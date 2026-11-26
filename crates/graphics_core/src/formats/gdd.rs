use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    ops::{Bound, Deref},
    path::PathBuf,
    sync::Arc,
};

use graphite_core_types::Type;
use graphite_core_types::{Color, NodeParameter, ProtoNodeIdentifier};
pub use graphite_document_container::archive::ArchiveFormat;
use graphite_document_container::{
    AnyContainer, ByteHolder, Container, ContainerError,
    archive::{Archive, ArchiveWriter, Xz, Zip},
    backends::memory::MemoryBackend,
    validate_path,
};
use graphite_document_format::{
    ExportFormat, ExportOptions, GddV1, GddV1Layout, Layout, MANIFEST_CODEC, Manifest,
    io::path_for,
    manifest::{FORMAT_MAGIC, SUPPORTED_FORMAT_VERSION},
};
use graphite_document_graph_storage::{
    AttributesRead, NetworkId, NetworkMetadataEntry, NoMetadata, NodeId as StorageNodeId,
    NodeMetadataEntry, PeerId, ROOT_NETWORK, Registry, RegistryDelta, from_runtime,
    to_runtime::ConversionError,
};
pub use graphite_editor::messages::portfolio::document::utility_types::network_interface::ImportOrExport;
use graphite_editor::messages::portfolio::document::{
    DocumentMessageHandler,
    node_graph::document_node_definitions::{DocumentNodeDefinition, resolve_proto_node_type},
    utility_types::{
        document_metadata::LayerNodeIdentifier,
        misc::PTZ,
        network_interface::{
            InputConnector, NavigationMetadata, NodeNetworkInterface, NodePosition, NodeTemplate,
            NodeTemplateImplementation, NodeTypePersistentMetadata, OutputConnector, Previewing,
            editor_delta::{
                ConstructedOps, EditorDelta, NetworkMetadataChange, NodeMetadataChange,
                construct_batch,
            },
            storage_metadata::{
                InterfaceRebuildError, StorageMetadataView, apply_network_view_settings,
                build_interface_from_storage, network_ids_from_entries,
            },
        },
    },
};
use graphite_editor::{
    consts::VIEWPORT_ZOOM_TO_FIT_PADDING_SCALE_FACTOR,
    messages::{clipboard::utility_types::ClipboardItem, prelude::NavigationMessageHandler},
};
use graphite_glam::{DAffine2, DVec2, IVec2, UVec2};
use graphite_graph_craft::application_io::resource::{
    HashMapResourceStorage, ResourceHash, ResourceStorage,
};
use graphite_graph_craft::{
    document::{
        DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
        value::{RenderOutput, TaggedValue},
    },
    runtime_delta::RuntimeDelta,
};
use graphite_interpreted_executor::node_registry::NODE_REGISTRY;
use graphite_math_nodes::modulo;
use graphite_std::{
    application_io::{ExportFormat as SvgExportFormat, RenderConfig},
    transform::Footprint,
    vector_nodes::{
        fill,
        generator_nodes::{ellipse, rectangle},
    },
};
use zip::ZipArchive;

use super::{OriginalJson, legacy::PreservedLegacyDocument};
use crate::evaluate::{
    CpuSvgEvaluationBudget, NUMERIC_NETWORK_WRAPPER, NumericGraphEvaluator,
    NumericGraphPreparationError, UnaryNumericOperation, evaluate_cpu_svg,
    typecheck_detached_numeric_network, validate_svg_configuration,
};

use crate::registry::{
    ARTBOARD_NODE_IDENTIFIER, ARTBOARD_NODE_NAME, CUSTOM_NODE_IDENTIFIER, CUSTOM_NODE_NAME,
    MERGE_NODE_IDENTIFIER, MERGE_NODE_NAME, artboard_node_definition, custom_node_definition,
    merge_node_definition,
};

mod xz;

#[derive(Clone, Copy)]
pub struct GddArchiveBudget {
    pub max_archive_bytes: usize,
    pub max_entry_bytes: u64,
    pub max_total_entry_bytes: u64,
    pub max_entries: usize,
}

#[derive(Clone, Copy)]
pub struct GddExportContent {
    pub include_registry: bool,
    pub include_history: bool,
}

impl Default for GddExportContent {
    fn default() -> Self {
        let options = ExportOptions::default();
        Self {
            include_registry: options.include_registry,
            include_history: options.include_history,
        }
    }
}

impl GddExportContent {
    pub fn validate(self) -> Result<(), GddArchiveError> {
        self.options()
            .validate()
            .map_err(GddArchiveError::InvalidExportOptions)?;
        Ok(())
    }

    fn options(self) -> ExportOptions {
        ExportOptions {
            include_registry: self.include_registry,
            include_history: self.include_history,
            ..ExportOptions::default()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GddArchiveError {
    #[error("This Graphite archive has an unsupported envelope.")]
    UnsupportedEnvelope,
    #[error("The Graphite archive exceeds its input or extracted-data budget.")]
    BudgetExceeded,
    #[error("The Graphite archive has inconsistent entry records.")]
    InconsistentEntries,
    #[error("Unsupported Graphite container format {format:?} version {version}.")]
    UnsupportedManifest { format: String, version: u32 },
    #[error("The Graphite archive is missing its recorded registry and history.")]
    MissingDocument,
    #[error("The Graphite document has no history to export.")]
    NoExportableHistory,
    #[error("The Graphite document has no pending edits to recover.")]
    NoPendingEdits,
    #[error("The recovered Graphite folder did not retain the replayed edits.")]
    IncompleteRecovery,
    #[error("Recover the pending Graphite edits before changing the graph or its saved history.")]
    PendingGraphEdits,
    #[error("The Graphite copy did not restore the saved history position.")]
    IncompleteHistoryStep,
    #[error("The selected node belongs to an earlier Graphite snapshot. Select the node again.")]
    StaleGraphNode,
    #[error("The Graphite graph is missing a saved network identity.")]
    MissingGraphNetwork,
    #[error(
        "The selected network belongs to an earlier Graphite snapshot. Choose the network again."
    )]
    StaleGraphNetwork,
    #[error("Select a standalone nested Graphite network to append an export.")]
    UnsupportedGraphExportAppend,
    #[error("Select a standalone nested Graphite network to append an import.")]
    UnsupportedGraphImportAppend,
    #[error("Select an eligible import in a standalone nested Graphite network.")]
    UnsupportedGraphImportRemoval,
    #[error("Select an eligible import and a different slot in the same nested Graphite network.")]
    UnsupportedGraphImportMove,
    #[error("Select an eligible export in a standalone nested Graphite network.")]
    UnsupportedGraphExportRemoval,
    #[error("Select an eligible export and a different slot in the same nested Graphite network.")]
    UnsupportedGraphExportMove,
    #[error("Select an eligible port in a standalone nested Graphite network.")]
    UnsupportedGraphPortRename,
    #[error("Enter a different Graphite port name.")]
    UnchangedGraphPortName,
    #[error("Select a literal primary import in a standalone nested Graphite network.")]
    UnsupportedGraphPrimaryImportExposure,
    #[error("Select an eligible primary export in a standalone nested Graphite network.")]
    UnsupportedGraphPrimaryExportExposure,
    #[error("Select an ordinary numeric output in the same Graphite network and an empty export.")]
    UnsupportedGraphExportConnection,
    #[error("Graphite could not resolve the selected export type. {0}")]
    GraphExportTypeMismatch(String),
    #[error(
        "Select a compatible exposed numeric import and node input in the same Graphite network."
    )]
    UnsupportedGraphImportConnection,
    #[error("Graphite could not type-check the selected import connection. {0}")]
    GraphImportTypeMismatch(String),
    #[error("Select an ordinary connected numeric input for the Graphite node insertion.")]
    UnsupportedGraphWireInsertion,
    #[error("Graphite could not type-check the inserted node on this wire. {0}")]
    GraphWireInsertionTypeMismatch(String),
    #[error("Select supported numeric nodes in one Graphite network.")]
    UnsupportedGraphClipboardSelection,
    #[error(
        "To cut a numeric group, select both ends of every connected wire in a network outside an active preview."
    )]
    UnsupportedGraphCutSelection,
    #[error("The clipboard does not contain an eligible Graphite numeric node group.")]
    UnsupportedGraphClipboard,
    #[error("Could not encode or decode the Graphite clipboard. {0}")]
    GraphClipboardFormat(serde_json::Error),
    #[error("Graphite could not type-check the node group. {0}")]
    GraphClipboardTypeMismatch(String),
    #[error("Enter a different Graphite node name.")]
    UnchangedGraphNodeName,
    #[error("The Graphite node already has the requested pin state.")]
    UnchangedGraphNodePin,
    #[error("The selected Graphite node cannot move in that direction in pinned order.")]
    InvalidPinnedNodeMove,
    #[error("The selected Graphite node has no supported numeric input.")]
    UnsupportedGraphNumericInput,
    #[error("Hide the Graphite Value port before editing its literal.")]
    ExposedGraphNumericInput,
    #[error("Enter a finite number for the Graphite Value input.")]
    InvalidGraphNumber,
    #[error("Enter a whole number in the 64-bit integer range for the Graphite Value input.")]
    InvalidGraphInteger,
    #[error("Enter two finite numbers separated by a comma for the Graphite Value input.")]
    InvalidGraphVector,
    #[error("Enter finite numbers separated by commas, or leave the Graphite Value list blank.")]
    InvalidGraphNumberList,
    #[error("The Graphite Value input must keep its original number type.")]
    GraphInputTypeChanged,
    #[error("Enter a different value for the Graphite Value input.")]
    UnchangedGraphInput,
    #[error("The Graphite Value input already has the requested exposure.")]
    UnchangedGraphInputExposure,
    #[error("The selected Graphite Value input has no supported direct connection.")]
    UnsupportedGraphConnection,
    #[error(
        "The Graphite connection requires a fallback type or layout change that this copy operation cannot save."
    )]
    UnsupportedGraphDisconnection,
    #[error("Choose a compatible Value output from a different node in the same Graphite network.")]
    InvalidGraphConnection,
    #[error("The Graphite wire requires a layout change that this copy operation cannot save.")]
    UnsupportedGraphConnectionLayout,
    #[error("The pinned Graphite definition for {0} is unavailable.")]
    MissingGraphNodeDefinition(&'static str),
    #[error("Choose a visible standalone Graphite node with stored literal defaults.")]
    UnsupportedCatalogNodeDefinition,
    #[error("Choose the document root or an editable nested Graphite network.")]
    UnsupportedCustomNodeScope,
    #[error(
        "Only an isolated Graphite node outside an active preview can be removed in a folder copy."
    )]
    UnsupportedGraphNodeDeletion,
    #[error(
        "Select an ordinary numeric Graphite node with room for its duplicate on the graph grid."
    )]
    UnsupportedGraphNodeDuplication,
    #[error("Select an ordinary Graphite node with a free grid position.")]
    UnsupportedGraphGridMove,
    #[error("Enter a different Graphite grid position.")]
    UnchangedGraphGridPosition,
    #[error("The change in Graphite grid coordinates is outside the supported range.")]
    GraphGridMoveOutOfRange,
    #[error("Graphite preview requires a supported numeric graph with literal inputs.")]
    UnsupportedGraphNumericPreview,
    #[error("The selected Graphite numeric graph has a cycle.")]
    CyclicGraphNumericPreview,
    #[error("The selected Graphite numeric graph exceeds the preview budget.")]
    GraphNumericPreviewBudget,
    #[error("Graphite could not evaluate the selected node. {0}")]
    GraphNumericPreviewEvaluation(String),
    #[error(
        "Graphite preview requires a saved root Merge or Artboard with supported filled shapes."
    )]
    UnsupportedGraphLayerPreview,
    #[error("The saved layer graph exceeds the preview budget.")]
    GraphLayerPreviewBudget,
    #[error("Graphite could not render the saved layer graph. {0}")]
    GraphLayerPreviewEvaluation(String),
    #[error("Choose a supported root Artboard and an empty document output.")]
    UnsupportedDocumentArtboard,
    #[error("Choose compatible Graphite nodes and an empty Content input.")]
    UnsupportedGraphContentConnection,
    #[error("Choose the exported root Artboard to save its SVG.")]
    UnsupportedArtboardSvgExport,
    #[error("Choose an Artboard with editable Location and Dimensions.")]
    UnsupportedArtboardGeometry,
    #[error("The Artboard geometry exceeds the finite SVG bounds.")]
    InvalidArtboardGeometry,
    #[error("Choose an Artboard with an editable Background color.")]
    UnsupportedArtboardBackground,
    #[error("Enter finite sRGB channels between zero and one for edited Background values.")]
    InvalidArtboardBackground,
    #[error("Choose a root Ellipse with editable radii.")]
    UnsupportedEllipseRadius,
    #[error("Enter an Ellipse radius in pixels with a finite diameter.")]
    InvalidEllipseRadius,
    #[error("Enter a different Ellipse radius.")]
    UnchangedEllipseRadius,
    #[error("Graphite could not use radius source {node:?}. {reason}")]
    InvalidEllipseRadiusSource { node: NodeId, reason: String },
    #[error("Choose a supported node with a hidden Boolean literal.")]
    UnsupportedGraphBooleanInput,
    #[error("Graphite could not allocate an unused node ID.")]
    GraphNodeIdentityCollision,
    #[error("Saved layers could not be read from this Graphite layer graph.")]
    InvalidRootLayerGraph,
    #[error("Choose a layer in the Graphite document network.")]
    UnsupportedRootLayerEdit,
    #[error("The Graphite layer already has this setting.")]
    UnchangedRootLayer,
    #[error("The Graphite copy could not save the addressed node's change.")]
    IncompleteGraphNodeEdit,
    #[error("This Graphite document already contains a different legacy document.")]
    ConflictingLegacyDocument,
    #[error("{0}")]
    InvalidExportOptions(&'static str),
    #[error("An embedded Graphite resource has no recorded content hash.")]
    UnresolvedResource,
    #[error("A linked Graphite resource has no recorded content hash.")]
    UnresolvedLinkedResource,
    #[error("No selected file matches the linked Graphite resource {0}.")]
    MissingApprovedResource(ResourceHash),
    #[error("The selected file with hash {0} does not match a missing linked Graphite resource.")]
    UnexpectedApprovedResource(ResourceHash),
    #[error("The embedded Graphite resource {0} is missing or does not match its content hash.")]
    InvalidResource(ResourceHash),
    #[error("An archive entry could not be read: {0}")]
    Archive(#[from] zip::result::ZipError),
    #[error("An archive entry could not be decoded: {0}")]
    Io(#[from] std::io::Error),
    #[error("The Graphite manifest could not be read: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("The Graphite container could not be staged: {0}")]
    Container(#[from] ContainerError),
    #[error("The pinned Graphite document storage operation failed: {0}")]
    Storage(#[from] graphite_document_format::Error),
    #[error("The saved Graphite graph could not be resolved: {0}")]
    RuntimeRegistry(#[source] ConversionError),
    #[error("The saved Graphite graph could not restore its network metadata: {0}")]
    RuntimeInterface(#[source] InterfaceRebuildError),
    #[error("The pinned Graphite editor could not prepare the node change: {0}")]
    GraphNodePreparation(#[source] from_runtime::ConversionError),
    #[error("The pinned Graphite writer could not export the archive: {0}")]
    Export(#[source] ContainerError),
}

#[derive(serde::Deserialize)]
struct ManifestHeader {
    format: String,
    format_version: u32,
}

pub struct PreservedGddV1 {
    original_bytes: Arc<[u8]>,
    inspected: InspectedGddV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GddRecoverySource {
    RegistryAndHistory,
    RegistryOnly,
    HistoryOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GddHistoryDirection {
    Undo,
    Redo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GddRecoverySummary {
    pub source: GddRecoverySource,
    pub network_count: usize,
    pub node_count: usize,
    pub history_len: usize,
    pub hot_log_len: usize,
}

pub struct GddSavedView {
    pub document: Option<GddDocumentView>,
    pub networks: Vec<GddNetworkView>,
    pub unresolved_networks: usize,
}

pub struct GddDocumentView {
    pub camera: PTZ,
    pub rulers_visible: bool,
}

pub struct GddArtboardCanvasOutput {
    pub render: RenderOutput,
    pub camera: PTZ,
    pub artboard_size: DVec2,
    pub pick_target: Option<GddArtboardCanvasPick>,
    pub overlay: Option<GddArtboardCanvasOverlay>,
}

pub struct GddArtboardCanvasOverlay {
    pub name: String,
    pub outline: [DVec2; 4],
    pub name_transform: DAffine2,
}

impl GddArtboardCanvasOverlay {
    fn from_document_bounds(
        name: String,
        [min, max]: [DVec2; 2],
        transform: DAffine2,
        camera: PTZ,
    ) -> Self {
        let left = if camera.flip { max.x } else { min.x };
        let label_corner = DVec2::new(left, min.y);
        let flip = DVec2::new(if camera.flip { -1. } else { 1. }, 1.);
        Self {
            name,
            outline: [min, DVec2::new(max.x, min.y), max, DVec2::new(min.x, max.y)]
                .map(|corner| transform.transform_point2(corner)),
            // Match DrawArtboardOverlays, including the unscaled four-pixel gap.
            name_transform: transform
                * DAffine2::from_translation(label_corner)
                * DAffine2::from_scale(DVec2::splat(camera.zoom().recip()))
                * DAffine2::from_translation(-DVec2::Y * 4.)
                * DAffine2::from_scale(flip),
        }
    }
}

pub struct GddArtboardCanvasPick {
    storage_id: StorageNodeId,
    snapshot: Arc<()>,
    pane_to_document: DAffine2,
    layer_to_document: DAffine2,
    targets: Vec<Arc<graphite_std::vector::click_target::ClickTarget>>,
}

impl GddArtboardCanvasPick {
    pub fn contains(&self, pane_point: DVec2) -> bool {
        let document_point = self.pane_to_document.transform_point2(pane_point);
        document_point.is_finite()
            && self
                .targets
                .iter()
                .any(|target| target.intersect_point(document_point, self.layer_to_document))
    }

    pub fn matches_node(&self, node: &GddGraphNode) -> bool {
        self.storage_id == node.storage_id && Arc::ptr_eq(&self.snapshot, &node.snapshot)
    }
}

pub struct GddNetworkView {
    pub id: NetworkId,
    pub path: Vec<NodeId>,
    pub name: String,
    pub navigation: NavigationMetadata,
    pub previewing: Previewing,
}

pub struct GddGraphInspection {
    pub nodes: Vec<GddGraphNode>,
    pub networks: Vec<GddGraphNetwork>,
}

pub struct GddNumericGraphEvaluator {
    evaluator: NumericGraphEvaluator,
    snapshot: Arc<()>,
    network_path: Vec<NodeId>,
}

pub struct GddResolvedNumericPorts {
    pub inputs: Vec<Type>,
    pub output: Type,
}

impl GddNumericGraphEvaluator {
    pub fn resolved_ports(&self, node: &GddGraphNode) -> Option<GddResolvedNumericPorts> {
        if !Arc::ptr_eq(&self.snapshot, &node.snapshot) || self.network_path != node.network_path {
            return None;
        }
        let types = self.evaluator.node_types(node.local_id)?;
        // The source map stores the evaluation context before the actual input slots.
        let inputs = types.inputs.get(1..)?;
        (inputs.len() == node.input_ports.len()).then(|| GddResolvedNumericPorts {
            inputs: inputs.to_vec(),
            output: types.output,
        })
    }

    pub async fn evaluate(&self) -> Result<GddNumericValue, GddArchiveError> {
        let output = self
            .evaluator
            .evaluate()
            .await
            .map_err(GddArchiveError::GraphNumericPreviewEvaluation)?;
        preview_numeric_value(&output)
    }
}

pub struct GddGraphGeometry {
    pub nodes: Vec<GddGraphNodeGeometry>,
    pub boundary_ports: Vec<(ImportOrExport, [f64; 2])>,
    pub wires: Vec<[[f64; 2]; 2]>,
    pub root_layers: Result<Vec<GddRootLayer>, GddArchiveError>,
}

impl GddGraphGeometry {
    pub fn nodes_intersecting_box(
        &self,
        bounds: [[f64; 2]; 2],
        exclude_layers: bool,
    ) -> impl Iterator<Item = StorageNodeId> + '_ {
        use graphite_std::math::{math_ext::QuadExt, quad::Quad};

        let [start, end] = bounds.map(DVec2::from_array);
        let quad = (start.is_finite() && end.is_finite())
            .then(|| Quad::from_box([start.min(end), start.max(end)]));
        self.nodes
            .iter()
            .filter(move |node| {
                (!exclude_layers || !node.is_layer)
                    && quad.as_ref().zip(node.click_target.as_ref()).is_some_and(
                        |(quad, target)| {
                            target.intersect_path(|| quad.to_lines(), DAffine2::IDENTITY)
                        },
                    )
            })
            .map(|node| node.storage_id)
    }
}

pub struct GddGraphNodeGeometry {
    pub storage_id: StorageNodeId,
    pub bounds: [[f64; 2]; 2],
    pub primary_input: Option<[f64; 2]>,
    pub primary_output: Option<[f64; 2]>,
    click_target: Option<graphite_std::vector::click_target::ClickTarget>,
    pub is_layer: bool,
}

pub struct GddRootLayer {
    pub storage_id: StorageNodeId,
    pub parent: Option<StorageNodeId>,
    pub depth: usize,
    pub visible: bool,
    pub locked: bool,
    pub select_all_eligible: bool,
}

#[derive(Clone, Copy)]
pub enum GddRootLayerChange {
    Visibility(bool),
    Locked(bool),
}

#[derive(Clone)]
pub struct GddGraphNetwork {
    pub id: NetworkId,
    pub parent: Option<NetworkId>,
    pub name: String,
    pub import_count: usize,
    pub export_count: usize,
    import_names: Vec<String>,
    export_names: Vec<String>,
    snapshot: Arc<()>,
    path: Vec<NodeId>,
    can_extend_ports: bool,
    untouched_custom_node: bool,
    primary_import_exposed: Option<bool>,
    primary_export_exposed: Option<bool>,
    numeric_imports: Vec<(usize, std::mem::Discriminant<GddNumericValue>)>,
    empty_exports: Vec<usize>,
    imports: Vec<GddGraphInputPort>,
    exports: Vec<GddGraphInputPort>,
}

impl GddGraphNetwork {
    pub fn imports(&self) -> &[GddGraphInputPort] {
        &self.imports
    }

    pub fn exports(&self) -> &[GddGraphInputPort] {
        &self.exports
    }

    pub fn connected_export<'nodes>(
        &self,
        nodes: &'nodes [GddGraphNode],
    ) -> Option<(&'nodes GddGraphNode, usize)> {
        let [export] = self.exports.as_slice() else {
            return None;
        };
        let Some(GddGraphPortSource::Node {
            storage_id,
            output_index,
        }) = &export.source
        else {
            return None;
        };
        let node = nodes.iter().find(|node| {
            node.storage_id == *storage_id
                && node.network_id == self.id
                && Arc::ptr_eq(&self.snapshot, &node.snapshot)
        })?;
        Some((node, *output_index))
    }

    pub fn import_name(&self, index: usize) -> Option<&str> {
        self.import_names
            .get(index)
            .map(String::as_str)
            .filter(|name| !name.is_empty())
    }

    pub fn export_name(&self, index: usize) -> Option<&str> {
        self.export_names
            .get(index)
            .map(String::as_str)
            .filter(|name| !name.is_empty())
    }

    pub fn can_extend_ports_in_copy(&self) -> bool {
        self.can_extend_ports
    }

    pub fn can_append_ports_in_copy(&self) -> bool {
        self.can_extend_ports || self.untouched_custom_node
    }

    pub fn can_insert_custom_node_in_copy(&self) -> bool {
        self.parent.is_none() || self.can_extend_ports
    }

    fn append_port_deltas(&self, edits: &mut Vec<EditorDelta>) {
        if self.untouched_custom_node {
            edits.push(EditorDelta::NetworkMetadata {
                network_path: self.path.clone(),
                change: NetworkMetadataChange::Reference(None),
            });
        }
    }

    fn port_operations<'operation>(
        &self,
        operations: &'operation mut [RegistryDelta],
    ) -> Result<&'operation mut [RegistryDelta], GddArchiveError> {
        use graphite_document_graph_storage::attr::node::ui::REFERENCE;

        if !self.untouched_custom_node {
            return Ok(operations);
        }
        let Some((RegistryDelta::ChangeNetworkAttribute { id, delta }, ports)) =
            operations.split_last_mut()
        else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        if *id != self.id || delta.key != REFERENCE || delta.value.is_some() {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(ports)
    }

    pub fn primary_import_exposed(&self) -> Option<bool> {
        self.primary_import_exposed
    }

    pub fn primary_export_exposed(&self) -> Option<bool> {
        self.primary_export_exposed
    }

    pub fn empty_exports(&self) -> &[usize] {
        &self.empty_exports
    }

    pub fn can_connect_export_from(&self, node: &GddGraphNode) -> bool {
        self.can_extend_ports
            && self.id == node.network_id
            && Arc::ptr_eq(&self.snapshot, &node.snapshot)
            && node.can_supply_numeric_export
    }

    pub fn compatible_numeric_imports(
        &self,
        node: &GddGraphNode,
    ) -> impl Iterator<Item = usize> + '_ {
        let current = node.numeric_import_source();
        let target = if self.can_extend_ports
            && self.id == node.network_id
            && Arc::ptr_eq(&self.snapshot, &node.snapshot)
        {
            match current {
                Some(current) => self
                    .numeric_imports
                    .iter()
                    .find_map(|&(index, kind)| (index == current).then_some(kind)),
                None => node
                    .numeric_port()
                    .filter(|port| port.exposed)
                    .map(|port| std::mem::discriminant(&port.value)),
            }
        } else {
            None
        };
        self.numeric_imports
            .iter()
            .filter_map(move |&(index, kind)| {
                (Some(index) != current && target == Some(kind)).then_some(index)
            })
    }

    pub fn can_start_numeric_import(&self, index: usize) -> bool {
        self.can_extend_ports
            && self
                .numeric_imports
                .iter()
                .any(|&(import, _)| import == index)
    }
}

#[derive(Clone, Copy)]
enum NestedPortEdit {
    Remove(usize),
    Move { from: usize, to: usize },
}

impl NestedPortEdit {
    fn unsupported_import(self) -> GddArchiveError {
        match self {
            Self::Remove(_) => GddArchiveError::UnsupportedGraphImportRemoval,
            Self::Move { .. } => GddArchiveError::UnsupportedGraphImportMove,
        }
    }

    fn unsupported_export(self) -> GddArchiveError {
        match self {
            Self::Remove(_) => GddArchiveError::UnsupportedGraphExportRemoval,
            Self::Move { .. } => GddArchiveError::UnsupportedGraphExportMove,
        }
    }

    fn valid_for(self, count: usize) -> bool {
        match self {
            Self::Remove(index) => index < count,
            Self::Move { from, to } => from < count && to < count && from != to,
        }
    }

    fn edit_slots<T>(self, slots: &mut Vec<T>) {
        match self {
            Self::Remove(index) => {
                slots.remove(index);
            }
            Self::Move { from, to } => {
                let moved = slots.remove(from);
                slots.insert(to, moved);
            }
        }
    }

    fn slot_order(self, count: usize) -> Result<(Vec<usize>, Vec<Option<usize>>), GddArchiveError> {
        let mut original_indices: Vec<_> = (0..count).collect();
        self.edit_slots(&mut original_indices);
        let mut new_indices = vec![None; count];
        for (new, &old) in original_indices.iter().enumerate() {
            let index = new_indices
                .get_mut(old)
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            *index = Some(new);
        }
        Ok((original_indices, new_indices))
    }
}

#[derive(Clone, Copy)]
pub enum GddNumericImportKind {
    Number,
    NumberList,
}

#[derive(Clone, Copy)]
pub enum GddGraphLayerPreview {
    Merge,
    Artboard { origin: DVec2, size: DVec2 },
}

#[derive(Clone, Copy)]
enum SavedGraphicProfile {
    Rectangle,
    Ellipse([f64; 2]),
    EllipseList,
    EllipseWired,
    Fill,
    Layer(GddGraphLayerPreview),
}

#[derive(Clone, Copy)]
pub enum GddEllipseAxis {
    X,
    Y,
}

const LAYER_CONTENT_INPUT: usize = 1;
const ARTBOARD_LOCATION_INPUT: usize = 2;
const ARTBOARD_DIMENSIONS_INPUT: usize = 3;
const ARTBOARD_BACKGROUND_INPUT: usize = 4;
const ARTBOARD_CLIP_INPUT: usize = 5;
const ELLIPSE_RADIUS_INPUTS: [usize; 2] =
    [ellipse::RadiusXInput::INDEX, ellipse::RadiusYInput::INDEX];
const MAX_PREVIEW_VALUES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GddArtboardGeometry {
    pub location: DVec2,
    pub dimensions: DVec2,
}

impl GddArtboardGeometry {
    pub fn parse(location: &str, dimensions: &str) -> Result<Self, GddArchiveError> {
        Ok(Self {
            location: parse_graph_vector(location)?,
            dimensions: parse_graph_vector(dimensions)?,
        })
    }

    fn read(document: &DocumentNode) -> Option<Self> {
        let Some(TaggedValue::DVec2(location)) =
            document.inputs.get(ARTBOARD_LOCATION_INPUT)?.as_value()
        else {
            return None;
        };
        let Some(TaggedValue::DVec2(dimensions)) =
            document.inputs.get(ARTBOARD_DIMENSIONS_INPUT)?.as_value()
        else {
            return None;
        };
        Some(Self {
            location: *location,
            dimensions: *dimensions,
        })
    }

    fn hidden(document: &DocumentNode) -> Option<Self> {
        [ARTBOARD_LOCATION_INPUT, ARTBOARD_DIMENSIONS_INPUT]
            .into_iter()
            .all(|index| {
                matches!(
                    document.inputs.get(index),
                    Some(NodeInput::Value { exposed: false, .. })
                )
            })
            .then(|| Self::read(document))
            .flatten()
    }

    fn normalized(self) -> Option<(DVec2, DVec2)> {
        let opposite_corner = self.location + self.dimensions;
        // CreateArtboard normalizes the starting corner before clamping the extent.
        let origin = self.location.min(opposite_corner);
        let size = self.dimensions.abs().max(DVec2::ONE);
        (self.location.is_finite()
            && self.dimensions.is_finite()
            && opposite_corner.is_finite()
            && (origin + size).is_finite())
        .then_some((origin, size))
    }
}

#[derive(Clone)]
pub struct GddGraphNode {
    snapshot: Arc<()>,
    storage_id: StorageNodeId,
    local_id: NodeId,
    network_path: Vec<NodeId>,
    network_id: NetworkId,
    nested_network: Option<NetworkId>,
    label: String,
    display_name: String,
    name: String,
    pinned: bool,
    pinned_order: Option<(usize, usize)>,
    numeric_input: Option<GddNumericInput>,
    boolean_input: Option<GddGraphBooleanInput>,
    artboard_geometry: Option<GddArtboardGeometry>,
    artboard_background: Option<Color>,
    can_delete_in_copy: bool,
    can_duplicate_in_copy: bool,
    can_preview_numeric_output: bool,
    graphic_profile: Option<SavedGraphicProfile>,
    is_document_output: bool,
    can_set_document_artboard: bool,
    can_supply_numeric_export: bool,
    can_copy_graph_node: bool,
    grid_position: Option<[i32; 2]>,
    input_ports: Vec<GddGraphInputPort>,
    output_names: Vec<String>,
    unavailable_proto: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GddGraphPortSource {
    Node {
        storage_id: StorageNodeId,
        output_index: usize,
    },
    Import(usize),
    UnavailableNode {
        local_id: NodeId,
        output_index: usize,
    },
}

#[derive(Clone, Debug)]
pub struct GddGraphInputPort {
    pub index: usize,
    pub name: Option<String>,
    pub exposed: bool,
    pub stored_type: Option<Type>,
    pub source: Option<GddGraphPortSource>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GddGraphBooleanInput {
    index: usize,
    pub name: &'static str,
    pub value: bool,
}

#[derive(Clone)]
enum GddNumericInput {
    Literal(GddNumericPort),
    Connected { index: usize, upstream: NodeId },
    Imported { index: usize, import_index: usize },
}

impl GddNumericInput {
    fn index(&self) -> usize {
        match self {
            Self::Literal(port) => port.index,
            Self::Connected { index, .. } | Self::Imported { index, .. } => *index,
        }
    }

    fn same_input(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Literal(first), Self::Literal(second)) => first.same_input(second),
            (
                Self::Connected { index, upstream },
                Self::Connected {
                    index: other_index,
                    upstream: other_upstream,
                },
            ) => index == other_index && upstream == other_upstream,
            (
                Self::Imported {
                    index,
                    import_index,
                },
                Self::Imported {
                    index: other_index,
                    import_index: other_import,
                },
            ) => index == other_index && import_index == other_import,
            _ => false,
        }
    }
}

#[derive(Clone)]
struct GddNumericPort {
    index: usize,
    value: GddNumericValue,
    exposed: bool,
}

struct GraphWireInsertion {
    consumer: StorageNodeId,
    input: InputConnector,
}

impl GddNumericPort {
    fn same_input(&self, other: &Self) -> bool {
        self.index == other.index
            && self.exposed == other.exposed
            && self.value.same_value(&other.value)
    }
}

#[derive(Clone, Debug)]
pub enum GddNumericValue {
    Number(f64),
    Integer(i64),
    Vector([f64; 2]),
    NumberList(Arc<[f64]>),
}

fn finite_graph_number(text: &str) -> Option<f64> {
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn parse_graph_vector(text: &str) -> Result<DVec2, GddArchiveError> {
    let (first, second) = text
        .split_once(',')
        .ok_or(GddArchiveError::InvalidGraphVector)?;
    let first = finite_graph_number(first).ok_or(GddArchiveError::InvalidGraphVector)?;
    let second = finite_graph_number(second).ok_or(GddArchiveError::InvalidGraphVector)?;
    Ok(DVec2::new(first, second))
}

impl GddNumericValue {
    pub fn parse_like(&self, text: &str) -> Result<Self, GddArchiveError> {
        let text = text.trim();
        match self {
            Self::Number(_) => finite_graph_number(text)
                .map(Self::Number)
                .ok_or(GddArchiveError::InvalidGraphNumber),
            Self::Integer(_) => text
                .parse::<i64>()
                .map(Self::Integer)
                .map_err(|_| GddArchiveError::InvalidGraphInteger),
            Self::Vector(_) => parse_graph_vector(text).map(|value| Self::Vector(value.to_array())),
            Self::NumberList(_) if text.is_empty() => Ok(Self::NumberList(Arc::from([]))),
            Self::NumberList(_) => text
                .split(',')
                .map(|part| {
                    finite_graph_number(part).ok_or(GddArchiveError::InvalidGraphNumberList)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|values| Self::NumberList(values.into())),
        }
    }

    fn same_value(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number(first), Self::Number(second)) => first.to_bits() == second.to_bits(),
            (Self::Integer(first), Self::Integer(second)) => first == second,
            (Self::Vector(first), Self::Vector(second)) => {
                first.map(f64::to_bits) == second.map(f64::to_bits)
            }
            (Self::NumberList(first), Self::NumberList(second)) => first
                .iter()
                .map(|value| value.to_bits())
                .eq(second.iter().map(|value| value.to_bits())),
            _ => false,
        }
    }

    fn same_type(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl fmt::Display for GddNumericValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(value) => value.fmt(formatter),
            Self::Integer(value) => value.fmt(formatter),
            Self::Vector([first, second]) => write!(formatter, "{first}, {second}"),
            Self::NumberList(values) => {
                if let Some((first, remaining)) = values.split_first() {
                    first.fmt(formatter)?;
                    for value in remaining {
                        write!(formatter, ", {value}")?;
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum GddPinnedNodeMove {
    Up,
    Down,
}

impl GddGraphNode {
    pub fn storage_id(&self) -> StorageNodeId {
        self.storage_id
    }

    pub fn input_ports(&self) -> &[GddGraphInputPort] {
        &self.input_ports
    }

    pub fn output_name(&self, index: usize) -> Option<&str> {
        self.output_names
            .get(index)
            .map(String::as_str)
            .filter(|name| !name.is_empty())
    }

    pub fn unavailable_proto(&self) -> Option<&str> {
        self.unavailable_proto.as_deref()
    }

    pub fn network_id(&self) -> NetworkId {
        self.network_id
    }

    pub fn nested_network(&self) -> Option<NetworkId> {
        self.nested_network
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    pub fn can_delete_in_copy(&self) -> bool {
        self.can_delete_in_copy
    }

    pub fn can_duplicate_in_copy(&self) -> bool {
        self.can_duplicate_in_copy
    }

    pub fn grid_position(&self) -> Option<[i32; 2]> {
        self.grid_position
    }

    pub fn can_preview_numeric_output(&self) -> bool {
        self.can_preview_numeric_output
    }

    pub fn layer_preview(&self) -> Option<GddGraphLayerPreview> {
        match self.graphic_profile {
            Some(SavedGraphicProfile::Layer(preview)) if self.is_document_output => Some(preview),
            _ => None,
        }
    }

    pub fn can_set_document_artboard(&self) -> bool {
        self.can_set_document_artboard
    }

    pub fn graphic_content_input_from(&self, source: &Self) -> Option<usize> {
        if !self.same_network_as(source) {
            return None;
        }
        let index = match (self.graphic_profile?, source.graphic_profile?) {
            (
                SavedGraphicProfile::Fill,
                SavedGraphicProfile::Rectangle
                | SavedGraphicProfile::Ellipse(_)
                | SavedGraphicProfile::EllipseList
                | SavedGraphicProfile::EllipseWired,
            ) => fill::ContentInput::INDEX,
            (
                SavedGraphicProfile::Layer(GddGraphLayerPreview::Artboard { .. }),
                SavedGraphicProfile::Layer(GddGraphLayerPreview::Merge),
            ) if self.is_document_output => LAYER_CONTENT_INPUT,
            (
                SavedGraphicProfile::Layer(GddGraphLayerPreview::Merge),
                SavedGraphicProfile::Fill,
            ) if matches!(
                source
                    .input_ports
                    .get(fill::ContentInput::INDEX)
                    .and_then(|port| port.source.as_ref()),
                Some(GddGraphPortSource::Node {
                    output_index: 0,
                    ..
                })
            ) =>
            {
                LAYER_CONTENT_INPUT
            }
            _ => return None,
        };
        self.input_ports
            .get(index)
            .filter(|port| port.exposed && port.source.is_none())
            .map(|_| index)
    }

    pub fn can_copy_graph_node(&self) -> bool {
        self.can_copy_graph_node
    }

    pub fn validate_grid_move(&self, destination: [i32; 2]) -> Result<(), GddArchiveError> {
        self.grid_move_offset(destination).map(|_| ())
    }

    fn grid_move_offset(&self, destination: [i32; 2]) -> Result<IVec2, GddArchiveError> {
        let original = self
            .grid_position
            .ok_or(GddArchiveError::UnsupportedGraphGridMove)?;
        if original == destination {
            return Err(GddArchiveError::UnchangedGraphGridPosition);
        }
        let [x, y] = original;
        let [destination_x, destination_y] = destination;
        match (destination_x.checked_sub(x), destination_y.checked_sub(y)) {
            (Some(horizontal), Some(vertical)) => Ok(IVec2::new(horizontal, vertical)),
            _ => Err(GddArchiveError::GraphGridMoveOutOfRange),
        }
    }

    pub fn numeric_input(&self) -> Option<&GddNumericValue> {
        self.numeric_port()
            .filter(|port| !port.exposed)
            .map(|port| &port.value)
    }

    pub fn numeric_input_exposed(&self) -> Option<bool> {
        self.numeric_port().map(|port| port.exposed)
    }

    pub fn boolean_input(&self) -> Option<GddGraphBooleanInput> {
        self.boolean_input
    }

    pub fn artboard_geometry(&self) -> Option<GddArtboardGeometry> {
        self.artboard_geometry
    }

    pub fn artboard_background(&self) -> Option<Color> {
        self.artboard_background
    }

    pub fn ellipse_radii(&self) -> Option<[f64; 2]> {
        match self.graphic_profile? {
            SavedGraphicProfile::Ellipse(radii) => Some(radii),
            _ => None,
        }
    }

    pub fn has_numeric_connection(&self) -> bool {
        matches!(
            self.numeric_input.as_ref(),
            Some(GddNumericInput::Connected { .. } | GddNumericInput::Imported { .. })
        )
    }

    pub fn can_connect_numeric_input_from(&self, source: &Self) -> bool {
        self.same_network_as(source)
            && self.local_id != source.local_id
            && self
                .numeric_port()
                .zip(source.numeric_port())
                .is_some_and(|(input, output)| {
                    input.exposed && input.value.same_type(&output.value)
                })
    }

    pub fn can_start_numeric_output(&self) -> bool {
        self.can_supply_numeric_export && self.numeric_port().is_some()
    }

    pub fn numeric_connection_source<'nodes>(&self, nodes: &'nodes [Self]) -> Option<&'nodes Self> {
        let Some(GddNumericInput::Connected { upstream, .. }) = self.numeric_input.as_ref() else {
            return None;
        };
        nodes
            .iter()
            .find(|node| self.same_network_as(node) && node.local_id == *upstream)
    }

    pub fn numeric_import_source(&self) -> Option<usize> {
        match self.numeric_input.as_ref() {
            Some(GddNumericInput::Imported { import_index, .. }) => Some(*import_index),
            _ => None,
        }
    }

    pub fn can_reconnect_numeric_input_from(&self, previous: &Self, replacement: &Self) -> bool {
        matches!(
            self.numeric_input.as_ref(),
            Some(GddNumericInput::Connected { upstream, .. }) if *upstream == previous.local_id
        ) && self.same_network_as(previous)
            && self.same_network_as(replacement)
            && self.local_id != replacement.local_id
            && previous.local_id != replacement.local_id
            && previous
                .numeric_port()
                .zip(replacement.numeric_port())
                .is_some_and(|(first, second)| first.value.same_type(&second.value))
    }

    fn same_network_as(&self, source: &Self) -> bool {
        Arc::ptr_eq(&self.snapshot, &source.snapshot) && self.network_path == source.network_path
    }

    fn numeric_port(&self) -> Option<&GddNumericPort> {
        match self.numeric_input.as_ref()? {
            GddNumericInput::Literal(port) => Some(port),
            GddNumericInput::Connected { .. } | GddNumericInput::Imported { .. } => None,
        }
    }

    pub fn pinned_position(&self) -> Option<(usize, usize)> {
        self.pinned_order.map(|(index, count)| (index + 1, count))
    }

    pub fn can_move_pinned(&self, movement: GddPinnedNodeMove) -> bool {
        self.pinned_insertion_index(movement).is_some()
    }

    fn pinned_insertion_index(&self, movement: GddPinnedNodeMove) -> Option<usize> {
        let (index, count) = self.pinned_order?;
        match movement {
            GddPinnedNodeMove::Up => index.checked_sub(1),
            // Source insertion slots include the moving node. Moving down one skips
            // both the node and its following neighbor before source removal.
            GddPinnedNodeMove::Down if index + 1 < count => Some(index + 2),
            GddPinnedNodeMove::Down => None,
        }
    }
}

struct GddRuntimeGraph {
    network: NodeNetwork,
    nodes: Vec<NodeMetadataEntry>,
    networks: Vec<NetworkMetadataEntry>,
}

impl GddRuntimeGraph {
    fn into_interface(self) -> Result<NodeNetworkInterface, GddArchiveError> {
        build_interface_from_storage(self.network, self.nodes, self.networks)
            .map_err(GddArchiveError::RuntimeInterface)
    }
}

pub struct InspectedGddV1 {
    entries: StagedArchive,
    document: GddV1,
    recovery: GddRecoverySummary,
    snapshot: Arc<()>,
}

impl PreservedGddV1 {
    pub fn original_bytes(&self) -> &[u8] {
        &self.original_bytes
    }

    pub fn into_inspected(self) -> InspectedGddV1 {
        self.inspected
    }
}

impl Deref for PreservedGddV1 {
    type Target = InspectedGddV1;

    fn deref(&self) -> &Self::Target {
        &self.inspected
    }
}

impl InspectedGddV1 {
    pub fn manifest(&self) -> &Manifest {
        self.document.manifest()
    }

    pub fn read_entry(&self, path: &str) -> Result<ByteHolder, ContainerError> {
        self.entries.working.read(path)
    }

    pub fn history_len(&self) -> usize {
        self.recovery.history_len
    }

    pub fn hot_log_len(&self) -> usize {
        self.recovery.hot_log_len
    }

    pub fn recovery(&self) -> GddRecoverySummary {
        self.recovery
    }

    pub fn matches_saved_history(&self, saved: &Self) -> bool {
        let current_session = self.document.session();
        let saved_session = saved.document.session();
        self.manifest().document_id == saved.manifest().document_id
            && self.history_len() == saved.history_len()
            && self.hot_log_len() == 0
            && saved.hot_log_len() == 0
            && current_session.head_rev() == saved_session.head_rev()
            && current_session.redo_stack() == saved_session.redo_stack()
            && current_session.last_broadcast_rev() == saved_session.last_broadcast_rev()
            && current_session
                .history()
                .map(|delta| delta.id)
                .eq(saved_session.history().map(|delta| delta.id))
    }

    pub fn can_move_history(&self, direction: GddHistoryDirection) -> bool {
        self.hot_log_len() == 0
            && match direction {
                GddHistoryDirection::Undo => self.document.can_undo(),
                GddHistoryDirection::Redo => self.document.can_redo(),
            }
    }

    pub async fn move_history_in_copy(
        &self,
        direction: GddHistoryDirection,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::CrdtError;

        if self.hot_log_len() != 0 {
            return Err(GddArchiveError::PendingGraphEdits);
        }
        if !self.can_move_history(direction) {
            let error = match direction {
                GddHistoryDirection::Undo => CrdtError::NothingToUndo,
                GddHistoryDirection::Redo => CrdtError::NothingToRedo,
            };
            return Err(graphite_document_format::Error::from(error).into());
        }
        let (mut owner, staged) = self.fork_working_copy(&budget).await?;
        match direction {
            GddHistoryDirection::Undo => owner.undo()?,
            GddHistoryDirection::Redo => owner.redo()?,
        };
        let copy = staged.reopen_after_edit(&budget).await?;
        let actual = copy.document.session();
        let expected = owner.session();
        if actual.head_rev() != expected.head_rev()
            || actual.redo_stack() != expected.redo_stack()
            || actual.last_broadcast_rev() != expected.last_broadcast_rev()
            || copy.history_len() != self.history_len()
            || copy.hot_log_len() != 0
            || !copy.document.registry().value_equal(owner.registry())
        {
            return Err(GddArchiveError::IncompleteHistoryStep);
        }
        Ok(copy)
    }

    pub async fn graph_nodes(&self) -> Result<Vec<GddGraphNode>, GddArchiveError> {
        Ok(self.graph_inspection().await?.nodes)
    }

    pub async fn graph_geometry(
        &self,
        network: &GddGraphNetwork,
    ) -> Result<GddGraphGeometry, GddArchiveError> {
        if !Arc::ptr_eq(&self.snapshot, &network.snapshot) {
            return Err(GddArchiveError::StaleGraphNetwork);
        }
        let path = network.path.as_slice();
        let graph = self.runtime_graph().await?;
        let mut ids = graph
            .nodes
            .iter()
            .filter(|node| node.network_path == path)
            .map(|node| (node.local_id, node.storage_id))
            .collect::<Vec<_>>();
        ids.sort_unstable();
        let mut interface = graph.into_interface()?;
        let root_layers = if path.is_empty() {
            project_root_layers(&mut interface, &ids)
        } else {
            Ok(Vec::new())
        };
        let nodes = ids
            .iter()
            .filter_map(|(local_id, storage_id)| {
                let bounds = interface.node_bounding_box(local_id, path)?;
                bounds
                    .iter()
                    .all(|point| point.is_finite())
                    .then(|| GddGraphNodeGeometry {
                        storage_id: *storage_id,
                        bounds: bounds.map(|point| point.to_array()),
                        primary_input: finite_graph_point(
                            interface
                                .input_position(&InputConnector::primary_input(*local_id), path),
                        ),
                        primary_output: finite_graph_point(
                            interface
                                .output_position(&OutputConnector::primary_output(*local_id), path),
                        ),
                        click_target: interface
                            .node_click_targets(local_id, path)
                            .map(|targets| targets.node_click_target.clone()),
                        is_layer: interface.is_layer(local_id, path),
                    })
            })
            .collect();
        let nested = interface
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        let boundary_ports = (0..network.import_count)
            .filter_map(|index| {
                let point = finite_graph_point(
                    interface.output_position(&OutputConnector::Import(index), path),
                )?;
                Some((ImportOrExport::Import(index), point))
            })
            .chain((0..network.export_count).filter_map(|index| {
                let point = finite_graph_point(
                    interface.input_position(&InputConnector::Export(index), path),
                )?;
                Some((ImportOrExport::Export(index), point))
            }))
            .collect();
        let wires = ids
            .iter()
            .flat_map(|(node, _)| {
                nested.nodes.get(node).into_iter().flat_map(|document| {
                    (0..document.inputs.len())
                        .map(|index| InputConnector::node_at_index(*node, index))
                })
            })
            .chain((0..nested.exports.len()).map(InputConnector::Export))
            .filter_map(|input| {
                let output = interface.upstream_output_connector(&input, path)?;
                let start = finite_graph_point(interface.output_position(&output, path))?;
                let end = finite_graph_point(interface.input_position(&input, path))?;
                Some([start, end])
            })
            .collect();
        Ok(GddGraphGeometry {
            nodes,
            boundary_ports,
            wires,
            root_layers,
        })
    }

    pub async fn graph_inspection(&self) -> Result<GddGraphInspection, GddArchiveError> {
        let graph = self.runtime_graph().await?;
        let mut network_ids = network_ids_from_entries(&graph.networks);
        if self.graph_root_needs_initialization() {
            network_ids.insert(Vec::new(), ROOT_NETWORK);
        }
        let mut nodes: Vec<_> = graph
            .nodes
            .iter()
            .map(|node| {
                let mut owned_path = node.network_path.clone();
                owned_path.push(node.local_id);
                Ok(GddGraphNode {
                    snapshot: self.snapshot.clone(),
                    storage_id: node.storage_id,
                    local_id: node.local_id,
                    network_path: node.network_path.clone(),
                    network_id: *network_ids
                        .get(&node.network_path)
                        .ok_or(GddArchiveError::MissingGraphNetwork)?,
                    nested_network: network_ids.get(&owned_path).copied(),
                    label: String::new(),
                    display_name: String::new(),
                    name: node.display_name.clone().unwrap_or_default(),
                    pinned: node.pinned,
                    pinned_order: None,
                    numeric_input: None,
                    boolean_input: None,
                    artboard_geometry: None,
                    artboard_background: None,
                    can_delete_in_copy: false,
                    can_duplicate_in_copy: false,
                    can_preview_numeric_output: false,
                    graphic_profile: None,
                    is_document_output: false,
                    can_set_document_artboard: false,
                    can_supply_numeric_export: false,
                    can_copy_graph_node: false,
                    grid_position: None,
                    input_ports: Vec::new(),
                    output_names: Vec::new(),
                    unavailable_proto: None,
                })
            })
            .collect::<Result<_, GddArchiveError>>()?;
        let node_ids = nodes
            .iter()
            .map(|node| ((node.network_id, node.local_id), node.storage_id))
            .collect::<BTreeMap<_, _>>();
        let mut interface = graph.into_interface()?;
        let mut network_paths = network_ids.iter().collect::<Vec<_>>();
        network_paths.sort_unstable_by_key(|(path, _)| *path);
        let networks = network_paths
            .into_iter()
            .map(|(path, &id)| {
                let (name, parent) = match path.split_last() {
                    Some((node, parent_path)) => (
                        interface.display_name(node, parent_path),
                        Some(
                            *network_ids
                                .get(parent_path)
                                .ok_or(GddArchiveError::MissingGraphNetwork)?,
                        ),
                    ),
                    None => ("Root graph".to_owned(), None),
                };
                let owner_metadata = path.split_last().and_then(|(owner_id, parent_path)| {
                    interface.node_metadata(owner_id, parent_path)
                });
                let (import_names, export_names) = owner_metadata.map_or_else(
                    || (Vec::new(), Vec::new()),
                    |metadata| {
                        let metadata = &metadata.persistent_metadata;
                        let imports: Vec<_> = metadata
                            .input_metadata
                            .iter()
                            .map(|input| input.persistent_metadata.input_name.clone())
                            .collect();
                        (imports, metadata.output_names.clone())
                    },
                );
                let nested = interface
                    .nested_network(path)
                    .ok_or(GddArchiveError::MissingGraphNetwork)?;
                let exports = &nested.exports;
                let imports = enclosing_network_inputs(interface.document_network(), path)
                    .ok_or(GddArchiveError::MissingGraphNetwork)?;
                let editable_network = path.split_last().is_some_and(|(node_id, parent_path)| {
                    !interface.is_layer(node_id, parent_path)
                });
                let network_metadata = interface.network_metadata(path);
                let reference = network_metadata
                    .and_then(|metadata| metadata.persistent_metadata.reference.as_deref());
                let import_scope = parent.unwrap_or(id);
                let import_ports = imports
                    .iter()
                    .enumerate()
                    .map(|(index, input)| {
                        project_graph_input(
                            index,
                            input,
                            import_names.get(index).map(String::as_str),
                            import_scope,
                            &node_ids,
                        )
                    })
                    .collect();
                let export_ports = exports
                    .iter()
                    .enumerate()
                    .map(|(index, input)| {
                        project_graph_input(
                            index,
                            input,
                            export_names.get(index).map(String::as_str),
                            id,
                            &node_ids,
                        )
                    })
                    .collect();
                Ok(GddGraphNetwork {
                    id,
                    parent,
                    name,
                    import_count: interface.number_of_imports(path),
                    export_count: interface.number_of_exports(path),
                    import_names,
                    export_names,
                    snapshot: self.snapshot.clone(),
                    path: path.clone(),
                    can_extend_ports: editable_network
                        && network_metadata.is_some()
                        && reference.is_none(),
                    untouched_custom_node: editable_network
                        && reference == Some(CUSTOM_NODE_NAME)
                        && imports.is_empty()
                        && exports.is_empty()
                        && nested.nodes.is_empty()
                        && nested.scope_injections.is_empty(),
                    primary_import_exposed: match imports.first() {
                        Some(NodeInput::Value { exposed, .. }) => Some(*exposed),
                        _ => None,
                    },
                    primary_export_exposed: match exports.first() {
                        Some(NodeInput::Value { exposed, .. }) => Some(*exposed),
                        Some(NodeInput::Node { .. }) => Some(true),
                        _ => None,
                    },
                    numeric_imports: imports
                        .iter()
                        .enumerate()
                        .filter_map(|(index, input)| {
                            exposed_numeric_literal(input)
                                .map(|value| (index, std::mem::discriminant(&value)))
                        })
                        .collect(),
                    empty_exports: exports
                        .iter()
                        .enumerate()
                        .filter_map(|(index, input)| empty_graph_export(input).then_some(index))
                        .collect(),
                    imports: import_ports,
                    exports: export_ports,
                })
            })
            .collect::<Result<_, GddArchiveError>>()?;
        nodes.sort_by(|left, right| {
            left.network_path
                .cmp(&right.network_path)
                .then_with(|| left.local_id.cmp(&right.local_id))
        });
        let mut pinned_indexes = BTreeMap::new();
        let mut wired_nodes = BTreeMap::new();
        let document_exports = &interface.document_network().exports;
        let empty_document_output = empty_document_output(document_exports);
        let document_output_node = match document_exports.as_slice() {
            [
                NodeInput::Node {
                    node_id,
                    output_index: 0,
                },
            ] => Some(*node_id),
            _ => None,
        };
        for node in &mut nodes {
            let path = &node.network_path;
            if node.pinned {
                let (indexes, count) = pinned_indexes.entry(path.clone()).or_insert_with(|| {
                    let order = interface.ordered_pinned_nodes(path);
                    let mut indexes = BTreeMap::new();
                    for (index, node_id) in order.iter().enumerate() {
                        indexes.entry(*node_id).or_insert(index);
                    }
                    (indexes, order.len())
                });
                node.pinned_order = indexes.get(&node.local_id).map(|index| (*index, *count));
            }
            let parent = if path.is_empty() {
                "Root graph".into()
            } else {
                path.iter()
                    .enumerate()
                    .map(|(depth, node)| interface.display_name(node, &path[..depth]))
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            node.display_name = interface.display_name(&node.local_id, path);
            node.label = format!("{parent} · {} [{}]", node.display_name, node.local_id.0);
            let metadata = interface
                .node_metadata(&node.local_id, path)
                .map(|node| &node.persistent_metadata);
            if let Some(metadata) = metadata {
                node.output_names.clone_from(&metadata.output_names);
            }
            if let Some(document) = interface.document_node(&node.local_id, path) {
                if let DocumentNodeImplementation::ProtoNode(identifier) = &document.implementation
                    && !NODE_REGISTRY.contains_key(identifier)
                {
                    node.unavailable_proto = Some(identifier.to_string());
                }
                node.input_ports = document
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| {
                        let name = metadata
                            .and_then(|metadata| metadata.input_metadata.get(index))
                            .map(|metadata| metadata.persistent_metadata.input_name.as_str());
                        project_graph_input(index, input, name, node.network_id, &node_ids)
                    })
                    .collect();
            }
            node.numeric_input = graph_numeric_input(&interface, node);
            node.can_preview_numeric_output = !interface.is_layer(&node.local_id, path)
                && interface
                    .document_node(&node.local_id, path)
                    .is_some_and(|document| numeric_preview_inputs(document).is_some());
            if node.can_preview_numeric_output {
                node.boolean_input = interface
                    .document_node(&node.local_id, path)
                    .and_then(graph_boolean_literal);
            }
            if path.is_empty()
                && let Some(document) = interface.document_node(&node.local_id, path)
                && let Some((_, profile)) = checked_saved_graphic_node(document)
            {
                node.graphic_profile = Some(profile);
                node.is_document_output = document_output_node == Some(node.local_id);
                if matches!(
                    profile,
                    SavedGraphicProfile::Layer(GddGraphLayerPreview::Artboard { .. })
                ) {
                    node.can_set_document_artboard =
                        empty_document_output && interface.is_layer(&node.local_id, path);
                    node.boolean_input =
                        hidden_graph_boolean(&document.inputs, ARTBOARD_CLIP_INPUT, "Clip");
                    node.artboard_geometry = GddArtboardGeometry::hidden(document);
                    node.artboard_background = hidden_artboard_background(document);
                }
            }
            node.can_supply_numeric_export = node.can_preview_numeric_output
                && !interface.is_chain(&node.local_id, path)
                && interface.number_of_outputs(&node.local_id, path) == 1;
            node.can_copy_graph_node = (node.can_supply_numeric_export
                || node.nested_network.is_some())
                && !interface.is_layer(&node.local_id, path)
                && !interface.is_chain(&node.local_id, path)
                && interface
                    .position(&node.local_id, path)
                    .is_some_and(|position| {
                        position.x.checked_add(2).is_some() && position.y.checked_add(2).is_some()
                    });
            node.can_duplicate_in_copy = matches!(
                node.numeric_input,
                Some(GddNumericInput::Literal(_) | GddNumericInput::Connected { .. })
            ) && !interface.is_chain(&node.local_id, path)
                && interface
                    .position(&node.local_id, path)
                    .is_some_and(|position| {
                        position.x.checked_add(2).is_some() && position.y.checked_add(2).is_some()
                    });
            let wired = wired_nodes
                .entry(path.clone())
                .or_insert_with(|| graph_wired_output_nodes(&mut interface, path));
            if graph_node_has_isolated_layout(&interface, node) {
                node.can_delete_in_copy = wired
                    .as_ref()
                    .is_some_and(|wired| !wired.contains(&node.local_id));
                node.grid_position = interface
                    .position(&node.local_id, path)
                    .map(|position| position.to_array());
            }
        }
        Ok(GddGraphInspection { nodes, networks })
    }

    pub async fn insert_root_numeric_node_in_copy(
        &self,
        operation: UnaryNumericOperation,
        position: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        if self.hot_log_len() != 0 {
            return Err(GddArchiveError::PendingGraphEdits);
        }
        let template = numeric_node_template(operation)?;
        self.insert_node_template_in_copy(&[], template, position, budget)
            .await
    }

    pub async fn insert_numeric_node_in_network_copy(
        &self,
        network: &GddGraphNetwork,
        operation: UnaryNumericOperation,
        position: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        let template = numeric_node_template(operation)?;
        self.insert_node_template_in_copy(&network.path, template, position, budget)
            .await
    }

    pub async fn append_nested_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.append_network_import(network, false, TaggedValue::None, budget)
            .await
    }

    pub async fn append_secondary_nested_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.append_network_import(network, true, TaggedValue::None, budget)
            .await
    }

    pub async fn append_nested_numeric_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        kind: GddNumericImportKind,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        let default_value = match kind {
            GddNumericImportKind::Number => TaggedValue::Number(0.),
            GddNumericImportKind::NumberList => TaggedValue::Numbers(Vec::new()),
        };
        self.append_network_import(network, false, default_value, budget)
            .await
    }

    async fn append_network_import(
        &self,
        network: &GddGraphNetwork,
        secondary: bool,
        default_value: TaggedValue,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        if !network.can_append_ports_in_copy() {
            return Err(GddArchiveError::UnsupportedGraphImportAppend);
        }
        let path = &network.path;
        let Some((&owner_id, parent_path)) = path.split_last() else {
            return Err(GddArchiveError::UnsupportedGraphImportAppend);
        };
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let mut expected_inputs = interface
            .document_node(&owner_id, parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .inputs
            .clone();
        let original_count = expected_inputs.len();
        if original_count != network.import_count {
            return Err(GddArchiveError::StaleGraphNetwork);
        }
        let previous_metadata = &interface
            .node_metadata(&owner_id, parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .persistent_metadata;
        let owner_storage_id = previous_metadata
            .storage_id
            .map(StorageNodeId)
            .ok_or(GddArchiveError::MissingGraphNetwork)?;
        let previous_metadata = previous_metadata.input_metadata.clone();
        if secondary && original_count == 0 {
            let nested = interface
                .nested_network(path)
                .ok_or(GddArchiveError::StaleGraphNetwork)?;
            if nested
                .exports
                .iter()
                .chain(nested.nodes.values().flat_map(|node| &node.inputs))
                .any(|input| matches!(input, NodeInput::Import { .. }))
            {
                return Err(GddArchiveError::UnsupportedGraphImportAppend);
            }
            let hidden = NodeInput::value(TaggedValue::None, false);
            interface.add_import(TaggedValue::None, true, -1, "", "", path);
            interface.set_input(
                &InputConnector::primary_input(owner_id),
                hidden.clone(),
                parent_path,
            );
            expected_inputs.push(hidden);
        }
        expected_inputs.push(NodeInput::value(default_value.clone(), true));
        interface.add_import(default_value, true, -1, "", "", path);
        if !interface
            .document_node(&owner_id, parent_path)
            .is_some_and(|node| node.inputs == expected_inputs)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let input_metadata = interface
            .node_metadata(&owner_id, parent_path)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
            .persistent_metadata
            .input_metadata
            .clone();
        if !input_metadata.starts_with(&previous_metadata) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let mut edits = vec![
            EditorDelta::Graph(RuntimeDelta::SetInputs {
                network_path: parent_path.to_vec(),
                node_id: owner_id,
                inputs: expected_inputs.clone(),
            }),
            EditorDelta::NodeInputMetadata {
                network_path: parent_path.to_vec(),
                node_id: owner_id,
                input_metadata: input_metadata.clone(),
            },
        ];
        network.append_port_deltas(&mut edits);
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let original_slots = self
            .document
            .registry()
            .node_instances
            .get(&owner_storage_id)
            .ok_or(GddArchiveError::MissingGraphNetwork)?
            .inputs();
        let mut batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let [RegistryDelta::SetNodeInputs { id, inputs }, rest @ ..] =
            network.port_operations(&mut batch.ops)?
        else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        if *id != owner_storage_id
            || original_slots.len() != original_count
            || inputs.len() != expected_inputs.len()
            || !rest.iter().all(|operation| {
                matches!(operation,
                    RegistryDelta::ChangeNodeInputAttribute { id, index, .. }
                        if *id == owner_storage_id && (*index as usize) < inputs.len()
                )
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        for (current, original) in inputs.iter_mut().zip(original_slots) {
            if current.input != original.input {
                return Err(GddArchiveError::IncompleteGraphNodeEdit);
            }
            // An append leaves existing slots in place, including attributes the runtime cannot decode.
            current.clone_from(original);
        }
        batch.ops.retain(|operation| !matches!(operation,
            RegistryDelta::ChangeNodeInputAttribute { index, .. } if (*index as usize) < original_count
        ));
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        let retained = copy
            .document
            .registry()
            .node_instances
            .get(&owner_storage_id)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
            .inputs();
        if !actual
            .network_metadata(path)
            .is_some_and(|metadata| metadata.persistent_metadata.reference.is_none())
            || !actual
                .document_node(&owner_id, parent_path)
                .is_some_and(|node| node.inputs == expected_inputs)
            || !actual
                .node_metadata(&owner_id, parent_path)
                .is_some_and(|metadata| {
                    metadata.persistent_metadata.input_metadata == input_metadata
                })
            || retained.len() != expected_inputs.len()
            || !retained
                .iter()
                .zip(original_slots)
                .all(|(current, original)| {
                    current.input == original.input && current.attributes == original.attributes
                })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn remove_nested_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        import_index: usize,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.edit_nested_network_import_in_copy(
            network,
            NestedPortEdit::Remove(import_index),
            budget,
        )
        .await
    }

    pub async fn move_nested_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        from: usize,
        to: usize,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.edit_nested_network_import_in_copy(network, NestedPortEdit::Move { from, to }, budget)
            .await
    }

    async fn edit_nested_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        edit: NestedPortEdit,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        let unsupported = || edit.unsupported_import();
        if !network.can_extend_ports_in_copy() || !edit.valid_for(network.import_count) {
            return Err(unsupported());
        }
        let path = &network.path;
        let (&owner_id, parent_path) = path.split_last().ok_or_else(unsupported)?;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let mut owner_inputs = interface
            .document_node(&owner_id, parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .inputs
            .clone();
        let owner_metadata = &interface
            .node_metadata(&owner_id, parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .persistent_metadata;
        let owner_storage = owner_metadata
            .storage_id
            .map(StorageNodeId)
            .ok_or(GddArchiveError::MissingGraphNetwork)?;
        let mut input_metadata = owner_metadata.input_metadata.clone();
        if owner_inputs.len() != network.import_count || input_metadata.len() != owner_inputs.len()
        {
            return Err(unsupported());
        }
        edit.edit_slots(&mut owner_inputs);
        edit.edit_slots(&mut input_metadata);
        let original_slots = self
            .document
            .registry()
            .node_instances
            .get(&owner_storage)
            .ok_or(GddArchiveError::MissingGraphNetwork)?
            .inputs();
        if original_slots.len() != network.import_count {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let (original_indices, new_indices) = edit.slot_order(original_slots.len())?;
        let surviving_slots = original_indices
            .into_iter()
            .map(|old| {
                original_slots
                    .get(old)
                    .ok_or(GddArchiveError::IncompleteGraphNodeEdit)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let nested = interface
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        let mut affected = Vec::new();
        for (&node_id, node) in &nested.nodes {
            for (input_index, input) in node.inputs.iter().enumerate() {
                if let NodeInput::Import {
                    import_index: previous,
                    import_type,
                } = input
                {
                    let &next = new_indices.get(*previous).ok_or_else(unsupported)?;
                    if next != Some(*previous)
                        && (interface.is_layer(&node_id, path)
                            || interface.is_chain(&node_id, path))
                    {
                        return Err(unsupported());
                    }
                    affected.push((node_id, input_index, *previous, next, import_type.clone()));
                }
            }
        }
        if nested.exports.iter().any(|input| {
            matches!(input, NodeInput::Import { import_index: previous, .. }
                if new_indices.get(*previous) != Some(&Some(*previous)))
        }) {
            return Err(unsupported());
        }
        affected.sort_unstable_by_key(|&(node_id, input_index, ..)| (node_id, input_index));
        let exports = nested.exports.clone();
        match edit {
            NestedPortEdit::Remove(index) => interface.remove_import(index, path),
            NestedPortEdit::Move { from, to } => {
                // Graphite's destination is the insertion boundary before removing the old slot.
                interface.reorder_import(from, to + usize::from(to > from), path);
            }
        }
        let has_expected_signature = |interface: &NodeNetworkInterface| {
            interface
                .document_node(&owner_id, parent_path)
                .is_some_and(|node| node.inputs == owner_inputs)
                && interface
                    .node_metadata(&owner_id, parent_path)
                    .is_some_and(|metadata| {
                        metadata.persistent_metadata.input_metadata == input_metadata
                    })
                && interface
                    .nested_network(path)
                    .is_some_and(|nested| nested.exports == exports)
        };
        if !has_expected_signature(&interface) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let mut edits = vec![EditorDelta::Graph(RuntimeDelta::SetInputs {
            network_path: parent_path.to_vec(),
            node_id: owner_id,
            inputs: owner_inputs.clone(),
        })];
        let mut rewired_inputs = Vec::with_capacity(affected.len());
        for (node_id, input_index, previous, next, import_type) in affected {
            let connector = InputConnector::Node {
                node_id,
                input_index,
            };
            let input = interface
                .input_from_connector(&connector, path)
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            let persisted_input = if let Some(current) = next {
                // The pinned single-slot delta carries the connection index and retains its stored type.
                matches!(input, NodeInput::Import { import_index, .. } if *import_index == current)
                    .then_some(NodeInput::Import {
                        import_type,
                        import_index: current,
                    })
            } else {
                matches!(input, NodeInput::Value { exposed: true, .. }).then(|| input.clone())
            };
            let persisted_input =
                persisted_input.ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            if next != Some(previous) {
                edits.push(EditorDelta::Graph(RuntimeDelta::SetInput {
                    network_path: path.clone(),
                    node_id,
                    input_index,
                    input: input.clone(),
                }));
            }
            rewired_inputs.push((connector, persisted_input));
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let mut batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let [RegistryDelta::SetNodeInputs { id, inputs }, ..] = batch.ops.as_mut_slice() else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        if *id != owner_storage || inputs.len() != owner_inputs.len() {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        for (input, &original) in inputs.iter_mut().zip(&surviving_slots) {
            if input.input != original.input {
                return Err(GddArchiveError::IncompleteGraphNodeEdit);
            }
            // Source metadata excludes attributes this version does not recognize.
            input.clone_from(original);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        let retained = copy
            .document
            .registry()
            .node_instances
            .get(&owner_storage)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
            .inputs();
        if !has_expected_signature(&actual)
            || rewired_inputs.iter().any(|(connector, input)| {
                actual.input_from_connector(connector, path) != Some(input)
            })
            || retained.len() != owner_inputs.len()
            || retained
                .iter()
                .zip(surviving_slots)
                .any(|(current, original)| {
                    current.input != original.input || current.attributes != original.attributes
                })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_primary_network_import_exposed_in_copy(
        &self,
        network: &GddGraphNetwork,
        exposed: bool,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        let unsupported = || GddArchiveError::UnsupportedGraphPrimaryImportExposure;
        if !network.can_extend_ports_in_copy() {
            return Err(unsupported());
        }
        let path = &network.path;
        let (&owner_id, parent_path) = path.split_last().ok_or_else(unsupported)?;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let primary = InputConnector::primary_input(owner_id);
        let mut input = interface
            .input_from_connector(&primary, parent_path)
            .cloned()
            .ok_or_else(unsupported)?;
        let NodeInput::Value {
            exposed: current, ..
        } = &mut input
        else {
            return Err(unsupported());
        };
        if *current == exposed {
            return Err(GddArchiveError::UnchangedGraphInputExposure);
        }
        *current = exposed;
        let mut consumers = interface
            .outward_wires(path)
            .and_then(|wires| wires.get(&OutputConnector::Import(0)))
            .cloned()
            .ok_or_else(unsupported)?;
        if consumers.iter().any(|connector| {
            connector.node_id().is_none_or(|node_id| {
                interface.is_layer(&node_id, path) || interface.is_chain(&node_id, path)
            })
        }) {
            return Err(unsupported());
        }
        consumers.sort_unstable_by_key(|connector| (connector.node_id(), connector.input_index()));
        interface.set_input(&primary, input.clone(), parent_path);
        let mut edits = Vec::with_capacity(consumers.len() + 1);
        edits.push(write_graph_connector(parent_path, primary, input.clone()));
        let mut detached = Vec::with_capacity(consumers.len());
        // The pinned exposure command detaches old inner consumers in both directions.
        for connector in consumers {
            interface.disconnect_input(&connector, path);
            let fallback = interface
                .input_from_connector(&connector, path)
                .filter(|input| matches!(input, NodeInput::Value { exposed: true, .. }))
                .cloned()
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            edits.push(write_graph_connector(path, connector, fallback.clone()));
            detached.push((connector, fallback));
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.input_from_connector(&primary, parent_path) != Some(&input)
            || detached.iter().any(|(connector, fallback)| {
                actual.input_from_connector(connector, path) != Some(fallback)
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn append_nested_network_export_in_copy(
        &self,
        network: &GddGraphNetwork,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.append_network_export(network, false, budget).await
    }

    pub async fn append_secondary_nested_network_export_in_copy(
        &self,
        network: &GddGraphNetwork,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.append_network_export(network, true, budget).await
    }

    async fn append_network_export(
        &self,
        network: &GddGraphNetwork,
        secondary: bool,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::node::ui::OUTPUT_NAMES;

        self.check_network_edit_target(network)?;
        if !network.can_append_ports_in_copy() {
            return Err(GddArchiveError::UnsupportedGraphExportAppend);
        }
        let path = &network.path;
        let Some((&owner_id, parent_path)) = path.split_last() else {
            return Err(GddArchiveError::UnsupportedGraphExportAppend);
        };
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let mut expected_exports = interface
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .exports
            .clone();
        let original_count = expected_exports.len();
        if original_count != network.export_count {
            return Err(GddArchiveError::StaleGraphNetwork);
        }
        if !self.graph_export_slots_contiguous(network.id, original_count) {
            return Err(GddArchiveError::UnsupportedGraphExportAppend);
        }
        if secondary && original_count == 0 {
            let parent = interface
                .nested_network(parent_path)
                .ok_or(GddArchiveError::StaleGraphNetwork)?;
            if parent.exports.iter().chain(parent.nodes.values().flat_map(|node| &node.inputs))
                .any(|input| matches!(input, NodeInput::Node { node_id, .. } if *node_id == owner_id))
            {
                return Err(GddArchiveError::UnsupportedGraphExportAppend);
            }
            let hidden = NodeInput::value(TaggedValue::None, false);
            interface.add_export(TaggedValue::None, -1, "", path);
            interface.set_input(&InputConnector::Export(0), hidden.clone(), path);
            expected_exports.push(hidden);
        }
        expected_exports.push(NodeInput::value(TaggedValue::None, true));
        interface.add_export(TaggedValue::None, -1, "", path);
        if !interface
            .nested_network(path)
            .is_some_and(|actual| actual.exports == expected_exports)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let owner_metadata = &interface
            .node_metadata(&owner_id, parent_path)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
            .persistent_metadata;
        let owner_storage_id = owner_metadata
            .storage_id
            .map(StorageNodeId)
            .ok_or(GddArchiveError::MissingGraphNetwork)?;
        let output_names = owner_metadata.output_names.clone();
        let mut edits: Vec<_> = expected_exports
            .iter()
            .enumerate()
            .skip(original_count)
            .map(|(index, input)| {
                write_graph_connector(path, InputConnector::Export(index), input.clone())
            })
            .collect();
        edits.push(EditorDelta::NodeMetadata {
            network_path: parent_path.to_vec(),
            node_id: owner_id,
            change: NodeMetadataChange::OutputNames(output_names.clone()),
        });
        network.append_port_deltas(&mut edits);
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let mut batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let Some((name_edit, ports)) = network.port_operations(&mut batch.ops)?.split_last() else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        if !matches!(name_edit,
            RegistryDelta::ChangeNodeAttribute { id, delta }
                if *id == owner_storage_id && delta.key == OUTPUT_NAMES)
            || ports.len() != expected_exports.len() - original_count
            || !ports.iter().enumerate().all(|(offset, operation)| {
                matches!(
                    operation,
                    RegistryDelta::SetNetworkExport { id, index, export: Some(_) }
                        if *id == network.id && *index as usize == original_count + offset
                )
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !actual
            .network_metadata(path)
            .is_some_and(|metadata| metadata.persistent_metadata.reference.is_none())
            || !actual
                .nested_network(path)
                .is_some_and(|nested| nested.exports == expected_exports)
            || !actual
                .node_metadata(&owner_id, parent_path)
                .is_some_and(|metadata| metadata.persistent_metadata.output_names == output_names)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn remove_nested_network_export_in_copy(
        &self,
        network: &GddGraphNetwork,
        export_index: usize,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.edit_nested_network_export_in_copy(
            network,
            NestedPortEdit::Remove(export_index),
            budget,
        )
        .await
    }

    pub async fn move_nested_network_export_in_copy(
        &self,
        network: &GddGraphNetwork,
        from: usize,
        to: usize,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.edit_nested_network_export_in_copy(network, NestedPortEdit::Move { from, to }, budget)
            .await
    }

    async fn edit_nested_network_export_in_copy(
        &self,
        network: &GddGraphNetwork,
        edit: NestedPortEdit,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        let unsupported = || edit.unsupported_export();
        if !network.can_extend_ports_in_copy() || !edit.valid_for(network.export_count) {
            return Err(unsupported());
        }
        let path = &network.path;
        let (&owner_id, parent_path) = path.split_last().ok_or_else(unsupported)?;
        let parent_id = network.parent.ok_or_else(unsupported)?;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let nested = interface
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        let mut exports = nested.exports.clone();
        if exports.len() != network.export_count
            || !self.graph_export_slots_contiguous(network.id, exports.len())
        {
            return Err(unsupported());
        }
        let parent = interface
            .nested_network(parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        if !self.graph_export_slots_contiguous(parent_id, parent.exports.len()) {
            return Err(unsupported());
        }
        let owner_metadata = &interface
            .node_metadata(&owner_id, parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .persistent_metadata;
        let mut names = owner_metadata.output_names.clone();
        match edit {
            NestedPortEdit::Remove(index) => {
                if index < names.len() {
                    names.remove(index);
                }
            }
            NestedPortEdit::Move { from, to } => {
                let name = if from < names.len() {
                    names.remove(from)
                } else {
                    String::new()
                };
                names.insert(to.min(names.len()), name);
            }
        }
        edit.edit_slots(&mut exports);
        let (original_indices, new_indices) = edit.slot_order(network.export_count)?;

        let layout = graph_port_edit_layout(&interface, [path, parent_path])?;
        let connections = interface
            .outward_wires(parent_path)
            .ok_or_else(unsupported)?;
        let mut affected = Vec::new();
        for (previous, next) in new_indices.into_iter().enumerate() {
            let connections = connections
                .get(&OutputConnector::node(owner_id, previous))
                .ok_or_else(unsupported)?;
            affected.extend(
                connections
                    .iter()
                    .map(|&connector| (connector, previous, next)),
            );
        }
        affected
            .sort_unstable_by_key(|(connector, ..)| (connector.node_id(), connector.input_index()));
        match edit {
            NestedPortEdit::Remove(index) => interface.remove_export(index, path),
            NestedPortEdit::Move { from, to } => {
                interface.reorder_export(from, to + usize::from(to > from), path)
            }
        }
        let expected_signature = |interface: &NodeNetworkInterface| {
            interface
                .nested_network(path)
                .is_some_and(|nested| nested.exports == exports)
                && interface
                    .node_metadata(&owner_id, parent_path)
                    .is_some_and(|metadata| metadata.persistent_metadata.output_names == names)
        };
        if !expected_signature(&interface) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        if graph_port_edit_layout(&interface, [path, parent_path])? != layout {
            return Err(unsupported());
        }

        let mut edits: Vec<_> = exports
            .iter()
            .zip(original_indices)
            .enumerate()
            .filter(|(new, (_, old))| new != old)
            .map(|(export_index, (input, _))| {
                EditorDelta::Graph(RuntimeDelta::SetExport {
                    network_path: path.clone(),
                    export_index,
                    input: Some(input.clone()),
                })
            })
            .collect();
        if matches!(edit, NestedPortEdit::Remove(_)) {
            edits.push(EditorDelta::Graph(RuntimeDelta::SetExport {
                network_path: path.clone(),
                export_index: exports.len(),
                input: None,
            }));
        }
        edits.push(EditorDelta::NodeMetadata {
            network_path: parent_path.to_vec(),
            node_id: owner_id,
            change: NodeMetadataChange::OutputNames(names.clone()),
        });
        let mut rewired = Vec::with_capacity(affected.len());
        for (connector, previous, next) in affected {
            let input = interface
                .input_from_connector(&connector, parent_path)
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            let expected_wiring = if let Some(index) = next {
                input == &NodeInput::node(owner_id, index)
            } else {
                matches!(input, NodeInput::Value { exposed: true, .. })
            };
            if !expected_wiring {
                return Err(GddArchiveError::IncompleteGraphNodeEdit);
            }
            if next != Some(previous) {
                edits.push(write_graph_connector(parent_path, connector, input.clone()));
            }
            rewired.push((connector, input.clone()));
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !expected_signature(&actual)
            || rewired.iter().any(|(connector, input)| {
                actual.input_from_connector(connector, parent_path) != Some(input)
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_primary_network_export_exposed_in_copy(
        &self,
        network: &GddGraphNetwork,
        exposed: bool,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        let unsupported = || GddArchiveError::UnsupportedGraphPrimaryExportExposure;
        if !network.can_extend_ports_in_copy() {
            return Err(unsupported());
        }
        let current = network.primary_export_exposed().ok_or_else(unsupported)?;
        if current == exposed {
            return Err(GddArchiveError::UnchangedGraphInputExposure);
        }
        let path = &network.path;
        let (&owner_id, parent_path) = path.split_last().ok_or_else(unsupported)?;
        let parent = network.parent.ok_or_else(unsupported)?;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        if !self.graph_export_slots_contiguous(network.id, network.export_count)
            || !self.graph_export_slots_contiguous(parent, interface.number_of_exports(parent_path))
        {
            return Err(unsupported());
        }
        let layout = graph_port_edit_layout(&interface, [path, parent_path])?;
        let mut consumers = interface
            .outward_wires(parent_path)
            .and_then(|wires| wires.get(&OutputConnector::primary_output(owner_id)))
            .cloned()
            .ok_or_else(unsupported)?;
        consumers.sort_unstable_by_key(|connector| (connector.node_id(), connector.input_index()));
        let primary = InputConnector::Export(0);
        if !exposed {
            interface.disconnect_input(&primary, path);
        }
        let mut input = interface
            .input_from_connector(&primary, path)
            .cloned()
            .ok_or_else(unsupported)?;
        let NodeInput::Value { exposed: state, .. } = &mut input else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        *state = exposed;
        interface.set_input(&primary, input.clone(), path);
        let mut edits = vec![write_graph_connector(path, primary, input.clone())];
        let mut detached = Vec::with_capacity(consumers.len());
        // Changing exposure detaches the enclosing node's primary output consumers in the pinned editor.
        for connector in consumers {
            interface.disconnect_input(&connector, parent_path);
            let fallback = interface
                .input_from_connector(&connector, parent_path)
                .filter(|input| matches!(input, NodeInput::Value { exposed: true, .. }))
                .cloned()
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            edits.push(write_graph_connector(
                parent_path,
                connector,
                fallback.clone(),
            ));
            detached.push((connector, fallback));
        }
        if graph_port_edit_layout(&interface, [path, parent_path])? != layout {
            return Err(unsupported());
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.input_from_connector(&primary, path) != Some(&input)
            || detached.iter().any(|(connector, fallback)| {
                actual.input_from_connector(connector, parent_path) != Some(fallback)
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn rename_nested_network_port_in_copy(
        &self,
        network: &GddGraphNetwork,
        port: ImportOrExport,
        name: String,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::node::{input::ui::NAME, ui::OUTPUT_NAMES};

        self.check_network_edit_target(network)?;
        let port_exists = match port {
            ImportOrExport::Import(index) => index < network.import_count,
            ImportOrExport::Export(index) => index < network.export_count,
        };
        if !network.can_extend_ports_in_copy() || !port_exists {
            return Err(GddArchiveError::UnsupportedGraphPortRename);
        }
        if name.len() as u64 > budget.max_entry_bytes {
            return Err(GddArchiveError::BudgetExceeded);
        }
        let path = &network.path;
        let (&owner_id, parent_path) = path
            .split_last()
            .ok_or(GddArchiveError::UnsupportedGraphPortRename)?;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let metadata = interface
            .node_metadata(&owner_id, parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        let mut expected = metadata.persistent_metadata.clone();
        let owner_storage = expected
            .storage_id
            .map(StorageNodeId)
            .ok_or(GddArchiveError::MissingGraphNetwork)?;
        let original_name = match port {
            ImportOrExport::Import(index) => expected
                .input_metadata
                .get_mut(index)
                .map(|input| &mut input.persistent_metadata.input_name),
            ImportOrExport::Export(index) => expected.output_names.get_mut(index),
        }
        .ok_or(GddArchiveError::UnsupportedGraphPortRename)?;
        if *original_name == name {
            return Err(GddArchiveError::UnchangedGraphPortName);
        }
        original_name.clone_from(&name);
        let owner_inputs = interface
            .document_node(&owner_id, parent_path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .inputs
            .clone();
        let exports = interface
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?
            .exports
            .clone();
        let change = match port {
            ImportOrExport::Import(index) => NodeMetadataChange::InputName {
                index,
                name: name.clone(),
            },
            ImportOrExport::Export(_) => {
                NodeMetadataChange::OutputNames(expected.output_names.clone())
            }
        };
        let has_expected_metadata = move |interface: &NodeNetworkInterface| {
            interface
                .node_metadata(&owner_id, parent_path)
                .is_some_and(|metadata| metadata.persistent_metadata == expected)
        };
        interface.set_import_export_name(name, port, path);
        if !has_expected_metadata(&interface) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let edit = EditorDelta::NodeMetadata {
            network_path: parent_path.to_vec(),
            node_id: owner_id,
            change,
        };
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        let valid_edit = match (port, batch.ops.as_slice()) {
            (
                ImportOrExport::Import(port),
                [RegistryDelta::ChangeNodeInputAttribute { id, index, delta }],
            ) => *id == owner_storage && *index as usize == port && delta.key == NAME,
            (ImportOrExport::Export(_), [RegistryDelta::ChangeNodeAttribute { id, delta }]) => {
                *id == owner_storage && delta.key == OUTPUT_NAMES
            }
            _ => false,
        };
        if !valid_edit {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !has_expected_metadata(&actual)
            || !actual
                .document_node(&owner_id, parent_path)
                .is_some_and(|node| node.inputs == owner_inputs)
            || !actual
                .nested_network(path)
                .is_some_and(|network| network.exports == exports)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    fn graph_export_slots_contiguous(&self, network: NetworkId, count: usize) -> bool {
        self.document
            .registry()
            .networks
            .get(&network)
            .and_then(|network| network.exports.get(..count))
            .is_some_and(|slots| slots.iter().all(|slot| slot.target.is_some()))
    }

    pub async fn connect_numeric_network_export_in_copy(
        &self,
        network: &GddGraphNetwork,
        source: &GddGraphNode,
        export_index: usize,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(source)?;
        self.check_network_edit_target(network)?;
        if !network.can_connect_export_from(source)
            || !network.empty_exports().contains(&export_index)
        {
            return Err(GddArchiveError::UnsupportedGraphExportConnection);
        }
        let path = &network.path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let nested = interface
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        let mut expected_exports = nested.exports.clone();
        let export = expected_exports
            .get_mut(export_index)
            .filter(|input| empty_graph_export(input))
            .ok_or(GddArchiveError::UnsupportedGraphExportConnection)?;
        let imports = enclosing_network_inputs(interface.document_network(), path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        let detached = numeric_preview_network(nested, source.local_id, imports)?;
        typecheck_detached_numeric_network(detached)
            .map_err(GddArchiveError::GraphExportTypeMismatch)?;

        let template = interface
            .create_node_template(&source.local_id, path)
            .ok_or(GddArchiveError::StaleGraphNode)?;
        let wire = NodeInput::node(source.local_id, 0);
        *export = wire.clone();
        interface.create_wire(
            &OutputConnector::primary_output(source.local_id),
            &InputConnector::Export(export_index),
            path,
        );
        if !interface
            .nested_network(path)
            .is_some_and(|nested| nested.exports == expected_exports)
            || interface
                .create_node_template(&source.local_id, path)
                .as_ref()
                != Some(&template)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let edit = EditorDelta::Graph(RuntimeDelta::SetExport {
            network_path: path.clone(),
            export_index,
            input: Some(wire),
        });
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        if !matches!(
            batch.ops.as_slice(),
            [RegistryDelta::SetNetworkExport {
                id, index, export: Some(graphite_document_graph_storage::NodeInput::Node {id: source_id, index: 0})
            }] if *id == network.id && *index as usize == export_index && *source_id == source.storage_id
        ) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !actual
            .nested_network(path)
            .is_some_and(|nested| nested.exports == expected_exports)
            || actual.create_node_template(&source.local_id, path) != Some(template)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn connect_numeric_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        import_index: usize,
        target: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.write_numeric_import_wire(network, import_index, target, false, budget)
            .await
    }

    pub async fn reconnect_numeric_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        import_index: usize,
        target: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.write_numeric_import_wire(network, import_index, target, true, budget)
            .await
    }

    async fn write_numeric_import_wire(
        &self,
        network: &GddGraphNetwork,
        import_index: usize,
        target: &GddGraphNode,
        reconnect: bool,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        self.check_node_edit_target(target)?;
        let unsupported = || GddArchiveError::UnsupportedGraphImportConnection;
        if !network
            .compatible_numeric_imports(target)
            .any(|index| index == import_index)
        {
            return Err(unsupported());
        }
        let path = &network.path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let metadata = interface
            .node_metadata(&target.local_id, path)
            .ok_or_else(unsupported)?
            .persistent_metadata
            .clone();
        let imports =
            enclosing_network_inputs(interface.document_network(), path).ok_or_else(unsupported)?;
        let (input_index, current_value) =
            match (graph_numeric_input(&interface, target), reconnect) {
                (Some(GddNumericInput::Literal(port)), false) if port.exposed => {
                    (port.index, port.value)
                }
                (
                    Some(GddNumericInput::Imported {
                        index,
                        import_index: previous,
                    }),
                    true,
                ) if Some(previous) == target.numeric_import_source() => {
                    let previous = imports
                        .get(previous)
                        .and_then(exposed_numeric_literal)
                        .ok_or_else(unsupported)?;
                    (index, previous)
                }
                _ => return Err(unsupported()),
            };
        let replacement = imports
            .get(import_index)
            .and_then(exposed_numeric_literal)
            .ok_or_else(unsupported)?;
        if !current_value.same_type(&replacement) {
            return Err(unsupported());
        }
        let imports = imports.to_vec();
        let input = InputConnector::node_at_index(target.local_id, input_index);
        let mut edits = Vec::with_capacity(1 + usize::from(reconnect));
        if reconnect {
            interface.disconnect_input(&input, path);
            let fallback = interface
                .input_from_connector(&input, path)
                .filter(|input| matches!(input, NodeInput::Value { exposed: true, .. }))
                .cloned()
                .ok_or(GddArchiveError::UnsupportedGraphDisconnection)?;
            edits.push(write_graph_connector(path, input, fallback));
        }
        let output = OutputConnector::Import(import_index);
        interface.create_wire(&output, &input, path);
        let wire = interface.input_from_connector(&input, path)
            .filter(|input| matches!(input, NodeInput::Import { import_index: actual, .. } if *actual == import_index))
            .cloned()
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        if !interface
            .node_metadata(&target.local_id, path)
            .is_some_and(|node| node.persistent_metadata == metadata)
        {
            return Err(GddArchiveError::UnsupportedGraphConnectionLayout);
        }
        let nested = interface.nested_network(path).ok_or_else(unsupported)?;
        let detached = numeric_preview_network(nested, target.local_id, &imports)?;
        typecheck_detached_numeric_network(detached)
            .map_err(GddArchiveError::GraphImportTypeMismatch)?;
        edits.push(write_graph_connector(path, input, wire));
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let Some((final_wire, fallback)) = batch.ops.split_last() else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        if !matches!(final_wire,
            RegistryDelta::ChangeNodeInput {
                id, index, new_input: graphite_document_graph_storage::NodeInput::Import { index: source },
            } if *id == target.storage_id && *index as usize == input_index && *source as usize == import_index
        ) || fallback.len() != usize::from(reconnect)
            || !fallback.iter().all(|edit| matches!(edit,
                RegistryDelta::ChangeNodeInput {
                    id, index, new_input: graphite_document_graph_storage::NodeInput::Value { exposed: true, .. },
                } if *id == target.storage_id && *index as usize == input_index
            ))
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !matches!(
            actual.input_from_connector(&input, path),
            Some(NodeInput::Import { import_index: actual, .. }) if *actual == import_index
        ) || enclosing_network_inputs(actual.document_network(), path)
            != Some(imports.as_slice())
            || !actual
                .node_metadata(&target.local_id, path)
                .is_some_and(|node| node.persistent_metadata == metadata)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn copy_graph_nodes_to_clipboard(
        &self,
        targets: &[GddGraphNode],
        budget: GddArchiveBudget,
    ) -> Result<String, GddArchiveError> {
        let (_, nodes) = self.graph_templates_for_selection(targets, &budget).await?;
        let encoded = serde_json::to_string(&[ClipboardItem::Nodes(nodes)])
            .map_err(GddArchiveError::GraphClipboardFormat)?;
        let text = format!("graphite: {encoded}");
        if text.len() as u64 > budget.max_entry_bytes {
            return Err(GddArchiveError::BudgetExceeded);
        }
        Ok(text)
    }

    async fn graph_templates_for_selection(
        &self,
        targets: &[GddGraphNode],
        budget: &GddArchiveBudget,
    ) -> Result<(NodeNetworkInterface, Vec<(NodeId, NodeTemplate)>), GddArchiveError> {
        let unsupported = || GddArchiveError::UnsupportedGraphClipboardSelection;
        let first = targets.first().ok_or_else(unsupported)?;
        if targets.len() > budget.max_entries {
            return Err(GddArchiveError::BudgetExceeded);
        }
        let mut ids = BTreeSet::new();
        for target in targets {
            self.check_node_edit_target(target)?;
            if !first.same_network_as(target)
                || !target.can_copy_graph_node()
                || !ids.insert(target.local_id)
            {
                return Err(unsupported());
            }
        }
        let path = &first.network_path;
        let interface = self.runtime_graph().await?.into_interface()?;
        let temporary = ids
            .into_iter()
            .enumerate()
            .map(|(index, node)| (node, NodeId(index as u64)))
            .collect::<HashMap<_, _>>();
        let mut nodes: Vec<_> = interface.copy_nodes(&temporary, path).collect();
        nodes.sort_unstable_by_key(|(id, _)| *id);
        if nodes.len() != targets.len() || !valid_graph_clipboard_nodes(&nodes) {
            return Err(unsupported());
        }
        if graph_clipboard_node_count(&nodes) > budget.max_entries {
            return Err(GddArchiveError::BudgetExceeded);
        }
        Ok((interface, nodes))
    }

    pub async fn duplicate_graph_nodes_in_copy(
        &self,
        network: &GddGraphNetwork,
        targets: &[GddGraphNode],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        if (network.parent.is_some() && !network.can_extend_ports_in_copy())
            || targets.iter().any(|node| node.network_id != network.id)
        {
            return Err(GddArchiveError::UnsupportedGraphClipboardSelection);
        }
        let (interface, nodes) = self.graph_templates_for_selection(targets, &budget).await?;
        self.insert_graph_node_templates_in_copy(network, interface, nodes, budget)
            .await
    }

    pub async fn paste_graph_nodes_in_copy(
        &self,
        network: &GddGraphNetwork,
        clipboard: &str,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        if network.parent.is_some() && !network.can_extend_ports_in_copy() {
            return Err(GddArchiveError::UnsupportedGraphClipboard);
        }
        if clipboard.len() as u64 > budget.max_entry_bytes {
            return Err(GddArchiveError::BudgetExceeded);
        }
        let encoded = clipboard
            .strip_prefix("graphite: ")
            .ok_or(GddArchiveError::UnsupportedGraphClipboard)?;
        let items: [ClipboardItem; 1] =
            serde_json::from_str(encoded).map_err(GddArchiveError::GraphClipboardFormat)?;
        let [ClipboardItem::Nodes(nodes)] = items else {
            return Err(GddArchiveError::UnsupportedGraphClipboard);
        };
        if graph_clipboard_node_count(&nodes) > budget.max_entries {
            return Err(GddArchiveError::BudgetExceeded);
        }
        if !valid_graph_clipboard_nodes(&nodes) {
            return Err(GddArchiveError::UnsupportedGraphClipboard);
        }
        self.insert_graph_node_templates_in_copy(
            network,
            self.runtime_graph().await?.into_interface()?,
            nodes,
            budget,
        )
        .await
    }

    async fn insert_graph_node_templates_in_copy(
        &self,
        network: &GddGraphNetwork,
        mut interface: NodeNetworkInterface,
        mut nodes: Vec<(NodeId, NodeTemplate)>,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        let path = network.path.as_slice();
        let mut used = BTreeSet::new();
        let mut identifiers = HashMap::new();
        for (id, template) in &mut nodes {
            template.normalize_stored_types();
            let inserted = NodeId::new();
            if !used.insert(inserted) || interface.document_node(&inserted, path).is_some() {
                return Err(GddArchiveError::GraphNodeIdentityCollision);
            }
            identifiers.insert(*id, inserted);
        }
        interface.insert_node_group(nodes, identifiers, path);
        let nested = interface
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        let mut expected = BTreeMap::new();
        let mut expected_networks = BTreeSet::new();
        let mut expected_nodes = 0;
        let mut edits = Vec::with_capacity(used.len() * 2);
        for node_id in used {
            let node = interface
                .document_node(&node_id, path)
                .cloned()
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            let metadata = interface
                .node_metadata(&node_id, path)
                .map(|node| node.persistent_metadata.clone())
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            let template = interface
                .create_node_template(&node_id, path)
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            expected_nodes += 1;
            match &node.implementation {
                DocumentNodeImplementation::Network(inner) => {
                    let mut child_path = path.to_vec();
                    child_path.push(node_id);
                    expected_networks.insert(child_path);
                    expected_nodes += inner.nodes.len();
                    if !path.is_empty() || !is_pinned_layer_template(&template) {
                        for &child in inner.nodes.keys() {
                            check_clipboard_numeric_branch(inner, child, &node.inputs)?;
                        }
                    }
                }
                _ => check_clipboard_numeric_branch(nested, node_id, &[])?,
            }
            expected.insert(node_id, template);
            edits.push(EditorDelta::Graph(RuntimeDelta::AddNode {
                network_path: path.to_vec(),
                node_id,
                node: Box::new(node),
            }));
            edits.push(EditorDelta::NodeMetadataSnapshot {
                network_path: path.to_vec(),
                node_id,
                metadata: Box::new(metadata),
            });
        }
        let (mut owner, staged) = self.fork_working_copy(&budget).await?;
        if self.graph_root_needs_initialization() {
            initialize_graph_root_history(&mut owner)?;
        }
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let added_networks = batch
            .ops
            .iter()
            .filter_map(|operation| match operation {
                RegistryDelta::AddNetwork { id, .. } => Some(*id),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let added = batch
            .ops
            .iter()
            .filter_map(|operation| match operation {
                RegistryDelta::AddNode { id, node }
                    if node.network() == network.id || added_networks.contains(&node.network()) =>
                {
                    Some(*id)
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if added.len() != expected_nodes
            || added_networks.len() != expected_networks.len()
            || !batch.ops.iter().all(|operation| match operation {
                RegistryDelta::AddNode { id, .. } => {
                    added.contains(id) && !owner.registry().node_instances.contains_key(id)
                }
                RegistryDelta::ChangeNodeAttribute { id, .. }
                | RegistryDelta::ChangeNodeInputAttribute { id, .. } => added.contains(id),
                RegistryDelta::AddNetwork { id, .. } => !owner.registry().networks.contains_key(id),
                RegistryDelta::ChangeNetworkAttribute { id, .. } => added_networks.contains(id),
                RegistryDelta::AddResource { entry, .. } => {
                    entry.has_embedded_source()
                        && entry
                            .hash
                            .is_some_and(|hash| batch.declarations.bytes.contains_key(&hash))
                }
                _ => false,
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let graph = copy.runtime_graph().await?;
        if graph
            .networks
            .iter()
            .filter(|network| {
                added_networks.contains(&network.network_id)
                    && expected_networks.contains(&network.network_path)
            })
            .count()
            != expected_networks.len()
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        if graph
            .nodes
            .iter()
            .filter(|node| {
                node.network_path.as_slice() == path
                    && added.contains(&node.storage_id)
                    && expected.contains_key(&node.local_id)
            })
            .count()
            != expected.len()
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let actual = graph.into_interface()?;
        for (node_id, template) in expected {
            if actual.create_node_template(&node_id, path) != Some(template) {
                return Err(GddArchiveError::IncompleteGraphNodeEdit);
            }
        }
        Ok(copy)
    }

    pub async fn insert_root_catalog_node_in_copy(
        &self,
        identifier: &str,
        position: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        if self.hot_log_len() != 0 {
            return Err(GddArchiveError::PendingGraphEdits);
        }
        let definition = catalog_copy_node_definition(identifier)
            .ok_or(GddArchiveError::UnsupportedCatalogNodeDefinition)?;
        if identifier == CUSTOM_NODE_IDENTIFIER
            || matches!(
                definition.node_template.node_type_metadata,
                NodeTypePersistentMetadata::Layer(_)
            )
        {
            let inspection = self.graph_inspection().await?;
            let network = inspection
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .ok_or(GddArchiveError::MissingGraphNetwork)?;
            if identifier == CUSTOM_NODE_IDENTIFIER {
                return self
                    .insert_custom_node_in_network_copy(network, position, budget)
                    .await;
            }
            let mut template = definition.default_node_template();
            template.node_type_metadata =
                NodeTypePersistentMetadata::layer(IVec2::from_array(position));
            return self
                .insert_graph_node_templates_in_copy(
                    network,
                    self.runtime_graph().await?.into_interface()?,
                    vec![(NodeId(0), template)],
                    budget,
                )
                .await;
        }
        self.insert_node_template_in_copy(&[], definition.default_node_template(), position, budget)
            .await
    }

    pub async fn insert_custom_node_in_network_copy(
        &self,
        network: &GddGraphNetwork,
        position: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        if !network.can_insert_custom_node_in_copy() {
            return Err(GddArchiveError::UnsupportedCustomNodeScope);
        }
        let definition = custom_node_definition().ok_or(
            GddArchiveError::MissingGraphNodeDefinition(CUSTOM_NODE_NAME),
        )?;
        let mut template = definition.default_node_template();
        template.node_type_metadata = NodeTypePersistentMetadata::node(IVec2::from_array(position));
        let interface = self.runtime_graph().await?.into_interface()?;
        self.insert_graph_node_templates_in_copy(
            network,
            interface,
            vec![(NodeId(0), template)],
            budget,
        )
        .await
    }

    async fn insert_node_template_in_copy(
        &self,
        path: &[NodeId],
        mut template: NodeTemplate,
        position: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        template.node_type_metadata = NodeTypePersistentMetadata::node(IVec2::from_array(position));
        let mut interface = self.runtime_graph().await?.into_interface()?;
        if interface.nested_network(path).is_none() {
            return Err(GddArchiveError::StaleGraphNetwork);
        }
        let node_id = NodeId::new();
        if interface.document_node(&node_id, path).is_some() {
            return Err(GddArchiveError::GraphNodeIdentityCollision);
        }
        if interface
            .insert_node(node_id, template.clone(), path)
            .is_some()
        {
            return Err(GddArchiveError::GraphNodeIdentityCollision);
        }
        self.save_added_graph_node(interface, path, node_id, template, None, budget)
            .await
    }

    pub async fn insert_numeric_node_on_wire_in_copy(
        &self,
        target: &GddGraphNode,
        operation: UnaryNumericOperation,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(target)?;
        let unsupported = || GddArchiveError::UnsupportedGraphWireInsertion;
        let path = target.network_path.as_slice();
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let original_input = graph_numeric_input(&interface, target)
            .filter(|input| !matches!(input, GddNumericInput::Literal(_)))
            .ok_or_else(unsupported)?;
        if !interface.is_absolute(&target.local_id, path) {
            return Err(unsupported());
        }
        let connector = InputConnector::node_at_index(target.local_id, original_input.index());
        let source = interface
            .upstream_output_connector(&connector, path)
            .ok_or_else(unsupported)?;
        if let Some(source) = source.node_id()
            && (interface.is_layer(&source, path) || !interface.is_absolute(&source, path))
        {
            return Err(unsupported());
        }
        let position = interface
            .position(&target.local_id, path)
            .ok_or_else(unsupported)?;
        let horizontal = position.x.checked_sub(4).ok_or_else(unsupported)?;
        let mut template = numeric_node_template(operation)?;
        if !matches!(
            template.inputs.as_slice(),
            [NodeInput::Value { exposed: true, .. }]
        ) {
            return Err(unsupported());
        }
        template.node_type_metadata =
            NodeTypePersistentMetadata::node(IVec2::new(horizontal, position.y));
        let layout = graph_port_edit_layout(&interface, [path])?;
        let node_id = NodeId::new();
        if interface.document_node(&node_id, path).is_some() {
            return Err(GddArchiveError::GraphNodeIdentityCollision);
        }
        if interface.insert_node(node_id, template, path).is_some() {
            return Err(GddArchiveError::GraphNodeIdentityCollision);
        }
        interface.insert_node_between(&node_id, &connector, 0, path);
        if interface.upstream_output_connector(&connector, path)
            != Some(OutputConnector::primary_output(node_id))
            || interface.upstream_output_connector(&InputConnector::primary_input(node_id), path)
                != Some(source)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let mut updated_layout = graph_port_edit_layout(&interface, [path])?;
        if updated_layout.remove(&(path, node_id)).is_none() || updated_layout != layout {
            return Err(unsupported());
        }
        let nested = interface.nested_network(path).ok_or_else(unsupported)?;
        let imports =
            enclosing_network_inputs(interface.document_network(), path).ok_or_else(unsupported)?;
        let detached = numeric_preview_network(nested, target.local_id, imports)?;
        typecheck_detached_numeric_network(detached)
            .map_err(GddArchiveError::GraphWireInsertionTypeMismatch)?;
        let template = interface
            .create_node_template(&node_id, path)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        self.save_added_graph_node(
            interface,
            path,
            node_id,
            template,
            Some(GraphWireInsertion {
                consumer: target.storage_id,
                input: connector,
            }),
            budget,
        )
        .await
    }

    pub async fn preview_graph_numeric_output(
        &self,
        node: &GddGraphNode,
    ) -> Result<GddNumericValue, GddArchiveError> {
        self.prepare_graph_numeric_output(node)
            .await?
            .evaluate()
            .await
    }

    async fn prepare_graph_numeric_output(
        &self,
        node: &GddGraphNode,
    ) -> Result<GddNumericGraphEvaluator, GddArchiveError> {
        self.check_node_edit_target(node)?;
        if !node.can_preview_numeric_output() {
            return Err(GddArchiveError::UnsupportedGraphNumericPreview);
        }
        let graph = self.runtime_graph().await?;
        let source = graph
            .network
            .nested_network(&node.network_path)
            .ok_or(GddArchiveError::StaleGraphNode)?;
        let imports = enclosing_network_inputs(&graph.network, &node.network_path)
            .ok_or(GddArchiveError::StaleGraphNode)?;
        let detached = numeric_preview_network(source, node.local_id, imports)?;
        let evaluator = NumericGraphEvaluator::prepare(detached)
            .await
            .map_err(|error| GddArchiveError::GraphNumericPreviewEvaluation(error.to_string()))?;
        Ok(GddNumericGraphEvaluator {
            evaluator,
            snapshot: self.snapshot.clone(),
            network_path: node.network_path.clone(),
        })
    }

    pub async fn prepare_document_numeric_output(
        &self,
        node: &GddGraphNode,
        output_index: usize,
    ) -> Result<GddNumericGraphEvaluator, GddArchiveError> {
        self.check_node_edit_target(node)?;
        if !node.network_path.is_empty() {
            return Err(GddArchiveError::StaleGraphNode);
        }
        let graph = self.runtime_graph().await?;
        let [
            NodeInput::Node {
                node_id,
                output_index: saved_output,
            },
        ] = graph.network.exports.as_slice()
        else {
            return Err(GddArchiveError::UnsupportedGraphNumericPreview);
        };
        if *node_id != node.local_id || *saved_output != output_index {
            return Err(GddArchiveError::StaleGraphNode);
        }
        if node.nested_network.is_none() && !node.can_preview_numeric_output() {
            return Err(GddArchiveError::UnsupportedGraphNumericPreview);
        }
        let (detached, path) =
            document_numeric_export_network(&graph.network, *node_id, output_index)?;
        let evaluator = NumericGraphEvaluator::prepare(detached)
            .await
            .map_err(|error| document_numeric_compilation_error(error, &path, &graph.nodes))?;
        Ok(GddNumericGraphEvaluator {
            evaluator,
            snapshot: self.snapshot.clone(),
            network_path: path,
        })
    }

    pub async fn render_document_layer_svg(
        &self,
        node: &GddGraphNode,
        configuration: RenderConfig,
        budget: CpuSvgEvaluationBudget,
    ) -> Result<RenderOutput, GddArchiveError> {
        self.check_node_edit_target(node)?;
        if node.layer_preview().is_none() {
            return Err(GddArchiveError::UnsupportedGraphLayerPreview);
        }
        let graph = self.runtime_graph().await?;
        let selected = node.local_id;
        if graph.network.exports != [NodeInput::node(selected, 0)] {
            return Err(GddArchiveError::UnsupportedGraphLayerPreview);
        }
        Self::render_saved_layer_svg(&graph.network, selected, configuration, budget).await
    }

    pub async fn render_document_artboard_canvas_svg(
        &self,
        resolution: UVec2,
        scale: f64,
        camera: Option<PTZ>,
        budget: CpuSvgEvaluationBudget,
    ) -> Result<GddArtboardCanvasOutput, GddArchiveError> {
        if self.hot_log_len() != 0 {
            return Err(GddArchiveError::PendingGraphEdits);
        }
        let graph = self.runtime_graph().await?;
        let [
            NodeInput::Node {
                node_id,
                output_index: 0,
            },
        ] = graph.network.exports.as_slice()
        else {
            return Err(GddArchiveError::UnsupportedGraphLayerPreview);
        };
        let Some((_, SavedGraphicProfile::Layer(GddGraphLayerPreview::Artboard { origin, size }))) =
            graph
                .network
                .nodes
                .get(node_id)
                .and_then(checked_saved_graphic_node)
        else {
            return Err(GddArchiveError::UnsupportedGraphLayerPreview);
        };
        let logical_size = resolution.as_dvec2() / scale;
        if !logical_size.is_finite() || logical_size.min_element() <= 0. {
            return Err(GddArchiveError::GraphLayerPreviewBudget);
        }
        let camera = camera.unwrap_or_else(|| {
            let center = origin + size / 2.;
            let mut camera = PTZ::default();
            if center.abs().max_element() > 0.5 {
                camera.pan = -center;
            }
            camera.set_zoom(
                (logical_size / size).min_element() * VIEWPORT_ZOOM_TO_FIT_PADDING_SCALE_FACTOR,
            );
            camera
        });
        let transform = NavigationMessageHandler::default()
            .calculate_offset_transform(logical_size / 2., &camera);
        let configuration = RenderConfig {
            viewport: Footprint {
                transform,
                resolution,
                ..Footprint::DEFAULT
            },
            scale,
            export_format: SvgExportFormat::Svg,
            for_export: true,
            ..Default::default()
        };
        let node_id = *node_id;
        let render =
            Self::render_saved_layer_svg(&graph.network, node_id, configuration, budget).await?;
        let storage_id = graph
            .nodes
            .iter()
            .find(|entry| entry.local_id == node_id && entry.network_path.is_empty())
            .map(|entry| entry.storage_id);
        let (pick_target, overlay) = if let Some((storage_id, targets)) =
            storage_id.zip(render.metadata.click_targets.get(&node_id))
        {
            let interface = graph.into_interface()?;
            let metadata = &render.metadata;
            let pane_to_document = transform.inverse();
            let parent_to_pane = metadata
                .upstream_footprints
                .get(&node_id)
                .map_or(transform, |footprint| footprint.transform);
            let local = metadata
                .local_transforms
                .get(&node_id)
                .copied()
                .unwrap_or_default();
            let layer_to_document = pane_to_document * parent_to_pane * local;
            if interface.is_layer(&node_id, &[])
                && interface.is_artboard(&node_id, &[])
                && interface.is_visible(&node_id, &[])
                && !targets.is_empty()
                && pane_to_document.is_finite()
                && layer_to_document.inverse().is_finite()
            {
                let bounds = targets
                    .iter()
                    .filter_map(|target| target.bounding_box_with_transform(layer_to_document))
                    .reduce(|[first_min, first_max], [second_min, second_max]| {
                        [first_min.min(second_min), first_max.max(second_max)]
                    });
                let overlay = bounds.map(|bounds| {
                    GddArtboardCanvasOverlay::from_document_bounds(
                        interface.display_name(&node_id, &[]),
                        bounds,
                        transform,
                        camera,
                    )
                });
                let pick = (!interface.is_locked(&node_id, &[])).then(|| GddArtboardCanvasPick {
                    storage_id,
                    snapshot: self.snapshot.clone(),
                    pane_to_document,
                    layer_to_document,
                    targets: targets.clone(),
                });
                (pick, overlay)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        Ok(GddArtboardCanvasOutput {
            render,
            camera,
            artboard_size: size,
            pick_target,
            overlay,
        })
    }

    async fn render_saved_layer_svg(
        network: &NodeNetwork,
        selected: NodeId,
        configuration: RenderConfig,
        budget: CpuSvgEvaluationBudget,
    ) -> Result<RenderOutput, GddArchiveError> {
        validate_svg_configuration(configuration, budget)
            .map_err(|error| GddArchiveError::GraphLayerPreviewEvaluation(error.to_string()))?;
        let mut detached = NodeNetwork {
            exports: vec![NodeInput::node(selected, 0)],
            ..Default::default()
        };
        collect_layer_preview_node(network, selected, &mut detached, &mut BTreeSet::new())?;
        validate_saved_ellipse_radius_sources(&detached).await?;
        evaluate_cpu_svg(&detached, configuration, budget, None)
            .await
            .map_err(|error| GddArchiveError::GraphLayerPreviewEvaluation(error.to_string()))
    }

    pub async fn export_document_artboard_svg(
        &self,
        node: &GddGraphNode,
        budget: CpuSvgEvaluationBudget,
    ) -> Result<RenderOutput, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let Some(GddGraphLayerPreview::Artboard { origin, size }) = node.layer_preview() else {
            return Err(GddArchiveError::UnsupportedArtboardSvgExport);
        };
        // The pinned Artboard export subtracts the document bounds and rounds the resolution.
        let resolution = ((origin + size) - origin).round();
        if resolution.min_element() < 1. || resolution.max_element() > f64::from(u32::MAX) {
            return Err(GddArchiveError::GraphLayerPreviewBudget);
        }
        self.render_document_layer_svg(
            node,
            RenderConfig {
                viewport: Footprint {
                    transform: DAffine2::from_translation(-origin),
                    resolution: resolution.as_uvec2(),
                    ..Footprint::DEFAULT
                },
                scale: 1.,
                export_format: SvgExportFormat::Svg,
                for_export: true,
                ..Default::default()
            },
            budget,
        )
        .await
    }

    pub async fn set_document_artboard_in_copy(
        &self,
        node: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        if !node.can_set_document_artboard() {
            return Err(GddArchiveError::UnsupportedDocumentArtboard);
        }
        let path = &[][..];
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let root = interface.document_network();
        if !empty_document_output(&root.exports)
            || !interface.is_layer(&node.local_id, path)
            || !matches!(
                root.nodes
                    .get(&node.local_id)
                    .and_then(checked_saved_graphic_node),
                Some((
                    _,
                    SavedGraphicProfile::Layer(GddGraphLayerPreview::Artboard { .. })
                ))
            )
        {
            return Err(GddArchiveError::UnsupportedDocumentArtboard);
        }
        let connector = InputConnector::Export(0);
        let wire = NodeInput::node(node.local_id, 0);
        let layout = graph_port_edit_layout(&interface, [path])?;
        if root.exports.is_empty() {
            interface.add_export(TaggedValue::None, -1, "", path);
        }
        interface.create_wire(
            &OutputConnector::primary_output(node.local_id),
            &connector,
            path,
        );
        if interface.document_network().exports != [wire.clone()]
            || graph_port_edit_layout(&interface, [path])? != layout
        {
            return Err(GddArchiveError::UnsupportedDocumentArtboard);
        }
        let edit = write_graph_connector(path, connector, wire.clone());
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        if !matches!(batch.ops.as_slice(),
            [RegistryDelta::SetNetworkExport {
                id: ROOT_NETWORK, index: 0,
                export: Some(graphite_document_graph_storage::NodeInput::Node { id, index: 0 }),
            }] if *id == node.storage_id
        ) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.document_network().exports != [wire]
            || graph_port_edit_layout(&actual, [path])? != layout
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn connect_graphic_content_in_copy(
        &self,
        node: &GddGraphNode,
        source: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::node::ui::{IS_LAYER, POSITION};

        self.check_node_edit_target(node)?;
        self.check_node_edit_target(source)?;
        let input_index = node
            .graphic_content_input_from(source)
            .ok_or(GddArchiveError::UnsupportedGraphContentConnection)?;
        let path = node.network_path.as_slice();
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let exports = interface.document_network().exports.clone();
        let connector = InputConnector::node_at_index(node.local_id, input_index);
        if !matches!(
            interface.input_from_connector(&connector, path),
            Some(NodeInput::Value { exposed: true, .. })
        ) {
            return Err(GddArchiveError::UnsupportedGraphContentConnection);
        }
        let layout = graph_port_edit_layout(&interface, [path])?;
        let wire = NodeInput::node(source.local_id, 0);
        interface.create_wire(
            &OutputConnector::primary_output(source.local_id),
            &connector,
            path,
        );
        if interface.input_from_connector(&connector, path) != Some(&wire) {
            return Err(GddArchiveError::UnsupportedGraphContentConnection);
        }
        let mut detached = NodeNetwork::default();
        collect_layer_preview_node(
            interface.document_network(),
            node.local_id,
            &mut detached,
            &mut BTreeSet::new(),
        )?;
        let upstream_shape = if matches!(source.graphic_profile, Some(SavedGraphicProfile::Fill)) {
            let Some(NodeInput::Node {
                node_id,
                output_index: 0,
            }) = detached
                .nodes
                .get(&source.local_id)
                .and_then(|node| node.inputs.get(fill::ContentInput::INDEX))
            else {
                return Err(GddArchiveError::UnsupportedGraphContentConnection);
            };
            if !matches!(
                detached.nodes.get(node_id).map(|node| &node.implementation),
                Some(DocumentNodeImplementation::ProtoNode(identifier))
                    if *identifier == rectangle::IDENTIFIER || *identifier == ellipse::IDENTIFIER
            ) {
                return Err(GddArchiveError::UnsupportedGraphContentConnection);
            }
            Some(*node_id)
        } else {
            None
        };
        let updated_layout = graph_port_edit_layout(&interface, [path])?;
        let copy = if layout == updated_layout {
            self.finish_graph_input(node, interface, input_index, None, budget)
                .await?
        } else {
            let mut edits = vec![write_graph_connector(path, connector, wire.clone())];
            let mut changed = Vec::new();
            for (&key, position) in &updated_layout {
                if layout.get(&key) == Some(position) {
                    continue;
                }
                let (_, local_id) = key;
                if (local_id != source.local_id && Some(local_id) != upstream_shape)
                    || !matches!(layout.get(&key), Some(NodeTypePersistentMetadata::Node(node)) if matches!(node.position(), NodePosition::Absolute(_)))
                    || !matches!(position, NodeTypePersistentMetadata::Node(node) if *node.position() == NodePosition::Chain)
                {
                    return Err(GddArchiveError::UnsupportedGraphContentConnection);
                }
                let metadata = &interface
                    .node_metadata(&local_id, path)
                    .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
                    .persistent_metadata;
                changed.push(
                    metadata
                        .storage_id
                        .map(StorageNodeId)
                        .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?,
                );
                edits.push(EditorDelta::NodeMetadata {
                    network_path: path.to_vec(),
                    node_id: local_id,
                    change: NodeMetadataChange::NodeType(position.clone()),
                });
            }
            let (owner, staged) = self.fork_working_copy(&budget).await?;
            let mut batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
            let Some((wire_edit, layout_edits)) = batch.ops.split_first() else {
                return Err(GddArchiveError::IncompleteGraphNodeEdit);
            };
            let (pairs, remainder) = layout_edits.as_chunks::<2>();
            if !matches!(
                wire_edit,
                RegistryDelta::ChangeNodeInput {
                    id, index,
                    new_input: graphite_document_graph_storage::NodeInput::Node { id: source_id, index: 0 },
                } if *id == node.storage_id && usize::try_from(*index).ok() == Some(input_index) && *source_id == source.storage_id
            ) || !remainder.is_empty()
                || pairs.len() != changed.len()
                || !pairs.iter().zip(changed).all(|(pair, expected)| {
                    matches!(
                        pair,
                        [RegistryDelta::ChangeNodeAttribute { id, delta },
                         RegistryDelta::ChangeNodeAttribute { id: mode_id, delta: mode }]
                            if *id == expected && *mode_id == expected
                                && delta.key == POSITION && mode.key == IS_LAYER && mode.value.is_none()
                    )
                })
            {
                return Err(GddArchiveError::IncompleteGraphNodeEdit);
            }
            batch.ops.retain(|operation| {
                !matches!(
                    operation,
                    RegistryDelta::ChangeNodeAttribute { delta, .. } if delta.key == IS_LAYER
                )
            });
            finish_graph_node_edit(owner, staged, batch, &budget).await?
        };
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.document_network().exports != exports
            || actual.input_from_connector(&connector, path) != Some(&wire)
            || graph_port_edit_layout(&actual, [path])? != updated_layout
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn duplicate_graph_numeric_node_in_copy(
        &self,
        node: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        if !node.can_duplicate_in_copy() {
            return Err(GddArchiveError::UnsupportedGraphNodeDuplication);
        }
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let new_id = NodeId::new();
        if interface.document_node(&new_id, path).is_some() {
            return Err(GddArchiveError::GraphNodeIdentityCollision);
        }
        let copied = interface
            .copy_nodes(&HashMap::from([(node.local_id, NodeId(0))]), path)
            .collect::<Vec<_>>();
        let [(temporary_id, template)] = copied.as_slice() else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        let temporary_id = *temporary_id;
        let template = template.clone();
        interface.insert_node_group(copied, HashMap::from([(temporary_id, new_id)]), path);
        self.save_added_graph_node(interface, path, new_id, template, None, budget)
            .await
    }

    async fn save_added_graph_node(
        &self,
        interface: NodeNetworkInterface,
        path: &[NodeId],
        node_id: NodeId,
        template: NodeTemplate,
        insertion: Option<GraphWireInsertion>,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        let node = interface
            .document_node(&node_id, path)
            .cloned()
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        let metadata = interface
            .node_metadata(&node_id, path)
            .map(|metadata| metadata.persistent_metadata.clone())
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        let mut edits = vec![
            EditorDelta::Graph(RuntimeDelta::AddNode {
                network_path: path.to_vec(),
                node_id,
                node: Box::new(node),
            }),
            EditorDelta::NodeMetadataSnapshot {
                network_path: path.to_vec(),
                node_id,
                metadata: Box::new(metadata),
            },
        ];
        let output = NodeInput::node(node_id, 0);
        if let Some(insertion) = &insertion {
            edits.push(write_graph_connector(path, insertion.input, output.clone()));
        }
        let (mut owner, staged) = self.fork_working_copy(&budget).await?;
        if self.graph_root_needs_initialization() {
            initialize_graph_root_history(&mut owner)?;
        }
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        drop(interface);
        let [RegistryDelta::AddNode { id, .. }, rest @ ..] = batch.ops.as_slice() else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        let id = *id;
        if owner.registry().node_instances.contains_key(&id) {
            return Err(GddArchiveError::GraphNodeIdentityCollision);
        }
        if !rest.iter().all(|operation| match operation {
            RegistryDelta::AddResource { entry, .. } => {
                entry.has_embedded_source()
                    && entry
                        .hash
                        .is_some_and(|hash| batch.declarations.bytes.contains_key(&hash))
            }
            RegistryDelta::ChangeNodeAttribute { id: target, .. }
            | RegistryDelta::ChangeNodeInputAttribute { id: target, .. } => *target == id,
            RegistryDelta::ChangeNodeInput {
                id: target,
                index,
                new_input:
                    graphite_document_graph_storage::NodeInput::Node {
                        id: upstream,
                        index: 0,
                    },
            } => insertion.as_ref().is_some_and(|insertion| {
                *target == insertion.consumer
                    && *index as usize == insertion.input.input_index()
                    && *upstream == id
            }),
            _ => false,
        }) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let graph = copy.runtime_graph().await?;
        if !graph.nodes.iter().any(|node| {
            node.storage_id == id && node.local_id == node_id && node.network_path == path
        }) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let actual = graph.into_interface()?;
        if actual.create_node_template(&node_id, path) != Some(template)
            || insertion.is_some_and(|insertion| {
                actual.input_from_connector(&insertion.input, path) != Some(&output)
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn delete_isolated_graph_node_in_copy(
        &self,
        node: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        if !node.can_delete_in_copy() {
            return Err(GddArchiveError::UnsupportedGraphNodeDeletion);
        }
        self.remove_graph_nodes_in_copy(
            std::slice::from_ref(node),
            &node.network_path,
            self.runtime_graph().await?.into_interface()?,
            budget,
        )
        .await
    }

    pub async fn cut_graph_nodes_in_copy(
        &self,
        targets: &[GddGraphNode],
        budget: GddArchiveBudget,
    ) -> Result<(Self, String), GddArchiveError> {
        let clipboard = self.copy_graph_nodes_to_clipboard(targets, budget).await?;
        if let [node] = targets {
            let copy = self
                .delete_isolated_graph_node_in_copy(node, budget)
                .await?;
            return Ok((copy, clipboard));
        }
        let path = &targets
            .first()
            .ok_or(GddArchiveError::UnsupportedGraphCutSelection)?
            .network_path;
        let mut original = self.runtime_graph().await?.into_interface()?;
        let identifiers = targets.iter().map(|node| node.local_id).collect();
        if original.previewing(path) != Previewing::No
            || !targets.iter().all(|target| {
                target.can_supply_numeric_export
                    && original
                        .document_node(&target.local_id, path)
                        .is_some_and(|node| {
                            node.inputs
                                .iter()
                                .all(|input| graph_clipboard_input(input, &identifiers, &[]))
                        })
            })
            || !original.outward_wires(path).is_some_and(|wires| {
                wires
                    .iter()
                    .filter(|(output, _)| {
                        output
                            .node_id()
                            .is_some_and(|node| identifiers.contains(&node))
                    })
                    .flat_map(|(_, inputs)| inputs)
                    .all(|input| {
                        input
                            .node_id()
                            .is_some_and(|node| identifiers.contains(&node))
                    })
            })
        {
            return Err(GddArchiveError::UnsupportedGraphCutSelection);
        }
        let network = original
            .nested_network(path)
            .ok_or(GddArchiveError::StaleGraphNetwork)?;
        for target in targets {
            check_clipboard_numeric_branch(network, target.local_id, &[])?;
        }
        let copy = self
            .remove_graph_nodes_in_copy(targets, path, original, budget)
            .await?;
        Ok((copy, clipboard))
    }

    async fn remove_graph_nodes_in_copy(
        &self,
        targets: &[GddGraphNode],
        path: &[NodeId],
        original: NodeNetworkInterface,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::network::PINNED_ORDER;

        let network_id = targets
            .first()
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
            .network_id;
        let expected = targets
            .iter()
            .map(|node| node.storage_id)
            .collect::<BTreeSet<_>>();
        let mut interface = original.clone();
        let old_order = interface
            .network_metadata(path)
            .ok_or(GddArchiveError::StaleGraphNode)?
            .persistent_metadata
            .pinned_node_order
            .clone();
        interface.delete_nodes(
            targets.iter().map(|node| node.local_id).collect(),
            false,
            path,
        );
        if targets
            .iter()
            .any(|node| interface.document_node(&node.local_id, path).is_some())
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let new_order = &interface
            .network_metadata(path)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
            .persistent_metadata
            .pinned_node_order;
        let order_changed = *new_order != old_order;
        let mut edits = targets
            .iter()
            .map(|node| {
                EditorDelta::Graph(RuntimeDelta::RemoveNode {
                    network_path: path.to_vec(),
                    node_id: node.local_id,
                })
            })
            .collect::<Vec<_>>();
        if order_changed {
            edits.push(EditorDelta::NetworkMetadata {
                network_path: path.to_vec(),
                change: NetworkMetadataChange::PinnedOrder(new_order.clone()),
            });
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        // The source deletion removes the persistent storage ID along with the node.
        let batch = prepare_graph_node_edit(&owner, &original, &edits)?;
        let mut removed = BTreeSet::new();
        let mut order_updates = 0;
        if !batch.ops.iter().all(|operation| match operation {
            RegistryDelta::RemoveNode { id, snapshot } => {
                removed.insert(*id)
                    && owner
                        .registry()
                        .node_instances
                        .get(id)
                        .is_some_and(|stored| snapshot.value_equal(stored))
            }
            RegistryDelta::ChangeNetworkAttribute { id, delta }
                if *id == network_id && delta.key == PINNED_ORDER =>
            {
                order_updates += 1;
                true
            }
            RegistryDelta::RemoveResource { .. } => true,
            _ => false,
        }) || removed != expected
            || order_updates != usize::from(order_changed)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if targets
            .iter()
            .any(|node| actual.document_node(&node.local_id, path).is_some())
            || !actual.network_metadata(path).is_some_and(|metadata| {
                metadata.persistent_metadata.pinned_node_order == *new_order
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn move_graph_node_in_copy(
        &self,
        node: &GddGraphNode,
        destination: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.move_graph_nodes_to_positions(&[(node, destination)], budget)
            .await
    }

    pub async fn move_graph_nodes_in_copy(
        &self,
        nodes: &[GddGraphNode],
        offset: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        let positions = nodes
            .iter()
            .map(|node| {
                self.check_node_edit_target(node)?;
                let [x, y] = node
                    .grid_position()
                    .ok_or(GddArchiveError::UnsupportedGraphGridMove)?;
                let destination = [
                    x.checked_add(offset[0])
                        .ok_or(GddArchiveError::GraphGridMoveOutOfRange)?,
                    y.checked_add(offset[1])
                        .ok_or(GddArchiveError::GraphGridMoveOutOfRange)?,
                ];
                Ok((node, destination))
            })
            .collect::<Result<Vec<_>, GddArchiveError>>()?;
        self.move_graph_nodes_to_positions(&positions, budget).await
    }

    async fn move_graph_nodes_to_positions(
        &self,
        positions: &[(&GddGraphNode, [i32; 2])],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::node::ui::{IS_LAYER, POSITION};

        let (first, _) = positions
            .first()
            .ok_or(GddArchiveError::UnsupportedGraphGridMove)?;
        let path = &first.network_path;
        let mut ids = BTreeSet::new();
        let mut shifts = Vec::with_capacity(positions.len());
        for &(node, destination) in positions {
            self.check_node_edit_target(node)?;
            if node.network_path != *path || !ids.insert(node.storage_id) {
                return Err(GddArchiveError::UnsupportedGraphGridMove);
            }
            shifts.push(node.grid_move_offset(destination)?);
        }
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let mut edits = Vec::with_capacity(positions.len());
        for (&(node, destination), shift) in positions.iter().zip(shifts) {
            interface.shift_absolute_node_position(&node.local_id, shift, path);
            if interface.position(&node.local_id, path) != Some(IVec2::from_array(destination)) {
                return Err(GddArchiveError::IncompleteGraphNodeEdit);
            }
            let metadata = interface
                .node_metadata(&node.local_id, path)
                .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
            edits.push(EditorDelta::NodeMetadata {
                network_path: path.clone(),
                node_id: node.local_id,
                change: NodeMetadataChange::NodeType(
                    metadata.persistent_metadata.node_type_metadata.clone(),
                ),
            });
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let mut batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let (pairs, remainder) = batch.ops.as_chunks::<2>();
        if !remainder.is_empty()
            || pairs.len() != positions.len()
            || !positions.iter().zip(pairs).all(|((node, _), operations)| {
                matches!(operations,
                    [RegistryDelta::ChangeNodeAttribute { id, delta },
                     RegistryDelta::ChangeNodeAttribute { id: mode_id, delta: mode }]
                        if *id == node.storage_id && *mode_id == *id
                            && delta.key == POSITION && mode.key == IS_LAYER && mode.value.is_none()
                )
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        // The layer mode is an independent stored attribute and a grid move leaves it intact.
        batch.ops.retain(|operation| matches!(operation, RegistryDelta::ChangeNodeAttribute { delta, .. } if delta.key == POSITION));
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !positions.iter().all(|(node, destination)| {
            actual.position(&node.local_id, path) == Some(IVec2::from_array(*destination))
        }) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn rename_graph_node_in_copy(
        &self,
        node: &GddGraphNode,
        name: String,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::node::ui::DISPLAY_NAME;

        self.check_node_edit_target(node)?;
        if node.name == name {
            return Err(GddArchiveError::UnchangedGraphNodeName);
        }
        if name.len() as u64 > budget.max_entry_bytes {
            return Err(GddArchiveError::BudgetExceeded);
        }
        let interface = self.runtime_graph().await?.into_interface()?;
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let edit = EditorDelta::NodeMetadata {
            network_path: node.network_path.clone(),
            node_id: node.local_id,
            change: NodeMetadataChange::DisplayName(name.clone()),
        };
        let batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        if !matches!(
            batch.ops.as_slice(),
            [RegistryDelta::ChangeNodeAttribute { id, delta }]
                if *id == node.storage_id && delta.key == DISPLAY_NAME
        ) {
            return Err(GddArchiveError::StaleGraphNode);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let updated_name = copy
            .document
            .registry()
            .node_instances
            .get(&node.storage_id)
            .map(|node| {
                node.attributes()
                    .get_typed::<String>(DISPLAY_NAME)
                    .unwrap_or_default()
            });
        if updated_name.as_deref() != Some(name.as_str()) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn edit_root_layer_in_copy(
        &self,
        node: &GddGraphNode,
        change: GddRootLayerChange,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::{
            Value,
            attr::node::{VISIBLE, ui::LOCKED},
        };

        self.check_node_edit_target(node)?;
        let node_id = node.local_id;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        if !node.network_path.is_empty() || !interface.is_layer(&node_id, &[]) {
            return Err(GddArchiveError::UnsupportedRootLayerEdit);
        }
        let (key, value, edit) = match change {
            GddRootLayerChange::Visibility(visible)
                if interface.is_visible(&node_id, &[]) != visible =>
            {
                interface.set_visibility(&node_id, &[], visible);
                (
                    VISIBLE,
                    (!visible).then_some(Value::Bool(false)),
                    EditorDelta::Graph(RuntimeDelta::SetVisibility {
                        network_path: Vec::new(),
                        node_id,
                        visible,
                    }),
                )
            }
            GddRootLayerChange::Locked(locked) if interface.is_locked(&node_id, &[]) != locked => {
                interface.set_locked(&node_id, &[], locked);
                (
                    LOCKED,
                    locked.then_some(Value::Bool(true)),
                    EditorDelta::NodeMetadata {
                        network_path: Vec::new(),
                        node_id,
                        change: NodeMetadataChange::Locked(locked),
                    },
                )
            }
            _ => return Err(GddArchiveError::UnchangedRootLayer),
        };
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        if !matches!(batch.ops.as_slice(), [RegistryDelta::ChangeNodeAttribute { id, delta }]
            if *id == node.storage_id && delta.key == key && delta.value == value)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        let matches = match change {
            GddRootLayerChange::Visibility(visible) => actual.is_visible(&node_id, &[]) == visible,
            GddRootLayerChange::Locked(locked) => actual.is_locked(&node_id, &[]) == locked,
        };
        if !matches {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_graph_node_pinned_in_copy(
        &self,
        node: &GddGraphNode,
        pinned: bool,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::{network::PINNED_ORDER, node::ui::PINNED};

        self.check_node_edit_target(node)?;
        if node.pinned == pinned {
            return Err(GddArchiveError::UnchangedGraphNodePin);
        }
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let original_order = interface
            .network_metadata(path)
            .ok_or(GddArchiveError::StaleGraphNode)?
            .persistent_metadata
            .pinned_node_order
            .clone();
        interface.set_pinned(&node.local_id, path, pinned);
        let new_order = &interface
            .network_metadata(path)
            .ok_or(GddArchiveError::StaleGraphNode)?
            .persistent_metadata
            .pinned_node_order;
        let mut edits = vec![EditorDelta::NodeMetadata {
            network_path: path.clone(),
            node_id: node.local_id,
            change: NodeMetadataChange::Pinned(pinned),
        }];
        let order_changed = original_order != *new_order;
        if order_changed {
            edits.push(EditorDelta::NetworkMetadata {
                network_path: path.clone(),
                change: NetworkMetadataChange::PinnedOrder(new_order.clone()),
            });
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let network = owner
            .registry()
            .node_instances
            .get(&node.storage_id)
            .ok_or(GddArchiveError::StaleGraphNode)?
            .network();
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let [RegistryDelta::ChangeNodeAttribute { id, delta }, rest @ ..] = batch.ops.as_slice()
        else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        let valid_order = match (order_changed, rest) {
            (false, []) => true,
            (true, [RegistryDelta::ChangeNetworkAttribute { id, delta }]) => {
                *id == network && delta.key == PINNED_ORDER
            }
            _ => false,
        };
        if *id != node.storage_id || delta.key != PINNED || !valid_order {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let expected_order = interface.ordered_pinned_nodes(path);
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.ordered_pinned_nodes(path) != expected_order
            || copy
                .document
                .registry()
                .node_instances
                .get(&node.storage_id)
                .map(|stored| {
                    stored
                        .attributes()
                        .get_typed::<bool>(PINNED)
                        .unwrap_or_default()
                })
                != Some(pinned)
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn move_pinned_graph_node_in_copy(
        &self,
        node: &GddGraphNode,
        movement: GddPinnedNodeMove,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::network::PINNED_ORDER;

        self.check_node_edit_target(node)?;
        let insertion_index = node
            .pinned_insertion_index(movement)
            .ok_or(GddArchiveError::InvalidPinnedNodeMove)?;
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        interface.reorder_pinned_node(node.local_id, insertion_index, path);
        let order = &interface
            .network_metadata(path)
            .ok_or(GddArchiveError::StaleGraphNode)?
            .persistent_metadata
            .pinned_node_order;
        let expected = interface.ordered_pinned_nodes(path);
        let edit = EditorDelta::NetworkMetadata {
            network_path: path.clone(),
            change: NetworkMetadataChange::PinnedOrder(order.clone()),
        };
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let network = owner
            .registry()
            .node_instances
            .get(&node.storage_id)
            .ok_or(GddArchiveError::StaleGraphNode)?
            .network();
        let batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        if !matches!(
            batch.ops.as_slice(),
            [RegistryDelta::ChangeNetworkAttribute { id, delta }]
                if *id == network && delta.key == PINNED_ORDER
        ) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.ordered_pinned_nodes(path) != expected {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_graph_numeric_input_in_copy(
        &self,
        node: &GddGraphNode,
        value: GddNumericValue,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let interface = self.runtime_graph().await?.into_interface()?;
        let port = graph_numeric_port(&interface, node)
            .ok_or(GddArchiveError::UnsupportedGraphNumericInput)?;
        if port.exposed {
            return Err(GddArchiveError::ExposedGraphNumericInput);
        }
        let tagged_value = match (&port.value, &value) {
            (GddNumericValue::Number(_), GddNumericValue::Number(number)) if number.is_finite() => {
                TaggedValue::Number(*number)
            }
            (GddNumericValue::Number(_), GddNumericValue::Number(_)) => {
                return Err(GddArchiveError::InvalidGraphNumber);
            }
            (GddNumericValue::Integer(_), GddNumericValue::Integer(number)) => {
                TaggedValue::Integer(*number)
            }
            (GddNumericValue::Vector(_), GddNumericValue::Vector(vector))
                if vector.iter().all(|component| component.is_finite()) =>
            {
                TaggedValue::DVec2(DVec2::from_array(*vector))
            }
            (GddNumericValue::Vector(_), GddNumericValue::Vector(_)) => {
                return Err(GddArchiveError::InvalidGraphVector);
            }
            (GddNumericValue::NumberList(_), GddNumericValue::NumberList(numbers))
                if numbers.iter().all(|number| number.is_finite()) =>
            {
                TaggedValue::Numbers(numbers.to_vec())
            }
            (GddNumericValue::NumberList(_), GddNumericValue::NumberList(_)) => {
                return Err(GddArchiveError::InvalidGraphNumberList);
            }
            _ => return Err(GddArchiveError::GraphInputTypeChanged),
        };
        let input = NodeInput::value(tagged_value, false);
        self.write_graph_numeric_port(
            node,
            interface,
            GddNumericPort { value, ..port },
            input,
            budget,
        )
        .await
    }

    pub async fn set_graph_boolean_input_in_copy(
        &self,
        node: &GddGraphNode,
        value: bool,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let previous = node
            .boolean_input()
            .ok_or(GddArchiveError::UnsupportedGraphBooleanInput)?;
        if value == previous.value {
            return Err(GddArchiveError::UnchangedGraphInput);
        }
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let connector = InputConnector::node_at_index(node.local_id, previous.index);
        let boolean_literal = |interface: &NodeNetworkInterface| {
            interface
                .document_node(&node.local_id, path)
                .and_then(graph_boolean_literal)
        };
        if boolean_literal(&interface) != Some(previous) {
            return Err(GddArchiveError::UnsupportedGraphBooleanInput);
        }
        interface.set_input(
            &connector,
            NodeInput::value(TaggedValue::Bool(value), false),
            path,
        );
        let expected = Some(GddGraphBooleanInput { value, ..previous });
        if boolean_literal(&interface) != expected {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = self
            .finish_graph_input(node, interface, previous.index, None, budget)
            .await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if boolean_literal(&actual) != expected {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_artboard_geometry_in_copy(
        &self,
        node: &GddGraphNode,
        requested: GddArtboardGeometry,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let previous = node
            .artboard_geometry()
            .ok_or(GddArchiveError::UnsupportedArtboardGeometry)?;
        let same_vector = |left: DVec2, right: DVec2| {
            left.to_array().map(f64::to_bits) == right.to_array().map(f64::to_bits)
        };
        let rounded_edit = |original: DVec2, entered: DVec2| {
            // The pinned integer widgets round edited values, while legacy literals remain exact.
            DVec2::from_array([(original.x, entered.x), (original.y, entered.y)].map(
                |(original, entered)| {
                    if original.to_bits() == entered.to_bits() {
                        original
                    } else {
                        entered.round()
                    }
                },
            ))
        };
        let proposed = GddArtboardGeometry {
            location: rounded_edit(previous.location, requested.location),
            dimensions: rounded_edit(previous.dimensions, requested.dimensions),
        };
        if proposed.normalized().is_none() {
            return Err(GddArchiveError::InvalidArtboardGeometry);
        }
        let path = node.network_path.as_slice();
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let matching_geometry = |interface: &NodeNetworkInterface,
                                 expected: GddArtboardGeometry| {
            interface
                .document_node(&node.local_id, path)
                .and_then(GddArtboardGeometry::hidden)
                .is_some_and(|actual| {
                    same_vector(actual.location, expected.location)
                        && same_vector(actual.dimensions, expected.dimensions)
                })
        };
        if !matching_geometry(&interface, previous) {
            return Err(GddArchiveError::UnsupportedArtboardGeometry);
        }
        let mut edits = Vec::with_capacity(2);
        let mut indices = Vec::with_capacity(2);
        for (index, before, after) in [
            (
                ARTBOARD_LOCATION_INPUT,
                previous.location,
                proposed.location,
            ),
            (
                ARTBOARD_DIMENSIONS_INPUT,
                previous.dimensions,
                proposed.dimensions,
            ),
        ] {
            if same_vector(before, after) {
                continue;
            }
            let connector = InputConnector::node_at_index(node.local_id, index);
            let input = NodeInput::value(TaggedValue::DVec2(after), false);
            interface.set_input(&connector, input.clone(), path);
            edits.push(write_graph_connector(path, connector, input));
            indices.push(index);
        }
        if edits.is_empty() {
            return Err(GddArchiveError::UnchangedGraphInput);
        }
        if !matching_geometry(&interface, proposed) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        if batch.ops.len() != indices.len()
            || !batch.ops.iter().zip(indices).all(|(operation, expected)| {
                matches!(
                    operation,
                    RegistryDelta::ChangeNodeInput { id, index, .. }
                        if *id == node.storage_id && usize::try_from(*index).ok() == Some(expected)
                )
            })
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !matching_geometry(&actual, proposed) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_artboard_background_in_copy(
        &self,
        node: &GddGraphNode,
        channels: [f32; 4],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let original = node
            .artboard_background()
            .ok_or(GddArchiveError::UnsupportedArtboardBackground)?;
        let current = original.to_gamma_srgb_channels();
        let changed: [bool; 4] = std::array::from_fn(|index| current[index] != channels[index]);
        if changed
            .into_iter()
            .zip(channels)
            .any(|(changed, channel)| changed && !(0.0..=1.0).contains(&channel))
        {
            return Err(GddArchiveError::InvalidArtboardBackground);
        }
        let [red, green, blue, alpha] = channels;
        let entered = Color::from_gamma_srgb_channels(red, green, blue, alpha);
        let original_linear = [original.r(), original.g(), original.b(), original.a()];
        let entered_linear = [entered.r(), entered.g(), entered.b(), entered.a()];
        // Decoding an unchanged display value can change the saved linear RGB bits.
        let [red, green, blue, alpha] = std::array::from_fn(|index| {
            if changed[index] {
                entered_linear[index]
            } else {
                original_linear[index]
            }
        });
        let color = Color::from_rgbaf32_unchecked(red, green, blue, alpha);
        if color == original {
            return Err(GddArchiveError::UnchangedGraphInput);
        }
        let path = node.network_path.as_slice();
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let background = |interface: &NodeNetworkInterface| {
            interface
                .document_node(&node.local_id, path)
                .and_then(hidden_artboard_background)
        };
        if background(&interface) != Some(original) {
            return Err(GddArchiveError::UnsupportedArtboardBackground);
        }
        interface.set_input(
            &InputConnector::node_at_index(node.local_id, ARTBOARD_BACKGROUND_INPUT),
            NodeInput::value(TaggedValue::Color(color), false),
            path,
        );
        if background(&interface) != Some(color) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = self
            .finish_graph_input(node, interface, ARTBOARD_BACKGROUND_INPUT, None, budget)
            .await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if background(&actual) != Some(color) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_ellipse_radius_in_copy(
        &self,
        node: &GddGraphNode,
        axis: GddEllipseAxis,
        value: f64,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let previous = node
            .ellipse_radii()
            .ok_or(GddArchiveError::UnsupportedEllipseRadius)?;
        if !valid_saved_ellipse_radius(value) {
            return Err(GddArchiveError::InvalidEllipseRadius);
        }
        let [mut radius_x, mut radius_y] = previous;
        let (index, radius) = match axis {
            GddEllipseAxis::X => (ellipse::RadiusXInput::INDEX, &mut radius_x),
            GddEllipseAxis::Y => (ellipse::RadiusYInput::INDEX, &mut radius_y),
        };
        if *radius == value {
            return Err(GddArchiveError::UnchangedEllipseRadius);
        }
        *radius = value;
        let expected = [radius_x, radius_y].map(f64::to_bits);
        let path = node.network_path.as_slice();
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let exact_radii = |interface: &NodeNetworkInterface| {
            let document = interface.document_node(&node.local_id, path)?;
            match checked_saved_graphic_node(document)?.1 {
                SavedGraphicProfile::Ellipse(radii) => Some(radii.map(f64::to_bits)),
                _ => None,
            }
        };
        if exact_radii(&interface) != Some(previous.map(f64::to_bits)) {
            return Err(GddArchiveError::UnsupportedEllipseRadius);
        }
        interface.set_input(
            &InputConnector::node_at_index(node.local_id, index),
            NodeInput::value(TaggedValue::Number(value), false),
            path,
        );
        if exact_radii(&interface) != Some(expected) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = self
            .finish_graph_input(node, interface, index, None, budget)
            .await?;
        if exact_radii(&copy.runtime_graph().await?.into_interface()?) != Some(expected) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn set_graph_numeric_input_exposed_in_copy(
        &self,
        node: &GddGraphNode,
        exposed: bool,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let interface = self.runtime_graph().await?.into_interface()?;
        let port = graph_numeric_port(&interface, node)
            .ok_or(GddArchiveError::UnsupportedGraphNumericInput)?;
        if exposed == port.exposed {
            return Err(GddArchiveError::UnchangedGraphInputExposure);
        }
        let connector = InputConnector::node_at_index(node.local_id, port.index);
        let mut input = interface
            .input_from_connector(&connector, &node.network_path)
            .cloned()
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        let NodeInput::Value { exposed: state, .. } = &mut input else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        *state = exposed;
        self.write_graph_numeric_port(
            node,
            interface,
            GddNumericPort { exposed, ..port },
            input,
            budget,
        )
        .await
    }

    pub async fn disconnect_graph_numeric_input_in_copy(
        &self,
        node: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let (index, upstream) = match graph_numeric_input(&interface, node) {
            Some(GddNumericInput::Connected { index, upstream }) => (index, Some(upstream)),
            Some(GddNumericInput::Imported {
                index,
                import_index,
            }) if enclosing_network_inputs(interface.document_network(), path)
                .and_then(|inputs| inputs.get(import_index))
                .is_some() =>
            {
                (index, None)
            }
            _ => return Err(GddArchiveError::UnsupportedGraphConnection),
        };
        let connector = InputConnector::node_at_index(node.local_id, index);
        let upstream_metadata = if let Some(upstream) = upstream {
            if !is_only_graph_wire(&mut interface, path, upstream, connector) {
                return Err(GddArchiveError::UnsupportedGraphDisconnection);
            }
            let metadata = interface
                .node_metadata(&upstream, path)
                .ok_or(GddArchiveError::UnsupportedGraphConnection)?
                .persistent_metadata
                .clone();
            Some((upstream, metadata))
        } else {
            None
        };
        interface.disconnect_input(&connector, path);
        let expected = graph_numeric_port(&interface, node)
            .filter(|port| port.index == index && port.exposed)
            .ok_or(GddArchiveError::UnsupportedGraphDisconnection)?;
        if let Some((upstream, original)) = upstream_metadata
            && interface
                .node_metadata(&upstream, path)
                .is_none_or(|metadata| metadata.persistent_metadata != original)
        {
            return Err(GddArchiveError::UnsupportedGraphDisconnection);
        }
        self.finish_graph_numeric_input(
            node,
            interface,
            GddNumericInput::Literal(expected),
            None,
            budget,
        )
        .await
    }

    pub async fn connect_graph_numeric_input_in_copy(
        &self,
        node: &GddGraphNode,
        source: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_node_edit_target(node)?;
        self.check_node_edit_target(source)?;
        if !node.can_connect_numeric_input_from(source) {
            return Err(GddArchiveError::InvalidGraphConnection);
        }
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let port = graph_numeric_port(&interface, node)
            .filter(|port| port.exposed)
            .ok_or(GddArchiveError::InvalidGraphConnection)?;
        let upstream = source.local_id;
        if interface.is_chain(&upstream, path) || interface.is_chain(&node.local_id, path) {
            return Err(GddArchiveError::UnsupportedGraphConnectionLayout);
        }
        let upstream_metadata = interface
            .node_metadata(&upstream, path)
            .ok_or(GddArchiveError::InvalidGraphConnection)?
            .persistent_metadata
            .clone();
        let connector = InputConnector::node_at_index(node.local_id, port.index);
        interface.create_wire(&OutputConnector::primary_output(upstream), &connector, path);
        if interface
            .node_metadata(&upstream, path)
            .is_none_or(|metadata| metadata.persistent_metadata != upstream_metadata)
        {
            return Err(GddArchiveError::UnsupportedGraphConnectionLayout);
        }
        let expected = GddNumericInput::Connected {
            index: port.index,
            upstream,
        };
        self.finish_graph_numeric_input(node, interface, expected, None, budget)
            .await
    }

    pub async fn reconnect_graph_numeric_input_in_copy(
        &self,
        node: &GddGraphNode,
        previous: &GddGraphNode,
        replacement: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        for target in [node, previous, replacement] {
            self.check_node_edit_target(target)?;
        }
        if !node.can_reconnect_numeric_input_from(previous, replacement) {
            return Err(GddArchiveError::InvalidGraphConnection);
        }
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let Some(GddNumericInput::Connected { index, upstream }) =
            graph_numeric_input(&interface, node)
        else {
            return Err(GddArchiveError::InvalidGraphConnection);
        };
        let connector = InputConnector::node_at_index(node.local_id, index);
        if upstream != previous.local_id
            || interface.is_chain(&node.local_id, path)
            || interface.is_chain(&replacement.local_id, path)
            || !is_only_graph_wire(&mut interface, path, upstream, connector)
        {
            return Err(GddArchiveError::UnsupportedGraphConnectionLayout);
        }
        let endpoints = [upstream, replacement.local_id];
        let persistent_metadata = |interface: &NodeNetworkInterface| {
            endpoints.map(|local_id| {
                interface
                    .node_metadata(&local_id, path)
                    .map(|metadata| metadata.persistent_metadata.clone())
            })
        };
        let original_metadata = persistent_metadata(&interface);
        if original_metadata.iter().any(Option::is_none) {
            return Err(GddArchiveError::InvalidGraphConnection);
        }
        interface.disconnect_input(&connector, path);
        let intermediate = interface
            .input_from_connector(&connector, path)
            .filter(|input| matches!(input, NodeInput::Value { exposed: true, .. }))
            .cloned()
            .ok_or(GddArchiveError::UnsupportedGraphDisconnection)?;
        interface.create_wire(
            &OutputConnector::primary_output(replacement.local_id),
            &connector,
            path,
        );
        if persistent_metadata(&interface) != original_metadata {
            return Err(GddArchiveError::UnsupportedGraphConnectionLayout);
        }
        let expected = GddNumericInput::Connected {
            index,
            upstream: replacement.local_id,
        };
        self.finish_graph_numeric_input(node, interface, expected, Some(intermediate), budget)
            .await
    }

    async fn write_graph_numeric_port(
        &self,
        node: &GddGraphNode,
        mut interface: NodeNetworkInterface,
        expected: GddNumericPort,
        input: NodeInput,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        let path = &node.network_path;
        let input_index = expected.index;
        let connector = InputConnector::node_at_index(node.local_id, input_index);
        if interface.input_from_connector(&connector, path) == Some(&input) {
            return Err(GddArchiveError::UnchangedGraphInput);
        }
        interface.set_input(&connector, input, path);
        self.finish_graph_numeric_input(
            node,
            interface,
            GddNumericInput::Literal(expected),
            None,
            budget,
        )
        .await
    }

    async fn finish_graph_numeric_input(
        &self,
        node: &GddGraphNode,
        interface: NodeNetworkInterface,
        expected: GddNumericInput,
        intermediate: Option<NodeInput>,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        if !graph_numeric_input(&interface, node).is_some_and(|actual| actual.same_input(&expected))
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = self
            .finish_graph_input(node, interface, expected.index(), intermediate, budget)
            .await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !graph_numeric_input(&actual, node).is_some_and(|actual| actual.same_input(&expected)) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    async fn finish_graph_input(
        &self,
        node: &GddGraphNode,
        interface: NodeNetworkInterface,
        input_index: usize,
        intermediate: Option<NodeInput>,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        let path = &node.network_path;
        let connector = InputConnector::node_at_index(node.local_id, input_index);
        let input = interface
            .input_from_connector(&connector, path)
            .cloned()
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        let edits = intermediate
            .into_iter()
            .chain([input])
            .map(|input| {
                EditorDelta::Graph(RuntimeDelta::SetInput {
                    network_path: path.clone(),
                    node_id: node.local_id,
                    input_index,
                    input,
                })
            })
            .collect::<Vec<_>>();
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let writes_expected_slot = |operation: &RegistryDelta| {
            matches!(
                operation,
                RegistryDelta::ChangeNodeInput { id, index, .. }
                    if *id == node.storage_id && usize::try_from(*index).ok() == Some(input_index)
            )
        };
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        if batch.ops.len() != edits.len() || !batch.ops.iter().all(writes_expected_slot) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        finish_graph_node_edit(owner, staged, batch, &budget).await
    }

    fn check_node_edit_target(&self, node: &GddGraphNode) -> Result<(), GddArchiveError> {
        if !Arc::ptr_eq(&self.snapshot, &node.snapshot) {
            return Err(GddArchiveError::StaleGraphNode);
        }
        if self.hot_log_len() != 0 {
            return Err(GddArchiveError::PendingGraphEdits);
        }
        Ok(())
    }

    fn check_network_edit_target(&self, network: &GddGraphNetwork) -> Result<(), GddArchiveError> {
        if !Arc::ptr_eq(&self.snapshot, &network.snapshot) {
            return Err(GddArchiveError::StaleGraphNetwork);
        }
        if self.hot_log_len() != 0 {
            return Err(GddArchiveError::PendingGraphEdits);
        }
        Ok(())
    }

    async fn runtime_graph(&self) -> Result<GddRuntimeGraph, GddArchiveError> {
        if self.graph_root_needs_initialization() {
            return Ok(GddRuntimeGraph {
                network: NodeNetwork::default(),
                nodes: Vec::new(),
                networks: Vec::new(),
            });
        }
        let declarations = self.document.declarations(&self.document).await;
        let (network, nodes, networks) = self
            .document
            .registry()
            .to_runtime_with_full_metadata(&declarations)
            .map_err(GddArchiveError::RuntimeRegistry)?;
        Ok(GddRuntimeGraph {
            network,
            nodes,
            networks,
        })
    }

    fn graph_root_needs_initialization(&self) -> bool {
        let registry = self.document.registry();
        registry.networks.is_empty() && registry.node_instances.is_empty()
    }

    pub async fn recover_pending_edits(
        &self,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        if self.hot_log_len() == 0 {
            return Err(GddArchiveError::NoPendingEdits);
        }
        let (mut owner, staged) = self.fork_working_copy(&budget).await?;
        owner.retire_pending_interaction()?;
        let recovered = staged.reopen_after_edit(&budget).await?;
        if recovered.hot_log_len() != 0
            || !recovered
                .document
                .registry()
                .value_equal(self.document.registry())
        {
            return Err(GddArchiveError::IncompleteRecovery);
        }
        Ok(recovered)
    }

    async fn fork_working_copy(
        &self,
        budget: &GddArchiveBudget,
    ) -> Result<(GddV1, StagedArchive), GddArchiveError> {
        let synthesized_history =
            self.recovery.source == GddRecoverySource::RegistryOnly && self.history_len() > 0;
        let entries = if synthesized_history {
            // Registry-only recovery synthesizes its history in memory. Materialize that lineage
            // before appending new revisions whose parents refer to it.
            let bytes = self
                .document
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &HashMapResourceStorage::new(),
                    None,
                )
                .await?;
            let entries = stage_zip(
                &bytes,
                &GddArchiveBudget {
                    max_archive_bytes: bytes.len(),
                    ..*budget
                },
            )?;
            self.restore_export_entries(entries, budget, true)?
        } else {
            self.copy_entries(budget)?.finish()?
        };
        let mut owner =
            GddV1::open_in(AnyContainer::Memory(entries.working.clone()), GddV1Layout).await?;
        owner
            .session()
            .verify_history()
            .map_err(graphite_document_format::Error::from)?;
        if synthesized_history && let Some(base) = owner.session().head_rev() {
            use graphite_document_graph_storage::{Value, attr};

            // The synthetic base has no interaction marker for a later edit to undo to.
            owner.annotate_delta(base, attr::delta::INTERACTION_END, Value::Bool(true))?;
        }
        Ok((owner, entries))
    }

    fn copy_entries<'budget>(
        &self,
        budget: &'budget GddArchiveBudget,
    ) -> Result<StagedEntries<'budget>, GddArchiveError> {
        let mut entries = StagedEntries::new(budget);
        for name in &self.entries.names {
            entries.install(name, self.read_entry(name)?.as_slice())?;
        }
        Ok(entries)
    }

    pub fn has_saved_view(&self) -> bool {
        !self.document.view_settings().is_empty()
            || !self.document.network_view_settings().is_empty()
    }

    pub async fn saved_view(&self) -> Result<GddSavedView, GddArchiveError> {
        let settings = self.document.view_settings();
        let document = (!settings.is_empty()).then(|| {
            let mut owner = DocumentMessageHandler::default();
            owner.apply_stored_document_settings(settings);
            GddDocumentView {
                camera: owner.document_ptz,
                rulers_visible: owner.rulers_visible,
            }
        });
        let network_settings = self.document.network_view_settings();
        let mut unresolved = network_settings.keys().copied().collect::<BTreeSet<_>>();
        let mut networks = Vec::new();
        if !network_settings.is_empty() {
            let graph = self.runtime_graph().await?;
            let ids = network_ids_from_entries(&graph.networks);
            let mut interface = graph.into_interface()?;
            apply_network_view_settings(&mut interface, &ids, network_settings);
            for (path, id) in ids {
                if network_settings.contains_key(&id)
                    && let Some(metadata) = interface.network_metadata(&path)
                {
                    unresolved.remove(&id);
                    let name = match path.split_last() {
                        Some((node, parent)) => interface.display_name(node, parent),
                        None => "Root graph".into(),
                    };
                    networks.push(GddNetworkView {
                        id,
                        path,
                        name,
                        navigation: metadata.persistent_metadata.navigation_metadata.clone(),
                        previewing: metadata.persistent_metadata.previewing,
                    });
                }
            }
            networks.sort_by_key(|network| (!network.path.is_empty(), network.id.0));
        }
        Ok(GddSavedView {
            document,
            networks,
            unresolved_networks: unresolved.len(),
        })
    }

    pub fn missing_linked_resources(&self) -> Result<BTreeSet<ResourceHash>, GddArchiveError> {
        let mut missing = BTreeSet::new();
        for resource in self.document.registry().resources.values() {
            if resource.has_embedded_source() {
                continue;
            }
            let hash = resource
                .hash
                .ok_or(GddArchiveError::UnresolvedLinkedResource)?;
            if !self
                .entries
                .working
                .exists(&GddV1Layout.resource_path(&hash))
            {
                missing.insert(hash);
            }
        }
        Ok(missing)
    }

    pub async fn with_embedded_linked_resources(
        &self,
        content: GddExportContent,
        approved_files: Vec<Vec<u8>>,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.with_export_attachments(content, Some(approved_files), None, budget)
            .await
    }

    pub async fn with_export_attachments(
        &self,
        content: GddExportContent,
        approved_resources: Option<Vec<Vec<u8>>>,
        legacy_document: Option<&PreservedLegacyDocument>,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        content.validate()?;
        if !content.include_registry && self.history_len() == 0 {
            return Err(GddArchiveError::NoExportableHistory);
        }
        let layout = GddV1Layout;
        let legacy_path = layout.legacy_path();
        let include_all = content.include_registry && content.include_history;
        let supplied_legacy = legacy_document.map(PreservedLegacyDocument::original_bytes);
        let original_legacy = ((include_all || supplied_legacy.is_some())
            && self.entries.names.contains(legacy_path))
        .then(|| self.read_entry(legacy_path))
        .transpose()?;
        if let Some(supplied) = supplied_legacy
            && let Some(original) = &original_legacy
            && supplied != original.as_slice()
        {
            return Err(GddArchiveError::ConflictingLegacyDocument);
        }
        let legacy = supplied_legacy.or(original_legacy.as_ref().map(ByteHolder::as_slice));
        let Some(approved_resources) = approved_resources else {
            if include_all {
                let mut entries = self.copy_entries(&budget)?;
                if let Some(legacy) = legacy
                    && original_legacy.is_none()
                {
                    entries.install(legacy_path, legacy)?;
                }
                return open_staged_gdd(entries.finish()?).await;
            }
            let bytes = self
                .export_filtered_archive(ArchiveFormat::Zip, content, legacy, &budget)
                .await?;
            return open_staged_gdd(stage_zip(&bytes, &budget)?).await;
        };
        let missing = self.missing_linked_resources()?;
        let sources = approved_resource_storage(missing, approved_resources, &budget)?;
        let bytes = self
            .document
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions {
                    embed_all_resources: true,
                    ..content.options()
                },
                &sources,
                legacy,
            )
            .await?;
        let exported = stage_zip(
            &bytes,
            &GddArchiveBudget {
                max_archive_bytes: bytes.len(),
                ..budget
            },
        )?;
        let exported = self.restore_export_entries(exported, &budget, include_all)?;
        open_staged_gdd(exported).await
    }

    fn restore_export_entries(
        &self,
        exported: StagedArchive,
        budget: &GddArchiveBudget,
        include_opaque: bool,
    ) -> Result<StagedArchive, GddArchiveError> {
        let layout = GddV1Layout;
        let codecs = self.manifest().codecs;
        let manifest_path = path_for(layout.manifest_basename(), MANIFEST_CODEC);
        let owned = [
            path_for(layout.registry_basename(), codecs.registry),
            path_for(layout.history_basename(), codecs.history),
            path_for(layout.hot_log_basename(), codecs.hot_log),
            path_for(layout.session_basename(), codecs.session),
        ];
        let mut entries = StagedEntries {
            budget,
            working: exported.working,
            names: exported.names,
            total_bytes: exported.total_bytes,
        };
        let previous_length = entries.working.read(&manifest_path)?.as_slice().len() as u64;
        entries.total_bytes = entries
            .total_bytes
            .checked_sub(previous_length)
            .ok_or(GddArchiveError::InconsistentEntries)?;
        entries.names.remove(&manifest_path);
        entries.install(&manifest_path, self.read_entry(&manifest_path)?.as_slice())?;
        if include_opaque {
            for path in &self.entries.names {
                if !owned.contains(path) && !entries.names.contains(path) {
                    entries.install(path, self.read_entry(path)?.as_slice())?;
                }
            }
        }
        entries.finish()
    }

    pub fn folder_files(&self) -> Result<Vec<(PathBuf, Vec<u8>)>, ContainerError> {
        self.entries.folder_files()
    }

    pub async fn export_folder_files_with_content(
        &self,
        content: GddExportContent,
        budget: GddArchiveBudget,
    ) -> Result<Vec<(PathBuf, Vec<u8>)>, GddArchiveError> {
        if content.include_registry && content.include_history {
            return Ok(self.folder_files()?);
        }
        let bytes = self
            .export_filtered_archive(ArchiveFormat::Zip, content, None, &budget)
            .await?;
        Ok(stage_zip(&bytes, &budget)?.folder_files()?)
    }

    pub fn export_archive(
        &self,
        format: ArchiveFormat,
        max_archive_bytes: usize,
    ) -> Result<Vec<u8>, GddArchiveError> {
        match format {
            ArchiveFormat::Zip => self.write_archive::<Zip>(max_archive_bytes),
            ArchiveFormat::Xz => self.write_archive::<Xz>(max_archive_bytes),
        }
        .map_err(GddArchiveError::Export)
    }

    pub async fn export_archive_with_content(
        &self,
        format: ArchiveFormat,
        content: GddExportContent,
        budget: GddArchiveBudget,
    ) -> Result<Vec<u8>, GddArchiveError> {
        if content.include_registry && content.include_history {
            return self.export_archive(format, budget.max_archive_bytes);
        }
        self.export_filtered_archive(format, content, None, &budget)
            .await
    }

    async fn export_filtered_archive(
        &self,
        format: ArchiveFormat,
        content: GddExportContent,
        legacy_document: Option<&[u8]>,
        budget: &GddArchiveBudget,
    ) -> Result<Vec<u8>, GddArchiveError> {
        content.validate()?;
        if content.include_history && self.history_len() == 0 {
            return Err(GddArchiveError::NoExportableHistory);
        }
        let format = match format {
            ArchiveFormat::Zip => ExportFormat::Zip,
            ArchiveFormat::Xz => ExportFormat::Xz,
        };
        let bytes = self
            .document
            .export_to_bytes(
                format,
                content.options(),
                &HashMapResourceStorage::new(),
                legacy_document,
            )
            .await?;
        if bytes.len() > budget.max_archive_bytes {
            return Err(GddArchiveError::BudgetExceeded);
        }
        Ok(bytes)
    }

    fn write_archive<Format: Archive>(
        &self,
        max_archive_bytes: usize,
    ) -> Result<Vec<u8>, ContainerError>
    where
        Format::Writer<BoundedArchive>: ArchiveWriter<Sink = BoundedArchive>,
    {
        let mut archive = Format::writer(BoundedArchive {
            bytes: Cursor::new(Vec::new()),
            limit: max_archive_bytes as u64,
        })?;
        for path in &self.entries.names {
            archive.write_entry(path, self.read_entry(path)?.as_slice())?;
        }
        Ok(archive.finish_into()?.bytes.into_inner())
    }
}

pub fn can_insert_graphite_catalog_node(identifier: &str) -> bool {
    catalog_copy_node_definition(identifier).is_some()
}

fn catalog_copy_node_definition(identifier: &str) -> Option<&'static DocumentNodeDefinition> {
    if identifier == CUSTOM_NODE_IDENTIFIER {
        return custom_node_definition();
    }
    if identifier == MERGE_NODE_IDENTIFIER {
        return merge_node_definition();
    }
    if identifier == ARTBOARD_NODE_IDENTIFIER {
        return artboard_node_definition();
    }
    let identifier = identifier.strip_prefix("PROTONODE:")?;
    let identifier = ProtoNodeIdentifier::with_owned_string(identifier.to_owned());
    let definition = resolve_proto_node_type(identifier.clone())?;
    let template = &definition.node_template;
    (!definition.category.is_empty()
        && matches!(&template.implementation, NodeTemplateImplementation::ProtoNode(actual) if actual == &identifier)
        && matches!(template.node_type_metadata, NodeTypePersistentMetadata::Node(_))
        && template.inputs.iter().all(|input| matches!(input, NodeInput::Value { .. })))
    .then_some(definition)
}

fn is_pinned_layer_template(template: &NodeTemplate) -> bool {
    if !matches!(
        template.node_type_metadata,
        NodeTypePersistentMetadata::Layer(_)
    ) {
        return false;
    }
    let NodeTemplateImplementation::Network(network) = &template.implementation else {
        return false;
    };
    let definition = match network.reference.as_deref() {
        Some(MERGE_NODE_NAME) => merge_node_definition(),
        Some(ARTBOARD_NODE_NAME) => artboard_node_definition(),
        _ => None,
    };
    definition.is_some_and(|definition| {
        let mut expected = definition.default_node_template();
        expected.normalize_stored_types();
        expected.node_type_metadata = template.node_type_metadata.clone();
        expected == *template
    })
}

fn numeric_node_identifier(operation: UnaryNumericOperation) -> ProtoNodeIdentifier {
    use graphite_math_nodes::{absolute_value, ceiling, floor, round, sign};

    match operation {
        UnaryNumericOperation::AbsoluteValue => absolute_value::IDENTIFIER,
        UnaryNumericOperation::Ceiling => ceiling::IDENTIFIER,
        UnaryNumericOperation::Floor => floor::IDENTIFIER,
        UnaryNumericOperation::Round => round::IDENTIFIER,
        UnaryNumericOperation::Sign => sign::IDENTIFIER,
    }
}

fn numeric_node_template(
    operation: UnaryNumericOperation,
) -> Result<NodeTemplate, GddArchiveError> {
    let identifier = numeric_node_identifier(operation);
    let definition = resolve_proto_node_type(identifier.clone()).ok_or(
        GddArchiveError::MissingGraphNodeDefinition(operation.name()),
    )?;
    let template = definition.default_node_template();
    if !matches!(
        &template.implementation,
        NodeTemplateImplementation::ProtoNode(actual) if actual == &identifier
    ) || !matches!(template.inputs.as_slice(), [NodeInput::Value { .. }])
    {
        return Err(GddArchiveError::MissingGraphNodeDefinition(
            operation.name(),
        ));
    }
    Ok(template)
}

fn valid_graph_clipboard_nodes(nodes: &[(NodeId, NodeTemplate)]) -> bool {
    let identifiers = nodes.iter().map(|(id, _)| *id).collect::<BTreeSet<_>>();
    !identifiers.is_empty()
        && identifiers.len() == nodes.len()
        && nodes.iter().all(|(_, node)| {
            graph_clipboard_node_position(node)
                && node
                    .inputs
                    .iter()
                    .all(|input| graph_clipboard_input(input, &identifiers, &[]))
                && match &node.implementation {
                    NodeTemplateImplementation::ProtoNode(identifier) => {
                        numeric_proto_input_count(identifier) == Some(node.inputs.len())
                            && node.context_features == Default::default()
                    }
                    NodeTemplateImplementation::Network(inner) => {
                        if inner.reference.is_some()
                            || inner.previewing != Previewing::No
                            || inner.navigation_metadata != NavigationMetadata::default()
                            || !inner.scope_injections.is_empty()
                            || !node
                                .inputs
                                .iter()
                                .all(|input| matches!(input, NodeInput::Value { .. }))
                        {
                            return false;
                        }
                        let ids = inner.nodes.keys().copied().collect::<BTreeSet<_>>();
                        inner.nodes.values().all(|child| {
                            graph_clipboard_node_position(child)
                                && matches!(
                                    &child.implementation,
                                    NodeTemplateImplementation::ProtoNode(identifier)
                                        if numeric_proto_input_count(identifier)
                                            == Some(child.inputs.len())
                                            && child.context_features == Default::default()
                                )
                                && child
                                    .inputs
                                    .iter()
                                    .all(|input| graph_clipboard_input(input, &ids, &node.inputs))
                        }) && inner
                            .exports
                            .iter()
                            .all(|input| graph_clipboard_input(input, &ids, &[]))
                    }
                    NodeTemplateImplementation::Extract => false,
                }
        })
}

fn project_graph_input(
    index: usize,
    input: &NodeInput,
    name: Option<&str>,
    scope: NetworkId,
    node_ids: &BTreeMap<(NetworkId, NodeId), StorageNodeId>,
) -> GddGraphInputPort {
    let source = match input {
        NodeInput::Node {
            node_id,
            output_index,
        } => Some(node_ids.get(&(scope, *node_id)).map_or(
            GddGraphPortSource::UnavailableNode {
                local_id: *node_id,
                output_index: *output_index,
            },
            |id| GddGraphPortSource::Node {
                storage_id: *id,
                output_index: *output_index,
            },
        )),
        NodeInput::Import { import_index, .. } => Some(GddGraphPortSource::Import(*import_index)),
        _ => None,
    };
    let stored_type = match input {
        NodeInput::Value { tagged_value, .. } => Some(tagged_value.ty()),
        NodeInput::Import { import_type, .. } => Some(import_type.clone()),
        NodeInput::Inline(code) => Some(code.ty.clone()),
        NodeInput::Reflection(metadata) => Some(metadata.ty()),
        NodeInput::Node { .. } | NodeInput::Scope(_) => None,
    };
    GddGraphInputPort {
        index,
        name: name.filter(|name| !name.is_empty()).map(str::to_owned),
        exposed: input.is_exposed(),
        stored_type,
        source,
    }
}

fn finite_graph_point(point: Option<DVec2>) -> Option<[f64; 2]> {
    point
        .filter(|point| point.is_finite())
        .map(|point| point.to_array())
}

fn project_root_layers(
    interface: &mut NodeNetworkInterface,
    ids: &[(NodeId, StorageNodeId)],
) -> Result<Vec<GddRootLayer>, GddArchiveError> {
    if !ids.iter().any(|(node, _)| interface.is_layer(node, &[])) {
        return Ok(Vec::new());
    }
    // The pinned traversal assumes acyclic inputs and reserves both ends of the layer ID range.
    if [NodeId(0), NodeId(u64::MAX)]
        .iter()
        .any(|node| interface.document_node(node, &[]).is_some() && interface.is_layer(node, &[]))
        || !interface.document_network().is_acyclic()
    {
        return Err(GddArchiveError::InvalidRootLayerGraph);
    }
    interface.load_structure();
    let metadata = interface.document_metadata();
    let selection = interface.selected_nodes();
    let mut pending = Vec::new();
    if let Some(layer) = LayerNodeIdentifier::ROOT_PARENT.last_child(metadata) {
        pending.extend(
            layer
                .downstream_siblings(metadata)
                .map(|layer| (layer, 0, None)),
        );
    }
    let mut layers = Vec::new();
    while let Some((layer, depth, parent)) = pending.pop() {
        let id = layer.to_node();
        let storage_id = ids
            .binary_search_by_key(&id, |(node, _)| *node)
            .ok()
            .and_then(|index| ids.get(index))
            .map(|(_, storage_id)| *storage_id)
            .ok_or(GddArchiveError::InvalidRootLayerGraph)?;
        layers.push(GddRootLayer {
            storage_id,
            parent,
            depth,
            visible: interface.is_visible(&id, &[]),
            locked: interface.is_locked(&id, &[]),
            select_all_eligible: !interface.is_artboard(&id, &[])
                && selection.layer_visible(layer, interface)
                && !selection.layer_locked(layer, interface),
        });
        if let Some(child) = layer.last_child(metadata) {
            pending.extend(
                child
                    .downstream_siblings(metadata)
                    .map(|child| (child, depth + 1, Some(storage_id))),
            );
        }
    }
    Ok(layers)
}

fn graph_clipboard_node_position(node: &NodeTemplate) -> bool {
    matches!(&node.node_type_metadata, NodeTypePersistentMetadata::Node(metadata)
        if matches!(metadata.position(), NodePosition::Absolute(_)))
}

fn graph_clipboard_node_count(nodes: &[(NodeId, NodeTemplate)]) -> usize {
    nodes
        .iter()
        .map(|(_, node)| {
            1 + match &node.implementation {
                NodeTemplateImplementation::Network(inner) => inner.nodes.len(),
                _ => 0,
            }
        })
        .sum()
}

fn graph_clipboard_input(
    input: &NodeInput,
    local_ids: &BTreeSet<NodeId>,
    imports: &[NodeInput],
) -> bool {
    match input {
        NodeInput::Value { tagged_value, .. } => {
            matches!(**tagged_value, TaggedValue::Bool(_))
                || preview_numeric_value(tagged_value).is_ok()
        }
        NodeInput::Node {
            node_id,
            output_index: 0,
        } => local_ids.contains(node_id),
        NodeInput::Import { import_index, .. } => imports
            .get(*import_index)
            .is_some_and(|input| matches!(input, NodeInput::Value { .. })),
        _ => false,
    }
}

fn check_clipboard_numeric_branch(
    network: &NodeNetwork,
    node: NodeId,
    imports: &[NodeInput],
) -> Result<(), GddArchiveError> {
    let detached = numeric_preview_network(network, node, imports)?;
    typecheck_detached_numeric_network(detached)
        .map_err(GddArchiveError::GraphClipboardTypeMismatch)
}

fn empty_graph_export(input: &NodeInput) -> bool {
    matches!(input, NodeInput::Value { tagged_value, exposed: true } if **tagged_value == TaggedValue::None)
}

fn empty_document_output(exports: &[NodeInput]) -> bool {
    match exports {
        [] => true,
        [export] => empty_graph_export(export),
        _ => false,
    }
}

fn enclosing_network_inputs<'a>(root: &'a NodeNetwork, path: &[NodeId]) -> Option<&'a [NodeInput]> {
    match path.split_last() {
        Some((owner, parent_path)) => {
            let parent = root.nested_network(parent_path)?;
            Some(&parent.nodes.get(owner)?.inputs)
        }
        None => Some(&[]),
    }
}

fn collect_layer_preview_node(
    source: &NodeNetwork,
    node_id: NodeId,
    detached: &mut NodeNetwork,
    visiting: &mut BTreeSet<NodeId>,
) -> Result<(), GddArchiveError> {
    const MAX_PREVIEW_NODES: usize = 32;
    if let Some(existing) = detached.nodes.get(&node_id) {
        if numeric_preview_inputs(existing).is_some() {
            return Err(GddArchiveError::UnsupportedGraphLayerPreview);
        }
        return Ok(());
    }
    if !visiting.insert(node_id) {
        return Err(GddArchiveError::UnsupportedGraphLayerPreview);
    }
    if detached.nodes.len() + visiting.len() > MAX_PREVIEW_NODES {
        return Err(GddArchiveError::GraphLayerPreviewBudget);
    }
    let source_node = source
        .nodes
        .get(&node_id)
        .ok_or(GddArchiveError::UnsupportedGraphLayerPreview)?;
    let (node, profile) = checked_saved_graphic_node(source_node)
        .ok_or(GddArchiveError::UnsupportedGraphLayerPreview)?;
    for (index, input) in node.inputs.iter().enumerate() {
        if let NodeInput::Node {
            node_id,
            output_index: 0,
        } = input
        {
            if matches!(profile, SavedGraphicProfile::EllipseWired)
                && ELLIPSE_RADIUS_INPUTS.contains(&index)
            {
                let branch = numeric_preview_network(source, *node_id, &[]).map_err(|error| {
                    GddArchiveError::InvalidEllipseRadiusSource {
                        node: *node_id,
                        reason: error.to_string(),
                    }
                })?;
                let additional = branch
                    .nodes
                    .keys()
                    .filter(|id| !detached.nodes.contains_key(id))
                    .count();
                if detached.nodes.len() + visiting.len() + additional > MAX_PREVIEW_NODES {
                    return Err(GddArchiveError::GraphLayerPreviewBudget);
                }
                detached.nodes.extend(branch.nodes);
            } else {
                collect_layer_preview_node(source, *node_id, detached, visiting)?;
            }
        }
    }
    visiting.remove(&node_id);
    detached.nodes.insert(node_id, node);
    Ok(())
}

fn valid_saved_ellipse_radius(radius: f64) -> bool {
    // The source forms the diameter by subtracting the two opposite radii.
    (radius * 2.).is_finite()
}

fn valid_saved_ellipse_radius_list(radii: &[f64]) -> bool {
    radii.len() <= MAX_PREVIEW_VALUES && radii.iter().copied().all(valid_saved_ellipse_radius)
}

fn valid_saved_ellipse_radius_value(value: &TaggedValue) -> bool {
    match value {
        TaggedValue::Number(radius) => valid_saved_ellipse_radius(*radius),
        TaggedValue::Numbers(radii) => valid_saved_ellipse_radius_list(radii),
        _ => false,
    }
}

async fn validate_saved_ellipse_radius_sources(
    network: &NodeNetwork,
) -> Result<(), GddArchiveError> {
    let source_ids: BTreeSet<_> = network
        .nodes
        .values()
        .filter(|node| {
            matches!(
                &node.implementation,
                DocumentNodeImplementation::ProtoNode(identifier) if *identifier == ellipse::IDENTIFIER
            )
        })
        .flat_map(|node| {
            ELLIPSE_RADIUS_INPUTS.into_iter().filter_map(|index| {
                let NodeInput::Node { node_id, output_index: 0 } = node.inputs.get(index)? else {
                    return None;
                };
                Some(*node_id)
            })
        })
        .collect();
    for node in source_ids {
        let source_error = |reason| GddArchiveError::InvalidEllipseRadiusSource { node, reason };
        let branch = numeric_preview_network(network, node, &[])
            .map_err(|error| source_error(error.to_string()))?;
        let evaluator = NumericGraphEvaluator::prepare(branch)
            .await
            .map_err(|error| source_error(error.to_string()))?;
        let value = evaluator.evaluate().await.map_err(source_error)?;
        if !valid_saved_ellipse_radius_value(&value) {
            return Err(source_error(
                "Expected finite pixel radii with finite diameters within the preview limit."
                    .into(),
            ));
        }
    }
    Ok(())
}

fn checked_saved_graphic_node(node: &DocumentNode) -> Option<(DocumentNode, SavedGraphicProfile)> {
    let output_wire = |input: &NodeInput| {
        matches!(
            input,
            NodeInput::Node {
                output_index: 0,
                ..
            }
        )
    };
    let (definition, replacements, profile) = match &node.implementation {
        DocumentNodeImplementation::Network(_) => {
            let mut replacements = node
                .inputs
                .iter()
                .take(2)
                .enumerate()
                .filter(|(_, input)| output_wire(input))
                .collect::<Vec<_>>();
            match node.inputs.as_slice() {
                [_, _] => (
                    merge_node_definition()?,
                    replacements,
                    SavedGraphicProfile::Layer(GddGraphLayerPreview::Merge),
                ),
                [_, _, location, dimensions, background, clip] => {
                    let (origin, size) = GddArtboardGeometry::read(node)?.normalized()?;
                    if !matches!(background.as_value(), Some(TaggedValue::Color(color)) if color.to_gamma_srgb_channels().into_iter().all(|channel| channel.is_finite()) && (color.a() * 1000.).is_finite())
                        || !matches!(clip.as_value(), Some(TaggedValue::Bool(_)))
                    {
                        return None;
                    }
                    replacements.extend([
                        (ARTBOARD_LOCATION_INPUT, location),
                        (ARTBOARD_DIMENSIONS_INPUT, dimensions),
                        (ARTBOARD_BACKGROUND_INPUT, background),
                        (ARTBOARD_CLIP_INPUT, clip),
                    ]);
                    (
                        artboard_node_definition()?,
                        replacements,
                        SavedGraphicProfile::Layer(GddGraphLayerPreview::Artboard { origin, size }),
                    )
                }
                _ => return None,
            }
        }
        DocumentNodeImplementation::ProtoNode(identifier)
            if identifier == &rectangle::IDENTIFIER =>
        {
            let dimensions = [rectangle::WidthInput::INDEX, rectangle::HeightInput::INDEX];
            let mut replacements = Vec::with_capacity(dimensions.len());
            for index in dimensions {
                let input = node.inputs.get(index)?;
                if !matches!(input.as_value(), Some(TaggedValue::Number(value)) if value.is_finite() && *value >= 0.)
                {
                    return None;
                }
                replacements.push((index, input));
            }
            (
                resolve_proto_node_type(rectangle::IDENTIFIER)?,
                replacements,
                SavedGraphicProfile::Rectangle,
            )
        }
        DocumentNodeImplementation::ProtoNode(identifier) if identifier == &ellipse::IDENTIFIER => {
            let horizontal = node.inputs.get(ellipse::RadiusXInput::INDEX)?;
            let vertical = node.inputs.get(ellipse::RadiusYInput::INDEX)?;
            let profile = match (
                horizontal.as_non_exposed_value(),
                vertical.as_non_exposed_value(),
            ) {
                (Some(TaggedValue::Number(x)), Some(TaggedValue::Number(y)))
                    if valid_saved_ellipse_radius(*x) && valid_saved_ellipse_radius(*y) =>
                {
                    SavedGraphicProfile::Ellipse([*x, *y])
                }
                (Some(TaggedValue::Numbers(x)), Some(TaggedValue::Numbers(y)))
                    if valid_saved_ellipse_radius_list(x) && valid_saved_ellipse_radius_list(y) =>
                {
                    SavedGraphicProfile::EllipseList
                }
                _ if [horizontal, vertical].into_iter().any(output_wire)
                    && [horizontal, vertical].into_iter().all(|input| {
                        output_wire(input)
                            || input
                                .as_non_exposed_value()
                                .is_some_and(valid_saved_ellipse_radius_value)
                    }) =>
                {
                    SavedGraphicProfile::EllipseWired
                }
                _ => return None,
            };
            (
                resolve_proto_node_type(ellipse::IDENTIFIER)?,
                Vec::from([
                    (ellipse::RadiusXInput::INDEX, horizontal),
                    (ellipse::RadiusYInput::INDEX, vertical),
                ]),
                profile,
            )
        }
        DocumentNodeImplementation::ProtoNode(identifier) if identifier == &fill::IDENTIFIER => {
            let content_index = fill::ContentInput::INDEX;
            let paint_index = fill::PaintInput::INDEX;
            let content = node.inputs.get(content_index)?;
            let paint = node.inputs.get(paint_index)?;
            if !matches!(paint.as_value(), Some(TaggedValue::Color(color)) if color.to_gamma_srgb_channels().into_iter().all(|channel| channel.is_finite()))
            {
                return None;
            }
            let mut replacements = vec![(paint_index, paint)];
            if output_wire(content) {
                replacements.push((content_index, content));
            }
            (
                resolve_proto_node_type(fill::IDENTIFIER)?,
                replacements,
                SavedGraphicProfile::Fill,
            )
        }
        _ => return None,
    };
    let mut expected = definition.default_node_template();
    expected.normalize_stored_types();
    let mut expected = expected.into_parts().0;
    for (index, input) in replacements {
        *expected.inputs.get_mut(index)? = input.clone();
    }
    (expected == *node).then_some((expected, profile))
}

fn document_numeric_export_network(
    mut network: &NodeNetwork,
    mut node_id: NodeId,
    mut output_index: usize,
) -> Result<(NodeNetwork, Vec<NodeId>), GddArchiveError> {
    const MAX_EXPORT_DEPTH: usize = 32;
    let mut imports = &[][..];
    let mut path = Vec::new();
    for _ in 0..MAX_EXPORT_DEPTH {
        let document = network
            .nodes
            .get(&node_id)
            .ok_or(GddArchiveError::UnsupportedGraphNumericPreview)?;
        let DocumentNodeImplementation::Network(nested) = &document.implementation else {
            if output_index != 0 {
                return Err(GddArchiveError::UnsupportedGraphNumericPreview);
            }
            return Ok((numeric_preview_network(network, node_id, imports)?, path));
        };
        // The pinned compiler replaces a hidden wrapper with a passthrough.
        if !document.visible || document.context_features != Default::default() {
            return Err(GddArchiveError::UnsupportedGraphNumericPreview);
        }
        let Some(NodeInput::Node {
            node_id: source_id,
            output_index: source_output,
        }) = nested.exports.get(output_index)
        else {
            return Err(GddArchiveError::UnsupportedGraphNumericPreview);
        };
        network = nested;
        imports = &document.inputs;
        path.push(node_id);
        node_id = *source_id;
        output_index = *source_output;
    }
    Err(GddArchiveError::GraphNumericPreviewBudget)
}

fn document_numeric_compilation_error(
    error: NumericGraphPreparationError,
    network_path: &[NodeId],
    nodes: &[NodeMetadataEntry],
) -> GddArchiveError {
    let NumericGraphPreparationError::Nodes(errors) = error else {
        return GddArchiveError::GraphNumericPreviewEvaluation(error.to_string());
    };
    let scope = if network_path.is_empty() {
        "Root graph".to_owned()
    } else {
        let path = network_path
            .iter()
            .map(|node| node.0.to_string())
            .collect::<Vec<_>>()
            .join(" / ");
        format!("Network {path}")
    };
    let details = errors
        .iter()
        .map(|error| {
            // Any path segments after the saved node belong to pinned compiler adapters.
            let [NUMERIC_NETWORK_WRAPPER, local_id, ..] = error.node_path.as_slice() else {
                return format!("{error:?}");
            };
            let Some(node) = nodes
                .iter()
                .find(|node| node.local_id == *local_id && node.network_path == network_path)
            else {
                return format!("{error:?}");
            };
            let name = node.display_name.as_deref().unwrap_or(&error.identifier);
            format!("{scope} · {name} [{}]\n{:?}", local_id.0, error.error)
        })
        .collect::<Vec<_>>()
        .join("\n");
    GddArchiveError::GraphNumericPreviewEvaluation(details)
}

fn numeric_preview_network(
    source: &NodeNetwork,
    selected: NodeId,
    imports: &[NodeInput],
) -> Result<NodeNetwork, GddArchiveError> {
    let mut detached = NodeNetwork {
        exports: vec![NodeInput::node(selected, 0)],
        ..Default::default()
    };
    collect_numeric_preview_node(
        source,
        selected,
        imports,
        &mut detached,
        &mut BTreeSet::new(),
    )?;
    Ok(detached)
}

fn collect_numeric_preview_node(
    source: &NodeNetwork,
    node_id: NodeId,
    imports: &[NodeInput],
    detached: &mut NodeNetwork,
    visiting: &mut BTreeSet<NodeId>,
) -> Result<(), GddArchiveError> {
    const MAX_PREVIEW_NODES: usize = 32;
    if detached.nodes.contains_key(&node_id) {
        return Ok(());
    }
    if !visiting.insert(node_id) {
        return Err(GddArchiveError::CyclicGraphNumericPreview);
    }
    if detached.nodes.len() + visiting.len() > MAX_PREVIEW_NODES {
        return Err(GddArchiveError::GraphNumericPreviewBudget);
    }
    let document = source
        .nodes
        .get(&node_id)
        .ok_or(GddArchiveError::UnsupportedGraphNumericPreview)?;
    let inputs =
        numeric_preview_inputs(document).ok_or(GddArchiveError::UnsupportedGraphNumericPreview)?;
    let mut resolved = Vec::with_capacity(inputs.len());
    let boolean_input = match &document.implementation {
        DocumentNodeImplementation::ProtoNode(identifier) => {
            numeric_boolean_port(identifier).map(|(index, _)| index)
        }
        _ => None,
    };
    for (index, input) in inputs.iter().enumerate() {
        let input = match input {
            // A detached pure branch receives the same literal the enclosing network holds.
            NodeInput::Import { import_index, .. } => imports
                .get(*import_index)
                .filter(|input| matches!(input, NodeInput::Value { .. }))
                .ok_or(GddArchiveError::UnsupportedGraphNumericPreview)?,
            _ => input,
        };
        match input {
            NodeInput::Value { tagged_value, .. } => {
                if !(boolean_input == Some(index) && matches!(**tagged_value, TaggedValue::Bool(_)))
                {
                    preview_numeric_value(tagged_value)?;
                }
            }
            NodeInput::Node {
                node_id,
                output_index: 0,
            } => {
                collect_numeric_preview_node(source, *node_id, imports, detached, visiting)?;
            }
            _ => return Err(GddArchiveError::UnsupportedGraphNumericPreview),
        }
        resolved.push(input.clone());
    }
    visiting.remove(&node_id);
    let mut document = document.clone();
    document.inputs = resolved;
    detached.nodes.insert(node_id, document);
    Ok(())
}

fn numeric_preview_inputs(document: &DocumentNode) -> Option<&[NodeInput]> {
    let DocumentNodeImplementation::ProtoNode(identifier) = &document.implementation else {
        return None;
    };
    (numeric_proto_input_count(identifier) == Some(document.inputs.len())
        && document.context_features == Default::default())
    .then_some(document.inputs.as_slice())
}

fn numeric_proto_input_count(identifier: &ProtoNodeIdentifier) -> Option<usize> {
    use graphite_math_nodes::{
        absolute_value, add, ceiling, divide, exponent, floor, logarithm, multiply, reciprocal,
        root, round, sign, subtract,
    };

    if [
        absolute_value::IDENTIFIER,
        ceiling::IDENTIFIER,
        floor::IDENTIFIER,
        reciprocal::IDENTIFIER,
        round::IDENTIFIER,
        sign::IDENTIFIER,
    ]
    .contains(identifier)
    {
        Some(1)
    } else if [
        add::IDENTIFIER,
        divide::IDENTIFIER,
        exponent::IDENTIFIER,
        logarithm::IDENTIFIER,
        multiply::IDENTIFIER,
        root::IDENTIFIER,
        subtract::IDENTIFIER,
    ]
    .contains(identifier)
        || is_trigonometric_proto(identifier)
    {
        Some(2)
    } else if identifier == &modulo::IDENTIFIER {
        Some(3)
    } else {
        None
    }
}

const TRIGONOMETRIC_RADIANS_INDEX: usize = graphite_math_nodes::sine::RadiansInput::INDEX;

fn graph_boolean_literal(document: &DocumentNode) -> Option<GddGraphBooleanInput> {
    match &document.implementation {
        DocumentNodeImplementation::ProtoNode(identifier) => {
            let (index, name) = numeric_boolean_port(identifier)?;
            hidden_graph_boolean(numeric_preview_inputs(document)?, index, name)
        }
        DocumentNodeImplementation::Network(_)
            if matches!(
                checked_saved_graphic_node(document),
                Some((
                    _,
                    SavedGraphicProfile::Layer(GddGraphLayerPreview::Artboard { .. })
                ))
            ) =>
        {
            hidden_graph_boolean(&document.inputs, ARTBOARD_CLIP_INPUT, "Clip")
        }
        _ => None,
    }
}

fn hidden_artboard_background(document: &DocumentNode) -> Option<Color> {
    let NodeInput::Value {
        tagged_value,
        exposed: false,
    } = document.inputs.get(ARTBOARD_BACKGROUND_INPUT)?
    else {
        return None;
    };
    match **tagged_value {
        TaggedValue::Color(color) => Some(color),
        _ => None,
    }
}

fn hidden_graph_boolean(
    inputs: &[NodeInput],
    index: usize,
    name: &'static str,
) -> Option<GddGraphBooleanInput> {
    let NodeInput::Value {
        tagged_value,
        exposed: false,
    } = inputs.get(index)?
    else {
        return None;
    };
    match **tagged_value {
        TaggedValue::Bool(value) => Some(GddGraphBooleanInput { index, name, value }),
        _ => None,
    }
}

fn numeric_boolean_port(identifier: &ProtoNodeIdentifier) -> Option<(usize, &'static str)> {
    if is_trigonometric_proto(identifier) {
        Some((TRIGONOMETRIC_RADIANS_INDEX, "Radians"))
    } else if identifier == &modulo::IDENTIFIER {
        Some((modulo::AlwaysPositiveInput::INDEX, "Always Positive"))
    } else {
        None
    }
}

fn is_trigonometric_proto(identifier: &ProtoNodeIdentifier) -> bool {
    use graphite_math_nodes::{
        cosine, cosine_inverse, sine, sine_inverse, tangent, tangent_inverse,
    };

    [
        cosine::IDENTIFIER,
        cosine_inverse::IDENTIFIER,
        sine::IDENTIFIER,
        sine_inverse::IDENTIFIER,
        tangent::IDENTIFIER,
        tangent_inverse::IDENTIFIER,
    ]
    .contains(identifier)
}

fn preview_numeric_value(value: &TaggedValue) -> Result<GddNumericValue, GddArchiveError> {
    match value {
        TaggedValue::Number(value) if value.is_finite() => Ok(GddNumericValue::Number(*value)),
        TaggedValue::Integer(value) => Ok(GddNumericValue::Integer(*value)),
        TaggedValue::DVec2(value) if value.is_finite() => {
            Ok(GddNumericValue::Vector(value.to_array()))
        }
        TaggedValue::Numbers(values) if values.len() > MAX_PREVIEW_VALUES => {
            Err(GddArchiveError::GraphNumericPreviewBudget)
        }
        TaggedValue::Numbers(values) if values.iter().all(|value| value.is_finite()) => {
            Ok(GddNumericValue::NumberList(values.as_slice().into()))
        }
        _ => Err(GddArchiveError::UnsupportedGraphNumericPreview),
    }
}

fn exposed_numeric_literal(input: &NodeInput) -> Option<GddNumericValue> {
    if let NodeInput::Value {
        tagged_value,
        exposed: true,
    } = input
    {
        preview_numeric_value(tagged_value).ok()
    } else {
        None
    }
}

fn is_only_graph_wire(
    interface: &mut NodeNetworkInterface,
    path: &[NodeId],
    upstream: NodeId,
    input: InputConnector,
) -> bool {
    interface
        .outward_wires(path)
        .and_then(|wires| wires.get(&OutputConnector::primary_output(upstream)))
        .is_some_and(|wires| wires.as_slice() == [input])
}

fn graph_numeric_port(
    interface: &NodeNetworkInterface,
    node: &GddGraphNode,
) -> Option<GddNumericPort> {
    match graph_numeric_input(interface, node)? {
        GddNumericInput::Literal(port) => Some(port),
        GddNumericInput::Connected { .. } | GddNumericInput::Imported { .. } => None,
    }
}

fn graph_numeric_input(
    interface: &NodeNetworkInterface,
    node: &GddGraphNode,
) -> Option<GddNumericInput> {
    use graphite_math_nodes::{absolute_value, ceiling, floor, round, sign};

    let path = &node.network_path;
    let DocumentNodeImplementation::ProtoNode(identifier) =
        interface.implementation(&node.local_id, path)?
    else {
        return None;
    };
    let (_, input_index, integer_supported) = [
        (
            absolute_value::IDENTIFIER,
            absolute_value::ValueInput::INDEX,
            true,
        ),
        (ceiling::IDENTIFIER, ceiling::ValueInput::INDEX, false),
        (floor::IDENTIFIER, floor::ValueInput::INDEX, false),
        (round::IDENTIFIER, round::ValueInput::INDEX, false),
        (sign::IDENTIFIER, sign::ValueInput::INDEX, false),
    ]
    .into_iter()
    .find(|(admitted, _, _)| admitted == identifier)?;
    if interface.number_of_inputs(&node.local_id, path) != 1
        || interface.is_layer(&node.local_id, path)
    {
        return None;
    }
    let connector = InputConnector::node_at_index(node.local_id, input_index);
    let (tagged_value, exposed) = match interface.input_from_connector(&connector, path)? {
        NodeInput::Value {
            tagged_value,
            exposed,
        } => (tagged_value, *exposed),
        NodeInput::Node {
            node_id,
            output_index: 0,
        } if *node_id != node.local_id
            && interface.node_metadata(node_id, path).is_some()
            && !interface.is_layer(node_id, path)
            && !interface.is_chain(node_id, path) =>
        {
            return Some(GddNumericInput::Connected {
                index: input_index,
                upstream: *node_id,
            });
        }
        NodeInput::Import { import_index, .. } => {
            return Some(GddNumericInput::Imported {
                index: input_index,
                import_index: *import_index,
            });
        }
        _ => return None,
    };
    let value = match &**tagged_value {
        TaggedValue::Number(number) => GddNumericValue::Number(*number),
        TaggedValue::Integer(number) if integer_supported => GddNumericValue::Integer(*number),
        TaggedValue::DVec2(vector) => GddNumericValue::Vector(vector.to_array()),
        TaggedValue::Numbers(numbers) => GddNumericValue::NumberList(numbers.as_slice().into()),
        _ => return None,
    };
    Some(GddNumericInput::Literal(GddNumericPort {
        index: input_index,
        value,
        exposed,
    }))
}

fn graph_wired_output_nodes(
    interface: &mut NodeNetworkInterface,
    path: &[NodeId],
) -> Option<BTreeSet<NodeId>> {
    Some(
        interface
            .outward_wires(path)?
            .iter()
            .filter(|(_, inputs)| !inputs.is_empty())
            .filter_map(|(output, _)| output.node_id())
            .collect(),
    )
}

fn graph_node_has_isolated_layout(interface: &NodeNetworkInterface, node: &GddGraphNode) -> bool {
    let path = &node.network_path;
    interface.previewing(path) == Previewing::No
        && !interface.is_layer(&node.local_id, path)
        && !interface.is_chain(&node.local_id, path)
        && interface
            .document_node(&node.local_id, path)
            .is_some_and(|document| {
                matches!(
                    document.implementation,
                    DocumentNodeImplementation::ProtoNode(_)
                ) && document
                    .inputs
                    .iter()
                    .all(|input| matches!(input, NodeInput::Value { .. }))
            })
}

type GraphPortEditLayout<'a> = BTreeMap<(&'a [NodeId], NodeId), NodeTypePersistentMetadata>;

fn graph_port_edit_layout<'a, const SCOPES: usize>(
    interface: &NodeNetworkInterface,
    scopes: [&'a [NodeId]; SCOPES],
) -> Result<GraphPortEditLayout<'a>, GddArchiveError> {
    let mut layout = BTreeMap::new();
    for scope in scopes {
        let network = interface
            .nested_network(scope)
            .ok_or(GddArchiveError::MissingGraphNetwork)?;
        for node_id in network.nodes.keys() {
            let metadata = interface
                .node_metadata(node_id, scope)
                .ok_or(GddArchiveError::MissingGraphNetwork)?;
            layout.insert(
                (scope, *node_id),
                metadata.persistent_metadata.node_type_metadata.clone(),
            );
        }
    }
    Ok(layout)
}

fn write_graph_connector(
    path: &[NodeId],
    connector: InputConnector,
    input: NodeInput,
) -> EditorDelta {
    let network_path = path.to_vec();
    EditorDelta::Graph(match connector {
        InputConnector::Node {
            node_id,
            input_index,
        } => RuntimeDelta::SetInput {
            network_path,
            node_id,
            input_index,
            input,
        },
        InputConnector::Export(export_index) => RuntimeDelta::SetExport {
            network_path,
            export_index,
            input: Some(input),
        },
    })
}

fn prepare_graph_node_edit(
    owner: &GddV1,
    interface: &NodeNetworkInterface,
    edits: &[EditorDelta],
) -> Result<ConstructedOps, GddArchiveError> {
    let resources = owner
        .registry()
        .to_resource_registry()
        .map_err(GddArchiveError::RuntimeRegistry)?;
    construct_batch(
        edits,
        owner.registry(),
        &resources,
        &StorageMetadataView::new(interface),
        owner.session().peer(),
    )
    .map_err(GddArchiveError::GraphNodePreparation)
}

async fn finish_graph_node_edit(
    mut owner: GddV1,
    mut staged: StagedArchive,
    batch: ConstructedOps,
    budget: &GddArchiveBudget,
) -> Result<InspectedGddV1, GddArchiveError> {
    staged.store_declarations(&batch.declarations.bytes, budget)?;
    commit_graph_node_edit(&mut owner, batch)?;
    staged.reopen_after_edit(budget).await
}

fn initialize_graph_root_history(owner: &mut GddV1) -> Result<(), GddArchiveError> {
    let mut empty = Registry::from_runtime_with_metadata(
        &NodeNetwork::default(),
        &NoMetadata,
        &graphite_resource::ResourceRegistry::new(),
        owner.session().peer(),
    )
    .map_err(GddArchiveError::GraphNodePreparation)?;
    let network = empty
        .networks
        .remove(&ROOT_NETWORK)
        .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
    // The pinned cursor cannot undo the first interaction, so save the empty base first.
    commit_graph_node_edit(
        owner,
        ConstructedOps {
            ops: vec![RegistryDelta::AddNetwork {
                id: ROOT_NETWORK,
                network,
            }],
            declarations: Default::default(),
        },
    )
}

fn commit_graph_node_edit(owner: &mut GddV1, batch: ConstructedOps) -> Result<(), GddArchiveError> {
    let peer = owner.session().peer();
    let registers_peer = !batch.ops.is_empty() && !owner.registry().peer_users.contains_key(&peer);
    let expected = batch.ops.len() + usize::from(registers_peer);
    owner.stage_constructed_ops(
        batch.ops,
        &batch.declarations.bytes,
        &HashMapResourceStorage::new(),
    )?;
    let retired = owner.retire_pending_interaction()?;
    if retired.len() != expected
        || registers_peer
            && !retired
                .first()
                .and_then(|revision| owner.session().delta(*revision))
                .is_some_and(|delta| {
                    matches!(
                        delta.kind,
                        RegistryDelta::RegisterPeer { peer: registered, user }
                            if registered == peer && user.0 == peer.0
                    )
                })
    {
        return Err(GddArchiveError::IncompleteGraphNodeEdit);
    }
    Ok(())
}

fn approved_resource_storage(
    mut missing: BTreeSet<ResourceHash>,
    approved_files: Vec<Vec<u8>>,
    budget: &GddArchiveBudget,
) -> Result<HashMapResourceStorage, GddArchiveError> {
    if approved_files.len() > budget.max_entries {
        return Err(GddArchiveError::BudgetExceeded);
    }
    let storage = HashMapResourceStorage::new();
    let mut total = 0_u64;
    for bytes in approved_files {
        let length = bytes.len() as u64;
        total = total.saturating_add(length);
        if length > budget.max_entry_bytes || total > budget.max_total_entry_bytes {
            return Err(GddArchiveError::BudgetExceeded);
        }
        let hash = ResourceHash::from(bytes.as_slice());
        if !missing.remove(&hash) {
            return Err(GddArchiveError::UnexpectedApprovedResource(hash));
        }
        storage.store(&bytes);
    }
    if let Some(hash) = missing.into_iter().next() {
        return Err(GddArchiveError::MissingApprovedResource(hash));
    }
    Ok(storage)
}

struct BoundedArchive {
    bytes: Cursor<Vec<u8>>,
    limit: u64,
}

impl Write for BoundedArchive {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.bytes.position().saturating_add(bytes.len() as u64) > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "Graphite archive export exceeds its byte budget",
            ));
        }
        self.bytes.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.bytes.flush()
    }
}

impl Seek for BoundedArchive {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.bytes.seek(position)
    }
}

pub async fn open_gdd_zip(
    bytes: &Arc<[u8]>,
    budget: GddArchiveBudget,
) -> Result<PreservedGddV1, GddArchiveError> {
    let working = stage_zip(bytes, &budget)?;
    Ok(PreservedGddV1 {
        original_bytes: bytes.clone(),
        inspected: open_staged_gdd(working).await?,
    })
}

pub async fn open_gdd_archive(
    bytes: &Arc<[u8]>,
    budget: GddArchiveBudget,
) -> Result<PreservedGddV1, GddArchiveError> {
    match ArchiveFormat::detect(bytes) {
        Some(ArchiveFormat::Zip) => open_gdd_zip(bytes, budget).await,
        Some(ArchiveFormat::Xz) => {
            let working = xz::stage_xz(bytes, &budget)?;
            Ok(PreservedGddV1 {
                original_bytes: bytes.clone(),
                inspected: open_staged_gdd(working).await?,
            })
        }
        None => Err(GddArchiveError::UnsupportedEnvelope),
    }
}

pub async fn inspect_gdd_folder_files(
    files: Vec<(String, Vec<u8>)>,
    budget: GddArchiveBudget,
) -> Result<InspectedGddV1, GddArchiveError> {
    let mut entries = StagedEntries::new(&budget);
    for (name, bytes) in files {
        entries.install(&name, &bytes)?;
    }
    open_staged_gdd(entries.finish()?).await
}

pub async fn create_gdd_folder(
    document_id: u64,
    peer_id: u64,
    editor_version: String,
    budget: GddArchiveBudget,
) -> Result<InspectedGddV1, GddArchiveError> {
    let working = MemoryBackend::new();
    GddV1::create_in(
        AnyContainer::Memory(working.clone()),
        GddV1Layout,
        PeerId(peer_id),
        document_id,
        editor_version,
        crate::resources::GRAPHITE_IMAGE_SOURCE_REVISION.into(),
    )?;
    inspect_working_copy(&working, BTreeSet::new(), &budget).await
}

async fn inspect_working_copy(
    working: &MemoryBackend,
    mut names: BTreeSet<String>,
    budget: &GddArchiveBudget,
) -> Result<InspectedGddV1, GddArchiveError> {
    let mut entries = StagedEntries::new(budget);
    names.extend(working.list("")?);
    for name in names {
        entries.install(&name, working.read(&name)?.as_slice())?;
    }
    let copy = open_staged_gdd(entries.finish()?).await?;
    copy.document
        .session()
        .verify_history()
        .map_err(graphite_document_format::Error::from)?;
    Ok(copy)
}

async fn open_staged_gdd(entries: StagedArchive) -> Result<InspectedGddV1, GddArchiveError> {
    let working = &entries.working;
    let layout = GddV1Layout;
    let path = path_for(layout.manifest_basename(), MANIFEST_CODEC);
    let OriginalJson(value) = serde_json::from_slice(working.read(&path)?.as_slice())?;
    let header = serde::Deserialize::deserialize(&value)?;
    let ManifestHeader {
        format,
        format_version,
    } = header;
    if format != FORMAT_MAGIC || format_version != SUPPORTED_FORMAT_VERSION {
        return Err(GddArchiveError::UnsupportedManifest {
            format,
            version: format_version,
        });
    }
    let manifest: Manifest = serde_json::from_value(value)?;
    let registry_exists = working.exists(&path_for(
        layout.registry_basename(),
        manifest.codecs.registry,
    ));
    let history_exists = working.exists(&path_for(
        layout.history_basename(),
        manifest.codecs.history,
    ));
    let source = match (registry_exists, history_exists) {
        (true, true) => GddRecoverySource::RegistryAndHistory,
        (true, false) => GddRecoverySource::RegistryOnly,
        (false, true) => GddRecoverySource::HistoryOnly,
        (false, false) => return Err(GddArchiveError::MissingDocument),
    };
    let document = GddV1::open_in(AnyContainer::Memory(working.clone()), layout).await?;
    validate_embedded_resources(&document).await?;
    let recovery = GddRecoverySummary {
        source,
        network_count: document.registry().networks.len(),
        node_count: document.registry().node_instances.len(),
        history_len: document.session().history().count(),
        hot_log_len: document.session().hot_log().len(),
    };
    Ok(InspectedGddV1 {
        entries,
        document,
        recovery,
        snapshot: Arc::new(()),
    })
}

async fn validate_embedded_resources(document: &GddV1) -> Result<(), GddArchiveError> {
    let mut required = BTreeSet::new();
    for resource in document.registry().resources.values() {
        if resource.has_embedded_source() {
            required.insert(resource.hash.ok_or(GddArchiveError::UnresolvedResource)?);
        }
    }
    let session = document.session();
    let history = session
        .history()
        .flat_map(|delta| [&delta.kind, &delta.reverse]);
    let hot_log = session.hot_log().iter().map(|operation| &operation.op);
    for operation in history.chain(hot_log) {
        let entry = match operation {
            RegistryDelta::AddResource { entry, .. } => entry,
            RegistryDelta::RemoveResource { snapshot, .. } => snapshot,
            _ => continue,
        };
        if entry.has_embedded_source() {
            required.extend(entry.hash);
        }
    }
    for hash in required {
        if !document.has_resource(&hash).await {
            return Err(GddArchiveError::InvalidResource(hash));
        }
    }
    Ok(())
}

fn stage_zip(bytes: &[u8], budget: &GddArchiveBudget) -> Result<StagedArchive, GddArchiveError> {
    if bytes.len() > budget.max_archive_bytes {
        return Err(GddArchiveError::BudgetExceeded);
    }
    let (entry_count, directory_start) = zip_directory(bytes, budget.max_entries)?;
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    // The ZIP reader folds identical names into its index. Require every
    // original directory entry to survive before installing any payloads.
    if archive.len() != entry_count || archive.central_directory_start() != directory_start {
        return Err(GddArchiveError::InconsistentEntries);
    }

    let mut entries = StagedEntries::new(budget);
    let mut next_header = 0_u64;
    for index in 0..entry_count {
        let entry = archive.by_index(index)?;
        if !entry.is_file() || entry.header_start() != next_header {
            return Err(GddArchiveError::InconsistentEntries);
        }
        next_header = entry
            .data_start()
            .and_then(|start| start.checked_add(entry.compressed_size()))
            .filter(|end| *end <= directory_start)
            .ok_or(GddArchiveError::InconsistentEntries)?;
        let name = entry.name().to_owned();
        let size = entry.size();
        entries.stage(name, size, entry)?;
    }
    if next_header != directory_start {
        return Err(GddArchiveError::InconsistentEntries);
    }
    entries.finish()
}

struct StagedEntries<'a> {
    budget: &'a GddArchiveBudget,
    working: MemoryBackend,
    names: BTreeSet<String>,
    total_bytes: u64,
}

struct StagedArchive {
    working: MemoryBackend,
    names: BTreeSet<String>,
    total_bytes: u64,
}

impl StagedArchive {
    fn store_declarations(
        &mut self,
        declarations: &from_runtime::DeclarationBytes,
        budget: &GddArchiveBudget,
    ) -> Result<(), GddArchiveError> {
        for (hash, bytes) in declarations {
            if ResourceHash::from(bytes.as_slice()) != *hash {
                return Err(GddArchiveError::InvalidResource(*hash));
            }
            let path = GddV1Layout.resource_path(hash);
            if self.names.contains(&path) {
                if self.working.read(&path)?.as_slice() != bytes {
                    return Err(GddArchiveError::InvalidResource(*hash));
                }
                continue;
            }
            let size = bytes.len() as u64;
            let total = self.total_bytes.saturating_add(size);
            if size > budget.max_entry_bytes
                || total > budget.max_total_entry_bytes
                || self.names.len() >= budget.max_entries
            {
                return Err(GddArchiveError::BudgetExceeded);
            }
            self.working.write(&path, bytes)?;
            self.names.insert(path);
            self.total_bytes = total;
        }
        Ok(())
    }

    async fn reopen_after_edit(
        self,
        budget: &GddArchiveBudget,
    ) -> Result<InspectedGddV1, GddArchiveError> {
        // The source writer updates root payloads. Its nonrecursive listing misses the
        // admitted resources, new declarations and opaque files in nested paths.
        inspect_working_copy(&self.working, self.names, budget).await
    }

    fn folder_files(&self) -> Result<Vec<(PathBuf, Vec<u8>)>, ContainerError> {
        self.names
            .iter()
            .map(|path| {
                let bytes = match self.working.read(path)? {
                    ByteHolder::Owned(bytes) => bytes,
                    #[cfg(not(target_family = "wasm"))]
                    other => other.as_slice().to_vec(),
                };
                Ok((PathBuf::from(path), bytes))
            })
            .collect()
    }
}

impl<'a> StagedEntries<'a> {
    fn new(budget: &'a GddArchiveBudget) -> Self {
        Self {
            budget,
            working: MemoryBackend::new(),
            names: BTreeSet::new(),
            total_bytes: 0,
        }
    }

    fn stage(&mut self, name: String, size: u64, input: impl Read) -> Result<(), GddArchiveError> {
        self.admit(&name, size)?;
        let mut materialized = Vec::new();
        input
            .take(size.saturating_add(1))
            .read_to_end(&mut materialized)?;
        if u64::try_from(materialized.len()) != Ok(size) {
            return Err(GddArchiveError::InconsistentEntries);
        }
        self.write(&name, &materialized)
    }

    fn install(&mut self, name: &str, bytes: &[u8]) -> Result<(), GddArchiveError> {
        self.admit(name, bytes.len() as u64)?;
        self.write(name, bytes)
    }

    fn admit(&mut self, name: &str, size: u64) -> Result<(), GddArchiveError> {
        self.total_bytes = self.total_bytes.saturating_add(size);
        if self.names.len() >= self.budget.max_entries
            || size > self.budget.max_entry_bytes
            || self.total_bytes > self.budget.max_total_entry_bytes
        {
            return Err(GddArchiveError::BudgetExceeded);
        }
        validate_path(name)?;
        if name.contains('\0') || !self.names.insert(name.to_owned()) {
            return Err(GddArchiveError::InconsistentEntries);
        }
        Ok(())
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), GddArchiveError> {
        if let Some((directory, filename)) = name.split_once('/')
            && directory == GddV1Layout.resources_dir()
            && let Ok(hash) = filename.parse::<ResourceHash>()
            && ResourceHash::from(bytes) != hash
        {
            return Err(GddArchiveError::InvalidResource(hash));
        }
        self.working.write(name, bytes)?;
        Ok(())
    }

    fn finish(self) -> Result<StagedArchive, GddArchiveError> {
        if self.names.iter().any(|name| {
            let prefix = format!("{name}/");
            self.names
                .range::<str, _>((Bound::Included(prefix.as_str()), Bound::Unbounded))
                .next()
                .is_some_and(|descendant| descendant.starts_with(&prefix))
        }) {
            return Err(GddArchiveError::InconsistentEntries);
        }
        Ok(StagedArchive {
            working: self.working,
            names: self.names,
            total_bytes: self.total_bytes,
        })
    }
}

fn zip_directory(bytes: &[u8], max_entries: usize) -> Result<(usize, u64), GddArchiveError> {
    if !bytes.starts_with(b"PK\x03\x04") {
        return Err(GddArchiveError::UnsupportedEnvelope);
    }
    // The pinned exporter writes a single-volume ZIP with no archive comment.
    let footer = bytes
        .last_chunk::<22>()
        .ok_or(GddArchiveError::UnsupportedEnvelope)?;
    if !footer.starts_with(b"PK\x05\x06\0\0\0\0") || footer[20..] != [0, 0] {
        return Err(GddArchiveError::UnsupportedEnvelope);
    }
    let disk_entry_count = u16::from_le_bytes([footer[8], footer[9]]);
    let total_entry_count = u16::from_le_bytes([footer[10], footer[11]]);
    let entry_count = usize::from(total_entry_count);
    let directory_size = u64::from(u32::from_le_bytes([
        footer[12], footer[13], footer[14], footer[15],
    ]));
    let directory_start = u64::from(u32::from_le_bytes([
        footer[16], footer[17], footer[18], footer[19],
    ]));
    if entry_count == 0
        || disk_entry_count != total_entry_count
        || directory_start + directory_size != (bytes.len() - footer.len()) as u64
        || directory_size < 46 * entry_count as u64
    {
        return Err(GddArchiveError::InconsistentEntries);
    }
    if entry_count > max_entries {
        return Err(GddArchiveError::BudgetExceeded);
    }
    Ok((entry_count, directory_start))
}

#[cfg(test)]
mod tests {
    async fn evaluate_detached_numeric_network(
        network: NodeNetwork,
    ) -> Result<TaggedValue, String> {
        NumericGraphEvaluator::prepare(network)
            .await
            .map_err(|error| error.to_string())?
            .evaluate()
            .await
    }

    use super::*;
    use graphite_document_format::Codec;
    use graphite_editor::messages::portfolio::document::DocumentMessageHandler;
    use graphite_graph_craft::application_io::resource::ResourceId;
    use std::path::Path;

    fn budget() -> GddArchiveBudget {
        GddArchiveBudget {
            max_archive_bytes: 1024 * 1024,
            max_entry_bytes: 256 * 1024,
            max_total_entry_bytes: 1024 * 1024,
            max_entries: 32,
        }
    }

    #[test]
    fn at_gra_layer_001_1_rejects_invalid_root_layer_topology() {
        use graphite_document_graph_storage::Position;

        for (local_id, inputs) in [
            (NodeId(0), Vec::new()),
            (NodeId(u64::MAX), Vec::new()),
            (NodeId(42), vec![NodeInput::node(NodeId(42), 0)]),
        ] {
            let storage_id = StorageNodeId(1);
            let metadata = NodeMetadataEntry {
                network_path: Vec::new(),
                local_id,
                storage_id,
                position: Some(Position::Absolute([0, 0])),
                is_layer: true,
                display_name: None,
                locked: false,
                pinned: false,
                input_metadata: vec![Default::default(); inputs.len()],
                output_names: Vec::new(),
            };
            let network = NodeNetwork {
                nodes: [(
                    local_id,
                    DocumentNode {
                        inputs,
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                exports: vec![NodeInput::node(local_id, 0)],
                ..Default::default()
            };
            let mut interface = build_interface_from_storage(network, vec![metadata], Vec::new())
                .expect("read the saved layer metadata");
            assert!(matches!(
                project_root_layers(&mut interface, &[(local_id, storage_id)]),
                Err(GddArchiveError::InvalidRootLayerGraph)
            ));
        }
    }

    fn document() -> GddV1 {
        GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            0x459,
            "pinned Graphite editor".into(),
            "pinned Graphite library".into(),
        )
        .expect("actual Graphite working copy")
    }

    async fn export(document: &GddV1, legacy: Option<&[u8]>) -> Arc<[u8]> {
        document
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &HashMapResourceStorage::new(),
                legacy,
            )
            .await
            .expect("actual Graphite ZIP export")
            .into()
    }

    fn document_with_nested_graph_nodes() -> GddV1 {
        use graphite_document_graph_storage::NoMetadata;
        use graphite_graph_craft::document::{DocumentNode, DocumentNodeImplementation};
        use graphite_resource::ResourceRegistry;

        let repeated = NodeId(7);
        let owner_node = NodeId(24);
        let leaf = || DocumentNode {
            implementation: DocumentNodeImplementation::Network(NodeNetwork::default()),
            ..Default::default()
        };
        let network = NodeNetwork {
            nodes: [
                (repeated, leaf()),
                (
                    owner_node,
                    DocumentNode {
                        implementation: DocumentNodeImplementation::Network(NodeNetwork {
                            nodes: [(repeated, leaf()), (NodeId(9), leaf())]
                                .into_iter()
                                .collect(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let mut owner = document();
        owner
            .commit_from_runtime(
                &network,
                &NoMetadata,
                &ResourceRegistry::new(),
                &HashMapResourceStorage::new(),
            )
            .expect("save an actual nested source graph");
        owner
    }

    async fn numeric_graph_fixture() -> Arc<[u8]> {
        use graphite_document_graph_storage::NodeMetadataSource;
        use graphite_math_nodes::{absolute_value, sign};
        use graphite_resource::ResourceRegistry;

        struct NumericMetadata;

        impl NodeMetadataSource for NumericMetadata {
            fn storage_node_id(
                &self,
                network_path: &[NodeId],
                local_id: NodeId,
            ) -> Option<StorageNodeId> {
                (network_path == [NodeId(24)] && local_id == NodeId(9))
                    .then_some(StorageNodeId(0xbc11_824d_7ec3_201a))
            }

            fn input_name(
                &self,
                network_path: &[NodeId],
                local_id: NodeId,
                index: usize,
            ) -> Option<&str> {
                (network_path == [NodeId(24)] && local_id == NodeId(11) && index == 0)
                    .then_some("Original wired input")
            }
        }

        let node = |identifier, input| graphite_graph_craft::document::DocumentNode {
            implementation: DocumentNodeImplementation::ProtoNode(identifier),
            inputs: vec![input],
            ..Default::default()
        };
        let nested = NodeNetwork {
            nodes: [
                (
                    NodeId(7),
                    node(
                        sign::IDENTIFIER,
                        NodeInput::value(TaggedValue::Number(-2.5), false),
                    ),
                ),
                (
                    NodeId(9),
                    node(
                        sign::IDENTIFIER,
                        NodeInput::value(TaggedValue::Number(2.), true),
                    ),
                ),
                (
                    NodeId(11),
                    node(sign::IDENTIFIER, NodeInput::node(NodeId(7), 0)),
                ),
                (
                    NodeId(12),
                    node(
                        absolute_value::IDENTIFIER,
                        NodeInput::value(TaggedValue::Integer(-3), false),
                    ),
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let network = NodeNetwork {
            nodes: [
                (
                    NodeId(7),
                    node(
                        absolute_value::IDENTIFIER,
                        NodeInput::value(TaggedValue::Integer(-17), false),
                    ),
                ),
                (
                    NodeId(24),
                    graphite_graph_craft::document::DocumentNode {
                        implementation: DocumentNodeImplementation::Network(nested),
                        ..Default::default()
                    },
                ),
            ]
            .into_iter()
            .collect(),
            exports: vec![NodeInput::node(NodeId(7), 0)],
            ..Default::default()
        };
        let mut owner = document();
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(
                &network,
                &NumericMetadata,
                &ResourceRegistry::new(),
                &resources,
            )
            .expect("save an actual numeric source graph");
        let bytes: Arc<[u8]> = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("export the source graph and its node definitions")
            .into();
        bytes
    }

    #[test]
    fn at_gra_graph_005_1_inserts_only_in_the_addressed_nested_network() {
        futures::executor::block_on(async {
            let source = document_with_nested_graph_nodes();
            let bytes: Arc<[u8]> = source
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &HashMapResourceStorage::new(),
                    None,
                )
                .await
                .expect("source with repeated local IDs")
                .into();
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("open nested graph");
            let original_graph = original
                .graph_inspection()
                .await
                .expect("original scope IDs");
            let nested = original_graph
                .networks
                .iter()
                .find(|network| network.path == [NodeId(24)])
                .expect("target nested network");
            let foreign = open_gdd_zip(&bytes, budget())
                .await
                .expect("independently opened source");
            let operation = UnaryNumericOperation::Sign;
            let position = [6, -9];
            assert!(matches!(
                foreign
                    .insert_numeric_node_in_network_copy(nested, operation, position, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));

            let edited = original
                .insert_numeric_node_in_network_copy(nested, operation, position, budget())
                .await
                .expect("insert in exact scope");
            let graph = edited.graph_inspection().await.expect("new scoped graph");
            assert_eq!(graph.networks.len(), original_graph.networks.len());
            let added = graph
                .nodes
                .iter()
                .filter(|node| {
                    !original_graph
                        .nodes
                        .iter()
                        .any(|old| old.storage_id == node.storage_id)
                })
                .collect::<Vec<_>>();
            let [node] = added.as_slice() else {
                panic!("one added node")
            };
            assert_eq!(node.network_id(), nested.id);
            assert_eq!(node.grid_position(), Some(position));
            assert_eq!(node.display_name(), "Sign");
            let node_id = node.storage_id();
            assert!(matches!(
                edited
                    .insert_numeric_node_in_network_copy(nested, operation, position, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let undone = edited
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one interaction");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("restore insertion");
            assert!(
                redone
                    .graph_nodes()
                    .await
                    .expect("redone graph")
                    .iter()
                    .any(|node| node.storage_id() == node_id && node.network_id() == nested.id)
            );
        });
    }

    #[test]
    fn at_gra_graph_005_1_inserts_pinned_defaults_and_embeds_the_source_declaration() {
        futures::executor::block_on(async {
            let blank = create_gdd_folder(781, 419, "Pinned Graphite".into(), budget())
                .await
                .expect("new source working folder");
            assert!(blank.graph_nodes().await.expect("blank graph").is_empty());
            let opaque_path = "opaque/notes/provenance.txt";
            let opaque_bytes = b"retain this entry when adding the first node";
            let mut original_files = blank
                .folder_files()
                .expect("blank folder files")
                .into_iter()
                .map(|(path, bytes)| (path.to_str().expect("fixture path").to_owned(), bytes))
                .collect::<Vec<_>>();
            original_files.push((opaque_path.into(), opaque_bytes.to_vec()));
            let original = inspect_gdd_folder_files(original_files, budget())
                .await
                .expect("admit a blank graph with an opaque entry");
            let strict_budget = GddArchiveBudget {
                max_entries: original.entries.names.len(),
                ..budget()
            };
            assert!(matches!(
                original
                    .insert_root_numeric_node_in_copy(
                        UnaryNumericOperation::AbsoluteValue,
                        [0, 0],
                        strict_budget
                    )
                    .await,
                Err(GddArchiveError::BudgetExceeded)
            ));
            let operation = UnaryNumericOperation::AbsoluteValue;
            let grid_position = [-11, 6];
            let inserted = original
                .insert_root_numeric_node_in_copy(operation, grid_position, budget())
                .await
                .expect("insert the exact pinned node template");
            assert_eq!(inserted.recovery().network_count, 1);
            let graph = inserted.runtime_graph().await.expect("reopened graph");
            let [node] = graph.nodes.as_slice() else {
                panic!("expected the single inserted node");
            };
            assert!(node.network_path.is_empty());
            let node_id = node.local_id;
            let interface = graph.into_interface().expect("pinned source editor");
            let definition = resolve_proto_node_type(numeric_node_identifier(operation))
                .expect("pinned Absolute Value definition");
            let mut template = definition.default_node_template();
            let position = IVec2::from_array(grid_position);
            template.node_type_metadata = NodeTypePersistentMetadata::node(position);
            assert_eq!(
                interface.create_node_template(&node_id, &[]),
                Some(template)
            );
            assert_eq!(interface.position(&node_id, &[]), Some(position));
            let resource_hashes = inserted
                .document
                .resource_hashes()
                .await
                .expect("embedded node definition");
            assert_eq!(resource_hashes.len(), 1);
            assert_eq!(
                inserted
                    .read_entry(opaque_path)
                    .expect("retained provenance")
                    .as_slice(),
                opaque_bytes
            );

            let undo = inserted
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores an empty document");
            assert!(
                undo.graph_nodes()
                    .await
                    .expect("no remaining nodes")
                    .is_empty()
            );
            let empty_root = Registry::from_runtime_with_metadata(
                &NodeNetwork::default(),
                &NoMetadata,
                &graphite_resource::ResourceRegistry::new(),
                original.document.session().peer(),
            )
            .expect("pinned empty graph storage");
            assert!(undo.document.registry().value_equal(&empty_root));
            assert_eq!(
                undo.document
                    .resource_hashes()
                    .await
                    .expect("preserved redo declaration"),
                resource_hashes
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the exact node");
            assert!(
                redo.document
                    .registry()
                    .value_equal(inserted.document.registry())
            );
            assert!(
                original
                    .graph_nodes()
                    .await
                    .expect("unchanged original")
                    .is_empty()
            );

            let duplicate = inserted
                .insert_root_numeric_node_in_copy(operation, [3, 4], budget())
                .await
                .expect("insert another instance");
            let nodes = duplicate
                .graph_nodes()
                .await
                .expect("reopened distinct instances");
            let [first, second] = nodes.as_slice() else {
                panic!("expected two node instances");
            };
            assert_ne!(first.storage_id, second.storage_id);
            assert_eq!(
                duplicate
                    .document
                    .resource_hashes()
                    .await
                    .expect("one reused declaration"),
                resource_hashes
            );
            for operation in [
                UnaryNumericOperation::Ceiling,
                UnaryNumericOperation::Floor,
                UnaryNumericOperation::Round,
                UnaryNumericOperation::Sign,
            ] {
                let copy = original
                    .insert_root_numeric_node_in_copy(operation, [2, -3], budget())
                    .await
                    .expect("insert another admitted definition");
                let nodes = copy.graph_nodes().await.expect("reopened definition");
                let [node] = nodes.as_slice() else {
                    panic!("expected one inserted definition");
                };
                assert!(node.label().contains(operation.name()));
            }
        });
    }

    #[test]
    fn at_gra_graph_020_022_1_adds_secondary_ports_with_hidden_primary_defaults() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let graph = original.graph_inspection().await.expect("source networks");
            let [_, nested] = graph.networks.as_slice() else {
                panic!("expected one empty nested interface");
            };
            assert_eq!(nested.import_count, 0);
            assert_eq!(nested.export_count, 0);
            let imported = original
                .append_secondary_nested_network_import_in_copy(nested, budget())
                .await
                .expect("add both initial input slots in one interaction");
            assert_eq!(imported.history_len(), original.history_len() + 1);
            let graph = imported
                .graph_inspection()
                .await
                .expect("saved initial imports");
            let current = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same nested network");
            assert_eq!(current.import_count, 2);
            assert_eq!(current.export_count, 0);
            assert_eq!(current.primary_import_exposed(), Some(false));
            let both = imported
                .append_secondary_nested_network_export_in_copy(current, budget())
                .await
                .expect("also create the initial export slots");
            let graph = both.graph_inspection().await.expect("saved ports");
            let current = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same saved interface");
            assert_eq!(current.import_count, 2);
            assert_eq!(current.export_count, 2);
            assert_eq!(current.primary_export_exposed(), Some(false));
            assert_eq!(current.empty_exports(), &[1]);
            let interface = both
                .runtime_graph()
                .await
                .expect("saved runtime")
                .into_interface()
                .expect("pinned source editor");
            let Some((&owner_id, parent_path)) = nested.path.split_last() else {
                panic!("expected saved enclosing node");
            };
            let ports = [
                NodeInput::value(TaggedValue::None, false),
                NodeInput::value(TaggedValue::None, true),
            ];
            assert_eq!(
                interface
                    .document_node(&owner_id, parent_path)
                    .expect("original enclosing node")
                    .inputs,
                ports
            );
            assert_eq!(
                interface
                    .nested_network(&nested.path)
                    .expect("original nested graph")
                    .exports,
                ports
            );
            assert_eq!(both.history_len(), imported.history_len() + 3);
            let undo = both
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo removes both added exports");
            assert!(
                undo.document
                    .registry()
                    .value_equal(imported.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("redo restores both added exports");
            assert!(
                redo.document
                    .registry()
                    .value_equal(both.document.registry())
            );
            let before_import = imported
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo also removes both initial imports together");
            assert!(
                before_import
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_019_020_1_appends_source_imports_and_keeps_opaque_slot_metadata() {
        use graphite_document_graph_storage::{AttributeDelta, Value, attr::node::input::ui::NAME};

        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let graph = original.graph_inspection().await.expect("source networks");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected one nested network");
            };
            assert_eq!(nested.import_count, 0);
            assert!(matches!(
                original
                    .append_nested_network_import_in_copy(root, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphImportAppend)
            ));
            let primary = original
                .append_nested_network_import_in_copy(nested, budget())
                .await
                .expect("source first import");
            assert!(matches!(
                primary
                    .append_nested_network_import_in_copy(nested, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let primary_graph = primary.graph_inspection().await.expect("first import");
            let owner_node = primary_graph
                .nodes
                .iter()
                .find(|node| node.nested_network() == Some(nested.id))
                .expect("source owner");
            let (mut owner, staged) = primary
                .fork_working_copy(&budget())
                .await
                .expect("fixture fork");
            let ops = [
                (NAME, "Original input name"),
                ("fixture::opaque-input-attribute", "original opaque payload"),
            ]
            .into_iter()
            .map(|(key, value)| RegistryDelta::ChangeNodeInputAttribute {
                id: owner_node.storage_id,
                index: 0,
                delta: AttributeDelta {
                    key: key.into(),
                    value: Some(Value::Str(value.into())),
                },
            })
            .collect();
            commit_graph_node_edit(
                &mut owner,
                ConstructedOps {
                    ops,
                    declarations: Default::default(),
                },
            )
            .expect("add named and opaque metadata to the original slot");
            let named = staged
                .reopen_after_edit(&budget())
                .await
                .expect("read original slot metadata");
            let graph = named.graph_inspection().await.expect("addressed network");
            let selected = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same saved network");
            assert_eq!(selected.import_count, 1);
            let secondary = named
                .append_nested_network_import_in_copy(selected, budget())
                .await
                .expect("source next import");
            let actual = secondary
                .runtime_graph()
                .await
                .expect("saved copy")
                .into_interface()
                .expect("source editor");
            let Some((&owner_id, parent_path)) = nested.path.split_last() else {
                panic!("expected an owning node");
            };
            let input = NodeInput::value(TaggedValue::None, true);
            let owner = actual
                .document_node(&owner_id, parent_path)
                .expect("original owner");
            assert_eq!(owner.inputs, vec![input.clone(), input]);
            let registry = secondary.document.registry();
            let stored = &registry.node_instances[&owner_node.storage_id];
            let original_slot = named.document.registry().node_instances[&owner_node.storage_id]
                .inputs()
                .first()
                .expect("original named input");
            let retained_slot = stored.inputs().first().expect("retained first input");
            assert_eq!(retained_slot.attributes, original_slot.attributes);
            assert_eq!(
                retained_slot
                    .attributes
                    .get_typed::<String>("fixture::opaque-input-attribute")
                    .as_deref(),
                Some("original opaque payload")
            );
            assert_eq!(registry.networks, named.document.registry().networks);
            assert_eq!(secondary.history_len(), named.history_len() + 1);
            let undo = secondary
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo removes only the latest import");
            assert!(
                undo.document
                    .registry()
                    .value_equal(named.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores that import");
            assert!(redo.document.registry().value_equal(registry));
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    async fn network_import_fixture() -> (InspectedGddV1, StorageNodeId) {
        use graphite_document_graph_storage::{
            AttributeDelta, NoMetadata, Value, attr::node::input::ui::NAME,
        };
        use graphite_graph_craft::concrete;
        use graphite_math_nodes::{absolute_value, add};
        use graphite_resource::ResourceRegistry;

        let mut nested = NodeNetwork {
            nodes: [(NodeId(7), 0), (NodeId(9), 1), (NodeId(11), 1)]
                .into_iter()
                .map(|(node_id, index)| {
                    (
                        node_id,
                        DocumentNode {
                            implementation: DocumentNodeImplementation::ProtoNode(
                                absolute_value::IDENTIFIER,
                            ),
                            inputs: vec![NodeInput::import(concrete!(f64), index)],
                            ..Default::default()
                        },
                    )
                })
                .collect(),
            exports: vec![NodeInput::node(NodeId(13), 0)],
            ..Default::default()
        };
        nested.nodes.insert(
            NodeId(13),
            DocumentNode {
                implementation: DocumentNodeImplementation::ProtoNode(add::IDENTIFIER),
                inputs: vec![NodeInput::node(NodeId(7), 0), NodeInput::node(NodeId(9), 0)],
                ..Default::default()
            },
        );
        for (node_id, value) in [
            (NodeId(14), TaggedValue::Number(-37.)),
            (NodeId(15), TaggedValue::Integer(-37)),
        ] {
            nested.nodes.insert(
                node_id,
                DocumentNode {
                    implementation: DocumentNodeImplementation::ProtoNode(
                        absolute_value::IDENTIFIER,
                    ),
                    inputs: vec![NodeInput::value(value, true)],
                    ..Default::default()
                },
            );
        }
        let root = NodeNetwork {
            nodes: [(
                NodeId(24),
                DocumentNode {
                    implementation: DocumentNodeImplementation::Network(nested),
                    inputs: vec![
                        NodeInput::value(TaggedValue::Number(-2.), true),
                        NodeInput::value(TaggedValue::Number(7.), true),
                    ],
                    ..Default::default()
                },
            )]
            .into_iter()
            .collect(),
            exports: vec![NodeInput::node(NodeId(24), 0)],
            ..Default::default()
        };
        let mut owner = document();
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&root, &NoMetadata, &ResourceRegistry::new(), &resources)
            .expect("write source bindings");
        let bytes: Arc<[u8]> = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP")
            .into();
        let source = open_gdd_zip(&bytes, budget())
            .await
            .expect("original graph");
        let graph = source.graph_inspection().await.expect("source projection");
        let [_, nested] = graph.networks.as_slice() else {
            panic!("expected the root and its nested network");
        };
        let owner_id = graph
            .nodes
            .iter()
            .find(|node| node.nested_network() == Some(nested.id))
            .expect("encapsulating node")
            .storage_id;
        let consumer = graph
            .nodes
            .iter()
            .find(|node| node.local_id == NodeId(9))
            .expect("the first secondary consumer")
            .storage_id;
        let (mut owner, staged) = source
            .fork_working_copy(&budget())
            .await
            .expect("fixture fork");
        commit_graph_node_edit(
            &mut owner,
            ConstructedOps {
                ops: vec![
                    RegistryDelta::ChangeNodeInputAttribute {
                        id: owner_id,
                        index: 1,
                        delta: AttributeDelta {
                            key: "fixture::opaque-import".into(),
                            value: Some(Value::Str("surviving input".into())),
                        },
                    },
                    RegistryDelta::ChangeNodeInputAttribute {
                        id: consumer,
                        index: 0,
                        delta: AttributeDelta {
                            key: NAME.into(),
                            value: Some(Value::Str("Saved operand".into())),
                        },
                    },
                ],
                declarations: Default::default(),
            },
        )
        .expect("fixture adds an opaque input attribute");
        let original = staged
            .reopen_after_edit(&budget())
            .await
            .expect("named input fixture");
        assert_eq!(source.original_bytes(), bytes.as_ref());
        (original, owner_id)
    }

    #[test]
    fn at_gra_graph_023_1_removes_an_import_and_repairs_its_scoped_wires() {
        use graphite_graph_craft::concrete;

        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let nested_path = [NodeId(24)];
            let graph = original.graph_inspection().await.expect("bound networks");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the source networks after reopening");
            };
            for (network, index) in [(root, 0), (nested, nested.import_count)] {
                assert!(matches!(
                    original
                        .remove_nested_network_import_in_copy(network, index, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphImportRemoval)
                ));
            }
            let mut pinned = original
                .runtime_graph()
                .await
                .expect("pinned runtime")
                .into_interface()
                .expect("pinned interface");
            pinned.remove_import(0, &nested_path);
            let disconnected = InputConnector::Node {
                node_id: NodeId(7),
                input_index: 0,
            };
            let expected_default = pinned
                .input_from_connector(&disconnected, &nested_path)
                .expect("pinned disconnected input");
            assert!(matches!(
                expected_default,
                NodeInput::Value { exposed: true, .. }
            ));

            let copy = original
                .remove_nested_network_import_in_copy(nested, 0, budget())
                .await
                .expect("remove the addressed primary import");
            assert!(matches!(
                copy.remove_nested_network_import_in_copy(nested, 0, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let interface = copy
                .runtime_graph()
                .await
                .expect("saved graph")
                .into_interface()
                .expect("saved interface");
            assert_eq!(
                interface.input_from_connector(&disconnected, &nested_path),
                Some(expected_default)
            );
            for node_id in [NodeId(9), NodeId(11)] {
                let shifted = InputConnector::Node {
                    node_id,
                    input_index: 0,
                };
                assert_eq!(
                    interface.input_from_connector(&shifted, &nested_path),
                    Some(&NodeInput::import(concrete!(f64), 0))
                );
            }
            let registry = copy.document.registry();
            let original_registry = original.document.registry();
            let [_, survivor] = original_registry.node_instances[&owner_id].inputs() else {
                panic!("expected both source inputs");
            };
            let [retained] = registry.node_instances[&owner_id].inputs() else {
                panic!("expected just the shifted input");
            };
            assert_eq!(retained.input, survivor.input);
            assert_eq!(retained.attributes, survivor.attributes);
            assert_eq!(registry.networks, original_registry.networks);
            assert_eq!(copy.history_len(), original.history_len() + 4);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(undo.document.registry().value_equal(original_registry));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_011_1_pastes_a_nested_numeric_template_with_a_fresh_network_identity() {
        use graphite_document_graph_storage::attr::node::input::ui::NAME;

        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let graph = original.graph_inspection().await.expect("source topology");
            let [root, source_network] = graph.networks.as_slice() else {
                panic!("expected the source root and one nested network");
            };
            let holder = graph
                .nodes
                .iter()
                .find(|node| node.storage_id == owner_id)
                .expect("original encapsulating node");
            assert!(holder.can_copy_graph_node());
            let clipboard = original
                .copy_graph_nodes_to_clipboard(std::slice::from_ref(holder), budget())
                .await
                .expect("copy the complete published nested template");
            let copy = original
                .paste_graph_nodes_in_copy(root, &clipboard, budget())
                .await
                .expect("paste a nested graph under fresh storage IDs");
            let pasted = copy
                .graph_inspection()
                .await
                .expect("reopen the nested scope");
            assert_eq!(pasted.networks.len(), 3);
            let network = pasted
                .networks
                .iter()
                .find(|network| network.parent == Some(root.id) && network.id != source_network.id)
                .expect("fresh nested identity");
            assert_eq!(network.import_count, source_network.import_count);
            assert_eq!(network.export_count, source_network.export_count);
            assert_eq!(
                network.primary_import_exposed(),
                source_network.primary_import_exposed()
            );
            let child = pasted
                .nodes
                .iter()
                .find(|node| node.network_id() == network.id && node.local_id == NodeId(9))
                .expect("remapped original inner consumer");
            let original_child = graph
                .nodes
                .iter()
                .find(|node| {
                    node.network_id() == source_network.id && node.local_id == child.local_id
                })
                .expect("original inner consumer");
            assert_ne!(child.storage_id, original_child.storage_id);
            assert_eq!(child.numeric_import_source(), Some(1));
            let registry = copy.document.registry();
            let [stored] = registry.node_instances[&child.storage_id].inputs() else {
                panic!("expected one copied inner input");
            };
            assert_eq!(
                stored.attributes.get_typed::<String>(NAME).as_deref(),
                Some("Saved operand")
            );
            let result = pasted
                .nodes
                .iter()
                .find(|node| node.network_id() == network.id && node.local_id == NodeId(13))
                .expect("copied two-input output node");
            assert!(matches!(
                copy.preview_graph_numeric_output(result)
                    .await
                    .expect("execute the copied numeric network"),
                GddNumericValue::Number(9.)
            ));
            for (id, node) in &original.document.registry().node_instances {
                assert_eq!(registry.node_instances.get(id), Some(node));
            }
            for (id, scope) in &original.document.registry().networks {
                assert_eq!(registry.networks.get(id), Some(scope));
            }
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo removes the nested network and the whole copied subtree");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the complete group");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_011_1_pastes_into_a_fresh_graph_and_undo_restores_the_empty_root() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let source = open_gdd_zip(&bytes, budget())
                .await
                .expect("original numeric graph");
            let nodes = source.graph_nodes().await.expect("source nodes");
            let consumer = nodes
                .iter()
                .find(|node| node.local_id == NodeId(11))
                .expect("original downstream");
            let producer = consumer
                .numeric_connection_source(&nodes)
                .expect("original upstream");
            let clipboard = source
                .copy_graph_nodes_to_clipboard(&[producer.clone(), consumer.clone()], budget())
                .await
                .expect("pinned clipboard group");
            let empty = create_gdd_folder(792, 419, "Blank Graphite".into(), budget())
                .await
                .expect("fresh native folder");
            let original = empty.folder_files().expect("original blank bytes");
            assert!(empty.graph_root_needs_initialization());
            let graph = empty.graph_inspection().await.expect("inspect blank scope");
            assert!(graph.nodes.is_empty());
            let [root] = graph.networks.as_slice() else {
                panic!("expected the virtual root of the new folder");
            };
            assert_eq!(root.id, ROOT_NETWORK);
            assert_eq!(root.parent, None);
            let copy = empty
                .paste_graph_nodes_in_copy(root, &clipboard, budget())
                .await
                .expect("paste in the blank root");
            let nodes = copy.graph_nodes().await.expect("saved root nodes");
            assert_eq!(nodes.len(), 2);
            let consumer = nodes
                .iter()
                .find(|node| node.has_numeric_connection())
                .expect("pasted downstream");
            assert!(consumer.numeric_connection_source(&nodes).is_some());
            assert!(matches!(
                copy.preview_graph_numeric_output(consumer)
                    .await
                    .expect("execute the saved group"),
                GddNumericValue::Number(-1.)
            ));
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores the empty graph");
            let graph = undo.graph_inspection().await.expect("saved empty root");
            assert!(graph.nodes.is_empty());
            assert_eq!(graph.networks.len(), 1);
            assert_eq!(
                graph.networks.first().map(|network| network.id),
                Some(ROOT_NETWORK)
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the complete group");
            assert!(
                redo.document
                    .registry()
                    .value_equal(copy.document.registry())
            );
            assert_eq!(
                empty.folder_files().expect("original blank folder"),
                original
            );
        });
    }

    #[test]
    fn at_gra_graph_011_1_cuts_a_closed_numeric_group_in_one_interaction() {
        futures::executor::block_on(async {
            let original = open_gdd_zip(&numeric_graph_fixture().await, budget())
                .await
                .expect("source graph");
            let nodes = original.graph_nodes().await.expect("source nodes");
            let consumer = nodes
                .iter()
                .find(|node| node.local_id == NodeId(11))
                .expect("downstream node");
            let source = consumer
                .numeric_connection_source(&nodes)
                .expect("upstream node");
            let unrelated = nodes
                .iter()
                .find(|node| node.local_id == NodeId(9))
                .expect("unrelated node in the same scope");
            for linked in [source, consumer] {
                assert!(matches!(
                    original
                        .cut_graph_nodes_in_copy(&[linked.clone(), unrelated.clone()], budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphCutSelection)
                ));
            }
            let (cut, clipboard) = original
                .cut_graph_nodes_in_copy(&[source.clone(), consumer.clone()], budget())
                .await
                .expect("cut both connected nodes");
            let surviving = &cut.document.registry().node_instances;
            for (id, node) in &original.document.registry().node_instances {
                if [source.storage_id, consumer.storage_id].contains(id) {
                    assert!(!surviving.contains_key(id));
                } else {
                    assert_eq!(surviving.get(id), Some(node));
                }
            }
            let undo = cut
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores both nodes and their wire");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo cuts both nodes again");
            assert!(
                redo.document
                    .registry()
                    .value_equal(cut.document.registry())
            );
            let graph = cut.graph_inspection().await.expect("saved cut graph");
            let network = graph
                .networks
                .iter()
                .find(|network| network.id == consumer.network_id())
                .expect("original nested scope");
            let pasted = cut
                .paste_graph_nodes_in_copy(network, &clipboard, budget())
                .await
                .expect("paste the cut nodes");
            let nodes = pasted.graph_nodes().await.expect("restored group");
            let pasted_consumer = nodes
                .iter()
                .find(|node| node.has_numeric_connection())
                .expect("restored internal wire");
            assert_ne!(pasted_consumer.storage_id, consumer.storage_id);
            assert!(matches!(
                pasted
                    .preview_graph_numeric_output(pasted_consumer)
                    .await
                    .expect("evaluate the pasted group"),
                GddNumericValue::Number(-1.)
            ));
        });
    }

    #[test]
    fn at_gra_graph_011_1_copies_and_pastes_a_named_numeric_group_with_fresh_wires() {
        use graphite_document_graph_storage::attr::node::input::ui::NAME;

        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let graph = original.graph_inspection().await.expect("original graph");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected one nested graph");
            };
            let consumer = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(11))
                .expect("original named consumer");
            let producer = consumer
                .numeric_connection_source(&graph.nodes)
                .expect("original producer");
            let root_node = graph
                .nodes
                .iter()
                .find(|node| node.network_id() == root.id && node.can_copy_graph_node())
                .expect("root numeric node");
            assert!(matches!(
                original
                    .copy_graph_nodes_to_clipboard(&[root_node.clone(), consumer.clone()], budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphClipboardSelection)
            ));
            let clipboard = original
                .copy_graph_nodes_to_clipboard(&[consumer.clone(), producer.clone()], budget())
                .await
                .expect("copy the explicitly selected group");
            let encoded = clipboard
                .strip_prefix("graphite: ")
                .expect("pinned clipboard prefix");
            let [ClipboardItem::Nodes(nodes)] = serde_json::from_str::<[ClipboardItem; 1]>(encoded)
                .expect("deserialize with the pinned source type")
            else {
                panic!("expected the pinned node group");
            };
            let [(_, first), (_, second)] = nodes.as_slice() else {
                panic!("exactly two copied nodes");
            };
            assert!(matches!(first.inputs.as_slice(), [NodeInput::Value { .. }]));
            assert_eq!(second.inputs, [NodeInput::node(NodeId(0), 0)]);
            assert!(matches!(
                original
                    .paste_graph_nodes_in_copy(nested, "unrelated text", budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphClipboard)
            ));
            let copy = original
                .paste_graph_nodes_in_copy(nested, &clipboard, budget())
                .await
                .expect("paste both nodes in one history interaction");
            assert!(matches!(
                copy.paste_graph_nodes_in_copy(nested, &clipboard, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let pasted = copy.graph_nodes().await.expect("saved pasted nodes");
            let mut added = pasted
                .iter()
                .filter(|node| {
                    !graph
                        .nodes
                        .iter()
                        .any(|original| original.storage_id == node.storage_id)
                })
                .collect::<Vec<_>>();
            assert_eq!(added.len(), 2);
            assert!(added.iter().all(|node| node.network_id() == nested.id));
            let index = added
                .iter()
                .position(|node| node.has_numeric_connection())
                .expect("exactly one pasted consumer");
            let pasted_consumer = added.remove(index);
            let [pasted_producer] = added.as_slice() else {
                panic!("expected one pasted producer");
            };
            assert_eq!(
                pasted_consumer
                    .numeric_connection_source(&pasted)
                    .expect("fresh internal wire")
                    .storage_id,
                pasted_producer.storage_id
            );
            let stored = &copy.document.registry().node_instances[&pasted_consumer.storage_id];
            let [input] = stored.inputs() else {
                panic!("expected the retained named input");
            };
            assert_eq!(
                input.attributes.get_typed::<String>(NAME).as_deref(),
                Some("Original wired input")
            );
            let interface = copy
                .runtime_graph()
                .await
                .expect("saved positions")
                .into_interface()
                .expect("pinned editor");
            for node in [pasted_consumer, *pasted_producer] {
                assert_eq!(
                    interface.position(&node.local_id, &node.network_path),
                    Some(IVec2::new(2, 2))
                );
            }
            let output = copy
                .preview_graph_numeric_output(pasted_consumer)
                .await
                .expect("pinned copied graph output");
            assert!(output.same_value(&GddNumericValue::Number(-1.)));
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo removes the whole node group");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the group");
            assert!(
                redo.document
                    .registry()
                    .value_equal(copy.document.registry())
            );

            let (destination, _) = network_export_fixture(false).await;
            let graph = destination
                .graph_inspection()
                .await
                .expect("different Graphite document");
            let [root, _] = graph.networks.as_slice() else {
                panic!("expected root destination");
            };
            let pasted = destination
                .paste_graph_nodes_in_copy(root, &clipboard, budget())
                .await
                .expect("paste with fresh source declarations into a second document");
            assert_eq!(
                pasted
                    .graph_nodes()
                    .await
                    .expect("second saved graph")
                    .len(),
                graph.nodes.len() + 2
            );
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_010_1_inserts_a_typed_node_on_a_wire_as_one_history_interaction() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("node wire fixture");
            let nodes = original.graph_nodes().await.expect("source numeric nodes");
            let target = nodes
                .iter()
                .find(|node| node.local_id == NodeId(11))
                .expect("the existing connected consumer");
            let producer = target
                .numeric_connection_source(&nodes)
                .expect("prior wired source");
            assert!(matches!(
                original
                    .insert_numeric_node_on_wire_in_copy(
                        producer,
                        UnaryNumericOperation::AbsoluteValue,
                        budget()
                    )
                    .await,
                Err(GddArchiveError::UnsupportedGraphWireInsertion)
            ));
            for (operation, value) in [
                (UnaryNumericOperation::AbsoluteValue, 1.),
                (UnaryNumericOperation::Ceiling, -1.),
                (UnaryNumericOperation::Floor, -1.),
                (UnaryNumericOperation::Round, -1.),
                (UnaryNumericOperation::Sign, -1.),
            ] {
                let copy = original
                    .insert_numeric_node_on_wire_in_copy(target, operation, budget())
                    .await
                    .expect("insert an exact pinned source node on the wire");
                assert!(matches!(
                    copy.insert_numeric_node_on_wire_in_copy(target, operation, budget())
                        .await,
                    Err(GddArchiveError::StaleGraphNode)
                ));
                let nodes = copy.graph_nodes().await.expect("saved numeric graph");
                let consumer = nodes
                    .iter()
                    .find(|node| node.storage_id == target.storage_id)
                    .expect("same downstream identity");
                let added = consumer
                    .numeric_connection_source(&nodes)
                    .expect("new intermediary");
                assert_ne!(added.storage_id, producer.storage_id);
                assert!(added.label().contains(operation.name()));
                assert_eq!(
                    added
                        .numeric_connection_source(&nodes)
                        .expect("original upstream")
                        .storage_id,
                    producer.storage_id
                );
                let output = copy
                    .preview_graph_numeric_output(consumer)
                    .await
                    .expect("pinned output");
                assert!(output.same_value(&GddNumericValue::Number(value)));
                assert_eq!(
                    copy.document.registry().networks,
                    original.document.registry().networks
                );
                let undo = copy
                    .move_history_in_copy(GddHistoryDirection::Undo, budget())
                    .await
                    .expect("one undo removes the node and restores the prior wire");
                assert!(
                    undo.document
                        .registry()
                        .value_equal(original.document.registry())
                );
                let redo = undo
                    .move_history_in_copy(GddHistoryDirection::Redo, budget())
                    .await
                    .expect("one redo restores both new edges");
                assert!(
                    redo.document
                        .registry()
                        .value_equal(copy.document.registry())
                );
            }
            assert_eq!(original.original_bytes(), bytes.as_ref());

            let (imported, _) = network_import_fixture().await;
            let nodes = imported.graph_nodes().await.expect("import wire fixture");
            let target = nodes
                .iter()
                .find(|node| node.local_id == NodeId(9))
                .expect("same nested numeric import consumer");
            let copy = imported
                .insert_numeric_node_on_wire_in_copy(target, UnaryNumericOperation::Sign, budget())
                .await
                .expect("insert on a nested import");
            let nodes = copy.graph_nodes().await.expect("reopened source");
            let consumer = nodes
                .iter()
                .find(|node| node.storage_id == target.storage_id)
                .expect("same downstream node");
            let intermediary = consumer
                .numeric_connection_source(&nodes)
                .expect("inserted Sign");
            assert_eq!(intermediary.numeric_import_source(), Some(1));
            let output = copy
                .preview_graph_numeric_output(consumer)
                .await
                .expect("pinned result");
            assert!(output.same_value(&GddNumericValue::Number(1.)));
        });
    }

    #[test]
    fn at_gra_graph_008_1_reconnects_a_numeric_import_without_replacing_its_consumer_slot() {
        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let graph = original.graph_inspection().await.expect("source networks");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the source scoped network");
            };
            let target = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(9))
                .expect("original numeric import consumer");
            let other = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(11))
                .expect("another consumer of the prior import");
            let empty = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(14))
                .expect("an existing unconnected input");
            assert_eq!(target.numeric_import_source(), Some(1));
            assert_eq!(
                nested
                    .compatible_numeric_imports(target)
                    .collect::<Vec<_>>(),
                [0]
            );
            for (scope, index, target) in
                [(root, 0, target), (nested, 1, target), (nested, 0, empty)]
            {
                assert!(matches!(
                    original
                        .reconnect_numeric_network_import_in_copy(scope, index, target, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphImportConnection)
                ));
            }
            assert!(matches!(
                original
                    .connect_numeric_network_import_in_copy(nested, 0, target, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphImportConnection)
            ));
            let copy = original
                .reconnect_numeric_network_import_in_copy(nested, 0, target, budget())
                .await
                .expect("reconnect the existing consumer to the first import");
            assert!(matches!(
                copy.reconnect_numeric_network_import_in_copy(nested, 0, target, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let graph = copy.graph_inspection().await.expect("reopened graph");
            let wired = graph
                .nodes
                .iter()
                .find(|node| node.storage_id == target.storage_id)
                .expect("same saved consumer");
            assert_eq!(wired.numeric_import_source(), Some(0));
            let output = copy
                .preview_graph_numeric_output(wired)
                .await
                .expect("pinned replacement output");
            assert!(output.same_value(&GddNumericValue::Number(2.)));
            let unchanged = graph
                .nodes
                .iter()
                .find(|node| node.storage_id == other.storage_id)
                .expect("same secondary consumer");
            assert_eq!(unchanged.numeric_import_source(), Some(1));
            let output = copy
                .preview_graph_numeric_output(unchanged)
                .await
                .expect("pinned original output");
            assert!(output.same_value(&GddNumericValue::Number(7.)));
            let registry = copy.document.registry();
            let source = original.document.registry();
            assert_eq!(registry.networks, source.networks);
            assert_eq!(
                registry.node_instances[&owner_id],
                source.node_instances[&owner_id]
            );
            assert_eq!(
                registry.node_instances[&other.storage_id],
                source.node_instances[&other.storage_id]
            );
            let [prior] = source.node_instances[&target.storage_id].inputs() else {
                panic!("expected the original named slot");
            };
            let [replaced] = registry.node_instances[&target.storage_id].inputs() else {
                panic!("expected the same named slot after reconnect");
            };
            assert_eq!(replaced.attributes, prior.attributes);
            assert_eq!(copy.history_len(), original.history_len() + 2);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(undo.document.registry().value_equal(source));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_009_1_disconnects_the_addressed_import_and_keeps_its_other_consumers() {
        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let graph = original.graph_inspection().await.expect("original graph");
            let target = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(9))
                .expect("named secondary import consumer");
            let other = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(11))
                .expect("a second consumer of the same import");
            assert_eq!(target.numeric_import_source(), Some(1));
            assert!(target.has_numeric_connection());
            let plain = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(14))
                .expect("literal");
            assert!(matches!(
                original
                    .disconnect_graph_numeric_input_in_copy(plain, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphConnection)
            ));
            let input = InputConnector::primary_input(target.local_id);
            let path = &target.network_path;
            let mut pinned = original
                .runtime_graph()
                .await
                .expect("original runtime")
                .into_interface()
                .expect("pinned editor");
            pinned.disconnect_input(&input, path);
            let expected = pinned
                .input_from_connector(&input, path)
                .expect("source fallback");
            assert!(matches!(expected, NodeInput::Value { exposed: true, .. }));

            let copy = original
                .disconnect_graph_numeric_input_in_copy(target, budget())
                .await
                .expect("disconnect a single addressed import wire");
            assert!(matches!(
                copy.disconnect_graph_numeric_input_in_copy(target, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let actual = copy
                .runtime_graph()
                .await
                .expect("saved runtime")
                .into_interface()
                .expect("pinned source editor");
            assert_eq!(actual.input_from_connector(&input, path), Some(expected));
            let registry = copy.document.registry();
            let source = original.document.registry();
            assert_eq!(registry.networks, source.networks);
            assert_eq!(
                registry.node_instances[&owner_id],
                source.node_instances[&owner_id]
            );
            assert_eq!(
                registry.node_instances[&other.storage_id],
                source.node_instances[&other.storage_id]
            );
            let [prior] = source.node_instances[&target.storage_id].inputs() else {
                panic!("expected the original named input");
            };
            let [saved] = registry.node_instances[&target.storage_id].inputs() else {
                panic!("expected the same slot after disconnect");
            };
            assert_eq!(saved.attributes, prior.attributes);
            assert_eq!(copy.history_len(), original.history_len() + 1);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(undo.document.registry().value_equal(source));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_007_1_connects_a_typed_literal_import_to_the_addressed_node() {
        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let graph = original.graph_inspection().await.expect("source networks");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the root and a standalone imported scope");
            };
            let target = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(14))
                .expect("ordinary number input");
            let integer = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(15))
                .expect("incompatible integer input");
            assert_eq!(
                nested
                    .compatible_numeric_imports(target)
                    .collect::<Vec<_>>(),
                [0, 1]
            );
            for (network, import, target) in
                [(root, 1, target), (nested, 1, integer), (nested, 2, target)]
            {
                assert!(matches!(
                    original
                        .connect_numeric_network_import_in_copy(network, import, target, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphImportConnection)
                ));
            }
            let copy = original
                .connect_numeric_network_import_in_copy(nested, 1, target, budget())
                .await
                .expect("connect the compatible secondary import");
            assert!(matches!(
                copy.connect_numeric_network_import_in_copy(nested, 1, target, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let graph = copy.graph_inspection().await.expect("saved graph");
            let node = graph
                .nodes
                .iter()
                .find(|node| node.storage_id == target.storage_id)
                .expect("same consumer");
            assert_eq!(node.numeric_import_source(), Some(1));
            let value = copy
                .preview_graph_numeric_output(node)
                .await
                .expect("execute with the stored import");
            assert!(value.same_value(&GddNumericValue::Number(7.)));
            let before = original
                .preview_graph_numeric_output(target)
                .await
                .expect("original literal");
            assert!(before.same_value(&GddNumericValue::Number(37.)));
            let registry = copy.document.registry();
            let original_registry = original.document.registry();
            assert_eq!(registry.networks, original_registry.networks);
            assert_eq!(
                registry.node_instances[&owner_id],
                original_registry.node_instances[&owner_id]
            );
            assert_eq!(copy.history_len(), original.history_len() + 1);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(undo.document.registry().value_equal(original_registry));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_025_1_reorders_imports_with_values_metadata_and_identical_output() {
        use graphite_graph_craft::concrete;

        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let graph = original
                .graph_inspection()
                .await
                .expect("source projection");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the source nested network");
            };
            for (network, from, to) in [(root, 0, 1), (nested, 0, 0), (nested, 0, 2)] {
                assert!(matches!(
                    original
                        .move_nested_network_import_in_copy(network, from, to, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphImportMove)
                ));
            }
            let copy = original
                .move_nested_network_import_in_copy(nested, 0, 1, budget())
                .await
                .expect("move the addressed import down");
            assert!(matches!(
                copy.move_nested_network_import_in_copy(nested, 1, 0, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let original_registry = original.document.registry();
            let registry = copy.document.registry();
            let old_slots = original_registry.node_instances[&owner_id].inputs();
            let new_slots = registry.node_instances[&owner_id].inputs();
            assert_eq!(new_slots.len(), old_slots.len());
            for (current, previous) in new_slots.iter().zip(old_slots.iter().rev()) {
                assert_eq!(current.input, previous.input);
                assert_eq!(current.attributes, previous.attributes);
            }
            assert_eq!(registry.networks, original_registry.networks);
            let interface = copy
                .runtime_graph()
                .await
                .expect("saved graph")
                .into_interface()
                .expect("saved editor");
            for (node_id, import_index) in [(NodeId(7), 1), (NodeId(9), 0), (NodeId(11), 0)] {
                assert_eq!(
                    interface.input_from_connector(
                        &InputConnector::primary_input(node_id),
                        &[NodeId(24)]
                    ),
                    Some(&NodeInput::import(concrete!(f64), import_index))
                );
            }
            for archive in [&original, &copy] {
                let graph = archive.runtime_graph().await.expect("pinned numeric graph");
                assert_eq!(
                    evaluate_detached_numeric_network(graph.network)
                        .await
                        .expect("evaluate original scene output"),
                    TaggedValue::Number(9.)
                );
            }
            assert_eq!(copy.history_len(), original.history_len() + 4);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(undo.document.registry().value_equal(original_registry));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(redo.document.registry().value_equal(registry));

            let graph = copy
                .graph_inspection()
                .await
                .expect("current network scope");
            let selected = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same stable network");
            let moved_back = copy
                .move_nested_network_import_in_copy(selected, 1, 0, budget())
                .await
                .expect("move the same import up");
            assert!(
                moved_back
                    .document
                    .registry()
                    .value_equal(original_registry)
            );
        });
    }

    async fn network_export_fixture(wired_primary: bool) -> (PreservedGddV1, Arc<[u8]>) {
        use graphite_document_graph_storage::NodeMetadataSource;
        use graphite_math_nodes::absolute_value;
        use graphite_resource::ResourceRegistry;

        struct PortNames;

        impl NodeMetadataSource for PortNames {
            fn output_names(&self, path: &[NodeId], node: NodeId) -> Vec<String> {
                if path.is_empty() && node == NodeId(24) {
                    vec!["Removed port".into(), "Retained port".into()]
                } else {
                    Vec::new()
                }
            }
        }

        let owner_id = NodeId(24);
        let surviving = NodeInput::value(TaggedValue::Number(7.), true);
        let removed = NodeInput::value(TaggedValue::Number(-2.), true);
        let mut nested = NodeNetwork {
            exports: vec![removed.clone(), surviving],
            ..Default::default()
        };
        if wired_primary {
            let producer = NodeId(16);
            nested.nodes.insert(
                producer,
                DocumentNode {
                    implementation: DocumentNodeImplementation::ProtoNode(
                        absolute_value::IDENTIFIER,
                    ),
                    inputs: vec![removed],
                    ..Default::default()
                },
            );
            let primary = nested.exports.first_mut().expect("fixture primary export");
            *primary = NodeInput::node(producer, 0);
        }
        let mut root = NodeNetwork {
            exports: vec![NodeInput::node(owner_id, 0), NodeInput::node(owner_id, 1)],
            ..Default::default()
        };
        root.nodes.insert(
            owner_id,
            DocumentNode {
                implementation: DocumentNodeImplementation::Network(nested),
                ..Default::default()
            },
        );
        for (node_id, index) in [(NodeId(7), 0), (NodeId(9), 1)] {
            root.nodes.insert(
                node_id,
                DocumentNode {
                    implementation: DocumentNodeImplementation::ProtoNode(
                        absolute_value::IDENTIFIER,
                    ),
                    inputs: vec![NodeInput::node(owner_id, index)],
                    ..Default::default()
                },
            );
        }
        let mut owner = document();
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&root, &PortNames, &ResourceRegistry::new(), &resources)
            .expect("write source interface and parent bindings");
        let bytes: Arc<[u8]> = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP")
            .into();
        let original = open_gdd_zip(&bytes, budget())
            .await
            .expect("original graph");
        (original, bytes)
    }

    #[test]
    fn at_gra_graph_024_1_removes_an_export_and_repairs_parent_bindings() {
        futures::executor::block_on(async {
            let (original, bytes) = network_export_fixture(false).await;
            let owner_id = NodeId(24);
            let surviving = NodeInput::value(TaggedValue::Number(7.), true);
            let graph = original
                .graph_inspection()
                .await
                .expect("source projection");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the root and its nested network");
            };
            for (network, index) in [(root, 0), (nested, nested.export_count)] {
                assert!(matches!(
                    original
                        .remove_nested_network_export_in_copy(network, index, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphExportRemoval)
                ));
            }
            let copy = original
                .remove_nested_network_export_in_copy(nested, 0, budget())
                .await
                .expect("remove the addressed primary export");
            assert!(matches!(
                copy.remove_nested_network_export_in_copy(nested, 0, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let actual = copy
                .runtime_graph()
                .await
                .expect("saved graph")
                .into_interface()
                .expect("saved interface");
            let mut pinned = original
                .runtime_graph()
                .await
                .expect("pinned graph")
                .into_interface()
                .expect("pinned interface");
            pinned.remove_export(0, &[owner_id]);
            for connector in [
                InputConnector::primary_input(NodeId(7)),
                InputConnector::Export(0),
            ] {
                assert!(matches!(
                    pinned.input_from_connector(&connector, &[]),
                    Some(NodeInput::Value { exposed: true, .. })
                ));
                assert_eq!(
                    actual.input_from_connector(&connector, &[]),
                    pinned.input_from_connector(&connector, &[])
                );
            }
            for connector in [
                InputConnector::primary_input(NodeId(9)),
                InputConnector::Export(1),
            ] {
                assert_eq!(
                    actual.input_from_connector(&connector, &[]),
                    Some(&NodeInput::node(owner_id, 0))
                );
            }
            assert_eq!(
                actual
                    .nested_network(&[owner_id])
                    .expect("same network")
                    .exports,
                vec![surviving]
            );
            assert_eq!(
                actual
                    .node_metadata(&owner_id, &[])
                    .expect("same parent owner")
                    .persistent_metadata
                    .output_names,
                ["Retained port"]
            );
            assert_eq!(copy.history_len(), original.history_len() + 7);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(
                redo.document
                    .registry()
                    .value_equal(copy.document.registry())
            );
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_026_1_reorders_exports_with_names_and_parent_bindings() {
        futures::executor::block_on(async {
            let (original, bytes) = network_export_fixture(false).await;
            let graph = original
                .graph_inspection()
                .await
                .expect("source projection");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the source nested network");
            };
            for (network, from, to) in [(root, 0, 1), (nested, 0, 0), (nested, 0, 2)] {
                assert!(matches!(
                    original
                        .move_nested_network_export_in_copy(network, from, to, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphExportMove)
                ));
            }
            let copy = original
                .move_nested_network_export_in_copy(nested, 0, 1, budget())
                .await
                .expect("move the addressed export down");
            assert!(matches!(
                copy.move_nested_network_export_in_copy(nested, 1, 0, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let owner_id = NodeId(24);
            let actual = copy
                .runtime_graph()
                .await
                .expect("saved runtime")
                .into_interface()
                .expect("saved editor");
            let exports = actual
                .nested_network(&[owner_id])
                .expect("same owned network");
            assert_eq!(
                exports.exports,
                [
                    NodeInput::value(TaggedValue::Number(7.), true),
                    NodeInput::value(TaggedValue::Number(-2.), true),
                ]
            );
            for (index, node_id) in [(0, NodeId(7)), (1, NodeId(9))] {
                let expected = NodeInput::node(owner_id, 1 - index);
                for connector in [
                    InputConnector::Export(index),
                    InputConnector::primary_input(node_id),
                ] {
                    assert_eq!(
                        actual.input_from_connector(&connector, &[]),
                        Some(&expected)
                    );
                }
            }
            let graph = copy.graph_inspection().await.expect("saved projection");
            let selected = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same saved scope");
            assert_eq!(selected.export_name(0), Some("Retained port"));
            assert_eq!(selected.export_name(1), Some("Removed port"));
            assert_eq!(copy.history_len(), original.history_len() + 7);
            let original_registry = original.document.registry();
            let registry = copy.document.registry();
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(undo.document.registry().value_equal(original_registry));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(redo.document.registry().value_equal(registry));
            let moved_back = copy
                .move_nested_network_export_in_copy(selected, 1, 0, budget())
                .await
                .expect("move the same export up");
            assert!(
                moved_back
                    .document
                    .registry()
                    .value_equal(original_registry)
            );
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_027_1_renames_exact_ports_without_changing_bindings_or_types() {
        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let graph = original.graph_inspection().await.expect("source networks");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the root and its original nested network");
            };
            let port = ImportOrExport::Import(1);
            for (network, port) in [(root, port), (nested, ImportOrExport::Import(2))] {
                assert!(matches!(
                    original
                        .rename_nested_network_port_in_copy(
                            network,
                            port,
                            "Ignored".into(),
                            budget()
                        )
                        .await,
                    Err(GddArchiveError::UnsupportedGraphPortRename)
                ));
            }
            let name = "Amount Δ";
            let copy = original
                .rename_nested_network_port_in_copy(nested, port, name.into(), budget())
                .await
                .expect("rename the addressed import");
            assert!(matches!(
                copy.rename_nested_network_port_in_copy(nested, port, "Stale".into(), budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let expected = original.document.registry();
            let actual = copy.document.registry();
            assert_eq!(actual.networks, expected.networks);
            let [prior_first, prior_second] = expected.node_instances[&owner_id].inputs() else {
                panic!("expected both source slots");
            };
            let [first, second] = actual.node_instances[&owner_id].inputs() else {
                panic!("expected both renamed slots");
            };
            assert_eq!(first, prior_first);
            assert_eq!(second.input, prior_second.input);
            for slot in [second, prior_second] {
                assert_eq!(
                    slot.attributes
                        .get_typed::<String>("fixture::opaque-import")
                        .as_deref(),
                    Some("surviving input")
                );
            }
            let graph = copy.graph_inspection().await.expect("reopen imported name");
            let current = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same stable nested network");
            assert_eq!(current.import_name(0), None);
            assert_eq!(current.import_name(1), Some(name));
            assert!(matches!(
                copy.rename_nested_network_port_in_copy(current, port, name.into(), budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphPortName)
            ));
            let cleared = copy
                .rename_nested_network_port_in_copy(current, port, String::new(), budget())
                .await
                .expect("clear the addressed name with the pinned default");
            assert!(cleared.document.registry().value_equal(expected));
            assert_eq!(copy.history_len(), original.history_len() + 1);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one Undo restores the original label");
            assert!(undo.document.registry().value_equal(expected));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one Redo restores the renamed label");
            assert!(redo.document.registry().value_equal(actual));

            let (exports, bytes) = network_export_fixture(false).await;
            let graph = exports
                .graph_inspection()
                .await
                .expect("source export names");
            let [_, nested] = graph.networks.as_slice() else {
                panic!("expected the original nested outputs");
            };
            let copy = exports
                .rename_nested_network_port_in_copy(
                    nested,
                    ImportOrExport::Export(1),
                    "Computed output".into(),
                    budget(),
                )
                .await
                .expect("rename the addressed export");
            let graph = copy.graph_inspection().await.expect("reopen exported name");
            let current = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same stable nested network");
            assert_eq!(current.export_name(0), Some("Removed port"));
            assert_eq!(current.export_name(1), Some("Computed output"));
            assert_eq!(
                copy.document.registry().networks,
                exports.document.registry().networks
            );
            assert_eq!(copy.history_len(), exports.history_len() + 1);
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one Undo restores both export names");
            assert!(
                undo.document
                    .registry()
                    .value_equal(exports.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one Redo restores the renamed export");
            assert!(
                redo.document
                    .registry()
                    .value_equal(copy.document.registry())
            );
            assert_eq!(exports.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_029_1_toggles_primary_import_exposure_without_duplicating_its_value() {
        use graphite_graph_craft::concrete;

        futures::executor::block_on(async {
            let (original, owner_id) = network_import_fixture().await;
            let graph = original.graph_inspection().await.expect("source networks");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected the enclosing nested interface");
            };
            assert_eq!(root.primary_import_exposed(), None);
            assert_eq!(nested.primary_import_exposed(), Some(true));
            assert!(matches!(
                original
                    .set_primary_network_import_exposed_in_copy(root, true, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphPrimaryImportExposure)
            ));
            assert!(matches!(
                original
                    .set_primary_network_import_exposed_in_copy(nested, true, budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphInputExposure)
            ));
            let hidden = original
                .set_primary_network_import_exposed_in_copy(nested, false, budget())
                .await
                .expect("hide the nested primary port");
            assert!(matches!(
                hidden
                    .set_primary_network_import_exposed_in_copy(nested, true, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let hidden_graph = hidden
                .graph_inspection()
                .await
                .expect("hidden port in saved graph");
            let current = hidden_graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same nested network");
            assert_eq!(current.primary_import_exposed(), Some(false));
            assert_eq!(current.import_count, nested.import_count);
            assert_eq!(current.export_count, nested.export_count);
            let hidden_interface = hidden
                .runtime_graph()
                .await
                .expect("saved interface")
                .into_interface()
                .expect("pinned source interface");
            let holder = InputConnector::primary_input(NodeId(24));
            assert_eq!(
                hidden_interface.input_from_connector(&holder, &[]),
                Some(&NodeInput::value(TaggedValue::Number(-2.), false))
            );
            let disconnected = InputConnector::primary_input(NodeId(7));
            let saved_fallback =
                hidden_interface.input_from_connector(&disconnected, &[NodeId(24)]);
            let mut source = original
                .runtime_graph()
                .await
                .expect("original interface")
                .into_interface()
                .expect("pinned source interface");
            source.set_input(
                &holder,
                NodeInput::value(TaggedValue::Number(-2.), false),
                &[],
            );
            source.disconnect_input(&disconnected, &[NodeId(24)]);
            assert!(matches!(
                saved_fallback,
                Some(NodeInput::Value { exposed: true, .. })
            ));
            assert_eq!(
                saved_fallback,
                source.input_from_connector(&disconnected, &[NodeId(24)])
            );
            for node_id in [NodeId(9), NodeId(11)] {
                assert_eq!(
                    hidden_interface.input_from_connector(
                        &InputConnector::primary_input(node_id),
                        &[NodeId(24)]
                    ),
                    Some(&NodeInput::import(concrete!(f64), 1))
                );
            }
            let [first, second] = hidden.document.registry().node_instances[&owner_id].inputs()
            else {
                panic!("exactly two existing stored imports");
            };
            let [source_first, source_second] =
                original.document.registry().node_instances[&owner_id].inputs()
            else {
                panic!("exactly two original imports");
            };
            assert_eq!(first.attributes, source_first.attributes);
            assert_eq!(second, source_second);
            assert_eq!(
                hidden.document.registry().networks,
                original.document.registry().networks
            );
            assert_eq!(hidden.history_len(), original.history_len() + 2);
            let undo = hidden
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(
                redo.document
                    .registry()
                    .value_equal(hidden.document.registry())
            );

            let shown = hidden
                .set_primary_network_import_exposed_in_copy(current, true, budget())
                .await
                .expect("expose the existing value again");
            let interface = shown
                .runtime_graph()
                .await
                .expect("saved exposure")
                .into_interface()
                .expect("pinned interface");
            assert_eq!(
                interface.input_from_connector(&holder, &[]),
                Some(&NodeInput::value(TaggedValue::Number(-2.), true))
            );
            assert_eq!(
                interface.input_from_connector(&disconnected, &[NodeId(24)]),
                saved_fallback
            );
            assert_eq!(shown.history_len(), hidden.history_len() + 1);
        });
    }

    #[test]
    fn at_gra_graph_030_1_changes_primary_export_visibility_with_pinned_fallbacks() {
        futures::executor::block_on(async {
            let (original, bytes) = network_export_fixture(true).await;
            let graph = original.graph_inspection().await.expect("source networks");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected a nested source export");
            };
            let owner_id = NodeId(24);
            let path = [owner_id];
            let primary = InputConnector::Export(0);
            assert_eq!(nested.primary_export_exposed(), Some(true));
            assert!(matches!(
                original
                    .set_primary_network_export_exposed_in_copy(root, false, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphPrimaryExportExposure)
            ));
            assert!(matches!(
                original
                    .set_primary_network_export_exposed_in_copy(nested, true, budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphInputExposure)
            ));
            let hidden = original
                .set_primary_network_export_exposed_in_copy(nested, false, budget())
                .await
                .expect("hide the wired primary output");
            assert!(matches!(
                hidden
                    .set_primary_network_export_exposed_in_copy(nested, true, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let mut pinned = original
                .runtime_graph()
                .await
                .expect("saved source graph")
                .into_interface()
                .expect("pinned editor");
            pinned.disconnect_input(&primary, &path);
            let mut expected = pinned
                .input_from_connector(&primary, &path)
                .cloned()
                .expect("pinned disconnected primary");
            let NodeInput::Value { exposed, .. } = &mut expected else {
                panic!("expected the pinned literal fallback");
            };
            *exposed = false;
            pinned.set_input(&primary, expected.clone(), &path);
            let actual = hidden
                .runtime_graph()
                .await
                .expect("saved copy")
                .into_interface()
                .expect("pinned editor");
            assert_eq!(
                actual.input_from_connector(&primary, &path),
                Some(&expected)
            );
            for connector in [
                InputConnector::Export(0),
                InputConnector::primary_input(NodeId(7)),
            ] {
                pinned.disconnect_input(&connector, &[]);
                assert!(matches!(
                    actual.input_from_connector(&connector, &[]),
                    Some(NodeInput::Value { exposed: true, .. })
                ));
                assert_eq!(
                    actual.input_from_connector(&connector, &[]),
                    pinned.input_from_connector(&connector, &[])
                );
            }
            for connector in [
                InputConnector::Export(1),
                InputConnector::primary_input(NodeId(9)),
            ] {
                assert_eq!(
                    actual.input_from_connector(&connector, &[]),
                    Some(&NodeInput::node(owner_id, 1))
                );
            }
            assert_eq!(
                actual.input_from_connector(&InputConnector::Export(1), &path),
                Some(&NodeInput::value(TaggedValue::Number(7.), true))
            );
            let graph = hidden
                .graph_inspection()
                .await
                .expect("reopened hidden export");
            let current = graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same saved scope");
            assert_eq!(current.primary_export_exposed(), Some(false));
            assert_eq!(current.export_count, 2);
            assert_eq!(current.export_name(0), Some("Removed port"));
            assert_eq!(current.export_name(1), Some("Retained port"));
            assert_eq!(hidden.history_len(), original.history_len() + 3);
            let undo = hidden
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(
                redo.document
                    .registry()
                    .value_equal(hidden.document.registry())
            );
            let shown = hidden
                .set_primary_network_export_exposed_in_copy(current, true, budget())
                .await
                .expect("re-expose the retained fallback");
            let NodeInput::Value { exposed, .. } = &mut expected else {
                panic!("expected the existing saved literal");
            };
            *exposed = true;
            let interface = shown
                .runtime_graph()
                .await
                .expect("saved exposure")
                .into_interface()
                .expect("pinned editor");
            assert_eq!(
                interface.input_from_connector(&primary, &path),
                Some(&expected)
            );
            assert_eq!(shown.history_len(), hidden.history_len() + 1);
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_021_022_1_connects_only_the_selected_empty_export_in_one_history_step() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let original_graph = original
                .graph_inspection()
                .await
                .expect("source network list");
            let [_, original_network] = original_graph.networks.as_slice() else {
                panic!("expected a nested network");
            };
            let appended = original
                .append_nested_network_export_in_copy(original_network, budget())
                .await
                .expect("save an empty export");
            let graph = appended.graph_inspection().await.expect("addressed graph");
            let [_, selected_network] = graph.networks.as_slice() else {
                panic!("expected the saved nested network");
            };
            let [root_node, _, source, ..] = graph.nodes.as_slice() else {
                panic!("expected repeated local node IDs");
            };
            assert_eq!(root_node.local_id, source.local_id);
            assert_eq!(selected_network.empty_exports(), &[0]);
            assert!(!selected_network.can_connect_export_from(root_node));
            assert!(matches!(
                appended
                    .connect_numeric_network_export_in_copy(
                        selected_network,
                        root_node,
                        0,
                        budget()
                    )
                    .await,
                Err(GddArchiveError::UnsupportedGraphExportConnection)
            ));
            assert!(matches!(
                appended
                    .connect_numeric_network_export_in_copy(original_network, source, 0, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let wired = appended
                .connect_numeric_network_export_in_copy(selected_network, source, 0, budget())
                .await
                .expect("connect source primary output to the empty export");
            assert_eq!(wired.history_len(), appended.history_len() + 1);
            let registry = wired.document.registry();
            assert_eq!(
                registry.node_instances,
                appended.document.registry().node_instances
            );
            let stored = registry.networks[&selected_network.id]
                .exports
                .first()
                .expect("saved primary slot");
            assert!(matches!(&stored.target,
                Some(graphite_document_graph_storage::NodeInput::Node { id, index: 0 }) if *id == source.storage_id));
            let resolved = wired
                .graph_inspection()
                .await
                .expect("reopened connected export");
            let [_, resolved_network] = resolved.networks.as_slice() else {
                panic!("expected the connected nested network");
            };
            assert!(resolved_network.empty_exports().is_empty());
            let replacement = resolved
                .nodes
                .iter()
                .find(|node| node.storage_id == source.storage_id)
                .expect("same source node");
            assert!(matches!(
                wired
                    .connect_numeric_network_export_in_copy(
                        resolved_network,
                        replacement,
                        0,
                        budget()
                    )
                    .await,
                Err(GddArchiveError::UnsupportedGraphExportConnection)
            ));
            let undo = wired
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores the empty export");
            assert!(
                undo.document
                    .registry()
                    .value_equal(appended.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores that export wire");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_021_022_1_appends_pinned_export_defaults_to_the_addressed_network() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let inspection = original.graph_inspection().await.expect("saved networks");
            let [root, nested] = inspection.networks.as_slice() else {
                panic!("expected a root and a nested network");
            };
            assert_eq!(nested.export_count, 0);
            assert!(nested.can_extend_ports_in_copy());
            assert!(matches!(
                original
                    .append_nested_network_export_in_copy(root, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphExportAppend)
            ));
            let primary = original
                .append_nested_network_export_in_copy(nested, budget())
                .await
                .expect("first source export");
            assert!(matches!(
                primary
                    .append_nested_network_export_in_copy(nested, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            let primary_graph = primary.graph_inspection().await.expect("new snapshot");
            let selected = primary_graph
                .networks
                .iter()
                .find(|network| network.id == nested.id)
                .expect("same saved network");
            assert_eq!(selected.export_count, 1);
            let secondary = primary
                .append_nested_network_export_in_copy(selected, budget())
                .await
                .expect("appended secondary export");
            let expected = NodeInput::value(TaggedValue::None, true);
            let graph = secondary.runtime_graph().await.expect("saved copy");
            let actual = graph
                .network
                .nested_network(&nested.path)
                .expect("nested exports");
            assert_eq!(actual.exports, vec![expected.clone(), expected]);
            let registry = secondary.document.registry();
            assert!(
                registry.networks[&root.id]
                    .value_equal(&original.document.registry().networks[&root.id])
            );
            assert_eq!(
                registry.node_instances.len(),
                original.document.registry().node_instances.len()
            );
            let undo = secondary
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo removes the latest port");
            assert!(
                undo.document
                    .registry()
                    .value_equal(primary.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the saved secondary port");
            assert!(redo.document.registry().value_equal(registry));
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_016_017_1_graph_inspection_retains_stable_network_ownership() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let inspection = original.graph_inspection().await.expect("inspection");
            let [root, nested] = inspection.networks.as_slice() else {
                panic!("expected the root and one nested network");
            };
            assert_eq!(root.id, ROOT_NETWORK);
            assert_eq!(root.parent, None);
            assert_eq!(root.import_count, 0);
            assert_eq!(root.export_count, 1);
            assert_eq!(nested.parent, Some(root.id));
            assert_ne!(nested.id, root.id);
            let [root_value, owner, nested_value, ..] = inspection.nodes.as_slice() else {
                panic!("expected the network owner and repeated local IDs");
            };
            assert_eq!(root_value.local_id, nested_value.local_id);
            assert_eq!(owner.network_id(), root.id);
            assert_eq!(owner.nested_network(), Some(nested.id));
            assert_eq!(root_value.network_id(), root.id);
            assert_eq!(nested_value.network_id(), nested.id);
            assert_eq!(nested_value.nested_network(), None);
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_node_280_2_saves_the_exact_custom_node_and_can_undo_it() {
        use graphite_document_graph_storage::Position;

        futures::executor::block_on(async {
            let original = create_gdd_folder(820, 402, "Custom node".into(), budget())
                .await
                .expect("empty saved graph");
            let copy = original
                .insert_root_catalog_node_in_copy(CUSTOM_NODE_IDENTIFIER, [6, -9], budget())
                .await
                .expect("copy with exact source template");
            let graph = copy.runtime_graph().await.expect("saved graph metadata");
            let [node] = graph.nodes.as_slice() else {
                panic!("exactly one new node");
            };
            assert_eq!(node.position, Some(Position::Absolute([6, -9])));
            assert!(node.network_path.is_empty());
            let nested = graph
                .networks
                .iter()
                .find(|network| network.network_path == [node.local_id])
                .expect("owned nested network");
            assert_eq!(nested.reference.as_deref(), Some("Custom Node"));
            assert!(
                graph
                    .network
                    .nested_network(&nested.network_path)
                    .is_some_and(|network| network.nodes.is_empty() && network.exports.is_empty())
            );
            let undone = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo removes the node and its network");
            let undone_graph = undone.runtime_graph().await.expect("stored undo");
            assert!(undone_graph.nodes.is_empty());
            assert!(
                undone_graph
                    .networks
                    .iter()
                    .all(|network| network.network_path.is_empty())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the custom node");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(copy.document.registry())
            );
            assert!(
                original
                    .graph_nodes()
                    .await
                    .expect("original graph")
                    .is_empty()
            );
        });
    }

    #[test]
    fn at_gra_node_280_2_custom_node_first_boundary_edit_retires_its_reference() {
        futures::executor::block_on(async {
            let blank = create_gdd_folder(823, 407, "Custom ports".into(), budget())
                .await
                .expect("empty saved graph");
            let original = blank
                .insert_root_catalog_node_in_copy(CUSTOM_NODE_IDENTIFIER, [0, 0], budget())
                .await
                .expect("pinned Custom Node");
            let inspection = original.graph_inspection().await.expect("saved graph");
            let [_, custom] = inspection.networks.as_slice() else {
                panic!("owned custom network")
            };
            assert!(custom.can_append_ports_in_copy());
            assert!(!custom.can_extend_ports_in_copy());
            for add_import in [true, false] {
                let edited = if add_import {
                    original
                        .append_nested_network_import_in_copy(custom, budget())
                        .await
                } else {
                    original
                        .append_nested_network_export_in_copy(custom, budget())
                        .await
                }
                .expect("first boundary edit");
                let metadata = edited.runtime_graph().await.expect("stored graph metadata");
                let network = metadata
                    .networks
                    .iter()
                    .find(|network| network.network_id == custom.id)
                    .expect("same saved custom network");
                assert!(network.reference.is_none());
                let inspection = edited.graph_inspection().await.expect("edited graph");
                let retained = inspection
                    .networks
                    .iter()
                    .find(|network| network.id == custom.id)
                    .expect("same editable custom network");
                assert!(retained.can_extend_ports_in_copy());
                assert_eq!(
                    (retained.import_count, retained.export_count),
                    (usize::from(add_import), usize::from(!add_import))
                );
                let undone = edited
                    .move_history_in_copy(GddHistoryDirection::Undo, budget())
                    .await
                    .expect("restore initial reference and ports together");
                assert!(
                    undone
                        .document
                        .registry()
                        .value_equal(original.document.registry())
                );
                let redone = undone
                    .move_history_in_copy(GddHistoryDirection::Redo, budget())
                    .await
                    .expect("restore first boundary edit");
                assert!(
                    redone
                        .document
                        .registry()
                        .value_equal(edited.document.registry())
                );
            }
        });
    }

    #[test]
    fn at_gra_node_280_3_custom_number_list_import_drives_a_saved_export() {
        async fn inspect_custom(copy: &InspectedGddV1) -> (GddGraphNetwork, Option<GddGraphNode>) {
            let GddGraphInspection { nodes, networks } =
                copy.graph_inspection().await.expect("saved custom network");
            let Ok([_, custom]) = <[GddGraphNetwork; 2]>::try_from(networks) else {
                panic!("root and custom network")
            };
            let internal = nodes
                .into_iter()
                .find(|node| node.network_id() == custom.id);
            (custom, internal)
        }

        futures::executor::block_on(async {
            let blank = create_gdd_folder(827, 411, "Numeric Custom Node".into(), budget())
                .await
                .expect("empty saved graph");
            let original = blank
                .insert_root_catalog_node_in_copy(CUSTOM_NODE_IDENTIFIER, [0, 0], budget())
                .await
                .expect("pinned Custom Node");
            let (empty, _) = inspect_custom(&original).await;
            let imported = original
                .append_nested_numeric_import_in_copy(
                    &empty,
                    GddNumericImportKind::NumberList,
                    budget(),
                )
                .await
                .expect("first typed import");
            let (custom, _) = inspect_custom(&imported).await;
            assert_eq!(custom.id, empty.id);
            assert!(custom.can_start_numeric_import(0));
            assert_eq!(
                custom
                    .imports()
                    .first()
                    .and_then(|port| port.stored_type.as_ref()),
                Some(&TaggedValue::Numbers(Vec::new()).ty())
            );
            let exported = imported
                .append_nested_network_export_in_copy(&custom, budget())
                .await
                .expect("primary export");
            let (with_ports, _) = inspect_custom(&exported).await;
            let inserted = exported
                .insert_numeric_node_in_network_copy(
                    &with_ports,
                    UnaryNumericOperation::AbsoluteValue,
                    [3, 0],
                    budget(),
                )
                .await
                .expect("pinned internal numeric node");
            let (with_node, internal) = inspect_custom(&inserted).await;
            let internal = internal.expect("created numeric node");
            assert_eq!(internal.numeric_input_exposed(), Some(true));
            let wired = inserted
                .connect_numeric_network_import_in_copy(&with_node, 0, &internal, budget())
                .await
                .expect("typed import wire");
            let (wired_network, internal) = inspect_custom(&wired).await;
            let internal = internal.expect("wired numeric node");
            let connected = wired
                .connect_numeric_network_export_in_copy(&wired_network, &internal, 0, budget())
                .await
                .expect("typed internal export wire");
            let bytes: Arc<[u8]> = connected
                .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
                .expect("publish typed graph")
                .into();
            let reopened = open_gdd_zip(&bytes, budget())
                .await
                .expect("round-tripped typed graph");
            let (reopened_network, reopened_node) = inspect_custom(&reopened).await;
            let reopened_node = reopened_node.expect("round-tripped numeric node");
            assert_eq!(reopened_network.id, empty.id);
            assert_eq!(reopened_node.storage_id(), internal.storage_id());
            assert_eq!(
                reopened_node
                    .input_ports()
                    .first()
                    .and_then(|port| port.source.as_ref()),
                Some(&GddGraphPortSource::Import(0))
            );
            assert_eq!(
                reopened_network
                    .exports()
                    .first()
                    .and_then(|port| port.source.as_ref()),
                Some(&GddGraphPortSource::Node {
                    storage_id: internal.storage_id(),
                    output_index: 0
                })
            );
            assert!(
                matches!(reopened.preview_graph_numeric_output(&reopened_node).await,
                Ok(GddNumericValue::NumberList(values)) if values.is_empty())
            );
            let (unchanged, _) = inspect_custom(&original).await;
            assert_eq!((unchanged.import_count, unchanged.export_count), (0, 0));
        });
    }

    #[test]
    fn at_gra_node_281_2_saves_the_exact_merge_layer_and_its_owned_network() {
        futures::executor::block_on(assert_catalog_layer_roundtrip(
            MERGE_NODE_IDENTIFIER,
            merge_node_definition().expect("pinned Merge"),
            6,
        ));
    }

    #[test]
    fn at_gra_node_282_2_saves_the_exact_artboard_layer_and_its_owned_network() {
        futures::executor::block_on(assert_catalog_layer_roundtrip(
            ARTBOARD_NODE_IDENTIFIER,
            artboard_node_definition().expect("pinned Artboard"),
            5,
        ));
    }

    #[test]
    fn at_gra_node_282_3_validates_the_saved_artboard_preview_inputs() {
        let mut template = artboard_node_definition()
            .expect("pinned Artboard")
            .default_node_template();
        template.normalize_stored_types();
        let (mut board, _) = template.into_parts();
        let Some((_, SavedGraphicProfile::Layer(GddGraphLayerPreview::Artboard { origin, size }))) =
            checked_saved_graphic_node(&board)
        else {
            panic!("previewable pinned default Artboard")
        };
        assert_eq!((origin, size), (DVec2::ZERO, DVec2::new(1920., 1080.)));
        for (index, invalid) in [
            (1, NodeInput::node(NodeId(20), 1)),
            (
                2,
                NodeInput::value(TaggedValue::DVec2(DVec2::splat(f64::NAN)), false),
            ),
            (3, NodeInput::value(TaggedValue::Bool(false), false)),
            (4, NodeInput::node(NodeId(20), 0)),
            (5, NodeInput::node(NodeId(20), 0)),
        ] {
            let input = board.inputs.get_mut(index).expect("pinned input");
            let original = std::mem::replace(input, invalid);
            assert!(checked_saved_graphic_node(&board).is_none());
            *board.inputs.get_mut(index).expect("pinned input") = original;
        }
    }

    #[test]
    fn at_gra_node_282_2_edits_only_saved_artboard_geometry_literals() {
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
            let mut template = artboard_node_definition()
                .expect("pinned Artboard")
                .default_node_template();
            let original_geometry = GddArtboardGeometry {
                location: DVec2::new(-1.25, 3.25),
                dimensions: DVec2::new(100.5, 50.25),
            };
            for (index, value) in [
                (ARTBOARD_LOCATION_INPUT, original_geometry.location),
                (ARTBOARD_DIMENSIONS_INPUT, original_geometry.dimensions),
            ] {
                *template
                    .inputs
                    .get_mut(index)
                    .expect("original geometry slot") =
                    NodeInput::value(TaggedValue::DVec2(value), false);
            }
            let board_id = NodeId(15);
            let source = template.into_parts().0;
            let network = NodeNetwork {
                nodes: [(board_id, source.clone())].into_iter().collect(),
                exports: vec![NodeInput::node(board_id, 0)],
                ..Default::default()
            };
            let resources = HashMapResourceStorage::new();
            let mut owner = document();
            owner
                .commit_from_runtime(&network, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("save original signed and fractional coordinates");
            let original: Arc<[u8]> = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("source archive")
                .into();
            let archive = open_gdd_zip(&original, budget())
                .await
                .expect("reopen original board");
            let nodes = archive.graph_nodes().await.expect("saved graph");
            let node = nodes
                .iter()
                .find(|node| node.artboard_geometry().is_some())
                .expect("Artboard");
            assert_eq!(node.artboard_geometry(), Some(original_geometry));
            assert!(matches!(
                archive
                    .set_artboard_geometry_in_copy(node, original_geometry, budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphInput)
            ));
            assert!(matches!(
                archive
                    .set_artboard_geometry_in_copy(
                        node,
                        GddArtboardGeometry {
                            location: DVec2::splat(f64::MAX),
                            dimensions: DVec2::splat(f64::MAX),
                        },
                        budget()
                    )
                    .await,
                Err(GddArchiveError::InvalidArtboardGeometry)
            ));
            let requested =
                GddArtboardGeometry::parse("-1.25, -8.5", "100.5, 50.25").expect("finite geometry");
            let staged = archive
                .set_artboard_geometry_in_copy(node, requested, budget())
                .await
                .expect("edit and round only the changed component");
            assert_eq!(staged.history_len(), archive.history_len() + 1);
            let current = staged.runtime_graph().await.expect("saved revision");
            let mut expected = source;
            *expected
                .inputs
                .get_mut(ARTBOARD_LOCATION_INPUT)
                .expect("typed Location") =
                NodeInput::value(TaggedValue::DVec2(DVec2::new(-1.25, -9.)), false);
            assert_eq!(current.network.nodes.get(&board_id), Some(&expected));
            assert_eq!(current.network.exports, network.exports);
            assert!(matches!(
                staged
                    .set_artboard_geometry_in_copy(node, requested, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
        });
    }

    #[test]
    fn at_gra_node_282_2_edits_only_the_saved_artboard_background_literal() {
        use graphite_graph_craft::document::value::RenderOutputType;
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
            let mut template = artboard_node_definition()
                .expect("pinned Artboard")
                .default_node_template();
            let original = Color::from_rgbaf32_unchecked(0.12345679, 0., 1., 1.);
            *template
                .inputs
                .get_mut(ARTBOARD_BACKGROUND_INPUT)
                .expect("typed Background") = NodeInput::value(TaggedValue::Color(original), false);
            let board_id = NodeId(15);
            let source = template.into_parts().0;
            let network = NodeNetwork {
                nodes: [(board_id, source.clone())].into_iter().collect(),
                exports: vec![NodeInput::node(board_id, 0)],
                ..Default::default()
            };
            let resources = HashMapResourceStorage::new();
            let mut owner = document();
            owner
                .commit_from_runtime(&network, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("save an Artboard with a linear Background color");
            let original_zip: Arc<[u8]> = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("source archive")
                .into();
            let archive = open_gdd_zip(&original_zip, budget()).await.expect("reopen");
            let nodes = archive.graph_nodes().await.expect("saved graph");
            let node = nodes
                .iter()
                .find(|node| node.artboard_background().is_some())
                .expect("Artboard with a Background");
            assert_eq!(node.artboard_background(), Some(original));
            let [red, green, blue, alpha] = original.to_gamma_srgb_channels();
            assert!(matches!(
                archive
                    .set_artboard_background_in_copy(node, [red, green, blue, alpha], budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphInput)
            ));
            for invalid in [-0.1, 1.1, f32::NAN] {
                assert!(matches!(
                    archive
                        .set_artboard_background_in_copy(
                            node,
                            [red, invalid, blue, alpha],
                            budget()
                        )
                        .await,
                    Err(GddArchiveError::InvalidArtboardBackground)
                ));
            }
            let channels = [red, 0.25, blue, 0.5];
            let staged = archive
                .set_artboard_background_in_copy(node, channels, budget())
                .await
                .expect("decode edited Green and Alpha with the pinned transfer function");
            assert_eq!(staged.history_len(), archive.history_len() + 1);
            let current = staged.runtime_graph().await.expect("staged revision");
            let green = Color::from_gamma_srgb_channels(0., 0.25, 0., 1.).g();
            let color = Color::from_rgbaf32_unchecked(original.r(), green, original.b(), 0.5);
            let mut expected = source;
            *expected
                .inputs
                .get_mut(ARTBOARD_BACKGROUND_INPUT)
                .expect("typed Background") = NodeInput::value(TaggedValue::Color(color), false);
            assert_eq!(current.network.nodes.get(&board_id), Some(&expected));
            assert_eq!(current.network.exports, network.exports);
            assert!(matches!(
                staged
                    .set_artboard_background_in_copy(node, channels, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));

            let edited_zip: Arc<[u8]> = staged
                .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
                .expect("export edited archive")
                .into();
            let reopened = open_gdd_zip(&edited_zip, budget())
                .await
                .expect("reopen edit");
            let edited_nodes = reopened.graph_nodes().await.expect("reopened graph");
            let edited = edited_nodes
                .iter()
                .find(|edited| edited.storage_id() == node.storage_id())
                .expect("same Artboard address");
            assert_eq!(edited.artboard_background(), Some(color));
            let output = reopened
                .export_document_artboard_svg(
                    edited,
                    CpuSvgEvaluationBudget {
                        max_viewport_pixels: 1920 * 1080,
                        max_output_bytes: 64 * 1024,
                    },
                )
                .await
                .expect("pinned SVG background");
            let RenderOutputType::Svg { svg, .. } = output.data else {
                panic!("source SVG output")
            };
            assert!(svg.contains(r#"fill-opacity="0.5""#));
        });
    }

    #[test]
    fn at_gra_node_282_3_saved_artboard_svg_uses_the_source_document_bounds() {
        use graphite_graph_craft::document::value::RenderOutputType;
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
            let mut template = artboard_node_definition()
                .expect("pinned Artboard")
                .default_node_template();
            *template.inputs.get_mut(2).expect("Artboard location") =
                NodeInput::value(TaggedValue::DVec2(DVec2::new(72., -12.)), false);
            *template.inputs.get_mut(3).expect("Artboard dimensions") =
                NodeInput::value(TaggedValue::DVec2(DVec2::new(-64., 32.)), false);
            let board_id = NodeId(15);
            let network = NodeNetwork {
                nodes: [(board_id, template.into_parts().0)].into_iter().collect(),
                exports: vec![NodeInput::node(board_id, 0)],
                ..Default::default()
            };
            let resources = HashMapResourceStorage::new();
            let mut owner = document();
            owner
                .commit_from_runtime(&network, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("save the pinned Artboard");
            let original: Arc<[u8]> = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("source archive")
                .into();
            let archive = open_gdd_zip(&original, budget())
                .await
                .expect("reopen Artboard archive");
            let nodes = archive.graph_nodes().await.expect("saved graph");
            let node = nodes
                .iter()
                .find(|node| node.layer_preview().is_some())
                .expect("document Artboard");
            let limits = CpuSvgEvaluationBudget {
                max_viewport_pixels: 64 * 32,
                max_output_bytes: 64 * 1024,
            };
            let output = archive
                .export_document_artboard_svg(node, limits)
                .await
                .expect("export SVG");
            let RenderOutputType::Svg { svg, image_data } = output.data else {
                panic!("source SVG output")
            };
            assert!(image_data.is_empty());
            assert!(svg.contains(r#"viewBox="0 0 64 32" width="64" height="32""#));
            assert!(svg.contains("<clipPath"));
            let [target] = output
                .metadata
                .click_targets
                .get(&board_id)
                .expect("Artboard hit geometry")
                .as_slice()
            else {
                panic!("single source Artboard hit shape")
            };
            assert_eq!(
                target.bounding_box(),
                Some([DVec2::ZERO, DVec2::new(64., 32.)])
            );
            assert_eq!(
                output.metadata.local_transforms.get(&board_id),
                Some(&DAffine2::from_translation(DVec2::new(8., -12.)))
            );
            let viewport = UVec2::new(160, 80);
            let canvas = archive
                .render_document_artboard_canvas_svg(
                    viewport,
                    2.,
                    None,
                    CpuSvgEvaluationBudget {
                        max_viewport_pixels: viewport.x * viewport.y,
                        ..limits
                    },
                )
                .await
                .expect("render the document Artboard into a scaled pane");
            let RenderOutputType::Svg { svg, image_data } = canvas.render.data else {
                panic!("source Canvas output")
            };
            assert!(image_data.is_empty());
            assert!(svg.contains(r#"viewBox="0 0 80 40" width="80" height="40""#));
            assert!(svg.contains("<clipPath"));
            assert_eq!(
                canvas.render.metadata.local_transforms.get(&board_id),
                output.metadata.local_transforms.get(&board_id)
            );
            assert!(matches!(
                archive
                    .render_document_artboard_canvas_svg(viewport, 2., None, limits)
                    .await,
                Err(GddArchiveError::GraphLayerPreviewEvaluation(_))
            ));
            assert!(matches!(
                archive
                    .export_document_artboard_svg(
                        node,
                        CpuSvgEvaluationBudget {
                            max_viewport_pixels: 64 * 32 - 1,
                            ..limits
                        }
                    )
                    .await,
                Err(GddArchiveError::GraphLayerPreviewEvaluation(_))
            ));
            let other = open_gdd_zip(&original, budget())
                .await
                .expect("independent inspection");
            assert!(matches!(
                other.export_document_artboard_svg(node, limits).await,
                Err(GddArchiveError::StaleGraphNode)
            ));
        });
    }

    #[test]
    fn at_gra_node_282_2_sets_only_an_empty_document_artboard_output() {
        futures::executor::block_on(async {
            let initial = create_gdd_folder(621, 992, "Document output".into(), budget())
                .await
                .expect("source folder");
            assert!(matches!(
                initial
                    .render_document_artboard_canvas_svg(
                        UVec2::new(80, 40),
                        1.,
                        None,
                        CpuSvgEvaluationBudget {
                            max_viewport_pixels: 80 * 40,
                            max_output_bytes: 64 * 1024,
                        },
                    )
                    .await,
                Err(GddArchiveError::UnsupportedGraphLayerPreview)
            ));
            let staged = initial
                .insert_root_catalog_node_in_copy(ARTBOARD_NODE_IDENTIFIER, [3, -2], budget())
                .await
                .expect("place Artboard");
            let staged = staged
                .insert_root_catalog_node_in_copy(ARTBOARD_NODE_IDENTIFIER, [9, 5], budget())
                .await
                .expect("place a second Artboard");
            let nodes = staged.graph_nodes().await.expect("working graph");
            let eligible = nodes
                .iter()
                .filter(|node| node.can_set_document_artboard())
                .collect::<Vec<_>>();
            let [first, second] = eligible.as_slice() else {
                panic!("two eligible Artboards")
            };
            let bound = staged
                .set_document_artboard_in_copy(first, budget())
                .await
                .expect("fill the empty root output");
            assert_eq!(bound.history_len(), staged.history_len() + 1);
            assert_eq!(
                bound.document.registry().node_instances,
                staged.document.registry().node_instances
            );
            assert!(matches!(
                bound.set_document_artboard_in_copy(second, budget()).await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let current = bound.graph_inspection().await.expect("connected graph");
            let alternative = current
                .nodes
                .iter()
                .find(|node| node.storage_id == second.storage_id)
                .expect("second Artboard");
            assert!(!alternative.can_set_document_artboard());
            assert!(matches!(
                bound
                    .set_document_artboard_in_copy(alternative, budget())
                    .await,
                Err(GddArchiveError::UnsupportedDocumentArtboard)
            ));
            let root = current
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("document root");
            let geometry = bound.graph_geometry(root).await.expect("root canvas");
            let layers = geometry.root_layers.expect("saved layer hierarchy");
            let [layer] = layers.as_slice() else {
                panic!("single document Artboard")
            };
            assert_eq!(
                (layer.storage_id, layer.parent, layer.depth),
                (first.storage_id, None, 0)
            );
            let undone = bound
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo wire");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(staged.document.registry())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("redo wire");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(bound.document.registry())
            );
        });
    }

    #[test]
    fn at_r10_1_picks_only_the_current_visible_unlocked_artboard() {
        futures::executor::block_on(async {
            let stage = create_gdd_folder(621, 187, "Canvas".into(), budget())
                .await
                .expect("native document")
                .insert_root_catalog_node_in_copy(ARTBOARD_NODE_IDENTIFIER, [2, 0], budget())
                .await
                .expect("catalog Artboard");
            let nodes = stage.graph_nodes().await.expect("initial graph");
            let root = nodes
                .iter()
                .find(|node| node.can_set_document_artboard())
                .expect("admitted Artboard");
            let output = stage
                .set_document_artboard_in_copy(root, budget())
                .await
                .expect("document Artboard output");
            let nodes = output.graph_nodes().await.expect("Artboard output graph");
            let root = nodes
                .iter()
                .find(|node| node.storage_id() == root.storage_id())
                .expect("same stored Artboard");
            let document = output
                .set_artboard_geometry_in_copy(
                    root,
                    GddArtboardGeometry {
                        location: DVec2::new(72., -12.),
                        dimensions: DVec2::new(-64., 32.),
                    },
                    budget(),
                )
                .await
                .expect("translated and signed Artboard");
            let nodes = document.graph_nodes().await.expect("current graph");
            let board = nodes
                .iter()
                .find(|node| node.storage_id() == root.storage_id())
                .expect("current Artboard");
            let viewport = UVec2::new(160, 80);
            let limits = CpuSvgEvaluationBudget {
                max_viewport_pixels: viewport.x * viewport.y,
                max_output_bytes: 64 * 1024,
            };
            let mut camera = PTZ::default();
            camera.pan = DVec2::new(-20., -2.);
            camera.set_zoom(0.75);
            for scale in [1., 1.5, 2.] {
                let canvas = document
                    .render_document_artboard_canvas_svg(viewport, scale, Some(camera), limits)
                    .await
                    .expect("original rendered frame");
                let pick = canvas.pick_target.expect("Artboard render metadata");
                let pane = NavigationMessageHandler::default()
                    .calculate_offset_transform(viewport.as_dvec2() / (2. * scale), &camera);
                let overlay = canvas.overlay.expect("Artboard overlay geometry");
                assert_eq!(overlay.name, "Artboard");
                let expected_outline = [
                    DVec2::new(8., -12.),
                    DVec2::new(72., -12.),
                    DVec2::new(72., 20.),
                    DVec2::new(8., 20.),
                ]
                .map(|corner| pane.transform_point2(corner));
                for (actual, expected) in overlay.outline.into_iter().zip(expected_outline) {
                    assert!((actual - expected).length() < 1e-9);
                }
                let expected_name_anchor = expected_outline[0] - DVec2::Y * 4.;
                assert!(
                    (overlay.name_transform.translation - expected_name_anchor).length() < 1e-9
                );
                assert!((overlay.name_transform.matrix2.x_axis - DVec2::X).length() < 1e-9);
                assert!(pick.contains(pane.transform_point2(DVec2::new(24., -4.))));
                assert!(!pick.contains(pane.transform_point2(DVec2::new(78., -4.))));
                assert!(!pick.contains(DVec2::NAN));
                assert!(pick.matches_node(board));
                assert!(!pick.matches_node(root));
            }
            let hidden = document
                .edit_root_layer_in_copy(board, GddRootLayerChange::Visibility(false), budget())
                .await
                .expect("hide the root Artboard");
            assert!(matches!(
                hidden
                    .render_document_artboard_canvas_svg(viewport, 2., Some(camera), limits)
                    .await,
                Err(GddArchiveError::UnsupportedGraphLayerPreview)
            ));
            let locked = document
                .edit_root_layer_in_copy(board, GddRootLayerChange::Locked(true), budget())
                .await
                .expect("lock the root Artboard");
            let canvas = locked
                .render_document_artboard_canvas_svg(viewport, 2., Some(camera), limits)
                .await
                .expect("locked Artboard remains visible");
            assert!(canvas.pick_target.is_none());
            assert!(canvas.overlay.is_some());
        });
    }

    #[test]
    fn at_gra_node_282_2_attaches_only_admitted_graphic_content() {
        futures::executor::block_on(async {
            let initial = create_gdd_folder(621, 992, "Artboard Content".into(), budget())
                .await
                .expect("source folder");
            let artboard = initial
                .insert_root_catalog_node_in_copy(ARTBOARD_NODE_IDENTIFIER, [5, 0], budget())
                .await
                .expect("place Artboard");
            let working = artboard
                .insert_root_catalog_node_in_copy(MERGE_NODE_IDENTIFIER, [-3, 1], budget())
                .await
                .expect("place Merge");
            let nodes = working.graph_nodes().await.expect("working graph");
            let board = nodes
                .iter()
                .find(|node| node.can_set_document_artboard())
                .expect("catalog Artboard");
            let working = working
                .set_document_artboard_in_copy(board, budget())
                .await
                .expect("Artboard document output");
            let nodes = working.graph_nodes().await.expect("current root");
            let board = nodes
                .iter()
                .find(|node| node.storage_id == board.storage_id)
                .expect("current Artboard");
            let merge = nodes
                .iter()
                .find(|node| {
                    matches!(
                        node.graphic_profile,
                        Some(SavedGraphicProfile::Layer(GddGraphLayerPreview::Merge))
                    )
                })
                .expect("current Merge");
            assert!(matches!(
                working
                    .connect_graphic_content_in_copy(board, board, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphContentConnection)
            ));
            assert_eq!(
                board.graphic_content_input_from(merge),
                Some(LAYER_CONTENT_INPUT)
            );
            let attached = working
                .connect_graphic_content_in_copy(board, merge, budget())
                .await
                .expect("attach Merge Content");
            assert_eq!(attached.history_len(), working.history_len() + 1);
            let graph = attached.graph_inspection().await.expect("attached graph");
            let target = graph
                .nodes
                .iter()
                .find(|node| node.storage_id == board.storage_id)
                .expect("same Artboard address");
            let source = graph
                .nodes
                .iter()
                .find(|node| node.storage_id == merge.storage_id)
                .expect("same Merge address");
            assert!(matches!(
                target.input_ports.get(LAYER_CONTENT_INPUT).and_then(|port| port.source.as_ref()),
                Some(GddGraphPortSource::Node { storage_id, output_index: 0 }) if *storage_id == merge.storage_id
            ));
            assert!(matches!(
                attached
                    .connect_graphic_content_in_copy(target, source, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphContentConnection)
            ));
            let root = graph
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("document root");
            let geometry = attached
                .graph_geometry(root)
                .await
                .expect("saved layer graph");
            let layers = geometry.root_layers.expect("saved layer hierarchy");
            assert_eq!(
                layers
                    .iter()
                    .map(|layer| (layer.storage_id, layer.parent))
                    .collect::<Vec<_>>(),
                [
                    (board.storage_id, None),
                    (merge.storage_id, Some(board.storage_id)),
                ]
            );
        });
    }

    #[test]
    fn at_gra_node_262_2_admits_only_the_pinned_finite_ellipse_template() {
        let mut ellipse_node = resolve_proto_node_type(ellipse::IDENTIFIER)
            .expect("pinned Ellipse definition")
            .default_node_template()
            .into_parts()
            .0;
        assert!(checked_saved_graphic_node(&ellipse_node).is_some());
        for value in [-40., 0., 75.] {
            *ellipse_node
                .inputs
                .get_mut(ellipse::RadiusXInput::INDEX)
                .expect("horizontal radius") = NodeInput::value(TaggedValue::Number(value), false);
            assert!(matches!(
                checked_saved_graphic_node(&ellipse_node),
                Some((actual, SavedGraphicProfile::Ellipse([radius_x, radius_y])))
                    if actual == ellipse_node && radius_x == value && radius_y == 25.
            ));
        }
        for value in [f64::NAN, f64::INFINITY, f64::MAX] {
            *ellipse_node
                .inputs
                .get_mut(ellipse::RadiusYInput::INDEX)
                .expect("vertical radius") = NodeInput::value(TaggedValue::Number(value), false);
            assert!(checked_saved_graphic_node(&ellipse_node).is_none());
        }
        *ellipse_node
            .inputs
            .get_mut(ellipse::RadiusYInput::INDEX)
            .expect("vertical radius") = NodeInput::value(TaggedValue::Number(25.), true);
        assert!(checked_saved_graphic_node(&ellipse_node).is_none());

        *ellipse_node
            .inputs
            .get_mut(ellipse::RadiusXInput::INDEX)
            .expect("horizontal radius") =
            NodeInput::value(TaggedValue::Numbers(vec![32.5, 0.]), false);
        for (value, exposed) in [
            (TaggedValue::Number(25.), false),
            (TaggedValue::Numbers(vec![25.]), true),
            (TaggedValue::Numbers(vec![f64::MAX]), false),
            (
                TaggedValue::Numbers(vec![25.; MAX_PREVIEW_VALUES + 1]),
                false,
            ),
        ] {
            *ellipse_node
                .inputs
                .get_mut(ellipse::RadiusYInput::INDEX)
                .expect("vertical radius") = NodeInput::value(value, exposed);
            assert!(checked_saved_graphic_node(&ellipse_node).is_none());
        }
    }

    #[test]
    fn at_gra_node_262_2_edits_only_the_selected_saved_radius() {
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
            let mut template = resolve_proto_node_type(ellipse::IDENTIFIER)
                .expect("pinned Ellipse definition")
                .default_node_template();
            template.normalize_stored_types();
            let mut expected = template.into_parts().0;
            let ellipse_id = NodeId(15);
            let network = NodeNetwork {
                nodes: [(ellipse_id, expected.clone())].into_iter().collect(),
                ..Default::default()
            };
            let resources = HashMapResourceStorage::new();
            let mut owner = document();
            owner
                .commit_from_runtime(&network, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("save a pinned Ellipse");
            let original: Arc<[u8]> = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("original saved archive")
                .into();
            let archive = open_gdd_zip(&original, budget())
                .await
                .expect("open Ellipse");
            let nodes = archive.graph_nodes().await.expect("saved Graphite graph");
            let node = nodes.first().expect("saved Ellipse");
            let original_files = archive.folder_files().expect("original files");
            assert_eq!(node.ellipse_radii(), Some([50., 25.]));
            assert!(matches!(
                archive
                    .set_ellipse_radius_in_copy(node, GddEllipseAxis::X, 50., budget())
                    .await,
                Err(GddArchiveError::UnchangedEllipseRadius)
            ));
            for value in [f64::NAN, f64::INFINITY, f64::MAX] {
                assert!(matches!(
                    archive
                        .set_ellipse_radius_in_copy(node, GddEllipseAxis::X, value, budget())
                        .await,
                    Err(GddArchiveError::InvalidEllipseRadius)
                ));
            }
            let changed = archive
                .set_ellipse_radius_in_copy(node, GddEllipseAxis::X, -36.125, budget())
                .await
                .expect("edit the signed fractional radius");
            assert_eq!(changed.history_len(), archive.history_len() + 1);
            assert!(matches!(
                changed
                    .set_ellipse_radius_in_copy(node, GddEllipseAxis::Y, 0., budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let nodes = changed.graph_nodes().await.expect("changed graph");
            let current = nodes.first().expect("same Ellipse");
            assert_eq!(current.storage_id(), node.storage_id());
            assert_eq!(current.ellipse_radii(), Some([-36.125, 25.]));
            let changed = changed
                .set_ellipse_radius_in_copy(current, GddEllipseAxis::Y, 0., budget())
                .await
                .expect("edit the degenerate radius");
            for (index, value) in [
                (ellipse::RadiusXInput::INDEX, -36.125),
                (ellipse::RadiusYInput::INDEX, 0.),
            ] {
                *expected.inputs.get_mut(index).expect("pinned radius slot") =
                    NodeInput::value(TaggedValue::Number(value), false);
            }
            let result = changed.runtime_graph().await.expect("read final graph");
            assert_eq!(result.network.nodes.get(&ellipse_id), Some(&expected));
            assert_eq!(result.network.exports, network.exports);
            assert_eq!(changed.history_len(), archive.history_len() + 2);
            assert_eq!(
                archive.folder_files().expect("source files"),
                original_files
            );
        });
    }

    #[test]
    fn at_gra_node_262_3_reopens_and_renders_literal_and_wired_ellipse_radii() {
        use graphite_graph_craft::document::value::RenderOutputType;
        use graphite_math_nodes::absolute_value;
        use graphite_resource::ResourceRegistry;

        let shape_id = NodeId(15);
        let paint_id = NodeId(20);
        let merge_id = NodeId(25);
        let board_id = NodeId(30);
        let horizontal_source = NodeId(35);
        let vertical_source = NodeId(40);
        let mut shape = resolve_proto_node_type(ellipse::IDENTIFIER)
            .expect("pinned Ellipse definition")
            .default_node_template();
        let mut paint = resolve_proto_node_type(fill::IDENTIFIER)
            .expect("pinned Fill definition")
            .default_node_template();
        paint.normalize_stored_types();
        let loose_paint = paint.clone().into_parts().0;
        let mut merge = merge_node_definition()
            .expect("pinned Merge")
            .default_node_template();
        let mut board = artboard_node_definition()
            .expect("pinned Artboard")
            .default_node_template();
        for (node, index, source) in [
            (&mut paint, fill::ContentInput::INDEX, shape_id),
            (&mut merge, LAYER_CONTENT_INPUT, paint_id),
            (&mut board, LAYER_CONTENT_INPUT, merge_id),
        ] {
            *node
                .inputs
                .get_mut(index)
                .expect("pinned Content connector") = NodeInput::node(source, 0);
            node.normalize_stored_types();
        }
        *board
            .inputs
            .get_mut(ARTBOARD_DIMENSIONS_INPUT)
            .expect("pinned Dimensions") =
            NodeInput::value(TaggedValue::DVec2(DVec2::splat(128.)), false);
        shape.normalize_stored_types();
        let paint = paint.into_parts().0;
        let merge = merge.into_parts().0;
        let board = board.into_parts().0;
        let output_budget = CpuSvgEvaluationBudget {
            max_viewport_pixels: 128 * 128,
            max_output_bytes: 64 * 1024,
        };
        let configuration = RenderConfig {
            viewport: Footprint {
                resolution: UVec2::splat(128),
                ..Footprint::DEFAULT
            },
            scale: 1.,
            export_format: SvgExportFormat::Svg,
            for_export: true,
            ..Default::default()
        };
        let mut numeric = resolve_proto_node_type(absolute_value::IDENTIFIER)
            .expect("pinned Absolute Value definition")
            .default_node_template();
        numeric.normalize_stored_types();
        let numeric = numeric.into_parts().0;

        futures::executor::block_on(async {
            let list = || TaggedValue::Numbers(vec![32.5, 0., -12.75]);
            for (horizontal, vertical, expected_paths, wired) in [
                (list(), TaggedValue::Numbers(vec![17.25, 9.5]), 3, false),
                (list(), TaggedValue::Numbers(Vec::new()), 0, false),
                (list(), TaggedValue::Numbers(vec![17.25, 9.5]), 3, true),
                (list(), TaggedValue::Numbers(Vec::new()), 0, true),
                (
                    TaggedValue::Number(-32.5),
                    TaggedValue::Number(17.25),
                    1,
                    true,
                ),
            ] {
                let mut shape = shape.clone();
                let mut source = NodeNetwork {
                    nodes: [
                        (paint_id, paint.clone()),
                        (merge_id, merge.clone()),
                        (board_id, board.clone()),
                    ]
                    .into_iter()
                    .collect(),
                    exports: vec![NodeInput::node(board_id, 0)],
                    ..Default::default()
                };
                for (index, source_id, value) in [
                    (ellipse::RadiusXInput::INDEX, horizontal_source, horizontal),
                    (ellipse::RadiusYInput::INDEX, vertical_source, vertical),
                ] {
                    let literal = NodeInput::value(value, false);
                    let radius = if wired {
                        let mut node = numeric.clone();
                        *node
                            .inputs
                            .get_mut(absolute_value::ValueInput::INDEX)
                            .expect("pinned Value slot") = literal;
                        source.nodes.insert(source_id, node);
                        NodeInput::node(source_id, 0)
                    } else {
                        literal
                    };
                    *shape.inputs.get_mut(index).expect("pinned radius") = radius;
                }
                source.nodes.insert(shape_id, shape.into_parts().0);
                let expected = evaluate_cpu_svg(&source, configuration, output_budget, None)
                    .await
                    .expect("pinned unmodified source graph renders");
                let mut unconnected = source.clone();
                unconnected.nodes.insert(paint_id, loose_paint.clone());
                let resources = HashMapResourceStorage::new();
                let mut owner = document();
                owner
                    .commit_from_runtime(
                        &unconnected,
                        &NoMetadata,
                        &ResourceRegistry::new(),
                        &resources,
                    )
                    .expect("save the original list inputs with an unwired Fill");
                let zip: Arc<[u8]> = owner
                    .export_to_bytes(
                        ExportFormat::Zip,
                        ExportOptions::default(),
                        &resources,
                        None,
                    )
                    .await
                    .expect("source ZIP")
                    .into();
                let loose = open_gdd_zip(&zip, budget())
                    .await
                    .expect("open exact saved list graph");
                let nodes = loose.graph_nodes().await.expect("original graph addresses");
                let find = |id| {
                    nodes
                        .iter()
                        .find(|node| node.local_id == id && node.network_path.is_empty())
                        .expect("original root node")
                };
                let connected = loose
                    .connect_graphic_content_in_copy(find(paint_id), find(shape_id), budget())
                    .await
                    .expect("connect the list output to the original Fill Content port");
                assert_eq!(connected.history_len(), loose.history_len() + 1);
                let zip: Arc<[u8]> = connected
                    .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
                    .expect("save connected list graph")
                    .into();
                let saved = open_gdd_zip(&zip, budget())
                    .await
                    .expect("reopen connected list graph");
                let graph = saved.runtime_graph().await.expect("saved runtime graph");
                assert_eq!(graph.network.nodes, source.nodes);
                assert_eq!(graph.network.exports, source.exports);
                let graph_nodes = saved.graph_nodes().await.expect("saved graph addresses");
                let board = graph_nodes
                    .iter()
                    .find(|node| node.local_id == board_id && node.network_path.is_empty())
                    .expect("saved Artboard");
                let shape = graph_nodes
                    .iter()
                    .find(|node| node.local_id == shape_id && node.network_path.is_empty())
                    .expect("saved Ellipse list");
                assert!(matches!(
                    (shape.graphic_profile, wired),
                    (Some(SavedGraphicProfile::EllipseList), false)
                        | (Some(SavedGraphicProfile::EllipseWired), true)
                ));
                assert!(shape.ellipse_radii().is_none());
                let rendered = saved
                    .export_document_artboard_svg(board, output_budget)
                    .await
                    .expect("render saved list through Artboard export");
                let (
                    RenderOutputType::Svg { svg, image_data },
                    RenderOutputType::Svg {
                        svg: expected_svg,
                        image_data: expected_images,
                    },
                ) = (rendered.data, expected.data)
                else {
                    panic!("source CPU SVG outputs");
                };
                let normalize_clip_id = |svg: &str| {
                    let (_, definition) = svg
                        .split_once("<clipPath id=\"")
                        .expect("Artboard clip definition");
                    let (id, _) = definition.split_once('"').expect("generated clip ID");
                    assert!(svg.contains(&format!("clip-path=\"url(#{id})\"")));
                    // Each source render assigns its own clip ID.
                    svg.replace(id, "artboard-test-clip")
                };
                assert_eq!(normalize_clip_id(&svg), normalize_clip_id(&expected_svg));
                assert!(image_data.is_empty() && expected_images.is_empty());
                assert_eq!(
                    svg.matches("fill=\"#000000\"").count(),
                    expected_paths,
                    "{svg}"
                );
                let canvas = saved
                    .render_document_artboard_canvas_svg(UVec2::splat(128), 1., None, output_budget)
                    .await
                    .expect("render the saved list through the Canvas entry point");
                let RenderOutputType::Svg { svg, image_data } = canvas.render.data else {
                    panic!("Artboard Canvas SVG");
                };
                assert_eq!(
                    svg.matches("fill=\"#000000\"").count(),
                    expected_paths,
                    "{svg}"
                );
                assert!(image_data.is_empty());
                if wired && expected_paths == 1 {
                    for invalid_input in [
                        NodeInput::value(TaggedValue::Number(f64::MAX), false),
                        NodeInput::value(TaggedValue::DVec2(DVec2::ONE), false),
                        NodeInput::node(horizontal_source, 0),
                        NodeInput::node(paint_id, 0),
                    ] {
                        let mut invalid = graph.network.clone();
                        let numeric = invalid
                            .nodes
                            .get_mut(&horizontal_source)
                            .expect("stored Radius X source");
                        *numeric
                            .inputs
                            .get_mut(absolute_value::ValueInput::INDEX)
                            .expect("numeric Value slot") = invalid_input;
                        assert!(matches!(
                            InspectedGddV1::render_saved_layer_svg(&invalid, board_id, configuration, output_budget).await,
                            Err(GddArchiveError::InvalidEllipseRadiusSource { node, .. }) if node == horizontal_source
                        ));
                    }
                }
            }
        });
    }

    #[test]
    fn at_gra_node_266_272_2_connects_the_pinned_rectangle_to_fill_content() {
        use graphite_glam::UVec2;
        use graphite_graph_craft::document::value::RenderOutputType;

        futures::executor::block_on(async {
            let blank = create_gdd_folder(621, 992, "Painted rectangle".into(), budget())
                .await
                .expect("blank saved graph");
            let graph = blank
                .insert_root_catalog_node_in_copy(
                    "PROTONODE:vector_nodes::generator_nodes::RectangleNode",
                    [0, 0],
                    budget(),
                )
                .await
                .expect("pinned Rectangle catalog template");
            let graph = graph
                .insert_root_catalog_node_in_copy(
                    "PROTONODE:graphene_core::vector::FillNode",
                    [12, 0],
                    budget(),
                )
                .await
                .expect("pinned Fill catalog template");
            let nodes = graph.graph_nodes().await.expect("current graph");
            let fill = nodes
                .iter()
                .find(|node| matches!(node.graphic_profile, Some(SavedGraphicProfile::Fill)))
                .expect("pinned loose Fill");
            let rectangle = nodes
                .iter()
                .find(|node| matches!(node.graphic_profile, Some(SavedGraphicProfile::Rectangle)))
                .expect("pinned Rectangle");
            assert!(matches!(
                graph
                    .connect_graphic_content_in_copy(fill, fill, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphContentConnection)
            ));
            assert_eq!(
                fill.graphic_content_input_from(rectangle),
                Some(fill::ContentInput::INDEX)
            );
            let copy = graph
                .connect_graphic_content_in_copy(fill, rectangle, budget())
                .await
                .expect("connect original Vector output");
            assert_eq!(copy.history_len(), graph.history_len() + 1);
            let bytes: Arc<[u8]> = copy
                .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
                .expect("export connected GDD")
                .into();
            let saved = open_gdd_zip(&bytes, budget())
                .await
                .expect("reopen the exact connector");
            let nodes = saved.graph_nodes().await.expect("saved graph addresses");
            let current_fill = nodes
                .iter()
                .find(|node| node.storage_id == fill.storage_id)
                .expect("same Fill");
            let current_rectangle = nodes
                .iter()
                .find(|node| node.storage_id == rectangle.storage_id)
                .expect("same Rectangle");
            assert!(
                current_fill
                    .graphic_content_input_from(current_rectangle)
                    .is_none()
            );
            let saved_runtime = saved.runtime_graph().await.expect("reopened runtime");
            assert!(saved_runtime.network.exports.is_empty());
            let mut detached = NodeNetwork {
                exports: vec![NodeInput::node(current_fill.local_id, 0)],
                ..Default::default()
            };
            collect_layer_preview_node(
                &saved_runtime.network,
                current_fill.local_id,
                &mut detached,
                &mut BTreeSet::new(),
            )
            .expect("rebuild the admitted source branch");
            let output = evaluate_cpu_svg(
                &detached,
                RenderConfig {
                    viewport: Footprint {
                        transform: DAffine2::from_translation(DVec2::splat(64.)),
                        resolution: UVec2::splat(128),
                        ..Footprint::DEFAULT
                    },
                    scale: 1.,
                    export_format: SvgExportFormat::Svg,
                    for_export: true,
                    ..Default::default()
                },
                CpuSvgEvaluationBudget {
                    max_viewport_pixels: 128 * 128,
                    max_output_bytes: 64 * 1024,
                },
                None,
            )
            .await
            .expect("evaluate the original source Fill operator");
            let RenderOutputType::Svg { svg, .. } = output.data else {
                panic!("source rendered SVG");
            };
            assert!(svg.contains("fill=\"#000000\""), "{svg}");
            let undo = saved
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo one wire");
            assert!(
                undo.document
                    .registry()
                    .value_equal(graph.document.registry())
            );
        });
    }

    #[test]
    fn at_gra_node_281_2_journals_the_pinned_paint_chain_with_its_merge_wire() {
        futures::executor::block_on(async {
            for (merge_position, chained) in [([0, 0], true), ([0, 5], false)] {
                let blank = create_gdd_folder(621, 992, "Painted layer".into(), budget())
                    .await
                    .expect("blank saved graph");
                let graph = blank
                    .insert_root_catalog_node_in_copy(
                        "PROTONODE:vector_nodes::generator_nodes::RectangleNode",
                        [-14, 0],
                        budget(),
                    )
                    .await
                    .expect("pinned Rectangle");
                let graph = graph
                    .insert_root_catalog_node_in_copy(
                        "PROTONODE:graphene_core::vector::FillNode",
                        [-7, 0],
                        budget(),
                    )
                    .await
                    .expect("pinned Fill");
                let graph = graph
                    .insert_root_catalog_node_in_copy(
                        MERGE_NODE_IDENTIFIER,
                        merge_position,
                        budget(),
                    )
                    .await
                    .expect("pinned Merge");
                let nodes = graph.graph_nodes().await.expect("saved addresses");
                let profiles: [fn(Option<SavedGraphicProfile>) -> bool; 3] = [
                    |profile| matches!(profile, Some(SavedGraphicProfile::Rectangle)),
                    |profile| matches!(profile, Some(SavedGraphicProfile::Fill)),
                    |profile| {
                        matches!(
                            profile,
                            Some(SavedGraphicProfile::Layer(GddGraphLayerPreview::Merge))
                        )
                    },
                ];
                let [rectangle, fill, merge] = profiles.map(|matches| {
                    nodes
                        .iter()
                        .find(|node| matches(node.graphic_profile))
                        .expect("source catalog node")
                });
                assert_eq!(merge.graphic_content_input_from(fill), None);
                let ready = graph
                    .connect_graphic_content_in_copy(fill, rectangle, budget())
                    .await
                    .expect("typed Rectangle-to-Fill wire");
                let nodes = ready.graph_nodes().await.expect("current candidates");
                let [fill, merge] = [fill, merge].map(|previous| {
                    nodes
                        .iter()
                        .find(|node| node.storage_id == previous.storage_id)
                        .expect("same storage address")
                });
                assert_eq!(
                    merge.graphic_content_input_from(fill),
                    Some(LAYER_CONTENT_INPUT)
                );
                let linked = ready
                    .connect_graphic_content_in_copy(merge, fill, budget())
                    .await
                    .expect("journal the source wire and requested chain");
                assert_eq!(
                    linked.history_len(),
                    ready.history_len() + 1 + if chained { 2 } else { 0 }
                );
                let actual = linked
                    .runtime_graph()
                    .await
                    .expect("saved runtime")
                    .into_interface()
                    .expect("reopened layout");
                assert_eq!(actual.is_chain(&rectangle.local_id, &[]), chained);
                assert_eq!(actual.is_chain(&fill.local_id, &[]), chained);
                assert_eq!(
                    actual.position(&merge.local_id, &[]),
                    Some(IVec2::from_array(merge_position))
                );
                assert_eq!(
                    actual.input_from_connector(
                        &InputConnector::node_at_index(merge.local_id, LAYER_CONTENT_INPUT),
                        &[]
                    ),
                    Some(&NodeInput::node(fill.local_id, 0))
                );
                let undone = linked
                    .move_history_in_copy(GddHistoryDirection::Undo, budget())
                    .await
                    .expect("undo wire and chain together");
                assert!(
                    undone
                        .document
                        .registry()
                        .value_equal(ready.document.registry())
                );
                let redone = undone
                    .move_history_in_copy(GddHistoryDirection::Redo, budget())
                    .await
                    .expect("redo wire and chain together");
                assert!(
                    redone
                        .document
                        .registry()
                        .value_equal(linked.document.registry())
                );
            }
        });
    }

    async fn assert_catalog_layer_roundtrip(
        identifier: &str,
        definition: &DocumentNodeDefinition,
        expected_children: usize,
    ) {
        let source = create_gdd_folder(621, 992, definition.identifier.into(), budget())
            .await
            .expect("source folder");
        let original = source
            .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
            .expect("original archive");
        let position = [5, -8];
        let copy = source
            .insert_root_catalog_node_in_copy(identifier, position, budget())
            .await
            .expect("place the pinned layer");
        let graph = copy.graph_inspection().await.expect("saved layer graph");
        let [root, child] = graph.networks.as_slice() else {
            panic!("original root and one owned layer network");
        };
        let node = graph
            .nodes
            .iter()
            .find(|node| node.nested_network() == Some(child.id))
            .expect("root layer owner");
        assert_eq!(node.network_id(), root.id);
        assert_eq!(
            (child.import_count, child.export_count),
            (definition.node_template.inputs.len(), 1)
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| node.network_id() == child.id)
                .count(),
            expected_children
        );
        assert!(!child.can_extend_ports_in_copy());
        assert!(matches!(
            copy.append_nested_network_import_in_copy(child, budget())
                .await,
            Err(GddArchiveError::UnsupportedGraphImportAppend)
        ));
        assert!(matches!(
            copy.copy_graph_nodes_to_clipboard(std::slice::from_ref(node), budget())
                .await,
            Err(GddArchiveError::UnsupportedGraphClipboardSelection)
        ));
        let mut expected = definition.default_node_template();
        expected.normalize_stored_types();
        expected.node_type_metadata =
            NodeTypePersistentMetadata::layer(IVec2::from_array(position));
        let interface = copy
            .runtime_graph()
            .await
            .expect("saved runtime graph")
            .into_interface()
            .expect("saved layer metadata");
        assert_eq!(
            interface.create_node_template(&node.local_id, &[]),
            Some(expected)
        );

        let bytes: Arc<[u8]> = copy
            .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
            .expect("export saved layer")
            .into();
        let reopened = open_gdd_zip(&bytes, budget())
            .await
            .expect("reopen saved layer");
        let reopened_graph = reopened.graph_inspection().await.expect("reopened graph");
        let addresses = |graph: &GddGraphInspection| {
            graph
                .nodes
                .iter()
                .map(|node| (node.storage_id(), node.network_id()))
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(addresses(&reopened_graph), addresses(&graph));
        let undone = reopened
            .move_history_in_copy(GddHistoryDirection::Undo, budget())
            .await
            .expect("remove owner and child in one undo");
        assert!(
            undone
                .graph_nodes()
                .await
                .expect("undo insertion")
                .is_empty()
        );
        let restored = undone
            .move_history_in_copy(GddHistoryDirection::Redo, budget())
            .await
            .expect("restore exact layer group");
        assert_eq!(
            addresses(&restored.graph_inspection().await.expect("restored graph")),
            addresses(&graph)
        );
        assert!(source.graph_nodes().await.expect("source graph").is_empty());
        assert_eq!(
            source
                .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
                .expect("untouched archive"),
            original
        );
    }

    #[test]
    fn at_gra_node_281_3_pinned_merge_keeps_order_provenance_and_hit_geometry_after_reopen() {
        use graphite_core_types::{
            Color, Context,
            list::{Item, List, NodeIdPath},
        };
        use graphite_glam::UVec2;
        use graphite_graph_craft::{
            application_io::{PlatformApplicationIo, PlatformEditorApi},
            document::value::RenderOutputType,
            graphene_compiler::{Compiler, Executor},
        };
        use graphite_interpreted_executor::{
            dynamic_executor::DynamicExecutor, util::wrap_network_in_scope,
        };
        use graphite_resource::ResourceRegistry;
        use graphite_std::{
            ATTR_EDITOR_LAYER_PATH, Graphic,
            application_io::{ExportFormat as RenderFormat, RenderConfig},
            bounds::RenderBoundingBox,
            memo::IORecord,
            renderer::{
                Render, RenderParams, RenderSvgSegmentList, SvgRender, graphic_list_bounding_box,
            },
            transform::Footprint,
            vector_nodes::{fill, generator_nodes::rectangle},
        };

        let first_layer = NodeId(41);
        let second_layer = NodeId(53);
        let draw_shape = |rectangle_id, fill_id, dimensions: [f64; 2], color| {
            let mut shape = resolve_proto_node_type(rectangle::IDENTIFIER)
                .expect("source Rectangle")
                .default_node_template();
            let [width, height] = dimensions;
            for (index, dimension) in [
                (rectangle::WidthInput::INDEX, width),
                (rectangle::HeightInput::INDEX, height),
            ] {
                *shape.inputs.get_mut(index).expect("source dimension") =
                    NodeInput::value(TaggedValue::Number(dimension), false);
            }
            let mut paint = resolve_proto_node_type(fill::IDENTIFIER)
                .expect("source Fill")
                .default_node_template();
            *paint
                .inputs
                .get_mut(fill::ContentInput::INDEX)
                .expect("source paint content") = NodeInput::node(rectangle_id, 0);
            *paint
                .inputs
                .get_mut(fill::PaintInput::INDEX)
                .expect("source paint") = NodeInput::value(TaggedValue::Color(color), false);
            [
                (rectangle_id, shape.into_parts().0),
                (fill_id, paint.into_parts().0),
            ]
        };
        let merge = |base, content| {
            let mut node = merge_node_definition()
                .expect("source Merge")
                .default_node_template();
            let [base_port, content_port] = node.inputs.as_mut_slice() else {
                panic!("original Base and Content ports")
            };
            if let Some(base) = base {
                *base_port = NodeInput::node(base, 0);
            }
            *content_port = NodeInput::node(content, 0);
            node.into_parts().0
        };
        let source = NodeNetwork {
            nodes: draw_shape(NodeId(11), NodeId(12), [60., 18.], Color::RED)
                .into_iter()
                .chain(draw_shape(NodeId(21), NodeId(22), [14., 44.], Color::BLUE))
                .chain([
                    (first_layer, merge(None, NodeId(12))),
                    (second_layer, merge(Some(first_layer), NodeId(22))),
                ])
                .collect(),
            exports: vec![NodeInput::node(second_layer, 0)],
            ..Default::default()
        };
        let configuration = RenderConfig {
            viewport: Footprint {
                transform: DAffine2::from_translation(DVec2::splat(64.)),
                resolution: UVec2::splat(128),
                ..Footprint::DEFAULT
            },
            scale: 1.,
            export_format: RenderFormat::Svg,
            for_export: true,
            ..Default::default()
        };
        let render = |network: NodeNetwork| async move {
            let api = Arc::new(PlatformEditorApi {
                application_io: Some(Arc::new(PlatformApplicationIo::default())),
                ..Default::default()
            });
            let mut network = wrap_network_in_scope(network, api);
            graphite_preprocessor::Preprocessor::new()
                .preprocess(&mut network, &|_| None)
                .expect("exact source preprocessing");
            let proto = Compiler {}
                .compile_single(network)
                .expect("compile exact Merge network");
            let executor = DynamicExecutor::new(proto)
                .await
                .expect("construct the registered Merge pipeline");
            let output = Executor::execute(&&executor, configuration.into_context())
                .await
                .expect("CPU SVG Merge output");
            let TaggedValue::RenderOutput(output) = output else {
                panic!("original SVG output")
            };
            let RenderOutputType::Svg { svg, image_data } = output.data else {
                panic!("CPU-only SVG")
            };
            assert!(image_data.is_empty());
            let svg = svg.to_ascii_lowercase();
            let red = svg.find("#ff0000").expect("bottom layer paint");
            let blue = svg.find("#0000ff").expect("top layer paint");
            assert!(red < blue, "source stack follows Base then Content");
            assert_eq!(
                output
                    .metadata
                    .click_targets
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from([first_layer, second_layer])
            );
            let mut thumbnails = Vec::new();
            for (layer, size, visible_color, other_color) in [
                (first_layer, DVec2::new(60., 18.), "#ff0000", "#0000ff"),
                (second_layer, DVec2::new(14., 44.), "#0000ff", "#ff0000"),
            ] {
                let targets = output
                    .metadata
                    .click_targets
                    .get(&layer)
                    .expect("separate layer click targets");
                let [target] = targets.as_slice() else {
                    panic!("one target for each layer")
                };
                assert_eq!(target.bounding_box(), Some([-size / 2., size / 2.]));
                let monitored = executor
                    .introspect(&[layer, NodeId(4)])
                    .expect("unique source Monitor");
                let content = if let Some(record) =
                    monitored.downcast_ref::<IORecord<Context<'static>, List<Graphic>>>()
                {
                    record.output.clone()
                } else {
                    let record = monitored
                        .downcast_ref::<IORecord<Context<'static>, Item<Graphic>>>()
                        .expect("source Graphic Monitor output");
                    List::new_from_item(record.output.clone())
                };
                assert_eq!(content.len(), 1);
                let path = content
                    .attribute::<NodeIdPath>(ATTR_EDITOR_LAYER_PATH, 0)
                    .expect("original provenance attribute");
                assert!(path.0.iter_element_values().copied().eq([layer]));
                let RenderBoundingBox::Rectangle(bounds) =
                    graphic_list_bounding_box(&content, DAffine2::IDENTITY)
                else {
                    panic!("source thumbnail bounds");
                };
                assert_eq!(bounds, [-size / 2., size / 2.]);
                let [minimum, maximum] = bounds;
                let mut thumbnail = SvgRender::new();
                content.render_svg(
                    &mut thumbnail,
                    &RenderParams {
                        thumbnail: true,
                        ..Default::default()
                    },
                );
                thumbnail.format_svg(minimum, maximum);
                let thumbnail = thumbnail.svg.to_svg_string().to_ascii_lowercase();
                assert!(thumbnail.contains(visible_color));
                assert!(!thumbnail.contains(other_color));
                thumbnails.push(thumbnail);
            }
            (svg, thumbnails)
        };

        futures::executor::block_on(async {
            let expected = render(source.clone()).await;
            let mut owner = document();
            let resources = HashMapResourceStorage::new();
            owner
                .commit_from_runtime(&source, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("persist the exact source network");
            let original: Arc<[u8]> = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("source ZIP")
                .into();
            let budget = GddArchiveBudget {
                max_entries: 128,
                ..budget()
            };
            let saved = open_gdd_zip(&original, budget)
                .await
                .expect("reopened overlapping layers");
            let graph = saved.runtime_graph().await.expect("saved graph");
            assert_eq!(render(graph.network.clone()).await, expected);

            let nodes = saved.graph_nodes().await.expect("saved layer addresses");
            let root = nodes
                .iter()
                .find(|node| node.local_id == second_layer && node.network_path.is_empty())
                .expect("actual root export");
            let render_budget = CpuSvgEvaluationBudget {
                max_viewport_pixels: 128 * 128,
                max_output_bytes: 64 * 1024,
            };
            let output = saved
                .render_document_layer_svg(root, configuration, render_budget)
                .await
                .expect("render the saved Merge without external resources");
            let RenderOutputType::Svg { svg, image_data } = output.data else {
                panic!("bounded source SVG")
            };
            assert!(image_data.is_empty());
            assert_eq!(svg.to_ascii_lowercase(), expected.0);
            assert_eq!(
                output
                    .metadata
                    .click_targets
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from([first_layer, second_layer])
            );
            let wrong_root = nodes
                .iter()
                .find(|node| node.local_id == first_layer && node.network_path.is_empty())
                .expect("base layer");
            assert!(matches!(
                saved
                    .render_document_layer_svg(wrong_root, configuration, render_budget)
                    .await,
                Err(GddArchiveError::UnsupportedGraphLayerPreview)
            ));
            let foreign = open_gdd_zip(&original, budget)
                .await
                .expect("independent snapshot");
            assert!(matches!(
                foreign
                    .render_document_layer_svg(root, configuration, render_budget)
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let insufficient = CpuSvgEvaluationBudget {
                max_output_bytes: 1,
                ..render_budget
            };
            assert!(matches!(
                saved
                    .render_document_layer_svg(root, configuration, insufficient)
                    .await,
                Err(GddArchiveError::GraphLayerPreviewEvaluation(_))
            ));
            let insufficient = CpuSvgEvaluationBudget {
                max_viewport_pixels: 1,
                ..render_budget
            };
            assert!(matches!(
                saved
                    .render_document_layer_svg(root, configuration, insufficient)
                    .await,
                Err(GddArchiveError::GraphLayerPreviewEvaluation(_))
            ));

            let mut cyclic = graph.network.clone();
            let base = cyclic.nodes.get_mut(&first_layer).expect("base Merge");
            *base.inputs.first_mut().expect("Base input") = NodeInput::node(second_layer, 0);
            assert!(matches!(
                collect_layer_preview_node(
                    &cyclic,
                    second_layer,
                    &mut NodeNetwork::default(),
                    &mut BTreeSet::new()
                ),
                Err(GddArchiveError::UnsupportedGraphLayerPreview)
            ));
            let mut deep = NodeNetwork::default();
            let mut template = merge_node_definition()
                .expect("pinned layer")
                .default_node_template();
            template.normalize_stored_types();
            for index in 1..=33 {
                let mut layer = template.clone().into_parts().0;
                if index > 1 {
                    *layer.inputs.first_mut().expect("Base input") =
                        NodeInput::node(NodeId(index - 1), 0);
                }
                deep.nodes.insert(NodeId(index), layer);
            }
            let mut bounded = NodeNetwork::default();
            collect_layer_preview_node(&deep, NodeId(32), &mut bounded, &mut BTreeSet::new())
                .expect("maximum bounded preview");
            assert_eq!(bounded.nodes.len(), 32);
            assert!(matches!(
                collect_layer_preview_node(
                    &deep,
                    NodeId(33),
                    &mut NodeNetwork::default(),
                    &mut BTreeSet::new()
                ),
                Err(GddArchiveError::GraphLayerPreviewBudget)
            ));
            let mut altered = graph.network;
            let shape = altered
                .nodes
                .get_mut(&NodeId(11))
                .expect("reachable rectangle");
            *shape
                .inputs
                .get_mut(rectangle::WidthInput::INDEX)
                .expect("width port") = NodeInput::scope("unresolved-scope");
            assert!(matches!(
                collect_layer_preview_node(
                    &altered,
                    second_layer,
                    &mut NodeNetwork::default(),
                    &mut BTreeSet::new()
                ),
                Err(GddArchiveError::UnsupportedGraphLayerPreview)
            ));
        });
    }

    #[test]
    fn at_gra_graph_005_1_round_trips_eligible_catalog_proto_defaults() {
        use crate::registry::GraphiteNodeCatalog;

        futures::executor::block_on(async {
            let catalog = GraphiteNodeCatalog::load().expect("pinned catalog");
            let blank = create_gdd_folder(922, 271, "Pinned catalog".into(), budget())
                .await
                .expect("source folder");
            let position = [7, -8];
            let mut count = 0;
            let mut failures = Vec::new();
            for node in catalog.search("").iter().flat_map(|group| &group.nodes) {
                if can_insert_graphite_catalog_node(&node.identifier) {
                    count += 1;
                    if let Err(error) = blank
                        .insert_root_catalog_node_in_copy(&node.identifier, position, budget())
                        .await
                    {
                        failures.push(format!("{}: {error}", node.identifier));
                    }
                }
            }
            assert!(count > 5);
            assert!(failures.is_empty(), "{}", failures.join("\n"));
            for invalid in [
                "NETWORK:Blend",
                "PROTONODE:unknown-node",
                "Add",
                "Merge",
                "Artboard",
            ] {
                assert!(matches!(
                    blank
                        .insert_root_catalog_node_in_copy(invalid, position, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedCatalogNodeDefinition)
                ));
            }
            assert!(
                blank
                    .graph_nodes()
                    .await
                    .expect("original graph")
                    .is_empty()
            );
        });
    }

    #[test]
    fn at_gra_node_079_080_083_2_evaluates_pinned_arithmetic_in_a_reopened_graph() {
        use graphite_core_types::{Context, Node, list::Item};
        use graphite_math_nodes::{LogarithmNode, exponent, logarithm};
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
            let number = TaggedValue::Number;
            let integer = TaggedValue::Integer;
            let numbers = |values: &[f64]| TaggedValue::Numbers(values.to_vec());
            let vector = |x, y| TaggedValue::DVec2(DVec2::new(x, y));
            let positive = TaggedValue::Bool(true);
            let signed = TaggedValue::Bool(false);
            let natural_base = std::f64::consts::E;
            let natural_result = LogarithmNode::new(
                crate::Input(Item::new_from_element(12.3_f64)),
                crate::Input(Item::new_from_element(natural_base)),
            )
            .eval(Context::default())
            .await
            .into_element();
            let cases = [
                (
                    exponent::IDENTIFIER,
                    vec![number(-2.), number(3.)],
                    GddNumericValue::Number(-8.),
                ),
                (
                    exponent::IDENTIFIER,
                    vec![integer(3_037_000_499), integer(2)],
                    GddNumericValue::Integer(9_223_372_030_926_249_001),
                ),
                (
                    exponent::IDENTIFIER,
                    vec![integer(2), integer(63)],
                    GddNumericValue::Integer(0),
                ),
                (
                    exponent::IDENTIFIER,
                    vec![vector(2., 3.), vector(3., 2.)],
                    GddNumericValue::Vector([8., 9.]),
                ),
                (
                    exponent::IDENTIFIER,
                    vec![numbers(&[-2., 3.]), number(3.)],
                    GddNumericValue::NumberList(Arc::from([-8., 27.])),
                ),
                (
                    logarithm::IDENTIFIER,
                    vec![number(12.3), number(natural_base)],
                    GddNumericValue::Number(natural_result),
                ),
                (
                    logarithm::IDENTIFIER,
                    vec![number(1000.), number(10.)],
                    GddNumericValue::Number(3.),
                ),
                (
                    logarithm::IDENTIFIER,
                    vec![number(16.), vector(2., 4.)],
                    GddNumericValue::Vector([4., 2.]),
                ),
                (
                    logarithm::IDENTIFIER,
                    vec![numbers(&[4., 16.]), number(2.)],
                    GddNumericValue::NumberList(Arc::from([2., 4.])),
                ),
                (
                    modulo::IDENTIFIER,
                    vec![number(-11.), number(5.), signed],
                    GddNumericValue::Number(-1.),
                ),
                (
                    modulo::IDENTIFIER,
                    vec![number(-11.), number(5.), positive.clone()],
                    GddNumericValue::Number(4.),
                ),
                (
                    modulo::IDENTIFIER,
                    vec![
                        integer(-9_007_199_254_740_993),
                        integer(5),
                        positive.clone(),
                    ],
                    GddNumericValue::Integer(2),
                ),
                (
                    modulo::IDENTIFIER,
                    vec![vector(-11., 12.), vector(5., 7.), positive.clone()],
                    GddNumericValue::Vector([4., 5.]),
                ),
                (
                    modulo::IDENTIFIER,
                    vec![
                        numbers(&[-11., 12., -13.]),
                        numbers(&[5., 7.]),
                        positive.clone(),
                    ],
                    GddNumericValue::NumberList(Arc::from([4., 5., 1.])),
                ),
                (
                    modulo::IDENTIFIER,
                    vec![numbers(&[]), number(2.), positive.clone()],
                    GddNumericValue::NumberList(Arc::from([])),
                ),
            ];
            let mut network = NodeNetwork::default();
            let mut insert = |identifier, values: Vec<TaggedValue>| {
                let id = NodeId(network.nodes.len() as u64 + 1);
                network.nodes.insert(
                    id,
                    DocumentNode {
                        implementation: DocumentNodeImplementation::ProtoNode(identifier),
                        inputs: values
                            .into_iter()
                            .enumerate()
                            .map(|(index, value)| NodeInput::value(value, index == 0))
                            .collect(),
                        ..Default::default()
                    },
                );
                id
            };
            let expected = cases
                .into_iter()
                .map(|(identifier, values, output)| (insert(identifier, values), output))
                .collect::<BTreeMap<_, _>>();
            let invalid_domain = [
                insert(exponent::IDENTIFIER, vec![number(-2.), number(0.5)]),
                insert(logarithm::IDENTIFIER, vec![number(0.), number(2.)]),
                insert(
                    modulo::IDENTIFIER,
                    vec![number(5.), number(0.), positive.clone()],
                ),
                insert(
                    modulo::IDENTIFIER,
                    vec![positive.clone(), number(5.), positive.clone()],
                ),
            ];
            let wrong_flag = insert(modulo::IDENTIFIER, vec![number(5.), number(2.), number(1.)]);
            let export = NodeId(1);
            network.exports.push(NodeInput::node(export, 0));
            let resources = HashMapResourceStorage::new();
            let mut owner = document();
            owner
                .commit_from_runtime(&network, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("save actual pinned arithmetic nodes");
            let bytes: Arc<[u8]> = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("source ZIP")
                .into();
            let budget = GddArchiveBudget {
                max_entries: 128,
                ..budget()
            };
            let saved = open_gdd_zip(&bytes, budget)
                .await
                .expect("reopen actual arithmetic nodes");
            let nodes = saved.graph_nodes().await.expect("reopened nodes");
            assert_eq!(nodes.len(), expected.len() + invalid_domain.len() + 1);
            let original_bytes = saved
                .export_archive(ArchiveFormat::Zip, budget.max_archive_bytes)
                .expect("saved snapshot");
            for node in &nodes {
                let result = saved.preview_graph_numeric_output(node).await;
                if invalid_domain.contains(&node.local_id) {
                    assert!(
                        matches!(result, Err(GddArchiveError::UnsupportedGraphNumericPreview)),
                        "{:?}: {result:?}",
                        node.local_id
                    );
                } else if node.local_id == wrong_flag {
                    assert!(
                        matches!(
                            result,
                            Err(GddArchiveError::GraphNumericPreviewEvaluation(_))
                        ),
                        "{result:?}"
                    );
                } else {
                    let value = result.expect("typed arithmetic output");
                    assert!(
                        value.same_value(expected.get(&node.local_id).expect("source output")),
                        "{:?}: {value:?}",
                        node.local_id
                    );
                }
            }
            let node = nodes
                .iter()
                .find(|node| node.local_id == export)
                .expect("actual root export");
            let result = saved
                .prepare_document_numeric_output(node, 0)
                .await
                .expect("pinned root export")
                .evaluate()
                .await
                .expect("current numeric result");
            assert!(result.same_value(expected.get(&export).expect("source export result")));
            assert_eq!(
                saved
                    .export_archive(ArchiveFormat::Zip, budget.max_archive_bytes)
                    .expect("unchanged source after rejection"),
                original_bytes
            );
        });
    }

    #[test]
    fn at_gra_prop_002_1_updates_only_current_hidden_boolean_literals() {
        futures::executor::block_on(async {
            for (identifier, index, name, value) in [
                (
                    "PROTONODE:math_nodes::CosineInverseNode",
                    TRIGONOMETRIC_RADIANS_INDEX,
                    "Radians",
                    false,
                ),
                (
                    "PROTONODE:math_nodes::ModuloNode",
                    modulo::AlwaysPositiveInput::INDEX,
                    "Always Positive",
                    true,
                ),
                ("NETWORK:Artboard", ARTBOARD_CLIP_INPUT, "Clip", true),
            ] {
                let source = create_gdd_folder(841, 355, "Pinned Graphite".into(), budget())
                    .await
                    .expect("source folder")
                    .insert_root_catalog_node_in_copy(identifier, [2, 4], budget())
                    .await
                    .expect("exact pinned template");
                let nodes = source.graph_nodes().await.expect("saved nodes");
                let mut roots = nodes.iter().filter(|node| node.network_path.is_empty());
                let node = roots.next().expect("source root node");
                assert!(roots.next().is_none());
                let original = Some(GddGraphBooleanInput { index, name, value });
                assert_eq!(node.boolean_input(), original);
                let graph = source.runtime_graph().await.expect("source graph");
                let document = graph
                    .network
                    .nodes
                    .get(&node.local_id)
                    .expect("source node");
                let mut changed = document.clone();
                *changed.inputs.get_mut(index).expect("source Boolean slot") =
                    NodeInput::value(TaggedValue::Bool(value), true);
                assert_eq!(graph_boolean_literal(&changed), None);
                *changed.inputs.get_mut(index).expect("source Boolean slot") =
                    NodeInput::node(NodeId(900), 0);
                assert_eq!(graph_boolean_literal(&changed), None);
                assert!(matches!(
                    source
                        .set_graph_boolean_input_in_copy(node, value, budget())
                        .await,
                    Err(GddArchiveError::UnchangedGraphInput)
                ));

                let copy = source
                    .set_graph_boolean_input_in_copy(node, !value, budget())
                    .await
                    .expect("edit the hidden Boolean slot");
                let edited = copy.runtime_graph().await.expect("edited graph");
                let actual = edited.network.nodes.get(&node.local_id).expect("same node");
                let mut expected = document.clone();
                *expected.inputs.get_mut(index).expect("source Boolean slot") =
                    NodeInput::value(TaggedValue::Bool(!value), false);
                assert_eq!(*actual, expected);
                assert_eq!(copy.history_len(), source.history_len() + 1);
                assert!(matches!(
                    copy.set_graph_boolean_input_in_copy(node, !value, budget())
                        .await,
                    Err(GddArchiveError::StaleGraphNode)
                ));
                assert_eq!(graph_boolean_literal(document), original);
            }
        });
    }

    #[test]
    fn at_gra_node_125_to_130_2_saved_trigonometry_matches_pinned_numeric_nodes() {
        use graphite_core_types::{
            Context, Node,
            list::{Item, List},
        };
        use graphite_document_graph_storage::NoMetadata;
        use graphite_math_nodes::{
            CosineInverseNode, CosineInverseNodeMapped, CosineNode, CosineNodeMapped,
            SineInverseNode, SineInverseNodeMapped, SineNode, SineNodeMapped, TangentInverseNode,
            TangentInverseNodeMapped, TangentNode, TangentNodeMapped, cosine, cosine_inverse, sine,
            sine_inverse, tangent, tangent_inverse,
        };
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
            let mut source = NodeNetwork::default();
            let mut expected = BTreeMap::new();
            let mut insert = |identifier, value, radians, output| {
                let node = NodeId(source.nodes.len() as u64 + 1);
                source.nodes.insert(
                    node,
                    DocumentNode {
                        implementation: DocumentNodeImplementation::ProtoNode(identifier),
                        inputs: vec![
                            NodeInput::value(value, true),
                            NodeInput::value(radians, false),
                        ],
                        ..Default::default()
                    },
                );
                if let Some(output) = output {
                    expected.insert(node, output);
                }
                node
            };

            macro_rules! evaluate {
                ($node:ident, $value:expr, $radians:expr) => {
                    $node::new(
                        crate::Input(Item::new_from_element($value)),
                        crate::Input(Item::new_from_element($radians)),
                    )
                    .eval(Context::default())
                    .await
                    .into_element()
                };
            }

            for radians in [false, true] {
                for (degrees, inverse_value) in [(0_f64, 0.), (90., 1.), (180., -1.), (30., -0.5)] {
                    let theta = if radians {
                        degrees.to_radians()
                    } else {
                        degrees
                    };
                    let angle = TaggedValue::Number;
                    let vector = DVec2::new(theta, -theta * 0.5);
                    for (identifier, input, output) in [
                        (
                            cosine::IDENTIFIER,
                            angle(theta),
                            evaluate!(CosineNode, theta, radians),
                        ),
                        (
                            cosine_inverse::IDENTIFIER,
                            angle(inverse_value),
                            evaluate!(CosineInverseNode, inverse_value, radians),
                        ),
                        (
                            sine::IDENTIFIER,
                            angle(theta),
                            evaluate!(SineNode, theta, radians),
                        ),
                        (
                            sine_inverse::IDENTIFIER,
                            angle(inverse_value),
                            evaluate!(SineInverseNode, inverse_value, radians),
                        ),
                        (
                            tangent::IDENTIFIER,
                            angle(theta),
                            evaluate!(TangentNode, theta, radians),
                        ),
                        (
                            tangent_inverse::IDENTIFIER,
                            angle(inverse_value),
                            evaluate!(TangentInverseNode, inverse_value, radians),
                        ),
                        (
                            tangent_inverse::IDENTIFIER,
                            TaggedValue::DVec2(vector),
                            evaluate!(TangentInverseNode, vector, radians),
                        ),
                    ] {
                        insert(
                            identifier,
                            input,
                            TaggedValue::Bool(radians),
                            Some(GddNumericValue::Number(output)),
                        );
                    }
                    for (identifier, output) in [
                        (cosine::IDENTIFIER, evaluate!(CosineNode, vector, radians)),
                        (sine::IDENTIFIER, evaluate!(SineNode, vector, radians)),
                        (tangent::IDENTIFIER, evaluate!(TangentNode, vector, radians)),
                    ] {
                        insert(
                            identifier,
                            TaggedValue::DVec2(vector),
                            TaggedValue::Bool(radians),
                            Some(GddNumericValue::Vector(output.to_array())),
                        );
                    }
                }
                let angles = [0_f64, 90., 180.].map(|degrees| {
                    if radians {
                        degrees.to_radians()
                    } else {
                        degrees
                    }
                });
                let inverse_domain = [-1.5, -1., 0., 1., 1.5];
                let items = |values: &[f64]| {
                    crate::Input(
                        values
                            .iter()
                            .copied()
                            .map(Item::new_from_element)
                            .collect::<List<_>>(),
                    )
                };
                let flags = || {
                    crate::Input(
                        std::iter::once(Item::new_from_element(radians)).collect::<List<_>>(),
                    )
                };
                macro_rules! evaluate_list {
                    ($node:ident, $values:expr) => {
                        $node::new(items($values), flags())
                            .eval(Context::default())
                            .await
                            .into_iter()
                            .map(Item::into_element)
                            .collect::<Vec<_>>()
                            .into()
                    };
                }
                for (identifier, values, output) in [
                    (
                        cosine::IDENTIFIER,
                        angles.as_slice(),
                        evaluate_list!(CosineNodeMapped, &angles),
                    ),
                    (
                        cosine_inverse::IDENTIFIER,
                        inverse_domain.as_slice(),
                        evaluate_list!(CosineInverseNodeMapped, &inverse_domain),
                    ),
                    (
                        sine::IDENTIFIER,
                        angles.as_slice(),
                        evaluate_list!(SineNodeMapped, &angles),
                    ),
                    (
                        sine_inverse::IDENTIFIER,
                        inverse_domain.as_slice(),
                        evaluate_list!(SineInverseNodeMapped, &inverse_domain),
                    ),
                    (
                        tangent::IDENTIFIER,
                        angles.as_slice(),
                        evaluate_list!(TangentNodeMapped, &angles),
                    ),
                    (
                        tangent_inverse::IDENTIFIER,
                        inverse_domain.as_slice(),
                        evaluate_list!(TangentInverseNodeMapped, &inverse_domain),
                    ),
                    (
                        sine::IDENTIFIER,
                        &[][..],
                        evaluate_list!(SineNodeMapped, &[]),
                    ),
                ] {
                    insert(
                        identifier,
                        TaggedValue::Numbers(values.to_vec()),
                        TaggedValue::Bool(radians),
                        Some(GddNumericValue::NumberList(output)),
                    );
                }
                for value in [-1.5, 1.5] {
                    for (identifier, output) in [
                        (
                            cosine_inverse::IDENTIFIER,
                            evaluate!(CosineInverseNode, value, radians),
                        ),
                        (
                            sine_inverse::IDENTIFIER,
                            evaluate!(SineInverseNode, value, radians),
                        ),
                    ] {
                        insert(
                            identifier,
                            TaggedValue::Number(value),
                            TaggedValue::Bool(radians),
                            Some(GddNumericValue::Number(output)),
                        );
                    }
                }
            }
            let wrong_port = insert(
                sine::IDENTIFIER,
                TaggedValue::Bool(true),
                TaggedValue::Bool(false),
                None,
            );
            let wrong_type = insert(
                sine::IDENTIFIER,
                TaggedValue::Number(0.5),
                TaggedValue::Number(1.),
                None,
            );
            let output_id = NodeId(1);
            source.exports = vec![NodeInput::node(output_id, 0)];

            let mut owner = document();
            let resources = HashMapResourceStorage::new();
            owner
                .commit_from_runtime(&source, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("save the pinned trigonometric network");
            let bytes: Arc<[u8]> = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("export with the source node definitions")
                .into();
            let budget = GddArchiveBudget {
                max_entries: 512,
                ..budget()
            };
            let saved = open_gdd_zip(&bytes, budget)
                .await
                .expect("reopen the saved trigonometric network");
            let original_archive = saved
                .export_archive(ArchiveFormat::Zip, budget.max_archive_bytes)
                .expect("saved source archive");
            let nodes = saved.graph_nodes().await.expect("saved node addresses");
            assert_eq!(nodes.len(), expected.len() + 2);
            for node in &nodes {
                let result = saved.preview_graph_numeric_output(node).await;
                match node.local_id {
                    node_id if node_id == wrong_port => {
                        assert!(
                            matches!(result, Err(GddArchiveError::UnsupportedGraphNumericPreview)),
                            "{node_id:?}: {result:?}"
                        );
                    }
                    node_id if node_id == wrong_type => {
                        assert!(
                            matches!(
                                result,
                                Err(GddArchiveError::GraphNumericPreviewEvaluation(_))
                            ),
                            "{node_id:?}: {result:?}"
                        );
                    }
                    node_id => {
                        let output = result.expect("typed trigonometric output");
                        assert!(
                            output.same_value(expected.get(&node_id).expect("source result")),
                            "{node_id:?}: {output:?}"
                        );
                    }
                }
            }
            let node = nodes
                .iter()
                .find(|node| node.local_id == output_id)
                .expect("exact saved root export");
            let output = saved
                .prepare_document_numeric_output(node, 0)
                .await
                .expect("pinned root export")
                .evaluate()
                .await
                .expect("typed root export");
            assert!(output.same_value(expected.get(&output_id).expect("source root output")));
            let copied = saved
                .copy_graph_nodes_to_clipboard(std::slice::from_ref(node), budget)
                .await
                .expect("copy the pinned scalar and Boolean inputs");
            let root = saved
                .graph_inspection()
                .await
                .expect("saved root")
                .networks
                .into_iter()
                .find(|network| network.parent.is_none())
                .expect("root scope");
            let pasted = saved
                .paste_graph_nodes_in_copy(&root, &copied, budget)
                .await
                .expect("paste and type-check the source node");
            let copied_nodes = pasted.graph_nodes().await.expect("pasted node addresses");
            let copy = copied_nodes
                .iter()
                .find(|candidate| {
                    !nodes
                        .iter()
                        .any(|original| candidate.storage_id == original.storage_id)
                })
                .expect("fresh pasted identity");
            let result = pasted
                .preview_graph_numeric_output(copy)
                .await
                .expect("pasted trigonometric output");
            assert!(result.same_value(&output));
            assert_eq!(
                saved
                    .export_archive(ArchiveFormat::Zip, budget.max_archive_bytes)
                    .expect("untouched source"),
                original_archive
            );
        });
    }

    #[test]
    fn at_gra_graph_035_1_evaluates_only_the_selected_numeric_branch() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let nodes = original.graph_nodes().await.expect("source graph nodes");
            let [integer, unsupported, _, _, selected, _] = nodes.as_slice() else {
                panic!("expected the root number and nested wire");
            };
            let original_registry = original.document.registry().clone();
            let original_history = original.history_len();
            let original_head = original.document.session().head_rev();
            assert_eq!(
                original
                    .runtime_graph()
                    .await
                    .expect("source export")
                    .network
                    .exports,
                vec![NodeInput::node(integer.local_id, 0)]
            );
            assert!(matches!(
                original.preview_graph_numeric_output(unsupported).await,
                Err(GddArchiveError::UnsupportedGraphNumericPreview)
            ));
            assert!(matches!(
                original
                    .preview_graph_numeric_output(integer)
                    .await
                    .expect("integer output"),
                GddNumericValue::Integer(17)
            ));
            assert!(matches!(
                original
                    .preview_graph_numeric_output(selected)
                    .await
                    .expect("connected scalar output"),
                GddNumericValue::Number(-1.)
            ));
            assert!(original.document.registry().value_equal(&original_registry));
            assert_eq!(original.history_len(), original_history);
            assert_eq!(original.document.session().head_rev(), original_head);
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn numeric_preview_evaluates_pinned_binary_branches_and_reuses_shared_inputs() {
        use graphite_math_nodes::{
            absolute_value, add, divide, multiply, reciprocal, root, subtract,
        };
        use graphite_std::platform_application_io::resource;

        futures::executor::block_on(async {
            let document = |identifier, inputs| DocumentNode {
                implementation: DocumentNodeImplementation::ProtoNode(identifier),
                inputs,
                ..Default::default()
            };
            let literal = |value| NodeInput::value(TaggedValue::Number(value), false);
            let selected = NodeId(40);
            for (identifier, inputs, expected) in [
                (add::IDENTIFIER, vec![literal(8.), literal(2.)], 10.),
                (subtract::IDENTIFIER, vec![literal(8.), literal(2.)], 6.),
                (multiply::IDENTIFIER, vec![literal(8.), literal(2.)], 16.),
                (divide::IDENTIFIER, vec![literal(8.), literal(2.)], 4.),
                (root::IDENTIFIER, vec![literal(8.), literal(3.)], 2.),
                (reciprocal::IDENTIFIER, vec![literal(-8.)], -0.125),
            ] {
                let source = NodeNetwork {
                    nodes: [(selected, document(identifier, inputs))]
                        .into_iter()
                        .collect(),
                    ..Default::default()
                };
                let detached = numeric_preview_network(&source, selected, &[])
                    .expect("admitted source operation");
                assert!(
                    preview_numeric_value(
                        &evaluate_detached_numeric_network(detached)
                            .await
                            .expect("pinned numeric output")
                    )
                    .expect("numeric preview")
                    .same_value(&GddNumericValue::Number(expected))
                );
            }
            let shared = NodeId(12);
            let other = NodeId(13);
            let sum = NodeId(14);
            let mut source = NodeNetwork {
                nodes: [
                    (
                        shared,
                        document(absolute_value::IDENTIFIER, vec![literal(-7.)]),
                    ),
                    (
                        other,
                        document(absolute_value::IDENTIFIER, vec![literal(-4.)]),
                    ),
                    (
                        sum,
                        document(
                            add::IDENTIFIER,
                            vec![NodeInput::node(shared, 0), NodeInput::node(other, 0)],
                        ),
                    ),
                    (
                        selected,
                        document(
                            subtract::IDENTIFIER,
                            vec![NodeInput::node(sum, 0), NodeInput::node(shared, 0)],
                        ),
                    ),
                ]
                .into_iter()
                .collect(),
                exports: vec![NodeInput::node(sum, 0)],
                ..Default::default()
            };
            let detached = numeric_preview_network(&source, selected, &[])
                .expect("shared input is not a cycle");
            assert_eq!(detached.nodes.len(), 4);
            assert_eq!(detached.exports, vec![NodeInput::node(selected, 0)]);
            assert_eq!(source.exports, vec![NodeInput::node(sum, 0)]);
            assert!(matches!(
                evaluate_detached_numeric_network(detached)
                    .await
                    .expect("selected branch"),
                TaggedValue::Number(4.)
            ));
            assert!(
                source
                    .nodes
                    .insert(other, document(resource::IDENTIFIER, vec![literal(4.)]))
                    .is_some()
            );
            assert!(matches!(
                numeric_preview_network(&source, selected, &[]),
                Err(GddArchiveError::UnsupportedGraphNumericPreview)
            ));
        });
    }

    #[test]
    fn numeric_preview_preserves_literal_vector_and_list_outputs_and_rejects_unbounded_branches() {
        use graphite_graph_craft::document::DocumentNode;

        futures::executor::block_on(async {
            let operation = |input| DocumentNode {
                implementation: DocumentNodeImplementation::ProtoNode(numeric_node_identifier(
                    UnaryNumericOperation::Round,
                )),
                inputs: vec![input],
                ..Default::default()
            };
            let selected = NodeId(232);
            for (input, expected) in [
                (
                    TaggedValue::DVec2(DVec2::new(-1.5, 2.49)),
                    GddNumericValue::Vector([-2., 2.]),
                ),
                (
                    TaggedValue::Numbers(vec![2.5, -2.5, 2.5]),
                    GddNumericValue::NumberList(Arc::from([3., -3., 3.])),
                ),
            ] {
                let source = NodeNetwork {
                    nodes: [(selected, operation(NodeInput::value(input, false)))]
                        .into_iter()
                        .collect(),
                    ..Default::default()
                };
                let detached =
                    numeric_preview_network(&source, selected, &[]).expect("bounded pure branch");
                let actual = evaluate_detached_numeric_network(detached)
                    .await
                    .expect("pinned typed result");
                assert!(
                    preview_numeric_value(&actual)
                        .expect("numeric preview")
                        .same_value(&expected)
                );
            }
            let cyclic = NodeNetwork {
                nodes: [
                    (selected, operation(NodeInput::node(NodeId(445), 0))),
                    (NodeId(445), operation(NodeInput::node(selected, 0))),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            };
            assert!(matches!(
                numeric_preview_network(&cyclic, selected, &[]),
                Err(GddArchiveError::CyclicGraphNumericPreview)
            ));
            let overlong = NodeNetwork {
                nodes: (0..33)
                    .map(|index| {
                        (
                            NodeId(index),
                            operation(if index == 32 {
                                NodeInput::value(TaggedValue::Number(1.), false)
                            } else {
                                NodeInput::node(NodeId(index + 1), 0)
                            }),
                        )
                    })
                    .collect(),
                ..Default::default()
            };
            assert!(matches!(
                numeric_preview_network(&overlong, NodeId(0), &[]),
                Err(GddArchiveError::GraphNumericPreviewBudget)
            ));
            assert!(matches!(
                preview_numeric_value(&TaggedValue::Numbers(vec![0.; 1025])),
                Err(GddArchiveError::GraphNumericPreviewBudget)
            ));
        });
    }

    #[test]
    fn at_gra_graph_014_1_moves_a_scoped_group_in_one_interaction() {
        use graphite_document_graph_storage::attr::node::ui::POSITION;

        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("pinned numeric graph");
            let nodes = original.graph_nodes().await.expect("scoped graph nodes");
            let [root, _, first, second, _, _] = nodes.as_slice() else {
                panic!("group with repeated local IDs")
            };
            let offset = [4, -5];
            assert!(matches!(
                original
                    .move_graph_nodes_in_copy(&[root.clone(), first.clone()], offset, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphGridMove)
            ));
            assert!(matches!(
                original
                    .move_graph_nodes_in_copy(&[first.clone(), first.clone()], offset, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphGridMove)
            ));
            assert!(matches!(
                original
                    .move_graph_nodes_in_copy(&[first.clone(), second.clone()], [0, 0], budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphGridPosition)
            ));
            let moved = original
                .move_graph_nodes_in_copy(&[first.clone(), second.clone()], offset, budget())
                .await
                .expect("move the whole group");
            let changes = moved
                .document
                .session()
                .history()
                .skip(original.history_len())
                .collect::<Vec<_>>();
            assert_eq!(changes.len(), 2);
            assert!(changes.iter().all(|change| matches!(&change.kind, RegistryDelta::ChangeNodeAttribute { id, delta } if [first.storage_id(), second.storage_id()].contains(id) && delta.key == POSITION)));
            let moved_nodes = moved.graph_nodes().await.expect("reopened group positions");
            for original in &nodes {
                let current = moved_nodes
                    .iter()
                    .find(|node| node.storage_id() == original.storage_id())
                    .expect("stable node ID");
                let expected = original.grid_position().map(|[x, y]| {
                    if [first.storage_id(), second.storage_id()].contains(&original.storage_id()) {
                        [x + offset[0], y + offset[1]]
                    } else {
                        [x, y]
                    }
                });
                assert_eq!(current.grid_position(), expected);
            }
            let undone = moved
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one Undo for the group");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
        });
    }

    #[test]
    fn at_gra_graph_014_1_moves_the_nested_node_grid_position_without_rewriting_the_wire() {
        use graphite_document_graph_storage::attr::node::ui::POSITION;

        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source graph");
            let nodes = original.graph_nodes().await.expect("source nodes");
            let [root, _, upstream, _, downstream, _] = nodes.as_slice() else {
                panic!("expected repeated local IDs and a nested wire");
            };
            assert_eq!(root.local_id, upstream.local_id);
            let original_position = upstream.grid_position().expect("free grid position");
            assert!(matches!(
                original
                    .move_graph_node_in_copy(upstream, original_position, budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphGridPosition)
            ));
            assert!(matches!(
                original
                    .move_graph_node_in_copy(downstream, [-12, 34], budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphGridMove)
            ));
            let moved = original
                .move_graph_node_in_copy(upstream, [-12, 34], budget())
                .await
                .expect("move the selected node");
            assert_eq!(moved.history_len(), original.history_len() + 1);
            let saved = moved
                .document
                .session()
                .history()
                .last()
                .expect("saved move");
            assert!(
                matches!(&saved.kind, RegistryDelta::ChangeNodeAttribute { id, delta }
                if *id == upstream.storage_id && delta.key == POSITION)
            );
            let new_nodes = moved.graph_nodes().await.expect("reopened moved graph");
            let new_upstream = new_nodes
                .iter()
                .find(|node| node.storage_id == upstream.storage_id)
                .expect("same stable node");
            assert_eq!(new_upstream.grid_position(), Some([-12, 34]));
            assert!(matches!(
                moved
                    .move_graph_node_in_copy(new_upstream, [i32::MAX, 34], budget())
                    .await,
                Err(GddArchiveError::GraphGridMoveOutOfRange)
            ));
            assert!(matches!(
                moved
                    .move_graph_node_in_copy(upstream, [2, 5], budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let registry = moved.document.registry();
            let before = original.document.registry();
            for node in nodes
                .iter()
                .filter(|node| node.storage_id != upstream.storage_id)
            {
                assert_eq!(
                    registry.node_instances[&node.storage_id],
                    before.node_instances[&node.storage_id]
                );
            }
            let moved_node = &registry.node_instances[&upstream.storage_id];
            assert_eq!(
                moved_node.inputs(),
                before.node_instances[&upstream.storage_id].inputs()
            );
            let undo = moved
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores the position");
            assert!(undo.document.registry().value_equal(before));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo moves the same node");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_012_1_duplicates_a_group_with_only_its_internal_wires() {
        futures::executor::block_on(async {
            let original = open_gdd_zip(&numeric_graph_fixture().await, budget())
                .await
                .expect("source graph");
            let graph = original.graph_inspection().await.expect("inspect source");
            let [root, nested] = graph.networks.as_slice() else {
                panic!("expected one nested scope");
            };
            let consumer = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(11))
                .expect("original downstream node");
            let producer = consumer
                .numeric_connection_source(&graph.nodes)
                .expect("original upstream node");
            let independent = graph
                .nodes
                .iter()
                .find(|node| node.local_id == NodeId(9))
                .expect("independent node in the same scope");
            let mut targets = [consumer.clone(), producer.clone()];
            assert!(matches!(
                original
                    .duplicate_graph_nodes_in_copy(root, &targets, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphClipboardSelection)
            ));
            let linked = original
                .duplicate_graph_nodes_in_copy(nested, &targets, budget())
                .await
                .expect("duplicate both ends of the wire");
            let nodes = linked.graph_nodes().await.expect("saved duplicated graph");
            let is_fresh = |node: &GddGraphNode| {
                !original
                    .document
                    .registry()
                    .node_instances
                    .contains_key(&node.storage_id)
            };
            let linked_consumer = nodes
                .iter()
                .find(|node| is_fresh(node) && node.has_numeric_connection())
                .expect("duplicate with the internal wire");
            assert!(
                linked_consumer
                    .numeric_connection_source(&nodes)
                    .is_some_and(is_fresh)
            );
            assert_eq!(linked_consumer.network_id(), nested.id);
            assert!(matches!(
                linked.preview_graph_numeric_output(linked_consumer).await,
                Ok(GddNumericValue::Number(-1.))
            ));
            for (id, value) in &original.document.registry().node_instances {
                assert_eq!(
                    linked.document.registry().node_instances.get(id),
                    Some(value)
                );
            }
            let undo = linked
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo removes both duplicates");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            assert!(matches!(
                linked
                    .duplicate_graph_nodes_in_copy(nested, &targets, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNetwork)
            ));
            targets = [consumer.clone(), independent.clone()];
            let fallback = original
                .duplicate_graph_nodes_in_copy(nested, &targets, budget())
                .await
                .expect("duplicate with the original source outside the selection");
            let nodes = fallback
                .graph_nodes()
                .await
                .expect("saved duplicated graph");
            let added = nodes
                .iter()
                .filter(|node| is_fresh(node))
                .collect::<Vec<_>>();
            assert_eq!(added.len(), 2);
            assert!(added.iter().all(|node| !node.has_numeric_connection()));
            assert!(
                added
                    .iter()
                    .any(|node| node.numeric_input_exposed() == Some(true))
            );
            assert!(nodes.iter().any(|node| {
                node.storage_id == consumer.storage_id
                    && node
                        .numeric_connection_source(&nodes)
                        .is_some_and(|source| source.storage_id == producer.storage_id)
            }));
        });
    }

    #[test]
    fn at_gra_graph_012_1_duplicates_a_nested_operation_with_source_fallback_and_exact_layout() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("source archive");
            let nodes = original.graph_nodes().await.expect("source graph");
            let [_, unsupported, _, _, selected, _] = nodes.as_slice() else {
                panic!("expected source graph with a nested wired node");
            };
            assert!(!unsupported.can_duplicate_in_copy());
            assert!(matches!(
                original
                    .duplicate_graph_numeric_node_in_copy(unsupported, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphNodeDuplication)
            ));
            assert!(selected.can_duplicate_in_copy());
            let path = &selected.network_path;
            let original_interface = original
                .runtime_graph()
                .await
                .expect("runtime graph")
                .into_interface()
                .expect("source editor");
            let original_position = original_interface
                .position(&selected.local_id, path)
                .expect("source absolute position");
            let original_template = original_interface
                .create_node_template(&selected.local_id, path)
                .expect("source node template");
            let copy = original
                .duplicate_graph_numeric_node_in_copy(selected, budget())
                .await
                .expect("duplicate pinned numeric node");
            let copied_nodes = copy.graph_nodes().await.expect("reopened graph");
            assert_eq!(copied_nodes.len(), nodes.len() + 1);
            let inserted = copied_nodes
                .iter()
                .find(|new| {
                    !nodes
                        .iter()
                        .any(|original| original.storage_id == new.storage_id)
                })
                .expect("one fresh stable node ID");
            assert_eq!(inserted.network_path, *path);
            assert_ne!(inserted.local_id, selected.local_id);
            assert_eq!(inserted.numeric_input_exposed(), Some(true));
            let graph = copy
                .runtime_graph()
                .await
                .expect("saved graph")
                .into_interface()
                .expect("saved source editor");
            let inserted_template = graph
                .create_node_template(&inserted.local_id, path)
                .expect("saved duplicate template");
            assert_eq!(
                graph.position(&inserted.local_id, path),
                Some(original_position + IVec2::new(2, 2))
            );
            assert_eq!(
                inserted_template.input_metadata,
                original_template.input_metadata
            );
            assert!(matches!(
                inserted_template.inputs.as_slice(),
                [NodeInput::Value { tagged_value, exposed: true }]
                    if **tagged_value == TaggedValue::DVec2(DVec2::ZERO)
            ));
            let registry = copy.document.registry();
            let original_registry = original.document.registry();
            assert_eq!(registry.resources.len(), original_registry.resources.len());
            for node in &nodes {
                assert!(
                    registry.node_instances[&node.storage_id]
                        .value_equal(&original_registry.node_instances[&node.storage_id])
                );
            }
            assert!(matches!(
                copy.duplicate_graph_numeric_node_in_copy(selected, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let undo = copy
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo removes the duplicate");
            assert!(undo.document.registry().value_equal(original_registry));
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the duplicate");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_013_1_removes_only_the_addressed_isolated_node_in_one_interaction() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original archive");
            let nodes = original.graph_nodes().await.expect("original nodes");
            let [root, parent, source, isolated, wired, sibling] = nodes.as_slice() else {
                panic!("expected original root and nested graph identities");
            };
            assert_eq!(isolated.storage_id, StorageNodeId(0xbc11_824d_7ec3_201a));
            for node in [parent, source, wired] {
                assert!(!node.can_delete_in_copy());
                assert!(matches!(
                    original
                        .delete_isolated_graph_node_in_copy(node, budget())
                        .await,
                    Err(GddArchiveError::UnsupportedGraphNodeDeletion)
                ));
            }
            let pinned = original
                .set_graph_node_pinned_in_copy(isolated, true, budget())
                .await
                .expect("pin the nested node first");
            assert!(matches!(
                pinned
                    .delete_isolated_graph_node_in_copy(isolated, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let pinned_nodes = pinned.graph_nodes().await.expect("retained pinned node");
            let selected = pinned_nodes
                .iter()
                .find(|node| node.storage_id == isolated.storage_id)
                .expect("same addressed node");
            assert_eq!(selected.pinned_position(), Some((1, 1)));
            assert!(selected.can_delete_in_copy());
            let removed = pinned
                .delete_isolated_graph_node_in_copy(selected, budget())
                .await
                .expect("remove the isolated node");
            let registry = removed.document.registry();
            assert!(!registry.node_instances.contains_key(&isolated.storage_id));
            assert_eq!(registry.node_instances.len(), nodes.len() - 1);
            for node in [root, parent, source, wired, sibling] {
                assert!(
                    registry.node_instances[&node.storage_id]
                        .value_equal(&pinned.document.registry().node_instances[&node.storage_id])
                );
            }
            let actual = removed
                .runtime_graph()
                .await
                .expect("reopened graph")
                .into_interface()
                .expect("source editor");
            assert!(
                actual
                    .ordered_pinned_nodes(&isolated.network_path)
                    .is_empty()
            );
            assert_eq!(
                registry.resources.len(),
                pinned.document.registry().resources.len()
            );
            let undo = removed
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores the node and pin order");
            assert!(
                undo.document
                    .registry()
                    .value_equal(pinned.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo removes the same node");
            assert!(redo.document.registry().value_equal(registry));
        });
    }

    #[test]
    fn at_gra_graph_007_1_connects_only_compatible_ports_in_the_addressed_network() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original archive");
            let nodes = original.graph_nodes().await.expect("original graph nodes");
            let [root, _, source, target, wired, wrong_type] = nodes.as_slice() else {
                panic!("expected original root and nested graph identities");
            };
            assert!(target.can_connect_numeric_input_from(source));
            for incompatible in [root, target, wired, wrong_type] {
                assert!(!target.can_connect_numeric_input_from(incompatible));
                assert!(matches!(
                    original
                        .connect_graph_numeric_input_in_copy(target, incompatible, budget())
                        .await,
                    Err(GddArchiveError::InvalidGraphConnection)
                ));
            }
            assert!(matches!(
                original
                    .connect_graph_numeric_input_in_copy(source, target, budget())
                    .await,
                Err(GddArchiveError::InvalidGraphConnection)
            ));

            let connected = original
                .connect_graph_numeric_input_in_copy(target, source, budget())
                .await
                .expect("connect original compatible operations");
            assert_eq!(connected.history_len(), original.history_len() + 1);
            let graph = connected
                .runtime_graph()
                .await
                .expect("reopened runtime graph");
            let interface = graph.into_interface().expect("source editor");
            let connector = InputConnector::node_at_index(target.local_id, 0);
            assert_eq!(
                interface.input_from_connector(&connector, &target.network_path),
                Some(&NodeInput::node(source.local_id, 0))
            );
            for untouched in [root, source, wired, wrong_type] {
                assert_eq!(
                    connected
                        .document
                        .registry()
                        .node_instances
                        .get(&untouched.storage_id),
                    original
                        .document
                        .registry()
                        .node_instances
                        .get(&untouched.storage_id)
                );
            }
            let reopened_nodes = connected.graph_nodes().await.expect("reopened graph nodes");
            let connected_node = reopened_nodes
                .iter()
                .find(|node| node.storage_id == target.storage_id)
                .expect("same target");
            assert!(connected_node.has_numeric_connection());
            assert!(matches!(
                connected
                    .connect_graph_numeric_input_in_copy(connected_node, source, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let undone = connected
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo one wire insertion");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("redo one wire insertion");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(connected.document.registry())
            );
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_008_1_reconnects_the_exact_nested_input_in_one_interaction() {
        use graphite_document_graph_storage::NodeInput as StoredInput;

        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget()).await.expect("source ZIP");
            let nodes = original.graph_nodes().await.expect("original nodes");
            let [root, _, previous, replacement, target, wrong_type] = nodes.as_slice() else {
                panic!("expected original root and nested nodes");
            };
            assert_eq!(
                target
                    .numeric_connection_source(&nodes)
                    .map(|node| node.storage_id),
                Some(previous.storage_id)
            );
            assert!(target.can_reconnect_numeric_input_from(previous, replacement));
            for rejected in [root, previous, target, wrong_type] {
                assert!(!target.can_reconnect_numeric_input_from(previous, rejected));
                assert!(matches!(
                    original
                        .reconnect_graph_numeric_input_in_copy(target, previous, rejected, budget())
                        .await,
                    Err(GddArchiveError::InvalidGraphConnection)
                ));
            }
            assert!(matches!(
                original
                    .reconnect_graph_numeric_input_in_copy(target, replacement, previous, budget())
                    .await,
                Err(GddArchiveError::InvalidGraphConnection)
            ));

            let reconnected = original
                .reconnect_graph_numeric_input_in_copy(target, previous, replacement, budget())
                .await
                .expect("reconnect to the new source");
            let history = reconnected
                .document
                .session()
                .history()
                .skip(original.history_len())
                .collect::<Vec<_>>();
            let [disconnected, connected] = history.as_slice() else {
                panic!("expected both original source writes");
            };
            assert!(matches!(
                &disconnected.kind,
                RegistryDelta::ChangeNodeInput {
                    id,
                    index: 0,
                    new_input: StoredInput::Value { exposed: true, .. },
                } if *id == target.storage_id
            ));
            assert!(matches!(
                &connected.kind,
                RegistryDelta::ChangeNodeInput {
                    id,
                    index: 0,
                    new_input: StoredInput::Node { id: output, index: 0 },
                } if *id == target.storage_id && *output == replacement.storage_id
            ));
            for untouched in [root, previous, replacement, wrong_type] {
                assert_eq!(
                    reconnected
                        .document
                        .registry()
                        .node_instances
                        .get(&untouched.storage_id),
                    original
                        .document
                        .registry()
                        .node_instances
                        .get(&untouched.storage_id)
                );
            }
            let nodes = reconnected.graph_nodes().await.expect("reopened graph");
            let updated = nodes
                .iter()
                .find(|node| node.storage_id == target.storage_id)
                .expect("same target");
            assert_eq!(
                updated
                    .numeric_connection_source(&nodes)
                    .map(|node| node.storage_id),
                Some(replacement.storage_id)
            );
            assert!(matches!(
                reconnected
                    .reconnect_graph_numeric_input_in_copy(updated, replacement, previous, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let undone = reconnected
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores the original output");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo restores the replacement output");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(reconnected.document.registry())
            );
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_009_1_disconnects_the_exact_wire_and_keeps_its_input_attributes() {
        use graphite_document_graph_storage::attr::node::input::ui::NAME;

        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original archive");
            let nodes = original.graph_nodes().await.expect("original nodes");
            let [root, _, upstream, _, wired, ..] = nodes.as_slice() else {
                panic!("expected the original node identities");
            };
            assert!(wired.has_numeric_connection());
            assert!(!upstream.has_numeric_connection());
            assert!(matches!(
                original
                    .disconnect_graph_numeric_input_in_copy(upstream, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphConnection)
            ));

            let interface = original
                .runtime_graph()
                .await
                .expect("original runtime graph")
                .into_interface()
                .expect("original editor");
            let connector = InputConnector::node_at_index(wired.local_id, 0);
            let expected = interface.tagged_value_from_input(&connector, &wired.network_path);
            assert!(
                matches!(expected, TaggedValue::DVec2(value) if value == DVec2::ZERO),
                "source fallback was {expected:?}"
            );
            let disconnected = original
                .disconnect_graph_numeric_input_in_copy(wired, budget())
                .await
                .expect("disconnect one source wire");
            let interface = disconnected
                .runtime_graph()
                .await
                .expect("disconnected runtime graph")
                .into_interface()
                .expect("reopened editor");
            assert_eq!(
                interface.input_from_connector(&connector, &wired.network_path),
                Some(&NodeInput::value(expected, true))
            );
            let source_slot = original
                .document
                .registry()
                .node_instances
                .get(&wired.storage_id)
                .and_then(|node| node.inputs().first())
                .expect("original stored input");
            let reopened_slot = disconnected
                .document
                .registry()
                .node_instances
                .get(&wired.storage_id)
                .and_then(|node| node.inputs().first())
                .expect("new stored input");
            assert_eq!(
                source_slot.attributes.get_typed::<String>(NAME).as_deref(),
                Some("Original wired input")
            );
            assert_eq!(reopened_slot.attributes, source_slot.attributes);
            assert_eq!(disconnected.history_len(), original.history_len() + 1);
            for unchanged in [root, upstream] {
                assert_eq!(
                    disconnected
                        .document
                        .registry()
                        .node_instances
                        .get(&unchanged.storage_id),
                    original
                        .document
                        .registry()
                        .node_instances
                        .get(&unchanged.storage_id)
                );
            }
            assert!(matches!(
                disconnected
                    .disconnect_graph_numeric_input_in_copy(wired, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let undone = disconnected
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo one wire deletion");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("redo one wire deletion");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(disconnected.document.registry())
            );
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_028_1_round_trips_source_port_exposure_and_its_literal() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original archive");
            let nodes = original.graph_nodes().await.expect("original nodes");
            let [root, _, nested, _, wired, ..] = nodes.as_slice() else {
                panic!("expected the original node identities");
            };
            assert_eq!(nested.numeric_input_exposed(), Some(false));
            assert!(matches!(
                original
                    .set_graph_numeric_input_exposed_in_copy(nested, false, budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphInputExposure)
            ));
            assert!(matches!(
                original
                    .set_graph_numeric_input_exposed_in_copy(wired, true, budget())
                    .await,
                Err(GddArchiveError::UnsupportedGraphNumericInput)
            ));

            let exposed = original
                .set_graph_numeric_input_exposed_in_copy(nested, true, budget())
                .await
                .expect("expose the nested literal");
            assert_eq!(exposed.history_len(), original.history_len() + 1);
            let nodes = exposed.graph_nodes().await.expect("exposed graph");
            let selected = nodes
                .iter()
                .find(|node| node.storage_id == nested.storage_id)
                .expect("same nested slot");
            assert_eq!(selected.numeric_input_exposed(), Some(true));
            assert!(selected.numeric_input().is_none());
            assert!(
                selected
                    .numeric_port()
                    .is_some_and(|port| port.value.same_value(&GddNumericValue::Number(-2.5)))
            );
            for unchanged in [root, wired] {
                assert_eq!(
                    exposed
                        .document
                        .registry()
                        .node_instances
                        .get(&unchanged.storage_id),
                    original
                        .document
                        .registry()
                        .node_instances
                        .get(&unchanged.storage_id)
                );
            }
            assert!(matches!(
                exposed
                    .set_graph_numeric_input_exposed_in_copy(nested, false, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));

            let undo = exposed
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo exposure");
            assert!(
                undo.document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("redo exposure");
            assert!(
                redo.document
                    .registry()
                    .value_equal(exposed.document.registry())
            );
            let hidden = exposed
                .set_graph_numeric_input_exposed_in_copy(selected, false, budget())
                .await
                .expect("restore the literal property");
            assert!(
                hidden
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            assert_eq!(hidden.history_len(), original.history_len() + 2);
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_031_1_edits_the_inserted_number_list_default_without_changing_type() {
        futures::executor::block_on(async {
            let blank = create_gdd_folder(918, 501, "Pinned Graphite".into(), budget())
                .await
                .expect("source folder");
            let original = blank
                .insert_root_numeric_node_in_copy(UnaryNumericOperation::Sign, [1, 2], budget())
                .await
                .expect("insert the default Sign definition");
            let original_nodes = original.graph_nodes().await.expect("original default");
            let [original_node] = original_nodes.as_slice() else {
                panic!("expected one original node");
            };
            assert_eq!(original_node.numeric_input_exposed(), Some(true));
            let hidden = original
                .set_graph_numeric_input_exposed_in_copy(original_node, false, budget())
                .await
                .expect("expose the original literal editor");
            let hidden_nodes = hidden.graph_nodes().await.expect("hidden list port");
            let [node] = hidden_nodes.as_slice() else {
                panic!("expected one edited node");
            };
            let Some(default @ GddNumericValue::NumberList(numbers)) = node.numeric_input() else {
                panic!("expected the original typed list");
            };
            assert!(numbers.is_empty());
            for malformed in ["2,,3", "0, NaN", "Infinity", "1,"] {
                assert!(matches!(
                    default.parse_like(malformed),
                    Err(GddArchiveError::InvalidGraphNumberList)
                ));
            }
            assert!(matches!(
                hidden
                    .set_graph_numeric_input_in_copy(node, GddNumericValue::Number(1.), budget())
                    .await,
                Err(GddArchiveError::GraphInputTypeChanged)
            ));
            assert!(matches!(
                hidden
                    .set_graph_numeric_input_in_copy(
                        node,
                        GddNumericValue::NumberList(Arc::from([f64::NAN])),
                        budget(),
                    )
                    .await,
                Err(GddArchiveError::InvalidGraphNumberList)
            ));
            let values = [-1.25, -0., -1.25, 0.03];
            let draft = default
                .parse_like("-1.25, -0, -1.25, 3e-2")
                .expect("valid list");
            let edited = hidden
                .set_graph_numeric_input_in_copy(node, draft, budget())
                .await
                .expect("save the ordered list");
            let bytes: Arc<[u8]> = edited
                .export_archive(ArchiveFormat::Zip, budget().max_archive_bytes)
                .expect("serialize the list")
                .into();
            let reopened = open_gdd_zip(&bytes, budget())
                .await
                .expect("reopen the list");
            let nodes = reopened.graph_nodes().await.expect("reopened nodes");
            let [saved] = nodes.as_slice() else {
                panic!("expected one reopened node");
            };
            assert_eq!(saved.storage_id, original_node.storage_id);
            let Some(value @ GddNumericValue::NumberList(_)) = saved.numeric_input() else {
                panic!("expected the saved source list");
            };
            assert!(value.same_value(&GddNumericValue::NumberList(Arc::from(values))));
            assert_eq!(value.to_string(), "-1.25, -0, -1.25, 0.03");
            assert!(matches!(
                reopened
                    .set_graph_numeric_input_in_copy(saved, value.clone(), budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphInput)
            ));
            let empty = value
                .parse_like(" ")
                .expect("blank represents an empty list");
            let cleared = reopened
                .set_graph_numeric_input_in_copy(saved, empty, budget())
                .await
                .expect("clear the list");
            let undo = cleared
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo restores the list");
            assert!(
                undo.document
                    .registry()
                    .value_equal(reopened.document.registry())
            );
            let redo = undo
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo clears the list");
            assert!(
                redo.document
                    .registry()
                    .value_equal(cleared.document.registry())
            );
        });
    }

    #[test]
    fn at_gra_graph_031_1_preserves_typed_literals_and_targets_the_exact_nested_slot() {
        futures::executor::block_on(async {
            let bytes = numeric_graph_fixture().await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original document");
            let nodes = original.graph_nodes().await.expect("typed values");
            let [root, parent, nested, exposed, wired, ..] = nodes.as_slice() else {
                panic!("expected the original node identities");
            };
            assert!(parent.numeric_input().is_none());
            assert!(matches!(
                root.numeric_input(),
                Some(GddNumericValue::Integer(-17))
            ));
            assert!(matches!(
                nested.numeric_input(),
                Some(GddNumericValue::Number(-2.5))
            ));
            for (unavailable, expected) in [
                (exposed, GddArchiveError::ExposedGraphNumericInput),
                (wired, GddArchiveError::UnsupportedGraphNumericInput),
            ] {
                assert!(unavailable.numeric_input().is_none());
                let result = original
                    .set_graph_numeric_input_in_copy(
                        unavailable,
                        GddNumericValue::Number(1.),
                        budget(),
                    )
                    .await;
                let error = result.err().expect("source port cannot edit its value");
                assert_eq!(
                    std::mem::discriminant(&error),
                    std::mem::discriminant(&expected)
                );
            }
            for (value, expected) in [
                (
                    GddNumericValue::Integer(1),
                    GddArchiveError::GraphInputTypeChanged,
                ),
                (
                    GddNumericValue::Number(f64::INFINITY),
                    GddArchiveError::InvalidGraphNumber,
                ),
                (
                    GddNumericValue::Number(-2.5),
                    GddArchiveError::UnchangedGraphInput,
                ),
            ] {
                let error = original
                    .set_graph_numeric_input_in_copy(nested, value, budget())
                    .await
                    .err()
                    .expect("invalid input");
                assert_eq!(
                    std::mem::discriminant(&error),
                    std::mem::discriminant(&expected)
                );
            }
            let edited = original
                .set_graph_numeric_input_in_copy(nested, GddNumericValue::Number(-0.), budget())
                .await
                .expect("save the exact negative zero value");
            assert_eq!(edited.history_len(), original.history_len() + 1);
            for untouched in [root, parent, exposed, wired] {
                assert_eq!(
                    edited
                        .document
                        .registry()
                        .node_instances
                        .get(&untouched.storage_id),
                    original
                        .document
                        .registry()
                        .node_instances
                        .get(&untouched.storage_id)
                );
            }
            let new_nodes = edited.graph_nodes().await.expect("edited typed values");
            let edited_node = new_nodes
                .iter()
                .find(|node| node.storage_id == nested.storage_id)
                .expect("same storage identity");
            assert!(
                edited_node
                    .numeric_input()
                    .is_some_and(|value| value.same_value(&GddNumericValue::Number(-0.)))
            );
            assert!(matches!(
                edited
                    .set_graph_numeric_input_in_copy(
                        edited_node,
                        GddNumericValue::Number(0.),
                        budget()
                    )
                    .await,
                Err(GddArchiveError::UnchangedGraphInput)
            ));
            assert!(matches!(
                edited
                    .set_graph_numeric_input_in_copy(nested, GddNumericValue::Number(3.), budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let undone = edited
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo numeric edit");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("redo numeric edit");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(edited.document.registry())
            );

            let integer_edit = original
                .set_graph_numeric_input_in_copy(root, GddNumericValue::Integer(i64::MAX), budget())
                .await
                .expect("edit exact signed integer");
            let new_nodes = integer_edit
                .graph_nodes()
                .await
                .expect("exact integer input");
            let integer_node = new_nodes
                .iter()
                .find(|node| node.storage_id == root.storage_id)
                .expect("same root node");
            assert!(matches!(
                integer_node.numeric_input(),
                Some(GddNumericValue::Integer(i64::MAX))
            ));
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_032_1_renames_only_the_explicit_nested_node_in_a_reopened_copy() {
        use graphite_document_graph_storage::attr::node::ui::DISPLAY_NAME;

        futures::executor::block_on(async {
            let repeated = NodeId(7);
            let owner_node = NodeId(24);
            let owner = document_with_nested_graph_nodes();
            let bytes = export(&owner, Some(b"retain exact legacy entry")).await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original snapshot");
            let nodes = original
                .graph_nodes()
                .await
                .expect("stable native node addresses");
            let root = nodes
                .iter()
                .find(|node| node.local_id == repeated && node.network_path.is_empty())
                .expect("same local ID on root");
            let target = nodes
                .iter()
                .find(|node| node.local_id == repeated && node.network_path == [owner_node])
                .expect("exact nested target");
            assert_ne!(target.storage_id, root.storage_id);
            assert!(matches!(
                original
                    .rename_graph_node_in_copy(target, target.name.clone(), budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphNodeName)
            ));
            let reselected = open_gdd_zip(&bytes, budget())
                .await
                .expect("different snapshot of the same source");
            assert!(matches!(
                reselected
                    .rename_graph_node_in_copy(target, "stale".into(), budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));

            let renamed = original
                .rename_graph_node_in_copy(target, "Nested only".into(), budget())
                .await
                .expect("source editor prepares exactly the addressed rename");
            assert_eq!(renamed.history_len(), original.history_len() + 1);
            assert_eq!(renamed.hot_log_len(), 0);
            let history = renamed
                .document
                .session()
                .history()
                .last()
                .expect("new interaction");
            assert!(
                matches!(&history.kind, RegistryDelta::ChangeNodeAttribute { id, delta } if *id == target.storage_id && delta.key == DISPLAY_NAME)
            );
            assert_eq!(
                renamed
                    .read_entry(GddV1Layout.legacy_path())
                    .expect("legacy data")
                    .as_slice(),
                b"retain exact legacy entry",
            );
            let working = renamed
                .copy_entries(&budget())
                .expect("working entries")
                .finish()
                .expect("valid layout")
                .working;
            let mut reopened = GddV1::open_in(AnyContainer::Memory(working), GddV1Layout)
                .await
                .expect("pinned source reopens the renamed copy");
            assert_eq!(
                reopened.registry().node_instances.get(&root.storage_id),
                original
                    .document
                    .registry()
                    .node_instances
                    .get(&root.storage_id)
            );
            assert_eq!(
                reopened
                    .registry()
                    .node_instances
                    .get(&target.storage_id)
                    .and_then(|node| node.attributes().get_typed::<String>(DISPLAY_NAME))
                    .as_deref(),
                Some("Nested only")
            );
            reopened.undo().expect("undo one rename");
            assert!(
                reopened
                    .registry()
                    .value_equal(original.document.registry())
            );
            reopened.redo().expect("redo one rename");
            assert!(reopened.registry().value_equal(renamed.document.registry()));
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_033_1_preserves_pinned_property_order_in_the_addressed_network() {
        futures::executor::block_on(async {
            let owner = document_with_nested_graph_nodes();
            let bytes = export(&owner, None).await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original nested graph");
            let path = [NodeId(24)];
            let nodes = original.graph_nodes().await.expect("source nodes");
            let first = nodes
                .iter()
                .find(|node| node.network_path == path && node.local_id == NodeId(9))
                .expect("first pin target");
            let first_copy = original
                .set_graph_node_pinned_in_copy(first, true, budget())
                .await
                .expect("pin first nested node");
            assert!(matches!(
                first_copy
                    .set_graph_node_pinned_in_copy(first, false, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));
            let targets = first_copy
                .graph_nodes()
                .await
                .expect("fresh snapshot targets");
            let second = targets
                .iter()
                .find(|node| node.network_path == path && node.local_id == NodeId(7))
                .expect("second pin target");
            let both = first_copy
                .set_graph_node_pinned_in_copy(second, true, budget())
                .await
                .expect("pin second nested node");
            let interface = both
                .runtime_graph()
                .await
                .expect("rebuild source graph")
                .into_interface()
                .expect("pinned properties metadata");
            assert_eq!(
                interface.ordered_pinned_nodes(&path),
                [NodeId(9), NodeId(7)]
            );
            assert!(interface.ordered_pinned_nodes(&[]).is_empty());
            assert_eq!(both.history_len(), first_copy.history_len() + 2);

            let working = both
                .copy_entries(&budget())
                .expect("reopen copy")
                .finish()
                .expect("valid layout")
                .working;
            let mut editor = GddV1::open_in(AnyContainer::Memory(working), GddV1Layout)
                .await
                .expect("original editor");
            editor
                .undo()
                .expect("one-step undo for both stored attributes");
            assert!(
                editor
                    .registry()
                    .value_equal(first_copy.document.registry())
            );
            editor
                .redo()
                .expect("one-step redo for both stored attributes");
            assert!(editor.registry().value_equal(both.document.registry()));

            let nodes = both.graph_nodes().await.expect("both pinned nodes");
            let remove = nodes
                .iter()
                .find(|node| node.network_path == path && node.local_id == NodeId(9))
                .expect("unpin the first node");
            assert!(remove.is_pinned());
            assert!(matches!(
                both.set_graph_node_pinned_in_copy(remove, true, budget())
                    .await,
                Err(GddArchiveError::UnchangedGraphNodePin)
            ));
            let unpinned = both
                .set_graph_node_pinned_in_copy(remove, false, budget())
                .await
                .expect("unpin without moving the other node");
            let interface = unpinned
                .runtime_graph()
                .await
                .expect("rebuild unpinned graph")
                .into_interface()
                .expect("unpinned properties metadata");
            assert_eq!(interface.ordered_pinned_nodes(&path), [NodeId(7)]);
            assert_eq!(original.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_graph_034_1_moves_pinned_nodes_in_source_order_with_one_history_step() {
        futures::executor::block_on(async {
            let bytes = export(&document_with_nested_graph_nodes(), None).await;
            let mut pinned = open_gdd_zip(&bytes, budget())
                .await
                .expect("original ZIP")
                .into_inspected();
            let path = [NodeId(24)];
            for local_id in [NodeId(9), NodeId(7)] {
                let nodes = pinned.graph_nodes().await.expect("source nodes");
                let target = nodes
                    .iter()
                    .find(|node| node.network_path == path && node.local_id == local_id)
                    .expect("nested node");
                pinned = pinned
                    .set_graph_node_pinned_in_copy(target, true, budget())
                    .await
                    .expect("pinned copy");
            }
            let nodes = pinned.graph_nodes().await.expect("pinned nodes");
            let first = nodes
                .iter()
                .find(|node| node.pinned_position() == Some((1, 2)))
                .expect("first pinned node");
            let root = nodes
                .iter()
                .find(|node| node.network_path.is_empty())
                .expect("root node");
            for target in [first, root] {
                assert!(matches!(
                    pinned
                        .move_pinned_graph_node_in_copy(target, GddPinnedNodeMove::Up, budget())
                        .await,
                    Err(GddArchiveError::InvalidPinnedNodeMove)
                ));
            }

            let reordered = pinned
                .move_pinned_graph_node_in_copy(first, GddPinnedNodeMove::Down, budget())
                .await
                .expect("move first node down");
            assert_eq!(reordered.history_len(), pinned.history_len() + 1);
            assert_eq!(
                reordered.document.registry().node_instances,
                pinned.document.registry().node_instances
            );
            let interface = reordered
                .runtime_graph()
                .await
                .expect("read registry")
                .into_interface()
                .expect("source network");
            assert_eq!(
                interface.ordered_pinned_nodes(&path),
                [NodeId(7), NodeId(9)]
            );
            assert!(interface.ordered_pinned_nodes(&[]).is_empty());
            assert!(matches!(
                reordered
                    .move_pinned_graph_node_in_copy(first, GddPinnedNodeMove::Up, budget())
                    .await,
                Err(GddArchiveError::StaleGraphNode)
            ));

            let undone = reordered
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("one undo");
            assert!(
                undone
                    .document
                    .registry()
                    .value_equal(pinned.document.registry())
            );
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("one redo");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(reordered.document.registry())
            );

            let nodes = reordered.graph_nodes().await.expect("reordered nodes");
            let last = nodes
                .iter()
                .find(|node| node.pinned_position() == Some((2, 2)))
                .expect("moved node");
            assert_eq!(last.storage_id, first.storage_id);
            assert!(matches!(
                reordered
                    .move_pinned_graph_node_in_copy(last, GddPinnedNodeMove::Down, budget())
                    .await,
                Err(GddArchiveError::InvalidPinnedNodeMove)
            ));
            let restored = reordered
                .move_pinned_graph_node_in_copy(last, GddPinnedNodeMove::Up, budget())
                .await
                .expect("move the same node back up");
            assert!(
                restored
                    .document
                    .registry()
                    .value_equal(pinned.document.registry())
            );
        });
    }

    #[test]
    fn at_gra_gdd_017_018_restores_peer_and_nested_network_view_settings() {
        use graphite_document_graph_storage::{NoMetadata, ROOT_NETWORK, attr::session, to_value};
        use graphite_glam::DVec2;
        use graphite_graph_craft::document::{
            DocumentNode, DocumentNodeImplementation, NodeNetwork,
        };
        use graphite_resource::ResourceRegistry;
        use std::collections::BTreeMap;

        futures::executor::block_on(async {
            let mut owner = document();
            let nested_node = NodeId(17);
            let root_network = NodeNetwork {
                nodes: [(
                    nested_node,
                    DocumentNode {
                        implementation: DocumentNodeImplementation::Network(NodeNetwork::default()),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            };
            owner
                .commit_from_runtime(
                    &root_network,
                    &NoMetadata,
                    &ResourceRegistry::new(),
                    &HashMapResourceStorage::new(),
                )
                .expect("commit the nested pinned network");
            let ids = owner
                .network_ids(&root_network, &NoMetadata)
                .expect("stable storage IDs");
            let nested_id = *ids.get(&vec![nested_node]).expect("nested storage ID");
            let mut artwork_camera = PTZ::default();
            artwork_camera.pan = DVec2::new(12., -24.);
            artwork_camera.set_zoom(1.5);
            owner
                .set_view_settings(BTreeMap::from([
                    (
                        session::doc::PTZ.into(),
                        to_value(&artwork_camera).expect("PTZ"),
                    ),
                    (
                        session::doc::RULERS_VISIBLE.into(),
                        to_value(&false).expect("rulers"),
                    ),
                ]))
                .expect("persist peer view settings");
            let mut nested_camera = PTZ::default();
            nested_camera.pan = DVec2::new(-310., 640.);
            nested_camera.set_zoom(0.75);
            let previewing = Previewing::Yes {
                root_node_to_restore: None,
            };
            owner
                .set_network_view_settings(BTreeMap::from([
                    (
                        ROOT_NETWORK,
                        BTreeMap::from([(
                            session::network::NAV_WIDTH.into(),
                            to_value(&700.).expect("root width"),
                        )]),
                    ),
                    (
                        nested_id,
                        BTreeMap::from([
                            (
                                session::network::NAV_PTZ.into(),
                                to_value(&nested_camera).expect("nested PTZ"),
                            ),
                            (
                                session::network::PREVIEWING.into(),
                                to_value(&previewing).expect("preview state"),
                            ),
                        ]),
                    ),
                    (
                        NetworkId(u64::MAX),
                        BTreeMap::from([(
                            session::network::NAV_WIDTH.into(),
                            to_value(&300.).expect("stale width"),
                        )]),
                    ),
                ]))
                .expect("persist root and nested view settings");
            let source_bytes = export(&owner, None).await;
            let source = open_gdd_zip(&source_bytes, budget())
                .await
                .expect("reopen pinned export");
            let summary = source.recovery();
            let view = source
                .saved_view()
                .await
                .expect("restore pinned view state");
            let artwork = view.document.expect("per-peer artwork settings");
            assert_eq!(artwork.camera, artwork_camera);
            assert!(!artwork.rulers_visible);
            let [root, nested] = view.networks.as_slice() else {
                panic!("expected only the two attached graph views");
            };
            assert_eq!(root.id, ROOT_NETWORK);
            assert!(root.path.is_empty());
            assert_eq!(root.navigation.node_graph_width, 700.);
            assert_eq!(root.previewing, Previewing::No);
            assert_eq!(nested.id, nested_id);
            assert_eq!(nested.path, [nested_node]);
            assert_eq!(nested.navigation.node_graph_ptz, nested_camera);
            assert_eq!(nested.previewing, previewing);
            assert_eq!(view.unresolved_networks, 1);
            assert_eq!(source.recovery(), summary);
            assert!(source.document.registry().value_equal(owner.registry()));
            assert_eq!(source.original_bytes(), source_bytes.as_ref());
        });
    }

    fn document_with_pending_network() -> (GddV1, NetworkId) {
        use graphite_document_graph_storage::{HotOp, Network, NetworkId, ROOT_NETWORK, TimeStamp};

        let mut owner = document();
        let pending_network = NetworkId(42);
        let peer = owner.session().peer();
        for (id, counter) in [(ROOT_NETWORK, 1), (pending_network, 2)] {
            owner
                .apply_hot_op(HotOp {
                    op: RegistryDelta::AddNetwork {
                        id,
                        network: Network::default(),
                    },
                    timestamp: TimeStamp { counter, peer },
                })
                .expect("source graph operation");
            if id == ROOT_NETWORK {
                owner
                    .retire_pending_interaction()
                    .expect("one retired interaction");
            }
        }
        (owner, pending_network)
    }

    #[test]
    fn at_gra_hist_003_004_moves_saved_gdd_history_with_stable_nodes_and_resources() {
        use graphite_document_graph_storage::{HotOp, Network, TimeStamp};

        futures::executor::block_on(async {
            let (mut owner, _) = document_with_pending_network();
            let initial = export(&owner, None).await;
            let pending = open_gdd_zip(&initial, budget())
                .await
                .expect("open pending snapshot");
            assert!(matches!(
                pending
                    .move_history_in_copy(GddHistoryDirection::Undo, budget())
                    .await,
                Err(GddArchiveError::PendingGraphEdits)
            ));
            owner
                .retire_pending_interaction()
                .expect("finish the prior interaction");
            let prior = owner.registry().clone();
            let resource_id = ResourceId::from(23_u64);
            let payload = b"retained source data through history";
            let hash = ResourceHash::from(payload.as_slice());
            owner
                .add_resource(resource_id, payload)
                .expect("add a real resource");
            let network_id = NetworkId(51);
            owner
                .apply_hot_op(HotOp {
                    op: RegistryDelta::AddNetwork {
                        id: network_id,
                        network: Network::default(),
                    },
                    timestamp: TimeStamp {
                        counter: 100,
                        peer: owner.session().peer(),
                    },
                })
                .expect("add a graph network");
            owner
                .retire_pending_interaction()
                .expect("retire resource and graph together");
            let bytes = export(&owner, Some(b"original opaque legacy data")).await;
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("open saved source");
            assert!(original.can_move_history(GddHistoryDirection::Undo));
            assert!(!original.can_move_history(GddHistoryDirection::Redo));
            let layout = GddV1Layout;
            let history = path_for(layout.history_basename(), owner.manifest().codecs.history);
            let resource = layout.resource_path(&hash);

            let undone = original
                .move_history_in_copy(GddHistoryDirection::Undo, budget())
                .await
                .expect("undo the saved interaction in a fresh folder");
            assert!(undone.document.registry().value_equal(&prior));
            assert!(
                !undone
                    .document
                    .registry()
                    .networks
                    .contains_key(&network_id)
            );
            assert!(
                !undone
                    .document
                    .registry()
                    .resources
                    .contains_key(&resource_id)
            );
            assert!(undone.can_move_history(GddHistoryDirection::Redo));
            let redone = undone
                .move_history_in_copy(GddHistoryDirection::Redo, budget())
                .await
                .expect("redo the saved interaction in a fresh folder");
            assert!(
                redone
                    .document
                    .registry()
                    .value_equal(original.document.registry())
            );
            assert_eq!(
                redone.document.session().head_rev(),
                original.document.session().head_rev()
            );
            assert!(
                redone
                    .document
                    .registry()
                    .networks
                    .contains_key(&network_id)
            );
            assert_eq!(
                redone
                    .document
                    .registry()
                    .resources
                    .get(&resource_id)
                    .and_then(|entry| entry.hash),
                Some(hash)
            );
            for copy in [&undone, &redone] {
                assert_eq!(
                    copy.read_entry(&history)
                        .expect("retained history")
                        .as_slice(),
                    original
                        .read_entry(&history)
                        .expect("original history")
                        .as_slice()
                );
                assert_eq!(
                    copy.read_entry(&resource)
                        .expect("retained resource bytes")
                        .as_slice(),
                    payload
                );
                assert_eq!(
                    copy.read_entry(layout.legacy_path())
                        .expect("retained legacy")
                        .as_slice(),
                    b"original opaque legacy data"
                );
            }
            assert_eq!(original.original_bytes(), bytes.as_ref());

            let head = owner.session().head_rev().expect("last authored revision");
            owner
                .publish_up_to(head)
                .expect("mark a source published frontier");
            let bytes = export(&owner, None).await;
            let published = open_gdd_zip(&bytes, budget())
                .await
                .expect("published source");
            assert!(!published.can_move_history(GddHistoryDirection::Undo));
            assert!(matches!(
                published
                    .move_history_in_copy(GddHistoryDirection::Undo, budget())
                    .await,
                Err(GddArchiveError::Storage(
                    graphite_document_format::Error::Crdt(
                        graphite_document_graph_storage::CrdtError::NothingToUndo
                    )
                ))
            ));
        });
    }

    #[test]
    fn at_gra_gdd_016_1_recovers_registry_history_and_hot_frames_with_the_original_owner() {
        use graphite_document_graph_storage::TimeStamp;

        futures::executor::block_on(async {
            let (owner, pending_network) = document_with_pending_network();
            let peer = owner.session().peer();
            for (source, include_registry, include_history) in [
                (GddRecoverySource::RegistryAndHistory, true, true),
                (GddRecoverySource::RegistryOnly, true, false),
                (GddRecoverySource::HistoryOnly, false, true),
            ] {
                let original: Arc<[u8]> = owner
                    .export_to_bytes(
                        ExportFormat::Zip,
                        ExportOptions {
                            include_registry,
                            include_history,
                            ..ExportOptions::default()
                        },
                        &HashMapResourceStorage::new(),
                        None,
                    )
                    .await
                    .expect("actual source export")
                    .into();
                let recovered = open_gdd_zip(&original, budget())
                    .await
                    .expect("recover the recorded route");
                let summary = recovered.recovery();
                assert_eq!(summary.source, source);
                assert_eq!(summary.network_count, 2);
                assert_eq!(summary.node_count, 0);
                assert_eq!(summary.hot_log_len, usize::from(include_history));
                assert!(recovered.document.registry().value_equal(owner.registry()));
                if include_history {
                    assert_eq!(summary.history_len, 1);
                    let [pending] = recovered.document.session().hot_log() else {
                        panic!("expected a single replayed operation");
                    };
                    assert_eq!(pending.timestamp, TimeStamp { counter: 2, peer });
                    assert!(matches!(
                        pending.op,
                        RegistryDelta::AddNetwork { id, .. } if id == pending_network
                    ));
                    assert!(
                        !recovered
                            .document
                            .session()
                            .retired_registry()
                            .networks
                            .contains_key(&pending_network)
                    );
                }
                assert_eq!(recovered.original_bytes(), original.as_ref());
            }
        });
    }

    #[test]
    fn at_gra_gdd_016_1_saves_pending_edits_as_one_undoable_interaction() {
        use graphite_document_graph_storage::ROOT_NETWORK;

        futures::executor::block_on(async {
            let (owner, pending_network) = document_with_pending_network();
            let bytes = export(&owner, Some(b"original legacy data")).await;
            let source = open_gdd_zip(&bytes, budget())
                .await
                .expect("original GDD with an unfinished edit");
            let layout = GddV1Layout;
            let codecs = owner.manifest().codecs;
            let history_path = path_for(layout.history_basename(), codecs.history);
            let hot_log_path = path_for(layout.hot_log_basename(), codecs.hot_log);
            let source_files = source.folder_files().expect("exact working entries");
            for retain_history in [true, false] {
                let files = source_files
                    .iter()
                    .filter(|(path, _)| retain_history || path != Path::new(&history_path))
                    .map(|(path, bytes)| {
                        (
                            path.to_str().expect("admitted path").to_owned(),
                            bytes.clone(),
                        )
                    })
                    .collect();
                let original = inspect_gdd_folder_files(files, budget())
                    .await
                    .expect("source copy for recovery");
                let recovery = original
                    .recover_pending_edits(budget())
                    .await
                    .expect("retire pending Graphite state in isolated storage");
                assert_eq!(recovery.history_len(), original.history_len() + 1);
                assert_eq!(recovery.hot_log_len(), 0);
                assert!(
                    recovery
                        .document
                        .registry()
                        .value_equal(original.document.registry())
                );
                assert_eq!(
                    recovery
                        .read_entry(layout.legacy_path())
                        .expect("retained legacy entry")
                        .as_slice(),
                    b"original legacy data",
                );
                assert_eq!(
                    original
                        .read_entry(&hot_log_path)
                        .expect("source hot log")
                        .as_slice(),
                    source
                        .read_entry(&hot_log_path)
                        .expect("original hot log")
                        .as_slice(),
                );
                let working = recovery
                    .copy_entries(&budget())
                    .expect("inspect durable output")
                    .finish()
                    .expect("valid folder layout")
                    .working;
                let mut original_editor = GddV1::open_in(AnyContainer::Memory(working), layout)
                    .await
                    .expect("original editor reopens the recovered working folder");
                original_editor
                    .undo()
                    .expect("one-step undo of the recovered edit");
                assert!(
                    original_editor
                        .registry()
                        .networks
                        .contains_key(&ROOT_NETWORK)
                );
                assert!(
                    !original_editor
                        .registry()
                        .networks
                        .contains_key(&pending_network)
                );
                original_editor
                    .redo()
                    .expect("one-step redo of the recovered edit");
                assert!(
                    original_editor
                        .registry()
                        .value_equal(original.document.registry())
                );
                assert!(matches!(
                    recovery.recover_pending_edits(budget()).await,
                    Err(GddArchiveError::NoPendingEdits)
                ));
            }
            assert_eq!(source.original_bytes(), bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_gdd_015_1_embeds_original_legacy_bytes_under_the_pinned_path() {
        use crate::formats::legacy::inspect_legacy_document;

        futures::executor::block_on(async {
            let fixture = DocumentMessageHandler::default();
            let legacy_bytes: Arc<[u8]> = format!("{}\n\n", fixture.serialize_document())
                .into_bytes()
                .into();
            let legacy = inspect_legacy_document(&legacy_bytes, budget().max_archive_bytes)
                .expect("actual legacy document");
            let source_bytes = export(&document(), None).await;
            let source = open_gdd_zip(&source_bytes, budget())
                .await
                .expect("original GDD");
            let path = GddV1Layout.legacy_path();
            let portable = source
                .with_export_attachments(GddExportContent::default(), None, Some(&legacy), budget())
                .await
                .expect("append the approved legacy document");
            for format in [ArchiveFormat::Zip, ArchiveFormat::Xz] {
                let bytes: Arc<[u8]> = portable
                    .export_archive(format, budget().max_archive_bytes)
                    .expect("write portable archive")
                    .into();
                let reopened = open_gdd_archive(&bytes, budget())
                    .await
                    .expect("reopen original container format");
                assert_eq!(
                    reopened.read_entry(path).expect("legacy bytes").as_slice(),
                    legacy_bytes.as_ref()
                );
                for name in &source.entries.names {
                    assert_eq!(
                        reopened.read_entry(name).expect("copied entry").as_slice(),
                        source.read_entry(name).expect("original entry").as_slice()
                    );
                }
            }
            let filtered = source
                .with_export_attachments(
                    GddExportContent {
                        include_registry: true,
                        include_history: false,
                    },
                    None,
                    Some(&legacy),
                    budget(),
                )
                .await
                .expect("registry-only export with explicitly supplied legacy data");
            assert_eq!(
                filtered.read_entry(path).expect("legacy bytes").as_slice(),
                legacy_bytes.as_ref()
            );
            assert!(matches!(
                source
                    .with_export_attachments(
                        GddExportContent::default(),
                        None,
                        Some(&legacy),
                        GddArchiveBudget {
                            max_entries: source.entries.names.len(),
                            ..budget()
                        },
                    )
                    .await,
                Err(GddArchiveError::BudgetExceeded)
            ));
            let different_bytes: Arc<[u8]> = fixture.serialize_document().into_bytes().into();
            let different = inspect_legacy_document(&different_bytes, budget().max_archive_bytes)
                .expect("valid, distinct original bytes");
            assert!(matches!(
                portable
                    .with_export_attachments(
                        GddExportContent::default(),
                        None,
                        Some(&different),
                        budget()
                    )
                    .await,
                Err(GddArchiveError::ConflictingLegacyDocument)
            ));
            assert!(source.read_entry(path).is_err());
            assert_eq!(source.original_bytes(), source_bytes.as_ref());
        });
    }

    #[test]
    fn at_gra_gdd_003_1_creates_original_folder_payloads_at_the_current_pin() {
        futures::executor::block_on(async {
            let contents = create_gdd_folder(92, 37, "Made".into(), budget())
                .await
                .expect("pinned GDD owner creates the first snapshot");
            let manifest = contents.manifest();
            assert_eq!(manifest.format_version, SUPPORTED_FORMAT_VERSION);
            assert_eq!(manifest.document_id, 92);
            assert_eq!(manifest.editor_version, "Made");
            assert_eq!(
                manifest.stdlib_version,
                crate::resources::GRAPHITE_IMAGE_SOURCE_REVISION
            );
            assert_eq!(contents.document.session().peer(), PeerId(37));
            assert_eq!(contents.history_len(), 0);
            assert_eq!(contents.hot_log_len(), 0);
            assert_eq!(contents.entries.names.len(), 3);
            assert!(
                contents
                    .read_entry(&path_for(
                        GddV1Layout.registry_basename(),
                        manifest.codecs.registry
                    ))
                    .is_ok()
            );
            assert!(
                contents
                    .read_entry(&path_for(
                        GddV1Layout.session_basename(),
                        manifest.codecs.session
                    ))
                    .is_ok()
            );
        });
    }

    #[test]
    fn at_gra_gdd_013_1_embeds_only_approved_matching_bytes_in_the_copy() {
        use graphite_document_graph_storage::{NoMetadata, to_value};
        use graphite_graph_craft::{
            ProtoNodeIdentifier,
            document::{
                DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
                value::TaggedValue,
            },
        };
        use graphite_resource::{DataSource, ResourceRegistry};

        futures::executor::block_on(async {
            let mut owner = document();
            let resource_id = ResourceId::from(71_u64);
            let approved = b"bytes of the linked artwork resource".to_vec();
            let hash = ResourceHash::from(approved.as_slice());
            let link = DataSource::Url(
                "https://example.invalid/resource.bin"
                    .parse()
                    .expect("fixture URL"),
            );
            let mut resources = ResourceRegistry::new();
            resources.push_source_back(&resource_id, link.clone());
            resources.resolve(&resource_id, hash);
            let network = NodeNetwork {
                nodes: [(
                    NodeId(0),
                    DocumentNode {
                        inputs: vec![NodeInput::value(TaggedValue::Resource(resource_id), false)],
                        implementation: DocumentNodeImplementation::ProtoNode(
                            ProtoNodeIdentifier::new("graphene_core::ops::identity::IdentityNode"),
                        ),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            };
            let source_store = HashMapResourceStorage::new();
            owner
                .commit_from_runtime(&network, &NoMetadata, &resources, &source_store)
                .expect("original source link ownership");
            let bytes = Arc::<[u8]>::from(
                owner
                    .export_to_bytes(
                        ExportFormat::Zip,
                        ExportOptions::default(),
                        &source_store,
                        Some(b"retained embedded legacy"),
                    )
                    .await
                    .expect("actual Graphite ZIP export"),
            );
            let zip = open_gdd_archive(&bytes, budget())
                .await
                .expect("original linked archive");
            let mut files = zip
                .folder_files()
                .expect("original files")
                .into_iter()
                .map(|(path, bytes)| {
                    (
                        path.into_os_string()
                            .into_string()
                            .expect("UTF-8 source paths"),
                        bytes,
                    )
                })
                .collect::<Vec<_>>();
            let manifest = files
                .iter_mut()
                .find(|(path, _)| path == "manifest.json")
                .expect("manifest");
            let mut raw_manifest: serde_json::Value =
                serde_json::from_slice(&manifest.1).expect("original manifest codec");
            raw_manifest
                .as_object_mut()
                .expect("manifest object")
                .insert("unknown_field".into(), serde_json::json!("retain this"));
            manifest.1 = serde_json::to_vec(&raw_manifest).expect("literal unrecognized metadata");
            let original_manifest = manifest.1.clone();
            files.push(("opaque/extras.bin".into(), vec![0, 1, 255]));
            let source = inspect_gdd_folder_files(files, budget())
                .await
                .expect("staged source with opaque records");
            assert_eq!(
                source.missing_linked_resources().expect("source link hash"),
                BTreeSet::from([hash])
            );
            let missing = source
                .with_embedded_linked_resources(GddExportContent::default(), Vec::new(), budget())
                .await;
            assert!(
                matches!(missing, Err(GddArchiveError::MissingApprovedResource(missing)) if missing == hash)
            );
            let unapproved = source
                .with_embedded_linked_resources(
                    GddExportContent::default(),
                    vec![b"different data".to_vec()],
                    budget(),
                )
                .await;
            assert!(matches!(
                unapproved,
                Err(GddArchiveError::UnexpectedApprovedResource(_))
            ));

            let copy = source
                .with_embedded_linked_resources(
                    GddExportContent::default(),
                    vec![approved.clone()],
                    budget(),
                )
                .await
                .expect("embed an explicitly matched file");
            let resource_file = GddV1Layout.resource_path(&hash);
            assert_eq!(
                copy.read_entry(&resource_file)
                    .expect("embedded bytes")
                    .as_slice(),
                approved
            );
            let resource = copy
                .document
                .registry()
                .resources
                .get(&resource_id)
                .expect("original resource identity retained");
            assert!(resource.has_embedded_source());
            let link = to_value(&link).expect("original URL fallback");
            assert!(
                resource
                    .sources
                    .iter()
                    .any(|(_, entry)| entry.source == link)
            );
            assert_eq!(
                copy.read_entry("manifest.json")
                    .expect("original manifest retained")
                    .as_slice(),
                original_manifest
            );
            assert_eq!(
                copy.read_entry("opaque/extras.bin")
                    .expect("opaque entry retained")
                    .as_slice(),
                [0, 1, 255]
            );
            assert_eq!(
                copy.read_entry(GddV1Layout.legacy_path())
                    .expect("legacy entry retained")
                    .as_slice(),
                b"retained embedded legacy"
            );
            assert!(source.read_entry(&resource_file).is_err());
            assert_eq!(
                source
                    .missing_linked_resources()
                    .expect("original links unchanged"),
                BTreeSet::from([hash])
            );
        });
    }

    #[test]
    fn at_gra_gdd_009_010_011_exports_registry_history_and_complete_outputs() {
        futures::executor::block_on(async {
            for codec in [Codec::Postcard, Codec::Json] {
                let mut owner = document();
                if codec == Codec::Json {
                    owner
                        .update_manifest(|manifest| {
                            manifest.codecs.registry = Codec::Json;
                            manifest.codecs.history = Codec::JsonLines;
                        })
                        .expect("original codec selection");
                }
                let resource_id = ResourceId::from(32_u64);
                let resource = b"resource in source history";
                let hash = ResourceHash::from(resource.as_slice());
                owner
                    .add_resource(resource_id, resource)
                    .expect("authentic history before export");
                let original = export(&owner, Some(b"source legacy contents")).await;
                let source = open_gdd_archive(&original, budget())
                    .await
                    .expect("actual source GDD");
                let codecs = source.manifest().codecs;
                let registry = path_for(GddV1Layout.registry_basename(), codecs.registry);
                let history = path_for(GddV1Layout.history_basename(), codecs.history);
                for (include_registry, include_history) in
                    [(true, false), (false, true), (true, true)]
                {
                    let content = GddExportContent {
                        include_registry,
                        include_history,
                    };
                    for format in [ArchiveFormat::Zip, ArchiveFormat::Xz] {
                        let bytes = Arc::<[u8]>::from(
                            source
                                .export_archive_with_content(format, content, budget())
                                .await
                                .expect("source-qualified content export"),
                        );
                        let exported = open_gdd_archive(&bytes, budget())
                            .await
                            .expect("reopen selected archive contents");
                        assert_eq!(exported.manifest().codecs.registry, codec);
                        assert_eq!(exported.read_entry(&registry).is_ok(), include_registry);
                        assert_eq!(exported.read_entry(&history).is_ok(), include_history);
                        assert_eq!(
                            exported.read_entry(GddV1Layout.legacy_path()).is_ok(),
                            include_registry && include_history
                        );
                        assert_eq!(
                            exported
                                .document
                                .registry()
                                .resources
                                .get(&resource_id)
                                .and_then(|entry| entry.hash),
                            Some(hash)
                        );
                        assert_eq!(
                            exported
                                .read_entry(&GddV1Layout.resource_path(&hash))
                                .expect("retained embedded resource")
                                .as_slice(),
                            resource
                        );
                    }
                    let folder = source
                        .export_folder_files_with_content(content, budget())
                        .await
                        .expect("export folder contents");
                    assert_eq!(
                        folder.iter().any(|(path, _)| path == Path::new(&registry)),
                        include_registry
                    );
                    assert_eq!(
                        folder.iter().any(|(path, _)| path == Path::new(&history)),
                        include_history
                    );
                }
                assert_eq!(source.original_bytes(), original.as_ref());
                assert!(
                    source.read_entry(&registry).is_ok() && source.read_entry(&history).is_ok()
                );
            }
        });
    }

    #[test]
    fn at_gra_gdd_012_rejects_no_contents_and_an_empty_history_export() {
        futures::executor::block_on(async {
            let original = export(&document(), None).await;
            let source = open_gdd_archive(&original, budget())
                .await
                .expect("valid new document");
            let none = GddExportContent {
                include_registry: false,
                include_history: false,
            };
            let empty_history = GddExportContent {
                include_registry: false,
                include_history: true,
            };
            assert!(matches!(
                source
                    .export_archive_with_content(ArchiveFormat::Zip, none, budget())
                    .await,
                Err(GddArchiveError::InvalidExportOptions(_))
            ));
            assert!(matches!(
                source
                    .export_folder_files_with_content(none, budget())
                    .await,
                Err(GddArchiveError::InvalidExportOptions(_))
            ));
            assert!(matches!(
                source
                    .export_archive_with_content(ArchiveFormat::Zip, empty_history, budget())
                    .await,
                Err(GddArchiveError::NoExportableHistory)
            ));
            assert_eq!(source.original_bytes(), original.as_ref());
        });
    }

    #[test]
    fn at_gra_gdd_007_and_008_exports_every_entry_and_reopens_with_both_original_codecs() {
        futures::executor::block_on(async {
            for codec in [Codec::Postcard, Codec::Json] {
                let mut original = document();
                original
                    .add_resource(ResourceId::from(79_u64), b"original image bytes")
                    .expect("actual resource and its history");
                if codec == Codec::Json {
                    original
                        .update_manifest(|manifest| {
                            manifest.codecs.registry = Codec::Json;
                            manifest.codecs.history = Codec::JsonLines;
                        })
                        .expect("original codecs");
                }
                let legacy = DocumentMessageHandler::default().serialize_document();
                let original_bytes = export(&original, Some(legacy.as_bytes())).await;
                let source = open_gdd_zip(&original_bytes, budget())
                    .await
                    .expect("pinned ZIP");
                let mut fixture = Zip::writer(Cursor::new(Vec::new())).expect("pinned writer");
                for path in &source.entries.names {
                    fixture
                        .write_entry(
                            path,
                            source.read_entry(path).expect("source entry").as_slice(),
                        )
                        .expect("write raw source entry");
                }
                fixture
                    .write_entry("opaque/extension/state.bin", &[0, 213, 2])
                    .expect("unknown entry");
                let source_bytes = Arc::<[u8]>::from(
                    fixture
                        .finish_into()
                        .expect("valid original envelope")
                        .into_inner(),
                );
                let source = open_gdd_zip(&source_bytes, budget())
                    .await
                    .expect("admit full source");
                let entries = &source.entries.names;
                for format in [ArchiveFormat::Zip, ArchiveFormat::Xz] {
                    assert!(source.export_archive(format, 5).is_err());
                    let bytes = Arc::<[u8]>::from(
                        source
                            .export_archive(format, budget().max_archive_bytes)
                            .expect("export complete source copy"),
                    );
                    assert_eq!(ArchiveFormat::detect(&bytes), Some(format));
                    let copy = open_gdd_archive(&bytes, budget())
                        .await
                        .expect("reopen native export");
                    let pinned = GddV1::open_from_archive(
                        &bytes,
                        AnyContainer::Memory(MemoryBackend::new()),
                        GddV1Layout,
                    )
                    .await
                    .expect("original Graphite reader opens the export");
                    assert_eq!(copy.manifest().codecs.registry, codec);
                    assert_eq!(pinned.manifest().codecs.registry, codec);
                    assert_eq!(copy.history_len(), source.history_len());
                    assert_eq!(pinned.session().history().count(), source.history_len());
                    assert_eq!(&copy.entries.names, entries);
                    for path in entries {
                        assert_eq!(
                            copy.read_entry(path).expect("copy entry").as_slice(),
                            source.read_entry(path).expect("source entry").as_slice(),
                            "{path}"
                        );
                    }
                    assert_eq!(source.original_bytes(), source_bytes.as_ref());
                }
            }
        });
    }

    #[test]
    fn at_r06_2_rejects_a_file_that_is_also_an_archive_parent() {
        futures::executor::block_on(async {
            let mut writer = Zip::writer(Cursor::new(Vec::new())).expect("pinned ZIP writer");
            for path in ["opaque", "opaque-annotation", "opaque/child.bin"] {
                writer.write_entry(path, b"fixture").expect("source entry");
            }
            let bytes = Arc::<[u8]>::from(writer.finish_into().expect("ZIP envelope").into_inner());
            assert!(matches!(
                open_gdd_archive(&bytes, budget()).await,
                Err(GddArchiveError::InconsistentEntries)
            ));
        });
    }

    #[test]
    fn at_gra_gdd_005_1_opens_the_pinned_xz_archive_with_resources_and_history() {
        futures::executor::block_on(async {
            let mut source = document();
            let resource_id = ResourceId::from(19_u64);
            let resource = b"resource in the original compressed form";
            let hash = ResourceHash::from(resource.as_slice());
            source
                .add_resource(resource_id, resource)
                .expect("actual Graphite resource history");
            let legacy = DocumentMessageHandler::default().serialize_document();
            let bytes = Arc::<[u8]>::from(
                source
                    .export_to_bytes(
                        ExportFormat::Xz,
                        ExportOptions::default(),
                        &HashMapResourceStorage::new(),
                        Some(legacy.as_bytes()),
                    )
                    .await
                    .expect("original Graphite XZ export"),
            );
            let opened = open_gdd_archive(&bytes, budget())
                .await
                .expect("open source-authored XZ");
            assert_eq!(opened.original_bytes(), bytes.as_ref());
            assert_eq!(opened.manifest().document_id, 0x459);
            assert_eq!(opened.history_len(), source.session().history().count());
            assert_eq!(
                opened
                    .read_entry(&GddV1Layout.resource_path(&hash))
                    .expect("literal resource contents")
                    .as_slice(),
                resource
            );
            assert_eq!(
                opened
                    .read_entry(GddV1Layout.legacy_path())
                    .expect("original legacy contents")
                    .as_slice(),
                legacy.as_bytes()
            );
        });
    }

    #[test]
    fn at_gra_gdd_005_1_preserves_xz_failures_without_installing_entries() {
        futures::executor::block_on(async {
            let mut writer = Xz::writer(Cursor::new(Vec::new())).expect("pinned XZ writer");
            writer
                .write_entry("notes.txt", b"first")
                .expect("first record");
            writer
                .write_entry("notes.txt", b"second")
                .expect("duplicate record");
            let raw = writer
                .finish_into()
                .expect("complete XZ stream")
                .into_inner();
            let bytes = Arc::<[u8]>::from(raw.as_slice());
            assert!(matches!(
                open_gdd_archive(&bytes, budget()).await,
                Err(GddArchiveError::InconsistentEntries)
            ));
            assert_eq!(bytes.as_ref(), raw);

            let shortened = Arc::<[u8]>::from(&raw[..raw.len() - 1]);
            assert!(matches!(
                open_gdd_archive(&shortened, budget()).await,
                Err(GddArchiveError::InconsistentEntries)
            ));

            let mut tight = budget();
            tight.max_archive_bytes = 1;
            assert!(matches!(
                open_gdd_archive(&bytes, tight).await,
                Err(GddArchiveError::BudgetExceeded)
            ));
            let mut tight = budget();
            tight.max_entry_bytes = 4;
            assert!(matches!(
                open_gdd_archive(&bytes, tight).await,
                Err(GddArchiveError::BudgetExceeded)
            ));
        });
    }

    #[test]
    fn at_r06_2_opens_pinned_zip_with_recorded_codecs_history_and_resources() {
        futures::executor::block_on(async {
            for codec in [Codec::Postcard, Codec::Json] {
                let mut original = document();
                let resource_id = ResourceId::from(17_u64);
                let resource_bytes = b"persistent Graphite resource";
                let hash = ResourceHash::from(resource_bytes.as_slice());
                original
                    .add_resource(resource_id, resource_bytes)
                    .expect("commit the actual Graphite resource to history");
                if codec == Codec::Json {
                    original
                        .update_manifest(|manifest| {
                            manifest.codecs.registry = Codec::Json;
                            manifest.codecs.history = Codec::JsonLines;
                        })
                        .expect("configure the source-defined JSON layout");
                }
                let legacy = DocumentMessageHandler::default().serialize_document();
                let bytes = export(&original, Some(legacy.as_bytes())).await;
                let opened = open_gdd_zip(&bytes, budget())
                    .await
                    .expect("read the native source container");
                assert_eq!(opened.original_bytes(), bytes.as_ref());
                assert_eq!(opened.manifest().document_id, 0x459);
                assert_eq!(opened.manifest().codecs.registry, codec);
                assert!(opened.history_len() > 0);
                assert_eq!(opened.history_len(), original.session().history().count());
                assert_eq!(opened.hot_log_len(), original.session().hot_log().len());
                assert_eq!(
                    opened
                        .document
                        .registry()
                        .resources
                        .get(&resource_id)
                        .and_then(|entry| entry.hash),
                    Some(hash)
                );
                assert_eq!(
                    opened
                        .read_entry(&GddV1Layout.resource_path(&hash))
                        .expect("embedded bytes")
                        .as_slice(),
                    resource_bytes
                );
                assert_eq!(
                    opened
                        .read_entry(GddV1Layout.legacy_path())
                        .expect("embedded legacy snapshot")
                        .as_slice(),
                    legacy.as_bytes()
                );
            }
        });
    }

    #[test]
    fn at_r06_2_rejects_future_manifest_and_unreadable_archives_without_changing_original() {
        futures::executor::block_on(async {
            let mut original = document();
            original
                .update_manifest(|manifest| manifest.format_version = SUPPORTED_FORMAT_VERSION + 1)
                .expect("write a future manifest through the original owner");
            let bytes = export(&original, None).await;
            assert!(matches!(
                open_gdd_zip(&bytes, budget()).await,
                Err(GddArchiveError::UnsupportedManifest { format, version })
                    if format == FORMAT_MAGIC && version == SUPPORTED_FORMAT_VERSION + 1
            ));
            let truncated: Arc<[u8]> = bytes[..bytes.len() - 1].into();
            assert!(matches!(
                open_gdd_zip(&truncated, budget()).await,
                Err(GddArchiveError::UnsupportedEnvelope)
            ));
            let mut tight = budget();
            tight.max_entry_bytes = 1;
            assert!(matches!(
                open_gdd_zip(&bytes, tight).await,
                Err(GddArchiveError::BudgetExceeded)
            ));
            let restored: Arc<[u8]> = export(&original, None).await;
            assert_eq!(bytes, restored);
        });
    }

    #[test]
    fn at_r06_2_preserves_unknown_entries_and_rejects_duplicate_manifest_fields() {
        futures::executor::block_on(async {
            let preserved = |manifest: &[u8]| {
                let mut writer =
                    Zip::writer(Cursor::new(Vec::new())).expect("pinned archive writer");
                writer
                    .write_entry("manifest.json", manifest)
                    .expect("manifest");
                let document = document();
                let registry = Codec::Postcard
                    .write_single(document.registry())
                    .expect("pinned registry codec");
                writer
                    .write_entry("registry.bin", &registry)
                    .expect("registry");
                writer
                    .write_entry("future/opaque.bin", &[45, 0, 192])
                    .expect("opaque optional entry");
                Arc::<[u8]>::from(writer.finish_into().expect("complete archive").into_inner())
            };
            let source = document();
            let manifest = MANIFEST_CODEC
                .write_single(source.manifest())
                .expect("actual Graphite manifest");
            let bytes = preserved(&manifest);
            let opened = open_gdd_zip(&bytes, budget())
                .await
                .expect("open valid V1 storage");
            assert_eq!(opened.original_bytes(), bytes.as_ref());
            assert_eq!(
                opened
                    .read_entry("future/opaque.bin")
                    .expect("unknown bytes are retained")
                    .as_slice(),
                &[45, 0, 192]
            );
            let mut duplicate = br#"{"format_version":999,"#.to_vec();
            duplicate.extend_from_slice(&manifest[1..]);
            assert!(matches!(
                open_gdd_zip(&preserved(&duplicate), budget()).await,
                Err(GddArchiveError::Manifest(error)) if error.to_string().contains("duplicate object key")
            ));
        });
    }

    #[test]
    fn at_gra_gdd_014_1_requires_embedded_resources_for_saved_redo() {
        use graphite_document_graph_storage::NoMetadata;
        use graphite_graph_craft::document::NodeNetwork;
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
            let mut source = document();
            source
                .commit_from_runtime(
                    &NodeNetwork::default(),
                    &NoMetadata,
                    &ResourceRegistry::new(),
                    &HashMapResourceStorage::new(),
                )
                .expect("commit the original mount interaction");
            let id = ResourceId::from(35_u64);
            let resource = b"bytes needed to restore the undone interaction";
            let hash = ResourceHash::from(resource.as_slice());
            source
                .add_resource(id, resource)
                .expect("record embedded source bytes");
            source
                .undo()
                .expect("move the pinned cursor before the resource interaction");
            assert!(!source.registry().resources.contains_key(&id));
            let complete = export(&source, None).await;
            let captured = open_gdd_archive(&complete, budget())
                .await
                .expect("history resources are intact");
            assert!(!captured.document.registry().resources.contains_key(&id));
            assert_eq!(
                captured
                    .read_entry(&GddV1Layout.resource_path(&hash))
                    .expect("preserved historical bytes")
                    .as_slice(),
                resource
            );
            let mut original_reader = GddV1::open_from_archive(
                &complete,
                AnyContainer::Memory(MemoryBackend::new()),
                GddV1Layout,
            )
            .await
            .expect("reopen with original Graphite storage");
            original_reader
                .redo()
                .expect("redo using the retained history");
            assert_eq!(
                original_reader
                    .registry()
                    .resources
                    .get(&id)
                    .and_then(|entry| entry.hash),
                Some(hash)
            );
            assert!(original_reader.has_resource(&hash).await);

            source
                .remove_resource(&hash)
                .expect("construct a missing historical resource fixture");
            let missing = export(&source, None).await;
            assert!(
                matches!(open_gdd_archive(&missing, budget()).await, Err(GddArchiveError::InvalidResource(actual)) if actual == hash)
            );
            assert_eq!(complete, captured.original_bytes);
        });
    }

    #[test]
    fn at_r06_2_rejects_missing_and_mismatched_embedded_resources() {
        futures::executor::block_on(async {
            let mut source = document();
            let resource = b"actual resource";
            let hash = ResourceHash::from(resource.as_slice());
            source
                .add_resource(ResourceId::from(18_u64), resource)
                .expect("actual Graphite registry entry");
            let path = GddV1Layout.resource_path(&hash);
            let manifest = MANIFEST_CODEC
                .write_single(source.manifest())
                .expect("actual manifest");
            let registry = Codec::Postcard
                .write_single(source.registry())
                .expect("actual registry");
            for replacement in [None, Some(b"wrong contents".as_slice())] {
                let mut writer =
                    Zip::writer(Cursor::new(Vec::new())).expect("pinned archive writer");
                writer
                    .write_entry("manifest.json", &manifest)
                    .expect("manifest");
                writer
                    .write_entry("registry.bin", &registry)
                    .expect("registry");
                if let Some(replacement) = replacement {
                    writer
                        .write_entry(&path, replacement)
                        .expect("mismatched resource file");
                }
                let bytes =
                    Arc::<[u8]>::from(writer.finish_into().expect("complete archive").into_inner());
                assert!(matches!(
                    open_gdd_zip(&bytes, budget()).await,
                    Err(GddArchiveError::InvalidResource(actual)) if actual == hash
                ));
            }
        });
    }
}
