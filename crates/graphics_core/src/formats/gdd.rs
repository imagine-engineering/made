use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    ops::{Bound, Deref},
    path::PathBuf,
    sync::Arc,
};

use graphite_core_types::{NodeParameter, ProtoNodeIdentifier};
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
        misc::PTZ,
        network_interface::{
            InputConnector, NavigationMetadata, NodeNetworkInterface, NodeTemplate,
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
use graphite_glam::{DVec2, IVec2};
use graphite_graph_craft::application_io::resource::{
    HashMapResourceStorage, ResourceHash, ResourceStorage,
};
use graphite_graph_craft::{
    document::{
        DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
        value::TaggedValue,
    },
    runtime_delta::RuntimeDelta,
};
use zip::ZipArchive;

use super::{OriginalJson, legacy::PreservedLegacyDocument};
use crate::evaluate::{
    UnaryNumericOperation, evaluate_detached_numeric_network, typecheck_detached_numeric_network,
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
    #[error("Select an ordinary numeric output in the same Graphite network and an empty export.")]
    UnsupportedGraphExportConnection,
    #[error("Graphite could not resolve the selected export type. {0}")]
    GraphExportTypeMismatch(String),
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
    #[error("Graphite could not allocate an unused node ID.")]
    GraphNodeIdentityCollision,
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
    empty_exports: Vec<usize>,
}

impl GddGraphNetwork {
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

    pub fn empty_exports(&self) -> &[usize] {
        &self.empty_exports
    }

    pub fn can_connect_export_from(&self, node: &GddGraphNode) -> bool {
        self.can_extend_ports
            && self.id == node.network_id
            && Arc::ptr_eq(&self.snapshot, &node.snapshot)
            && node.can_supply_numeric_export
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

#[derive(Clone)]
pub struct GddGraphNode {
    snapshot: Arc<()>,
    storage_id: StorageNodeId,
    local_id: NodeId,
    network_path: Vec<NodeId>,
    network_id: NetworkId,
    nested_network: Option<NetworkId>,
    label: String,
    name: String,
    pinned: bool,
    pinned_order: Option<(usize, usize)>,
    numeric_input: Option<GddNumericInput>,
    can_delete_in_copy: bool,
    can_duplicate_in_copy: bool,
    can_preview_numeric_output: bool,
    can_supply_numeric_export: bool,
    grid_position: Option<[i32; 2]>,
}

#[derive(Clone)]
enum GddNumericInput {
    Literal(GddNumericPort),
    Connected { index: usize, upstream: NodeId },
}

impl GddNumericInput {
    fn index(&self) -> usize {
        match self {
            Self::Literal(port) => port.index,
            Self::Connected { index, .. } => *index,
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

impl GddNumericValue {
    pub fn parse_like(&self, text: &str) -> Result<Self, GddArchiveError> {
        let text = text.trim();
        let finite_number = |text: &str| {
            text.trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
        };
        match self {
            Self::Number(_) => finite_number(text)
                .map(Self::Number)
                .ok_or(GddArchiveError::InvalidGraphNumber),
            Self::Integer(_) => text
                .parse::<i64>()
                .map(Self::Integer)
                .map_err(|_| GddArchiveError::InvalidGraphInteger),
            Self::Vector(_) => {
                let (first, second) = text
                    .split_once(',')
                    .ok_or(GddArchiveError::InvalidGraphVector)?;
                let first = finite_number(first).ok_or(GddArchiveError::InvalidGraphVector)?;
                let second = finite_number(second).ok_or(GddArchiveError::InvalidGraphVector)?;
                Ok(Self::Vector([first, second]))
            }
            Self::NumberList(_) if text.is_empty() => Ok(Self::NumberList(Arc::from([]))),
            Self::NumberList(_) => text
                .split(',')
                .map(|part| finite_number(part).ok_or(GddArchiveError::InvalidGraphNumberList))
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
    pub fn network_id(&self) -> NetworkId {
        self.network_id
    }

    pub fn nested_network(&self) -> Option<NetworkId> {
        self.nested_network
    }

    pub fn label(&self) -> &str {
        &self.label
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

    pub fn has_numeric_connection(&self) -> bool {
        matches!(
            self.numeric_input.as_ref(),
            Some(GddNumericInput::Connected { .. })
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

    pub fn numeric_connection_source<'nodes>(&self, nodes: &'nodes [Self]) -> Option<&'nodes Self> {
        let Some(GddNumericInput::Connected { upstream, .. }) = self.numeric_input.as_ref() else {
            return None;
        };
        nodes
            .iter()
            .find(|node| self.same_network_as(node) && node.local_id == *upstream)
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
            GddNumericInput::Connected { .. } => None,
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

    pub async fn graph_inspection(&self) -> Result<GddGraphInspection, GddArchiveError> {
        let graph = self.runtime_graph().await?;
        let network_ids = network_ids_from_entries(&graph.networks);
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
                    name: node.display_name.clone().unwrap_or_default(),
                    pinned: node.pinned,
                    pinned_order: None,
                    numeric_input: None,
                    can_delete_in_copy: false,
                    can_duplicate_in_copy: false,
                    can_preview_numeric_output: false,
                    can_supply_numeric_export: false,
                    grid_position: None,
                })
            })
            .collect::<Result<_, GddArchiveError>>()?;
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
                        let imports = metadata
                            .input_metadata
                            .iter()
                            .map(|input| input.persistent_metadata.input_name.clone())
                            .collect();
                        (imports, metadata.output_names.clone())
                    },
                );
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
                    can_extend_ports: path.split_last().is_some_and(|(node_id, parent_path)| {
                        !interface.is_layer(node_id, parent_path)
                            && interface.network_metadata(path).is_some_and(|metadata| {
                                metadata.persistent_metadata.reference.is_none()
                            })
                    }),
                    empty_exports: interface
                        .nested_network(path)
                        .ok_or(GddArchiveError::MissingGraphNetwork)?
                        .exports
                        .iter()
                        .enumerate()
                        .filter_map(|(index, input)| empty_graph_export(input).then_some(index))
                        .collect(),
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
            let label = interface.display_name(&node.local_id, path);
            node.label = format!("{parent} · {label} [{}]", node.local_id.0);
            node.numeric_input = graph_numeric_input(&interface, node);
            node.can_preview_numeric_output = !interface.is_layer(&node.local_id, path)
                && interface
                    .document_node(&node.local_id, path)
                    .is_some_and(|document| numeric_preview_inputs(document).is_some());
            node.can_supply_numeric_export = node.can_preview_numeric_output
                && !interface.is_chain(&node.local_id, path)
                && interface.number_of_outputs(&node.local_id, path) == 1;
            node.can_duplicate_in_copy = node.numeric_input.is_some()
                && !interface.is_chain(&node.local_id, path)
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
        self.insert_root_node_template_in_copy(template, position, budget)
            .await
    }

    pub async fn append_nested_network_import_in_copy(
        &self,
        network: &GddGraphNetwork,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        self.check_network_edit_target(network)?;
        if !network.can_extend_ports_in_copy() {
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
        expected_inputs.push(NodeInput::value(TaggedValue::None, true));
        interface.add_import(TaggedValue::None, true, -1, "", "", path);
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
        let edits = [
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
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let original_slots = self
            .document
            .registry()
            .node_instances
            .get(&owner_storage_id)
            .ok_or(GddArchiveError::MissingGraphNetwork)?
            .inputs();
        let mut batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        let [RegistryDelta::SetNodeInputs { id, inputs }, rest @ ..] = batch.ops.as_mut_slice()
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

    pub async fn append_nested_network_export_in_copy(
        &self,
        network: &GddGraphNetwork,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::node::ui::OUTPUT_NAMES;

        self.check_network_edit_target(network)?;
        if !network.can_extend_ports_in_copy() {
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
        let export_index = expected_exports.len();
        if export_index != network.export_count {
            return Err(GddArchiveError::StaleGraphNetwork);
        }
        let expected_input = NodeInput::value(TaggedValue::None, true);
        expected_exports.push(expected_input.clone());
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
        let edits = [
            EditorDelta::Graph(RuntimeDelta::SetExport {
                network_path: path.clone(),
                export_index,
                input: Some(expected_input),
            }),
            EditorDelta::NodeMetadata {
                network_path: parent_path.to_vec(),
                node_id: owner_id,
                change: NodeMetadataChange::OutputNames(output_names.clone()),
            },
        ];
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &edits)?;
        if !matches!(
            batch.ops.as_slice(),
            [RegistryDelta::SetNetworkExport { id, index, export: Some(_) },
             RegistryDelta::ChangeNodeAttribute { id: node_id, delta }]
                if *id == network.id && *index as usize == export_index
                    && *node_id == owner_storage_id && delta.key == OUTPUT_NAMES
        ) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !actual
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

        let layout = [(path.as_slice(), nested), (parent_path, parent)]
            .into_iter()
            .flat_map(|(scope, network)| network.nodes.keys().map(move |node_id| (scope, *node_id)))
            .map(|(scope, node_id)| {
                let metadata = interface
                    .node_metadata(&node_id, scope)
                    .ok_or(GddArchiveError::MissingGraphNetwork)?;
                Ok((
                    scope,
                    node_id,
                    metadata.persistent_metadata.node_type_metadata.clone(),
                ))
            })
            .collect::<Result<Vec<_>, GddArchiveError>>()?;
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
        if layout.iter().any(|(scope, node_id, original)| {
            !interface
                .node_metadata(node_id, scope)
                .is_some_and(|metadata| {
                    metadata.persistent_metadata.node_type_metadata == *original
                })
        }) {
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
                let network_path = parent_path.to_vec();
                let delta = match connector {
                    InputConnector::Node {
                        node_id,
                        input_index,
                    } => RuntimeDelta::SetInput {
                        network_path,
                        node_id,
                        input_index,
                        input: input.clone(),
                    },
                    InputConnector::Export(export_index) => RuntimeDelta::SetExport {
                        network_path,
                        export_index,
                        input: Some(input.clone()),
                    },
                };
                edits.push(EditorDelta::Graph(delta));
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
        let detached = numeric_preview_network(nested, source.local_id)?;
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
        self.insert_root_node_template_in_copy(definition.default_node_template(), position, budget)
            .await
    }

    async fn insert_root_node_template_in_copy(
        &self,
        mut template: NodeTemplate,
        position: [i32; 2],
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        template.node_type_metadata = NodeTypePersistentMetadata::node(IVec2::from_array(position));
        let path = &[];
        let mut interface = self.runtime_graph().await?.into_interface()?;
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
        self.save_added_graph_node(interface, path, node_id, template, budget)
            .await
    }

    pub async fn preview_graph_numeric_output(
        &self,
        node: &GddGraphNode,
    ) -> Result<GddNumericValue, GddArchiveError> {
        self.check_node_edit_target(node)?;
        if !node.can_preview_numeric_output() {
            return Err(GddArchiveError::UnsupportedGraphNumericPreview);
        }
        let graph = self.runtime_graph().await?;
        let source = graph
            .network
            .nested_network(&node.network_path)
            .ok_or(GddArchiveError::StaleGraphNode)?;
        let detached = numeric_preview_network(source, node.local_id)?;
        let output = evaluate_detached_numeric_network(detached)
            .await
            .map_err(GddArchiveError::GraphNumericPreviewEvaluation)?;
        preview_numeric_value(&output)
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
        self.save_added_graph_node(interface, path, new_id, template, budget)
            .await
    }

    async fn save_added_graph_node(
        &self,
        interface: NodeNetworkInterface,
        path: &[NodeId],
        node_id: NodeId,
        template: NodeTemplate,
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
        let edits = [
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
        let (mut owner, staged) = self.fork_working_copy(&budget).await?;
        if self.graph_root_needs_initialization() {
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
            // The pinned cursor never undoes the first interaction, so the empty graph
            // needs a saved base before the first inserted node can be undone.
            commit_graph_node_edit(
                &mut owner,
                ConstructedOps {
                    ops: vec![RegistryDelta::AddNetwork {
                        id: ROOT_NETWORK,
                        network,
                    }],
                    declarations: Default::default(),
                },
            )?;
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
        if actual.create_node_template(&node_id, path) != Some(template) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
    }

    pub async fn delete_isolated_graph_node_in_copy(
        &self,
        node: &GddGraphNode,
        budget: GddArchiveBudget,
    ) -> Result<Self, GddArchiveError> {
        use graphite_document_graph_storage::attr::network::PINNED_ORDER;

        self.check_node_edit_target(node)?;
        if !node.can_delete_in_copy() {
            return Err(GddArchiveError::UnsupportedGraphNodeDeletion);
        }
        let path = &node.network_path;
        let original = self.runtime_graph().await?.into_interface()?;
        let mut interface = original.clone();
        let stored = self
            .document
            .registry()
            .node_instances
            .get(&node.storage_id)
            .ok_or(GddArchiveError::StaleGraphNode)?;
        let old_order = interface
            .network_metadata(path)
            .ok_or(GddArchiveError::StaleGraphNode)?
            .persistent_metadata
            .pinned_node_order
            .clone();
        interface.delete_nodes(vec![node.local_id], false, path);
        if interface.document_node(&node.local_id, path).is_some() {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let new_order = &interface
            .network_metadata(path)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?
            .persistent_metadata
            .pinned_node_order;
        let order_changed = *new_order != old_order;
        let mut edits = vec![EditorDelta::Graph(RuntimeDelta::RemoveNode {
            network_path: path.clone(),
            node_id: node.local_id,
        })];
        if order_changed {
            edits.push(EditorDelta::NetworkMetadata {
                network_path: path.clone(),
                change: NetworkMetadataChange::PinnedOrder(new_order.clone()),
            });
        }
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        // The source deletion removes the persistent storage ID along with the node.
        let batch = prepare_graph_node_edit(&owner, &original, &edits)?;
        let [RegistryDelta::RemoveNode { id, snapshot }, rest @ ..] = batch.ops.as_slice() else {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        };
        let resources = if order_changed {
            match rest {
                [
                    RegistryDelta::ChangeNetworkAttribute { id, delta },
                    resources @ ..,
                ] if *id == stored.network() && delta.key == PINNED_ORDER => resources,
                _ => return Err(GddArchiveError::IncompleteGraphNodeEdit),
            }
        } else {
            rest
        };
        if *id != node.storage_id
            || !snapshot.value_equal(stored)
            || !resources
                .iter()
                .all(|operation| matches!(operation, RegistryDelta::RemoveResource { .. }))
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.document_node(&node.local_id, path).is_some()
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
        use graphite_document_graph_storage::attr::node::ui::{IS_LAYER, POSITION};

        self.check_node_edit_target(node)?;
        let shift = node.grid_move_offset(destination)?;
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        interface.shift_absolute_node_position(&node.local_id, shift, path);
        let expected = Some(IVec2::from_array(destination));
        if interface.position(&node.local_id, path) != expected {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let metadata = interface
            .node_metadata(&node.local_id, path)
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        let edit = EditorDelta::NodeMetadata {
            network_path: path.clone(),
            node_id: node.local_id,
            change: NodeMetadataChange::NodeType(
                metadata.persistent_metadata.node_type_metadata.clone(),
            ),
        };
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let mut batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        if !matches!(
            batch.ops.as_slice(),
            [RegistryDelta::ChangeNodeAttribute { id, delta },
             RegistryDelta::ChangeNodeAttribute { id: mode_id, delta: mode }]
                if *id == node.storage_id && *mode_id == *id
                    && delta.key == POSITION && mode.key == IS_LAYER && mode.value.is_none()
        ) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        // The layer mode is an independent stored attribute and a grid move leaves it intact.
        batch.ops.truncate(1);
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if actual.position(&node.local_id, path) != expected {
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
        let Some(GddNumericInput::Connected { index, upstream }) =
            graph_numeric_input(&interface, node)
        else {
            return Err(GddArchiveError::UnsupportedGraphConnection);
        };
        let connector = InputConnector::node_at_index(node.local_id, index);
        if !is_only_graph_wire(&mut interface, path, upstream, connector) {
            return Err(GddArchiveError::UnsupportedGraphDisconnection);
        }
        let upstream_metadata = interface
            .node_metadata(&upstream, path)
            .ok_or(GddArchiveError::UnsupportedGraphConnection)?
            .persistent_metadata
            .clone();
        interface.disconnect_input(&connector, path);
        let expected = graph_numeric_port(&interface, node)
            .filter(|port| port.index == index && port.exposed)
            .ok_or(GddArchiveError::UnsupportedGraphDisconnection)?;
        if interface
            .node_metadata(&upstream, path)
            .is_none_or(|metadata| metadata.persistent_metadata != upstream_metadata)
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
        let path = &node.network_path;
        let input_index = expected.index();
        let connector = InputConnector::node_at_index(node.local_id, input_index);
        if !graph_numeric_input(&interface, node).is_some_and(|actual| actual.same_input(&expected))
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
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
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !graph_numeric_input(&actual, node).is_some_and(|actual| actual.same_input(&expected)) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        Ok(copy)
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

fn empty_graph_export(input: &NodeInput) -> bool {
    matches!(input, NodeInput::Value { tagged_value, exposed: true } if **tagged_value == TaggedValue::None)
}

fn numeric_preview_network(
    source: &NodeNetwork,
    selected: NodeId,
) -> Result<NodeNetwork, GddArchiveError> {
    let mut detached = NodeNetwork {
        exports: vec![NodeInput::node(selected, 0)],
        ..Default::default()
    };
    collect_numeric_preview_node(source, selected, &mut detached, &mut BTreeSet::new())?;
    Ok(detached)
}

fn collect_numeric_preview_node(
    source: &NodeNetwork,
    node_id: NodeId,
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
    for input in inputs {
        match input {
            NodeInput::Value { tagged_value, .. } => {
                preview_numeric_value(tagged_value)?;
            }
            NodeInput::Node {
                node_id,
                output_index: 0,
            } => {
                collect_numeric_preview_node(source, *node_id, detached, visiting)?;
            }
            _ => return Err(GddArchiveError::UnsupportedGraphNumericPreview),
        }
    }
    visiting.remove(&node_id);
    detached.nodes.insert(node_id, document.clone());
    Ok(())
}

fn numeric_preview_inputs(document: &DocumentNode) -> Option<&[NodeInput]> {
    use graphite_math_nodes::{
        absolute_value, add, ceiling, divide, floor, multiply, reciprocal, root, round, sign,
        subtract,
    };

    let DocumentNodeImplementation::ProtoNode(identifier) = &document.implementation else {
        return None;
    };
    let count = if [
        absolute_value::IDENTIFIER,
        ceiling::IDENTIFIER,
        floor::IDENTIFIER,
        reciprocal::IDENTIFIER,
        round::IDENTIFIER,
        sign::IDENTIFIER,
    ]
    .contains(identifier)
    {
        1
    } else if [
        add::IDENTIFIER,
        divide::IDENTIFIER,
        multiply::IDENTIFIER,
        root::IDENTIFIER,
        subtract::IDENTIFIER,
    ]
    .contains(identifier)
    {
        2
    } else {
        return None;
    };
    (document.inputs.len() == count && document.context_features == Default::default())
        .then_some(document.inputs.as_slice())
}

fn preview_numeric_value(value: &TaggedValue) -> Result<GddNumericValue, GddArchiveError> {
    const MAX_PREVIEW_VALUES: usize = 1024;
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
        GddNumericInput::Connected { .. } => None,
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
        use graphite_document_graph_storage::{AttributeDelta, NoMetadata, Value};
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
        let (mut owner, staged) = source
            .fork_working_copy(&budget())
            .await
            .expect("fixture fork");
        commit_graph_node_edit(
            &mut owner,
            ConstructedOps {
                ops: vec![RegistryDelta::ChangeNodeInputAttribute {
                    id: owner_id,
                    index: 1,
                    delta: AttributeDelta {
                        key: "fixture::opaque-import".into(),
                        value: Some(Value::Str("surviving input".into())),
                    },
                }],
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

    async fn network_export_fixture() -> (PreservedGddV1, Arc<[u8]>) {
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
        let nested = NodeNetwork {
            exports: vec![removed, surviving.clone()],
            ..Default::default()
        };
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
            let (original, bytes) = network_export_fixture().await;
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
            let (original, bytes) = network_export_fixture().await;
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

            let (exports, bytes) = network_export_fixture().await;
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
            for invalid in ["NETWORK:Custom Node", "PROTONODE:unknown-node", "Add"] {
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
                let detached =
                    numeric_preview_network(&source, selected).expect("admitted source operation");
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
            let detached =
                numeric_preview_network(&source, selected).expect("shared input is not a cycle");
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
                numeric_preview_network(&source, selected),
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
                    numeric_preview_network(&source, selected).expect("bounded pure branch");
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
                numeric_preview_network(&cyclic, selected),
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
                numeric_preview_network(&overlong, NodeId(0)),
                Err(GddArchiveError::GraphNumericPreviewBudget)
            ));
            assert!(matches!(
                preview_numeric_value(&TaggedValue::Numbers(vec![0.; 1025])),
                Err(GddArchiveError::GraphNumericPreviewBudget)
            ));
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
