use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    ops::{Bound, Deref},
    path::PathBuf,
    sync::Arc,
};

use graphite_core_types::NodeParameter;
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
    AttributesRead, NetworkId, NetworkMetadataEntry, NodeId as StorageNodeId, NodeMetadataEntry,
    PeerId, RegistryDelta, from_runtime, to_runtime::ConversionError,
};
use graphite_editor::messages::portfolio::document::{
    DocumentMessageHandler,
    utility_types::{
        misc::PTZ,
        network_interface::{
            InputConnector, NavigationMetadata, NodeNetworkInterface, Previewing,
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
use graphite_graph_craft::application_io::resource::{
    HashMapResourceStorage, ResourceHash, ResourceStorage,
};
use graphite_graph_craft::{
    document::{DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork, value::TaggedValue},
    runtime_delta::RuntimeDelta,
};
use zip::ZipArchive;

use super::{OriginalJson, legacy::PreservedLegacyDocument};

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
    #[error("Enter a different Graphite node name.")]
    UnchangedGraphNodeName,
    #[error("The Graphite node already has the requested pin state.")]
    UnchangedGraphNodePin,
    #[error("The selected Graphite node cannot move in that direction in pinned order.")]
    InvalidPinnedNodeMove,
    #[error("The selected Graphite node has no supported unexposed numeric value.")]
    UnsupportedGraphNumericInput,
    #[error("Enter a finite number for the Graphite Value input.")]
    InvalidGraphNumber,
    #[error("Enter a whole number in the 64-bit integer range for the Graphite Value input.")]
    InvalidGraphInteger,
    #[error("The Graphite Value input must keep its original number type.")]
    GraphInputTypeChanged,
    #[error("Enter a different value for the Graphite Value input.")]
    UnchangedGraphInput,
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

#[derive(Clone)]
pub struct GddGraphNode {
    snapshot: Arc<()>,
    storage_id: StorageNodeId,
    local_id: NodeId,
    network_path: Vec<NodeId>,
    label: String,
    name: String,
    pinned: bool,
    pinned_order: Option<(usize, usize)>,
    numeric_input: Option<GddNumericValue>,
}

#[derive(Clone, Copy, Debug)]
pub enum GddNumericValue {
    Number(f64),
    Integer(i64),
}

impl GddNumericValue {
    pub fn parse_like(self, text: &str) -> Result<Self, GddArchiveError> {
        let text = text.trim();
        match self {
            Self::Number(_) => text
                .parse::<f64>()
                .ok()
                .filter(|number| number.is_finite())
                .map(Self::Number)
                .ok_or(GddArchiveError::InvalidGraphNumber),
            Self::Integer(_) => text
                .parse::<i64>()
                .map(Self::Integer)
                .map_err(|_| GddArchiveError::InvalidGraphInteger),
        }
    }

    fn same_value(self, other: Self) -> bool {
        match (self, other) {
            (Self::Number(first), Self::Number(second)) => first.to_bits() == second.to_bits(),
            (Self::Integer(first), Self::Integer(second)) => first == second,
            _ => false,
        }
    }
}

impl fmt::Display for GddNumericValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(value) => value.fmt(formatter),
            Self::Integer(value) => value.fmt(formatter),
        }
    }
}

#[derive(Clone, Copy)]
pub enum GddPinnedNodeMove {
    Up,
    Down,
}

impl GddGraphNode {
    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    pub fn numeric_input(&self) -> Option<GddNumericValue> {
        self.numeric_input
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
        let graph = self.runtime_graph().await?;
        let mut nodes: Vec<_> = graph
            .nodes
            .iter()
            .map(|node| GddGraphNode {
                snapshot: self.snapshot.clone(),
                storage_id: node.storage_id,
                local_id: node.local_id,
                network_path: node.network_path.clone(),
                label: String::new(),
                name: node.display_name.clone().unwrap_or_default(),
                pinned: node.pinned,
                pinned_order: None,
                numeric_input: None,
            })
            .collect();
        let interface = graph.into_interface()?;
        nodes.sort_by(|left, right| {
            left.network_path
                .cmp(&right.network_path)
                .then_with(|| left.local_id.cmp(&right.local_id))
        });
        let mut pinned_indexes = BTreeMap::new();
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
            node.numeric_input = graph_numeric_input(&interface, node).map(|(_, value)| value);
        }
        Ok(nodes)
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
        let path = &node.network_path;
        let mut interface = self.runtime_graph().await?.into_interface()?;
        let (input_index, previous) = graph_numeric_input(&interface, node)
            .ok_or(GddArchiveError::UnsupportedGraphNumericInput)?;
        let tagged_value = match (previous, value) {
            (GddNumericValue::Number(_), GddNumericValue::Number(number)) if number.is_finite() => {
                TaggedValue::Number(number)
            }
            (GddNumericValue::Number(_), GddNumericValue::Number(_)) => {
                return Err(GddArchiveError::InvalidGraphNumber);
            }
            (GddNumericValue::Integer(_), GddNumericValue::Integer(number)) => {
                TaggedValue::Integer(number)
            }
            _ => return Err(GddArchiveError::GraphInputTypeChanged),
        };
        let connector = InputConnector::node_at_index(node.local_id, input_index);
        let input = NodeInput::value(tagged_value, false);
        if interface.input_from_connector(&connector, path) == Some(&input) {
            return Err(GddArchiveError::UnchangedGraphInput);
        }
        interface.set_input(&connector, input, path);
        if !graph_numeric_input(&interface, node)
            .is_some_and(|(_, accepted)| accepted.same_value(value))
        {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let input = interface
            .input_from_connector(&connector, path)
            .cloned()
            .ok_or(GddArchiveError::IncompleteGraphNodeEdit)?;
        let edit = EditorDelta::Graph(RuntimeDelta::SetInput {
            network_path: path.clone(),
            node_id: node.local_id,
            input_index,
            input,
        });
        let (owner, staged) = self.fork_working_copy(&budget).await?;
        let batch = prepare_graph_node_edit(&owner, &interface, &[edit])?;
        if !matches!(
            batch.ops.as_slice(),
            [RegistryDelta::ChangeNodeInput { id, index, .. }]
                if *id == node.storage_id && usize::try_from(*index).ok() == Some(input_index)
        ) {
            return Err(GddArchiveError::IncompleteGraphNodeEdit);
        }
        let copy = finish_graph_node_edit(owner, staged, batch, &budget).await?;
        let actual = copy.runtime_graph().await?.into_interface()?;
        if !graph_numeric_input(&actual, node)
            .is_some_and(|(index, actual)| index == input_index && actual.same_value(value))
        {
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

    async fn runtime_graph(&self) -> Result<GddRuntimeGraph, GddArchiveError> {
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

fn graph_numeric_input(
    interface: &NodeNetworkInterface,
    node: &GddGraphNode,
) -> Option<(usize, GddNumericValue)> {
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
    if interface.number_of_inputs(&node.local_id, path) != 1 {
        return None;
    }
    let connector = InputConnector::node_at_index(node.local_id, input_index);
    let NodeInput::Value {
        tagged_value,
        exposed: false,
    } = interface.input_from_connector(&connector, path)?
    else {
        return None;
    };
    let value = match &**tagged_value {
        TaggedValue::Number(number) => GddNumericValue::Number(*number),
        TaggedValue::Integer(number) if integer_supported => GddNumericValue::Integer(*number),
        _ => return None,
    };
    Some((input_index, value))
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
    staged: StagedArchive,
    batch: ConstructedOps,
    budget: &GddArchiveBudget,
) -> Result<InspectedGddV1, GddArchiveError> {
    let expected = batch.ops.len();
    owner.stage_constructed_ops(
        batch.ops,
        &batch.declarations.bytes,
        &HashMapResourceStorage::new(),
    )?;
    if owner.retire_pending_interaction()?.len() != expected {
        return Err(GddArchiveError::IncompleteGraphNodeEdit);
    }
    staged.reopen_after_edit(budget).await
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
    async fn reopen_after_edit(
        self,
        budget: &GddArchiveBudget,
    ) -> Result<InspectedGddV1, GddArchiveError> {
        // The source writer updates root payloads. Its nonrecursive listing misses retained
        // resources and opaque files in nested paths.
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

    #[test]
    fn at_gra_graph_031_1_preserves_typed_literals_and_targets_the_exact_nested_slot() {
        use graphite_document_graph_storage::NoMetadata;
        use graphite_math_nodes::{absolute_value, sign};
        use graphite_resource::ResourceRegistry;

        futures::executor::block_on(async {
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
                ..Default::default()
            };
            let mut owner = document();
            let resources = HashMapResourceStorage::new();
            owner
                .commit_from_runtime(&network, &NoMetadata, &ResourceRegistry::new(), &resources)
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
            let original = open_gdd_zip(&bytes, budget())
                .await
                .expect("original document");
            let nodes = original.graph_nodes().await.expect("typed values");
            let [root, parent, nested, exposed, wired] = nodes.as_slice() else {
                panic!("expected the original five nodes");
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
            for unavailable in [exposed, wired] {
                assert!(unavailable.numeric_input().is_none());
                assert!(matches!(
                    original
                        .set_graph_numeric_input_in_copy(
                            unavailable,
                            GddNumericValue::Number(1.),
                            budget()
                        )
                        .await,
                    Err(GddArchiveError::UnsupportedGraphNumericInput)
                ));
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
                    .is_some_and(|value| value.same_value(GddNumericValue::Number(-0.)))
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
