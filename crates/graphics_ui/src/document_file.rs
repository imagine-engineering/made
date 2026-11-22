use std::{
    collections::BTreeSet,
    future::Future,
    path::{Path, PathBuf},
    sync::{Arc, Weak},
};

use anyhow::{Context as _, anyhow, ensure};
use editor::Editor;
use gpui::{
    App, AppContext as _, AsyncWindowContext, ClipboardItem, Context, Entity, EntityId,
    EventEmitter, FocusHandle, Focusable, PathPromptOptions, Render, SharedString, Subscription,
    Task, WeakEntity, Window, actions,
};
use graphics_core::evaluate::UnaryNumericOperation;
use graphics_core::formats::{
    gdd::{
        ArchiveFormat, GddArchiveBudget, GddArchiveError, GddExportContent, GddGraphGeometry,
        GddGraphInspection, GddGraphNetwork, GddGraphNode, GddHistoryDirection, GddNumericValue,
        GddPinnedNodeMove, GddRecoverySource, GddSavedView, ImportOrExport, InspectedGddV1,
        create_gdd_folder, inspect_gdd_folder_files, open_gdd_archive,
    },
    legacy::{LegacyDocumentSchema, PreservedLegacyDocument, inspect_legacy_document},
};
use graphics_core::registry::GraphiteNodeCatalog;
use project::{DirectoryLister, Project, ProjectEntryId, ProjectPath, Worktree};
use ui::{
    Button, Checkbox, Color, ContextMenu, DropdownMenu, Label, LabelSize, ToggleState, prelude::*,
};
use workspace::{
    Item, Pane, Workspace,
    item::{ItemBufferKind, ItemEvent, SaveOptions},
    notifications::DetachAndPromptErr as _,
};

use crate::graph::{
    SavedGraphAction, SavedGraphBoxSelection, SavedGraphConnectionSource, SavedGraphDragStart,
    SavedGraphMenuTarget, SavedGraphView,
};
use crate::node_catalog::{CatalogNodeInsert, NodeCatalog};

actions!(
    graphite,
    [
        /// Opens a local Graphite GDD folder for inspection.
        OpenGraphiteFolder,
        /// Creates a new local Graphite GDD folder.
        NewGraphiteFolder,
    ]
);

pub(crate) fn init(cx: &mut App) {
    workspace::register_project_item::<GraphiteDocumentItem>(cx);
    cx.observe_new(|workspace: &mut Workspace, _, _cx| {
        workspace.register_action(|workspace, _: &OpenGraphiteFolder, window, cx| {
            open_graphite_folder(workspace, window, cx);
        });
        workspace.register_action(|workspace, _: &NewGraphiteFolder, window, cx| {
            new_graphite_folder(workspace, window, cx);
        });
    })
    .detach();
}

const ARCHIVE_BUDGET: GddArchiveBudget = GddArchiveBudget {
    max_archive_bytes: 64 * 1024 * 1024,
    max_entry_bytes: 64 * 1024 * 1024,
    max_total_entry_bytes: 128 * 1024 * 1024,
    max_entries: 8192,
};

const GRAPHITE_NUMERIC_OPERATIONS: [UnaryNumericOperation; 5] = [
    UnaryNumericOperation::AbsoluteValue,
    UnaryNumericOperation::Ceiling,
    UnaryNumericOperation::Floor,
    UnaryNumericOperation::Round,
    UnaryNumericOperation::Sign,
];

enum InspectedDocument {
    Gdd { contents: Arc<InspectedGddV1> },
    Legacy(PreservedLegacyDocument),
}

impl InspectedDocument {
    fn gdd(contents: InspectedGddV1) -> Self {
        Self::Gdd {
            contents: Arc::new(contents),
        }
    }
}

pub(crate) struct GraphiteDocumentProjectItem {
    name: SharedString,
    path: ProjectPath,
    entry_id: Option<ProjectEntryId>,
    // The project tracks hidden worktrees weakly.
    _worktree: Entity<Worktree>,
    original: Option<Arc<[u8]>>,
    document: InspectedDocument,
    revision: Arc<()>,
    working: Option<GddWorkingCopy>,
}

struct GddWorkingCopy {
    saved: Arc<InspectedGddV1>,
    inspection: Arc<GddGraphInspection>,
    save_in_progress: Weak<()>,
}

struct PreparedWorkingSnapshot {
    contents: Arc<InspectedGddV1>,
    inspection: GddGraphInspection,
    cut_clipboard: Option<String>,
}

impl GddWorkingCopy {
    fn new(saved: Arc<InspectedGddV1>, inspection: Arc<GddGraphInspection>) -> Self {
        Self {
            saved,
            inspection,
            save_in_progress: Weak::new(),
        }
    }
}

impl GraphiteDocumentProjectItem {
    fn gdd_contents(&self) -> Option<&Arc<InspectedGddV1>> {
        match &self.document {
            InspectedDocument::Gdd { contents } => Some(contents),
            InspectedDocument::Legacy(_) => None,
        }
    }

    fn working_dirty(&self) -> bool {
        self.working.as_ref().is_some_and(|working| {
            self.gdd_contents().is_some_and(|contents| {
                // Redo can restore the last saved cursor in a newly reopened container.
                !Arc::ptr_eq(contents, &working.saved)
                    && !contents.matches_saved_history(&working.saved)
            })
        })
    }

    fn start_working_copy(
        &mut self,
        expected: &Arc<()>,
        inspection: GddGraphInspection,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        ensure!(
            Arc::ptr_eq(&self.revision, expected),
            "The Graphite document changed while opening its working copy."
        );
        if self.working.is_some() {
            return Ok(());
        }
        let saved = self
            .gdd_contents()
            .cloned()
            .context("Open a GDD document.")?;
        ensure!(
            saved.hot_log_len() == 0,
            "Recover the pending Graphite edits before opening a working copy."
        );
        self.working = Some(GddWorkingCopy::new(saved, Arc::new(inspection)));
        cx.notify();
        Ok(())
    }

    fn replace_working_snapshot(
        &mut self,
        expected: &Arc<()>,
        contents: Arc<InspectedGddV1>,
        inspection: GddGraphInspection,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        ensure!(
            Arc::ptr_eq(&self.revision, expected),
            "The Graphite working copy changed. Review the current value and try again."
        );
        let inspection = Arc::new(inspection);
        if let Some(working) = &mut self.working {
            working.inspection = inspection;
        } else {
            let saved = self
                .gdd_contents()
                .cloned()
                .context("Open a GDD document.")?;
            self.working = Some(GddWorkingCopy::new(saved, inspection));
        }
        self.document = InspectedDocument::Gdd { contents };
        self.revision = Arc::new(());
        cx.notify();
        Ok(())
    }
}

impl project::ProjectItem for GraphiteDocumentProjectItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        let project_model = project.read(cx);
        let worktree = project_model.worktree_for_id(path.worktree_id, cx)?;
        let root = worktree.read(cx).abs_path();
        let name = path
            .path
            .file_name()
            .or_else(|| root.file_name()?.to_str())?;
        let extension = name.to_ascii_lowercase();
        let legacy = extension.ends_with(".graphite");
        if !legacy
            && !extension.ends_with(".gdd.zip")
            && !extension.ends_with(".gdd.tar.xz")
            && !extension.ends_with(".gdd")
        {
            return None;
        }
        if !project_model.is_local() {
            return Some(Task::ready(Err(anyhow!(
                "Open this Graphite document from a local workspace."
            ))));
        }
        let name = SharedString::from(name.to_owned());
        let path = path.clone();
        let entry_id = project_model
            .entry_for_path(&path, cx)
            .map(|entry| entry.id);
        let filesystem = project_model.fs().clone();
        let project = project.downgrade();
        let relative_path = path.path.clone();
        let load = cx.background_spawn({
            let root = root.clone();
            async move {
                let captured = filesystem.capture_confined_root(&root).await?;
                let source = filesystem
                    .read_confined_file(
                        &captured,
                        &root,
                        relative_path.as_std_path(),
                        ARCHIVE_BUDGET.max_archive_bytes,
                    )
                    .await?;
                let bytes = Arc::<[u8]>::from(source.bytes);
                if legacy {
                    let document =
                        inspect_legacy_document(&bytes, ARCHIVE_BUDGET.max_archive_bytes)?;
                    return Ok::<_, anyhow::Error>((bytes, InspectedDocument::Legacy(document)));
                }
                let archive =
                    futures::executor::block_on(open_gdd_archive(&bytes, ARCHIVE_BUDGET))?;
                Ok((bytes, InspectedDocument::gdd(archive.into_inspected())))
            }
        });
        Some(cx.spawn(async move |cx| {
            let (original, document) = load
                .await
                .with_context(|| format!("Opening Graphite document {name}"))?;
            project.read_with(cx, |project, cx| {
                let current_worktree = project.worktree_for_id(path.worktree_id, cx);
                ensure!(
                    project.is_local()
                        && current_worktree.is_some_and(|current| {
                            current == worktree && current.read(cx).abs_path() == root
                        })
                        && project.entry_for_path(&path, cx).map(|entry| entry.id) == entry_id,
                    "The Graphite document path changed while it was opening. Reopen the file."
                );
                Ok(())
            })??;
            Ok(cx.new(|_| Self {
                name,
                path,
                entry_id,
                _worktree: worktree,
                original: Some(original),
                document,
                revision: Arc::new(()),
                working: None,
            }))
        }))
    }

    fn entry_id(&self, _cx: &App) -> Option<ProjectEntryId> {
        self.entry_id
    }

    fn project_path(&self, _cx: &App) -> Option<ProjectPath> {
        Some(self.path.clone())
    }

    fn is_dirty(&self) -> bool {
        self.working_dirty()
    }
}

fn open_graphite_folder(
    workspace: &mut Workspace,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let project = workspace.project().clone();
    let project_model = project.read(cx);
    let filesystem = project_model.fs().clone();
    let selection = project_model.is_local().then(|| {
        cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Open Graphite Folder".into()),
        })
    });
    cx.spawn_in(window, async move |workspace, cx| {
        let Some(paths) = selection
            .context("Open a Graphite folder from a local workspace.")?
            .await??
        else {
            return Ok(());
        };
        let [path] = paths.as_slice() else {
            return Err(anyhow!("Choose one Graphite folder."));
        };
        let path = path.clone();
        let contents = cx
            .background_spawn({
                let path = path.clone();
                async move { read_graphite_folder(filesystem.as_ref(), &path).await }
            })
            .await
            .with_context(|| format!("Opening Graphite folder {}", path.display()))?;
        open_folder_item(workspace, &project, &path, contents, cx).await
    })
    .detach_and_prompt_err("Could not open Graphite folder", window, cx, |_, _, _| None);
}

fn new_graphite_folder(
    workspace: &mut Workspace,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let project = workspace.project().clone();
    let project_model = project.read(cx);
    let filesystem = project_model.fs().clone();
    let local = project_model.is_local();
    let selection = local.then(|| {
        workspace.prompt_for_new_path(
            DirectoryLister::Local(project.clone(), filesystem.clone()),
            Some("Untitled.gdd".into()),
            window,
            cx,
        )
    });
    cx.spawn_in(window, async move |workspace, cx| {
        let Some(paths) = selection
            .context("Create a Graphite folder from a local workspace.")?
            .await?
        else {
            return Ok(());
        };
        let [path] = paths.as_slice() else {
            return Err(anyhow!("Choose one Graphite folder."));
        };
        let (document_id, peer_id) = uuid::Uuid::new_v4().as_u64_pair();
        let (contents, files) = cx
            .background_spawn(async move {
                let contents =
                    create_gdd_folder(document_id, peer_id, "Made".into(), ARCHIVE_BUDGET).await?;
                let files = contents.folder_files()?;
                anyhow::Ok((contents, files))
            })
            .await?;
        ensure!(
            workspace.read_with(cx, |workspace, _| workspace.project() == &project)?,
            "The Graphite workspace has changed."
        );
        filesystem
            .create_new_directory_atomically(path.clone(), files)
            .await
            .with_context(|| format!("Creating Graphite folder {}", path.display()))?;
        open_folder_item(workspace, &project, path, contents, cx)
            .await
            .with_context(|| {
                format!(
                    "The Graphite folder was created at {}. Opening it failed",
                    path.display()
                )
            })
    })
    .detach_and_prompt_err("New Graphite folder", window, cx, |_, _, _| None);
}

async fn open_folder_item(
    workspace: WeakEntity<Workspace>,
    project: &Entity<Project>,
    path: &Path,
    contents: InspectedGddV1,
    cx: &mut AsyncWindowContext,
) -> anyhow::Result<()> {
    let name = path
        .file_name()
        .context("Graphite folder has no name")?
        .to_string_lossy()
        .into_owned();
    ensure!(
        workspace.read_with(cx, |workspace, _| workspace.project() == project)?,
        "The Graphite workspace has changed."
    );
    let (worktree, project_path) = cx
        .update(|_, cx| Workspace::project_path_for_path(project.clone(), path, false, cx))?
        .await?;
    workspace.update_in(cx, |workspace, window, cx| {
        ensure!(
            workspace.project() == project,
            "The Graphite workspace has changed."
        );
        let entry_id = project
            .read(cx)
            .entry_for_path(&project_path, cx)
            .map(|entry| entry.id);
        let model = cx.new(|_| GraphiteDocumentProjectItem {
            name: name.into(),
            path: project_path,
            entry_id,
            _worktree: worktree,
            original: None,
            document: InspectedDocument::gdd(contents),
            revision: Arc::new(()),
            working: None,
        });
        workspace.open_project_item::<GraphiteDocumentItem>(
            workspace.active_pane().clone(),
            model,
            true,
            true,
            false,
            false,
            window,
            cx,
        );
        anyhow::Ok(())
    })?
}

async fn read_graphite_folder(
    filesystem: &dyn project::Fs,
    root_path: &Path,
) -> anyhow::Result<InspectedGddV1> {
    let root = filesystem.capture_confined_root(root_path).await?;
    let mut pending = vec![PathBuf::new()];
    let mut snapshots = Vec::new();
    let mut remaining_entries = ARCHIVE_BUDGET.max_entries;
    let mut remaining_bytes = ARCHIVE_BUDGET.max_total_entry_bytes;
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = filesystem
            .list_confined_directory(&root, root_path, &directory, remaining_entries)
            .await?;
        remaining_entries = remaining_entries
            .checked_sub(entries.len())
            .context("Graphite folder exceeds its entry budget")?;
        for entry in &entries {
            let path = directory.join(&entry.name);
            if entry.is_dir {
                pending.push(path);
                continue;
            }
            let max_bytes = remaining_bytes.min(ARCHIVE_BUDGET.max_entry_bytes);
            ensure!(
                entry.len <= max_bytes,
                "Graphite folder exceeds its byte budget"
            );
            let file = filesystem
                .read_confined_file(&root, root_path, &path, usize::try_from(max_bytes)?)
                .await?;
            ensure!(
                file.bytes.len() as u64 == entry.len,
                "Graphite folder changed while opening"
            );
            remaining_bytes -= entry.len;
            let name = path
                .to_str()
                .context("Graphite folder entries must be UTF-8")?;
            files.push((name.replace(std::path::MAIN_SEPARATOR, "/"), file.bytes));
        }
        snapshots.push((directory, entries));
    }
    for (directory, entries) in snapshots {
        ensure!(
            filesystem
                .list_confined_directory(&root, root_path, &directory, entries.len())
                .await?
                == entries,
            "Graphite folder changed while opening"
        );
    }
    Ok(inspect_gdd_folder_files(files, ARCHIVE_BUDGET).await?)
}

async fn read_approved_resource_files(
    filesystem: &dyn project::Fs,
    paths: Vec<PathBuf>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    ensure!(
        paths.len() <= ARCHIVE_BUDGET.max_entries,
        "Too many Graphite resource files were selected."
    );
    let mut remaining = ARCHIVE_BUDGET.max_total_entry_bytes;
    let mut approved = Vec::new();
    for path in paths {
        let root = filesystem.capture_confined_root(&path).await?;
        let limit = usize::try_from(remaining.min(ARCHIVE_BUDGET.max_entry_bytes))?;
        let file = filesystem
            .read_confined_file(&root, &path, Path::new(""), limit)
            .await
            .with_context(|| format!("Reading selected Graphite resource {}", path.display()))?;
        remaining = remaining
            .checked_sub(file.bytes.len() as u64)
            .context("Selected Graphite resources exceed the byte budget.")?;
        approved.push(file.bytes);
    }
    Ok(approved)
}

async fn read_approved_legacy_document(
    filesystem: &dyn project::Fs,
    path: &Path,
) -> anyhow::Result<PreservedLegacyDocument> {
    let root = filesystem.capture_confined_root(path).await?;
    let file = filesystem
        .read_confined_file(&root, path, Path::new(""), ARCHIVE_BUDGET.max_archive_bytes)
        .await?;
    inspect_legacy_document(&file.bytes.into(), ARCHIVE_BUDGET.max_archive_bytes).with_context(
        || {
            format!(
                "Reading selected legacy Graphite document {}",
                path.display()
            )
        },
    )
}

pub(crate) struct GraphiteDocumentItem {
    model: Entity<GraphiteDocumentProjectItem>,
    _model_subscription: Subscription,
    project: WeakEntity<Project>,
    focus_handle: FocusHandle,
    copy_task: Option<Task<()>>,
    copy_feedback: Option<(SharedString, Color)>,
    export_content: GddExportContent,
    embed_linked_resources: bool,
    embed_legacy_document: bool,
    saved_view: SavedViewPanel,
    node_catalog: NodeCatalogPanel,
    node_copy: NodeCopyPanel,
    node_insert: Option<NodeInsertForm>,
}

struct NodeInsertForm {
    position: Entity<Editor>,
    destination: NodeInsertDestination,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeInsertDestination {
    WorkingCopy,
    SeparateFolder,
}

enum NodeCatalogPanel {
    Closed,
    Loading {
        _task: Task<()>,
    },
    Ready {
        catalog: Entity<NodeCatalog>,
        _insert_subscription: Subscription,
    },
    Failed(SharedString),
}

enum NodeCopyPanel {
    Closed,
    Loading { _task: Task<()> },
    Ready(Box<NodeCopyForm>),
    Failed(SharedString),
}

struct NodeCopyForm {
    snapshot: Arc<InspectedGddV1>,
    revision: Arc<()>,
    nodes: Vec<GddGraphNode>,
    networks: Vec<GddGraphNetwork>,
    active_network: Option<usize>,
    selected: Option<usize>,
    name: Entity<Editor>,
    value: Entity<Editor>,
    position: Entity<Editor>,
    output_preview: NodeOutputPreview,
    port_rename: Option<PortRenameForm>,
    node_selection: BTreeSet<usize>,
    show_saved_ports: bool,
    saved_graph: SavedGraphPanel,
    saved_graph_focus: FocusHandle,
    saved_graph_drag_start: SavedGraphDragStart,
    saved_graph_box_selection: SavedGraphBoxSelection,
    pending_canvas_source: Option<SavedGraphConnectionSource>,
}

enum SavedGraphPanel {
    Closed,
    Loading { _task: Task<()> },
    Ready(GddGraphGeometry),
    Failed(SharedString),
}

impl NodeCopyForm {
    fn new(
        snapshot: Arc<InspectedGddV1>,
        revision: Arc<()>,
        inspection: &GddGraphInspection,
        window: &mut Window,
        cx: &mut Context<GraphiteDocumentItem>,
    ) -> Self {
        Self {
            snapshot,
            revision,
            nodes: inspection.nodes.clone(),
            networks: inspection.networks.clone(),
            active_network: None,
            selected: None,
            name: cx.new(|cx| Editor::single_line(window, cx)),
            value: cx.new(|cx| Editor::single_line(window, cx)),
            position: cx.new(|cx| Editor::single_line(window, cx)),
            output_preview: NodeOutputPreview::Hidden,
            port_rename: None,
            node_selection: BTreeSet::new(),
            show_saved_ports: false,
            saved_graph: SavedGraphPanel::Closed,
            saved_graph_focus: cx.focus_handle(),
            saved_graph_drag_start: Default::default(),
            saved_graph_box_selection: Default::default(),
            pending_canvas_source: None,
        }
    }

    fn replace_inspection(
        &mut self,
        snapshot: Arc<InspectedGddV1>,
        revision: Arc<()>,
        inspection: &GddGraphInspection,
    ) -> Option<usize> {
        let network_id = self.active_network().map(|network| network.id);
        let selected_id = self.selected_node().map(GddGraphNode::storage_id);
        let group = self
            .node_selection
            .iter()
            .filter_map(|index| self.nodes.get(*index))
            .map(GddGraphNode::storage_id)
            .collect::<BTreeSet<_>>();
        self.snapshot = snapshot;
        self.revision = revision;
        self.nodes.clone_from(&inspection.nodes);
        self.networks.clone_from(&inspection.networks);
        let active_network =
            network_id.and_then(|id| self.networks.iter().position(|network| network.id == id));
        self.select_network(active_network);
        self.node_selection = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| self.includes_node(node) && group.contains(&node.storage_id()))
            .map(|(index, _)| index)
            .collect();
        selected_id.and_then(|id| {
            self.nodes
                .iter()
                .position(|node| self.includes_node(node) && node.storage_id() == id)
        })
    }

    fn selected_node(&self) -> Option<&GddGraphNode> {
        self.selected.and_then(|index| self.nodes.get(index))
    }

    fn active_network(&self) -> Option<&GddGraphNetwork> {
        self.active_network
            .and_then(|index| self.networks.get(index))
    }

    fn clipboard_scope(&self) -> Option<&GddGraphNetwork> {
        self.active_network().or_else(|| {
            self.networks
                .iter()
                .find(|network| network.parent.is_none())
        })
    }

    fn selected_graph_nodes(&self) -> Vec<GddGraphNode> {
        if self.node_selection.is_empty() {
            return self
                .selected_node()
                .filter(|node| node.can_copy_graph_node())
                .cloned()
                .into_iter()
                .collect();
        }
        self.node_selection
            .iter()
            .filter_map(|index| self.nodes.get(*index))
            .cloned()
            .collect()
    }

    fn includes_node(&self, node: &GddGraphNode) -> bool {
        self.active_network()
            .is_none_or(|network| network.id == node.network_id())
    }

    fn select_network(&mut self, index: Option<usize>) {
        self.active_network = index;
        self.selected = None;
        self.output_preview = NodeOutputPreview::Hidden;
        self.port_rename = None;
        self.node_selection.clear();
        self.saved_graph = SavedGraphPanel::Closed;
        self.saved_graph_drag_start.set(None);
        self.saved_graph_box_selection.set(None);
        self.pending_canvas_source = None;
    }

    fn connect_network_import(&self, import_index: usize) -> Option<GraphiteCopy> {
        let (network, target) = self.active_network().zip(self.selected_node())?;
        network
            .compatible_numeric_imports(target)
            .any(|index| index == import_index)
            .then(|| {
                GraphiteCopy::EditNetworkWire(NetworkWireChange::Import {
                    network: Box::new(network.clone()),
                    target: Box::new(target.clone()),
                    import_index,
                    reconnect: target.numeric_import_source().is_some(),
                })
            })
    }

    fn connect_numeric_output(&self, source_index: usize) -> Option<GraphiteCopy> {
        let target = self.selected_node()?;
        let source = self.nodes.get(source_index)?;
        let previous = target.numeric_connection_source(&self.nodes);
        let compatible = match previous {
            Some(previous) => target.can_reconnect_numeric_input_from(previous, source),
            None => target.can_connect_numeric_input_from(source),
        };
        compatible.then(|| GraphiteCopy::EditNode {
            node: Box::new(target.clone()),
            change: NodeCopyChange::ConnectNumericInput {
                source: Box::new(source.clone()),
                previous: previous.cloned().map(Box::new),
            },
        })
    }
}

struct PortRenameForm {
    kind: NetworkPortKind,
    index: usize,
    name: Entity<Editor>,
}

enum NodeOutputPreview {
    Hidden,
    Pending { _task: Task<()> },
    Ready(GddNumericValue),
    Failed(SharedString),
}

enum SavedViewPanel {
    Closed,
    Loading { _task: Task<()> },
    Ready(GddSavedView),
    Failed(SharedString),
}

#[derive(Clone, Copy)]
enum GddCopyFormat {
    Archive(ArchiveFormat),
    Folder,
}

enum GraphiteCopy {
    Original,
    Export(GddCopyFormat),
    RecoverFolder,
    MoveHistory(GddHistoryDirection),
    InsertNumericNode {
        operation: UnaryNumericOperation,
        position: [i32; 2],
    },
    InsertCatalogNode {
        identifier: &'static str,
        position: [i32; 2],
    },
    AppendNetworkPort {
        network: Box<GddGraphNetwork>,
        kind: NetworkPortKind,
        with_hidden_primary: bool,
    },
    SetPrimaryNetworkPortExposure {
        network: Box<GddGraphNetwork>,
        kind: NetworkPortKind,
        exposed: bool,
    },
    EditNetworkPort {
        network: Box<GddGraphNetwork>,
        change: NetworkPortChange,
    },
    RenameNetworkPort {
        network: Box<GddGraphNetwork>,
        port: ImportOrExport,
        name: String,
    },
    PasteNodeGroup {
        network: Box<GddGraphNetwork>,
        clipboard: String,
    },
    CutNodeGroup {
        nodes: Vec<GddGraphNode>,
    },
    DuplicateNodeGroup {
        network: Box<GddGraphNetwork>,
        nodes: Vec<GddGraphNode>,
    },
    EditNetworkWire(NetworkWireChange),
    EditNode {
        node: Box<GddGraphNode>,
        change: NodeCopyChange,
    },
}

enum NetworkWireChange {
    Export {
        network: Box<GddGraphNetwork>,
        source: Box<GddGraphNode>,
        export_index: usize,
    },
    Import {
        network: Box<GddGraphNetwork>,
        target: Box<GddGraphNode>,
        import_index: usize,
        reconnect: bool,
    },
}

impl NetworkWireChange {
    async fn stage(self, archive: &InspectedGddV1) -> Result<InspectedGddV1, GddArchiveError> {
        match self {
            Self::Export {
                network,
                source,
                export_index,
            } => {
                archive
                    .connect_numeric_network_export_in_copy(
                        &network,
                        &source,
                        export_index,
                        ARCHIVE_BUDGET,
                    )
                    .await
            }
            Self::Import {
                network,
                target,
                import_index,
                reconnect: true,
            } => {
                archive
                    .reconnect_numeric_network_import_in_copy(
                        &network,
                        import_index,
                        &target,
                        ARCHIVE_BUDGET,
                    )
                    .await
            }
            Self::Import {
                network,
                target,
                import_index,
                reconnect: false,
            } => {
                archive
                    .connect_numeric_network_import_in_copy(
                        &network,
                        import_index,
                        &target,
                        ARCHIVE_BUDGET,
                    )
                    .await
            }
        }
    }
}

#[derive(Clone, Copy)]
enum NetworkPortKind {
    Import,
    Export,
}

impl NetworkPortKind {
    fn count(self, network: &GddGraphNetwork) -> usize {
        match self {
            Self::Import => network.import_count,
            Self::Export => network.export_count,
        }
    }

    fn name(self, network: &GddGraphNetwork, index: usize) -> Option<&str> {
        match self {
            Self::Import => network.import_name(index),
            Self::Export => network.export_name(index),
        }
    }

    fn port(self, index: usize) -> ImportOrExport {
        match self {
            Self::Import => ImportOrExport::Import(index),
            Self::Export => ImportOrExport::Export(index),
        }
    }

    fn primary_exposure(self, network: &GddGraphNetwork) -> Option<bool> {
        match self {
            Self::Import => network.primary_import_exposed(),
            Self::Export => network.primary_export_exposed(),
        }
    }
}

#[derive(Clone, Copy)]
enum NetworkPortMenuAction {
    Copy(NetworkPortChange),
    Rename,
}

#[derive(Clone, Copy)]
enum NetworkPortChange {
    Remove {
        kind: NetworkPortKind,
        index: usize,
    },
    Move {
        kind: NetworkPortKind,
        from: usize,
        to: usize,
    },
}

enum NodeCopyChange {
    Rename(String),
    DuplicateNumeric,
    DeleteIsolated,
    GridPosition([i32; 2]),
    Pin(bool),
    MovePinned(GddPinnedNodeMove),
    NumericInput(GddNumericValue),
    NumericInputExposure(bool),
    DisconnectNumericInput,
    InsertNumericOnWire(UnaryNumericOperation),
    ConnectNumericInput {
        source: Box<GddGraphNode>,
        previous: Option<Box<GddGraphNode>>,
    },
}

impl NodeCopyChange {
    async fn stage(
        self,
        archive: &InspectedGddV1,
        node: &GddGraphNode,
    ) -> Result<InspectedGddV1, GddArchiveError> {
        match self {
            Self::Rename(name) => {
                archive
                    .rename_graph_node_in_copy(node, name, ARCHIVE_BUDGET)
                    .await
            }
            Self::DuplicateNumeric => {
                archive
                    .duplicate_graph_numeric_node_in_copy(node, ARCHIVE_BUDGET)
                    .await
            }
            Self::DeleteIsolated => {
                archive
                    .delete_isolated_graph_node_in_copy(node, ARCHIVE_BUDGET)
                    .await
            }
            Self::GridPosition(position) => {
                archive
                    .move_graph_node_in_copy(node, position, ARCHIVE_BUDGET)
                    .await
            }
            Self::Pin(pinned) => {
                archive
                    .set_graph_node_pinned_in_copy(node, pinned, ARCHIVE_BUDGET)
                    .await
            }
            Self::MovePinned(movement) => {
                archive
                    .move_pinned_graph_node_in_copy(node, movement, ARCHIVE_BUDGET)
                    .await
            }
            Self::NumericInput(value) => {
                archive
                    .set_graph_numeric_input_in_copy(node, value, ARCHIVE_BUDGET)
                    .await
            }
            Self::NumericInputExposure(exposed) => {
                archive
                    .set_graph_numeric_input_exposed_in_copy(node, exposed, ARCHIVE_BUDGET)
                    .await
            }
            Self::DisconnectNumericInput => {
                archive
                    .disconnect_graph_numeric_input_in_copy(node, ARCHIVE_BUDGET)
                    .await
            }
            Self::InsertNumericOnWire(operation) => {
                archive
                    .insert_numeric_node_on_wire_in_copy(node, operation, ARCHIVE_BUDGET)
                    .await
            }
            Self::ConnectNumericInput {
                source,
                previous: Some(previous),
            } => {
                archive
                    .reconnect_graph_numeric_input_in_copy(node, &previous, &source, ARCHIVE_BUDGET)
                    .await
            }
            Self::ConnectNumericInput {
                source,
                previous: None,
            } => {
                archive
                    .connect_graph_numeric_input_in_copy(node, &source, ARCHIVE_BUDGET)
                    .await
            }
        }
    }
}

enum PreparedCopy {
    File(Arc<[u8]>),
    Folder {
        files: Vec<(PathBuf, Vec<u8>)>,
        clipboard: Option<String>,
    },
}

fn parse_graph_grid_position(text: &str) -> anyhow::Result<[i32; 2]> {
    const INVALID_POSITION: &str = "Enter two whole grid coordinates separated by a comma.";
    let (x, y) = text.split_once(',').context(INVALID_POSITION)?;
    Ok([
        x.trim().parse().context(INVALID_POSITION)?,
        y.trim().parse().context(INVALID_POSITION)?,
    ])
}

fn gdd_copy_stem(name: &str) -> &str {
    let lowercase = name.to_ascii_lowercase();
    [".gdd.tar.xz", ".gdd.zip", ".gdd"]
        .into_iter()
        .find(|suffix| lowercase.ends_with(suffix))
        .and_then(|suffix| name.get(..name.len() - suffix.len()))
        .unwrap_or(name)
}

impl GraphiteDocumentItem {
    fn new(
        model: Entity<GraphiteDocumentProjectItem>,
        project: WeakEntity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let subscription = cx.observe_in(&model, window, |item, _, window, cx| {
            item.update_working_inspection(window, cx);
            cx.emit(ItemEvent::UpdateTab);
            cx.notify();
        });
        Self {
            model,
            _model_subscription: subscription,
            project,
            focus_handle: cx.focus_handle(),
            copy_task: None,
            copy_feedback: None,
            export_content: GddExportContent::default(),
            embed_linked_resources: false,
            embed_legacy_document: false,
            saved_view: SavedViewPanel::Closed,
            node_catalog: NodeCatalogPanel::Closed,
            node_copy: NodeCopyPanel::Closed,
            node_insert: None,
        }
    }

    fn update_working_inspection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let model = self.model.read(cx);
        let Some((contents, working)) = model.gdd_contents().zip(model.working.as_ref()) else {
            return;
        };
        let contents = contents.clone();
        let revision = model.revision.clone();
        let inspection = working.inspection.clone();
        match &mut self.node_copy {
            NodeCopyPanel::Closed => {}
            NodeCopyPanel::Ready(form) if !Arc::ptr_eq(&form.revision, &revision) => {
                let reopen_graph = matches!(
                    form.saved_graph,
                    SavedGraphPanel::Loading { .. } | SavedGraphPanel::Ready(_)
                );
                if form.saved_graph_focus.contains_focused(window, cx) {
                    self.focus_handle.focus(window, cx);
                }
                if let Some(index) = form.replace_inspection(contents, revision, &inspection) {
                    self.select_copy_node(index, window, cx);
                }
                if reopen_graph {
                    self.toggle_saved_graph(window, cx);
                }
            }
            NodeCopyPanel::Loading { .. } | NodeCopyPanel::Failed(_) => {
                self.node_copy = NodeCopyPanel::Ready(Box::new(NodeCopyForm::new(
                    contents,
                    revision,
                    &inspection,
                    window,
                    cx,
                )));
            }
            NodeCopyPanel::Ready(_) => {}
        }
    }

    fn open_working_copy(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let model = self.model.read(cx);
        if self.copy_task.is_some() || model.working.is_some() {
            return;
        }
        let Some(contents) = model
            .gdd_contents()
            .filter(|contents| contents.hot_log_len() == 0)
            .cloned()
        else {
            return;
        };
        let revision = model.revision.clone();
        let load = cx.background_spawn(async move { contents.graph_inspection().await });
        self.copy_feedback = None;
        self.copy_task = Some(cx.spawn_in(window, async move |item, cx| {
            let inspection = load.await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                item.copy_task = None;
                let result = inspection
                    .map_err(anyhow::Error::from)
                    .and_then(|inspection| {
                        item.model.update(cx, |model, cx| {
                            model.start_working_copy(&revision, inspection, cx)
                        })
                    });
                item.copy_feedback = result
                    .err()
                    .map(|error| (format!("{error:#}").into(), Color::Error));
                cx.notify();
            }) {
                log::debug!("Graphite working copy view closed: {error:#}");
            }
        }));
        cx.notify();
    }

    fn toggle_node_catalog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(
            self.node_catalog,
            NodeCatalogPanel::Loading { .. } | NodeCatalogPanel::Ready { .. }
        ) {
            self.node_catalog = NodeCatalogPanel::Closed;
            self.focus_handle.focus(window, cx);
            cx.notify();
            return;
        }
        let load = cx.background_spawn(async { GraphiteNodeCatalog::load() });
        self.node_catalog = NodeCatalogPanel::Loading {
            _task: cx.spawn_in(window, async move |item, cx| {
                let loaded = load.await;
                if let Err(error) = item.update_in(cx, |item, window, cx| {
                    if !matches!(item.node_catalog, NodeCatalogPanel::Loading { .. }) {
                        return;
                    }
                    item.node_catalog = match loaded {
                        Ok(source) => {
                            let admit_insertion = matches!(
                                &item.model.read(cx).document,
                                InspectedDocument::Gdd { contents } if contents.hot_log_len() == 0
                            );
                            let catalog =
                                cx.new(|cx| NodeCatalog::new(source, admit_insertion, window, cx));
                            let insert_subscription =
                                cx.subscribe_in(&catalog, window, |item, _, event, window, cx| {
                                    item.save_catalog_node(event, window, cx);
                                });
                            NodeCatalogPanel::Ready {
                                catalog,
                                _insert_subscription: insert_subscription,
                            }
                        }
                        Err(error) => NodeCatalogPanel::Failed(error.to_string().into()),
                    };
                    cx.notify();
                }) {
                    log::debug!("Graphite catalog closed: {error:#}");
                }
            }),
        };
        cx.notify();
    }

    fn render_node_catalog(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let label = match &self.node_catalog {
            NodeCatalogPanel::Closed => "Browse Nodes",
            NodeCatalogPanel::Loading { .. } => "Cancel Catalog Load",
            NodeCatalogPanel::Ready { .. } => "Close Node Catalog",
            NodeCatalogPanel::Failed(_) => "Retry Node Catalog",
        };
        let panel = v_flex().gap_2().child(
            div()
                .debug_selector(|| "GRAPHITE-NODE-CATALOG".into())
                .child(Button::new("graphite-node-catalog", label).on_click(
                    cx.listener(|item, _, window, cx| item.toggle_node_catalog(window, cx)),
                )),
        );
        match &self.node_catalog {
            NodeCatalogPanel::Ready { catalog, .. } => panel.child(catalog.clone()),
            NodeCatalogPanel::Failed(error) => {
                panel.child(Label::new(error.clone()).color(Color::Error))
            }
            NodeCatalogPanel::Closed | NodeCatalogPanel::Loading { .. } => panel,
        }
    }

    fn save_catalog_node(
        &mut self,
        event: &CatalogNodeInsert,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match parse_graph_grid_position(&event.position) {
            Ok(position) => self.save_copy(
                GraphiteCopy::InsertCatalogNode {
                    identifier: event.identifier,
                    position,
                },
                window,
                cx,
            ),
            Err(error) => {
                self.copy_feedback = Some((error.to_string().into(), Color::Error));
                cx.notify();
            }
        }
    }

    fn toggle_insert_node(
        &mut self,
        destination: NodeInsertDestination,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.node_insert = match self.node_insert.take() {
            Some(form) if form.destination == destination => None,
            Some(form) => Some(NodeInsertForm {
                destination,
                ..form
            }),
            None => Some(NodeInsertForm {
                position: cx.new(|cx| {
                    let mut editor = Editor::single_line(window, cx);
                    editor.set_text("0, 0", window, cx);
                    editor
                }),
                destination,
            }),
        };
        cx.notify();
    }

    fn insert_numeric_node(
        &mut self,
        operation: UnaryNumericOperation,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() {
            return;
        }
        let Some(form) = &self.node_insert else {
            return;
        };
        let position = match parse_graph_grid_position(&form.position.read(cx).text(cx)) {
            Ok(position) => position,
            Err(error) => {
                self.copy_feedback = Some((error.to_string().into(), Color::Error));
                cx.notify();
                return;
            }
        };
        match form.destination {
            NodeInsertDestination::SeparateFolder => self.save_copy(
                GraphiteCopy::InsertNumericNode {
                    operation,
                    position,
                },
                window,
                cx,
            ),
            NodeInsertDestination::WorkingCopy => {
                let revision = self.model.read(cx).revision.clone();
                self.insert_working_numeric_node(operation, position, None, revision, window, cx);
            }
        }
    }

    fn insert_working_numeric_node(
        &mut self,
        operation: UnaryNumericOperation,
        position: [i32; 2],
        network: Option<Arc<GddGraphNetwork>>,
        revision: Arc<()>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() {
            return;
        }
        let model = self.model.read(cx);
        if !Arc::ptr_eq(&model.revision, &revision) {
            self.copy_feedback = Some((
                "The Graphite working copy changed. Reopen the node picker.".into(),
                Color::Error,
            ));
            cx.notify();
            return;
        }
        let Some(contents) = model.gdd_contents().cloned() else {
            return;
        };
        self.stage_working_snapshot(
            revision,
            async move {
                if let Some(network) = network {
                    contents
                        .insert_numeric_node_in_network_copy(
                            &network,
                            operation,
                            position,
                            ARCHIVE_BUDGET,
                        )
                        .await
                } else {
                    contents
                        .insert_root_numeric_node_in_copy(operation, position, ARCHIVE_BUDGET)
                        .await
                }
            },
            window,
            cx,
        );
    }

    fn numeric_node_picker(
        pick: impl Fn(UnaryNumericOperation, &mut Window, &mut App) + Clone + 'static,
        window: &mut Window,
        cx: &mut App,
    ) -> Entity<ContextMenu> {
        ContextMenu::build(window, cx, |mut menu, _, _| {
            for operation in GRAPHITE_NUMERIC_OPERATIONS {
                let pick = pick.clone();
                menu = menu.custom_entry(
                    move |_, _| {
                        div()
                            .debug_selector(move || {
                                format!("GRAPHITE-GDD-ADD-{}", operation.name())
                            })
                            .p_2()
                            .child(Label::new(operation.name()))
                            .into_any_element()
                    },
                    move |window, cx| pick(operation, window, cx),
                );
            }
            menu
        })
    }

    fn render_insert_node(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .gap_2()
            .child(
                h_flex().gap_2().flex_wrap().children(
                    [
                        (
                            NodeInsertDestination::WorkingCopy,
                            "GRAPHITE-GDD-ADD-WORKING-NODE",
                            "Add Node to Working Copy",
                        ),
                        (
                            NodeInsertDestination::SeparateFolder,
                            "GRAPHITE-GDD-ADD-NODE",
                            "Add Node in a Copy",
                        ),
                    ]
                    .map(|(destination, selector, label)| {
                        let label = if self
                            .node_insert
                            .as_ref()
                            .is_some_and(|form| form.destination == destination)
                        {
                            "Close Add Node"
                        } else {
                            label
                        };
                        div().debug_selector(move || selector.into()).child(
                            Button::new(selector, label)
                                .disabled(self.copy_task.is_some())
                                .on_click(cx.listener(move |item, _, window, cx| {
                                    item.toggle_insert_node(destination, window, cx);
                                })),
                        )
                    }),
                ),
            )
            .when_some(
                self.node_insert.as_ref().map(|form| form.position.clone()),
                |panel, position| {
                    let item = cx.entity().downgrade();
                    let menu = Self::numeric_node_picker(
                        move |operation, window, cx| {
                            if let Some(item) = item.upgrade() {
                                item.update(cx, |item, cx| {
                                    item.insert_numeric_node(operation, window, cx);
                                });
                            }
                        },
                        window,
                        cx,
                    );
                    panel
                        .child(Label::new("Grid position (x, y)"))
                        .child(
                            div()
                                .id("graphite-gdd-new-node-position")
                                .aria_label("Grid position (x, y)")
                                .child(position),
                        )
                        .when(self.copy_task.is_none(), |panel| {
                            panel.child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-CHOOSE-NEW-NODE".into())
                                    .child(DropdownMenu::new(
                                        "graphite-gdd-choose-new-node",
                                        "Choose Numeric Node",
                                        menu,
                                    )),
                            )
                        })
                },
            )
    }

    fn toggle_node_copy(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match &self.node_copy {
            NodeCopyPanel::Loading { .. } => return,
            NodeCopyPanel::Ready(_) => self.node_copy = NodeCopyPanel::Closed,
            NodeCopyPanel::Closed | NodeCopyPanel::Failed(_) => {
                let model = self.model.read(cx);
                let Some(contents) = model.gdd_contents().cloned() else {
                    return;
                };
                let revision = model.revision.clone();
                self.node_copy = if let Some(working) = &model.working {
                    let inspection = working.inspection.clone();
                    NodeCopyPanel::Ready(Box::new(NodeCopyForm::new(
                        contents,
                        revision,
                        &inspection,
                        window,
                        cx,
                    )))
                } else {
                    let load = cx.background_spawn({
                        let contents = contents.clone();
                        async move { contents.graph_inspection().await }
                    });
                    NodeCopyPanel::Loading {
                        _task: cx.spawn_in(window, async move |item, cx| {
                            let loaded = load.await;
                            if let Err(error) = item.update_in(cx, |item, window, cx| {
                                if !matches!(item.node_copy, NodeCopyPanel::Loading { .. }) {
                                    return;
                                }
                                item.node_copy = match loaded {
                                    Ok(graph) => NodeCopyPanel::Ready(Box::new(NodeCopyForm::new(
                                        contents, revision, &graph, window, cx,
                                    ))),
                                    Err(error) => NodeCopyPanel::Failed(error.to_string().into()),
                                };
                                cx.notify();
                            }) {
                                log::debug!("Graphite node edit panel closed: {error:#}");
                            }
                        }),
                    }
                };
            }
        }
        cx.notify();
    }

    fn select_copy_node(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if let Some(node) = form
            .nodes
            .get(index)
            .filter(|node| form.includes_node(node))
        {
            if form.selected != Some(index) {
                form.output_preview = NodeOutputPreview::Hidden;
            }
            form.selected = Some(index);
            form.name
                .update(cx, |editor, cx| editor.set_text(node.name(), window, cx));
            if let Some(value) = node.numeric_input() {
                form.value.update(cx, |editor, cx| {
                    editor.set_text(value.to_string(), window, cx)
                });
            }
            if let Some([x, y]) = node.grid_position() {
                form.position.update(cx, |editor, cx| {
                    editor.set_text(format!("{x}, {y}"), window, cx)
                });
            }
            cx.notify();
        }
    }

    fn toggle_saved_graph(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if matches!(
            form.saved_graph,
            SavedGraphPanel::Loading { .. } | SavedGraphPanel::Ready(_)
        ) {
            form.saved_graph = SavedGraphPanel::Closed;
            form.saved_graph_drag_start.set(None);
            form.saved_graph_box_selection.set(None);
            form.pending_canvas_source = None;
            if form.saved_graph_focus.contains_focused(window, cx) {
                self.focus_handle.focus(window, cx);
            }
            cx.notify();
            return;
        }
        let Some(network) = form.clipboard_scope().cloned() else {
            return;
        };
        let InspectedDocument::Gdd { contents } = &self.model.read(cx).document else {
            return;
        };
        let contents = contents.clone();
        let scope = network.id;
        let load = cx.background_spawn(async move { contents.graph_geometry(&network).await });
        form.saved_graph = SavedGraphPanel::Loading {
            _task: cx.spawn_in(window, async move |item, cx| {
                let result = load.await;
                if let Err(error) = item.update_in(cx, |item, _, cx| {
                    if let NodeCopyPanel::Ready(form) = &mut item.node_copy
                        && matches!(form.saved_graph, SavedGraphPanel::Loading { .. })
                        && form
                            .clipboard_scope()
                            .is_some_and(|network| network.id == scope)
                    {
                        form.saved_graph = match result {
                            Ok(geometry) => SavedGraphPanel::Ready(geometry),
                            Err(error) => SavedGraphPanel::Failed(error.to_string().into()),
                        };
                        cx.notify();
                    }
                }) {
                    log::debug!("Graphite saved graph view closed: {error:#}");
                }
            }),
        };
        cx.notify();
    }

    fn enter_copy_network(&mut self, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        let Some(nested) = form.selected_node().and_then(GddGraphNode::nested_network) else {
            return;
        };
        if let Some(index) = form
            .networks
            .iter()
            .position(|network| network.id == nested)
        {
            form.select_network(Some(index));
            cx.notify();
        }
    }

    fn handle_saved_graph_action(
        &mut self,
        action: SavedGraphAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if !matches!(form.saved_graph, SavedGraphPanel::Ready(_)) {
            return;
        }
        match action {
            SavedGraphAction::SelectNode { index, extend } => {
                if form.pending_canvas_source.is_none() {
                    if extend {
                        self.toggle_saved_graph_node_selection(index, window, cx);
                        return;
                    }
                    form.node_selection.clear();
                }
                self.select_copy_node(index, window, cx);
            }
            SavedGraphAction::DisconnectInput(index) | SavedGraphAction::MoveNode { index, .. } => {
                self.select_copy_node(index, window, cx);
            }
            _ => {}
        }
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        match action {
            SavedGraphAction::ToggleSource(source) => {
                let eligible = match source {
                    SavedGraphConnectionSource::Import(index) => form
                        .active_network()
                        .is_some_and(|network| network.can_start_numeric_import(index)),
                    SavedGraphConnectionSource::NodeOutput(index) => form
                        .clipboard_scope()
                        .zip(form.nodes.get(index))
                        .is_some_and(|(network, node)| {
                            network.id == node.network_id() && node.can_start_numeric_output()
                        }),
                };
                if self.copy_task.is_none() && eligible {
                    form.pending_canvas_source =
                        (form.pending_canvas_source != Some(source)).then_some(source);
                    self.copy_feedback = None;
                    cx.notify();
                }
            }
            SavedGraphAction::CancelConnection => {
                form.pending_canvas_source = None;
                cx.notify();
            }
            SavedGraphAction::CopySelection => self.copy_graph_clipboard(window, cx),
            SavedGraphAction::CutSelection => self.cut_graph_selection(window, cx),
            SavedGraphAction::PasteNodes => self.paste_graph_clipboard(window, cx),
            SavedGraphAction::BoxSelection { bounds, modifiers } => {
                self.select_saved_graph_box(bounds, modifiers, window, cx);
            }
            SavedGraphAction::MoveNode { index, position } => {
                self.move_canvas_nodes(index, position, window, cx);
            }
            SavedGraphAction::ConnectExport(index) => {
                if form.pending_canvas_source.is_none() && self.copy_task.is_none() {
                    self.connect_network_export(index, window, cx);
                }
            }
            SavedGraphAction::DisconnectInput(index) => {
                if form.selected == Some(index)
                    && form.pending_canvas_source.is_none()
                    && self.copy_task.is_none()
                {
                    self.save_disconnected_numeric_input(window, cx);
                }
            }
            SavedGraphAction::SelectNode { index, .. } => {
                let Some(source) = form.pending_canvas_source.take() else {
                    return;
                };
                if self.copy_task.is_none() {
                    let (command, message) = match source {
                        SavedGraphConnectionSource::Import(index) => (
                            form.connect_network_import(index),
                            "Choose an exposed numeric input that accepts this import.",
                        ),
                        SavedGraphConnectionSource::NodeOutput(index) => (
                            form.connect_numeric_output(index),
                            "Choose an exposed numeric input that accepts this output.",
                        ),
                    };
                    if form.selected == Some(index)
                        && let Some(command) = command
                    {
                        self.apply_graph_edit(command, window, cx);
                    } else {
                        self.copy_feedback = Some((message.into(), Color::Error));
                        cx.notify();
                    }
                }
            }
        }
    }

    fn move_canvas_nodes(
        &mut self,
        index: usize,
        position: [i32; 2],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() || self.model.read(cx).working.is_none() {
            return;
        }
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if form.pending_canvas_source.is_some() {
            return;
        }
        let Some(node) = form
            .nodes
            .get(index)
            .filter(|node| form.includes_node(node))
        else {
            return;
        };
        if form.node_selection.len() < 2 || !form.node_selection.contains(&index) {
            let node = Box::new(node.clone());
            form.node_selection.clear();
            self.apply_graph_edit(
                GraphiteCopy::EditNode {
                    node,
                    change: NodeCopyChange::GridPosition(position),
                },
                window,
                cx,
            );
            return;
        }
        let (Some([x, y]), [destination_x, destination_y]) = (node.grid_position(), position)
        else {
            return;
        };
        let (Some(horizontal), Some(vertical)) =
            (destination_x.checked_sub(x), destination_y.checked_sub(y))
        else {
            return;
        };
        let Some(nodes) = form
            .node_selection
            .iter()
            .map(|&index| {
                form.nodes
                    .get(index)
                    .filter(|node| form.includes_node(node))
                    .cloned()
            })
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        let contents = form.snapshot.clone();
        let revision = form.revision.clone();
        self.stage_working_snapshot(
            revision,
            async move {
                contents
                    .move_graph_nodes_in_copy(&nodes, [horizontal, vertical], ARCHIVE_BUDGET)
                    .await
            },
            window,
            cx,
        );
    }

    fn apply_graph_edit(
        &mut self,
        request: GraphiteCopy,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() {
            return;
        }
        let model = self.model.read(cx);
        let Some(contents) = model
            .gdd_contents()
            .filter(|_| model.working.is_some())
            .cloned()
        else {
            self.save_copy(request, window, cx);
            return;
        };
        let revision = model.revision.clone();
        match request {
            GraphiteCopy::EditNode { node, change } => {
                self.stage_working_snapshot(
                    revision,
                    async move { change.stage(&contents, &node).await },
                    window,
                    cx,
                );
            }
            GraphiteCopy::EditNetworkWire(change) => {
                self.stage_working_snapshot(
                    revision,
                    async move { change.stage(&contents).await },
                    window,
                    cx,
                );
            }
            GraphiteCopy::PasteNodeGroup { network, clipboard } => {
                self.stage_working_snapshot(
                    revision,
                    async move {
                        contents
                            .paste_graph_nodes_in_copy(&network, &clipboard, ARCHIVE_BUDGET)
                            .await
                    },
                    window,
                    cx,
                );
            }
            GraphiteCopy::CutNodeGroup { nodes } => {
                let staged = cx.background_spawn(async move {
                    let (contents, text) = contents
                        .cut_graph_nodes_in_copy(&nodes, ARCHIVE_BUDGET)
                        .await?;
                    let inspection = contents.graph_inspection().await?;
                    anyhow::Ok(PreparedWorkingSnapshot {
                        contents: Arc::new(contents),
                        inspection,
                        cut_clipboard: Some(text),
                    })
                });
                self.install_working_snapshot(revision, staged, window, cx);
            }
            request => self.save_copy(request, window, cx),
        }
    }

    fn toggle_saved_graph_node_selection(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        let nodes = &form.nodes;
        let Some(node) = nodes.get(index).filter(|node| node.can_copy_graph_node()) else {
            return;
        };
        let scope = node.network_id();
        let same_scope = |index: usize| {
            nodes
                .get(index)
                .is_some_and(|node| node.network_id() == scope && node.can_copy_graph_node())
        };
        form.node_selection.retain(|&index| same_scope(index));
        if form.node_selection.is_empty() {
            form.node_selection
                .extend(form.selected.filter(|&index| same_scope(index)));
        }
        let active = if form.node_selection.remove(&index) {
            form.node_selection.first().copied()
        } else {
            form.node_selection.insert(index);
            Some(index)
        };
        if let Some(active) = active {
            self.select_copy_node(active, window, cx);
        } else {
            form.selected = None;
            form.output_preview = NodeOutputPreview::Hidden;
            cx.notify();
        }
    }

    fn select_saved_graph_box(
        &mut self,
        bounds: [[f64; 2]; 2],
        modifiers: gpui::Modifiers,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        let SavedGraphPanel::Ready(geometry) = &form.saved_graph else {
            return;
        };
        if form.pending_canvas_source.is_some() {
            return;
        }
        let Some(scope) = form.clipboard_scope().map(|network| network.id) else {
            return;
        };
        let hits = geometry
            .nodes_intersecting_box(bounds, modifiers.control)
            .collect::<BTreeSet<_>>();
        let selected = form.selected;
        let matches = form.nodes.iter().enumerate().filter_map(|(index, node)| {
            (node.network_id() == scope && hits.contains(&node.storage_id())).then_some(index)
        });
        if modifiers.alt || modifiers.shift {
            let in_scope = |index: usize| {
                form.nodes
                    .get(index)
                    .is_some_and(|node| node.network_id() == scope)
            };
            form.node_selection.retain(|&index| in_scope(index));
            if form.node_selection.is_empty() {
                form.node_selection
                    .extend(selected.filter(|&index| in_scope(index)));
            }
            if modifiers.alt && modifiers.control {
                // The pinned handler subtracts the remaining non-layer nodes from the prior selection.
                let layers = geometry
                    .nodes
                    .iter()
                    .filter(|node| node.is_layer)
                    .map(|node| node.storage_id)
                    .collect::<BTreeSet<_>>();
                form.node_selection.retain(|&index| {
                    form.nodes.get(index).is_some_and(|node| {
                        let id = node.storage_id();
                        layers.contains(&id) || hits.contains(&id)
                    })
                });
            } else if modifiers.alt {
                for index in matches {
                    form.node_selection.remove(&index);
                }
            } else {
                form.node_selection.extend(matches);
            }
        } else {
            form.node_selection = matches.collect();
        }
        if let Some(active) = selected
            .filter(|index| form.node_selection.contains(index))
            .or_else(|| form.node_selection.first().copied())
        {
            self.select_copy_node(active, window, cx);
        } else {
            form.selected = None;
            form.output_preview = NodeOutputPreview::Hidden;
            cx.notify();
        }
    }

    fn exit_copy_network(&mut self, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        let Some(parent) = form.active_network().and_then(|network| network.parent) else {
            return;
        };
        if let Some(index) = form
            .networks
            .iter()
            .position(|network| network.id == parent)
        {
            form.select_network(Some(index));
            cx.notify();
        }
    }

    fn show_all_graph_nodes(&mut self, cx: &mut Context<Self>) {
        if let NodeCopyPanel::Ready(form) = &mut self.node_copy {
            form.select_network(None);
            cx.notify();
        }
    }

    fn append_network_port(
        &mut self,
        kind: NetworkPortKind,
        with_hidden_primary: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        if let Some(network) = form.active_network().filter(|network| {
            network.can_extend_ports_in_copy() && (!with_hidden_primary || kind.count(network) == 0)
        }) {
            self.save_copy(
                GraphiteCopy::AppendNetworkPort {
                    network: Box::new(network.clone()),
                    kind,
                    with_hidden_primary,
                },
                window,
                cx,
            );
        }
    }

    fn connect_network_export(
        &mut self,
        export_index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some((source, network)) = form.selected_node().zip(form.active_network()) else {
            return;
        };
        if network.can_connect_export_from(source)
            && network.empty_exports().contains(&export_index)
        {
            self.apply_graph_edit(
                GraphiteCopy::EditNetworkWire(NetworkWireChange::Export {
                    network: Box::new(network.clone()),
                    source: Box::new(source.clone()),
                    export_index,
                }),
                window,
                cx,
            );
        }
    }

    fn connect_network_import(
        &mut self,
        import_index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        if let Some(command) = form.connect_network_import(import_index) {
            self.apply_graph_edit(command, window, cx);
        }
    }

    fn toggle_primary_network_port(
        &mut self,
        kind: NetworkPortKind,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(network) = form
            .active_network()
            .filter(|network| network.can_extend_ports_in_copy())
        else {
            return;
        };
        if let Some(exposed) = kind.primary_exposure(network) {
            self.save_copy(
                GraphiteCopy::SetPrimaryNetworkPortExposure {
                    network: Box::new(network.clone()),
                    kind,
                    exposed: !exposed,
                },
                window,
                cx,
            );
        }
    }

    fn edit_network_port(
        &mut self,
        network: &GddGraphNetwork,
        change: NetworkPortChange,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        if form
            .active_network()
            .is_some_and(|active| active.id == network.id)
        {
            self.save_copy(
                GraphiteCopy::EditNetworkPort {
                    network: Box::new(network.clone()),
                    change,
                },
                window,
                cx,
            );
        }
    }

    fn network_port_menu(
        item: WeakEntity<Self>,
        network: &GddGraphNetwork,
        kind: NetworkPortKind,
        indices: impl IntoIterator<Item = usize>,
        window: &mut Window,
        cx: &mut App,
    ) -> Entity<ContextMenu> {
        let (noun, upper) = match kind {
            NetworkPortKind::Import => ("Import", "IMPORT"),
            NetworkPortKind::Export => ("Export", "EXPORT"),
        };
        let count = kind.count(network);
        let network = Arc::new(network.clone());
        ContextMenu::build(window, cx, |mut menu, _, _| {
            for index in indices {
                let port = match kind.name(&network, index) {
                    Some(name) => format!("{noun} {} ({name})", index + 1),
                    None => format!("{noun} {}", index + 1),
                };
                let commands = [
                    (
                        NetworkPortMenuAction::Copy(NetworkPortChange::Remove { kind, index }),
                        format!("Remove {port}"),
                        format!("GRAPHITE-GDD-REMOVE-{upper}-{index}"),
                    ),
                    (
                        NetworkPortMenuAction::Rename,
                        format!("Rename {port}"),
                        format!("GRAPHITE-GDD-RENAME-{upper}-{index}"),
                    ),
                ];
                let moves = [
                    (index.checked_sub(1), "Up", "UP"),
                    ((index + 1 < count).then_some(index + 1), "Down", "DOWN"),
                ]
                .into_iter()
                .filter_map(|(target, direction, suffix)| {
                    Some((
                        NetworkPortMenuAction::Copy(NetworkPortChange::Move {
                            kind,
                            from: index,
                            to: target?,
                        }),
                        format!("Move {port} {direction}"),
                        format!("GRAPHITE-GDD-MOVE-{upper}-{index}-{suffix}"),
                    ))
                });
                for (action, label, selector) in commands.into_iter().chain(moves) {
                    let item = item.clone();
                    let network = network.clone();
                    menu = menu.custom_entry(
                        move |_, _| {
                            div()
                                .debug_selector({
                                    let selector = selector.clone();
                                    move || selector.clone()
                                })
                                .p_2()
                                .child(Label::new(label.clone()))
                                .into_any_element()
                        },
                        move |window, cx| {
                            if let Some(item) = item.upgrade() {
                                item.update(cx, |item, cx| match action {
                                    NetworkPortMenuAction::Copy(change) => {
                                        item.edit_network_port(&network, change, window, cx)
                                    }
                                    NetworkPortMenuAction::Rename => item
                                        .begin_network_port_rename(
                                            &network, kind, index, window, cx,
                                        ),
                                });
                            }
                        },
                    );
                }
            }
            menu
        })
    }

    fn begin_network_port_rename(
        &mut self,
        network: &GddGraphNetwork,
        kind: NetworkPortKind,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if !form
            .active_network()
            .is_some_and(|active| active.id == network.id)
            || index >= kind.count(network)
        {
            return;
        }
        let current_name = kind.name(network, index).unwrap_or_default();
        let name = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text(current_name, window, cx);
            editor
        });
        name.focus_handle(cx).focus(window, cx);
        form.port_rename = Some(PortRenameForm { kind, index, name });
        cx.notify();
    }

    fn save_network_port_name(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some((network, port)) = form.active_network().zip(form.port_rename.as_ref()) else {
            return;
        };
        self.save_copy(
            GraphiteCopy::RenameNetworkPort {
                network: Box::new(network.clone()),
                port: port.kind.port(port.index),
                name: port.name.read(cx).text(cx),
            },
            window,
            cx,
        );
    }

    fn toggle_node_output_preview(&mut self, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if matches!(
            form.output_preview,
            NodeOutputPreview::Pending { .. } | NodeOutputPreview::Ready(_)
        ) {
            form.output_preview = NodeOutputPreview::Hidden;
            cx.notify();
            return;
        }
        let Some(node) = form
            .selected_node()
            .filter(|node| node.can_preview_numeric_output())
        else {
            return;
        };
        let InspectedDocument::Gdd { contents } = &self.model.read(cx).document else {
            return;
        };
        let selected = form.selected;
        let node = node.clone();
        let contents = contents.clone();
        let evaluation = cx.background_spawn(async move {
            futures::executor::block_on(contents.preview_graph_numeric_output(&node))
        });
        form.output_preview = NodeOutputPreview::Pending {
            _task: cx.spawn(async move |item, cx| {
                let result = evaluation.await;
                if let Err(error) = item.update(cx, |item, cx| {
                    let NodeCopyPanel::Ready(form) = &mut item.node_copy else {
                        return;
                    };
                    if form.selected != selected
                        || !matches!(form.output_preview, NodeOutputPreview::Pending { .. })
                    {
                        return;
                    }
                    form.output_preview = match result {
                        Ok(value) => NodeOutputPreview::Ready(value),
                        Err(error) => NodeOutputPreview::Failed(error.to_string().into()),
                    };
                    cx.notify();
                }) {
                    log::debug!("Graphite node preview closed: {error:#}");
                }
            }),
        };
        cx.notify();
    }

    fn toggle_graph_selection(&mut self, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        let Some(index) = form.selected else {
            return;
        };
        let Some(node) = form
            .nodes
            .get(index)
            .filter(|node| node.can_copy_graph_node())
        else {
            return;
        };
        if form.node_selection.remove(&index) {
            cx.notify();
            return;
        }
        if form
            .node_selection
            .iter()
            .filter_map(|index| form.nodes.get(*index))
            .any(|selected| selected.network_id() != node.network_id())
        {
            self.copy_feedback = Some(("Select nodes in the same network.".into(), Color::Error));
        } else {
            form.node_selection.insert(index);
            self.copy_feedback = None;
        }
        cx.notify();
    }

    fn copy_graph_clipboard(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let targets = form.selected_graph_nodes();
        if targets.is_empty() || self.copy_task.is_some() {
            return;
        }
        let InspectedDocument::Gdd { contents } = &self.model.read(cx).document else {
            return;
        };
        let contents = contents.clone();
        let copy = cx.background_spawn(async move {
            contents
                .copy_graph_nodes_to_clipboard(&targets, ARCHIVE_BUDGET)
                .await
        });
        self.copy_task = Some(cx.spawn_in(window, async move |item, cx| {
            let result = copy.await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                item.copy_task = None;
                item.copy_feedback = Some(match result {
                    Ok(text) => {
                        cx.write_to_clipboard(ClipboardItem::new_string(text));
                        ("Copied Graphite nodes.".into(), Color::Success)
                    }
                    Err(error) => (error.to_string().into(), Color::Error),
                });
                cx.notify();
            }) {
                log::debug!("Graphite clipboard source closed: {error:#}");
            }
        }));
        cx.notify();
    }

    fn paste_graph_clipboard(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(network) = form.clipboard_scope() else {
            return;
        };
        let Some(clipboard) = cx
            .read_from_clipboard()
            .and_then(|item| item.text())
            .filter(|text| text.starts_with("graphite: "))
        else {
            self.copy_feedback =
                Some(("The clipboard has no Graphite nodes.".into(), Color::Error));
            cx.notify();
            return;
        };
        self.apply_graph_edit(
            GraphiteCopy::PasteNodeGroup {
                network: Box::new(network.clone()),
                clipboard,
            },
            window,
            cx,
        );
    }

    fn cut_graph_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let nodes = form.selected_graph_nodes();
        match nodes.as_slice() {
            [] => {}
            [node] if !node.can_delete_in_copy() => {
                self.copy_feedback = Some((
                    "Select an isolated node or a complete connected numeric group to cut.".into(),
                    Color::Error,
                ));
                cx.notify();
            }
            _ => self.apply_graph_edit(GraphiteCopy::CutNodeGroup { nodes }, window, cx),
        }
    }

    fn duplicate_graph_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let nodes = form.selected_graph_nodes();
        let Some(first) = nodes.first().filter(|_| nodes.len() > 1) else {
            return;
        };
        let Some(network) = form
            .networks
            .iter()
            .find(|network| network.id == first.network_id())
        else {
            return;
        };
        self.save_copy(
            GraphiteCopy::DuplicateNodeGroup {
                network: Box::new(network.clone()),
                nodes,
            },
            window,
            cx,
        );
    }

    fn render_node_output_preview(
        preview: &NodeOutputPreview,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let label = match preview {
            NodeOutputPreview::Hidden => "Preview Saved Output",
            NodeOutputPreview::Pending { .. } => "Cancel Numeric Preview",
            NodeOutputPreview::Ready(_) => "Hide Numeric Preview",
            NodeOutputPreview::Failed(_) => "Retry Numeric Preview",
        };
        let result = match preview {
            NodeOutputPreview::Ready(GddNumericValue::NumberList(values)) if values.is_empty() => {
                Some(("[]".into(), Color::Default))
            }
            NodeOutputPreview::Ready(value) => Some((value.to_string(), Color::Default)),
            NodeOutputPreview::Failed(error) => Some((error.to_string(), Color::Error)),
            NodeOutputPreview::Hidden | NodeOutputPreview::Pending { .. } => None,
        };
        v_flex()
            .gap_2()
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-GDD-PREVIEW-NUMERIC".into())
                    .child(Button::new("graphite-gdd-preview-numeric", label).on_click(
                        cx.listener(|item, _, _, cx| item.toggle_node_output_preview(cx)),
                    )),
            )
            .when_some(result, |panel, (message, color)| {
                panel.child(
                    div()
                        .debug_selector(|| "GRAPHITE-GDD-NUMERIC-RESULT".into())
                        .child(Label::new(message).color(color)),
                )
            })
    }

    fn save_node_grid_position(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(node) = form.selected_node() else {
            return;
        };
        let request =
            parse_graph_grid_position(&form.position.read(cx).text(cx)).and_then(|position| {
                node.validate_grid_move(position)?;
                Ok(GraphiteCopy::EditNode {
                    node: Box::new(node.clone()),
                    change: NodeCopyChange::GridPosition(position),
                })
            });
        match request {
            Ok(request) => self.save_copy(request, window, cx),
            Err(error) => {
                self.copy_feedback = Some((error.to_string().into(), Color::Error));
                cx.notify();
            }
        }
    }

    fn save_numeric_value(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.change_numeric_value(false, window, cx);
    }

    fn change_numeric_value(
        &mut self,
        in_working_copy: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() {
            return;
        }
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(node) = form.selected_node() else {
            return;
        };
        let Some(original_value) = node.numeric_input() else {
            return;
        };
        let value = match original_value.parse_like(&form.value.read(cx).text(cx)) {
            Ok(value) => value,
            Err(error) => {
                self.copy_feedback = Some((error.to_string().into(), Color::Error));
                cx.notify();
                return;
            }
        };
        if in_working_copy {
            let snapshot = form.snapshot.clone();
            let expected = form.revision.clone();
            let node = node.clone();
            let edit = async move {
                snapshot
                    .set_graph_numeric_input_in_copy(&node, value, ARCHIVE_BUDGET)
                    .await
            };
            self.stage_working_snapshot(expected, edit, window, cx);
        } else {
            self.save_copy(
                GraphiteCopy::EditNode {
                    node: Box::new(node.clone()),
                    change: NodeCopyChange::NumericInput(value),
                },
                window,
                cx,
            );
        }
    }

    fn stage_working_snapshot(
        &mut self,
        expected: Arc<()>,
        edit: impl Future<Output = Result<InspectedGddV1, GddArchiveError>> + Send + 'static,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let staged = cx.background_spawn(async move {
            let contents = edit.await?;
            let inspection = contents.graph_inspection().await?;
            anyhow::Ok(PreparedWorkingSnapshot {
                contents: Arc::new(contents),
                inspection,
                cut_clipboard: None,
            })
        });
        self.install_working_snapshot(expected, staged, window, cx);
    }

    fn install_working_snapshot(
        &mut self,
        expected: Arc<()>,
        staged: Task<anyhow::Result<PreparedWorkingSnapshot>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.copy_feedback = None;
        self.copy_task = Some(cx.spawn_in(window, async move |item, cx| {
            let staged = staged.await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                item.copy_task = None;
                let result = staged.and_then(|staged| {
                    item.model.update(cx, |model, cx| {
                        model.replace_working_snapshot(
                            &expected,
                            staged.contents,
                            staged.inspection,
                            cx,
                        )
                    })?;
                    Ok(staged.cut_clipboard)
                });
                match result {
                    Ok(clipboard) => {
                        if let Some(text) = clipboard {
                            cx.write_to_clipboard(ClipboardItem::new_string(text));
                            item.copy_feedback =
                                Some(("Cut Graphite nodes.".into(), Color::Success));
                        }
                        cx.emit(ItemEvent::Edit);
                    }
                    Err(error) => {
                        item.copy_feedback = Some((format!("{error:#}").into(), Color::Error));
                    }
                }
                cx.notify();
            }) {
                log::debug!("Graphite edit view closed: {error:#}");
            }
        }));
        cx.notify();
    }

    fn move_working_history(
        &mut self,
        direction: GddHistoryDirection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() {
            return;
        }
        let model = self.model.read(cx);
        let Some(contents) = model
            .gdd_contents()
            .filter(|contents| model.working.is_some() && contents.can_move_history(direction))
            .cloned()
        else {
            return;
        };
        let revision = model.revision.clone();
        self.stage_working_snapshot(
            revision,
            async move {
                contents
                    .move_history_in_copy(direction, ARCHIVE_BUDGET)
                    .await
            },
            window,
            cx,
        );
    }

    fn route_working_history(
        &mut self,
        direction: GddHistoryDirection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let editor_focused = |editor: &Entity<Editor>| editor.read(cx).is_focused(window);
        let field_focused = self
            .node_insert
            .as_ref()
            .is_some_and(|form| editor_focused(&form.position))
            || matches!(&self.node_copy, NodeCopyPanel::Ready(form) if
                [&form.name, &form.value, &form.position].into_iter().any(editor_focused)
                    || form.port_rename.as_ref().is_some_and(|rename| editor_focused(&rename.name)));
        if field_focused || self.model.read(cx).working.is_none() {
            cx.propagate();
            return;
        }
        self.move_working_history(direction, window, cx);
        cx.stop_propagation();
    }

    fn save_numeric_input_exposure(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(node) = form.selected_node() else {
            return;
        };
        if let Some(exposed) = node.numeric_input_exposed() {
            let request = GraphiteCopy::EditNode {
                node: Box::new(node.clone()),
                change: NodeCopyChange::NumericInputExposure(!exposed),
            };
            self.save_copy(request, window, cx);
        }
    }

    fn save_disconnected_numeric_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        if let Some(node) = form
            .selected_node()
            .filter(|node| node.has_numeric_connection())
        {
            let request = GraphiteCopy::EditNode {
                node: Box::new(node.clone()),
                change: NodeCopyChange::DisconnectNumericInput,
            };
            self.apply_graph_edit(request, window, cx);
        }
    }

    fn numeric_wire_insertion_menu(
        item: WeakEntity<Self>,
        target: &GddGraphNode,
        window: &mut Window,
        cx: &mut App,
    ) -> Entity<ContextMenu> {
        ContextMenu::build(window, cx, |mut menu, _, _| {
            for operation in GRAPHITE_NUMERIC_OPERATIONS {
                let item = item.clone();
                let target = target.clone();
                menu = menu.custom_entry(
                    move |_, _| {
                        div()
                            .debug_selector(move || {
                                format!("GRAPHITE-GDD-INSERT-ON-WIRE-{}", operation.name())
                            })
                            .p_2()
                            .child(Label::new(operation.name()))
                            .into_any_element()
                    },
                    move |window, cx| {
                        if let Some(item) = item.upgrade() {
                            item.update(cx, |item, cx| {
                                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                                    return;
                                };
                                if form.includes_node(&target) {
                                    item.apply_graph_edit(
                                        GraphiteCopy::EditNode {
                                            node: Box::new(target.clone()),
                                            change: NodeCopyChange::InsertNumericOnWire(operation),
                                        },
                                        window,
                                        cx,
                                    );
                                }
                            });
                        }
                    },
                );
            }
            menu
        })
    }

    fn render_numeric_wire_insertion(
        &self,
        target: &GddGraphNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let menu = Self::numeric_wire_insertion_menu(cx.entity().downgrade(), target, window, cx);
        let label = if self.model.read(cx).working.is_some() {
            "Insert a Node on Wire"
        } else {
            "Insert a Node on Wire in a Copy"
        };
        div()
            .debug_selector(|| "GRAPHITE-GDD-INSERT-ON-WIRE".into())
            .child(DropdownMenu::new(
                "graphite-gdd-insert-on-wire",
                label,
                menu,
            ))
    }

    fn render_numeric_connection_menu(
        &self,
        form: &NodeCopyForm,
        target: &GddGraphNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        let previous = target.numeric_connection_source(&form.nodes);
        let sources = form
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, source)| match previous {
                Some(previous) => target.can_reconnect_numeric_input_from(previous, source),
                None => target.can_connect_numeric_input_from(source),
            })
            .collect::<Vec<_>>();
        if sources.is_empty() || self.copy_task.is_some() {
            return None;
        }
        let item = cx.entity().downgrade();
        let menu = ContextMenu::build(window, cx, |mut menu, _, _| {
            for (index, source) in sources {
                let item = item.clone();
                let target = target.clone();
                let source = source.clone();
                let previous = previous.cloned();
                let label = SharedString::from(source.label().to_owned());
                menu = menu.custom_entry(
                    move |_, _| {
                        div()
                            .debug_selector(move || format!("GRAPHITE-GDD-CONNECT-SOURCE-{index}"))
                            .p_2()
                            .child(Label::new(label.clone()))
                            .into_any_element()
                    },
                    move |window, cx| {
                        if let Some(item) = item.upgrade() {
                            item.update(cx, |item, cx| {
                                item.save_copy(
                                    GraphiteCopy::EditNode {
                                        node: Box::new(target.clone()),
                                        change: NodeCopyChange::ConnectNumericInput {
                                            source: Box::new(source.clone()),
                                            previous: previous.clone().map(Box::new),
                                        },
                                    },
                                    window,
                                    cx,
                                );
                            });
                        }
                    },
                );
            }
            menu
        });
        let (selector, label) = if previous.is_some() {
            ("GRAPHITE-GDD-RECONNECT-VALUE", "Reconnect Value in a Copy")
        } else {
            ("GRAPHITE-GDD-CONNECT-VALUE", "Connect Value in a Copy")
        };
        Some(
            div()
                .debug_selector(move || selector.into())
                .child(DropdownMenu::new(selector, label, menu)),
        )
    }

    fn render_node_copy(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let label = match self.node_copy {
            NodeCopyPanel::Closed if self.model.read(cx).working.is_some() => "Open Node Editor",
            NodeCopyPanel::Closed => "Edit Node in a Copy",
            NodeCopyPanel::Loading { .. } => "Loading Graph Nodes",
            NodeCopyPanel::Ready(_) => "Close Node Edit",
            NodeCopyPanel::Failed(_) => "Retry Graph Load",
        };
        let panel = v_flex().gap_2().child(
            div()
                .debug_selector(|| "GRAPHITE-GDD-NODE-COPY".into())
                .child(
                    Button::new("graphite-gdd-node-copy", label)
                        .disabled(
                            self.copy_task.is_some()
                                || matches!(self.node_copy, NodeCopyPanel::Loading { .. }),
                        )
                        .on_click(
                            cx.listener(|item, _, window, cx| item.toggle_node_copy(window, cx)),
                        ),
                ),
        );
        match &self.node_copy {
            NodeCopyPanel::Ready(form) => {
                let working = self.model.read(cx).working.is_some();
                let edit_suffix = if working { "" } else { " in a Copy" };
                let item = cx.entity().downgrade();
                let menu = ContextMenu::build(window, cx, |mut menu, _, _| {
                    for (index, node) in form
                        .nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, node)| form.includes_node(node))
                    {
                        let item = item.clone();
                        let label = SharedString::from(node.label().to_owned());
                        menu = menu.custom_entry(
                            move |_, _| {
                                div()
                                    .debug_selector(move || {
                                        format!("GRAPHITE-GDD-NODE-COPY-ITEM-{index}")
                                    })
                                    .p_2()
                                    .child(Label::new(label.clone()))
                                    .into_any_element()
                            },
                            move |window, cx| {
                                if let Some(item) = item.upgrade() {
                                    item.update(cx, |item, cx| {
                                        item.select_copy_node(index, window, cx)
                                    });
                                }
                            },
                        );
                    }
                    menu
                });
                let selected = form.selected_node();
                let viewed_network = selected
                    .and_then(|node| {
                        form.networks
                            .iter()
                            .find(|network| network.id == node.network_id())
                    })
                    .or_else(|| form.clipboard_scope());
                let selected_count = if form.node_selection.is_empty() {
                    usize::from(selected.is_some_and(|node| node.can_copy_graph_node()))
                } else {
                    form.node_selection.len()
                };
                let has_nodes = form.nodes.iter().any(|node| form.includes_node(node));
                let empty_exports = selected
                    .zip(form.active_network())
                    .filter(|(source, network)| network.can_connect_export_from(source))
                    .map_or(&[][..], |(_, network)| network.empty_exports());
                panel
                    .when_some(form.active_network(), |panel, network| {
                        panel
                            .child(Label::new(format!(
                                "{} · Imports {} · Exports {}",
                                network.name, network.import_count, network.export_count
                            )))
                            .child(
                                h_flex()
                                    .gap_2()
                                    .flex_wrap()
                                    .when(network.parent.is_some(), |bar| {
                                        bar.child(
                                            div()
                                                .debug_selector(|| "GRAPHITE-GDD-PARENT-NETWORK".into())
                                                .child(Button::new("graphite-gdd-parent-network", "Parent Network").on_click(
                                                    cx.listener(|item, _, _, cx| item.exit_copy_network(cx)),
                                                )),
                                        )
                                    })
                                    .child(
                                        div()
                                            .debug_selector(|| "GRAPHITE-GDD-ALL-GRAPH-NODES".into())
                                            .child(Button::new("graphite-gdd-all-graph-nodes", "Show All Nodes").on_click(
                                                cx.listener(|item, _, _, cx| item.show_all_graph_nodes(cx)),
                                            )),
                                    )
                                    .when(network.can_extend_ports_in_copy(), |bar| {
                                        bar.children([
                                            (
                                                NetworkPortKind::Import,
                                                "GRAPHITE-GDD-ADD-NETWORK-IMPORT",
                                                if network.import_count == 0 {
                                                    "Add Primary Import in a Copy"
                                                } else {
                                                    "Add Secondary Import in a Copy"
                                                },
                                            ),
                                            (
                                                NetworkPortKind::Export,
                                                "GRAPHITE-GDD-ADD-NETWORK-EXPORT",
                                                if network.export_count == 0 {
                                                    "Add Primary Export in a Copy"
                                                } else {
                                                    "Add Secondary Export in a Copy"
                                                },
                                            ),
                                        ].map(|(kind, selector, label)| {
                                            div().debug_selector(move || selector.into())
                                                .child(Button::new(selector, label)
                                                    .disabled(self.copy_task.is_some())
                                                    .on_click(cx.listener(move |item, _, window, cx| {
                                                        item.append_network_port(kind, false, window, cx);
                                                    })))
                                        }))
                                        .children([
                                            (NetworkPortKind::Import, "GRAPHITE-GDD-ADD-SECONDARY-IMPORT", "Add Secondary Import in a Copy"),
                                            (NetworkPortKind::Export, "GRAPHITE-GDD-ADD-SECONDARY-EXPORT", "Add Secondary Export in a Copy"),
                                        ].into_iter().filter(|(kind, ..)| kind.count(network) == 0)
                                            .map(|(kind, selector, label)| {
                                                div().debug_selector(move || selector.into())
                                                    .child(Button::new(selector, label)
                                                        .disabled(self.copy_task.is_some())
                                                        .on_click(cx.listener(move |item, _, window, cx| {
                                                            item.append_network_port(kind, true, window, cx);
                                                        })))
                                            }))
                                        .children([
                                            (NetworkPortKind::Import, "Import", "GRAPHITE-GDD-PRIMARY-IMPORT-EXPOSURE"),
                                            (NetworkPortKind::Export, "Export", "GRAPHITE-GDD-PRIMARY-EXPORT-EXPOSURE"),
                                        ].into_iter().filter_map(|(kind, noun, selector)| {
                                            let exposed = kind.primary_exposure(network)?;
                                            let action = if exposed { "Hide" } else { "Expose" };
                                            Some(
                                                div()
                                                    .debug_selector(move || selector.into())
                                                    .child(Button::new(selector, format!("{action} Primary {noun} in a Copy"))
                                                        .disabled(self.copy_task.is_some())
                                                        .on_click(cx.listener(move |item, _, window, cx| {
                                                            item.toggle_primary_network_port(kind, window, cx);
                                                        }))),
                                            )
                                        }))
                                    }),
                            )
                            .when(network.can_extend_ports_in_copy() && self.copy_task.is_none(), |panel| {
                                panel.child(h_flex().gap_2().flex_wrap().children(
                                    [
                                        (NetworkPortKind::Import, "Import", "GRAPHITE-GDD-REMOVE-IMPORT"),
                                        (NetworkPortKind::Export, "Export", "GRAPHITE-GDD-REMOVE-EXPORT"),
                                    ].into_iter().filter_map(|(kind, noun, selector)| {
                                        let count = kind.count(network);
                                        if count == 0 {
                                            return None;
                                        }
                                        let menu = Self::network_port_menu(
                                            cx.entity().downgrade(), network, kind, 0..count, window, cx,
                                        );
                                        Some(div()
                                            .debug_selector(move || selector.into())
                                            .child(DropdownMenu::new(
                                                selector,
                                                format!("Edit {noun}s in a Copy"),
                                                menu,
                                            )))
                                    }),
                                ))
                            })
                            .when_some(form.port_rename.as_ref(), |panel, form| {
                                panel
                                    .child(Label::new(match form.kind {
                                        NetworkPortKind::Import => format!("Import {} name", form.index + 1),
                                        NetworkPortKind::Export => format!("Export {} name", form.index + 1),
                                    }))
                                    .child(
                                        div()
                                            .debug_selector(|| "GRAPHITE-GDD-PORT-NAME".into())
                                            .child(form.name.clone()),
                                    )
                                    .child(h_flex().gap_2().children([
                                        div().debug_selector(|| "GRAPHITE-GDD-SAVE-PORT-NAME".into()).child(
                                            Button::new("graphite-gdd-save-port-name", "Save Renamed Folder")
                                                .disabled(self.copy_task.is_some())
                                                .on_click(cx.listener(|item, _, window, cx| item.save_network_port_name(window, cx))),
                                        ),
                                        div().debug_selector(|| "GRAPHITE-GDD-CANCEL-PORT-NAME".into()).child(
                                            Button::new("graphite-gdd-cancel-port-name", "Cancel")
                                                .on_click(cx.listener(|item, _, window, cx| {
                                                    if let NodeCopyPanel::Ready(form) = &mut item.node_copy {
                                                        form.port_rename = None;
                                                        item.focus_handle.focus(window, cx);
                                                        cx.notify();
                                                    }
                                                })),
                                        ),
                                    ]))
                            })
                            .when_some(selected, |panel, node| {
                                let mut imports = network.compatible_numeric_imports(node).peekable();
                                let action = if node.numeric_import_source().is_some() {
                                    "Reconnect to"
                                } else {
                                    "Connect"
                                };
                                panel.when(imports.peek().is_some(), |panel| {
                                    panel.child(h_flex().gap_2().flex_wrap().children(imports.map(|index| {
                                        let label = match network.import_name(index) {
                                            Some(name) => format!("{action} Import {} ({name}){edit_suffix}", index + 1),
                                            None => format!("{action} Import {}{edit_suffix}", index + 1),
                                        };
                                        div()
                                            .debug_selector(move || format!("GRAPHITE-GDD-CONNECT-IMPORT-{index}"))
                                            .child(Button::new(("graphite-gdd-connect-import", index), label)
                                                .disabled(self.copy_task.is_some())
                                                .on_click(cx.listener(move |item, _, window, cx| {
                                                    item.connect_network_import(index, window, cx);
                                                })))
                                    })))
                                })
                            })
                    })
                    .when(has_nodes, |panel| panel.child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-NODE-COPY-CHOOSE".into())
                            .child(DropdownMenu::new(
                                "graphite-gdd-node-copy-choose",
                                selected.map_or_else(
                                    || "Choose a graph node".to_owned(),
                                    |node| node.label().to_owned(),
                                ),
                                menu,
                            )),
                    ))
                    .when_some(viewed_network, |panel, network| {
                        let parent = network.parent.and_then(|id| {
                            form.networks.iter().find(|network| network.id == id)
                        });
                        panel.child(
                            div().debug_selector(|| "GRAPHITE-GDD-TOGGLE-SAVED-PORTS".into()).child(
                                Button::new("graphite-gdd-toggle-saved-ports", if form.show_saved_ports {
                                    "Hide Saved Ports"
                                } else {
                                    "Show Saved Ports"
                                })
                                .on_click(cx.listener(|item, _, _, cx| {
                                    if let NodeCopyPanel::Ready(form) = &mut item.node_copy {
                                        form.show_saved_ports = !form.show_saved_ports;
                                        cx.notify();
                                    }
                                })),
                            ),
                        ).when(form.show_saved_ports, |panel| {
                            panel.child(crate::graph::render_saved_graph_ports(
                                network, parent, selected, &form.nodes,
                            ))
                        })
                    })
                    .when_some(form.clipboard_scope(), |panel, network| {
                        let label = match &form.saved_graph {
                            SavedGraphPanel::Closed | SavedGraphPanel::Failed(_) => "Show Saved Graph",
                            SavedGraphPanel::Loading { .. } => "Cancel Graph Loading",
                            SavedGraphPanel::Ready(_) => "Hide Saved Graph",
                        };
                        let panel = panel.child(div().debug_selector(|| "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH".into()).child(
                            Button::new("graphite-gdd-toggle-saved-graph", label)
                                .on_click(cx.listener(|item, _, window, cx| item.toggle_saved_graph(window, cx))),
                        ));
                        match &form.saved_graph {
                            SavedGraphPanel::Ready(geometry) => {
                                let item = cx.entity().downgrade();
                                let menu_item = item.clone();
                                let scope = network.id;
                                let visible_revision = form.revision.clone();
                                let menu_revision = visible_revision.clone();
                                panel.child(crate::graph::render_saved_graph(
                                    geometry, &form.nodes,
                                    SavedGraphView {
                                        network,
                                        focus: &form.saved_graph_focus,
                                        selected: form.selected,
                                        selection: &form.node_selection,
                                        pending_source: form.pending_canvas_source,
                                        can_start_connection: self.copy_task.is_none(),
                                        revision: &form.revision,
                                        drag_start: self.model.read(cx).working.as_ref().map(|_| &form.saved_graph_drag_start),
                                        box_selection: &form.saved_graph_box_selection,
                                    },
                                    window, cx,
                                    move |action, window, cx| {
                                        if let Err(error) = item.update(cx, |item, cx| {
                                            if let NodeCopyPanel::Ready(form) = &item.node_copy
                                                && Arc::ptr_eq(&form.revision, &visible_revision)
                                                && form.clipboard_scope().is_some_and(|network| network.id == scope)
                                            {
                                                item.handle_saved_graph_action(action, window, cx);
                                            }
                                        }) {
                                            log::debug!("Graphite saved graph selection closed: {error:#}");
                                        }
                                    },
                                    move |target, window, cx| {
                                        let item = menu_item.upgrade()?;
                                        let item_state = item.read(cx);
                                        let NodeCopyPanel::Ready(form) = &item_state.node_copy else {
                                            return None;
                                        };
                                        if item_state.copy_task.is_some()
                                            || form.pending_canvas_source.is_some()
                                            || !matches!(form.saved_graph, SavedGraphPanel::Ready(_))
                                            || !Arc::ptr_eq(&form.revision, &menu_revision)
                                            || !form.clipboard_scope().is_some_and(|network| network.id == scope)
                                        {
                                            return None;
                                        }
                                        match target {
                                            SavedGraphMenuTarget::Background(position) => {
                                                let network = Arc::new(form.clipboard_scope()?.clone());
                                                let revision = form.revision.clone();
                                                let item = item.downgrade();
                                                Some(Self::numeric_node_picker(move |operation, window, cx| {
                                                    if let Err(error) = item.update(cx, |item, cx| {
                                                        if matches!(&item.node_copy, NodeCopyPanel::Ready(form) if
                                                            matches!(form.saved_graph, SavedGraphPanel::Ready(_))
                                                                && form.pending_canvas_source.is_none()
                                                                && form.clipboard_scope().is_some_and(|network| network.id == scope))
                                                        {
                                                            item.insert_working_numeric_node(operation, position, Some(network.clone()), revision.clone(), window, cx);
                                                        } else {
                                                            item.copy_feedback = Some(("The Graphite canvas changed. Reopen the node picker.".into(), Color::Error));
                                                            cx.notify();
                                                        }
                                                    }) {
                                                        log::debug!("Graphite canvas picker closed: {error:#}");
                                                    }
                                                }, window, cx))
                                            }
                                            SavedGraphMenuTarget::NodeInput(index) => {
                                                let target = form.nodes.get(index).filter(|node| {
                                                    node.network_id() == scope
                                                        && node.has_numeric_connection()
                                                        && node.can_preview_numeric_output()
                                                })?.clone();
                                                Some(Self::numeric_wire_insertion_menu(menu_item.clone(), &target, window, cx))
                                            }
                                            SavedGraphMenuTarget::Boundary(port) => {
                                                let (kind, index) = match port {
                                                    ImportOrExport::Import(index) => (NetworkPortKind::Import, index),
                                                    ImportOrExport::Export(index) => (NetworkPortKind::Export, index),
                                                };
                                                let network = form.active_network().filter(|network| {
                                                    network.id == scope
                                                        && network.can_extend_ports_in_copy()
                                                        && index < kind.count(network)
                                                })?.clone();
                                                Some(Self::network_port_menu(menu_item.clone(), &network, kind, [index], window, cx))
                                            }
                                        }
                                    },
                                ))
                            }
                            SavedGraphPanel::Failed(error) => panel.child(Label::new(error.clone()).color(Color::Error)),
                            SavedGraphPanel::Closed | SavedGraphPanel::Loading { .. } => panel,
                        }
                    })
                    .child(h_flex().gap_2().flex_wrap().children([
                        div().debug_selector(|| "GRAPHITE-GDD-COPY-NODES".into()).child(
                            Button::new("graphite-gdd-copy-nodes", match selected_count {
                                0 => "Copy Nodes".into(),
                                1 => "Copy Node".into(),
                                count => format!("Copy {count} Nodes"),
                            })
                            .disabled(selected_count == 0 || self.copy_task.is_some())
                            .on_click(cx.listener(|item, _, window, cx| item.copy_graph_clipboard(window, cx))),
                        ),
                        div().debug_selector(|| "GRAPHITE-GDD-PASTE-NODES".into()).child(
                            Button::new("graphite-gdd-paste-nodes", format!("Paste Nodes{edit_suffix}"))
                                .disabled(form.clipboard_scope().is_none() || self.copy_task.is_some())
                                .on_click(cx.listener(|item, _, window, cx| item.paste_graph_clipboard(window, cx))),
                        ),
                    ]).when(selected_count > 1, |controls| controls.child(
                        div().debug_selector(|| "GRAPHITE-GDD-CUT-NODES".into()).child(
                            Button::new("graphite-gdd-cut-nodes", format!("Cut {selected_count} Nodes{edit_suffix}"))
                                .disabled(self.copy_task.is_some())
                                .on_click(cx.listener(|item, _, window, cx| {
                                    item.cut_graph_selection(window, cx);
                                })),
                        ),
                    ).child(
                        div().debug_selector(|| "GRAPHITE-GDD-DUPLICATE-NODES".into()).child(
                            Button::new("graphite-gdd-duplicate-nodes", format!("Duplicate {selected_count} Nodes in a Copy"))
                                .disabled(self.copy_task.is_some())
                                .on_click(cx.listener(|item, _, window, cx| item.duplicate_graph_selection(window, cx))),
                        ),
                    )))
                    .when(!has_nodes, |panel| panel.child(Label::new("This network has no nodes.")))
                    .when(!empty_exports.is_empty(), |panel| {
                        panel.child(
                            h_flex().gap_2().flex_wrap().children(empty_exports.iter().copied().map(|export_index| {
                                div()
                                    .debug_selector(move || format!("GRAPHITE-GDD-CONNECT-EXPORT-{export_index}"))
                                    .child(Button::new(
                                        ("graphite-gdd-connect-export", export_index),
                                        format!("Connect to Export {}{edit_suffix}", export_index + 1),
                                    )
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(move |item, _, window, cx| {
                                        item.connect_network_export(export_index, window, cx);
                                    })))
                            })),
                        )
                    })
                    .when(selected.is_some_and(|node| node.nested_network().is_some()), |panel| {
                        panel.child(
                            div()
                                .debug_selector(|| "GRAPHITE-GDD-ENTER-NETWORK".into())
                                .child(Button::new("graphite-gdd-enter-network", "Enter Network").on_click(
                                    cx.listener(|item, _, _, cx| item.enter_copy_network(cx)),
                                )),
                        )
                    })
                    .when_some(selected, |panel, selected| {
                        let pin_label = match (working, selected.is_pinned()) {
                            (true, false) => "Pin Node",
                            (true, true) => "Unpin Node",
                            (false, false) => "Save Pinned Folder",
                            (false, true) => "Save Unpinned Folder",
                        };
                        panel
                            .when(selected.can_copy_graph_node(), |panel| {
                                let included = form.selected.is_some_and(|index| form.node_selection.contains(&index));
                                panel.child(
                                    div().debug_selector(|| "GRAPHITE-GDD-CLIPBOARD-SELECT-NODE".into()).child(
                                        Button::new("graphite-gdd-clipboard-select-node", if included {
                                            "Remove from Selection"
                                        } else {
                                            "Add to Selection"
                                        })
                                        .disabled(self.copy_task.is_some())
                                        .on_click(cx.listener(|item, _, _, cx| item.toggle_graph_selection(cx))),
                                    ),
                                )
                            })
                            .child(Label::new("Node name"))
                            .child(
                                div()
                                    .id("graphite-gdd-node-copy-name")
                                    .aria_label("Node name")
                                    .child(form.name.clone()),
                            )
                            .child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-SAVE-RENAMED".into())
                                    .child(
                                        Button::new(
                                            "graphite-gdd-save-renamed",
                                            if working { "Rename Node" } else { "Save Renamed Folder" },
                                        )
                                        .disabled(self.copy_task.is_some())
                                        .on_click(
                                            cx.listener(|item, _, window, cx| {
                                                let NodeCopyPanel::Ready(form) = &item.node_copy
                                                else {
                                                    return;
                                                };
                                                if let Some(node) = form.selected_node() {
                                                    let request = GraphiteCopy::EditNode {
                                                        node: Box::new(node.clone()),
                                                        change: NodeCopyChange::Rename(
                                                            form.name.read(cx).text(cx),
                                                        ),
                                                    };
                                                    item.apply_graph_edit(request, window, cx);
                                                }
                                            }),
                                        ),
                                    ),
                            )
                            .child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-SAVE-PIN".into())
                                    .child(
                                        Button::new("graphite-gdd-save-pin", pin_label)
                                            .disabled(self.copy_task.is_some())
                                            .on_click(cx.listener(|item, _, window, cx| {
                                                let NodeCopyPanel::Ready(form) = &item.node_copy
                                                else {
                                                    return;
                                                };
                                                if let Some(node) = form.selected_node() {
                                                    let request = GraphiteCopy::EditNode {
                                                        node: Box::new(node.clone()),
                                                        change: NodeCopyChange::Pin(
                                                            !node.is_pinned(),
                                                        ),
                                                    };
                                                    item.apply_graph_edit(request, window, cx);
                                                }
                                            })),
                                    ),
                            )
                            .when(selected.can_duplicate_in_copy(), |panel| {
                                panel.child(
                                    div()
                                        .debug_selector(|| "GRAPHITE-GDD-DUPLICATE-NODE".into())
                                        .child(
                                            Button::new(
                                                "graphite-gdd-duplicate-node",
                                                "Duplicate Node in a Copy",
                                            )
                                            .disabled(self.copy_task.is_some())
                                            .on_click(cx.listener(|item, _, window, cx| {
                                                let NodeCopyPanel::Ready(form) = &item.node_copy
                                                else {
                                                    return;
                                                };
                                                if let Some(node) = form
                                                    .selected_node()
                                                    .filter(|node| node.can_duplicate_in_copy())
                                                {
                                                    item.save_copy(
                                                        GraphiteCopy::EditNode {
                                                            node: Box::new(node.clone()),
                                                            change: NodeCopyChange::DuplicateNumeric,
                                                        },
                                                        window,
                                                        cx,
                                                    );
                                                }
                                            })),
                                        ),
                                )
                            })
                            .when(selected.can_delete_in_copy(), |panel| {
                                panel.child(
                                    div()
                                        .debug_selector(|| "GRAPHITE-GDD-DELETE-NODE".into())
                                        .child(
                                            Button::new(
                                                "graphite-gdd-delete-node",
                                                "Remove Node in a Copy",
                                            )
                                            .disabled(self.copy_task.is_some())
                                            .on_click(cx.listener(|item, _, window, cx| {
                                                let NodeCopyPanel::Ready(form) = &item.node_copy
                                                else {
                                                    return;
                                                };
                                                if let Some(node) = form
                                                    .selected_node()
                                                    .filter(|node| node.can_delete_in_copy())
                                                {
                                                    item.save_copy(
                                                        GraphiteCopy::EditNode {
                                                            node: Box::new(node.clone()),
                                                            change: NodeCopyChange::DeleteIsolated,
                                                        },
                                                        window,
                                                        cx,
                                                    );
                                                }
                                            })),
                                        ),
                                )
                            })
                            .when(selected.can_delete_in_copy() && selected.can_copy_graph_node(), |panel| {
                                panel.child(
                                    div().debug_selector(|| "GRAPHITE-GDD-CUT-NODE".into()).child(
                                        Button::new("graphite-gdd-cut-node", format!("Cut This Node{edit_suffix}"))
                                            .disabled(self.copy_task.is_some())
                                            .on_click(cx.listener(|item, _, window, cx| {
                                                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                                                    return;
                                                };
                                                if let Some(node) = form.selected_node().filter(|node| {
                                                    node.can_delete_in_copy() && node.can_copy_graph_node()
                                                }) {
                                                    item.apply_graph_edit(
                                                        GraphiteCopy::CutNodeGroup { nodes: vec![node.clone()] },
                                                        window,
                                                        cx,
                                                    );
                                                }
                                            })),
                                    ),
                                )
                            })
                            .when_some(selected.pinned_position(), |panel, (position, count)| {
                                panel
                                    .child(Label::new(format!("Pinned order · {position} of {count}")))
                                    .child(h_flex().gap_2().children(
                                        [
                                            (
                                                GddPinnedNodeMove::Up,
                                                "GRAPHITE-GDD-MOVE-PIN-UP",
                                                "Move Up",
                                            ),
                                            (
                                                GddPinnedNodeMove::Down,
                                                "GRAPHITE-GDD-MOVE-PIN-DOWN",
                                                "Move Down",
                                            ),
                                        ]
                                        .into_iter()
                                        .map(|(movement, selector, label)| {
                                            div()
                                                .debug_selector(move || selector.into())
                                                .child(
                                                    Button::new(selector, format!("{label}{edit_suffix}"))
                                                        .disabled(
                                                            self.copy_task.is_some()
                                                                || !selected.can_move_pinned(movement),
                                                        )
                                                        .on_click(cx.listener(
                                                            move |item, _, window, cx| {
                                                                let NodeCopyPanel::Ready(form) =
                                                                    &item.node_copy
                                                                else {
                                                                    return;
                                                                };
                                                                if let Some(node) = form
                                                                    .selected_node()
                                                                    .filter(|node| node.can_move_pinned(movement))
                                                                {
                                                                    let request = GraphiteCopy::EditNode {
                                                                        node: Box::new(node.clone()),
                                                                        change: NodeCopyChange::MovePinned(movement),
                                                                    };
                                                                    item.apply_graph_edit(request, window, cx);
                                                                }
                                                            },
                                                        )),
                                                )
                                        }),
                                    ))
                            })
                            .when_some(selected.numeric_import_source(), |panel, index| {
                                panel.child(Label::new(format!("Value connected to Import {}", index + 1)))
                            })
                            .when_some(selected.numeric_input(), |panel, value| {
                                let label = match value {
                                    GddNumericValue::Vector(_) => "Value (x, y)",
                                    GddNumericValue::NumberList(_) => "Value (comma-separated numbers)",
                                    _ => "Value",
                                };
                                panel
                                    .child(Label::new(label))
                                    .child(
                                        div()
                                            .id("graphite-gdd-node-copy-value")
                                            .aria_label(label)
                                            .child(form.value.clone()),
                                    )
                                    .child(
                                        div()
                                            .debug_selector(|| "GRAPHITE-GDD-SAVE-NUMERIC-INPUT".into())
                                            .child(
                                                Button::new(
                                                    "graphite-gdd-save-numeric-input",
                                                    "Save Value in a Copy",
                                                )
                                                .disabled(self.copy_task.is_some())
                                                .on_click(cx.listener(|item, _, window, cx| {
                                                    item.save_numeric_value(window, cx);
                                                })),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .debug_selector(|| "GRAPHITE-GDD-APPLY-NUMERIC-INPUT".into())
                                            .child(
                                                Button::new(
                                                    "graphite-gdd-apply-numeric-input",
                                                    "Apply Value to Working Copy",
                                                )
                                                .disabled(self.copy_task.is_some())
                                                .on_click(cx.listener(|item, _, window, cx| {
                                                    item.change_numeric_value(true, window, cx);
                                                })),
                                            ),
                                    )
                            })
                            .when_some(selected.numeric_input_exposed(), |panel, exposed| {
                                let label = if exposed {
                                    "Hide Value Port in a Copy"
                                } else {
                                    "Expose Value Port in a Copy"
                                };
                                panel.child(
                                    div()
                                        .debug_selector(|| "GRAPHITE-GDD-SAVE-VALUE-EXPOSURE".into())
                                        .child(
                                            Button::new("graphite-gdd-save-value-exposure", label)
                                                .disabled(self.copy_task.is_some())
                                                .on_click(cx.listener(|item, _, window, cx| {
                                                    item.save_numeric_input_exposure(window, cx);
                                                })),
                                        ),
                                )
                            })
                            .when(selected.has_numeric_connection(), |panel| {
                                panel.child(
                                    div()
                                        .debug_selector(|| "GRAPHITE-GDD-DISCONNECT-VALUE".into())
                                        .child(
                                            Button::new(
                                                "graphite-gdd-disconnect-value",
                                                if self.model.read(cx).working.is_some() {
                                                    "Disconnect Value"
                                                } else {
                                                    "Disconnect Value in a Copy"
                                                },
                                            )
                                            .disabled(self.copy_task.is_some())
                                            .on_click(cx.listener(|item, _, window, cx| {
                                                item.save_disconnected_numeric_input(window, cx);
                                            })),
                                        ),
                                )
                            })
                            .when(
                                selected.has_numeric_connection()
                                    && selected.can_preview_numeric_output()
                                    && self.copy_task.is_none(),
                                |panel| panel.child(self.render_numeric_wire_insertion(selected, window, cx)),
                            )
                            .children(self.render_numeric_connection_menu(form, selected, window, cx))
                            .when(selected.can_preview_numeric_output(), |panel| {
                                panel.child(Self::render_node_output_preview(&form.output_preview, cx))
                            })
                            .when(selected.grid_position().is_some(), |panel| {
                                panel.child(
                                    h_flex()
                                        .gap_2()
                                        .child(Label::new("Grid position (x, y)"))
                                        .child(
                                            div()
                                                .id("graphite-gdd-grid-position")
                                                .aria_label("Grid position (x, y)")
                                                .w_32()
                                                .child(form.position.clone()),
                                        )
                                        .child(
                                            div()
                                                .debug_selector(|| "GRAPHITE-GDD-SAVE-POSITION".into())
                                                .child(
                                                    Button::new(
                                                        "graphite-gdd-save-position",
                                                        "Move Node in a Copy",
                                                    )
                                                    .disabled(self.copy_task.is_some())
                                                    .on_click(cx.listener(|item, _, window, cx| {
                                                        item.save_node_grid_position(window, cx);
                                                    })),
                                                ),
                                        ),
                                )
                            })
                    })
            }
            NodeCopyPanel::Failed(error) => {
                panel.child(Label::new(error.clone()).color(Color::Error))
            }
            NodeCopyPanel::Closed | NodeCopyPanel::Loading { .. } => panel,
        }
    }

    fn toggle_saved_view(&mut self, cx: &mut Context<Self>) {
        match &self.saved_view {
            SavedViewPanel::Loading { .. } => return,
            SavedViewPanel::Ready(_) => self.saved_view = SavedViewPanel::Closed,
            SavedViewPanel::Closed | SavedViewPanel::Failed(_) => {
                let InspectedDocument::Gdd { contents } = &self.model.read(cx).document else {
                    return;
                };
                let contents = contents.clone();
                let preparation = cx.background_spawn(async move { contents.saved_view().await });
                self.saved_view = SavedViewPanel::Loading {
                    _task: cx.spawn(async move |item, cx| {
                        let prepared = preparation.await;
                        if let Err(error) = item.update(cx, |item, cx| {
                            item.saved_view = match prepared {
                                Ok(view) => SavedViewPanel::Ready(view),
                                Err(error) => SavedViewPanel::Failed(error.to_string().into()),
                            };
                            cx.notify();
                        }) {
                            log::debug!("Graphite saved view closed: {error:#}");
                        }
                    }),
                };
            }
        }
        cx.notify();
    }

    fn render_saved_view(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let label = match self.saved_view {
            SavedViewPanel::Closed => "Show Saved View",
            SavedViewPanel::Loading { .. } => "Loading Saved View",
            SavedViewPanel::Ready(_) => "Hide Saved View",
            SavedViewPanel::Failed(_) => "Retry Saved View",
        };
        let panel = v_flex().gap_1().child(
            div()
                .debug_selector(|| "GRAPHITE-GDD-SAVED-VIEW".into())
                .child(
                    Button::new("graphite-gdd-saved-view", label)
                        .disabled(matches!(self.saved_view, SavedViewPanel::Loading { .. }))
                        .on_click(cx.listener(|item, _, _, cx| item.toggle_saved_view(cx))),
                ),
        );
        match &self.saved_view {
            SavedViewPanel::Ready(view) => panel
                .when_some(view.document.as_ref(), |panel, view| {
                    let rulers = if view.rulers_visible { "shown" } else { "hidden" };
                    panel.child(Label::new(format!(
                        "Artwork zoom {}% · rulers {rulers}",
                        view.camera.zoom() * 100.
                    )))
                })
                .children(view.networks.iter().map(|network| {
                    use graphite_editor::messages::portfolio::document::utility_types::network_interface::Previewing;

                    let preview = if matches!(network.previewing, Previewing::Yes { .. }) {
                        " · preview active"
                    } else {
                        ""
                    };
                    let camera = network.navigation.node_graph_ptz;
                    Label::new(format!(
                        "{} · zoom {}% · pan ({}, {}){preview}",
                        network.name,
                        camera.zoom() * 100.,
                        camera.pan.x,
                        camera.pan.y,
                    ))
                }))
                .when(view.unresolved_networks > 0, |panel| {
                    panel.child(
                        Label::new(format!(
                            "Saved view settings could not be matched to {} graphs.",
                            view.unresolved_networks,
                        ))
                        .color(Color::Warning),
                    )
                }),
            SavedViewPanel::Failed(error) => panel.child(Label::new(error.clone()).color(Color::Error)),
            SavedViewPanel::Closed | SavedViewPanel::Loading { .. } => panel,
        }
    }

    fn publish_working_folder(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<Task<anyhow::Result<Option<PathBuf>>>> {
        ensure!(
            self.copy_task.is_none(),
            "Wait for the current Graphite operation."
        );
        let model = self.model.read(cx);
        let snapshot = model
            .gdd_contents()
            .cloned()
            .context("Open a GDD document.")?;
        let working = model
            .working
            .as_ref()
            .context("Apply a Graphite edit first.")?;
        ensure!(
            working.save_in_progress.upgrade().is_none(),
            "A Graphite working copy save is already in progress."
        );
        let source = model.path.clone();
        let revision = model.revision.clone();
        let suggested_name = format!("Working copy of {}.gdd", gdd_copy_stem(&model.name));
        let project = self.project.clone();
        let (filesystem, prompt) = project.read_with(cx, |project, cx| {
            ensure!(project.is_local(), "Save Graphite from a local workspace.");
            let path = project
                .absolute_path(&source, cx)
                .context("The Graphite worktree was closed.")?;
            let directory = path
                .parent()
                .context("The Graphite file has no parent directory.")?;
            anyhow::Ok((
                project.fs().clone(),
                cx.prompt_for_new_path(directory, Some(&suggested_name)),
            ))
        })??;
        let model = self.model.clone();
        let reservation = Arc::new(());
        model.update(cx, |model, _| {
            if let Some(working) = &mut model.working {
                working.save_in_progress = Arc::downgrade(&reservation);
            }
        });
        Ok(cx.spawn_in(window, async move |item, cx| {
            let _reservation = reservation;
            let Some(path) = prompt.await?? else {
                return Ok(None);
            };
            ensure!(
                path.is_absolute(),
                "Choose an absolute path for the Graphite folder."
            );
            let files = cx
                .background_spawn({
                    let snapshot = snapshot.clone();
                    async move { snapshot.folder_files() }
                })
                .await?;
            item.read_with(cx, |_, _| ())?;
            ensure!(
                project.read_with(cx, |project, cx| project
                    .absolute_path(&source, cx)
                    .is_some())?,
                "The Graphite worktree was closed."
            );
            filesystem
                .create_new_directory_atomically(path.clone(), files)
                .await
                .with_context(|| format!("Could not save Graphite at {}", path.display()))?;
            model.update(cx, |model, cx| {
                // A newer working edit must keep its dirty state after an older save completes.
                if Arc::ptr_eq(&model.revision, &revision)
                    && let Some(working) = &mut model.working
                {
                    working.saved = snapshot;
                    cx.notify();
                }
            });
            Ok(Some(path))
        }))
    }

    fn save_working_folder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let save = match self.publish_working_folder(window, cx) {
            Ok(save) => save,
            Err(error) => {
                self.copy_feedback = Some((format!("{error:#}").into(), Color::Error));
                cx.notify();
                return;
            }
        };
        self.copy_feedback = None;
        self.copy_task = Some(cx.spawn_in(window, async move |item, cx| {
            let result = save.await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                item.copy_task = None;
                item.copy_feedback = match result {
                    Ok(Some(path)) if item.is_dirty(cx) => Some((
                        format!(
                            "Snapshot saved to {}. Newer edits are unsaved.",
                            path.display()
                        )
                        .into(),
                        Color::Default,
                    )),
                    Ok(Some(path)) => Some((
                        format!("Working copy saved to {}", path.display()).into(),
                        Color::Default,
                    )),
                    Ok(None) => None,
                    Err(error) => Some((format!("{error:#}").into(), Color::Error)),
                };
                cx.notify();
            }) {
                log::debug!("Graphite working copy view closed: {error:#}");
            }
        }));
        cx.notify();
    }

    fn save_copy(&mut self, request: GraphiteCopy, window: &mut Window, cx: &mut Context<Self>) {
        if self.copy_task.is_some() {
            return;
        }
        let explicit_export = matches!(&request, GraphiteCopy::Export(_));
        let content = if explicit_export {
            self.export_content
        } else {
            GddExportContent::default()
        };
        let embed_linked_resources = explicit_export && self.embed_linked_resources;
        let embed_legacy_document = explicit_export && self.embed_legacy_document;
        if explicit_export && let Err(error) = content.validate() {
            self.copy_feedback = Some((error.to_string().into(), Color::Error));
            cx.notify();
            return;
        }
        let model = self.model.read(cx);
        let source = model.path.clone();
        let project = self.project.clone();
        let format = match &request {
            GraphiteCopy::Export(format) => Some(*format),
            GraphiteCopy::Original if model.original.is_some() => None,
            _ => Some(GddCopyFormat::Folder),
        };
        let export = match (&model.document, format) {
            (InspectedDocument::Gdd { contents, .. }, Some(format)) => {
                Some((contents.clone(), format))
            }
            (_, None) => None,
            (InspectedDocument::Legacy(_), Some(_)) => return,
        };
        let original = model.original.clone();
        let suggested_name = match format {
            Some(format) => {
                let stem = gdd_copy_stem(&model.name);
                let suffix = match format {
                    GddCopyFormat::Archive(ArchiveFormat::Zip) => ".gdd.zip",
                    GddCopyFormat::Archive(ArchiveFormat::Xz) => ".gdd.tar.xz",
                    GddCopyFormat::Folder => ".gdd",
                };
                let prefix = match &request {
                    GraphiteCopy::Original | GraphiteCopy::Export(_) => "Copy of",
                    GraphiteCopy::RecoverFolder => "Recovered",
                    GraphiteCopy::AppendNetworkPort {
                        kind: NetworkPortKind::Import,
                        ..
                    } => "Added import in",
                    GraphiteCopy::AppendNetworkPort {
                        kind: NetworkPortKind::Export,
                        ..
                    } => "Added export in",
                    GraphiteCopy::EditNetworkWire(NetworkWireChange::Export { .. }) => {
                        "Connected export in"
                    }
                    GraphiteCopy::EditNetworkWire(NetworkWireChange::Import {
                        reconnect: false,
                        ..
                    }) => "Connected import in",
                    GraphiteCopy::EditNetworkWire(NetworkWireChange::Import {
                        reconnect: true,
                        ..
                    }) => "Reconnected import in",
                    GraphiteCopy::SetPrimaryNetworkPortExposure {
                        kind: NetworkPortKind::Import,
                        exposed: true,
                        ..
                    } => "Exposed primary import in",
                    GraphiteCopy::SetPrimaryNetworkPortExposure {
                        kind: NetworkPortKind::Import,
                        exposed: false,
                        ..
                    } => "Hidden primary import in",
                    GraphiteCopy::SetPrimaryNetworkPortExposure {
                        kind: NetworkPortKind::Export,
                        exposed: true,
                        ..
                    } => "Exposed primary export in",
                    GraphiteCopy::SetPrimaryNetworkPortExposure {
                        kind: NetworkPortKind::Export,
                        exposed: false,
                        ..
                    } => "Hidden primary export in",
                    GraphiteCopy::EditNetworkPort {
                        change:
                            NetworkPortChange::Remove {
                                kind: NetworkPortKind::Import,
                                ..
                            },
                        ..
                    } => "Removed import in",
                    GraphiteCopy::EditNetworkPort {
                        change:
                            NetworkPortChange::Remove {
                                kind: NetworkPortKind::Export,
                                ..
                            },
                        ..
                    } => "Removed export in",
                    GraphiteCopy::EditNetworkPort {
                        change:
                            NetworkPortChange::Move {
                                kind: NetworkPortKind::Import,
                                ..
                            },
                        ..
                    } => "Reordered imports in",
                    GraphiteCopy::EditNetworkPort {
                        change:
                            NetworkPortChange::Move {
                                kind: NetworkPortKind::Export,
                                ..
                            },
                        ..
                    } => "Reordered exports in",
                    GraphiteCopy::RenameNetworkPort { .. } => "Renamed port in",
                    GraphiteCopy::PasteNodeGroup { .. } => "Pasted nodes in",
                    GraphiteCopy::CutNodeGroup { nodes } if nodes.len() == 1 => "Cut node from",
                    GraphiteCopy::CutNodeGroup { .. } => "Cut nodes from",
                    GraphiteCopy::DuplicateNodeGroup { .. } => "Duplicated nodes in",
                    GraphiteCopy::MoveHistory(GddHistoryDirection::Undo) => "Undo of",
                    GraphiteCopy::MoveHistory(GddHistoryDirection::Redo) => "Redo of",
                    GraphiteCopy::InsertNumericNode { .. }
                    | GraphiteCopy::InsertCatalogNode { .. } => "Added node in",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::Rename(_),
                        ..
                    } => "Renamed",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::DuplicateNumeric,
                        ..
                    } => "Duplicated node in",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::DeleteIsolated,
                        ..
                    } => "Removed node in",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::GridPosition(_),
                        ..
                    } => "Moved node in",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::Pin(true),
                        ..
                    } => "Pinned",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::Pin(false),
                        ..
                    } => "Unpinned",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::MovePinned(_),
                        ..
                    } => "Reordered",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::NumericInput(_),
                        ..
                    } => "Edited",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::NumericInputExposure(true),
                        ..
                    } => "Exposed port in",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::NumericInputExposure(false),
                        ..
                    } => "Hidden port in",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::DisconnectNumericInput,
                        ..
                    } => "Disconnected",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::InsertNumericOnWire(_),
                        ..
                    } => "Inserted node on wire in",
                    GraphiteCopy::EditNode {
                        change:
                            NodeCopyChange::ConnectNumericInput {
                                previous: Some(_), ..
                            },
                        ..
                    } => "Reconnected",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::ConnectNumericInput { previous: None, .. },
                        ..
                    } => "Connected",
                };
                format!("{prefix} {stem}{suffix}")
            }
            None => format!("Copy of {}", model.name),
        };
        let selection = self
            .project
            .read_with(cx, |project, cx| {
                ensure!(
                    project.is_local(),
                    "Save a Graphite copy from a local workspace."
                );
                let source_path = project
                    .absolute_path(&model.path, cx)
                    .context("The Graphite worktree was closed.")?;
                let directory = source_path
                    .parent()
                    .context("The Graphite file has no parent directory.")?;
                Ok::<_, anyhow::Error>((
                    project.fs().clone(),
                    cx.prompt_for_new_path(directory, Some(&suggested_name)),
                ))
            })
            .and_then(|selection| selection);
        let (filesystem, prompt) = match selection {
            Ok(selection) => selection,
            Err(error) => {
                self.copy_feedback = Some((format!("{error:#}").into(), Color::Error));
                cx.notify();
                return;
            }
        };
        self.copy_feedback = None;
        self.copy_task = Some(cx.spawn_in(window, async move |item, cx| {
            let result: anyhow::Result<_> = async {
                let Some(path) = prompt.await?? else {
                    return Ok(None);
                };
                ensure!(
                    path.is_absolute(),
                    "Choose an absolute path for the Graphite copy."
                );
                let copy = if let Some((archive, format)) = export {
                    let legacy_document = if embed_legacy_document {
                        item.read_with(cx, |_, _| ())?;
                        let chooser = cx.update(|_, cx| {
                            cx.prompt_for_paths(PathPromptOptions {
                                files: true,
                                directories: false,
                                multiple: false,
                                prompt: Some("Choose a Legacy Graphite Document".into()),
                            })
                        })?;
                        let Some(paths) = chooser.await?? else {
                            return Ok(None);
                        };
                        let [path] = paths.as_slice() else {
                            anyhow::bail!("Select one legacy Graphite document.");
                        };
                        item.read_with(cx, |_, _| ())?;
                        Some(
                            cx.background_spawn({
                                let filesystem = filesystem.clone();
                                let path = path.clone();
                                async move {
                                    read_approved_legacy_document(filesystem.as_ref(), &path).await
                                }
                            })
                            .await?,
                        )
                    } else {
                        None
                    };
                    let approved_files = if embed_linked_resources
                        && !archive.missing_linked_resources()?.is_empty()
                    {
                        item.read_with(cx, |_, _| ())?;
                        let chooser = cx.update(|_, cx| {
                            cx.prompt_for_paths(PathPromptOptions {
                                files: true,
                                directories: false,
                                multiple: true,
                                prompt: Some("Choose Graphite Resource Files to Embed".into()),
                            })
                        })?;
                        let Some(paths) = chooser.await?? else {
                            return Ok(None);
                        };
                        item.read_with(cx, |_, _| ())?;
                        cx.background_spawn({
                                let filesystem = filesystem.clone();
                                async move {
                                    read_approved_resource_files(filesystem.as_ref(), paths).await
                                }
                            })
                            .await?
                    } else {
                        Vec::new()
                    };
                    cx.background_spawn(async move {
                        let revision = match request {
                            GraphiteCopy::Original | GraphiteCopy::Export(_) => None,
                            GraphiteCopy::RecoverFolder => {
                                Some(archive.recover_pending_edits(ARCHIVE_BUDGET).await?)
                            }
                            GraphiteCopy::MoveHistory(direction) => Some(
                                archive
                                    .move_history_in_copy(direction, ARCHIVE_BUDGET)
                                    .await?,
                            ),
                            GraphiteCopy::AppendNetworkPort {
                                network,
                                kind,
                                with_hidden_primary,
                            } => Some(match (kind, with_hidden_primary) {
                                (NetworkPortKind::Import, false) => {
                                    archive
                                        .append_nested_network_import_in_copy(
                                            &network,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                (NetworkPortKind::Export, false) => {
                                    archive
                                        .append_nested_network_export_in_copy(
                                            &network,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                (NetworkPortKind::Import, true) => {
                                    archive
                                        .append_secondary_nested_network_import_in_copy(
                                            &network,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                (NetworkPortKind::Export, true) => {
                                    archive
                                        .append_secondary_nested_network_export_in_copy(
                                            &network,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                            }),
                            GraphiteCopy::EditNetworkWire(change) => {
                                Some(change.stage(&archive).await?)
                            }
                            GraphiteCopy::SetPrimaryNetworkPortExposure {
                                network,
                                kind,
                                exposed,
                            } => Some(match kind {
                                NetworkPortKind::Import => {
                                    archive
                                        .set_primary_network_import_exposed_in_copy(
                                            &network,
                                            exposed,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                NetworkPortKind::Export => {
                                    archive
                                        .set_primary_network_export_exposed_in_copy(
                                            &network,
                                            exposed,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                            }),
                            GraphiteCopy::EditNetworkPort { network, change } => {
                                Some(match change {
                                    NetworkPortChange::Remove {
                                        kind: NetworkPortKind::Import,
                                        index,
                                    } => {
                                        archive
                                            .remove_nested_network_import_in_copy(
                                                &network,
                                                index,
                                                ARCHIVE_BUDGET,
                                            )
                                            .await?
                                    }
                                    NetworkPortChange::Remove {
                                        kind: NetworkPortKind::Export,
                                        index,
                                    } => {
                                        archive
                                            .remove_nested_network_export_in_copy(
                                                &network,
                                                index,
                                                ARCHIVE_BUDGET,
                                            )
                                            .await?
                                    }
                                    NetworkPortChange::Move {
                                        kind: NetworkPortKind::Import,
                                        from,
                                        to,
                                    } => {
                                        archive
                                            .move_nested_network_import_in_copy(
                                                &network,
                                                from,
                                                to,
                                                ARCHIVE_BUDGET,
                                            )
                                            .await?
                                    }
                                    NetworkPortChange::Move {
                                        kind: NetworkPortKind::Export,
                                        from,
                                        to,
                                    } => {
                                        archive
                                            .move_nested_network_export_in_copy(
                                                &network,
                                                from,
                                                to,
                                                ARCHIVE_BUDGET,
                                            )
                                            .await?
                                    }
                                })
                            }
                            GraphiteCopy::RenameNetworkPort {
                                network,
                                port,
                                name,
                            } => Some(
                                archive
                                    .rename_nested_network_port_in_copy(
                                        &network,
                                        port,
                                        name,
                                        ARCHIVE_BUDGET,
                                    )
                                    .await?,
                            ),
                            GraphiteCopy::PasteNodeGroup { network, clipboard } => Some(
                                archive
                                    .paste_graph_nodes_in_copy(&network, &clipboard, ARCHIVE_BUDGET)
                                    .await?,
                            ),
                            GraphiteCopy::CutNodeGroup { nodes } => {
                                let (copy, clipboard) = archive
                                    .cut_graph_nodes_in_copy(&nodes, ARCHIVE_BUDGET)
                                    .await?;
                                return Ok(PreparedCopy::Folder {
                                    files: copy.folder_files()?,
                                    clipboard: Some(clipboard),
                                });
                            }
                            GraphiteCopy::DuplicateNodeGroup { network, nodes } => Some(
                                archive
                                    .duplicate_graph_nodes_in_copy(&network, &nodes, ARCHIVE_BUDGET)
                                    .await?,
                            ),
                            GraphiteCopy::InsertNumericNode {
                                operation,
                                position,
                            } => Some(
                                archive
                                    .insert_root_numeric_node_in_copy(
                                        operation,
                                        position,
                                        ARCHIVE_BUDGET,
                                    )
                                    .await?,
                            ),
                            GraphiteCopy::InsertCatalogNode {
                                identifier,
                                position,
                            } => Some(
                                archive
                                    .insert_root_catalog_node_in_copy(
                                        identifier,
                                        position,
                                        ARCHIVE_BUDGET,
                                    )
                                    .await?,
                            ),
                            GraphiteCopy::EditNode { node, change } => {
                                Some(change.stage(&archive, &node).await?)
                            }
                        };
                        if let Some(copy) = revision {
                            return Ok(PreparedCopy::Folder {
                                files: copy.folder_files()?,
                                clipboard: None,
                            });
                        }
                        let enriched = if embed_linked_resources || legacy_document.is_some() {
                            Some(
                                archive
                                    .with_export_attachments(
                                        content,
                                        embed_linked_resources.then_some(approved_files),
                                        legacy_document.as_ref(),
                                        ARCHIVE_BUDGET,
                                    )
                                    .await?,
                            )
                        } else {
                            None
                        };
                        let contents = enriched.as_ref().unwrap_or(&archive);
                        let content = if enriched.is_some() {
                            GddExportContent::default()
                        } else {
                            content
                        };
                        Ok::<_, anyhow::Error>(match format {
                            GddCopyFormat::Archive(format) => PreparedCopy::File(
                                contents
                                    .export_archive_with_content(format, content, ARCHIVE_BUDGET)
                                    .await?
                                    .into(),
                            ),
                            GddCopyFormat::Folder => PreparedCopy::Folder {
                                files: contents
                                    .export_folder_files_with_content(content, ARCHIVE_BUDGET)
                                    .await?,
                                clipboard: None,
                            },
                        })
                    })
                    .await?
                } else {
                    PreparedCopy::File(
                        original.context("The original Graphite bytes are unavailable.")?,
                    )
                };
                item.read_with(cx, |_, _| ())?;
                ensure!(
                    project.read_with(cx, |project, cx| project
                        .absolute_path(&source, cx)
                        .is_some())?,
                    "The Graphite worktree was closed."
                );
                let (written, clipboard) = match copy {
                    PreparedCopy::File(bytes) => (
                        filesystem
                            .create_new_file_atomically(path.clone(), bytes)
                            .await,
                        None,
                    ),
                    PreparedCopy::Folder { files, clipboard } => (
                        filesystem
                            .create_new_directory_atomically(path.clone(), files)
                            .await,
                        clipboard,
                    ),
                };
                written.with_context(|| format!("Could not save a copy at {}", path.display()))?;
                Ok(Some((path, clipboard)))
            }
            .await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                item.copy_task = None;
                item.copy_feedback = match result {
                    Ok(Some((path, clipboard))) => {
                        if let Some(clipboard) = clipboard {
                            cx.write_to_clipboard(ClipboardItem::new_string(clipboard));
                        }
                        Some((
                            format!("Copy saved to {}", path.display()).into(),
                            Color::Default,
                        ))
                    }
                    Ok(None) => None,
                    Err(error) => Some((format!("{error:#}").into(), Color::Error)),
                };
                cx.notify();
            }) {
                log::debug!("Graphite copy view closed: {error:#}");
            }
        }));
        cx.notify();
    }
}

impl workspace::ProjectItem for GraphiteDocumentItem {
    type Item = GraphiteDocumentProjectItem;

    fn for_project_item(
        project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(item, project.downgrade(), window, cx)
    }
}

impl Focusable for GraphiteDocumentItem {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<ItemEvent> for GraphiteDocumentItem {}

impl Item for GraphiteDocumentItem {
    type Event = ItemEvent;

    fn to_item_events(event: &Self::Event, emit: &mut dyn FnMut(ItemEvent)) {
        emit(*event);
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        self.model.read(cx).name.clone()
    }

    fn buffer_kind(&self, _cx: &App) -> ItemBufferKind {
        ItemBufferKind::Singleton
    }

    fn for_each_project_item(
        &self,
        cx: &App,
        visit: &mut dyn FnMut(EntityId, &dyn project::ProjectItem),
    ) {
        visit(self.model.entity_id(), self.model.read(cx));
    }

    fn is_dirty(&self, cx: &App) -> bool {
        self.model.read(cx).working_dirty()
    }

    fn can_save(&self, cx: &App) -> bool {
        self.model.read(cx).working.is_some()
    }

    fn supports_autosave(&self, _cx: &App) -> bool {
        false
    }

    fn save(
        &mut self,
        _options: SaveOptions,
        _project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        let save = match self.publish_working_folder(window, cx) {
            Ok(save) => save,
            Err(error) => return Task::ready(Err(error)),
        };
        cx.spawn(async move |item, cx| {
            ensure!(
                save.await?.is_some(),
                "Graphite working copy save cancelled."
            );
            ensure!(
                item.read_with(cx, |item, cx| !item.is_dirty(cx))?,
                "The folder was saved. Newer Graphite edits still need to be saved."
            );
            Ok(())
        })
    }

    fn reload(
        &mut self,
        _project: Entity<Project>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<()>> {
        let model = self.model.read(cx);
        if !model.working_dirty() {
            return Task::ready(Ok(()));
        }
        let Some(working) = &model.working else {
            return Task::ready(Err(anyhow!("The Graphite working copy is unavailable.")));
        };
        if working.save_in_progress.upgrade().is_some() {
            return Task::ready(Err(anyhow!("A Graphite working copy save is in progress.")));
        }
        let revision = model.revision.clone();
        let saved = working.saved.clone();
        let inspect = cx.background_spawn({
            let saved = saved.clone();
            async move { saved.graph_inspection().await }
        });
        let model = self.model.clone();
        cx.spawn(async move |_, cx| {
            let inspection = inspect.await?;
            model.update(cx, |model, cx| {
                model.replace_working_snapshot(&revision, saved, inspection, cx)
            })
        })
    }

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>> {
        let model = self.model.clone();
        let project = self.project.clone();
        Task::ready(Some(cx.new(|cx| {
            let mut item = Self::new(model, project, window, cx);
            item.export_content = self.export_content;
            item.embed_linked_resources = self.embed_linked_resources;
            item.embed_legacy_document = self.embed_legacy_document;
            item
        })))
    }
}

impl Render for GraphiteDocumentItem {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let model = self.model.read(cx);
        let view = v_flex()
            .id("graphite-document-inspector")
            .track_focus(&self.focus_handle)
            .key_context("GraphiteGdd")
            .on_action(cx.listener(|item, _: &editor::actions::Undo, window, cx| {
                item.route_working_history(GddHistoryDirection::Undo, window, cx);
            }))
            .on_action(cx.listener(|item, _: &editor::actions::Redo, window, cx| {
                item.route_working_history(GddHistoryDirection::Redo, window, cx);
            }))
            .size_full()
            .p_6()
            .gap_2()
            .overflow_y_scroll()
            .child(Label::new(model.name.clone()).size(LabelSize::Large))
            .child(
                div().debug_selector(|| "GRAPHITE-SAVE-COPY".into()).child(
                    Button::new("graphite-save-copy", "Save a Copy")
                        .disabled(self.copy_task.is_some())
                        .on_click(cx.listener(|item, _, window, cx| {
                            item.save_copy(GraphiteCopy::Original, window, cx)
                        })),
                ),
            )
            .when_some(self.copy_feedback.clone(), |view, (message, color)| {
                view.child(Label::new(message).color(color))
            });
        match &model.document {
            InspectedDocument::Gdd { contents } => {
                let manifest = contents.manifest();
                let recovery = contents.recovery();
                let can_undo = contents.can_move_history(GddHistoryDirection::Undo);
                let can_redo = contents.can_move_history(GddHistoryDirection::Redo);
                let recovery_source = match recovery.source {
                    GddRecoverySource::RegistryAndHistory => "Opened saved registry and history",
                    GddRecoverySource::RegistryOnly => "Opened saved registry",
                    GddRecoverySource::HistoryOnly => "Rebuilt registry from history",
                };
                let label = if model.original.is_none() {
                    "Graphite GDD folder"
                } else {
                    "Graphite GDD"
                };
                let format_version = manifest.format_version;
                let working_dirty = model.working_dirty();
                let working = model.working.is_some();
                view.child(Label::new(if working {
                    format!("{label} v{format_version} · Working copy")
                } else {
                    format!("{label} v{format_version} · Read-only")
                }))
                .when(!working && recovery.hot_log_len == 0, |view| {
                    view.child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-EDIT-WORKING-COPY".into())
                            .child(
                                Button::new("graphite-gdd-edit-working-copy", "Edit Working Copy")
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(|item, _, window, cx| {
                                        item.open_working_copy(window, cx)
                                    })),
                            ),
                    )
                })
                .when(working, |view| {
                    view.child(Label::new(if working_dirty {
                        "Unsaved working edits"
                    } else {
                        "No unsaved edits"
                    }))
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-SAVE-WORKING-FOLDER".into())
                            .child(
                                Button::new(
                                    "graphite-gdd-save-working-folder",
                                    "Save Working Copy As",
                                )
                                .disabled(self.copy_task.is_some())
                                .on_click(cx.listener(
                                    |item, _, window, cx| {
                                        item.save_working_folder(window, cx);
                                    },
                                )),
                            ),
                    )
                })
                .child(Label::new(format!("Document ID {}", manifest.document_id)))
                .child(Label::new(format!(
                    "Recorded editor version {}",
                    manifest.editor_version
                )))
                .child(
                    v_flex()
                        .debug_selector(|| "GRAPHITE-GDD-RECOVERY".into())
                        .child(Label::new(recovery_source))
                        .child(Label::new(format!(
                            "{} networks · {} nodes · {} history entries · {} pending operations",
                            recovery.network_count,
                            recovery.node_count,
                            recovery.history_len,
                            recovery.hot_log_len,
                        ))),
                )
                .when(contents.has_saved_view(), |view| {
                    view.child(self.render_saved_view(cx))
                })
                .when(recovery.hot_log_len == 0, |view| {
                    view.child(self.render_node_copy(window, cx))
                })
                .when(recovery.hot_log_len == 0, |view| {
                    view.child(self.render_insert_node(window, cx))
                })
                .child(self.render_node_catalog(cx))
                .when(recovery.hot_log_len > 0, |view| {
                    view.child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-SAVE-RECOVERED".into())
                            .child(
                                Button::new("graphite-gdd-save-recovered", "Save Recovered Folder")
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(|item, _, window, cx| {
                                        item.save_copy(GraphiteCopy::RecoverFolder, window, cx)
                                    })),
                            ),
                    )
                })
                .when(can_undo || can_redo, |view| {
                    view.child(
                        h_flex().gap_2().children(
                            [
                                (
                                    GddHistoryDirection::Undo,
                                    if working { "Undo" } else { "Save Undo Folder" },
                                    if working {
                                        "GRAPHITE-GDD-UNDO"
                                    } else {
                                        "GRAPHITE-GDD-SAVE-UNDO"
                                    },
                                    can_undo,
                                ),
                                (
                                    GddHistoryDirection::Redo,
                                    if working { "Redo" } else { "Save Redo Folder" },
                                    if working {
                                        "GRAPHITE-GDD-REDO"
                                    } else {
                                        "GRAPHITE-GDD-SAVE-REDO"
                                    },
                                    can_redo,
                                ),
                            ]
                            .into_iter()
                            .filter(|(_, _, _, available)| *available)
                            .map(|(direction, label, selector, _)| {
                                div().debug_selector(move || selector.into()).child(
                                    Button::new(selector, label)
                                        .disabled(self.copy_task.is_some())
                                        .on_click(cx.listener(move |item, _, window, cx| {
                                            if working {
                                                item.move_working_history(direction, window, cx);
                                            } else {
                                                item.save_copy(
                                                    GraphiteCopy::MoveHistory(direction),
                                                    window,
                                                    cx,
                                                );
                                            }
                                        })),
                                )
                            }),
                        ),
                    )
                })
                .child(
                    h_flex()
                        .flex_wrap()
                        .gap_3()
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-EXPORT-INCLUDE-REGISTRY".into())
                                .child(
                                    Checkbox::new(
                                        "graphite-export-include-registry",
                                        ToggleState::from(self.export_content.include_registry),
                                    )
                                    .label("Include registry")
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(
                                        |item, choice: &ToggleState, _, cx| {
                                            item.export_content.include_registry =
                                                choice.selected();
                                            item.copy_feedback = None;
                                            cx.notify();
                                        },
                                    )),
                                ),
                        )
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-EXPORT-INCLUDE-HISTORY".into())
                                .child(
                                    Checkbox::new(
                                        "graphite-export-include-history",
                                        ToggleState::from(self.export_content.include_history),
                                    )
                                    .label("Include history")
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(
                                        |item, choice: &ToggleState, _, cx| {
                                            item.export_content.include_history = choice.selected();
                                            item.copy_feedback = None;
                                            cx.notify();
                                        },
                                    )),
                                ),
                        )
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-EXPORT-EMBED-RESOURCES".into())
                                .child(
                                    Checkbox::new(
                                        "graphite-export-embed-resources",
                                        ToggleState::from(self.embed_linked_resources),
                                    )
                                    .label("Embed linked resources")
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(
                                        |item, choice: &ToggleState, _, cx| {
                                            item.embed_linked_resources = choice.selected();
                                            item.copy_feedback = None;
                                            cx.notify();
                                        },
                                    )),
                                ),
                        )
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-EXPORT-EMBED-LEGACY".into())
                                .child(
                                    Checkbox::new(
                                        "graphite-export-embed-legacy",
                                        ToggleState::from(self.embed_legacy_document),
                                    )
                                    .label("Embed legacy Graphite document")
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(
                                        |item, choice: &ToggleState, _, cx| {
                                            item.embed_legacy_document = choice.selected();
                                            item.copy_feedback = None;
                                            cx.notify();
                                        },
                                    )),
                                ),
                        ),
                )
                .child(
                    h_flex().gap_2().children(
                        [
                            (
                                "GRAPHITE-EXPORT-GDD-ZIP",
                                "Export ZIP",
                                GddCopyFormat::Archive(ArchiveFormat::Zip),
                            ),
                            (
                                "GRAPHITE-EXPORT-GDD-XZ",
                                "Export XZ",
                                GddCopyFormat::Archive(ArchiveFormat::Xz),
                            ),
                            (
                                "GRAPHITE-EXPORT-GDD-FOLDER",
                                "Export Folder",
                                GddCopyFormat::Folder,
                            ),
                        ]
                        .map(|(selector, label, format)| {
                            div().debug_selector(move || selector.into()).child(
                                Button::new(selector, label)
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(move |item, _, window, cx| {
                                        item.save_copy(GraphiteCopy::Export(format), window, cx)
                                    })),
                            )
                        }),
                    ),
                )
            }
            InspectedDocument::Legacy(document) => {
                let layout = match document.schema() {
                    LegacyDocumentSchema::NetworkInterface => {
                        "Graphite legacy document · Read-only"
                    }
                    LegacyDocumentSchema::OriginalNetwork => {
                        "Graphite older network layout · Read-only"
                    }
                };
                let unregistered = document.unregistered_identifiers();
                view.child(Label::new(layout))
                    .when_some(document.build_hint(), |view, hint| {
                        view.child(Label::new(format!("Recorded Graphite build {hint}")))
                    })
                    .child(Label::new(format!("{} nodes", document.node_count())))
                    .when(!unregistered.is_empty(), |view| {
                        view.child(Label::new(format!(
                            "{} node identifiers are unregistered in this Graphite build",
                            unregistered.len()
                        )))
                        .children(
                            unregistered
                                .iter()
                                .take(20)
                                .map(|identifier| Label::new(identifier.clone())),
                        )
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png_workflow::tests::{click, native_workspace, paint};
    use gpui::TestAppContext;
    use graphite_document_container::{AnyContainer, backends::memory::MemoryBackend};
    use graphite_document_format::{ExportFormat, ExportOptions, GddV1, GddV1Layout};
    use graphite_graph_craft::application_io::resource::HashMapResourceStorage;
    use std::path::PathBuf;

    async fn archive_bytes(version: u32, format: ExportFormat) -> Vec<u8> {
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            171,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("create actual Graphite document storage");
        owner
            .update_manifest(|manifest| manifest.format_version = version)
            .expect("use the original manifest writer");
        owner
            .export_to_bytes(
                format,
                ExportOptions::default(),
                &HashMapResourceStorage::new(),
                None,
            )
            .await
            .expect("export actual Graphite storage")
    }

    async fn nested_graph_archive() -> (Vec<u8>, usize) {
        nested_graph_archive_with_sibling(false).await
    }

    async fn nested_graph_archive_with_sibling(include_sibling: bool) -> (Vec<u8>, usize) {
        use graphite_document_graph_storage::NoMetadata;
        use graphite_graph_craft::document::{
            DocumentNode, DocumentNodeImplementation, NodeId, NodeNetwork,
        };
        use graphite_resource::ResourceRegistry;

        let leaf = || DocumentNode {
            implementation: DocumentNodeImplementation::Network(NodeNetwork::default()),
            ..Default::default()
        };
        let mut nested = NodeNetwork::default();
        nested.nodes.insert(NodeId(7), leaf());
        if include_sibling {
            nested.nodes.insert(NodeId(9), leaf());
        }
        let network = NodeNetwork {
            nodes: [
                (NodeId(7), leaf()),
                (
                    NodeId(24),
                    DocumentNode {
                        implementation: DocumentNodeImplementation::Network(nested),
                        ..Default::default()
                    },
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            184,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("pinned GDD storage");
        let storage = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&network, &NoMetadata, &ResourceRegistry::new(), &storage)
            .expect("source fixture with repeated local IDs");
        let original = owner
            .export_to_bytes(ExportFormat::Zip, ExportOptions::default(), &storage, None)
            .await
            .expect("original ZIP payload");
        (original, owner.session().history().count())
    }

    fn choose_copy_node(window: &mut gpui::VisualTestContext, selector: &'static str) {
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        paint(window);
        click(window, selector);
        paint(window);
    }

    fn choose_scoped_copy_node(window: &mut gpui::VisualTestContext, selector: &'static str) {
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        paint(window);
        click(window, selector);
        paint(window);
    }

    fn choose_nested_copy_node(window: &mut gpui::VisualTestContext) {
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
    }

    fn right_click_saved_graph(window: &mut gpui::VisualTestContext, selector: &'static str) {
        click_saved_graph(
            window,
            selector,
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
    }

    fn click_saved_graph(
        window: &mut gpui::VisualTestContext,
        selector: &'static str,
        button: gpui::MouseButton,
        modifiers: gpui::Modifiers,
    ) {
        let position = window
            .debug_bounds(selector)
            .expect("visible saved graph control")
            .center();
        click_graph_point(window, position, button, modifiers);
    }

    fn click_graph_point(
        window: &mut gpui::VisualTestContext,
        position: gpui::Point<gpui::Pixels>,
        button: gpui::MouseButton,
        modifiers: gpui::Modifiers,
    ) {
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position,
                    button,
                    modifiers,
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position,
                    button,
                    modifiers,
                    click_count: 1,
                }),
                cx,
            );
        });
        window.run_until_parked();
        paint(window);
    }

    fn drag_graph_point(
        window: &mut gpui::VisualTestContext,
        start: gpui::Point<gpui::Pixels>,
        end: gpui::Point<gpui::Pixels>,
        during_drag: impl FnOnce(&gpui::VisualTestContext),
    ) {
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position: start,
                    button: gpui::MouseButton::Left,
                    modifiers: gpui::Modifiers::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            )
        });
        paint(window);
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                    position: start + gpui::point(px(7.), px(0.)),
                    pressed_button: Some(gpui::MouseButton::Left),
                    modifiers: gpui::Modifiers::default(),
                }),
                cx,
            );
            assert!(cx.has_active_drag());
        });
        during_drag(window);
        paint(window);
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                    position: end,
                    pressed_button: Some(gpui::MouseButton::Left),
                    modifiers: gpui::Modifiers::default(),
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position: end,
                    button: gpui::MouseButton::Left,
                    modifiers: gpui::Modifiers::default(),
                    click_count: 1,
                }),
                cx,
            );
            assert!(!cx.has_active_drag());
        });
        window.run_until_parked();
        paint(window);
    }

    fn saved_graph_shortcut(window: &mut gpui::VisualTestContext, key: &str) {
        let modifier = if cfg!(target_os = "macos") {
            "cmd"
        } else {
            "ctrl"
        };
        window.simulate_keystrokes(&format!("{modifier}-{key}"));
        window.run_until_parked();
        paint(window);
    }

    fn open_saved_boundary_menu(
        window: &mut gpui::VisualTestContext,
        kind: NetworkPortKind,
        index: usize,
        canvas: bool,
    ) {
        if canvas {
            if window.debug_bounds("GRAPHITE-GDD-SAVED-GRAPH").is_none() {
                click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
                window.run_until_parked();
                paint(window);
            }
            let sockets = match kind {
                NetworkPortKind::Import => {
                    ["GRAPHITE-GDD-GRAPH-Import-1", "GRAPHITE-GDD-GRAPH-Import-2"]
                }
                NetworkPortKind::Export => {
                    ["GRAPHITE-GDD-GRAPH-Export-1", "GRAPHITE-GDD-GRAPH-Export-2"]
                }
            };
            right_click_saved_graph(
                window,
                sockets
                    .get(index)
                    .copied()
                    .expect("fixture boundary socket"),
            );
        } else {
            let selector = match kind {
                NetworkPortKind::Import => "GRAPHITE-GDD-REMOVE-IMPORT",
                NetworkPortKind::Export => "GRAPHITE-GDD-REMOVE-EXPORT",
            };
            click(window, selector);
        }
    }

    enum NumericGraphFixture {
        SingleNode,
        ExposedInput,
        ExposedInputWithAlternative,
        Connection,
        Reconnection,
        NestedConnection,
    }

    async fn numeric_graph_archive(fixture: NumericGraphFixture) -> (Vec<u8>, usize) {
        numeric_graph_archive_with_layout(fixture, false).await
    }

    async fn numeric_graph_archive_with_layout(
        fixture: NumericGraphFixture,
        positioned: bool,
    ) -> (Vec<u8>, usize) {
        use graphite_document_graph_storage::{NodeMetadataSource, Position};
        use graphite_graph_craft::document::{
            DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
            value::TaggedValue,
        };
        use graphite_resource::ResourceRegistry;
        use graphite_std::math_nodes::absolute_value;

        struct NumericPositions(bool);

        impl NodeMetadataSource for NumericPositions {
            fn position(&self, _: &[NodeId], node: NodeId) -> Option<Position> {
                if !self.0 {
                    return None;
                }
                let grid = match node {
                    NodeId(7) => [0, 0],
                    NodeId(8) => [8, 2],
                    NodeId(9) => [0, 4],
                    _ => return None,
                };
                Some(Position::Absolute(grid))
            }
        }

        let node = DocumentNode {
            implementation: DocumentNodeImplementation::ProtoNode(absolute_value::IDENTIFIER),
            inputs: vec![NodeInput::value(TaggedValue::Integer(-17), false)],
            ..Default::default()
        };
        let mut graph = NodeNetwork {
            nodes: [(NodeId(7), node)].into_iter().collect(),
            ..Default::default()
        };
        let second_input = match fixture {
            NumericGraphFixture::SingleNode => None,
            NumericGraphFixture::ExposedInput
            | NumericGraphFixture::ExposedInputWithAlternative => {
                Some(NodeInput::value(TaggedValue::Integer(0), true))
            }
            NumericGraphFixture::Connection
            | NumericGraphFixture::Reconnection
            | NumericGraphFixture::NestedConnection => Some(NodeInput::node(NodeId(7), 0)),
        };
        if let Some(second_input) = second_input {
            graph.nodes.insert(
                NodeId(8),
                DocumentNode {
                    implementation: DocumentNodeImplementation::ProtoNode(
                        absolute_value::IDENTIFIER,
                    ),
                    inputs: vec![second_input],
                    ..Default::default()
                },
            );
        }
        if matches!(
            fixture,
            NumericGraphFixture::Reconnection | NumericGraphFixture::ExposedInputWithAlternative
        ) {
            graph.nodes.insert(
                NodeId(9),
                DocumentNode {
                    implementation: DocumentNodeImplementation::ProtoNode(
                        absolute_value::IDENTIFIER,
                    ),
                    inputs: vec![NodeInput::value(TaggedValue::Integer(-31), false)],
                    ..Default::default()
                },
            );
        }
        if matches!(fixture, NumericGraphFixture::NestedConnection) {
            graph.exports = vec![NodeInput::node(NodeId(8), 0)];
            graph = NodeNetwork {
                nodes: [(
                    NodeId(24),
                    DocumentNode {
                        implementation: DocumentNodeImplementation::Network(graph),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            };
        }
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            184,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("pinned GDD storage");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(
                &graph,
                &NumericPositions(positioned),
                &ResourceRegistry::new(),
                &resources,
            )
            .expect("save pinned integer fixture");
        let original = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP");
        (original, owner.session().history().count())
    }

    async fn graph_navigation_archive() -> (Vec<u8>, usize) {
        graph_navigation_archive_with_number_target(false).await
    }

    async fn graph_layer_archive() -> (Vec<u8>, usize) {
        use graphite_document_graph_storage::{NodeMetadataSource, Position};
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_network_node_type;
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork};
        use graphite_resource::ResourceRegistry;

        struct LayerMetadata;

        impl NodeMetadataSource for LayerMetadata {
            fn is_layer(&self, path: &[NodeId], _: NodeId) -> bool {
                path.is_empty()
            }

            fn display_name(&self, path: &[NodeId], node: NodeId) -> Option<&str> {
                if !path.is_empty() {
                    return None;
                }
                match node {
                    NodeId(10) => Some("Background"),
                    NodeId(20) => Some("Artwork"),
                    NodeId(30) => Some("Child"),
                    NodeId(40) => Some("Unconnected"),
                    _ => None,
                }
            }

            fn position(&self, path: &[NodeId], node: NodeId) -> Option<Position> {
                if !path.is_empty() {
                    return None;
                }
                let position = match node {
                    NodeId(10) => [8, 4],
                    NodeId(20) => [8, 0],
                    NodeId(30) => [0, 0],
                    NodeId(40) => [16, 4],
                    _ => return None,
                };
                Some(Position::Absolute(position))
            }

            fn locked(&self, path: &[NodeId], node: NodeId) -> bool {
                path.is_empty() && node == NodeId(30)
            }
        }

        let merge = resolve_network_node_type("Merge").expect("pinned Merge template");
        let mut artwork = merge.node_template.clone();
        let [sibling, child] = artwork.inputs.as_mut_slice() else {
            panic!("two pinned Merge inputs")
        };
        *sibling = NodeInput::node(NodeId(10), 0);
        *child = NodeInput::node(NodeId(30), 0);
        let mut hidden = merge.node_template.clone();
        hidden.visible = false;
        let graph = NodeNetwork {
            nodes: [
                (NodeId(10), merge.node_template.clone().into_parts().0),
                (NodeId(20), artwork.into_parts().0),
                (NodeId(30), hidden.into_parts().0),
                (NodeId(40), merge.node_template.clone().into_parts().0),
            ]
            .into_iter()
            .collect(),
            exports: vec![NodeInput::node(NodeId(20), 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            623,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("source GDD");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&graph, &LayerMetadata, &ResourceRegistry::new(), &resources)
            .expect("source layer hierarchy");
        let source = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP");
        (source, owner.session().history().count())
    }

    async fn graph_navigation_archive_with_number_target(number_target: bool) -> (Vec<u8>, usize) {
        use graphite_document_graph_storage::{NodeMetadataSource, Position};
        use graphite_graph_craft::{
            ProtoNodeIdentifier, concrete,
            document::{
                DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
                value::TaggedValue,
            },
        };
        use graphite_resource::ResourceRegistry;
        use graphite_std::{math_nodes::absolute_value, ops::passthrough};

        struct Names;

        impl NodeMetadataSource for Names {
            fn position(&self, _: &[NodeId], local_id: NodeId) -> Option<Position> {
                let position = match local_id {
                    NodeId(7) => [0, 0],
                    NodeId(9) => [7, 0],
                    NodeId(10) => [0, 3],
                    NodeId(11) => [7, 3],
                    NodeId(12) => [0, 6],
                    NodeId(24) => [7, 6],
                    _ => return None,
                };
                Some(Position::Absolute(position))
            }

            fn display_name(&self, _: &[NodeId], local_id: NodeId) -> Option<&str> {
                (local_id == NodeId(7)).then_some("Repeated value")
            }

            fn input_name(&self, path: &[NodeId], local_id: NodeId, index: usize) -> Option<&str> {
                if path.is_empty() && local_id == NodeId(24) {
                    ["First numeric input", "Second numeric input"]
                        .get(index)
                        .copied()
                } else {
                    None
                }
            }

            fn output_names(&self, path: &[NodeId], local_id: NodeId) -> Vec<String> {
                if path.is_empty() && local_id == NodeId(24) {
                    vec![
                        "First numeric output".into(),
                        "Second numeric output".into(),
                    ]
                } else {
                    Vec::new()
                }
            }
        }

        let proto = |identifier, inputs| DocumentNode {
            implementation: DocumentNodeImplementation::ProtoNode(identifier),
            inputs,
            ..Default::default()
        };
        let nested = NodeNetwork {
            nodes: [
                (
                    NodeId(7),
                    proto(
                        absolute_value::IDENTIFIER,
                        vec![NodeInput::import(concrete!(f64), 0)],
                    ),
                ),
                (
                    NodeId(9),
                    proto(
                        absolute_value::IDENTIFIER,
                        vec![NodeInput::import(concrete!(f64), 1)],
                    ),
                ),
                (
                    NodeId(10),
                    proto(
                        passthrough::IDENTIFIER,
                        vec![NodeInput::value(
                            TaggedValue::String("literal".into()),
                            false,
                        )],
                    ),
                ),
                (
                    NodeId(11),
                    proto(
                        absolute_value::IDENTIFIER,
                        vec![NodeInput::node(NodeId(10), 0)],
                    ),
                ),
                (
                    NodeId(12),
                    proto(
                        absolute_value::IDENTIFIER,
                        vec![if number_target {
                            NodeInput::value(TaggedValue::Number(-17.), true)
                        } else {
                            NodeInput::value(TaggedValue::Integer(-17), false)
                        }],
                    ),
                ),
                (
                    NodeId(24),
                    DocumentNode {
                        implementation: DocumentNodeImplementation::Network(NodeNetwork::default()),
                        ..Default::default()
                    },
                ),
            ]
            .into_iter()
            .collect(),
            exports: vec![NodeInput::node(NodeId(7), 0), NodeInput::node(NodeId(9), 0)],
            ..Default::default()
        };
        let root = NodeNetwork {
            nodes: [
                (
                    NodeId(7),
                    proto(
                        ProtoNodeIdentifier::with_owned_string(
                            "graphite_fixture::Unregistered".into(),
                        ),
                        Vec::new(),
                    ),
                ),
                (
                    NodeId(24),
                    DocumentNode {
                        implementation: DocumentNodeImplementation::Network(nested),
                        inputs: vec![
                            NodeInput::value(TaggedValue::Number(-17.), true),
                            NodeInput::value(TaggedValue::Number(23.), true),
                        ],
                        ..Default::default()
                    },
                ),
            ]
            .into_iter()
            .collect(),
            exports: vec![NodeInput::node(NodeId(24), 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            622,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("source GDD");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&root, &Names, &ResourceRegistry::new(), &resources)
            .expect("nested fixture with an incompatible link and unknown proto node");
        let source = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP");
        (source, owner.session().history().count())
    }

    #[gpui::test]
    async fn at_gra_graph_019_020_2_workspace_appends_primary_and_secondary_imports(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/import-network.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");

        for (inside_child, destination, original_imports, added_history) in [
            (true, "/project/first-import.gdd", 0, 1),
            (false, "/project/third-import.gdd", 2, 2),
        ] {
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
            click(window, "GRAPHITE-GDD-ENTER-NETWORK");
            paint(window);
            if inside_child {
                click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
                paint(window);
                click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-7");
                paint(window);
                click(window, "GRAPHITE-GDD-ENTER-NETWORK");
                paint(window);
            }
            let (network_id, export_count) = current.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected the scoped graph");
                };
                let network = form.active_network().expect("original nested network");
                assert_eq!(network.import_count, original_imports);
                (network.id, network.export_count)
            });
            click(window, "GRAPHITE-GDD-ADD-NETWORK-IMPORT");
            window.simulate_new_path_selection(|_| None);
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-ADD-NETWORK-IMPORT");
            let destination = PathBuf::from(destination);
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopened import folder");
            assert_ne!(reopened, current);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected the new Graphite folder");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let graph = contents.graph_inspection().await.expect("saved imports");
            let network = graph
                .networks
                .iter()
                .find(|network| network.id == network_id)
                .expect("same addressed network");
            assert_eq!(network.import_count, original_imports + 1);
            assert_eq!(network.export_count, export_count);
            assert_eq!(contents.history_len(), original_history + added_history);
            current = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_007_2_workspace_connects_and_previews_a_typed_network_import(
        cx: &mut TestAppContext,
    ) {
        verify_scoped_numeric_import_wire(cx, ScopedImportScenario::Connect).await;
    }

    #[gpui::test]
    async fn at_gra_graph_008_2_workspace_reconnects_and_previews_a_typed_network_import(
        cx: &mut TestAppContext,
    ) {
        verify_scoped_numeric_import_wire(cx, ScopedImportScenario::Reconnect).await;
    }

    #[gpui::test]
    async fn at_r17_3_graphite_wire_publication_preserves_the_open_sim_workflow(
        cx: &mut TestAppContext,
    ) {
        verify_scoped_numeric_import_wire(cx, ScopedImportScenario::ConnectBesideSim).await;
    }

    #[gpui::test]
    async fn at_gra_graph_007_2_workspace_connects_a_typed_import_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_scoped_numeric_import_wire(cx, ScopedImportScenario::ConnectFromCanvas).await;
    }

    enum ScopedImportScenario {
        Connect,
        Reconnect,
        ConnectBesideSim,
        ConnectFromCanvas,
    }

    async fn verify_scoped_numeric_import_wire(
        cx: &mut TestAppContext,
        scenario: ScopedImportScenario,
    ) {
        let reconnect = matches!(scenario, ScopedImportScenario::Reconnect);
        let canvas = matches!(scenario, ScopedImportScenario::ConnectFromCanvas);
        let (target, selector, import, expected, added_history) = if reconnect {
            (
                "GRAPHITE-GDD-NODE-COPY-ITEM-3",
                "GRAPHITE-GDD-CONNECT-IMPORT-0",
                0,
                17.,
                2,
            )
        } else {
            (
                "GRAPHITE-GDD-NODE-COPY-ITEM-6",
                "GRAPHITE-GDD-CONNECT-IMPORT-1",
                1,
                23.,
                1,
            )
        };
        let (original, original_history) =
            graph_navigation_archive_with_number_target(!reconnect).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let sim_workflow = if matches!(scenario, ScopedImportScenario::ConnectBesideSim) {
            let bytes = br#"{"version":0.4,"last_node_id":1,"last_link_id":0,"nodes":[{"id":"workflow-source","type":"Source","pos":[20,30],"size":[140,80],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"image","type":"IMAGE","links":[]}],"properties":{},"widgets_values":[]}],"links":[],"groups":[],"config":{},"extra":{}}"#;
            let path = PathBuf::from("/project/independent.workflow.json");
            window.update(|_, cx| sim_ui::register_native_workflow_project_item(cx));
            filesystem
                .write(&path, bytes)
                .await
                .expect("native Sim workflow fixture");
            window.run_until_parked();
            let item = workspace
                .update_in(window, |workspace, window, cx| {
                    workspace.open_abs_path(path.clone(), Default::default(), window, cx)
                })
                .await
                .expect("open the independent native Sim workflow")
                .downcast::<sim_ui::GraphWorkspaceItem>()
                .expect("Sim owns the native workflow");
            window.run_until_parked();
            paint(window);
            let (saved, exported) = item.read_with(window, |item, cx| {
                let model = item.model();
                let graph = model.document().unwrap_or_else(|| {
                    panic!(
                        "expected an editable Sim graph, got {:?}",
                        model.read_only_diagnostic()
                    )
                });
                assert_eq!(graph.root.nodes.len(), 1);
                assert_eq!(model.original_bytes(), bytes);
                assert!(!item.is_dirty(cx));
                (
                    model.encode().expect("serialize the open Sim workflow"),
                    graph
                        .to_workflow_bytes()
                        .expect("serialize the Sim workflow format"),
                )
            });
            Some((item, path, bytes, saved, exported))
        } else {
            None
        };
        let source_path = PathBuf::from("/project/typed-import-wire.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        if sim_workflow.is_some() {
            choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-5");
            click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
            window.run_until_parked();
            source.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected the nested Graphite graph");
                };
                assert!(matches!(form.output_preview, NodeOutputPreview::Failed(_)));
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected the unchanged GDD");
                };
                assert_eq!(contents.history_len(), original_history);
                assert!(!item.is_dirty(cx));
            });
            paint(window);
            click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
            paint(window);
            click(window, target);
            paint(window);
        } else {
            choose_scoped_copy_node(window, target);
        }
        if canvas {
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-Import-2");
            paint(window);
            click(window, "GRAPHITE-GDD-CANCEL-GRAPH-CONNECTION");
            paint(window);
            assert!(
                window
                    .debug_bounds("GRAPHITE-GDD-CANCEL-GRAPH-CONNECTION")
                    .is_none()
            );
            click(window, "GRAPHITE-GDD-GRAPH-Import-2");
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-4");
            paint(window);
            source.read_with(window, |item, cx| {
                assert!(matches!(&item.copy_feedback, Some((message, Color::Error))
                    if message.as_ref() == "Choose an exposed numeric input that accepts this import."));
                assert!(item.copy_task.is_none());
                assert!(!item.is_dirty(cx));
            });
        }
        let connect = |window: &mut gpui::VisualTestContext| {
            if canvas {
                click(window, "GRAPHITE-GDD-GRAPH-Import-2");
                paint(window);
                click(window, "GRAPHITE-GDD-GRAPH-NODE-6");
            } else {
                click(window, selector);
            }
        };
        connect(window);
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));
        paint(window);
        connect(window);
        let destination = PathBuf::from("/project/wired-numeric-import.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the typed-wire folder");
        assert_ne!(reopened, source);
        choose_scoped_copy_node(window, target);
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        window.run_until_parked();
        reopened.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the numeric preview");
            };
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the original Graphite format");
            };
            let node = form.selected_node().expect("same target node");
            assert_eq!(node.numeric_import_source(), Some(import));
            assert!(matches!(
                form.output_preview,
                NodeOutputPreview::Ready(GddNumericValue::Number(value)) if value == expected
            ));
            assert_eq!(contents.history_len(), original_history + added_history);
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
        if let Some((item, path, original_sim, before, exported)) = sim_workflow {
            assert!(workspace.read_with(window, |workspace, cx| {
                workspace
                    .items_of_type::<sim_ui::GraphWorkspaceItem>(cx)
                    .any(|open_item| open_item == item)
            }));
            item.read_with(window, |item, cx| {
                assert_eq!(item.model().encode().expect("serialize Sim again"), before);
                let graph = item.model().document().expect("Sim stayed editable");
                assert_eq!(
                    graph
                        .to_workflow_bytes()
                        .expect("serialize the Sim workflow again"),
                    exported
                );
                assert!(!item.is_dirty(cx));
            });
            assert_eq!(
                filesystem
                    .load_bytes(&path)
                    .await
                    .expect("Sim source bytes"),
                original_sim
            );
        }
    }

    #[gpui::test]
    async fn at_gra_graph_009_2_workspace_disconnects_a_scoped_import_and_previews_its_fallback(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_disconnects_scoped_import(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_009_2_workspace_disconnects_a_scoped_import_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_disconnects_scoped_import(cx, true).await;
    }

    async fn verify_workspace_disconnects_scoped_import(cx: &mut TestAppContext, canvas: bool) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/disconnect-import.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        let selector = "GRAPHITE-GDD-NODE-COPY-ITEM-3";
        choose_scoped_copy_node(window, selector);
        let command = if canvas {
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            assert!(
                window
                    .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-INPUT-6")
                    .is_none()
            );
            "GRAPHITE-GDD-GRAPH-NODE-INPUT-3"
        } else {
            "GRAPHITE-GDD-DISCONNECT-VALUE"
        };
        click(window, command);
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));
        paint(window);
        click(window, command);
        let destination = PathBuf::from("/project/disconnected-import.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the disconnected folder");
        assert_ne!(reopened, source);
        choose_scoped_copy_node(window, selector);
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        window.run_until_parked();
        reopened.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the numeric preview");
            };
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the original Graphite format");
            };
            let node = form.selected_node().expect("same target node");
            assert!(!node.has_numeric_connection());
            assert_eq!(node.numeric_input_exposed(), Some(true));
            assert!(matches!(
                form.output_preview,
                NodeOutputPreview::Ready(GddNumericValue::Number(0.))
            ));
            let network = form.active_network().expect("same scope");
            assert_eq!(network.import_name(1), Some("Second numeric input"));
            assert_eq!(network.import_count, 2);
            assert_eq!(contents.history_len(), original_history + 1);
            assert!(!item.is_dirty(cx));
        });
        if canvas {
            paint(window);
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            assert!(
                window
                    .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-INPUT-3")
                    .is_none()
            );
            assert!(
                window
                    .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-INPUT-2")
                    .is_some()
            );
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_010_2_workspace_inserts_a_numeric_node_on_the_selected_wire(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_inserts_numeric_node_on_wire(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_010_2_workspace_inserts_a_numeric_node_from_the_canvas_wire_menu(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_inserts_numeric_node_on_wire(cx, true).await;
    }

    #[gpui::test]
    async fn at_gra_graph_010_2_workspace_inserts_a_node_on_a_working_canvas_wire(
        cx: &mut TestAppContext,
    ) {
        let (original, _) =
            numeric_graph_archive_with_layout(NumericGraphFixture::Connection, true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/working-insert-wire.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("original archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open wired archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let (field, source_id, target_id) = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("numeric graph")
            };
            let [source, target] = form.nodes.as_slice() else {
                panic!("one original wire")
            };
            (form.value.clone(), source.storage_id(), target.storage_id())
        });
        field.update_in(window, |editor, window, cx| {
            editor.set_text("-19", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let canvas = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH")
            .expect("visible working canvas");
        right_click_saved_graph(window, "GRAPHITE-GDD-GRAPH-NODE-INPUT-1");
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-INSERT-ON-WIRE-Sign")
                .is_some()
        );
        click_graph_point(
            window,
            canvas.origin + gpui::point(px(550.), px(180.)),
            gpui::MouseButton::Left,
            gpui::Modifiers::default(),
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-INSERT-ON-WIRE-Sign")
                .is_none()
        );
        let before = item.read_with(window, |item, cx| {
            assert!(matches!(&item.node_copy, NodeCopyPanel::Ready(form) if form.nodes.len() == 2));
            item.model.read(cx).revision.clone()
        });

        right_click_saved_graph(window, "GRAPHITE-GDD-GRAPH-NODE-INPUT-1");
        click(window, "GRAPHITE-GDD-INSERT-ON-WIRE-Sign");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("could not type-check the inserted node")));
            assert!(Arc::ptr_eq(&item.model.read(cx).revision, &before));
        });

        right_click_saved_graph(window, "GRAPHITE-GDD-GRAPH-NODE-INPUT-1");
        click(window, "GRAPHITE-GDD-INSERT-ON-WIRE-Absolute Value");
        window.run_until_parked();
        paint(window);
        let inserted_id = item.read_with(window, |item, cx| {
            assert!(item.is_dirty(cx));
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working graph")
            };
            let target = form
                .nodes
                .iter()
                .find(|node| node.storage_id() == target_id)
                .expect("original target");
            let inserted = target
                .numeric_connection_source(&form.nodes)
                .expect("rewired target");
            assert_eq!(inserted.display_name(), "Absolute Value");
            assert_ne!(inserted.storage_id(), source_id);
            assert_eq!(
                inserted
                    .numeric_connection_source(&form.nodes)
                    .map(GddGraphNode::storage_id),
                Some(source_id)
            );
            assert!(
                matches!(&form.saved_graph, SavedGraphPanel::Ready(graph) if graph.wires.len() == 2)
            );
            inserted.storage_id()
        });
        click(window, "GRAPHITE-GDD-GRAPH-NODE-1");
        paint(window);
        for (history, upstream) in [
            (None, inserted_id),
            (Some("GRAPHITE-GDD-UNDO"), source_id),
            (Some("GRAPHITE-GDD-REDO"), inserted_id),
        ] {
            if let Some(action) = history {
                click(window, action);
                window.run_until_parked();
                paint(window);
            }
            click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("working graph")
                };
                let target = form.selected_node().expect("retained target");
                assert_eq!(target.storage_id(), target_id);
                assert_eq!(
                    target
                        .numeric_connection_source(&form.nodes)
                        .map(GddGraphNode::storage_id),
                    Some(upstream)
                );
                assert!(matches!(
                    form.output_preview,
                    NodeOutputPreview::Ready(GddNumericValue::Integer(19))
                ));
            });
        }

        let destination = PathBuf::from("/project/saved-working-insert-wire.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("published working copy");
        let nodes = saved.graph_nodes().await.expect("reopened working wires");
        let target = nodes
            .iter()
            .find(|node| node.storage_id() == target_id)
            .expect("durable original target");
        assert_eq!(
            target
                .numeric_connection_source(&nodes)
                .map(GddGraphNode::storage_id),
            Some(inserted_id)
        );
        assert!(matches!(
            saved.preview_graph_numeric_output(target).await,
            Ok(GddNumericValue::Integer(19))
        ));
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("preserved original archive"),
            original
        );
    }

    async fn verify_workspace_inserts_numeric_node_on_wire(cx: &mut TestAppContext, canvas: bool) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/insert-on-wire.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-3");
        let label = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the scoped graph");
            };
            let node = form.selected_node().expect("source wire target");
            assert_eq!(node.numeric_import_source(), Some(1));
            node.label().to_owned()
        });
        if canvas {
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-2");
            source.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected the saved graph");
                };
                assert_eq!(
                    form.selected_node()
                        .and_then(GddGraphNode::numeric_import_source),
                    Some(0)
                );
            });
        }
        let insert = |window: &mut gpui::VisualTestContext| {
            if canvas {
                right_click_saved_graph(window, "GRAPHITE-GDD-GRAPH-NODE-INPUT-3");
            } else {
                click(window, "GRAPHITE-GDD-INSERT-ON-WIRE");
            }
            paint(window);
            click(window, "GRAPHITE-GDD-INSERT-ON-WIRE-Sign");
        };
        insert(window);
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));
        paint(window);
        insert(window);
        let destination = PathBuf::from("/project/inserted-on-wire.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the inserted node");
        assert_ne!(reopened, source);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected native Graphite folder");
            };
            contents.clone()
        });
        let graph = contents.graph_inspection().await.expect("saved graph");
        let selected = match graph.nodes.iter().position(|node| node.label() == label) {
            Some(3) => "GRAPHITE-GDD-NODE-COPY-ITEM-3",
            Some(4) => "GRAPHITE-GDD-NODE-COPY-ITEM-4",
            _ => panic!("expected the original node and one new node in the selected scope"),
        };
        choose_scoped_copy_node(window, selected);
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        window.run_until_parked();
        reopened.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the numeric preview");
            };
            let node = form.selected_node().expect("same target node");
            let added = node
                .numeric_connection_source(&form.nodes)
                .expect("inserted upstream Sign");
            assert!(added.label().contains("Sign"));
            assert_eq!(added.numeric_import_source(), Some(1));
            assert!(matches!(
                form.output_preview,
                NodeOutputPreview::Ready(GddNumericValue::Number(1.))
            ));
            assert!(contents.history_len() > original_history);
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_pastes_and_enters_a_copied_nested_graph(
        cx: &mut TestAppContext,
    ) {
        let (bytes, _) = numeric_graph_archive(NumericGraphFixture::NestedConnection).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/clipboard-nested-source.gdd.zip");
        filesystem
            .write(&source_path, &bytes)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let original = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected original node scope");
            };
            form.selected_node()
                .and_then(|node| node.nested_network())
                .expect("original nested identity")
        });
        click(window, "GRAPHITE-GDD-COPY-NODES");
        window.run_until_parked();
        assert!(
            window
                .read_from_clipboard()
                .and_then(|item| item.text())
                .is_some_and(|text| text.starts_with("graphite: "))
        );
        paint(window);
        click(window, "GRAPHITE-GDD-PASTE-NODES");
        let destination = PathBuf::from("/project/clipboard-nested-paste.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the pasted graph");
        assert_ne!(reopened, source);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the native pasted Graphite folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let graph = contents.graph_inspection().await.expect("pasted hierarchy");
        let scope = graph
            .networks
            .iter()
            .find(|network| network.parent.is_some() && network.id != original)
            .expect("new stable network identity");
        let selected = match graph
            .nodes
            .iter()
            .position(|node| node.nested_network() == Some(scope.id))
        {
            Some(0) => "GRAPHITE-GDD-NODE-COPY-ITEM-0",
            Some(1) => "GRAPHITE-GDD-NODE-COPY-ITEM-1",
            _ => panic!("expected two encapsulating nodes in root"),
        };
        choose_copy_node(window, selected);
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        reopened.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the copied inner scope");
            };
            assert_eq!(
                form.active_network().map(|network| network.id),
                Some(scope.id)
            );
            assert_eq!(
                form.nodes
                    .iter()
                    .filter(|node| form.includes_node(node))
                    .count(),
                2
            );
        });
        let consumer = graph
            .nodes
            .iter()
            .find(|node| node.network_id() == scope.id && node.has_numeric_connection())
            .expect("new linked inner node");
        assert!(matches!(
            contents
                .preview_graph_numeric_output(consumer)
                .await
                .expect("execute the copied linkage"),
            GddNumericValue::Integer(17)
        ));
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            bytes
        );
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_pastes_a_numeric_node_into_a_new_blank_graphite_folder(
        cx: &mut TestAppContext,
    ) {
        let (bytes, _) = numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let original_path = PathBuf::from("/project/clipboard-origin.gdd.zip");
        filesystem
            .write(&original_path, &bytes)
            .await
            .expect("clipboard source");
        window.run_until_parked();
        workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(original_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-COPY-NODES");
        window.run_until_parked();
        assert!(
            window
                .read_from_clipboard()
                .and_then(|item| item.text())
                .is_some_and(|text| text.starts_with("graphite: "))
        );

        let blank_path = PathBuf::from("/project/clipboard-blank.gdd");
        window.dispatch_action(NewGraphiteFolder);
        window.simulate_new_path_selection(|_| Some(blank_path.clone()));
        window.run_until_parked();
        let blank = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("native blank Graphite item");
        let original = read_graphite_folder(filesystem.as_ref(), &blank_path)
            .await
            .expect("original blank folder")
            .folder_files()
            .expect("original bytes");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY");
        window.run_until_parked();
        paint(window);
        assert!(blank.read_with(window, |item, _| {
            matches!(&item.node_copy, NodeCopyPanel::Ready(form) if form.nodes.is_empty())
        }));
        click(window, "GRAPHITE-GDD-PASTE-NODES");
        let destination = PathBuf::from("/project/clipboard-populated.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the pasted folder");
        assert_ne!(reopened, blank);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the native pasted Graphite folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents.graph_nodes().await.expect("saved root nodes");
        let [node] = nodes.as_slice() else {
            panic!("expected the single copied node");
        };
        assert!(matches!(
            contents
                .preview_graph_numeric_output(node)
                .await
                .expect("pasted source node"),
            GddNumericValue::Integer(17)
        ));
        assert!(contents.can_move_history(GddHistoryDirection::Undo));
        assert_eq!(
            read_graphite_folder(filesystem.as_ref(), &blank_path)
                .await
                .expect("original blank folder")
                .folder_files()
                .expect("original bytes"),
            original
        );
        assert_eq!(
            filesystem
                .load_bytes(&original_path)
                .await
                .expect("clipboard source archive"),
            bytes
        );
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_cuts_a_node_only_after_saving_its_folder_copy(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_cuts_numeric_node(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_cut_shortcut_waits_for_folder_publication(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_cuts_numeric_node(cx, true).await;
    }

    async fn verify_workspace_cuts_numeric_node(cx: &mut TestAppContext, keyboard: bool) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/cut-source.gdd.zip");
        let blocked = PathBuf::from("/project/existing.gdd");
        let destination = PathBuf::from("/project/cut-copy.gdd");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        filesystem
            .write(&blocked, b"existing destination")
            .await
            .expect("existing file");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        if keyboard {
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-0");
        }
        let cut = |window: &mut gpui::VisualTestContext| {
            if keyboard {
                saved_graph_shortcut(window, "x");
            } else {
                click(window, "GRAPHITE-GDD-CUT-NODE");
            }
        };
        window.write_to_clipboard(ClipboardItem::new_string("previous clipboard".into()));
        for path in [None, Some(blocked.clone())] {
            paint(window);
            cut(window);
            window.simulate_new_path_selection(|_| path);
            window.run_until_parked();
            assert_eq!(
                window
                    .read_from_clipboard()
                    .and_then(|item| item.text())
                    .as_deref(),
                Some("previous clipboard")
            );
            assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));
        }
        assert_eq!(
            filesystem
                .load_bytes(&blocked)
                .await
                .expect("read the existing target"),
            b"existing destination"
        );
        paint(window);
        cut(window);
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let clipboard = window
            .read_from_clipboard()
            .and_then(|item| item.text())
            .expect("the clipboard was updated after publication");
        assert!(clipboard.starts_with("graphite: "));

        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the cut folder");
        assert_ne!(reopened, source);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected a native Graphite folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        assert!(
            contents
                .graph_nodes()
                .await
                .expect("saved graph")
                .is_empty()
        );
        assert!(contents.history_len() > original_history);
        let original_nodes = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the original node scope");
            };
            form.nodes.len()
        });
        assert_eq!(original_nodes, 1);
        let restored = contents
            .move_history_in_copy(GddHistoryDirection::Undo, ARCHIVE_BUDGET)
            .await
            .expect("one undo restores the cut node");
        assert_eq!(
            restored.graph_nodes().await.expect("restored graph").len(),
            1
        );
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_cuts_the_selected_connected_group(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = numeric_graph_archive(NumericGraphFixture::Reconnection).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/cut-group.gdd.zip");
        let destination = PathBuf::from("/project/cut-group-copy.gdd");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-CLIPBOARD-SELECT-NODE");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        paint(window);
        click(window, "GRAPHITE-GDD-CLIPBOARD-SELECT-NODE");
        paint(window);
        window.write_to_clipboard(ClipboardItem::new_string("previous clipboard".into()));
        click(window, "GRAPHITE-GDD-CUT-NODES");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert_eq!(
            window
                .read_from_clipboard()
                .and_then(|item| item.text())
                .as_deref(),
            Some("previous clipboard")
        );
        paint(window);
        click(window, "GRAPHITE-GDD-CUT-NODES");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let clipboard = window
            .read_from_clipboard()
            .and_then(|item| item.text())
            .expect("clipboard after folder publication");
        assert!(clipboard.starts_with("graphite: "));
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the cut folder");
        assert_ne!(reopened, source);
        let cut = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected native Graphite folder");
            };
            contents.clone()
        });
        let nodes = cut.graph_nodes().await.expect("saved cut graph");
        let [survivor] = nodes.as_slice() else {
            panic!("expected only the unselected node");
        };
        assert!(matches!(
            cut.preview_graph_numeric_output(survivor)
                .await
                .expect("execute the unselected node"),
            GddNumericValue::Integer(31)
        ));
        let restored = cut
            .move_history_in_copy(GddHistoryDirection::Undo, ARCHIVE_BUDGET)
            .await
            .expect("one undo restores the complete selection");
        let nodes = restored.graph_nodes().await.expect("restored group");
        assert_eq!(nodes.len(), 3);
        assert!(
            nodes
                .iter()
                .any(|node| node.numeric_connection_source(&nodes).is_some())
        );
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_cuts_and_pastes_in_the_working_canvas(
        cx: &mut TestAppContext,
    ) {
        let (original, _) =
            numeric_graph_archive_with_layout(NumericGraphFixture::Reconnection, true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/working-clipboard.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        window.run_until_parked();
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let snapshot = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, cx| {
                let model = item.model.read(cx);
                (
                    model.gdd_contents().expect("GDD").clone(),
                    model.revision.clone(),
                    item.is_dirty(cx),
                )
            })
        };
        let (source, initial_revision, dirty) = snapshot(window);
        assert!(!dirty);
        let initial_nodes = source.graph_nodes().await.expect("original nodes");
        let [source, target, survivor] = initial_nodes.as_slice() else {
            panic!("wired pair and survivor")
        };
        let initial_ids = initial_nodes
            .iter()
            .map(GddGraphNode::storage_id)
            .collect::<BTreeSet<_>>();
        let survivor_id = survivor.storage_id();
        assert_eq!(
            target
                .numeric_connection_source(&initial_nodes)
                .map(GddGraphNode::storage_id),
            Some(source.storage_id())
        );
        let clipboard = |window: &mut gpui::VisualTestContext| {
            window
                .read_from_clipboard()
                .and_then(|item| item.text())
                .expect("clipboard contents")
        };

        click(window, "GRAPHITE-GDD-GRAPH-NODE-0");
        paint(window);
        window.write_to_clipboard(ClipboardItem::new_string("previous clipboard".into()));
        saved_graph_shortcut(window, "x");
        assert_eq!(clipboard(window), "previous clipboard");
        let (_, revision, dirty) = snapshot(window);
        assert!(Arc::ptr_eq(&initial_revision, &revision));
        assert!(!dirty);

        click_saved_graph(
            window,
            "GRAPHITE-GDD-GRAPH-NODE-1",
            gpui::MouseButton::Left,
            gpui::Modifiers {
                shift: true,
                ..Default::default()
            },
        );
        saved_graph_shortcut(window, "x");
        let cut_clipboard = clipboard(window);
        assert!(cut_clipboard.starts_with("graphite: "));
        let (cut, revision, dirty) = snapshot(window);
        assert!(!Arc::ptr_eq(&initial_revision, &revision));
        assert!(dirty);
        let remaining = cut.graph_nodes().await.expect("working cut");
        assert!(
            remaining
                .iter()
                .map(GddGraphNode::storage_id)
                .eq([survivor_id])
        );

        click(window, "GRAPHITE-GDD-UNDO");
        window.run_until_parked();
        paint(window);
        let (restored, _, dirty) = snapshot(window);
        assert!(dirty);
        assert!(restored.can_move_history(GddHistoryDirection::Redo));
        let restored_nodes = restored.graph_nodes().await.expect("restored nodes");
        assert_eq!(
            restored_nodes
                .iter()
                .map(GddGraphNode::storage_id)
                .collect::<BTreeSet<_>>(),
            initial_ids
        );
        let restored_target = restored_nodes
            .iter()
            .find(|node| node.storage_id() == target.storage_id())
            .expect("restored wire target");
        assert_eq!(
            restored_target
                .numeric_connection_source(&restored_nodes)
                .map(GddGraphNode::storage_id),
            Some(source.storage_id())
        );
        click(window, "GRAPHITE-GDD-REDO");
        window.run_until_parked();
        paint(window);
        click_saved_graph(
            window,
            "GRAPHITE-GDD-SAVED-GRAPH",
            gpui::MouseButton::Left,
            Default::default(),
        );
        let (_, before_paste, _) = snapshot(window);
        window.write_to_clipboard(ClipboardItem::new_string("graphite: {".into()));
        saved_graph_shortcut(window, "v");
        assert!(Arc::ptr_eq(&before_paste, &snapshot(window).1));
        assert_eq!(clipboard(window), "graphite: {");
        item.read_with(window, |item, _| {
            assert!(matches!(&item.copy_feedback, Some((_, Color::Error))))
        });

        window.write_to_clipboard(ClipboardItem::new_string(cut_clipboard.clone()));
        saved_graph_shortcut(window, "v");
        let (pasted, revision, dirty) = snapshot(window);
        assert!(!Arc::ptr_eq(&before_paste, &revision));
        assert!(dirty);
        assert_eq!(clipboard(window), cut_clipboard);
        let check_paste = async |document: &InspectedGddV1| {
            let nodes = document.graph_nodes().await.expect("pasted working graph");
            assert_eq!(nodes.len(), 3);
            assert!(nodes.iter().any(|node| node.storage_id() == survivor_id));
            let target = nodes
                .iter()
                .find(|node| node.has_numeric_connection())
                .expect("restored internal wire");
            let source = target
                .numeric_connection_source(&nodes)
                .expect("wire source");
            assert!(!initial_ids.contains(&source.storage_id()));
            assert!(!initial_ids.contains(&target.storage_id()));
            assert!(matches!(
                document.preview_graph_numeric_output(target).await,
                Ok(GddNumericValue::Integer(17))
            ));
            [source.storage_id(), target.storage_id()]
        };
        let pasted_ids = check_paste(&pasted).await;
        click(window, "GRAPHITE-GDD-UNDO");
        window.run_until_parked();
        paint(window);
        assert!(
            snapshot(window)
                .0
                .graph_nodes()
                .await
                .expect("undo paste")
                .iter()
                .map(GddGraphNode::storage_id)
                .eq([survivor_id])
        );
        click(window, "GRAPHITE-GDD-REDO");
        window.run_until_parked();
        paint(window);
        assert_eq!(check_paste(&snapshot(window).0).await, pasted_ids);
        let destination = PathBuf::from("/project/saved-working-clipboard.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen working clipboard changes");
        assert_eq!(check_paste(&saved).await, pasted_ids);
        assert!(!snapshot(window).2);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_copies_and_pastes_a_connected_graph_node_group(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_graph_group_copy(cx, "GRAPHITE-GDD-PASTE-NODES", GraphGroupRoute::Picker)
            .await;
    }

    #[gpui::test]
    async fn at_gra_graph_012_2_workspace_duplicates_the_selected_graph_group(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_graph_group_copy(
            cx,
            "GRAPHITE-GDD-DUPLICATE-NODES",
            GraphGroupRoute::Picker,
        )
        .await;
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_copies_a_canvas_modifier_selection(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_graph_group_copy(cx, "GRAPHITE-GDD-PASTE-NODES", GraphGroupRoute::Canvas)
            .await;
    }

    #[gpui::test]
    async fn at_gra_graph_011_2_workspace_copies_and_pastes_with_canvas_shortcuts(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_graph_group_copy(
            cx,
            "GRAPHITE-GDD-PASTE-NODES",
            GraphGroupRoute::Keyboard,
        )
        .await;
    }

    enum GraphGroupRoute {
        Picker,
        Canvas,
        Keyboard,
    }

    async fn verify_workspace_graph_group_copy(
        cx: &mut TestAppContext,
        action: &'static str,
        route: GraphGroupRoute,
    ) {
        let canvas = !matches!(route, GraphGroupRoute::Picker);
        let keyboard = matches!(route, GraphGroupRoute::Keyboard);
        let (original, original_history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::Connection, canvas).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/graph-clipboard.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let original_labels = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected clipboard nodes");
            };
            form.nodes
                .iter()
                .map(|node| node.label().to_owned())
                .collect::<BTreeSet<_>>()
        });
        window.write_to_clipboard(ClipboardItem::new_string("previous clipboard".into()));
        if keyboard {
            saved_graph_shortcut(window, "c");
            assert_eq!(
                window
                    .read_from_clipboard()
                    .and_then(|item| item.text())
                    .as_deref(),
                Some("previous clipboard")
            );
        }
        if canvas {
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-0");
            let shift = gpui::Modifiers {
                shift: true,
                ..Default::default()
            };
            let accelerator = gpui::Modifiers::secondary_key();
            for (selector, modifiers, selection) in [
                ("GRAPHITE-GDD-GRAPH-NODE-1", shift, &[0, 1][..]),
                ("GRAPHITE-GDD-GRAPH-NODE-1", shift, &[0][..]),
                ("GRAPHITE-GDD-GRAPH-NODE-0", accelerator, &[][..]),
            ] {
                click_saved_graph(window, selector, gpui::MouseButton::Left, modifiers);
                source.read_with(window, |item, cx| {
                    let NodeCopyPanel::Ready(form) = &item.node_copy else {
                        panic!("expected the saved graph selection");
                    };
                    assert_eq!(
                        form.node_selection.iter().copied().collect::<Vec<_>>(),
                        selection
                    );
                    assert_eq!(form.selected_graph_nodes().len(), selection.len());
                    assert!(!item.is_dirty(cx));
                });
            }
            click(window, "GRAPHITE-GDD-GRAPH-NODE-0");
            click_saved_graph(
                window,
                "GRAPHITE-GDD-GRAPH-NODE-1",
                gpui::MouseButton::Left,
                accelerator,
            );
            if !keyboard {
                click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            }
        } else {
            click(window, "GRAPHITE-GDD-CLIPBOARD-SELECT-NODE");
            paint(window);
            click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
            paint(window);
            click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
            paint(window);
            click(window, "GRAPHITE-GDD-CLIPBOARD-SELECT-NODE");
            paint(window);
        }
        let paste = action == "GRAPHITE-GDD-PASTE-NODES";
        if paste {
            if keyboard {
                saved_graph_shortcut(window, "c");
            } else {
                click(window, "GRAPHITE-GDD-COPY-NODES");
            }
            window.run_until_parked();
        }
        let clipboard = window
            .read_from_clipboard()
            .and_then(|item| item.text())
            .expect("native clipboard text");
        if paste {
            assert!(clipboard.starts_with("graphite: "));
        } else {
            assert_eq!(clipboard, "previous clipboard");
        }
        assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));
        paint(window);
        let publish = |window: &mut gpui::VisualTestContext| {
            if keyboard {
                saved_graph_shortcut(window, "v");
            } else {
                click(window, action);
            }
        };
        publish(window);
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        publish(window);
        let destination = PathBuf::from("/project/graph-group-copy.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        assert_eq!(
            window
                .read_from_clipboard()
                .and_then(|item| item.text())
                .as_deref(),
            Some(clipboard.as_str())
        );
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the copied node group");
        assert_ne!(reopened, source);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the native Graphite folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents
            .graph_nodes()
            .await
            .expect("saved original and pasted nodes");
        assert_eq!(nodes.len(), 4);
        let consumer = nodes
            .iter()
            .find(|node| node.has_numeric_connection() && !original_labels.contains(node.label()))
            .expect("new downstream node");
        let producer = consumer
            .numeric_connection_source(&nodes)
            .expect("new linked source");
        assert!(!original_labels.contains(producer.label()));
        assert!(matches!(
            contents
                .preview_graph_numeric_output(consumer)
                .await
                .expect("execute pasted group"),
            GddNumericValue::Integer(17)
        ));
        assert!(contents.history_len() > original_history);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_021_2_workspace_adds_the_first_export_in_an_empty_network(
        cx: &mut TestAppContext,
    ) {
        verify_initial_network_port_actions(cx, &[(NetworkPortKind::Export, false)]).await;
    }

    #[gpui::test]
    async fn at_gra_graph_020_022_2_workspace_starts_secondary_interfaces_with_hidden_primaries(
        cx: &mut TestAppContext,
    ) {
        verify_initial_network_port_actions(
            cx,
            &[
                (NetworkPortKind::Import, true),
                (NetworkPortKind::Export, true),
            ],
        )
        .await;
    }

    async fn verify_initial_network_port_actions(
        cx: &mut TestAppContext,
        actions: &[(NetworkPortKind, bool)],
    ) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/empty-ports.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        let mut import_count = 0;
        let mut export_count = 0;
        let mut added_history = 0;
        for (step, &(kind, secondary)) in actions.iter().enumerate() {
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
            click(window, "GRAPHITE-GDD-ENTER-NETWORK");
            paint(window);
            click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
            paint(window);
            click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-7");
            paint(window);
            click(window, "GRAPHITE-GDD-ENTER-NETWORK");
            paint(window);
            let network_id = current.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected the scoped graph");
                };
                let network = form.active_network().expect("original scoped network");
                assert_eq!(kind.count(network), 0);
                assert_eq!(network.import_count, import_count);
                assert_eq!(network.export_count, export_count);
                network.id
            });
            let count = 1 + usize::from(secondary);
            let selector = match kind {
                NetworkPortKind::Import => {
                    import_count = count;
                    added_history += 1;
                    if secondary {
                        "GRAPHITE-GDD-ADD-SECONDARY-IMPORT"
                    } else {
                        "GRAPHITE-GDD-ADD-NETWORK-IMPORT"
                    }
                }
                NetworkPortKind::Export => {
                    export_count = count;
                    added_history += count + 1;
                    if secondary {
                        "GRAPHITE-GDD-ADD-SECONDARY-EXPORT"
                    } else {
                        "GRAPHITE-GDD-ADD-NETWORK-EXPORT"
                    }
                }
            };
            click(window, selector);
            window.simulate_new_path_selection(|_| None);
            window.run_until_parked();
            assert!(!current.read_with(window, |item, cx| item.is_dirty(cx)));
            paint(window);
            click(window, selector);
            let destination = PathBuf::from(format!("/project/initial-ports-{step}.gdd"));
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopen the edited folder");
            assert_ne!(reopened, current);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected native Graphite folder");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let graph = contents
                .graph_inspection()
                .await
                .expect("reopened nested network");
            let network = graph
                .networks
                .iter()
                .find(|network| network.id == network_id)
                .expect("same nested network identity");
            assert_eq!(kind.primary_exposure(network), Some(!secondary));
            assert_eq!(network.import_count, import_count);
            assert_eq!(network.export_count, export_count);
            assert_eq!(contents.history_len(), original_history + added_history);
            current = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_023_2_workspace_removes_the_addressed_nested_import(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_removes_nested_import(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_023_2_workspace_removes_a_nested_import_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_removes_nested_import(cx, true).await;
    }

    async fn verify_workspace_removes_nested_import(cx: &mut TestAppContext, canvas: bool) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/remove-import.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        let network_id = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected scoped graph");
            };
            let network = form.active_network().expect("source network");
            assert_eq!(network.import_count, 2);
            assert_eq!(network.export_count, 2);
            network.id
        });
        open_saved_boundary_menu(window, NetworkPortKind::Import, 1, canvas);
        click(window, "GRAPHITE-GDD-REMOVE-IMPORT-1");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));

        open_saved_boundary_menu(window, NetworkPortKind::Import, 0, canvas);
        click(window, "GRAPHITE-GDD-REMOVE-IMPORT-0");
        let path = PathBuf::from("/project/removed-primary-import.gdd");
        window.simulate_new_path_selection(|_| Some(path.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![path]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened edited folder");
        assert_ne!(reopened, source);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the new Graphite folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let graph = contents
            .graph_inspection()
            .await
            .expect("reopened nested network");
        let network = graph
            .networks
            .iter()
            .find(|network| network.id == network_id)
            .expect("same saved network");
        assert_eq!(network.import_count, 1);
        assert_eq!(network.export_count, 2);
        assert_eq!(network.import_name(0), Some("Second numeric input"));
        assert_eq!(contents.history_len(), original_history + 3);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_025_2_workspace_reorders_named_imports_in_a_folder_copy(
        cx: &mut TestAppContext,
    ) {
        verify_network_port_reorder(cx, NetworkPortKind::Import, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_026_2_workspace_reorders_named_exports_in_a_folder_copy(
        cx: &mut TestAppContext,
    ) {
        verify_network_port_reorder(cx, NetworkPortKind::Export, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_025_2_workspace_reorders_imports_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_network_port_reorder(cx, NetworkPortKind::Import, true).await;
    }

    #[gpui::test]
    async fn at_gra_graph_026_2_workspace_reorders_exports_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_network_port_reorder(cx, NetworkPortKind::Export, true).await;
    }

    async fn verify_network_port_reorder(
        cx: &mut TestAppContext,
        kind: NetworkPortKind,
        canvas: bool,
    ) {
        let (move_up, move_down, first, second, added_history) = match kind {
            NetworkPortKind::Import => (
                "GRAPHITE-GDD-MOVE-IMPORT-1-UP",
                "GRAPHITE-GDD-MOVE-IMPORT-0-DOWN",
                "First numeric input",
                "Second numeric input",
                3,
            ),
            NetworkPortKind::Export => (
                "GRAPHITE-GDD-MOVE-EXPORT-1-UP",
                "GRAPHITE-GDD-MOVE-EXPORT-0-DOWN",
                "First numeric output",
                "Second numeric output",
                4,
            ),
        };
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/reorder-ports.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        let (network_id, imports, exports) = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected scoped graph");
            };
            let network = form.active_network().expect("original nested network");
            assert_eq!(kind.name(network, 0), Some(first));
            assert_eq!(kind.name(network, 1), Some(second));
            let imports = form
                .nodes
                .iter()
                .filter(|node| node.network_id() == network.id)
                .filter_map(|node| Some((node.storage_id(), node.numeric_import_source()?)))
                .collect::<Vec<_>>();
            let exports = network
                .exports()
                .iter()
                .map(|port| port.source.clone())
                .collect::<Vec<_>>();
            (network.id, imports, exports)
        });
        open_saved_boundary_menu(window, kind, 1, canvas);
        click(window, move_up);
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));
        open_saved_boundary_menu(window, kind, 0, canvas);
        click(window, move_down);
        let destination = PathBuf::from("/project/moved-port.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the edited folder");
        assert_ne!(reopened, source);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected native Graphite folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let graph = contents
            .graph_inspection()
            .await
            .expect("reopened nested network");
        let network = graph
            .networks
            .iter()
            .find(|network| network.id == network_id)
            .expect("same nested network identity");
        assert_eq!(kind.name(network, 0), Some(second));
        assert_eq!(kind.name(network, 1), Some(first));
        assert_eq!(network.import_count, 2);
        assert_eq!(network.export_count, 2);
        for (stored_id, previous) in imports {
            let node = graph
                .nodes
                .iter()
                .find(|node| node.storage_id() == stored_id)
                .expect("preserved importing node");
            let expected = match kind {
                NetworkPortKind::Import => 1usize
                    .checked_sub(previous)
                    .expect("one of two fixture imports"),
                NetworkPortKind::Export => previous,
            };
            assert_eq!(node.numeric_import_source(), Some(expected));
        }
        let expected_exports = match kind {
            NetworkPortKind::Import => exports,
            NetworkPortKind::Export => exports.into_iter().rev().collect(),
        };
        assert_eq!(
            network
                .exports()
                .iter()
                .map(|port| port.source.clone())
                .collect::<Vec<_>>(),
            expected_exports
        );
        assert_eq!(contents.history_len(), original_history + added_history);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_029_2_workspace_hides_and_reexposes_the_primary_import(
        cx: &mut TestAppContext,
    ) {
        verify_primary_network_port_exposure(cx, NetworkPortKind::Import).await;
    }

    #[gpui::test]
    async fn at_gra_graph_030_2_workspace_hides_and_reexposes_the_primary_export(
        cx: &mut TestAppContext,
    ) {
        verify_primary_network_port_exposure(cx, NetworkPortKind::Export).await;
    }

    async fn verify_primary_network_port_exposure(cx: &mut TestAppContext, kind: NetworkPortKind) {
        let selector = match kind {
            NetworkPortKind::Import => "GRAPHITE-GDD-PRIMARY-IMPORT-EXPOSURE",
            NetworkPortKind::Export => "GRAPHITE-GDD-PRIMARY-EXPORT-EXPOSURE",
        };
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/primary-port.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        for (exposed, destination, added_history) in [
            (false, "/project/hidden-primary-port.gdd", 2),
            (true, "/project/exposed-primary-port.gdd", 3),
        ] {
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
            click(window, "GRAPHITE-GDD-ENTER-NETWORK");
            paint(window);
            let network_id = current.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected scoped graph");
                };
                let network = form.active_network().expect("original scoped network");
                assert_eq!(kind.primary_exposure(network), Some(!exposed));
                network.id
            });
            click(window, selector);
            window.simulate_new_path_selection(|_| None);
            window.run_until_parked();
            assert!(!current.read_with(window, |item, cx| item.is_dirty(cx)));

            paint(window);
            click(window, selector);
            let path = PathBuf::from(destination);
            window.simulate_new_path_selection(|_| Some(path.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![path]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopen the edited folder");
            assert_ne!(reopened, current);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected native Graphite folder");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let graph = contents
                .graph_inspection()
                .await
                .expect("reopened nested network");
            let network = graph
                .networks
                .iter()
                .find(|network| network.id == network_id)
                .expect("same nested network identity");
            assert_eq!(kind.primary_exposure(network), Some(exposed));
            assert_eq!(network.import_name(0), Some("First numeric input"));
            assert_eq!(network.import_name(1), Some("Second numeric input"));
            assert_eq!(network.export_name(0), Some("First numeric output"));
            assert_eq!(network.export_name(1), Some("Second numeric output"));
            assert_eq!(network.import_count, 2);
            assert_eq!(network.export_count, 2);
            assert_eq!(contents.history_len(), original_history + added_history);
            current = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_027_2_workspace_renames_addressed_import_and_export_names(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_renames_saved_ports(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_027_2_workspace_renames_nested_ports_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_renames_saved_ports(cx, true).await;
    }

    async fn verify_workspace_renames_saved_ports(cx: &mut TestAppContext, canvas: bool) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/rename-ports.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        for (kind, menu, selector, old_name, name, destination, added_history) in [
            (
                NetworkPortKind::Import,
                "GRAPHITE-GDD-REMOVE-IMPORT",
                "GRAPHITE-GDD-RENAME-IMPORT-1",
                "Second numeric input",
                "Adjustment",
                "/project/renamed-import.gdd",
                1,
            ),
            (
                NetworkPortKind::Export,
                "GRAPHITE-GDD-REMOVE-EXPORT",
                "GRAPHITE-GDD-RENAME-EXPORT-1",
                "Second numeric output",
                "Computed result",
                "/project/renamed-export.gdd",
                2,
            ),
        ] {
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
            click(window, "GRAPHITE-GDD-ENTER-NETWORK");
            paint(window);
            let network_id = current.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected scoped graph");
                };
                form.active_network().expect("same nested network").id
            });
            if canvas {
                click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
                window.run_until_parked();
                paint(window);
            }
            let rename = |window: &mut gpui::VisualTestContext| {
                if canvas {
                    let (socket, sibling) = match kind {
                        NetworkPortKind::Import => (
                            "GRAPHITE-GDD-GRAPH-Import-2",
                            "GRAPHITE-GDD-RENAME-IMPORT-0",
                        ),
                        NetworkPortKind::Export => (
                            "GRAPHITE-GDD-GRAPH-Export-2",
                            "GRAPHITE-GDD-RENAME-EXPORT-0",
                        ),
                    };
                    right_click_saved_graph(window, socket);
                    assert!(window.debug_bounds(sibling).is_none());
                } else {
                    click(window, menu);
                }
                click(window, selector);
            };
            rename(window);
            current.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected the name form");
                };
                let port = form.port_rename.as_ref().expect("selected rename");
                assert_eq!(port.name.read(cx).text(cx), old_name);
            });
            click(window, "GRAPHITE-GDD-CANCEL-PORT-NAME");
            paint(window);
            assert!(!current.read_with(window, |item, cx| item.is_dirty(cx)));

            rename(window);
            click(window, "GRAPHITE-GDD-PORT-NAME");
            window.simulate_keystrokes("ctrl-a");
            window.simulate_input(name);
            paint(window);
            click(window, "GRAPHITE-GDD-SAVE-PORT-NAME");
            let path = PathBuf::from(destination);
            window.simulate_new_path_selection(|_| Some(path.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![path]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopen the renamed folder");
            assert_ne!(reopened, current);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected native Graphite folder");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let graph = contents
                .graph_inspection()
                .await
                .expect("reopened nested network");
            let network = graph
                .networks
                .iter()
                .find(|network| network.id == network_id)
                .expect("same nested network identity");
            assert_eq!(kind.name(network, 1), Some(name));
            assert_eq!(network.import_name(0), Some("First numeric input"));
            assert_eq!(network.export_name(0), Some("First numeric output"));
            assert_eq!(network.import_count, 2);
            assert_eq!(network.export_count, 2);
            assert_eq!(contents.history_len(), original_history + added_history);
            current = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_024_2_workspace_removes_and_restores_a_nested_export(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_removes_nested_export(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_024_2_workspace_removes_and_restores_an_export_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_removes_nested_export(cx, true).await;
    }

    async fn verify_workspace_removes_nested_export(cx: &mut TestAppContext, canvas: bool) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/remove-export.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        let network_id = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the scoped graph");
            };
            let network = form.active_network().expect("original nested network");
            assert_eq!(network.export_count, 2);
            network.id
        });
        open_saved_boundary_menu(window, NetworkPortKind::Export, 1, canvas);
        click(window, "GRAPHITE-GDD-REMOVE-EXPORT-1");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(!source.read_with(window, |item, cx| item.is_dirty(cx)));

        let mut previous = source;
        for (selector, path, expected_exports) in [
            (
                "GRAPHITE-GDD-REMOVE-EXPORT-0",
                "/project/removed-export.gdd",
                1,
            ),
            ("GRAPHITE-GDD-SAVE-UNDO", "/project/restored-export.gdd", 2),
            ("GRAPHITE-GDD-SAVE-REDO", "/project/redone-export.gdd", 1),
        ] {
            paint(window);
            if selector == "GRAPHITE-GDD-REMOVE-EXPORT-0" {
                open_saved_boundary_menu(window, NetworkPortKind::Export, 0, canvas);
            }
            click(window, selector);
            let destination = PathBuf::from(path);
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopen the saved signature");
            assert_ne!(reopened, previous);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected native Graphite folder");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let graph = contents
                .graph_inspection()
                .await
                .expect("reopened nested scope");
            let network = graph
                .networks
                .iter()
                .find(|network| network.id == network_id)
                .expect("same nested network identity");
            assert_eq!(network.import_count, 2);
            assert_eq!(network.export_count, expected_exports);
            if expected_exports == 2 {
                assert_eq!(network.export_name(0), Some("First numeric output"));
                assert_eq!(network.export_name(1), Some("Second numeric output"));
            } else {
                assert_eq!(network.export_name(0), Some("Second numeric output"));
            }
            assert_eq!(contents.history_len(), original_history + 4);
            previous = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_007_022_2_workspace_wires_nested_ports_in_the_working_canvas(
        cx: &mut TestAppContext,
    ) {
        use graphics_core::formats::gdd::GddGraphPortSource;

        let (bytes, _) = graph_navigation_archive_with_number_target(true).await;
        let seed = open_gdd_archive(&bytes.into(), ARCHIVE_BUDGET)
            .await
            .expect("pinned source network");
        let graph = seed.graph_inspection().await.expect("pinned nested scopes");
        let scope = graph
            .networks
            .iter()
            .find(|network| network.parent.is_some() && network.export_count == 2)
            .expect("nested network with pinned ports");
        let original = seed
            .append_nested_network_export_in_copy(scope, ARCHIVE_BUDGET)
            .await
            .expect("source with an empty third export")
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source archive contents");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/live-nested-ports.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("write pinned source");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite Inspector");
        choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-6");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let canvas = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH")
            .expect("open nested canvas");
        click_graph_point(
            window,
            canvas.origin + gpui::point(px(650.), px(170.)),
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        click(window, "GRAPHITE-GDD-ADD-Sign");
        window.run_until_parked();
        paint(window);
        let (target_id, base_history) = item.read_with(window, |item, cx| {
            assert!(item.is_dirty(cx));
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working scoped document")
            };
            (
                form.selected_node()
                    .expect("retained numeric target")
                    .storage_id(),
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("working snapshot")
                    .history_len(),
            )
        });

        click(window, "GRAPHITE-GDD-GRAPH-Import-2");
        paint(window);
        click(window, "GRAPHITE-GDD-GRAPH-NODE-6");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working canvas")
            };
            assert_eq!(
                form.selected_node()
                    .expect("wired target")
                    .numeric_import_source(),
                Some(1)
            );
            assert!(matches!(
                form.output_preview,
                NodeOutputPreview::Ready(GddNumericValue::Number(23.))
            ));
        });

        click(window, "GRAPHITE-GDD-GRAPH-Export-3");
        window.run_until_parked();
        paint(window);
        let assert_exports = |window: &gpui::VisualTestContext, connected: bool| {
            item.read_with(window, |item, cx| {
                assert!(item.is_dirty(cx));
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("working canvas")
                };
                let network = form.active_network().expect("original scoped network");
                assert_eq!(network.id, scope.id);
                assert_eq!(network.export_count, 3);
                assert_eq!(network.empty_exports().is_empty(), connected);
                assert_eq!(
                    network
                        .exports()
                        .get(2)
                        .and_then(|port| port.source.clone()),
                    connected.then_some(GddGraphPortSource::Node {
                        storage_id: target_id,
                        output_index: 0
                    })
                );
                assert!(matches!(form.saved_graph, SavedGraphPanel::Ready(_)));
            });
        };
        assert_exports(window, true);
        click(window, "GRAPHITE-GDD-UNDO");
        window.run_until_parked();
        paint(window);
        assert_exports(window, false);
        click(window, "GRAPHITE-GDD-UNDO");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working canvas")
            };
            assert_eq!(
                form.selected_node()
                    .expect("target input")
                    .numeric_import_source(),
                None
            );
        });
        for _ in 0..2 {
            click(window, "GRAPHITE-GDD-REDO");
            window.run_until_parked();
            paint(window);
        }
        assert_exports(window, true);

        let destination = PathBuf::from("/project/saved-working-ports.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        item.read_with(window, |item, cx| assert!(!item.is_dirty(cx)));
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("read published working graph");
        let graph = saved
            .graph_inspection()
            .await
            .expect("reopened working graph");
        let network = graph
            .networks
            .iter()
            .find(|network| network.id == scope.id)
            .expect("same nested scope");
        let target = graph
            .nodes
            .iter()
            .find(|node| node.storage_id() == target_id)
            .expect("same target ID");
        assert_eq!(target.numeric_import_source(), Some(1));
        assert!(matches!(
            saved.preview_graph_numeric_output(target).await,
            Ok(GddNumericValue::Number(23.))
        ));
        assert_eq!(
            network
                .exports()
                .get(2)
                .and_then(|port| port.source.as_ref()),
            Some(&GddGraphPortSource::Node {
                storage_id: target_id,
                output_index: 0
            })
        );
        assert_eq!(saved.history_len(), base_history + 2);
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_022_2_workspace_appends_and_connects_a_nested_export_in_copies(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_adds_and_connects_export(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_022_2_workspace_connects_an_empty_export_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_adds_and_connects_export(cx, true).await;
    }

    async fn verify_workspace_adds_and_connects_export(cx: &mut TestAppContext, canvas: bool) {
        use graphics_core::formats::gdd::GddGraphPortSource;

        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/export-network.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source network")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        let original_network = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected scoped graph");
            };
            let network = form.active_network().expect("source nested network");
            assert_eq!(network.export_count, 2);
            network.id
        });
        click(window, "GRAPHITE-GDD-ADD-NETWORK-EXPORT");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-ADD-NETWORK-EXPORT");
        let destination = PathBuf::from("/project/added-export.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened export copy");
        assert_ne!(reopened, item);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the new Graphite folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let networks = contents
            .graph_inspection()
            .await
            .expect("saved network list")
            .networks;
        let network = networks
            .iter()
            .find(|network| network.id == original_network)
            .expect("same saved network");
        assert_eq!((network.import_count, network.export_count), (2, 3));
        assert_eq!(network.empty_exports(), &[2]);
        assert_eq!(contents.history_len(), original_history + 2);

        let command = if canvas {
            choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-4");
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-Export-3");
            reopened.read_with(window, |item, cx| {
                assert!(item.copy_task.is_none());
                assert!(!item.is_dirty(cx));
            });
            click(window, "GRAPHITE-GDD-GRAPH-NODE-6");
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-Export-1");
            reopened.read_with(window, |item, _| assert!(item.copy_task.is_none()));
            "GRAPHITE-GDD-GRAPH-Export-3"
        } else {
            choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-6");
            "GRAPHITE-GDD-CONNECT-EXPORT-2"
        };
        let source_id = reopened.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the scoped graph");
            };
            form.selected_node()
                .expect("selected numeric source")
                .storage_id()
        });
        click(window, command);
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        click(window, command);
        let destination = PathBuf::from("/project/wired-export.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let wired = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened wired copy");
        assert_ne!(wired, reopened);
        let wired_contents = wired.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the connected export folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let wired_networks = wired_contents
            .graph_inspection()
            .await
            .expect("saved connected exports")
            .networks;
        let wired_network = wired_networks
            .iter()
            .find(|network| network.id == original_network)
            .expect("same saved network");
        assert_eq!(wired_network.export_count, 3);
        assert!(wired_network.empty_exports().is_empty());
        assert_eq!(
            wired_network
                .exports()
                .get(2)
                .and_then(|port| port.source.as_ref()),
            Some(&GddGraphPortSource::Node {
                storage_id: source_id,
                output_index: 0
            }),
        );
        assert_eq!(wired_contents.history_len(), original_history + 3);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_r17_2_workspace_paints_saved_graph_wires_and_selects_their_scoped_node(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/saved-graph-canvas.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("original archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open saved graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        for selector in [
            "GRAPHITE-GDD-SAVED-GRAPH",
            "GRAPHITE-GDD-GRAPH-Import-1",
            "GRAPHITE-GDD-GRAPH-Export-2",
            "GRAPHITE-GDD-GRAPH-NODE-3",
        ] {
            assert!(
                window.debug_bounds(selector).is_some(),
                "missing {selector}"
            );
        }
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_none());
        click(window, "GRAPHITE-GDD-GRAPH-NODE-3");
        paint(window);
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the scoped graph");
            };
            let SavedGraphPanel::Ready(geometry) = &form.saved_graph else {
                panic!("expected the saved graph geometry");
            };
            let node = form.selected_node().expect("selected node from the canvas");
            assert_eq!(node.numeric_import_source(), Some(1));
            assert_eq!(
                node.network_id(),
                form.active_network().expect("nested scope").id
            );
            assert_eq!(geometry.nodes.len(), 6);
            assert_eq!(geometry.wires.len(), 5);
            assert!(!item.is_dirty(cx));
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected source history");
            };
            assert_eq!(contents.history_len(), original_history);
        });
        click(window, "GRAPHITE-GDD-PARENT-NETWORK");
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-SAVED-GRAPH").is_none());
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_layer_001_2_workspace_selects_saved_layers_and_graph_nodes_together(
        cx: &mut TestAppContext,
    ) {
        let (original, history) = graph_layer_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/saved-layers.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the layer document")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let revision = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready Inspector")
            };
            let SavedGraphPanel::Ready(graph) = &form.saved_graph else {
                panic!("ready saved graph")
            };
            let layers = graph.root_layers.as_ref().expect("pinned layer structure");
            assert_eq!(
                layers
                    .iter()
                    .map(|layer| {
                        let node = form
                            .nodes
                            .iter()
                            .find(|node| node.storage_id() == layer.storage_id)
                            .expect("same layer and graph identity");
                        (node.name(), layer.depth, layer.visible, layer.locked)
                    })
                    .collect::<Vec<_>>(),
                [
                    ("Artwork", 0, true, false),
                    ("Child", 1, false, true),
                    ("Background", 0, true, false)
                ]
            );
            item.model.read(cx).revision.clone()
        });
        let parent = window
            .debug_bounds("GRAPHITE-GDD-SAVED-LAYER-1")
            .expect("parent layer");
        let child = window
            .debug_bounds("GRAPHITE-GDD-SAVED-LAYER-2")
            .expect("child layer");
        let sibling = window
            .debug_bounds("GRAPHITE-GDD-SAVED-LAYER-0")
            .expect("sibling layer");
        assert!(parent.origin.y < child.origin.y && child.origin.y < sibling.origin.y);
        assert!(child.origin.x > parent.origin.x);
        assert!(window.debug_bounds("GRAPHITE-GDD-SAVED-LAYER-3").is_none());
        for (selector, label) in [
            ("GRAPHITE-GDD-SAVED-LAYER-2", "Child"),
            ("GRAPHITE-GDD-GRAPH-NODE-1", "Artwork"),
            ("GRAPHITE-GDD-SAVED-LAYER-0", "Background"),
        ] {
            click(window, selector);
            paint(window);
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Inspector")
                };
                assert_eq!(form.selected_node().map(GddGraphNode::name), Some(label));
                assert_eq!(form.name.read(cx).text(cx), label);
                assert!(!item.is_dirty(cx));
                let model = item.model.read(cx);
                assert!(Arc::ptr_eq(&model.revision, &revision));
                assert_eq!(model.gdd_contents().expect("GDD").history_len(), history);
            });
        }
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_r17_1_workspace_projects_scoped_ports_and_saved_wires(cx: &mut TestAppContext) {
        use graphics_core::formats::gdd::GddGraphPortSource;

        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/scoped-graph-ports.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("original GDD archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the scoped graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-3");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-PORTS");
        paint(window);
        for selector in [
            "GRAPHITE-GDD-SAVED-Import-1",
            "GRAPHITE-GDD-SAVED-Import-2",
            "GRAPHITE-GDD-SAVED-Export-1",
            "GRAPHITE-GDD-SAVED-Export-2",
            "GRAPHITE-GDD-SAVED-Input-1",
        ] {
            assert!(
                window.debug_bounds(selector).is_some(),
                "missing {selector}"
            );
        }
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the scoped graph");
            };
            let scope = form.active_network().expect("nested scope");
            let [primary, secondary] = scope.imports() else {
                panic!("expected primary and secondary saved imports");
            };
            assert_eq!(primary.name.as_deref(), Some("First numeric input"));
            assert_eq!(secondary.name.as_deref(), Some("Second numeric input"));
            assert_eq!(secondary.index, 1);
            assert!(primary.exposed && secondary.exposed);
            assert_eq!(
                secondary
                    .stored_type
                    .as_ref()
                    .map(|ty| ty.identifier_name())
                    .as_deref(),
                Some("f64")
            );
            let node = form.selected_node().expect("consumer of the second import");
            let [input] = node.input_ports() else {
                panic!("expected one node input");
            };
            assert_eq!(input.source, Some(GddGraphPortSource::Import(1)));
            let [_, export] = scope.exports() else {
                panic!("expected two saved exports");
            };
            assert_eq!(export.name.as_deref(), Some("Second numeric output"));
            assert_eq!(
                export.source,
                Some(GddGraphPortSource::Node {
                    storage_id: node.storage_id(),
                    output_index: 0,
                })
            );
        });
        click(window, "GRAPHITE-GDD-PARENT-NETWORK");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-SAVED-UNKNOWN-NODE")
                .is_some()
        );
        assert!(window.debug_bounds("GRAPHITE-GDD-SAVED-Import-2").is_none());
        item.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the source document");
            };
            assert_eq!(contents.history_len(), original_history);
        });
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_016_017_2_workspace_enters_exact_nested_network_and_returns_to_parent(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/networks.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open nested source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the graph picker");
            };
            let network = form.active_network().expect("addressed nested network");
            assert_eq!((network.import_count, network.export_count), (2, 2));
            assert!(form.selected_node().is_none());
            assert_eq!(
                form.nodes
                    .iter()
                    .filter(|node| form.includes_node(node))
                    .count(),
                6
            );
            assert_eq!(
                form.nodes
                    .iter()
                    .filter(|node| node.name() == "Repeated value" && form.includes_node(node))
                    .count(),
                1
            );
        });
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-NODE-COPY-ITEM-0")
                .is_none()
        );
        click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-7");
        paint(window);
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-NODE-COPY-CHOOSE")
                .is_none()
        );
        click(window, "GRAPHITE-GDD-PARENT-NETWORK");
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-NODE-COPY-CHOOSE")
                .is_some()
        );
        click(window, "GRAPHITE-GDD-PARENT-NETWORK");
        paint(window);
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the root graph picker");
            };
            let root = form.active_network().expect("root scope");
            assert_eq!(root.parent, None);
            assert_eq!(
                form.nodes
                    .iter()
                    .filter(|node| form.includes_node(node))
                    .count(),
                2
            );
            assert!(!item.is_dirty(cx));
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected unchanged source document");
            };
            assert_eq!(contents.history_len(), original_history);
        });
        assert!(window.debug_bounds("GRAPHITE-GDD-PARENT-NETWORK").is_none());
        click(window, "GRAPHITE-GDD-ALL-GRAPH-NODES");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-NODE-COPY-ITEM-0")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-NODE-COPY-ITEM-7")
                .is_some()
        );
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_005_2_workspace_inserts_the_selected_catalog_template(
        cx: &mut TestAppContext,
    ) {
        let blank = create_gdd_folder(621, 187, "Made".into(), ARCHIVE_BUDGET)
            .await
            .expect("source folder");
        let original = blank
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/catalog-insert.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source document")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-SEARCH");
        window.simulate_input("type: Number math Add");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-POSITION");
        window.simulate_keystrokes("ctrl-a");
        window.simulate_input("bad");
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-ADD-Add");
        item.read_with(window, |item, _| {
            assert!(item.copy_task.is_none());
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error))
                if message.contains("two whole grid coordinates")));
        });
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-POSITION");
        window.simulate_keystrokes("ctrl-a");
        window.simulate_input("6, -9");
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-ADD-Add");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-ADD-Add");
        let destination = PathBuf::from("/project/catalog-node.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened catalog copy");
        assert_ne!(reopened, item);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the new GDD folder");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents.graph_nodes().await.expect("saved catalog nodes");
        let [added] = nodes.as_slice() else {
            panic!("expected one added node");
        };
        assert!(added.label().contains("Add"));
        assert_eq!(added.grid_position(), Some([6, -9]));
        let saved_history = contents.history_len();
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        window.run_until_parked();
        reopened.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the selected source Add node");
            };
            assert!(matches!(&form.output_preview,
                NodeOutputPreview::Ready(GddNumericValue::NumberList(values)) if values.is_empty()));
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(contents.history_len(), saved_history);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_001_002_003_004_2_workspace_browses_and_searches_pinned_node_categories(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/catalog.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source document")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG");
        window.run_until_parked();
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-NODE-CATALOG-NODE-Add")
                .is_none()
        );
        click(window, "GRAPHITE-NODE-CATALOG-CATEGORY-Math: Arithmetic");
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-NODE-CATALOG-NODE-Add")
                .is_some()
        );

        click(window, "GRAPHITE-NODE-CATALOG-SEARCH");
        for (query, expected) in [
            (
                "aBsOlUtE VaLuE",
                "GRAPHITE-NODE-CATALOG-NODE-Absolute Value",
            ),
            (
                "TYPE: number math subtract",
                "GRAPHITE-NODE-CATALOG-NODE-Subtract",
            ),
            ("lAyEr", "GRAPHITE-NODE-CATALOG-NODE-Merge"),
            (
                "zqx nowhere in the source catalog",
                "GRAPHITE-NODE-CATALOG-EMPTY",
            ),
        ] {
            window.simulate_keystrokes("ctrl-a");
            window.simulate_input(query);
            window.run_until_parked();
            paint(window);
            assert!(window.debug_bounds(expected).is_some(), "query {query}");
        }
        click(window, "GRAPHITE-NODE-CATALOG");
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-NODE-CATALOG-SEARCH")
                .is_none()
        );
        item.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            assert!(item.copy_task.is_none());
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the unchanged GDD document");
            };
            assert_eq!(contents.history_len(), original_history);
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_035_2_workspace_previews_a_wired_numeric_output_without_dirtying(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::Connection).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/preview.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the numeric graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        window.run_until_parked();
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the selected numeric node");
            };
            assert!(matches!(
                form.output_preview,
                NodeOutputPreview::Ready(GddNumericValue::Integer(17))
            ));
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the unchanged GDD document");
            };
            assert_eq!(contents.history_len(), original_history);
            assert!(item.copy_task.is_none());
            assert!(!item.is_dirty(cx));
        });
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-NUMERIC-RESULT").is_some());
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-NUMERIC-RESULT").is_none());

        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-NUMERIC-RESULT").is_none());
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("byte-exact source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_014_2_workspace_drags_a_node_in_the_working_canvas(
        cx: &mut TestAppContext,
    ) {
        let (original, _) =
            numeric_graph_archive_with_layout(NumericGraphFixture::Connection, true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/working-drag.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the wired Graphite archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let (field, source_id, target_id) = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("loaded source graph")
            };
            let [source, target] = form.nodes.as_slice() else {
                panic!("two wired nodes")
            };
            assert_eq!(source.grid_position(), Some([0, 0]));
            (form.value.clone(), source.storage_id(), target.storage_id())
        });
        field.update_in(window, |editor, window, cx| {
            editor.set_text("-19", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let before = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());

        let drag = |window: &mut gpui::VisualTestContext, end: gpui::Point<gpui::Pixels>| {
            let start = window
                .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0")
                .expect("visible source node")
                .center();
            drag_graph_point(window, start, end, |window| {
                item.read_with(window, |item, cx| {
                    assert!(Arc::ptr_eq(&item.model.read(cx).revision, &before))
                });
            });
        };
        let canvas = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH")
            .expect("open canvas");
        let start = window
            .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0")
            .expect("source node")
            .center();
        for cancelled in [
            canvas.origin + gpui::point(canvas.size.width + px(30.), px(40.)),
            start + gpui::point(px(8.), px(0.)),
        ] {
            drag(window, cancelled);
            item.read_with(window, |item, cx| {
                assert!(Arc::ptr_eq(&item.model.read(cx).revision, &before))
            });
        }
        drag(window, start + gpui::point(px(72.), px(48.)));
        let check = |window: &gpui::VisualTestContext, position| {
            item.read_with(window, |item, cx| {
                assert!(item.is_dirty(cx));
                let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("working graph") };
                let source = form.nodes.iter().find(|node| node.storage_id() == source_id).expect("same moved node");
                let target = form.nodes.iter().find(|node| node.storage_id() == target_id).expect("same connected target");
                assert_eq!(source.grid_position(), Some(position));
                assert_eq!(target.numeric_connection_source(&form.nodes).map(GddGraphNode::storage_id), Some(source_id));
                assert!(matches!(&form.saved_graph, SavedGraphPanel::Ready(graph) if graph.wires.len() == 1));
            });
        };
        check(window, [3, 2]);
        for (action, position) in [("GRAPHITE-GDD-UNDO", [0, 0]), ("GRAPHITE-GDD-REDO", [3, 2])] {
            click(window, action);
            window.run_until_parked();
            paint(window);
            check(window, position);
        }

        let destination = PathBuf::from("/project/saved-working-drag.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("published GDD folder");
        let nodes = saved.graph_nodes().await.expect("reopened graph");
        let source = nodes
            .iter()
            .find(|node| node.storage_id() == source_id)
            .expect("moved source");
        let target = nodes
            .iter()
            .find(|node| node.storage_id() == target_id)
            .expect("connected target");
        assert_eq!(source.grid_position(), Some([3, 2]));
        assert_eq!(
            target
                .numeric_connection_source(&nodes)
                .map(GddGraphNode::storage_id),
            Some(source_id)
        );
        assert!(matches!(
            saved.preview_graph_numeric_output(target).await,
            Ok(GddNumericValue::Integer(19))
        ));
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_014_2_workspace_drags_selected_nodes_as_one_step(
        cx: &mut TestAppContext,
    ) {
        let (original, history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::Reconnection, true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/selected-working-drag.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("root source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        window.run_until_parked();
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-GRAPH-NODE-0");
        paint(window);
        click_saved_graph(
            window,
            "GRAPHITE-GDD-GRAPH-NODE-2",
            gpui::MouseButton::Left,
            gpui::Modifiers {
                shift: true,
                ..Default::default()
            },
        );
        let (source_id, target_id, alternate_id, revision) = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working graph")
            };
            assert_eq!(form.node_selection.len(), 2);
            assert!(!item.is_dirty(cx));
            let [source, target, alternate] = form.nodes.as_slice() else {
                panic!("two source nodes and a connected target")
            };
            assert!(form.pending_canvas_source.is_none());
            assert_eq!(source.grid_position(), Some([0, 0]));
            assert_eq!(target.grid_position(), None);
            assert_eq!(alternate.grid_position(), Some([0, 4]));
            (
                source.storage_id(),
                target.storage_id(),
                alternate.storage_id(),
                form.revision.clone(),
            )
        });
        let start = window
            .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0")
            .expect("dragging the first selected node")
            .center();
        drag_graph_point(
            window,
            start,
            start + gpui::point(px(72.), px(48.)),
            |window| {
                item.read_with(window, |item, cx| {
                    assert!(!item.is_dirty(cx));
                    assert!(Arc::ptr_eq(&item.model.read(cx).revision, &revision));
                });
            },
        );
        let check_group = |window: &gpui::VisualTestContext,
                           source_position,
                           alternate_position| {
            item.read_with(window, |item, cx| {
                assert!(item.is_dirty(cx));
                let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("working graph") };
                assert_eq!(form.node_selection.len(), 2);
                let [source, target, alternate] = form.nodes.as_slice() else { panic!("same node group") };
                assert_eq!(source.storage_id(), source_id);
                assert_eq!(target.storage_id(), target_id);
                assert_eq!(alternate.storage_id(), alternate_id);
                assert_eq!(source.grid_position(), Some(source_position));
                assert_eq!(target.grid_position(), None);
                assert_eq!(alternate.grid_position(), Some(alternate_position));
                assert_eq!(target.numeric_connection_source(&form.nodes).map(GddGraphNode::storage_id), Some(source_id));
                assert!(matches!(&form.saved_graph, SavedGraphPanel::Ready(graph) if graph.wires.len() == 1));
            });
        };
        check_group(window, [3, 2], [3, 6]);
        for (action, source_position, alternate_position) in [
            ("GRAPHITE-GDD-UNDO", [0, 0], [0, 4]),
            ("GRAPHITE-GDD-REDO", [3, 2], [3, 6]),
        ] {
            click(window, action);
            window.run_until_parked();
            paint(window);
            check_group(window, source_position, alternate_position);
        }
        let destination = PathBuf::from("/project/saved-working-group-drag.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen the moved group");
        let nodes = saved.graph_nodes().await.expect("reopened scope");
        let [source, target, alternate] = nodes.as_slice() else {
            panic!("saved node group")
        };
        assert_eq!(source.storage_id(), source_id);
        assert_eq!(target.storage_id(), target_id);
        assert_eq!(alternate.storage_id(), alternate_id);
        assert_eq!(source.grid_position(), Some([3, 2]));
        assert_eq!(target.grid_position(), None);
        assert_eq!(alternate.grid_position(), Some([3, 6]));
        assert_eq!(saved.history_len(), history + 2);
        assert!(matches!(
            saved.preview_graph_numeric_output(target).await,
            Ok(GddNumericValue::Integer(17))
        ));
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_015_2_workspace_selects_nodes_with_a_canvas_box(cx: &mut TestAppContext) {
        let (original, history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::Reconnection, true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/canvas-box.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let revision = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        let source = window
            .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0")
            .expect("first node");
        let target = window
            .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-1")
            .expect("connected node");
        let alternate = window
            .debug_bounds("GRAPHITE-GDD-GRAPH-NODE-2")
            .expect("third node");
        let margin = gpui::point(px(12.), px(8.));
        let check = |window: &gpui::VisualTestContext, indexes: &[usize]| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready graph")
                };
                assert!(
                    form.node_selection
                        .iter()
                        .copied()
                        .eq(indexes.iter().copied())
                );
                assert_eq!(form.selected, indexes.first().copied());
                assert!(!item.is_dirty(cx));
                let model = item.model.read(cx);
                assert!(Arc::ptr_eq(&model.revision, &revision));
                let InspectedDocument::Gdd { contents } = &model.document else {
                    panic!("GDD")
                };
                assert_eq!(contents.history_len(), history);
            });
        };
        let drag = |window: &mut gpui::VisualTestContext, start, end, modifiers, cancel| {
            window.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                        position: start,
                        button: gpui::MouseButton::Left,
                        modifiers,
                        click_count: 1,
                        first_mouse: false,
                    }),
                    cx,
                );
                window.dispatch_event(
                    gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                        position: end,
                        pressed_button: Some(gpui::MouseButton::Left),
                        modifiers,
                    }),
                    cx,
                );
            });
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-GDD-BOX-SELECTION").is_some());
            if cancel {
                window.simulate_keystrokes("escape");
                paint(window);
                assert!(window.debug_bounds("GRAPHITE-GDD-BOX-SELECTION").is_none());
            }
            window.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                        position: end,
                        button: gpui::MouseButton::Left,
                        modifiers,
                        click_count: 1,
                    }),
                    cx,
                );
            });
            paint(window);
            assert!(window.debug_bounds("GRAPHITE-GDD-BOX-SELECTION").is_none());
        };
        drag(
            window,
            alternate.bottom_right() + margin,
            source.origin - margin,
            Default::default(),
            false,
        );
        check(window, &[0, 2]);
        let target_start = target.origin - margin;
        let target_end = target.bottom_right() + margin;
        drag(
            window,
            target_start,
            target_end,
            gpui::Modifiers {
                shift: true,
                ..Default::default()
            },
            false,
        );
        check(window, &[0, 1, 2]);
        drag(
            window,
            target_start,
            target_end,
            gpui::Modifiers {
                alt: true,
                ..Default::default()
            },
            false,
        );
        check(window, &[0, 2]);
        drag(window, target_start, target_end, Default::default(), true);
        check(window, &[0, 2]);
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_014_2_workspace_moves_a_node_to_an_exact_saved_grid_position(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/position.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source node")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let editor = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected a selected node");
            };
            assert_eq!(form.position.read(cx).text(cx), "0, 0");
            form.position.clone()
        });
        for (draft, feedback) in [
            ("0, 0", "different Graphite grid position"),
            ("2.5, -4", "two whole grid coordinates"),
        ] {
            editor.update_in(window, |editor, window, cx| {
                editor.set_text(draft, window, cx)
            });
            paint(window);
            click(window, "GRAPHITE-GDD-SAVE-POSITION");
            item.read_with(window, |item, cx| {
                assert!(item.copy_task.is_none());
                assert!(!item.is_dirty(cx));
                assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains(feedback)));
            });
        }
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("-12, 34", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-POSITION");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-POSITION");
        let destination = PathBuf::from("/project/moved.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened moved folder");
        assert_ne!(reopened, item);
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        reopened.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the moved node");
            };
            assert_eq!(form.position.read(cx).text(cx), "-12, 34");
            assert_eq!(form.value.read(cx).text(cx), "-17");
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_012_2_workspace_duplicates_the_selected_numeric_node_in_a_copy(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = numeric_graph_archive(NumericGraphFixture::Connection).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/duplicate-source.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source nodes")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-DUPLICATE-NODE");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        item.read_with(window, |item, cx| {
            assert!(item.copy_task.is_none());
            assert!(!item.is_dirty(cx));
        });
        paint(window);
        click(window, "GRAPHITE-GDD-DUPLICATE-NODE");
        let destination = PathBuf::from("/project/duplicated.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened copy");
        assert_ne!(reopened, item);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected GDD copy");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents.graph_nodes().await.expect("reopened duplicate");
        assert_eq!(nodes.len(), 3);
        let wired = nodes
            .iter()
            .find(|node| node.has_numeric_connection())
            .expect("original wire");
        let upstream = wired
            .numeric_connection_source(&nodes)
            .expect("original upstream node");
        assert!(matches!(
            upstream.numeric_input(),
            Some(GddNumericValue::Integer(-17))
        ));
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.numeric_input_exposed() == Some(true))
                .count(),
            1
        );
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_013_2_workspace_removes_the_last_isolated_node_and_undoes_in_a_copy(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/isolated.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the isolated node")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-DELETE-NODE");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        current.read_with(window, |item, cx| {
            assert!(item.copy_task.is_none());
            assert!(!item.is_dirty(cx));
        });
        for (selector, destination, restores_node) in [
            ("GRAPHITE-GDD-DELETE-NODE", "/project/removed.gdd", false),
            ("GRAPHITE-GDD-SAVE-UNDO", "/project/restored.gdd", true),
        ] {
            paint(window);
            click(window, selector);
            let destination = PathBuf::from(destination);
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopened folder");
            assert_ne!(reopened, current);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected GDD folder");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            assert_eq!(contents.history_len(), original_history + 2);
            let nodes = contents.graph_nodes().await.expect("read saved node set");
            if restores_node {
                let [node] = nodes.as_slice() else {
                    panic!("expected the original restored node");
                };
                assert!(matches!(
                    node.numeric_input(),
                    Some(GddNumericValue::Integer(-17))
                ));
            } else {
                assert!(nodes.is_empty());
            }
            current = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_007_2_workspace_connects_an_exposed_value_in_a_new_folder(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_connects_numeric_nodes(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_007_2_workspace_connects_numeric_nodes_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_connects_numeric_nodes(cx, true).await;
    }

    async fn verify_workspace_connects_numeric_nodes(cx: &mut TestAppContext, canvas: bool) {
        let (original, original_history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::ExposedInput, canvas).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/connect.gdd.zip");
        let destination = PathBuf::from("/project/connected.gdd");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let original_item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        if canvas {
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-OUTPUT-0");
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-0");
            original_item.read_with(window, |item, cx| {
                assert!(item.copy_task.is_none());
                assert!(!item.is_dirty(cx));
                assert!(matches!(&item.copy_feedback, Some((message, Color::Error))
                    if message.as_ref() == "Choose an exposed numeric input that accepts this output."));
            });
        }
        for publish in [false, true] {
            if canvas {
                click(window, "GRAPHITE-GDD-GRAPH-NODE-OUTPUT-0");
                paint(window);
                click(window, "GRAPHITE-GDD-GRAPH-NODE-1");
            } else {
                click(window, "GRAPHITE-GDD-CONNECT-VALUE");
                assert!(
                    window
                        .debug_bounds("GRAPHITE-GDD-CONNECT-SOURCE-1")
                        .is_none()
                );
                click(window, "GRAPHITE-GDD-CONNECT-SOURCE-0");
            }
            window.simulate_new_path_selection(|_| publish.then(|| destination.clone()));
            window.run_until_parked();
            assert_eq!(
                filesystem
                    .metadata(&destination)
                    .await
                    .expect("copy publication")
                    .is_some(),
                publish
            );
            paint(window);
        }
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("new document");
        assert_ne!(reopened, original_item);
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        assert!(window.debug_bounds("GRAPHITE-GDD-CONNECT-VALUE").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-DISCONNECT-VALUE")
                .is_some()
        );
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected GDD document");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents.graph_nodes().await.expect("original nodes");
        let [source, connected] = nodes.as_slice() else {
            panic!("expected the original two nodes");
        };
        assert!(matches!(
            source.numeric_input(),
            Some(GddNumericValue::Integer(-17))
        ));
        assert_eq!(
            connected
                .numeric_connection_source(&nodes)
                .map(GddGraphNode::storage_id),
            Some(source.storage_id())
        );
        assert!(matches!(
            contents.preview_graph_numeric_output(connected).await,
            Ok(GddNumericValue::Integer(17))
        ));
        assert_eq!(contents.history_len(), original_history + 1);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_007_008_009_2_workspace_edits_numeric_wires_in_the_working_canvas(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = numeric_graph_archive_with_layout(
            NumericGraphFixture::ExposedInputWithAlternative,
            true,
        )
        .await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/working-wires.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open wired source")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let (field, before_ids) = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("graph properties")
            };
            let [first, target, second] = form.nodes.as_slice() else {
                panic!("one target and two sources")
            };
            (
                form.value.clone(),
                [first.storage_id(), target.storage_id(), second.storage_id()],
            )
        });
        field.update_in(window, |editor, window, cx| {
            editor.set_text("-19", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let check_wire = |window: &mut gpui::VisualTestContext,
                          source: Option<usize>,
                          value: GddNumericValue| {
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, cx| {
                assert!(item.is_dirty(cx));
                let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("working graph") };
                let selected = form.selected_node().expect("retained target selection");
                assert_eq!(selected.storage_id(), before_ids[1]);
                assert_eq!(selected.numeric_connection_source(&form.nodes).map(GddGraphNode::storage_id), source.map(|index| before_ids[index]));
                assert!(matches!(&form.saved_graph, SavedGraphPanel::Ready(geometry) if geometry.wires.len() == usize::from(source.is_some())));
                match (&form.output_preview, value) {
                    (NodeOutputPreview::Ready(GddNumericValue::Integer(result)), GddNumericValue::Integer(expected)) => assert_eq!(*result, expected),
                    (NodeOutputPreview::Ready(GddNumericValue::Vector(result)), GddNumericValue::Vector(expected)) => assert_eq!(*result, expected),
                    _ => panic!("expected the pinned typed output"),
                }
            });
        };
        for (socket, index, expected) in [
            ("GRAPHITE-GDD-GRAPH-NODE-OUTPUT-2", 2, 31),
            ("GRAPHITE-GDD-GRAPH-NODE-OUTPUT-0", 0, 19),
        ] {
            click(window, socket);
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-1");
            check_wire(window, Some(index), GddNumericValue::Integer(expected));
        }
        click(window, "GRAPHITE-GDD-GRAPH-NODE-INPUT-1");
        check_wire(window, None, GddNumericValue::Vector([0., 0.]));

        for (action, source, expected) in [
            ("GRAPHITE-GDD-UNDO", Some(0), GddNumericValue::Integer(19)),
            ("GRAPHITE-GDD-UNDO", Some(2), GddNumericValue::Integer(31)),
            ("GRAPHITE-GDD-UNDO", None, GddNumericValue::Integer(0)),
            ("GRAPHITE-GDD-REDO", Some(2), GddNumericValue::Integer(31)),
            ("GRAPHITE-GDD-REDO", Some(0), GddNumericValue::Integer(19)),
            ("GRAPHITE-GDD-REDO", None, GddNumericValue::Vector([0., 0.])),
            ("GRAPHITE-GDD-UNDO", Some(0), GddNumericValue::Integer(19)),
        ] {
            click(window, action);
            check_wire(window, source, expected);
        }
        let destination = PathBuf::from("/project/saved-working-wires.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        item.read_with(window, |item, cx| assert!(!item.is_dirty(cx)));
        let contents = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("saved working folder");
        let nodes = contents.graph_nodes().await.expect("reopened graph");
        let [first, target, second] = nodes.as_slice() else {
            panic!("same three saved nodes")
        };
        assert_eq!(
            [first.storage_id(), target.storage_id(), second.storage_id()],
            before_ids
        );
        assert_eq!(
            target
                .numeric_connection_source(&nodes)
                .map(GddGraphNode::storage_id),
            Some(first.storage_id())
        );
        assert!(matches!(
            contents.preview_graph_numeric_output(target).await,
            Ok(GddNumericValue::Integer(19))
        ));
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_008_2_workspace_reconnects_a_value_in_a_new_folder(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_reconnects_numeric_nodes(cx, false).await;
    }

    #[gpui::test]
    async fn at_gra_graph_008_2_workspace_reconnects_numeric_nodes_from_the_canvas(
        cx: &mut TestAppContext,
    ) {
        verify_workspace_reconnects_numeric_nodes(cx, true).await;
    }

    async fn verify_workspace_reconnects_numeric_nodes(cx: &mut TestAppContext, canvas: bool) {
        let (original, original_history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::Reconnection, canvas).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/old-wire.gdd.zip");
        let destination = PathBuf::from("/project/reconnected.gdd");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let original_item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        if canvas {
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-OUTPUT-0");
            paint(window);
            click(window, "GRAPHITE-GDD-GRAPH-NODE-1");
            original_item.read_with(window, |item, cx| {
                assert!(item.copy_task.is_none());
                assert!(!item.is_dirty(cx));
            });
        }
        for publish in [false, true] {
            if canvas {
                click(window, "GRAPHITE-GDD-GRAPH-NODE-OUTPUT-2");
                paint(window);
                click(window, "GRAPHITE-GDD-GRAPH-NODE-1");
            } else {
                click(window, "GRAPHITE-GDD-RECONNECT-VALUE");
                assert!(
                    window
                        .debug_bounds("GRAPHITE-GDD-CONNECT-SOURCE-0")
                        .is_none()
                );
                assert!(
                    window
                        .debug_bounds("GRAPHITE-GDD-CONNECT-SOURCE-1")
                        .is_none()
                );
                click(window, "GRAPHITE-GDD-CONNECT-SOURCE-2");
            }
            window.simulate_new_path_selection(|_| publish.then(|| destination.clone()));
            window.run_until_parked();
            assert_eq!(
                filesystem
                    .metadata(&destination)
                    .await
                    .expect("copy publication")
                    .is_some(),
                publish
            );
            paint(window);
        }
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("new document");
        assert_ne!(reopened, original_item);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected GDD document");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents.graph_nodes().await.expect("original nodes");
        let [previous, target, replacement] = nodes.as_slice() else {
            panic!("expected the original three nodes");
        };
        assert!(matches!(
            previous.numeric_input(),
            Some(GddNumericValue::Integer(-17))
        ));
        assert!(matches!(
            replacement.numeric_input(),
            Some(GddNumericValue::Integer(-31))
        ));
        let current_source = target
            .numeric_connection_source(&nodes)
            .expect("replacement wire");
        assert_eq!(current_source.storage_id(), replacement.storage_id());
        assert!(matches!(
            contents.preview_graph_numeric_output(target).await,
            Ok(GddNumericValue::Integer(31))
        ));
        assert_eq!(contents.history_len(), original_history + 2);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_009_2_workspace_disconnects_a_value_and_reopens_its_fallback(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::Connection).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/wired.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");

        for (selector, destination, draft) in [
            (
                "GRAPHITE-GDD-DISCONNECT-VALUE",
                "/project/disconnected.gdd",
                None,
            ),
            (
                "GRAPHITE-GDD-SAVE-VALUE-EXPOSURE",
                "/project/fallback.gdd",
                None,
            ),
            (
                "GRAPHITE-GDD-SAVE-NUMERIC-INPUT",
                "/project/edited-fallback.gdd",
                Some("3.5, -2"),
            ),
        ] {
            let destination = PathBuf::from(destination);
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
            if let Some(draft) = draft {
                let field = current.read_with(window, |item, cx| {
                    let NodeCopyPanel::Ready(form) = &item.node_copy else {
                        panic!("expected the source fallback Value form");
                    };
                    assert_eq!(form.value.read(cx).text(cx), "0, 0");
                    form.value.clone()
                });
                field.update_in(window, |editor, window, cx| {
                    editor.set_text("3.5,", window, cx)
                });
                click(window, selector);
                current.read_with(window, |item, _| {
                    assert!(item.copy_task.is_none());
                    assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("two finite numbers")));
                });
                field.update_in(window, |editor, window, cx| {
                    editor.set_text(draft, window, cx)
                });
                paint(window);
            }
            click(window, selector);
            window.simulate_new_path_selection(|_| None);
            window.run_until_parked();
            assert!(
                filesystem
                    .metadata(&destination)
                    .await
                    .expect("cancelled save")
                    .is_none()
            );
            paint(window);
            click(window, selector);
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("new folder");
            assert_ne!(reopened, current);
            current = reopened;
        }

        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        let contents = current.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the fallback Value form");
            };
            assert_eq!(form.value.read(cx).text(cx), "3.5, -2");
            assert!(!item.is_dirty(cx));
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected GDD document");
            };
            contents.clone()
        });
        assert_eq!(contents.history_len(), original_history + 3);
        let nodes = contents.graph_nodes().await.expect("original nodes");
        let [upstream, disconnected] = nodes.as_slice() else {
            panic!("expected both original nodes");
        };
        assert!(matches!(
            upstream.numeric_input(),
            Some(GddNumericValue::Integer(-17))
        ));
        assert!(matches!(
            disconnected.numeric_input(),
            Some(GddNumericValue::Vector([3.5, -2.]))
        ));
        assert!(!disconnected.has_numeric_connection());
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_028_2_workspace_round_trips_the_exposed_value_port(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/ports.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        for (path, was_exposed, added_history) in [
            ("/project/exposed.gdd", false, 1),
            ("/project/hidden.gdd", true, 2),
        ] {
            let destination = PathBuf::from(path);
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
            current.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("expected the Value port form");
                };
                let selected = form.selected_node().expect("selected Value port");
                assert_eq!(selected.numeric_input_exposed(), Some(was_exposed));
                if !was_exposed {
                    assert_eq!(form.value.read(cx).text(cx), "-17");
                }
            });
            assert_eq!(
                window
                    .debug_bounds("GRAPHITE-GDD-SAVE-NUMERIC-INPUT")
                    .is_some(),
                !was_exposed,
            );
            click(window, "GRAPHITE-GDD-SAVE-VALUE-EXPOSURE");
            window.simulate_new_path_selection(|_| None);
            window.run_until_parked();
            assert!(
                filesystem
                    .metadata(&destination)
                    .await
                    .expect("cancelled save")
                    .is_none()
            );
            paint(window);
            click(window, "GRAPHITE-GDD-SAVE-VALUE-EXPOSURE");
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("open new Value port state");
            assert_ne!(reopened, current);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected GDD document");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let nodes = contents.graph_nodes().await.expect("reopened Value port");
            let [node] = nodes.as_slice() else {
                panic!("expected the original node");
            };
            assert_eq!(node.numeric_input_exposed(), Some(!was_exposed));
            if was_exposed {
                assert!(matches!(
                    node.numeric_input(),
                    Some(GddNumericValue::Integer(-17))
                ));
            }
            assert_eq!(contents.history_len(), original_history + added_history);
            current = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_031_2_workspace_edits_a_new_nodes_number_list_default(
        cx: &mut TestAppContext,
    ) {
        let blank = create_gdd_folder(451, 184, "Made".into(), ARCHIVE_BUDGET)
            .await
            .expect("source GDD folder");
        let inserted = blank
            .insert_root_numeric_node_in_copy(UnaryNumericOperation::Round, [3, 5], ARCHIVE_BUDGET)
            .await
            .expect("source Round defaults");
        let original = inserted
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/round.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source file");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the inserted Round node")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        for (selector, destination, draft) in [
            (
                "GRAPHITE-GDD-SAVE-VALUE-EXPOSURE",
                "/project/list-input.gdd",
                None,
            ),
            (
                "GRAPHITE-GDD-SAVE-NUMERIC-INPUT",
                "/project/edited-list.gdd",
                Some("-2.5, -0, -2.5"),
            ),
        ] {
            let destination = PathBuf::from(destination);
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
            if let Some(draft) = draft {
                let editor = current.read_with(window, |item, cx| {
                    let NodeCopyPanel::Ready(form) = &item.node_copy else {
                        panic!("expected the list editor");
                    };
                    assert_eq!(form.value.read(cx).text(cx), "");
                    form.value.clone()
                });
                editor.update_in(window, |editor, window, cx| {
                    editor.set_text("1,,2", window, cx)
                });
                click(window, selector);
                current.read_with(window, |item, _| {
                    assert!(item.copy_task.is_none());
                    assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("finite numbers separated by commas")));
                });
                editor.update_in(window, |editor, window, cx| {
                    editor.set_text(draft, window, cx)
                });
                paint(window);
            }
            click(window, selector);
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopened list copy");
            assert_ne!(reopened, current);
            current = reopened;
        }
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        current.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the reopened list editor");
            };
            assert_eq!(form.value.read(cx).text(cx), "-2.5, -0, -2.5");
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_031_2_workspace_saves_and_reopens_a_typed_numeric_value(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/value.gdd.zip");
        let destination = PathBuf::from("/project/edited.gdd");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source file");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let field = source.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the node form");
            };
            assert_eq!(form.value.read(cx).text(cx), "-17");
            form.value.clone()
        });
        field.update_in(window, |editor, window, cx| {
            editor.set_text("3.5", window, cx)
        });
        click(window, "GRAPHITE-GDD-SAVE-NUMERIC-INPUT");
        source.read_with(window, |item, cx| {
            assert!(item.copy_task.is_none());
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("64-bit integer")));
            assert!(!item.is_dirty(cx));
        });
        field.update_in(window, |editor, window, cx| {
            editor.set_text("-9007199254740993", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-NUMERIC-INPUT");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(
            filesystem
                .metadata(&destination)
                .await
                .expect("cancelled save")
                .is_none()
        );
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-NUMERIC-INPUT");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("open the edited graph");
        assert_ne!(reopened, source);
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        reopened.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected reopened node form");
            };
            assert_eq!(form.value.read(cx).text(cx), "-9007199254740993");
            assert!(!item.is_dirty(cx));
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected GDD document");
            };
            assert_eq!(contents.history_len(), original_history + 1);
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained source"),
            original
        );
    }

    #[gpui::test]
    async fn at_r04_1_workspace_opens_a_clean_working_copy_in_both_splits(cx: &mut TestAppContext) {
        let (original, history) = numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/open-working.gdd.zip");
        let other_path = PathBuf::from("/project/independent-archive.gdd.zip");
        for source in [&path, &other_path] {
            filesystem
                .write(source, &original)
                .await
                .expect("write original archive");
        }
        window.run_until_parked();
        let independent = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(other_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open independent archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("independent Inspector");
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("source Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let (base, revision, value_field) = source.read_with(window, |item, cx| {
            assert!(!item.can_save(cx));
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("source properties")
            };
            (
                form.snapshot.clone(),
                form.revision.clone(),
                form.value.clone(),
            )
        });
        let pane = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.split_and_clone(
                    workspace.active_pane().clone(),
                    workspace::SplitDirection::Right,
                    window,
                    cx,
                )
            })
            .await
            .expect("clone the source into a real split");
        let split = pane.read_with(window, |pane, _| {
            pane.active_item()
                .expect("split source tab")
                .downcast::<GraphiteDocumentItem>()
                .expect("split source Inspector")
        });
        assert_ne!(source, split);
        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        window.run_until_parked();
        paint(window);
        for view in [&source, &split] {
            view.read_with(window, |item, cx| {
                assert!(item.can_save(cx));
                assert!(!item.is_dirty(cx));
                let model = item.model.read(cx);
                let working = model.working.as_ref().expect("shared working model");
                assert!(Arc::ptr_eq(&working.saved, &base));
                assert!(Arc::ptr_eq(
                    model.gdd_contents().expect("working snapshot"),
                    &base
                ));
                assert!(Arc::ptr_eq(&model.revision, &revision));
                assert_eq!(base.history_len(), history);
            });
        }
        value_field.update_in(window, |editor, window, cx| {
            editor.set_text("-21", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
        window.run_until_parked();
        for view in [&source, &split] {
            view.read_with(window, |item, cx| {
                assert!(item.is_dirty(cx));
                let model = item.model.read(cx);
                let working = model.working.as_ref().expect("shared edited projection");
                assert!(Arc::ptr_eq(&working.saved, &base));
                assert_eq!(
                    model
                        .gdd_contents()
                        .expect("working snapshot")
                        .history_len(),
                    history + 1
                );
                let [node] = working.inspection.nodes.as_slice() else {
                    panic!("the original node")
                };
                assert!(matches!(
                    node.numeric_input(),
                    Some(GddNumericValue::Integer(-21))
                ));
            });
        }
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace
                .items_of_type::<GraphiteDocumentItem>(cx)
                .any(|item| item == independent)
        }));
        independent.read_with(window, |item, cx| {
            assert!(!item.can_save(cx));
            assert!(!item.is_dirty(cx));
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("independent source")
                    .history_len(),
                history
            );
        });
        for source in [&path, &other_path] {
            assert_eq!(
                filesystem
                    .load_bytes(source)
                    .await
                    .expect("original archive data"),
                original
            );
        }
    }

    #[gpui::test]
    async fn at_r04_r07_numeric_working_edits_share_a_snapshot_and_save_fresh_folders(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/working-value.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source file");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source")
            .downcast::<GraphiteDocumentItem>()
            .expect("GDD item");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");

        let set_draft =
            |item: &Entity<GraphiteDocumentItem>, draft, window: &mut gpui::VisualTestContext| {
                let field = item.read_with(window, |item, _| {
                    let NodeCopyPanel::Ready(form) = &item.node_copy else {
                        panic!("open node editor")
                    };
                    form.value.clone()
                });
                field.update_in(window, |editor, window, cx| {
                    editor.set_text(draft, window, cx)
                });
            };
        let check_view = |item: &Entity<GraphiteDocumentItem>,
                          expected,
                          dirty,
                          window: &gpui::VisualTestContext| {
            item.read_with(window, |item, cx| {
                assert_eq!(item.is_dirty(cx), dirty);
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("open node editor")
                };
                assert_eq!(form.value.read(cx).text(cx), expected);
                assert!(Arc::ptr_eq(
                    &form.snapshot,
                    item.model.read(cx).gdd_contents().expect("GDD snapshot")
                ));
                form.selected_node()
                    .expect("same node remains selected")
                    .storage_id()
            })
        };
        let node_id = check_view(&source, "-17", false, window);
        set_draft(&source, "3.5", window);
        click(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
        source.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("64-bit integer")));
        });
        set_draft(&source, "12", window);
        paint(window);
        click(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
        window.run_until_parked();
        assert_eq!(check_view(&source, "12", true, window), node_id);
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert_eq!(check_view(&source, "12", true, window), node_id);

        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(source_path.clone()));
        window.run_until_parked();
        source.read_with(window, |item, cx| {
            assert!(item.is_dirty(cx));
            assert!(matches!(&item.copy_feedback, Some((_, Color::Error))));
        });

        let split_pane = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.split_and_clone(
                    workspace.active_pane().clone(),
                    workspace::SplitDirection::Right,
                    window,
                    cx,
                )
            })
            .await
            .expect("split the native item");
        let split = split_pane.read_with(window, |pane, _| {
            pane.active_item()
                .expect("active split item")
                .downcast::<GraphiteDocumentItem>()
                .expect("GDD split")
        });
        split.update_in(window, |item, window, cx| {
            item.toggle_node_copy(window, cx);
            item.select_copy_node(0, window, cx);
        });
        assert_eq!(check_view(&split, "12", true, window), node_id);
        let (stale_revision, stale_snapshot, stale_node) = split.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("open node editor")
            };
            (
                form.revision.clone(),
                form.snapshot.clone(),
                form.selected_node().expect("selected node").clone(),
            )
        });

        let older_folder = PathBuf::from("/project/older-working.gdd");
        source.update_in(window, |item, window, cx| {
            item.save_working_folder(window, cx)
        });
        set_draft(&split, "36", window);
        split.update_in(window, |item, window, cx| {
            item.change_numeric_value(true, window, cx)
        });
        window.run_until_parked();
        assert_eq!(check_view(&source, "36", true, window), node_id);
        window.simulate_new_path_selection(|_| Some(older_folder.clone()));
        window.run_until_parked();
        let older = read_graphite_folder(filesystem.as_ref(), &older_folder)
            .await
            .expect("published older snapshot");
        assert_eq!(older.history_len(), original_history + 1);
        let older_nodes = older.graph_nodes().await.expect("older graph");
        assert!(matches!(
            older_nodes.first().and_then(GddGraphNode::numeric_input),
            Some(GddNumericValue::Integer(12))
        ));
        assert_eq!(check_view(&split, "36", true, window), node_id);
        source.update_in(window, |item, window, cx| {
            item.stage_working_snapshot(
                stale_revision,
                async move {
                    stale_snapshot
                        .set_graph_numeric_input_in_copy(
                            &stale_node,
                            GddNumericValue::Integer(99),
                            ARCHIVE_BUDGET,
                        )
                        .await
                },
                window,
                cx,
            );
        });
        window.run_until_parked();
        source.read_with(window, |item, _| {
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("working copy changed")));
        });
        assert_eq!(check_view(&source, "36", true, window), node_id);

        let destination = PathBuf::from("/project/saved-working.gdd");
        let project = workspace.read_with(window, |workspace, _| workspace.project().clone());
        let save = split.update_in(window, |item, window, cx| {
            item.save(SaveOptions::default(), project, window, cx)
        });
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        save.await
            .expect("workspace save waits for folder publication");
        window.run_until_parked();
        assert_eq!(check_view(&source, "36", false, window), node_id);
        assert_eq!(check_view(&split, "36", false, window), node_id);

        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace.read_with(window, |workspace, cx| {
            workspace
                .active_item_as::<GraphiteDocumentItem>(cx)
                .expect("opened working folder")
        });
        assert_ne!(reopened, split);
        let saved = reopened.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            item.model
                .read(cx)
                .gdd_contents()
                .expect("saved GDD")
                .clone()
        });
        assert_eq!(saved.history_len(), original_history + 2);
        let nodes = saved.graph_nodes().await.expect("saved working graph");
        let [node] = nodes.as_slice() else {
            panic!("expected the saved node")
        };
        assert_eq!(node.storage_id(), node_id);
        assert!(matches!(
            node.numeric_input(),
            Some(GddNumericValue::Integer(36))
        ));
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_r05_2_working_gdd_history_is_shared_across_focused_split_views(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::SingleNode).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/history-working.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let (value_field, node_id) = source.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("open node editor")
            };
            (
                form.value.clone(),
                form.selected_node().expect("selected node").storage_id(),
            )
        });
        for draft in ["12", "36"] {
            value_field.update_in(window, |editor, window, cx| {
                editor.set_text(draft, window, cx)
            });
            paint(window);
            click(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
            window.run_until_parked();
        }
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some("/project/history-baseline.gdd".into()));
        window.run_until_parked();

        let pane = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.split_and_clone(
                    workspace.active_pane().clone(),
                    workspace::SplitDirection::Right,
                    window,
                    cx,
                )
            })
            .await
            .expect("native split");
        let split = pane.read_with(window, |pane, _| {
            pane.active_item()
                .expect("split tab")
                .downcast::<GraphiteDocumentItem>()
                .expect("working GDD split")
        });
        split.update_in(window, |item, window, cx| {
            item.toggle_node_copy(window, cx);
            item.select_copy_node(0, window, cx);
        });
        let assert_working = |current: &Entity<GraphiteDocumentItem>,
                              value,
                              dirty,
                              window: &gpui::VisualTestContext| {
            current.read_with(window, |item, cx| {
                assert_eq!(item.is_dirty(cx), dirty);
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("open node editor")
                };
                let node = form.selected_node().expect("selected node");
                assert_eq!(node.storage_id(), node_id);
                assert_eq!(form.value.read(cx).text(cx), value);
                let snapshot = item.model.read(cx).gdd_contents().expect("current GDD");
                assert_eq!(snapshot.history_len(), original_history + 2);
            });
        };
        assert_working(&source, "36", false, window);
        assert_working(&split, "36", false, window);
        let (undo, redo) = if cfg!(target_os = "macos") {
            ("cmd-z", "cmd-shift-z")
        } else {
            ("ctrl-z", "ctrl-shift-z")
        };
        window.update(|_, cx| {
            use settings::{KeymapFile, KeymapFileLoadResult};

            for (platform, contents, expected) in [
                (
                    "macos",
                    include_str!("../../../assets/keymaps/default-macos.json"),
                    2,
                ),
                (
                    "linux",
                    include_str!("../../../assets/keymaps/default-linux.json"),
                    5,
                ),
                (
                    "windows",
                    include_str!("../../../assets/keymaps/default-windows.json"),
                    3,
                ),
            ] {
                let bindings = match KeymapFile::load(contents, cx) {
                    KeymapFileLoadResult::Success { key_bindings }
                    | KeymapFileLoadResult::SomeFailedToLoad { key_bindings, .. } => key_bindings,
                    KeymapFileLoadResult::JsonParseFailure { error } => {
                        panic!("{platform} keymap: {error:#}")
                    }
                };
                let context = "GraphiteArtboard || GraphiteGdd";
                let bindings = bindings
                    .into_iter()
                    .filter(|binding| {
                        binding
                            .predicate()
                            .is_some_and(|predicate| predicate.to_string() == context)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(bindings.len(), expected, "{platform} history bindings");
                if platform == std::env::consts::OS {
                    cx.bind_keys(bindings);
                }
            }
        });
        value_field.update_in(window, |editor, window, cx| {
            editor.focus_handle(cx).focus(window, cx)
        });
        paint(window);
        window.simulate_keystrokes(undo);
        window.run_until_parked();
        assert_working(&split, "36", false, window);

        for (focused, key, value, dirty) in [
            (&split, undo, "12", true),
            (&source, undo, "-17", true),
            (&split, redo, "12", true),
            (&source, redo, "36", false),
        ] {
            focused.update_in(window, |item, window, cx| {
                item.focus_handle.focus(window, cx)
            });
            paint(window);
            window.simulate_keystrokes(key);
            window.run_until_parked();
            assert_working(&source, value, dirty, window);
            assert_working(&split, value, dirty, window);
        }

        paint(window);
        click(window, "GRAPHITE-GDD-UNDO");
        window.run_until_parked();
        assert_working(&source, "12", true, window);
        let reverted_path = PathBuf::from("/project/history-with-redo.gdd");
        source.update_in(window, |item, window, cx| {
            item.save_working_folder(window, cx)
        });
        window.simulate_new_path_selection(|_| Some(reverted_path.clone()));
        window.run_until_parked();
        assert_working(&split, "12", false, window);
        let reopened = read_graphite_folder(filesystem.as_ref(), &reverted_path)
            .await
            .expect("reopen published undone history");
        assert!(reopened.can_move_history(GddHistoryDirection::Redo));
        let redone = reopened
            .move_history_in_copy(GddHistoryDirection::Redo, ARCHIVE_BUDGET)
            .await
            .expect("redo the saved interaction");
        let nodes = redone.graph_nodes().await.expect("redone graph");
        let [node] = nodes.as_slice() else {
            panic!("retained node")
        };
        assert_eq!(node.storage_id(), node_id);
        assert!(matches!(
            node.numeric_input(),
            Some(GddNumericValue::Integer(36))
        ));
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("preserved source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_032_2_workspace_edits_working_nested_name_and_pin_order(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/working-node-metadata.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open scoped graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        let invoke = |window: &mut gpui::VisualTestContext, selector| {
            paint(window);
            click(window, selector);
            window.run_until_parked();
            paint(window);
        };
        invoke(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
        let (root_id, first_id, second_id, scope, field, revision) =
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Inspector")
                };
                let [root, _, first, second, ..] = form.nodes.as_slice() else {
                    panic!("scoped graph")
                };
                assert_eq!(root.name(), first.name());
                assert_ne!(root.network_id(), first.network_id());
                assert_eq!(
                    form.selected_node().map(GddGraphNode::storage_id),
                    Some(first.storage_id())
                );
                (
                    root.storage_id(),
                    first.storage_id(),
                    second.storage_id(),
                    first.network_id(),
                    form.name.clone(),
                    item.model.read(cx).revision.clone(),
                )
            });
        invoke(window, "GRAPHITE-GDD-SAVE-RENAMED");
        item.read_with(window, |item, cx| {
            assert!(Arc::ptr_eq(&item.model.read(cx).revision, &revision));
            assert!(!item.is_dirty(cx));
            assert!(matches!(&item.copy_feedback, Some((_, Color::Error))));
        });

        field.update_in(window, |editor, window, cx| {
            editor.set_text("Working nested value", window, cx);
        });
        invoke(window, "GRAPHITE-GDD-SAVE-RENAMED");
        invoke(window, "GRAPHITE-GDD-SAVE-PIN");
        invoke(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        invoke(window, "GRAPHITE-GDD-NODE-COPY-ITEM-3");
        invoke(window, "GRAPHITE-GDD-SAVE-PIN");
        let check = |nodes: &[GddGraphNode], first_order, second_order| {
            let root = nodes
                .iter()
                .find(|node| node.storage_id() == root_id)
                .expect("root with repeated name");
            let first = nodes
                .iter()
                .find(|node| node.storage_id() == first_id)
                .expect("renamed nested node");
            let second = nodes
                .iter()
                .find(|node| node.storage_id() == second_id)
                .expect("other nested node");
            assert_eq!(root.name(), "Repeated value");
            assert!(!root.is_pinned());
            assert!(root.unavailable_proto().is_some());
            assert_eq!(first.network_id(), scope);
            assert_eq!(second.network_id(), scope);
            assert_eq!(first.name(), "Working nested value");
            assert_eq!(first.pinned_position(), Some((first_order, 2)));
            assert_eq!(second.pinned_position(), Some((second_order, 2)));
            assert_eq!(first.numeric_import_source(), Some(0));
            assert_eq!(second.numeric_import_source(), Some(1));
        };
        let check_view = |window: &gpui::VisualTestContext, first_order, second_order| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("working Inspector")
                };
                assert_eq!(form.active_network().map(|network| network.id), Some(scope));
                assert_eq!(
                    form.selected_node().map(GddGraphNode::storage_id),
                    Some(second_id)
                );
                assert!(item.is_dirty(cx));
                check(&form.nodes, first_order, second_order);
            });
        };
        check_view(window, 1, 2);
        for (selector, first_order, second_order) in [
            ("GRAPHITE-GDD-MOVE-PIN-UP", 2, 1),
            ("GRAPHITE-GDD-UNDO", 1, 2),
            ("GRAPHITE-GDD-REDO", 2, 1),
        ] {
            invoke(window, selector);
            check_view(window, first_order, second_order);
        }

        let destination = PathBuf::from("/project/saved-working-node-metadata.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen working metadata");
        check(
            &saved.graph_nodes().await.expect("scoped saved nodes"),
            2,
            1,
        );
        assert!(!item.read_with(window, |item, cx| item.is_dirty(cx)));
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_032_2_workspace_renames_a_selected_nested_node_in_a_folder_copy(
        cx: &mut TestAppContext,
    ) {
        let (original, saved_history_len) = nested_graph_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/nested.gdd.zip");
        let destination = PathBuf::from("/project/renamed.gdd");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("write source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_nested_copy_node(window);
        let editor = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("expected the rename form");
            };
            assert_eq!(form.selected, Some(2));
            form.name.clone()
        });
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("Nested label", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-RENAMED");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(
            filesystem
                .metadata(&destination)
                .await
                .expect("check cancellation")
                .is_none()
        );
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-RENAMED");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopen the renamed folder");
        assert_ne!(reopened, item);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the renamed GDD");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents
            .graph_nodes()
            .await
            .expect("rebuild source graph IDs");
        let [root, _, nested] = nodes.as_slice() else {
            panic!("expected the original three nodes");
        };
        assert_eq!(root.name(), "");
        assert_eq!(nested.name(), "Nested label");
        assert_eq!(contents.history_len(), saved_history_len + 1);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original bytes"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_hist_003_004_workspace_moves_saved_gdd_history_in_new_folders(
        cx: &mut TestAppContext,
    ) {
        let (fixture, retired) = nested_graph_archive().await;
        let fixture = open_gdd_archive(&fixture.into(), ARCHIVE_BUDGET)
            .await
            .expect("source nested graph");
        let opaque_path = "opaque/notes/provenance.txt";
        let opaque_bytes = b"keep this original nested record";
        let mut entries = fixture
            .folder_files()
            .expect("source entries")
            .into_iter()
            .map(|(path, bytes)| (path.to_str().expect("source path").to_owned(), bytes))
            .collect::<Vec<_>>();
        entries.push((opaque_path.into(), opaque_bytes.to_vec()));
        let fixture = inspect_gdd_folder_files(entries, ARCHIVE_BUDGET)
            .await
            .expect("nested graph with opaque source data");
        let graph_nodes = fixture.graph_nodes().await.expect("stable source nodes");
        let nested = graph_nodes
            .get(2)
            .expect("nested node with a repeated local ID");
        let edited = fixture
            .rename_graph_node_in_copy(nested, "Latest nested label".into(), ARCHIVE_BUDGET)
            .await
            .expect("original saved interaction");
        let original = edited
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("archive with saved edit");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/edited.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("original archive");
        window.run_until_parked();
        let first = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open native GDD")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        let undone_path = PathBuf::from("/project/undo.gdd");
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-UNDO");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(
            filesystem
                .metadata(&undone_path)
                .await
                .expect("cancelled save")
                .is_none()
        );

        let mut previous = first;
        for (selector, destination, name, opposite) in [
            (
                "GRAPHITE-GDD-SAVE-UNDO",
                undone_path,
                "",
                GddHistoryDirection::Redo,
            ),
            (
                "GRAPHITE-GDD-SAVE-REDO",
                PathBuf::from("/project/redo.gdd"),
                "Latest nested label",
                GddHistoryDirection::Undo,
            ),
        ] {
            paint(window);
            click(window, selector);
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            assert!(
                filesystem
                    .metadata(&destination)
                    .await
                    .expect("published history")
                    .is_some()
            );
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("open the saved history copy");
            assert_ne!(reopened, previous);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected the native GDD folder");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let nodes = contents
                .graph_nodes()
                .await
                .expect("retained node identities");
            assert_eq!(nodes.first().expect("same root local ID").name(), "");
            assert_eq!(nodes.get(2).expect("same nested local ID").name(), name);
            assert_eq!(contents.history_len(), retired + 1);
            assert_eq!(
                contents
                    .read_entry(opaque_path)
                    .expect("original nested opaque record")
                    .as_slice(),
                opaque_bytes,
            );
            assert!(contents.can_move_history(opposite));
            previous = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_033_2_workspace_pins_and_unpins_only_the_selected_node_in_copies(
        cx: &mut TestAppContext,
    ) {
        let (original, originally_retired) = nested_graph_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/graph.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("original archive");
        window.run_until_parked();
        let mut current = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        for (path, pinned, added_history) in [
            ("/project/pinned.gdd", true, 2),
            ("/project/unpinned.gdd", false, 4),
        ] {
            let destination = PathBuf::from(path);
            choose_nested_copy_node(window);
            click(window, "GRAPHITE-GDD-SAVE-PIN");
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            assert!(
                filesystem
                    .metadata(&destination)
                    .await
                    .expect("published folder")
                    .is_some()
            );
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![destination]));
            window.run_until_parked();
            let reopened = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("open the new pinned state");
            assert_ne!(reopened, current);
            let contents = reopened.read_with(window, |item, cx| {
                let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                    panic!("expected the native GDD model");
                };
                assert!(!item.is_dirty(cx));
                contents.clone()
            });
            let nodes = contents
                .graph_nodes()
                .await
                .expect("original node identities");
            let [root, parent, nested] = nodes.as_slice() else {
                panic!("expected the three original nodes");
            };
            assert!(!root.is_pinned() && !parent.is_pinned());
            assert_eq!(nested.is_pinned(), pinned);
            assert_eq!(contents.history_len(), originally_retired + added_history);
            current = reopened;
        }
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("retained source"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_034_2_workspace_reorders_pinned_nodes_in_a_new_folder(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) = nested_graph_archive_with_sibling(true).await;
        let mut pinned = open_gdd_archive(&original.into(), ARCHIVE_BUDGET)
            .await
            .expect("source archive")
            .into_inspected();
        for index in [2, 3] {
            let nodes = pinned.graph_nodes().await.expect("nested nodes");
            pinned = pinned
                .set_graph_node_pinned_in_copy(
                    nodes.get(index).expect("selected nested node"),
                    true,
                    ARCHIVE_BUDGET,
                )
                .await
                .expect("pin nested node in the source fixture");
        }
        let bytes = pinned
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("serialize the pinned fixture");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/pins.gdd.zip");
        let destination = PathBuf::from("/project/reordered.gdd");
        filesystem
            .write(&source_path, &bytes)
            .await
            .expect("write source fixture");
        window.run_until_parked();
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_nested_copy_node(window);
        click(window, "GRAPHITE-GDD-MOVE-PIN-DOWN");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(
            filesystem
                .metadata(&destination)
                .await
                .expect("no saved folder")
                .is_none()
        );
        paint(window);
        click(window, "GRAPHITE-GDD-MOVE-PIN-DOWN");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("select the reopened destination");
        assert_ne!(reopened, source);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected the native GDD model");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents.graph_nodes().await.expect("reopened nodes");
        let [root, parent, first, second] = nodes.as_slice() else {
            panic!("expected two root nodes and two nested nodes");
        };
        assert_eq!(root.pinned_position(), None);
        assert_eq!(parent.pinned_position(), None);
        assert_eq!(first.pinned_position(), Some((2, 2)));
        assert_eq!(second.pinned_position(), Some((1, 2)));
        assert_eq!(contents.history_len(), original_history + 5);
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source archive"),
            bytes
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_017_018_workspace_shows_the_saved_view_without_dirtying_the_document(
        cx: &mut TestAppContext,
    ) {
        use graphite_document_graph_storage::{
            HotOp, Network, PeerId, ROOT_NETWORK, RegistryDelta, TimeStamp, attr::session, to_value,
        };
        use graphite_editor::messages::portfolio::document::utility_types::{
            misc::PTZ, network_interface::Previewing,
        };
        use std::collections::BTreeMap;

        let peer = PeerId(9);
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            peer,
            173,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("create source storage");
        owner
            .apply_hot_op(HotOp {
                op: RegistryDelta::AddNetwork {
                    id: ROOT_NETWORK,
                    network: Network::default(),
                },
                timestamp: TimeStamp { counter: 1, peer },
            })
            .expect("create the original root graph");
        owner
            .retire_pending_interaction()
            .expect("retire the graph");
        let mut artwork_camera = PTZ::default();
        artwork_camera.set_zoom(1.75);
        owner
            .set_view_settings(BTreeMap::from([
                (
                    session::doc::PTZ.into(),
                    to_value(&artwork_camera).expect("artwork PTZ"),
                ),
                (
                    session::doc::RULERS_VISIBLE.into(),
                    to_value(&false).expect("rulers"),
                ),
            ]))
            .expect("save per-peer preferences");
        let mut graph_camera = PTZ::default();
        graph_camera.pan.x = -56.;
        graph_camera.set_zoom(0.75);
        let previewing = Previewing::Yes {
            root_node_to_restore: None,
        };
        owner
            .set_network_view_settings(BTreeMap::from([(
                ROOT_NETWORK,
                BTreeMap::from([
                    (
                        session::network::NAV_PTZ.into(),
                        to_value(&graph_camera).expect("root PTZ"),
                    ),
                    (
                        session::network::PREVIEWING.into(),
                        to_value(&previewing).expect("preview"),
                    ),
                ]),
            )]))
            .expect("save per-network navigation");
        let original = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &HashMapResourceStorage::new(),
                None,
            )
            .await
            .expect("source archive");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/saved-view.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("original settings fixture");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the saved GDD")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-SAVED-VIEW");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let SavedViewPanel::Ready(view) = &item.saved_view else {
                panic!("the native view should be ready");
            };
            let artwork = view.document.as_ref().expect("saved artwork view");
            assert_eq!(artwork.camera, artwork_camera);
            assert!(!artwork.rulers_visible);
            let [root] = view.networks.as_slice() else {
                panic!("the saved root graph view");
            };
            assert_eq!(root.id, ROOT_NETWORK);
            assert_eq!(root.navigation.node_graph_ptz, graph_camera);
            assert_eq!(root.previewing, previewing);
            assert!(!item.is_dirty(cx));
        });
        click(window, "GRAPHITE-GDD-SAVED-VIEW");
        item.read_with(window, |item, cx| {
            assert!(matches!(item.saved_view, SavedViewPanel::Closed));
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("source GDD"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_016_2_workspace_displays_recovered_history_and_rejects_a_truncated_hot_log(
        cx: &mut TestAppContext,
    ) {
        use graphite_document_container::archive::{Archive, ArchiveWriter, Zip};
        use graphite_document_format::{Layout, io::path_for};
        use graphite_document_graph_storage::{
            HotOp, Network, NetworkId, PeerId, ROOT_NETWORK, RegistryDelta, TimeStamp,
        };
        use std::io::Cursor;

        let peer = PeerId(7);
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            peer,
            172,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("create original GDD owner");
        for (id, counter) in [(ROOT_NETWORK, 1), (NetworkId(41), 2)] {
            owner
                .apply_hot_op(HotOp {
                    op: RegistryDelta::AddNetwork {
                        id,
                        network: Network::default(),
                    },
                    timestamp: TimeStamp { counter, peer },
                })
                .expect("original network operation");
            if id == ROOT_NETWORK {
                owner.retire_pending_interaction().expect("retired history");
            }
        }
        let original: Arc<[u8]> = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions {
                    include_registry: false,
                    ..ExportOptions::default()
                },
                &HashMapResourceStorage::new(),
                None,
            )
            .await
            .expect("original history and hot log export")
            .into();
        let hot_path = path_for(
            GddV1Layout.hot_log_basename(),
            owner.manifest().codecs.hot_log,
        );
        let admitted = open_gdd_archive(&original, ARCHIVE_BUDGET)
            .await
            .expect("pinned archive with pending operation");
        let mut writer = Zip::writer(Cursor::new(Vec::new())).expect("pinned ZIP writer");
        for (name, mut bytes) in admitted.folder_files().expect("exact source records") {
            if name == Path::new(&hot_path) {
                bytes.truncate(3);
            }
            writer
                .write_entry(name.to_str().expect("validated UTF-8 entry"), &bytes)
                .expect("write the malformed history fixture");
        }
        let malformed = writer
            .finish_into()
            .expect("valid ZIP envelope")
            .into_inner();
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let original_path = PathBuf::from("/project/history-recovery.gdd.zip");
        let malformed_path = PathBuf::from("/project/truncated-pending.gdd.zip");
        filesystem
            .write(&original_path, &original)
            .await
            .expect("original history");
        filesystem
            .write(&malformed_path, &malformed)
            .await
            .expect("malformed pending log");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(original_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("recover through Workspace File Open")
            .downcast::<GraphiteDocumentItem>()
            .expect("native GDD Inspector");
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-RECOVERY").is_some());
        item.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected recovered GDD");
            };
            let recovery = contents.recovery();
            assert_eq!(recovery.source, GddRecoverySource::HistoryOnly);
            assert_eq!(recovery.network_count, 2);
            assert_eq!(recovery.history_len, 1);
            assert_eq!(recovery.hot_log_len, 1);
            assert!(!item.is_dirty(cx));
        });
        let rejected = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(malformed_path.clone(), Default::default(), window, cx)
            })
            .await;
        assert!(
            matches!(rejected, Err(error) if format!("{error:#}").contains("frame length prefix truncated"))
        );
        assert_eq!(
            workspace.read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            }),
            Some(item.clone())
        );
        let occupied = PathBuf::from("/project/occupied.gdd");
        filesystem
            .create_new_directory_atomically(
                occupied.clone(),
                vec![(PathBuf::from("original.txt"), b"keep this folder".to_vec())],
            )
            .await
            .expect("existing directory fixture");
        window.run_until_parked();
        let recovered_path = PathBuf::from("/project/recovered.gdd");
        for (selection, published) in [
            (None, false),
            (Some(occupied.clone()), false),
            (Some(recovered_path.clone()), true),
        ] {
            paint(window);
            click(window, "GRAPHITE-GDD-SAVE-RECOVERED");
            window.simulate_new_path_selection(|_| selection);
            window.run_until_parked();
            assert_eq!(
                filesystem
                    .metadata(&recovered_path)
                    .await
                    .expect("recovered folder publication")
                    .is_some(),
                published,
            );
        }
        assert_eq!(
            filesystem
                .load_bytes(&occupied.join("original.txt"))
                .await
                .expect("existing folder data"),
            b"keep this folder",
        );
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|options| {
            assert!(options.directories && !options.files && !options.multiple);
            Some(vec![recovered_path])
        });
        window.run_until_parked();
        let recovered = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened recovered folder");
        assert_ne!(recovered, item);
        recovered.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected a reopened GDD");
            };
            let summary = contents.recovery();
            assert_eq!(summary.source, GddRecoverySource::RegistryAndHistory);
            assert_eq!(summary.network_count, 2);
            assert_eq!(summary.history_len, 2);
            assert_eq!(summary.hot_log_len, 0);
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&original_path)
                .await
                .expect("original recovery fixture"),
            original.as_ref(),
        );
        assert_eq!(
            filesystem
                .load_bytes(&malformed_path)
                .await
                .expect("original malformed fixture"),
            malformed,
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_015_2_workspace_embeds_a_legacy_document_selected_for_export(
        cx: &mut TestAppContext,
    ) {
        use graphite_document_format::{Layout, manifest::SUPPORTED_FORMAT_VERSION};
        use graphite_editor::messages::portfolio::document::DocumentMessageHandler;

        let fixture = DocumentMessageHandler::default();
        let payload = format!("{}\n\n", fixture.serialize_document()).into_bytes();
        let original = archive_bytes(SUPPORTED_FORMAT_VERSION, ExportFormat::Zip).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/source.gdd.zip");
        let output_path = PathBuf::from("/project/with-legacy.gdd.tar.xz");
        let legacy_path = PathBuf::from("/project/original.graphite");
        let invalid_path = PathBuf::from("/project/invalid.graphite");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("original archive fixture");
        filesystem
            .write(&legacy_path, &payload)
            .await
            .expect("legacy fixture");
        filesystem
            .write(&invalid_path, b"{}")
            .await
            .expect("invalid fixture");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open original archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite Inspector");
        paint(window);
        click(window, "GRAPHITE-EXPORT-EMBED-LEGACY");
        for (selected, expect_publish) in [
            (None, false),
            (Some(invalid_path), false),
            (Some(legacy_path.clone()), true),
        ] {
            click(window, "GRAPHITE-EXPORT-GDD-XZ");
            window.simulate_new_path_selection(|_| Some(output_path.clone()));
            window.run_until_parked();
            window.simulate_path_prompt_response(|options| {
                assert!(options.files && !options.multiple && !options.directories);
                selected.map(|path| vec![path])
            });
            window.run_until_parked();
            paint(window);
            assert_eq!(
                filesystem
                    .metadata(&output_path)
                    .await
                    .expect("publication status")
                    .is_some(),
                expect_publish
            );
        }
        let copy = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(output_path, Default::default(), window, cx)
            })
            .await
            .expect("open the published XZ archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("reopened native Inspector");
        copy.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents, .. } = &item.model.read(cx).document else {
                panic!("expected GDD archive");
            };
            let embedded = contents
                .read_entry(GddV1Layout.legacy_path())
                .expect("legacy payload in the source layout");
            assert_eq!(embedded.as_slice(), payload);
        });
        item.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents, .. } = &item.model.read(cx).document else {
                panic!("expected GDD source");
            };
            assert!(contents.read_entry(GddV1Layout.legacy_path()).is_err());
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("source GDD"),
            original
        );
        assert_eq!(
            filesystem
                .load_bytes(&legacy_path)
                .await
                .expect("legacy file"),
            payload
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_013_2_embeds_only_matching_files_chosen_in_workspace(
        cx: &mut TestAppContext,
    ) {
        use graphite_document_format::Layout;
        use graphite_document_graph_storage::NoMetadata;
        use graphite_graph_craft::{
            ProtoNodeIdentifier,
            document::{
                DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
                value::TaggedValue,
            },
        };
        use graphite_resource::{DataSource, ResourceHash, ResourceId, ResourceRegistry};

        let mut source = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            175,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("original Graphite working copy");
        let resource_id = ResourceId::from(77_u64);
        let payload = b"exact local resource bytes";
        let hash = ResourceHash::from(payload.as_slice());
        let mut resources = ResourceRegistry::new();
        resources.push_source_back(
            &resource_id,
            DataSource::Url(
                "https://example.invalid/linked.bin"
                    .parse()
                    .expect("fixture URL"),
            ),
        );
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
        source
            .commit_from_runtime(&network, &NoMetadata, &resources, &source_store)
            .expect("original URL-linked resource");
        let original = source
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &source_store,
                None,
            )
            .await
            .expect("source ZIP retaining the link");

        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/linked.gdd.zip");
        let output = PathBuf::from("/project/portable.gdd.zip");
        let approved = PathBuf::from("/project/selected.bin");
        let wrong = PathBuf::from("/project/unmatched.bin");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source fixture");
        filesystem
            .write(&approved, payload)
            .await
            .expect("selected resource file");
        filesystem
            .write(&wrong, b"bytes with a different hash")
            .await
            .expect("unmatched fixture");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open linked archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite inspector");
        paint(window);
        click(window, "GRAPHITE-EXPORT-EMBED-RESOURCES");
        for (selected, expect_publish) in
            [(None, false), (Some(wrong), false), (Some(approved), true)]
        {
            click(window, "GRAPHITE-EXPORT-GDD-ZIP");
            window.simulate_new_path_selection(|_| Some(output.clone()));
            window.run_until_parked();
            window.simulate_path_prompt_response(|options| {
                assert!(options.files && options.multiple && !options.directories);
                selected.map(|path| vec![path])
            });
            window.run_until_parked();
            paint(window);
            assert_eq!(
                filesystem
                    .metadata(&output)
                    .await
                    .expect("check publication")
                    .is_some(),
                expect_publish
            );
        }
        let bytes = Arc::<[u8]>::from(
            filesystem
                .load_bytes(&output)
                .await
                .expect("portable archive"),
        );
        let copy = open_gdd_archive(&bytes, ARCHIVE_BUDGET)
            .await
            .expect("portable archive reopens");
        let resource_path = GddV1Layout.resource_path(&hash);
        assert_eq!(
            copy.read_entry(&resource_path)
                .expect("embedded file")
                .as_slice(),
            payload
        );
        item.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents, .. } = &item.model.read(cx).document else {
                panic!("expected source archive");
            };
            assert!(contents.read_entry(&resource_path).is_err());
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("unchanged source"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_014_2_workspace_rejects_missing_historical_resource(
        cx: &mut TestAppContext,
    ) {
        use graphite_document_graph_storage::NoMetadata;
        use graphite_graph_craft::document::NodeNetwork;
        use graphite_resource::{ResourceHash, ResourceId, ResourceRegistry};

        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            176,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("original Graphite working copy");
        let store = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(
                &NodeNetwork::default(),
                &NoMetadata,
                &ResourceRegistry::new(),
                &store,
            )
            .expect("mount interaction");
        let bytes = b"resource retained for redo";
        let hash = ResourceHash::from(bytes.as_slice());
        owner
            .add_resource(ResourceId::from(14_u64), bytes)
            .expect("pinned resource history");
        owner.undo().expect("leave resource only in history");
        let complete = owner
            .export_to_bytes(ExportFormat::Zip, ExportOptions::default(), &store, None)
            .await
            .expect("complete original history archive");
        owner
            .remove_resource(&hash)
            .expect("construct a missing-resource fixture");
        let incomplete = owner
            .export_to_bytes(ExportFormat::Zip, ExportOptions::default(), &store, None)
            .await
            .expect("source emits a history archive with the fixture bytes removed");

        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let complete_path = PathBuf::from("/project/complete.gdd.zip");
        let incomplete_path = PathBuf::from("/project/incomplete.gdd.zip");
        filesystem
            .write(&complete_path, &complete)
            .await
            .expect("valid source fixture");
        filesystem
            .write(&incomplete_path, &incomplete)
            .await
            .expect("incomplete source fixture");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(complete_path, Default::default(), window, cx)
            })
            .await
            .expect("open complete history archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite inspector");
        let rejected = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(incomplete_path.clone(), Default::default(), window, cx)
            })
            .await;
        assert!(matches!(rejected, Err(error) if format!("{error:#}").contains(&hash.to_string())));
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<GraphiteDocumentItem>(cx)),
            Some(item)
        );
        assert_eq!(
            filesystem
                .load_bytes(&incomplete_path)
                .await
                .expect("incomplete original is intact"),
            incomplete
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_003_2_workspace_creates_and_reopens_a_new_gdd_folder(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        window.dispatch_action(NewGraphiteFolder);
        window.run_until_parked();
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        assert!(
            workspace
                .read_with(window, |workspace, cx| workspace
                    .active_item_as::<GraphiteDocumentItem>(cx))
                .is_none()
        );

        let path = PathBuf::from("/new.gdd");
        window.dispatch_action(NewGraphiteFolder);
        window.run_until_parked();
        window.simulate_new_path_selection(|parent| {
            assert_eq!(parent, Path::new("/project"));
            Some(path.clone())
        });
        window.run_until_parked();
        let item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("new GDD folder in native Workspace");
        let contents = read_graphite_folder(filesystem.as_ref(), &path)
            .await
            .expect("reopen the persisted source layout");
        assert_ne!(contents.manifest().document_id, 0);
        assert_eq!(contents.manifest().editor_version, "Made");
        item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            let InspectedDocument::Gdd {
                contents: active, ..
            } = &model.document
            else {
                panic!("expected the created GDD folder");
            };
            assert_eq!(
                active.manifest().document_id,
                contents.manifest().document_id
            );
            assert_eq!(
                active.folder_files().expect("saved snapshot"),
                contents.folder_files().expect("same records on disk")
            );
            assert!(!item.is_dirty(cx));
        });
        window.dispatch_action(NewGraphiteFolder);
        window.run_until_parked();
        window.simulate_new_path_selection(|_| Some(path.clone()));
        window.run_until_parked();
        let still_original = read_graphite_folder(filesystem.as_ref(), &path)
            .await
            .expect("original directory remains usable");
        assert_eq!(
            still_original.manifest().document_id,
            contents.manifest().document_id
        );
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<GraphiteDocumentItem>(cx)),
            Some(item)
        );
    }

    #[gpui::test]
    async fn at_gra_graph_005_2_workspace_inserts_in_the_visible_nested_canvas(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = nested_graph_archive_with_sibling(true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/nested-insertion.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("nested source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open nested archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        paint(window);
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let (network, original_nodes) = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("saved scoped graph")
            };
            let network = form
                .active_network()
                .filter(|network| network.parent.is_some())
                .expect("nested network");
            (
                network.id,
                form.nodes
                    .iter()
                    .map(|node| (node.storage_id(), node.network_id()))
                    .collect::<BTreeSet<_>>(),
            )
        });
        let bounds = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH")
            .expect("open nested canvas");
        click_graph_point(
            window,
            bounds.origin + gpui::point(px(250.), px(177.)),
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        click(window, "GRAPHITE-GDD-ADD-Sign");
        window.run_until_parked();
        paint(window);
        let (new_id, position) = item.read_with(window, |item, cx| {
            assert!(item.is_dirty(cx));
            let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("working scoped graph") };
            assert_eq!(form.active_network().expect("same active scope").id, network);
            let added = form.nodes.iter().filter(|node| !original_nodes.contains(&(node.storage_id(), node.network_id()))).collect::<Vec<_>>();
            let [node] = added.as_slice() else { panic!("one added scoped node") };
            assert_eq!(node.network_id(), network);
            assert_eq!(node.display_name(), "Sign");
            assert!(matches!(&form.saved_graph, SavedGraphPanel::Ready(geometry) if geometry.nodes.iter().any(|geometry| geometry.storage_id == node.storage_id())));
            (node.storage_id(), node.grid_position().expect("saved snapped position"))
        });
        let destination = PathBuf::from("/project/saved-nested-insertion.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopened nested insertion");
        let nodes = saved.graph_nodes().await.expect("reopened graph");
        assert_eq!(nodes.len(), original_nodes.len() + 1);
        let new = nodes
            .iter()
            .find(|node| node.storage_id() == new_id)
            .expect("same new storage ID");
        assert_eq!(new.network_id(), network);
        assert_eq!(new.grid_position(), Some(position));
        assert!(original_nodes.iter().all(|address| {
            nodes
                .iter()
                .any(|node| (node.storage_id(), node.network_id()) == *address)
        }));
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("source ZIP remains exact"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_005_2_workspace_inserts_at_the_canvas_right_click_position(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/blank-pointer.gdd");
        window.dispatch_action(NewGraphiteFolder);
        window.run_until_parked();
        window.simulate_new_path_selection(|_| Some(source.clone()));
        window.run_until_parked();
        let item = workspace.read_with(window, |workspace, cx| {
            workspace
                .active_item_as::<GraphiteDocumentItem>(cx)
                .expect("blank Graphite folder")
        });
        let original = read_graphite_folder(filesystem.as_ref(), &source)
            .await
            .expect("source folder")
            .folder_files()
            .expect("source entries");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            assert!(matches!(&item.node_copy, NodeCopyPanel::Ready(form) if
                matches!(&form.saved_graph, SavedGraphPanel::Ready(graph) if graph.nodes.is_empty() && graph.boundary_ports.is_empty())));
        });
        let bounds = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH")
            .expect("blank root canvas");
        let position = bounds.origin + gpui::point(px(108.), px(91.));
        click_graph_point(
            window,
            position,
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        assert!(window.debug_bounds("GRAPHITE-GDD-ADD-Sign").is_some());
        click_graph_point(
            window,
            bounds.origin + gpui::point(px(8.), px(8.)),
            gpui::MouseButton::Left,
            gpui::Modifiers::default(),
        );
        assert!(window.debug_bounds("GRAPHITE-GDD-ADD-Sign").is_none());
        item.read_with(window, |item, cx| assert!(!item.is_dirty(cx)));

        click_graph_point(
            window,
            position,
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        click(window, "GRAPHITE-GDD-ADD-Sign");
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_some());
        let node_id = item.read_with(window, |item, cx| {
            assert!(item.is_dirty(cx));
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working graph projection")
            };
            let [node] = form.nodes.as_slice() else {
                panic!("one inserted node")
            };
            assert_eq!(node.display_name(), "Sign");
            assert_eq!(node.grid_position(), Some([3, 2]));
            node.storage_id()
        });
        right_click_saved_graph(window, "GRAPHITE-GDD-GRAPH-NODE-0");
        assert!(window.debug_bounds("GRAPHITE-GDD-ADD-Sign").is_none());
        click(window, "GRAPHITE-GDD-UNDO");
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-SAVED-GRAPH").is_some());
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_none());
        click(window, "GRAPHITE-GDD-REDO");
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_some());

        let destination = PathBuf::from("/project/pointer-position.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("published pointer-positioned graph");
        let nodes = saved.graph_nodes().await.expect("reopened saved root");
        let [node] = nodes.as_slice() else {
            panic!("one saved node")
        };
        assert_eq!(node.storage_id(), node_id);
        assert_eq!(node.grid_position(), Some([3, 2]));
        assert_eq!(
            read_graphite_folder(filesystem.as_ref(), &source)
                .await
                .expect("retained source folder")
                .folder_files()
                .expect("source bytes"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_005_2_workspace_adds_nodes_to_a_blank_working_copy(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/blank-working.gdd");
        window.dispatch_action(NewGraphiteFolder);
        window.run_until_parked();
        window.simulate_new_path_selection(|_| Some(source.clone()));
        window.run_until_parked();
        let item = workspace.read_with(window, |workspace, cx| {
            workspace
                .active_item_as::<GraphiteDocumentItem>(cx)
                .expect("native Graphite folder")
        });
        let original = read_graphite_folder(filesystem.as_ref(), &source)
            .await
            .expect("blank folder")
            .folder_files()
            .expect("original folder entries");
        paint(window);
        click(window, "GRAPHITE-GDD-NODE-COPY");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-ADD-WORKING-NODE");
        let position = item.read_with(window, |item, _| {
            item.node_insert
                .as_ref()
                .expect("working node form")
                .position
                .clone()
        });
        position.update_in(window, |editor, window, cx| {
            editor.set_text("bad", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-CHOOSE-NEW-NODE");
        click(window, "GRAPHITE-GDD-ADD-Sign");
        item.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("two whole grid coordinates")));
        });
        position.update_in(window, |editor, window, cx| {
            editor.set_text("4, -8", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-CHOOSE-NEW-NODE");
        click(window, "GRAPHITE-GDD-ADD-Sign");
        window.run_until_parked();
        let first_id = item.read_with(window, |item, cx| {
            assert!(item.is_dirty(cx));
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("shared node projection")
            };
            let [node] = form.nodes.as_slice() else {
                panic!("expected one working node")
            };
            assert_eq!(node.display_name(), "Sign");
            assert_eq!(node.grid_position(), Some([4, -8]));
            node.storage_id()
        });
        paint(window);
        click(window, "GRAPHITE-GDD-UNDO");
        window.run_until_parked();
        item.read_with(window, |item, _| {
            assert!(matches!(&item.node_copy, NodeCopyPanel::Ready(form) if form.nodes.is_empty()));
        });
        paint(window);
        click(window, "GRAPHITE-GDD-REDO");
        window.run_until_parked();
        item.read_with(window, |item, _| {
            assert!(matches!(&item.node_copy, NodeCopyPanel::Ready(form) if form.nodes.first().is_some_and(|node| node.storage_id() == first_id)));
        });

        position.update_in(window, |editor, window, cx| {
            editor.set_text("-7, 13", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-CHOOSE-NEW-NODE");
        click(window, "GRAPHITE-GDD-ADD-Ceiling");
        window.run_until_parked();
        let destination = PathBuf::from("/project/working-nodes.gdd");
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        item.read_with(window, |item, cx| assert!(!item.is_dirty(cx)));

        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace.read_with(window, |workspace, cx| {
            workspace
                .active_item_as::<GraphiteDocumentItem>(cx)
                .expect("reopened working folder")
        });
        let contents = reopened.read_with(window, |item, cx| {
            item.model
                .read(cx)
                .gdd_contents()
                .expect("opened GDD")
                .clone()
        });
        let nodes = contents.graph_nodes().await.expect("reopened graph");
        assert_eq!(nodes.len(), 2);
        assert!(
            nodes
                .iter()
                .any(|node| node.storage_id() == first_id && node.grid_position() == Some([4, -8]))
        );
        assert!(
            nodes
                .iter()
                .any(|node| node.display_name() == "Ceiling"
                    && node.grid_position() == Some([-7, 13]))
        );
        assert_eq!(
            read_graphite_folder(filesystem.as_ref(), &source)
                .await
                .expect("original folder")
                .folder_files()
                .expect("original entries"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_005_2_workspace_adds_a_pinned_node_to_a_new_folder_copy(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/blank.gdd");
        let destination = PathBuf::from("/project/with-node.gdd");
        window.dispatch_action(NewGraphiteFolder);
        window.run_until_parked();
        window.simulate_new_path_selection(|_| Some(source.clone()));
        window.run_until_parked();
        let source_item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("new Graphite folder");
        let original = read_graphite_folder(filesystem.as_ref(), &source)
            .await
            .expect("original blank folder")
            .folder_files()
            .expect("original bytes");

        paint(window);
        click(window, "GRAPHITE-GDD-ADD-NODE");
        let position = source_item.read_with(window, |item, cx| {
            let form = item.node_insert.as_ref().expect("grid position editor");
            assert_eq!(form.position.read(cx).text(cx), "0, 0");
            form.position.clone()
        });
        position.update_in(window, |editor, window, cx| {
            editor.set_text("-11,", window, cx)
        });
        paint(window);
        click(window, "GRAPHITE-GDD-CHOOSE-NEW-NODE");
        click(window, "GRAPHITE-GDD-ADD-Absolute Value");
        source_item.read_with(window, |item, _| {
            assert!(item.copy_task.is_none());
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error)) if message.contains("two whole grid coordinates")));
        });

        position.update_in(window, |editor, window, cx| {
            editor.set_text("-11, 6", window, cx)
        });
        paint(window);
        for publish in [false, true] {
            click(window, "GRAPHITE-GDD-CHOOSE-NEW-NODE");
            click(window, "GRAPHITE-GDD-ADD-Absolute Value");
            window.simulate_new_path_selection(|_| publish.then(|| destination.clone()));
            window.run_until_parked();
            assert_eq!(
                filesystem
                    .metadata(&destination)
                    .await
                    .expect("separate folder publication")
                    .is_some(),
                publish
            );
            paint(window);
        }
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("open the newly populated Graphite folder");
        assert_ne!(reopened, source_item);
        let contents = reopened.read_with(window, |item, cx| {
            let InspectedDocument::Gdd { contents } = &item.model.read(cx).document else {
                panic!("expected GDD document");
            };
            assert!(!item.is_dirty(cx));
            contents.clone()
        });
        let nodes = contents
            .graph_nodes()
            .await
            .expect("load the embedded source declaration");
        let [node] = nodes.as_slice() else {
            panic!("expected exactly the inserted node");
        };
        assert!(node.label().contains("Absolute Value"));
        assert!(contents.can_move_history(GddHistoryDirection::Undo));
        assert_eq!(
            read_graphite_folder(filesystem.as_ref(), &source)
                .await
                .expect("original folder")
                .folder_files()
                .expect("original bytes"),
            original,
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_004_2_workspace_opens_read_only_graphite_archive(cx: &mut TestAppContext) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let original_path = PathBuf::from("/project/original.gdd.zip");
        let original = archive_bytes(1, ExportFormat::Zip).await;
        filesystem
            .write(&original_path, &original)
            .await
            .expect("write the source fixture");
        window.run_until_parked();
        let handle = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(original_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open through Workspace File Open");
        let item = handle
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite archive view");
        item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            assert_eq!(item.tab_content_text(0, cx).as_ref(), "original.gdd.zip");
            let InspectedDocument::Gdd { contents, .. } = &model.document else {
                panic!("expected the Graphite archive model");
            };
            assert_eq!(contents.manifest().document_id, 171);
            assert_eq!(model.original.as_deref(), Some(original.as_slice()));
            assert!(!item.is_dirty(cx));
            assert!(!item.can_save(cx));
            assert!(!item.can_save_as(cx));
        });

        let future_path = PathBuf::from("/project/future.gdd.zip");
        let future = archive_bytes(2, ExportFormat::Zip).await;
        filesystem
            .write(&future_path, &future)
            .await
            .expect("write a future fixture");
        window.run_until_parked();
        let result = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(future_path.clone(), Default::default(), window, cx)
            })
            .await;
        assert!(matches!(result, Err(error) if format!("{error:#}").contains("version 2")));
        assert_eq!(
            filesystem
                .load_bytes(&future_path)
                .await
                .expect("future file is preserved"),
            future
        );
        assert_eq!(
            filesystem
                .load_bytes(&original_path)
                .await
                .expect("original file is preserved"),
            original
        );
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<GraphiteDocumentItem>(cx)),
            Some(item)
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_005_2_workspace_opens_source_xz_archive(cx: &mut TestAppContext) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/source.gdd.tar.xz");
        let bytes = archive_bytes(1, ExportFormat::Xz).await;
        filesystem
            .write(&path, &bytes)
            .await
            .expect("write actual source XZ export");
        window.run_until_parked();
        let handle = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the XZ archive through Workspace File Open");
        let item = handle
            .downcast::<GraphiteDocumentItem>()
            .expect("native Graphite archive inspection");
        item.read_with(window, |item, cx| {
            assert_eq!(item.tab_content_text(0, cx), "source.gdd.tar.xz");
            let model = item.model.read(cx);
            let InspectedDocument::Gdd { .. } = &model.document else {
                panic!("expected the Graphite archive model");
            };
            assert_eq!(model.original.as_deref(), Some(bytes.as_slice()));
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("source file is intact"),
            bytes
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_003_2_opens_local_folder_and_preserves_its_entries(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let selected = PathBuf::from("/source.gdd");
        let saved_copy = PathBuf::from("/project/copy.gdd");
        let archive = Arc::<[u8]>::from(archive_bytes(1, ExportFormat::Zip).await);
        let mut entries = open_gdd_archive(&archive, ARCHIVE_BUDGET)
            .await
            .expect("source archive")
            .folder_files()
            .expect("literal source entries");
        let opaque = PathBuf::from("opaque/new/value.bin");
        entries.push((opaque.clone(), vec![0, 255, 9]));
        filesystem
            .create_new_directory_atomically(selected.clone(), entries.clone())
            .await
            .expect("source-qualified working folder");
        window.run_until_parked();

        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|options| {
            assert!(!options.files && options.directories && !options.multiple);
            Some(vec![selected.clone()])
        });
        window.run_until_parked();
        let item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("opened native folder item");
        item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            assert!(model.original.is_none());
            let InspectedDocument::Gdd { contents, .. } = &model.document else {
                panic!("expected folder-backed source");
            };
            assert_eq!(contents.manifest().document_id, 171);
            assert_eq!(
                contents
                    .read_entry("opaque/new/value.bin")
                    .expect("retained opaque bytes")
                    .as_slice(),
                [0, 255, 9]
            );
            assert!(!item.is_dirty(cx));
        });

        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| None);
        window.run_until_parked();
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<GraphiteDocumentItem>(cx)),
            Some(item.clone())
        );
        paint(window);
        click(window, "GRAPHITE-SAVE-COPY");
        window.simulate_new_path_selection(|_| Some(saved_copy.clone()));
        window.run_until_parked();
        for (entry, bytes) in entries {
            assert_eq!(
                filesystem
                    .load_bytes(&selected.join(&entry))
                    .await
                    .expect("original folder entry"),
                bytes
            );
            assert_eq!(
                filesystem
                    .load_bytes(&saved_copy.join(entry))
                    .await
                    .expect("preserved folder copy"),
                bytes
            );
        }
    }

    #[gpui::test]
    async fn at_gra_gdd_009_010_012_workspace_selects_archive_contents_without_editing_source(
        cx: &mut TestAppContext,
    ) {
        use graphite_document_format::{Layout, io::path_for};
        use graphite_graph_craft::application_io::resource::ResourceId;

        let mut source = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            172,
            "pinned Graphite".into(),
            "pinned library".into(),
        )
        .expect("original Graphite working copy");
        source
            .add_resource(ResourceId::from(12_u64), b"fixture resource")
            .expect("original Graphite history");
        let codecs = source.manifest().codecs;
        let registry_entry = path_for(GddV1Layout.registry_basename(), codecs.registry);
        let history_entry = path_for(GddV1Layout.history_basename(), codecs.history);
        let original = source
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &HashMapResourceStorage::new(),
                None,
            )
            .await
            .expect("source ZIP with history");

        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/source.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source fixture");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open via native Workspace")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite inspector");
        paint(window);

        let registry_path = PathBuf::from("/project/registry-only.gdd.zip");
        click(window, "GRAPHITE-EXPORT-INCLUDE-HISTORY");
        click(window, "GRAPHITE-EXPORT-GDD-ZIP");
        window.simulate_new_path_selection(|_| Some(registry_path.clone()));
        window.run_until_parked();
        paint(window);
        let registry_bytes = Arc::<[u8]>::from(
            filesystem
                .load_bytes(&registry_path)
                .await
                .expect("registry ZIP saved"),
        );
        let registry_copy = open_gdd_archive(&registry_bytes, ARCHIVE_BUDGET)
            .await
            .expect("registry export reopens");
        assert!(registry_copy.read_entry(&registry_entry).is_ok());
        assert!(registry_copy.read_entry(&history_entry).is_err());

        click(window, "GRAPHITE-EXPORT-INCLUDE-REGISTRY");
        click(window, "GRAPHITE-EXPORT-GDD-XZ");
        assert!(!window.did_prompt_for_new_path());
        item.read_with(window, |item, cx| {
            let (message, color) = item
                .copy_feedback
                .as_ref()
                .expect("invalid selection warning");
            assert_eq!(*color, Color::Error);
            assert!(message.contains("at least one"));
            assert!(!item.is_dirty(cx));
        });

        let history_path = PathBuf::from("/project/history-only.gdd.tar.xz");
        click(window, "GRAPHITE-EXPORT-INCLUDE-HISTORY");
        click(window, "GRAPHITE-EXPORT-GDD-XZ");
        window.simulate_new_path_selection(|_| Some(history_path.clone()));
        window.run_until_parked();
        let history_bytes = Arc::<[u8]>::from(
            filesystem
                .load_bytes(&history_path)
                .await
                .expect("history XZ saved"),
        );
        let history_copy = open_gdd_archive(&history_bytes, ARCHIVE_BUDGET)
            .await
            .expect("history export replays");
        assert!(history_copy.read_entry(&registry_entry).is_err());
        assert!(history_copy.read_entry(&history_entry).is_ok());
        assert_eq!(
            history_copy.history_len(),
            source.session().history().count()
        );
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original retained"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_006_2_workspace_exports_a_folder_with_exact_graphite_payloads(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/source.gdd.zip");
        let destination = PathBuf::from("/project/copy.gdd");
        let existing = PathBuf::from("/project/existing.gdd");
        let original = Arc::<[u8]>::from(archive_bytes(1, ExportFormat::Zip).await);
        let expected = open_gdd_archive(&original, ARCHIVE_BUDGET)
            .await
            .expect("source contents")
            .folder_files()
            .expect("snapshot files");
        filesystem
            .write(&source, &original)
            .await
            .expect("source fixture");
        filesystem
            .create_dir(&existing)
            .await
            .expect("existing directory");
        filesystem
            .write(&existing.join("keep.txt"), b"leave intact")
            .await
            .expect("existing bytes");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the original ZIP via Workspace")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite inspector");
        paint(window);
        click(window, "GRAPHITE-EXPORT-GDD-FOLDER");
        window.simulate_new_path_selection(|_| Some(existing.clone()));
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            assert!(matches!(item.copy_feedback, Some((_, Color::Error))))
        });
        assert_eq!(
            filesystem
                .load_bytes(&existing.join("keep.txt"))
                .await
                .expect("retained existing directory"),
            b"leave intact"
        );
        assert!(
            filesystem
                .metadata(&existing.join("manifest.json"))
                .await
                .expect("inspect preserved folder")
                .is_none()
        );
        click(window, "GRAPHITE-EXPORT-GDD-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        item.read_with(window, |item, cx| {
            assert!(matches!(item.copy_feedback, Some((_, Color::Default))));
            assert!(!item.is_dirty(cx));
        });
        let on_disk = tempfile::tempdir().expect("temporary pinned folder reader fixture");
        for (path, bytes) in expected {
            let exported = filesystem
                .load_bytes(&destination.join(&path))
                .await
                .expect("exported folder entry");
            assert_eq!(exported, bytes, "{}", path.display());
            let target = on_disk.path().join(path);
            std::fs::create_dir_all(target.parent().expect("exported fixture parent"))
                .expect("create pinned fixture directory");
            std::fs::write(target, exported)
                .expect("inspect exact exported bytes with the pinned reader");
        }
        let original_reader = GddV1::open(on_disk.path())
            .await
            .expect("original Graphite folder reader opens the export");
        assert_eq!(original_reader.manifest().document_id, 171);
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("source file remains exact"),
            original.as_ref()
        );
    }

    #[gpui::test]
    async fn at_gra_gdd_007_and_008_workspace_exports_and_reopens_both_archive_formats(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/source.gdd.zip");
        let original = archive_bytes(1, ExportFormat::Zip).await;
        filesystem
            .write(&source, &original)
            .await
            .expect("source fixture");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source ZIP")
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite inspector");
        paint(window);
        click(window, "GRAPHITE-EXPORT-GDD-XZ");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| assert!(item.copy_feedback.is_none()));

        for (selector, filename, format) in [
            (
                "GRAPHITE-EXPORT-GDD-ZIP",
                "export.gdd.zip",
                ArchiveFormat::Zip,
            ),
            (
                "GRAPHITE-EXPORT-GDD-XZ",
                "export.gdd.tar.xz",
                ArchiveFormat::Xz,
            ),
        ] {
            let destination = PathBuf::from("/project").join(filename);
            click(window, selector);
            window.simulate_new_path_selection(|_| Some(destination.clone()));
            window.run_until_parked();
            item.read_with(window, |item, cx| {
                assert!(matches!(item.copy_feedback, Some((_, Color::Default))));
                assert!(!item.is_dirty(cx));
            });
            let exported = filesystem
                .load_bytes(&destination)
                .await
                .expect("saved export");
            assert_eq!(ArchiveFormat::detect(&exported), Some(format));
            window.run_until_parked();
            let copy = workspace
                .update_in(window, |workspace, window, cx| {
                    workspace.open_abs_path(destination, Default::default(), window, cx)
                })
                .await
                .expect("reopen export through normal Workspace File Open")
                .downcast::<GraphiteDocumentItem>()
                .expect("saved Graphite archive");
            copy.read_with(window, |copy, cx| {
                let model = copy.model.read(cx);
                let InspectedDocument::Gdd { contents, .. } = &model.document else {
                    panic!("expected an archive");
                };
                assert_eq!(contents.manifest().document_id, 171);
                assert_eq!(model.original.as_deref(), Some(exported.as_slice()));
            });
            workspace
                .update_in(window, |workspace, window, cx| {
                    workspace.open_abs_path(source.clone(), Default::default(), window, cx)
                })
                .await
                .expect("reactivate original");
            paint(window);
        }
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("source file remains exact"),
            original
        );
    }

    #[gpui::test]
    async fn at_r06_1_workspace_save_copy_preserves_source_and_existing_destination(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/source.gdd.tar.xz");
        let existing = PathBuf::from("/project/existing.gdd.tar.xz");
        let destination = PathBuf::from("/project/new.gdd.tar.xz");
        let original = archive_bytes(1, ExportFormat::Xz).await;
        filesystem.write(&source, &original).await.expect("source");
        filesystem
            .write(&existing, b"existing")
            .await
            .expect("existing destination");
        window.run_until_parked();
        let handle = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source through Workspace File Open");
        let item = handle
            .downcast::<GraphiteDocumentItem>()
            .expect("Graphite inspector");
        paint(window);
        click(window, "GRAPHITE-SAVE-COPY");
        window.simulate_new_path_selection(|directory| {
            assert_eq!(directory, source.parent().expect("source directory"));
            None
        });
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            assert!(item.copy_task.is_none());
            assert!(item.copy_feedback.is_none());
        });

        click(window, "GRAPHITE-SAVE-COPY");
        window.simulate_new_path_selection(|_| Some(existing.clone()));
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            assert!(matches!(item.copy_feedback, Some((_, Color::Error))));
        });
        assert_eq!(
            filesystem
                .load_bytes(&existing)
                .await
                .expect("existing file"),
            b"existing"
        );

        click(window, "GRAPHITE-SAVE-COPY");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        item.read_with(window, |item, cx| {
            assert!(matches!(item.copy_feedback, Some((_, Color::Default))));
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("original source"),
            original
        );
        assert_eq!(
            filesystem
                .load_bytes(&destination)
                .await
                .expect("exact copy"),
            original
        );
        window.run_until_parked();
        let copy = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(destination, Default::default(), window, cx)
            })
            .await
            .expect("reopen the saved copy");
        assert!(copy.downcast::<GraphiteDocumentItem>().is_some());
    }

    #[gpui::test]
    async fn at_r06_1_workspace_inspects_standalone_original_legacy_document(
        cx: &mut TestAppContext,
    ) {
        use graphite_editor::messages::portfolio::document::DocumentMessageHandler;

        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let mut owner = DocumentMessageHandler::default();
        owner.commit_hash = "actual-authoring-build".into();
        let raw = owner.serialize_document().into_bytes();
        let path = PathBuf::from("/project/pinned.graphite");
        filesystem
            .write(&path, &raw)
            .await
            .expect("write original Graphite JSON");
        window.run_until_parked();
        let handle = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open through standard Workspace File Open");
        let item = handle
            .downcast::<GraphiteDocumentItem>()
            .expect("native read-only inspector");
        item.read_with(window, |item, cx| {
            let InspectedDocument::Legacy(document) = &item.model.read(cx).document else {
                panic!("expected legacy document model");
            };
            assert_eq!(document.build_hint(), Some("actual-authoring-build"));
            assert_eq!(document.original_bytes(), raw);
            assert!(!item.is_dirty(cx));
            assert!(!item.can_save(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("legacy source was preserved"),
            raw
        );
    }
}
