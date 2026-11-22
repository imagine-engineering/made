use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, anyhow, ensure};
use editor::Editor;
use gpui::{
    App, AppContext as _, AsyncWindowContext, Context, Entity, EntityId, EventEmitter, FocusHandle,
    Focusable, PathPromptOptions, Render, SharedString, Subscription, Task, WeakEntity, Window,
    actions,
};
use graphics_core::evaluate::UnaryNumericOperation;
use graphics_core::formats::{
    gdd::{
        ArchiveFormat, GddArchiveBudget, GddExportContent, GddGraphNetwork, GddGraphNode,
        GddHistoryDirection, GddNumericValue, GddPinnedNodeMove, GddRecoverySource, GddSavedView,
        InspectedGddV1, create_gdd_folder, inspect_gdd_folder_files, open_gdd_archive,
    },
    legacy::{LegacyDocumentSchema, PreservedLegacyDocument, inspect_legacy_document},
};
use graphics_core::registry::GraphiteNodeCatalog;
use project::{DirectoryLister, Project, ProjectEntryId, ProjectPath, Worktree};
use ui::{
    Button, Checkbox, Color, ContextMenu, DropdownMenu, Label, LabelSize, ToggleState, prelude::*,
};
use workspace::{
    Item, Pane, Workspace, item::ItemBufferKind, notifications::DetachAndPromptErr as _,
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
        false
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
    new_node_position: Option<Entity<Editor>>,
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
    nodes: Vec<GddGraphNode>,
    networks: Vec<GddGraphNetwork>,
    active_network: Option<usize>,
    selected: Option<usize>,
    name: Entity<Editor>,
    value: Entity<Editor>,
    position: Entity<Editor>,
    output_preview: NodeOutputPreview,
}

impl NodeCopyForm {
    fn selected_node(&self) -> Option<&GddGraphNode> {
        self.selected.and_then(|index| self.nodes.get(index))
    }

    fn active_network(&self) -> Option<&GddGraphNetwork> {
        self.active_network
            .and_then(|index| self.networks.get(index))
    }

    fn includes_node(&self, node: &GddGraphNode) -> bool {
        self.active_network()
            .is_none_or(|network| network.id == node.network_id())
    }

    fn select_network(&mut self, index: Option<usize>) {
        self.active_network = index;
        self.selected = None;
        self.output_preview = NodeOutputPreview::Hidden;
    }
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
    EditNode {
        node: Box<GddGraphNode>,
        change: NodeCopyChange,
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
    ConnectNumericInput {
        source: Box<GddGraphNode>,
        previous: Option<Box<GddGraphNode>>,
    },
}

enum PreparedCopy {
    File(Arc<[u8]>),
    Folder(Vec<(PathBuf, Vec<u8>)>),
}

fn parse_graph_grid_position(text: &str) -> anyhow::Result<[i32; 2]> {
    const INVALID_POSITION: &str = "Enter two whole grid coordinates separated by a comma.";
    let (x, y) = text.split_once(',').context(INVALID_POSITION)?;
    Ok([
        x.trim().parse().context(INVALID_POSITION)?,
        y.trim().parse().context(INVALID_POSITION)?,
    ])
}

impl GraphiteDocumentItem {
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

    fn toggle_insert_node(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.new_node_position.take().is_none() {
            self.new_node_position = Some(cx.new(|cx| {
                let mut editor = Editor::single_line(window, cx);
                editor.set_text("0, 0", window, cx);
                editor
            }));
        }
        cx.notify();
    }

    fn save_inserted_node(
        &mut self,
        operation: UnaryNumericOperation,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(editor) = &self.new_node_position else {
            return;
        };
        match parse_graph_grid_position(&editor.read(cx).text(cx)) {
            Ok(position) => self.save_copy(
                GraphiteCopy::InsertNumericNode {
                    operation,
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

    fn render_insert_node(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let label = if self.new_node_position.is_some() {
            "Close Add Node"
        } else {
            "Add Node in a Copy"
        };
        v_flex()
            .gap_2()
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-GDD-ADD-NODE".into())
                    .child(
                        Button::new("graphite-gdd-add-node", label)
                            .disabled(self.copy_task.is_some())
                            .on_click(cx.listener(|item, _, window, cx| {
                                item.toggle_insert_node(window, cx);
                            })),
                    ),
            )
            .when_some(self.new_node_position.clone(), |panel, position| {
                let item = cx.entity().downgrade();
                let menu = ContextMenu::build(window, cx, |mut menu, _, _| {
                    for operation in [
                        UnaryNumericOperation::AbsoluteValue,
                        UnaryNumericOperation::Ceiling,
                        UnaryNumericOperation::Floor,
                        UnaryNumericOperation::Round,
                        UnaryNumericOperation::Sign,
                    ] {
                        let item = item.clone();
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
                            move |window, cx| {
                                if let Some(item) = item.upgrade() {
                                    item.update(cx, |item, cx| {
                                        item.save_inserted_node(operation, window, cx);
                                    });
                                }
                            },
                        );
                    }
                    menu
                });
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
            })
    }

    fn toggle_node_copy(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match &self.node_copy {
            NodeCopyPanel::Loading { .. } => return,
            NodeCopyPanel::Ready(_) => self.node_copy = NodeCopyPanel::Closed,
            NodeCopyPanel::Closed | NodeCopyPanel::Failed(_) => {
                let InspectedDocument::Gdd { contents } = &self.model.read(cx).document else {
                    return;
                };
                let contents = contents.clone();
                let load = cx.background_spawn(async move { contents.graph_inspection().await });
                self.node_copy = NodeCopyPanel::Loading {
                    _task: cx.spawn_in(window, async move |item, cx| {
                        let loaded = load.await;
                        if let Err(error) = item.update_in(cx, |item, window, cx| {
                            item.node_copy = match loaded {
                                Ok(graph) if graph.nodes.is_empty() => {
                                    NodeCopyPanel::Failed("This graph has no nodes.".into())
                                }
                                Ok(graph) => NodeCopyPanel::Ready(Box::new(NodeCopyForm {
                                    nodes: graph.nodes,
                                    networks: graph.networks,
                                    active_network: None,
                                    selected: None,
                                    name: cx.new(|cx| Editor::single_line(window, cx)),
                                    value: cx.new(|cx| Editor::single_line(window, cx)),
                                    position: cx.new(|cx| Editor::single_line(window, cx)),
                                    output_preview: NodeOutputPreview::Hidden,
                                })),
                                Err(error) => NodeCopyPanel::Failed(error.to_string().into()),
                            };
                            cx.notify();
                        }) {
                            log::debug!("Graphite node edit panel closed: {error:#}");
                        }
                    }),
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
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(node) = form.selected_node() else {
            return;
        };
        let Some(original_value) = node.numeric_input() else {
            return;
        };
        match original_value.parse_like(&form.value.read(cx).text(cx)) {
            Ok(value) => {
                let request = GraphiteCopy::EditNode {
                    node: Box::new(node.clone()),
                    change: NodeCopyChange::NumericInput(value),
                };
                self.save_copy(request, window, cx);
            }
            Err(error) => {
                self.copy_feedback = Some((error.to_string().into(), Color::Error));
                cx.notify();
            }
        }
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
            self.save_copy(request, window, cx);
        }
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
                let has_nodes = form.nodes.iter().any(|node| form.includes_node(node));
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
                                    ),
                            )
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
                    .when(!has_nodes, |panel| panel.child(Label::new("This network has no nodes.")))
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
                        let pin_label = if selected.is_pinned() {
                            "Save Unpinned Folder"
                        } else {
                            "Save Pinned Folder"
                        };
                        panel
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
                                            "Save Renamed Folder",
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
                                                    item.save_copy(request, window, cx);
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
                                                    item.save_copy(request, window, cx);
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
                            .when_some(selected.pinned_position(), |panel, (position, count)| {
                                panel
                                    .child(Label::new(format!("Pinned order · {position} of {count}")))
                                    .child(h_flex().gap_2().children(
                                        [
                                            (
                                                GddPinnedNodeMove::Up,
                                                "GRAPHITE-GDD-MOVE-PIN-UP",
                                                "Move Up in a Copy",
                                            ),
                                            (
                                                GddPinnedNodeMove::Down,
                                                "GRAPHITE-GDD-MOVE-PIN-DOWN",
                                                "Move Down in a Copy",
                                            ),
                                        ]
                                        .into_iter()
                                        .map(|(movement, selector, label)| {
                                            div()
                                                .debug_selector(move || selector.into())
                                                .child(
                                                    Button::new(selector, label)
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
                                                                    item.save_copy(request, window, cx);
                                                                }
                                                            },
                                                        )),
                                                )
                                        }),
                                    ))
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
                                                "Disconnect Value in a Copy",
                                            )
                                            .disabled(self.copy_task.is_some())
                                            .on_click(cx.listener(|item, _, window, cx| {
                                                item.save_disconnected_numeric_input(window, cx);
                                            })),
                                        ),
                                )
                            })
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
                let name = model.name.as_ref();
                let lower_name = name.to_ascii_lowercase();
                let stem = [".gdd.tar.xz", ".gdd.zip", ".gdd"]
                    .into_iter()
                    .find(|suffix| lower_name.ends_with(suffix))
                    .and_then(|suffix| name.get(..name.len() - suffix.len()))
                    .unwrap_or(name);
                let suffix = match format {
                    GddCopyFormat::Archive(ArchiveFormat::Zip) => ".gdd.zip",
                    GddCopyFormat::Archive(ArchiveFormat::Xz) => ".gdd.tar.xz",
                    GddCopyFormat::Folder => ".gdd",
                };
                let prefix = match &request {
                    GraphiteCopy::Original | GraphiteCopy::Export(_) => "Copy of",
                    GraphiteCopy::RecoverFolder => "Recovered",
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
            let result: anyhow::Result<Option<PathBuf>> = async {
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
                            GraphiteCopy::EditNode { node, change } => Some(match change {
                                NodeCopyChange::Rename(name) => {
                                    archive
                                        .rename_graph_node_in_copy(&node, name, ARCHIVE_BUDGET)
                                        .await?
                                }
                                NodeCopyChange::DuplicateNumeric => {
                                    archive
                                        .duplicate_graph_numeric_node_in_copy(&node, ARCHIVE_BUDGET)
                                        .await?
                                }
                                NodeCopyChange::DeleteIsolated => {
                                    archive
                                        .delete_isolated_graph_node_in_copy(&node, ARCHIVE_BUDGET)
                                        .await?
                                }
                                NodeCopyChange::GridPosition(position) => {
                                    archive
                                        .move_graph_node_in_copy(&node, position, ARCHIVE_BUDGET)
                                        .await?
                                }
                                NodeCopyChange::Pin(pinned) => {
                                    archive
                                        .set_graph_node_pinned_in_copy(
                                            &node,
                                            pinned,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                NodeCopyChange::MovePinned(movement) => {
                                    archive
                                        .move_pinned_graph_node_in_copy(
                                            &node,
                                            movement,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                NodeCopyChange::NumericInput(value) => {
                                    archive
                                        .set_graph_numeric_input_in_copy(
                                            &node,
                                            value,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                NodeCopyChange::NumericInputExposure(exposed) => {
                                    archive
                                        .set_graph_numeric_input_exposed_in_copy(
                                            &node,
                                            exposed,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                NodeCopyChange::DisconnectNumericInput => {
                                    archive
                                        .disconnect_graph_numeric_input_in_copy(
                                            &node,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                NodeCopyChange::ConnectNumericInput {
                                    source,
                                    previous: Some(previous),
                                } => {
                                    archive
                                        .reconnect_graph_numeric_input_in_copy(
                                            &node,
                                            &previous,
                                            &source,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                                NodeCopyChange::ConnectNumericInput {
                                    source,
                                    previous: None,
                                } => {
                                    archive
                                        .connect_graph_numeric_input_in_copy(
                                            &node,
                                            &source,
                                            ARCHIVE_BUDGET,
                                        )
                                        .await?
                                }
                            }),
                        };
                        if let Some(copy) = revision {
                            return Ok(PreparedCopy::Folder(copy.folder_files()?));
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
                            GddCopyFormat::Folder => PreparedCopy::Folder(
                                contents
                                    .export_folder_files_with_content(content, ARCHIVE_BUDGET)
                                    .await?,
                            ),
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
                match copy {
                    PreparedCopy::File(bytes) => {
                        filesystem
                            .create_new_file_atomically(path.clone(), bytes)
                            .await
                    }
                    PreparedCopy::Folder(files) => {
                        filesystem
                            .create_new_directory_atomically(path.clone(), files)
                            .await
                    }
                }
                .with_context(|| format!("Could not save a copy at {}", path.display()))?;
                Ok(Some(path))
            }
            .await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                item.copy_task = None;
                item.copy_feedback = match result {
                    Ok(Some(path)) => Some((
                        format!("Copy saved to {}", path.display()).into(),
                        Color::Default,
                    )),
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
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            model: item,
            project: project.downgrade(),
            focus_handle: cx.focus_handle(),
            copy_task: None,
            copy_feedback: None,
            export_content: GddExportContent::default(),
            embed_linked_resources: false,
            embed_legacy_document: false,
            saved_view: SavedViewPanel::Closed,
            node_catalog: NodeCatalogPanel::Closed,
            node_copy: NodeCopyPanel::Closed,
            new_node_position: None,
        }
    }
}

impl Focusable for GraphiteDocumentItem {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<()> for GraphiteDocumentItem {}

impl Item for GraphiteDocumentItem {
    type Event = ();

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

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>> {
        let model = self.model.clone();
        let project = self.project.clone();
        Task::ready(Some(cx.new(|cx| Self {
            model,
            project,
            focus_handle: cx.focus_handle(),
            copy_task: None,
            copy_feedback: None,
            export_content: self.export_content,
            embed_linked_resources: self.embed_linked_resources,
            embed_legacy_document: self.embed_legacy_document,
            saved_view: SavedViewPanel::Closed,
            node_catalog: NodeCatalogPanel::Closed,
            node_copy: NodeCopyPanel::Closed,
            new_node_position: None,
        })))
    }
}

impl Render for GraphiteDocumentItem {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let model = self.model.read(cx);
        let view = v_flex()
            .id("graphite-document-inspector")
            .track_focus(&self.focus_handle)
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
                view.child(Label::new(format!(
                    "{label} v{} · Read-only",
                    manifest.format_version
                )))
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
                .when(
                    recovery.node_count > 0 && recovery.hot_log_len == 0,
                    |view| view.child(self.render_node_copy(window, cx)),
                )
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
                                    "Save Undo Folder",
                                    "GRAPHITE-GDD-SAVE-UNDO",
                                    can_undo,
                                ),
                                (
                                    GddHistoryDirection::Redo,
                                    "Save Redo Folder",
                                    "GRAPHITE-GDD-SAVE-REDO",
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
                                            item.save_copy(
                                                GraphiteCopy::MoveHistory(direction),
                                                window,
                                                cx,
                                            );
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

    fn choose_nested_copy_node(window: &mut gpui::VisualTestContext) {
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
    }

    enum NumericGraphFixture {
        SingleNode,
        ExposedInput,
        Connection,
        Reconnection,
    }

    async fn numeric_graph_archive(fixture: NumericGraphFixture) -> (Vec<u8>, usize) {
        use graphite_document_graph_storage::NoMetadata;
        use graphite_graph_craft::document::{
            DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
            value::TaggedValue,
        };
        use graphite_resource::ResourceRegistry;
        use graphite_std::math_nodes::absolute_value;

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
            NumericGraphFixture::ExposedInput => {
                Some(NodeInput::value(TaggedValue::Integer(0), true))
            }
            NumericGraphFixture::Connection | NumericGraphFixture::Reconnection => {
                Some(NodeInput::node(NodeId(7), 0))
            }
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
        if matches!(fixture, NumericGraphFixture::Reconnection) {
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
            .commit_from_runtime(&graph, &NoMetadata, &ResourceRegistry::new(), &resources)
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
        use graphite_document_graph_storage::NodeMetadataSource;
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
            fn display_name(&self, _: &[NodeId], local_id: NodeId) -> Option<&str> {
                (local_id == NodeId(7)).then_some("Repeated value")
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
                5
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
        click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-6");
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
                .debug_bounds("GRAPHITE-GDD-NODE-COPY-ITEM-6")
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
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::ExposedInput).await;
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
        for publish in [false, true] {
            click(window, "GRAPHITE-GDD-CONNECT-VALUE");
            assert!(
                window
                    .debug_bounds("GRAPHITE-GDD-CONNECT-SOURCE-1")
                    .is_none()
            );
            click(window, "GRAPHITE-GDD-CONNECT-SOURCE-0");
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
        assert!(connected.has_numeric_connection());
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
    async fn at_gra_graph_008_2_workspace_reconnects_a_value_in_a_new_folder(
        cx: &mut TestAppContext,
    ) {
        let (original, original_history) =
            numeric_graph_archive(NumericGraphFixture::Reconnection).await;
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
        for publish in [false, true] {
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
        assert_eq!(current_source.label(), replacement.label());
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
            let position = item
                .new_node_position
                .as_ref()
                .expect("grid position editor");
            assert_eq!(position.read(cx).text(cx), "0, 0");
            position.clone()
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
