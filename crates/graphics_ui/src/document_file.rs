use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    future::Future,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Weak},
    time::Instant,
};

use anyhow::{Context as _, anyhow, ensure};
use editor::Editor;
use gpui::{
    App, AppContext as _, AsyncWindowContext, ClipboardItem, Context, Entity, EntityId,
    EventEmitter, FocusHandle, Focusable, ImageSource, PathPromptOptions, Render, RenderImage,
    ScrollHandle, SharedString, Subscription, SvgRenderer, Task, WeakEntity, Window, actions, img,
};
use graphics_core::evaluate::{CpuSvgEvaluationBudget, UnaryNumericOperation};
use graphics_core::formats::{
    gdd::{
        ArchiveFormat, GddArchiveBudget, GddArchiveError, GddArtboardCanvasLayer,
        GddArtboardCanvasPick, GddArtboardGeometry, GddEllipseAxis, GddExportContent,
        GddGraphGeometry, GddGraphInspection, GddGraphLayerPreview, GddGraphNetwork, GddGraphNode,
        GddGraphPortSource, GddHistoryDirection, GddNumericGraphEvaluator, GddNumericImportKind,
        GddNumericValue, GddPinnedNodeMove, GddRecoverySource, GddRootLayer, GddRootLayerChange,
        GddSavedView, ImportOrExport, InspectedGddV1, create_gdd_folder, inspect_gdd_folder_files,
        open_gdd_archive,
    },
    legacy::{LegacyDocumentSchema, PreservedLegacyDocument, inspect_legacy_document},
};
use graphics_core::registry::{CUSTOM_NODE_IDENTIFIER, CUSTOM_NODE_NAME, GraphiteNodeCatalog};
use graphite_editor::messages::portfolio::document::{
    DocumentMessageHandler,
    overlays::utility_types::OverlaysVisibilitySettings,
    utility_types::misc::{PTZ, SnappingState},
};
use graphite_graph_craft::document::value::{DAffine2, DVec2, RenderOutputType, UVec2};
use graphite_std::{
    application_io::{ExportFormat as RenderFormat, RenderConfig},
    transform::Footprint,
};
use project::{DirectoryLister, Project, ProjectEntryId, ProjectPath, Worktree};
use ui::{
    Button, Checkbox, Color, ContextMenu, DropdownMenu, IconPosition, Label, LabelSize, Switch,
    SwitchLabelPosition, ToggleState, prelude::*,
};
use workspace::{
    Item, Pane, Workspace,
    item::{ItemBufferKind, ItemEvent, SaveOptions},
    notifications::DetachAndPromptErr as _,
};

use crate::canvas::{
    ArtboardDragSnap, ArtboardDraw, ArtboardMove, ArtboardOverlay, ArtboardResize,
    CANVAS_OVERLAY_OPTIONS, CANVAS_OVERLAYS_ALL, CANVAS_SNAP_ENABLED, CanvasOverlayFlag,
    CanvasSnapFlag, DocumentNavigation, artboard_auto_pan_delta, artboard_outline,
    canvas_snap_groups, logical_vector, paint_artboard_outline, set_artboard_snap_camera,
};
use crate::canvas_grid::{CanvasGrid, GridOptionsForm};
use crate::canvas_overlay::CanvasOverlay;
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
        /// Cancels the current Artboard Canvas pan.
        CancelArtboardCanvasPan,
        /// Cancels the current Artboard gesture on the Canvas.
        CancelArtboardCanvasEdit,
        /// Toggles snapping for the active Graphite Canvas.
        ToggleCanvasSnapping,
        /// Toggles the document grid for the active Graphite Canvas.
        ToggleCanvasGrid,
        /// Toggles all overlays for the active Graphite Canvas.
        ToggleCanvasOverlays,
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

const LAYER_PREVIEW_SIZE: u32 = 128;
const SAVED_SCENE_PIXEL_BUDGET: u32 = 16 * 1024 * 1024;

async fn preview_layer_image(
    contents: &InspectedGddV1,
    node: &GddGraphNode,
    preview: GddGraphLayerPreview,
    renderer: &SvgRenderer,
) -> Result<Arc<RenderImage>, String> {
    let resolution = UVec2::splat(LAYER_PREVIEW_SIZE);
    let center = resolution.as_dvec2() / 2.;
    let transform = match preview {
        GddGraphLayerPreview::Merge => DAffine2::from_translation(center),
        GddGraphLayerPreview::Artboard { origin, size } => {
            let scale = f64::from(LAYER_PREVIEW_SIZE) / size.max_element();
            DAffine2::from_scale_angle_translation(
                DVec2::splat(scale),
                0.,
                center - (origin * scale + size * (scale / 2.)),
            )
        }
    };
    let configuration = RenderConfig {
        viewport: Footprint {
            transform,
            resolution,
            ..Footprint::DEFAULT
        },
        scale: 1.,
        export_format: RenderFormat::Svg,
        for_export: true,
        ..Default::default()
    };
    let budget = CpuSvgEvaluationBudget {
        max_viewport_pixels: LAYER_PREVIEW_SIZE * LAYER_PREVIEW_SIZE,
        max_output_bytes: 64 * 1024,
    };
    let output = contents
        .render_document_layer_svg(node, configuration, budget)
        .await
        .map_err(|error| error.to_string())?;
    let RenderOutputType::Svg { svg, .. } = output.data else {
        return Err("Graphite did not return an SVG preview.".to_owned());
    };
    crate::render::prepare_svg_bytes_at_target(
        svg.as_bytes(),
        resolution.x,
        resolution.y,
        configuration.scale,
        renderer,
        (LAYER_PREVIEW_SIZE * LAYER_PREVIEW_SIZE * 4) as usize,
    )
    .map_err(|error| error.to_string())
}

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
    snapping_preferences: Option<Arc<SnappingState>>,
    overlay_preferences: Option<OverlaysVisibilitySettings>,
    grid_visible: Option<bool>,
    grid_options_edited: bool,
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

    fn picked_artboard(
        &self,
        pick: &GddArtboardCanvasPick,
    ) -> Option<(&GddGraphNode, GddArtboardGeometry)> {
        let node = self
            .inspection
            .nodes
            .iter()
            .find(|node| pick.matches_node(node))?;
        Some((node, node.artboard_geometry()?))
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
                snapping_preferences: None,
                overlay_preferences: None,
                grid_visible: None,
                grid_options_edited: false,
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
            snapping_preferences: None,
            overlay_preferences: None,
            grid_visible: None,
            grid_options_edited: false,
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
    scene_canvas: Option<SavedSceneCanvas>,
    node_catalog: NodeCatalogPanel,
    node_copy: NodeCopyPanel,
    node_insert: Option<NodeInsertForm>,
}

struct SavedSceneCanvas {
    instance: Arc<()>,
    revision: Arc<()>,
    target: Option<SavedSceneTarget>,
    camera_override: Option<PTZ>,
    navigation: Option<DocumentNavigation>,
    snapping_document: Option<Box<DocumentMessageHandler>>,
    applied_snapping_preferences: Option<Arc<SnappingState>>,
    source_grid_visible: bool,
    grid_options: Option<GridOptionsForm>,
    grid_frame: Option<SavedSceneGridFrame>,
    grid_task: Option<Task<()>>,
    pan: Option<SavedScenePan>,
    pick_armed: bool,
    move_armed: bool,
    resize_armed: bool,
    draw_armed: bool,
    artboard_draw: Option<SavedSceneArtboardDraw>,
    artboard_drag: Option<SavedSceneArtboardDrag>,
    artboard_view: Option<SavedSceneArtboardView>,
    artboards: Vec<SavedSceneArtboard>,
    can_draw_artboard: bool,
    frame: SavedSceneFrame,
}

struct SavedSceneGridFrame {
    request: CanvasGrid,
    image: Result<Option<Arc<RenderImage>>, SharedString>,
}

struct SavedScenePan {
    start: gpui::Point<gpui::Pixels>,
    original_camera: DocumentNavigation,
    original_override: Option<PTZ>,
}

struct SavedSceneArtboardDraw {
    gesture: ArtboardDraw,
    placement: Option<GddArtboardGeometry>,
}

struct SavedSceneArtboardDrag {
    node: GddGraphNode,
    artboard_index: usize,
    gesture: SavedSceneArtboardGesture,
    snapping: ArtboardDragSnap,
    outline_offset: DVec2,
    placement: Option<GddArtboardGeometry>,
}

struct SavedSceneArtboardView {
    identity: Arc<()>,
    pane: gpui::Bounds<gpui::Pixels>,
    initial: DocumentNavigation,
    navigation: DocumentNavigation,
    pointer: Option<(gpui::Point<gpui::Pixels>, gpui::Modifiers)>,
    auto_pan_since: Option<Instant>,
    frame_queued: bool,
    snap_guides: Option<Arc<CanvasOverlay>>,
    snap_guide_error: Option<SharedString>,
    draft_frame: Option<SavedSceneArtboardDraft>,
    draft_task: Option<Task<()>>,
    draft_error: Option<SharedString>,
}

impl SavedSceneArtboardView {
    fn new(pane: gpui::Bounds<gpui::Pixels>, navigation: DocumentNavigation) -> Self {
        Self {
            identity: Arc::new(()),
            pane,
            initial: navigation,
            navigation,
            pointer: None,
            auto_pan_since: None,
            frame_queued: false,
            snap_guides: None,
            snap_guide_error: None,
            draft_frame: None,
            draft_task: None,
            draft_error: None,
        }
    }

    fn offset_from(&self, target: SavedSceneTarget, camera: PTZ) -> DVec2 {
        let from = DocumentNavigation {
            camera,
            ..self.initial
        };
        self.navigation
            .document_to_pane(target.logical_size())
            .translation
            - from.document_to_pane(target.logical_size()).translation
    }

    fn retire_draft(&mut self, window: Option<&mut Window>, cx: &mut App) {
        if let Some(draft) = self.draft_frame.take() {
            cx.drop_image(draft.image, window);
        }
    }
}

enum SavedSceneArtboardGesture {
    Move(ArtboardMove),
    Resize(ArtboardResize, Arc<GddArtboardCanvasPick>),
}

impl SavedSceneArtboardGesture {
    fn placement(
        &self,
        position: DVec2,
        modifiers: gpui::Modifiers,
        document: &DocumentMessageHandler,
        snapping: &mut ArtboardDragSnap,
    ) -> Option<(GddArtboardGeometry, DVec2)> {
        match self {
            Self::Move(gesture) => gesture.placement(position, modifiers, document, snapping),
            Self::Resize(gesture, _) => gesture
                .placement(position, modifiers, document, snapping)
                .map(|geometry| (geometry, DVec2::ZERO)),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct SavedSceneArtboardRender {
    placement: Option<GddArtboardGeometry>,
    camera: PTZ,
}

struct SavedSceneArtboardDraft {
    request: SavedSceneArtboardRender,
    image: Arc<RenderImage>,
    artboards: Vec<SavedSceneArtboard>,
}

struct SavedSceneArtboard {
    pick_target: Option<GddArtboardCanvasPick>,
    overlay: Option<Arc<ArtboardOverlay>>,
}

impl From<GddArtboardCanvasLayer> for SavedSceneArtboard {
    fn from(layer: GddArtboardCanvasLayer) -> Self {
        Self {
            pick_target: layer.pick_target,
            overlay: layer
                .overlay
                .map(|overlay| Arc::new(ArtboardOverlay::prepare(overlay))),
        }
    }
}

impl SavedSceneCanvas {
    fn overlay_visibility(
        &self,
        model: &GraphiteDocumentProjectItem,
    ) -> Option<OverlaysVisibilitySettings> {
        let document = self.snapping_document.as_ref()?;
        Some(
            model
                .overlay_preferences
                .unwrap_or(document.overlays_visibility_settings),
        )
    }

    fn grid_request(&self, model: &GraphiteDocumentProjectItem) -> Option<CanvasGrid> {
        if !model.grid_visible.unwrap_or(self.source_grid_visible)
            || !self.overlay_visibility(model)?.all()
        {
            return None;
        }
        self.grid_viewport()
    }

    fn grid_viewport(&self) -> Option<CanvasGrid> {
        let document = self.snapping_document.as_ref()?;
        let target = self.target?;
        let navigation = self
            .artboard_view
            .as_ref()
            .map(|view| view.navigation)
            .or_else(|| self.drawing_navigation())
            .or(self.navigation)?;
        Some(CanvasGrid {
            resolution: target.resolution,
            scale: target.scale,
            camera: navigation.camera,
            options: document.snapping_state.grid.clone(),
        })
    }

    fn retire_grid_frame(&mut self, window: Option<&mut Window>, cx: &mut App) {
        if let Some(frame) = self.grid_frame.take()
            && let Ok(Some(image)) = frame.image
        {
            cx.drop_image(image, window);
        }
    }

    fn apply_snapping_preferences(&mut self, preferences: &Arc<SnappingState>) -> bool {
        let Some(document) = &mut self.snapping_document else {
            return false;
        };
        if self
            .applied_snapping_preferences
            .as_ref()
            .is_some_and(|applied| Arc::ptr_eq(applied, preferences))
        {
            return false;
        }
        document.snapping_state = (**preferences).clone();
        self.applied_snapping_preferences = Some(preferences.clone());
        if let Some(drag) = &mut self.artboard_drag {
            drag.snapping.refresh_candidates(document);
        }
        true
    }

    fn drawing_navigation(&self) -> Option<DocumentNavigation> {
        match self.frame {
            SavedSceneFrame::Blank => Some(DocumentNavigation {
                camera: self.camera_override.unwrap_or_default(),
                document_size: DVec2::ZERO,
            }),
            SavedSceneFrame::Ready(_) if self.can_draw_artboard => self.navigation,
            _ => None,
        }
    }

    fn artboard_render_request(&self) -> Option<SavedSceneArtboardRender> {
        let view = self.artboard_view.as_ref()?;
        let placement = self.artboard_drag.as_ref().and_then(|drag| drag.placement);
        let camera = view.navigation.camera;
        (placement.is_some()
            || (camera != view.initial.camera && matches!(self.frame, SavedSceneFrame::Ready(_))))
        .then_some(SavedSceneArtboardRender { placement, camera })
    }

    fn take_artboard_view(
        &mut self,
        window: Option<&mut Window>,
        cx: &mut App,
    ) -> Option<SavedSceneArtboardView> {
        let mut view = self.artboard_view.take()?;
        view.retire_draft(window, cx);
        view.draft_task = None;
        Some(view)
    }

    fn clear_artboard_edit(&mut self, window: Option<&mut Window>, cx: &mut App) {
        self.artboard_draw = None;
        self.artboard_drag = None;
        drop(self.take_artboard_view(window, cx));
    }

    fn retire(self, window: Option<&mut Window>, cx: &mut App) {
        let mut canvas = self;
        let mut window = window;
        canvas.clear_artboard_edit(window.as_deref_mut(), cx);
        canvas.retire_grid_frame(window.as_deref_mut(), cx);
        if let SavedSceneFrame::Ready(image) = canvas.frame {
            cx.drop_image(image, window);
        }
    }
}

enum SavedSceneFrame {
    Empty,
    Blank,
    Pending { request: Arc<()>, _task: Task<()> },
    Ready(Arc<RenderImage>),
    Failed(SharedString),
}

#[derive(Clone, Copy, PartialEq)]
struct SavedSceneTarget {
    resolution: UVec2,
    scale: f64,
}

impl SavedSceneTarget {
    fn logical_size(self) -> DVec2 {
        self.resolution.as_dvec2() / self.scale
    }

    fn render_image(
        self,
        output: graphite_graph_craft::document::value::RenderOutput,
        renderer: &SvgRenderer,
    ) -> Result<Arc<RenderImage>, String> {
        let RenderOutputType::Svg { svg, .. } = output.data else {
            return Err("Graphite did not return an SVG Canvas frame.".to_owned());
        };
        crate::render::prepare_svg_bytes_at_target(
            svg.as_bytes(),
            self.resolution.x,
            self.resolution.y,
            self.scale,
            renderer,
            SAVED_SCENE_PIXEL_BUDGET as usize * 4,
        )
        .map_err(|error| error.to_string())
    }

    fn measure(size: gpui::Size<gpui::Pixels>, scale: f32) -> Option<Self> {
        let scale = f64::from(scale);
        let resolution = (DVec2::new(
            f64::from(f32::from(size.width)),
            f64::from(f32::from(size.height)),
        ) * scale)
            .round();
        if scale <= 0. || !resolution.is_finite() || resolution.min_element() < 1. {
            return None;
        }
        Some(Self {
            resolution: resolution.as_uvec2(),
            scale,
        })
    }
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

#[derive(Clone, Copy)]
enum GraphiteInsertNode {
    Numeric(UnaryNumericOperation),
    Custom,
}

impl GraphiteInsertNode {
    fn name(self) -> &'static str {
        match self {
            Self::Numeric(operation) => operation.name(),
            Self::Custom => CUSTOM_NODE_NAME,
        }
    }
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
    artboard_background: Entity<Editor>,
    ellipse_radii: [Entity<Editor>; 2],
    output_preview: NodeOutputPreview,
    document_run: NodeOutputPreview,
    document_run_evaluator: Option<Arc<GddNumericGraphEvaluator>>,
    port_rename: Option<PortRenameForm>,
    node_selection: BTreeSet<usize>,
    layer_selection_anchor: Option<usize>,
    collapsed_layers: BTreeSet<usize>,
    show_saved_ports: bool,
    saved_graph: SavedGraphPanel,
    saved_graph_viewport: ScrollHandle,
    saved_graph_focus: FocusHandle,
    saved_layers_focus: FocusHandle,
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
            artboard_background: cx.new(|cx| Editor::single_line(window, cx)),
            ellipse_radii: std::array::from_fn(|_| cx.new(|cx| Editor::single_line(window, cx))),
            output_preview: NodeOutputPreview::Hidden,
            document_run: NodeOutputPreview::Hidden,
            document_run_evaluator: None,
            port_rename: None,
            node_selection: BTreeSet::new(),
            layer_selection_anchor: None,
            collapsed_layers: BTreeSet::new(),
            show_saved_ports: false,
            saved_graph: SavedGraphPanel::Closed,
            saved_graph_viewport: ScrollHandle::new(),
            saved_graph_focus: cx.focus_handle(),
            saved_layers_focus: cx.focus_handle(),
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
        let layer_anchor = self
            .layer_selection_anchor
            .and_then(|index| self.nodes.get(index))
            .map(GddGraphNode::storage_id);
        let group = self
            .node_selection
            .iter()
            .filter_map(|index| self.nodes.get(*index))
            .map(GddGraphNode::storage_id)
            .collect::<BTreeSet<_>>();
        let collapsed = self
            .collapsed_layers
            .iter()
            .filter_map(|&index| self.nodes.get(index))
            .map(GddGraphNode::storage_id)
            .collect::<BTreeSet<_>>();
        self.snapshot = snapshot;
        self.revision = revision;
        self.document_run = NodeOutputPreview::Hidden;
        self.document_run_evaluator = None;
        self.nodes.clone_from(&inspection.nodes);
        self.networks.clone_from(&inspection.networks);
        let active_network =
            network_id.and_then(|id| self.networks.iter().position(|network| network.id == id));
        self.select_network(active_network);
        self.layer_selection_anchor = layer_anchor.and_then(|id| {
            self.nodes
                .iter()
                .position(|node| self.includes_node(node) && node.storage_id() == id)
        });
        self.node_selection = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| self.includes_node(node) && group.contains(&node.storage_id()))
            .map(|(index, _)| index)
            .collect();
        self.collapsed_layers = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| collapsed.contains(&node.storage_id()))
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

    fn saved_layer_rows(&self) -> Option<Vec<(usize, &GddRootLayer)>> {
        let SavedGraphPanel::Ready(graph) = &self.saved_graph else {
            return None;
        };
        let layers = graph.root_layers.as_ref().ok()?;
        let scope = self
            .clipboard_scope()
            .filter(|network| network.parent.is_none())?
            .id;
        let nodes_by_id = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.network_id() == scope)
            .map(|(index, node)| (node.storage_id(), index))
            .collect::<BTreeMap<_, _>>();
        Some(
            layers
                .iter()
                .filter_map(|layer| Some((*nodes_by_id.get(&layer.storage_id)?, layer)))
                .collect(),
        )
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
        if self.active_network != index {
            self.saved_graph_viewport = ScrollHandle::new();
        }
        self.active_network = index;
        self.selected = None;
        self.output_preview = NodeOutputPreview::Hidden;
        self.port_rename = None;
        self.node_selection.clear();
        self.layer_selection_anchor = None;
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
    Pending { request: Arc<()>, _task: Task<()> },
    Cancelled,
    Ready(GddNumericValue),
    ReadyImage(Arc<RenderImage>),
    Failed(SharedString),
}

impl NodeOutputPreview {
    fn is_pending_request(&self, request: &Arc<()>) -> bool {
        matches!(self, Self::Pending { request: current, .. } if Arc::ptr_eq(current, request))
    }

    fn feedback(&self) -> Option<(String, Color)> {
        match self {
            Self::Ready(GddNumericValue::NumberList(values)) if values.is_empty() => {
                Some(("[]".into(), Color::Default))
            }
            Self::Ready(value) => Some((value.to_string(), Color::Default)),
            Self::Failed(error) => Some((error.to_string(), Color::Error)),
            Self::Cancelled => Some(("Graph run canceled.".into(), Color::Muted)),
            Self::Hidden | Self::Pending { .. } | Self::ReadyImage(_) => None,
        }
    }
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
    ExportArtboardSvg(Box<GddGraphNode>),
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
    AppendNumericImport {
        network: Box<GddGraphNetwork>,
        kind: GddNumericImportKind,
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
    BooleanInput(bool),
    ArtboardGeometry(GddArtboardGeometry),
    ArtboardBackground([f32; 4]),
    EllipseRadius(GddEllipseAxis, f64),
    EllipseRadiusSource(GddEllipseAxis, Option<Box<GddGraphNode>>),
    SetDocumentArtboard,
    ConnectGraphicContent(Box<GddGraphNode>),
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
            Self::BooleanInput(value) => {
                archive
                    .set_graph_boolean_input_in_copy(node, value, ARCHIVE_BUDGET)
                    .await
            }
            Self::ArtboardGeometry(geometry) => {
                archive
                    .set_artboard_geometry_in_copy(node, geometry, ARCHIVE_BUDGET)
                    .await
            }
            Self::ArtboardBackground(channels) => {
                archive
                    .set_artboard_background_in_copy(node, channels, ARCHIVE_BUDGET)
                    .await
            }
            Self::EllipseRadius(axis, radius) => {
                archive
                    .set_ellipse_radius_in_copy(node, axis, radius, ARCHIVE_BUDGET)
                    .await
            }
            Self::EllipseRadiusSource(axis, source) => {
                archive
                    .set_ellipse_radius_source_in_copy(
                        node,
                        axis,
                        source.as_deref(),
                        ARCHIVE_BUDGET,
                    )
                    .await
            }
            Self::SetDocumentArtboard => {
                archive
                    .set_document_artboard_in_copy(node, ARCHIVE_BUDGET)
                    .await
            }
            Self::ConnectGraphicContent(source) => {
                archive
                    .connect_graphic_content_in_copy(node, &source, ARCHIVE_BUDGET)
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
            if let Some(target) = item.scene_canvas.as_ref().and_then(|canvas| canvas.target) {
                item.update_scene_canvas(Some(target), false, window, cx);
                item.refresh_scene_snapping(window, cx);
            }
            cx.emit(ItemEvent::UpdateTab);
            cx.notify();
        });
        cx.observe_window_activation(window, |item, window, cx| {
            if !window.is_window_active() {
                item.finish_scene_pan(true, window, cx);
                item.cancel_scene_artboard_edit(window, cx);
            }
        })
        .detach();
        cx.on_release(|item, cx| {
            if let Some(canvas) = item.scene_canvas.take() {
                canvas.retire(None, cx);
            }
        })
        .detach();
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
            scene_canvas: None,
            node_catalog: NodeCatalogPanel::Closed,
            node_copy: NodeCopyPanel::Closed,
            node_insert: None,
        }
    }

    fn open_scene_canvas(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let model = self.model.read(cx);
        if self.scene_canvas.is_some()
            || model
                .gdd_contents()
                .is_none_or(|contents| contents.hot_log_len() != 0)
        {
            return;
        }
        self.scene_canvas = Some(SavedSceneCanvas {
            instance: Arc::new(()),
            revision: model.revision.clone(),
            target: None,
            camera_override: None,
            navigation: None,
            snapping_document: None,
            applied_snapping_preferences: None,
            source_grid_visible: false,
            grid_options: None,
            grid_frame: None,
            grid_task: None,
            pan: None,
            pick_armed: false,
            move_armed: false,
            resize_armed: false,
            draw_armed: false,
            artboard_draw: None,
            artboard_drag: None,
            artboard_view: None,
            artboards: Vec::new(),
            can_draw_artboard: false,
            frame: SavedSceneFrame::Empty,
        });
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    fn close_scene_canvas(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(canvas) = self.scene_canvas.take() {
            canvas.retire(Some(window), cx);
        }
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    fn toggle_scene_snapping(
        &self,
        instance: &Arc<()>,
        revision: &Arc<()>,
        flag: CanvasSnapFlag,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(canvas) = self.scene_canvas.as_ref().filter(|canvas| {
            Arc::ptr_eq(instance, &canvas.instance) && Arc::ptr_eq(revision, &canvas.revision)
        }) else {
            return false;
        };
        let Some(document) = &canvas.snapping_document else {
            return false;
        };
        let model = self.model.read(cx);
        if !Arc::ptr_eq(revision, &model.revision) {
            return false;
        }
        let mut preferences = model
            .snapping_preferences
            .as_deref()
            .unwrap_or(&document.snapping_state)
            .clone();
        let value = flag(&mut preferences);
        *value = !*value;
        self.model.update(cx, |model, cx| {
            model.snapping_preferences = Some(Arc::new(preferences));
            cx.notify();
        });
        true
    }

    fn refresh_scene_snapping(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let model = self.model.read(cx);
        let Some(preferences) = &model.snapping_preferences else {
            return;
        };
        if Arc::ptr_eq(&canvas.revision, &model.revision)
            && canvas.apply_snapping_preferences(preferences)
            && let Some((position, modifiers)) =
                canvas.artboard_view.as_ref().and_then(|view| view.pointer)
            && self.update_scene_artboard_placement(position, modifiers, window, cx)
        {
            self.render_scene_artboard_draft(window, cx);
        }
    }

    fn toggle_scene_grid(
        &self,
        instance: &Arc<()>,
        revision: &Arc<()>,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(canvas) = self.scene_canvas.as_ref().filter(|canvas| {
            Arc::ptr_eq(instance, &canvas.instance)
                && Arc::ptr_eq(revision, &canvas.revision)
                && canvas.snapping_document.is_some()
        }) else {
            return false;
        };
        let model = self.model.read(cx);
        if !Arc::ptr_eq(revision, &model.revision) {
            return false;
        }
        let visible = model.grid_visible.unwrap_or(canvas.source_grid_visible);
        self.model.update(cx, |model, cx| {
            model.grid_visible = Some(!visible);
            cx.notify();
        });
        true
    }

    fn toggle_scene_overlay(
        &self,
        instance: &Arc<()>,
        revision: &Arc<()>,
        flag: CanvasOverlayFlag,
        cx: &mut Context<Self>,
    ) -> bool {
        let model = self.model.read(cx);
        let Some(mut visibility) = self
            .scene_canvas
            .as_ref()
            .filter(|canvas| {
                Arc::ptr_eq(instance, &canvas.instance)
                    && Arc::ptr_eq(revision, &canvas.revision)
                    && Arc::ptr_eq(revision, &model.revision)
            })
            .and_then(|canvas| canvas.overlay_visibility(model))
        else {
            return false;
        };
        let visible = flag(&mut visibility);
        *visible = !*visible;
        self.model.update(cx, |model, cx| {
            model.overlay_preferences = Some(visibility);
            cx.notify();
        });
        true
    }

    fn open_scene_grid_options(
        &mut self,
        instance: &Arc<()>,
        revision: &Arc<()>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = self.scene_canvas.as_mut().filter(|canvas| {
            Arc::ptr_eq(instance, &canvas.instance) && Arc::ptr_eq(revision, &canvas.revision)
        }) else {
            return;
        };
        let model = self.model.read(cx);
        if !Arc::ptr_eq(revision, &model.revision) {
            return;
        }
        if canvas.grid_options.take().is_some() {
            self.focus_handle.focus(window, cx);
            cx.notify();
            return;
        }
        let Some(document) = &canvas.snapping_document else {
            return;
        };
        let preferences = model
            .snapping_preferences
            .as_deref()
            .unwrap_or(&document.snapping_state);
        let form = GridOptionsForm::new(
            preferences.grid.clone(),
            canvas.revision.clone(),
            window,
            cx,
        );
        form.focus(window, cx);
        canvas.grid_options = Some(form);
        cx.notify();
    }

    fn apply_scene_grid_options(
        &mut self,
        id: EntityId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some(form) = canvas.grid_options.as_ref().filter(|form| form.id() == id) else {
            return;
        };
        let result = (|| {
            let model = self.model.read(cx);
            ensure!(
                Arc::ptr_eq(&form.revision, &model.revision),
                "The document changed. Reopen Grid Options."
            );
            let document = canvas
                .snapping_document
                .as_ref()
                .context("Wait for the Canvas to render.")?;
            let current = model
                .snapping_preferences
                .as_deref()
                .unwrap_or(&document.snapping_state);
            ensure!(
                current.grid == form.baseline,
                "Grid options changed in another view. Reopen Grid Options."
            );
            let mut grid = canvas
                .grid_viewport()
                .context("Wait for the Canvas to render.")?;
            grid.options = form.options(cx)?;
            grid.validate()?;
            let mut preferences = current.clone();
            preferences.grid = grid.options;
            anyhow::Ok(preferences)
        })();
        match result {
            Ok(preferences) => {
                if preferences.grid != form.baseline {
                    self.model.update(cx, |model, cx| {
                        model.grid_options_edited = true;
                        model.snapping_preferences = Some(Arc::new(preferences));
                        cx.notify();
                    });
                }
                canvas.grid_options = None;
                self.focus_handle.focus(window, cx);
            }
            Err(error) => {
                if let Some(form) = &mut canvas.grid_options {
                    form.error = Some(error.to_string().into());
                }
            }
        }
        cx.notify();
    }

    fn render_scene_grid(
        &mut self,
        instance: &Arc<()>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let model = self.model.read(cx);
        if !Arc::ptr_eq(&canvas.instance, instance)
            || !Arc::ptr_eq(&canvas.revision, &model.revision)
        {
            return;
        }
        let Some(request) = canvas.grid_request(model) else {
            canvas.retire_grid_frame(Some(window), cx);
            return;
        };
        if canvas.grid_task.is_some()
            || canvas
                .grid_frame
                .as_ref()
                .is_some_and(|frame| frame.request == request)
        {
            return;
        }
        let render = cx.background_spawn({
            let request = request.clone();
            async move {
                request.render(SAVED_SCENE_PIXEL_BUDGET).map_err(|error| {
                    SharedString::from(format!("Unable to render the grid. {error:#}"))
                })
            }
        });
        let instance = instance.clone();
        canvas.grid_task = Some(cx.spawn_in(window, async move |item, cx| {
            let image = render.await;
            if let Err(error) = item.update_in(cx, |item, window, cx| {
                let Some(canvas) = &mut item.scene_canvas else {
                    return;
                };
                if !Arc::ptr_eq(&canvas.instance, &instance) {
                    return;
                }
                canvas.grid_task = None;
                let model = item.model.read(cx);
                if Arc::ptr_eq(&canvas.revision, &model.revision)
                    && canvas.grid_request(model).as_ref() == Some(&request)
                {
                    canvas.retire_grid_frame(Some(window), cx);
                    canvas.grid_frame = Some(SavedSceneGridFrame { request, image });
                }
                cx.notify();
            }) {
                log::debug!("Graphite Canvas closed before rendering the grid: {error:#}");
            }
        }));
    }

    fn update_scene_canvas(
        &mut self,
        target: Option<SavedSceneTarget>,
        retry: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let model = self.model.read(cx);
        if !retry && canvas.target == target && Arc::ptr_eq(&canvas.revision, &model.revision) {
            return;
        }
        let revision = model.revision.clone();
        let contents = model.gdd_contents().cloned();
        let snapping_preferences = model.snapping_preferences.clone();
        let working_copy = model.working.is_some();
        let blank_working_graph = model
            .working
            .as_ref()
            .is_some_and(|working| working.inspection.document_output_is_empty());
        let revision_changed = !Arc::ptr_eq(&canvas.revision, &revision);
        let target_changed = canvas.target != target;
        if revision_changed || target_changed {
            canvas.clear_artboard_edit(Some(window), cx);
        }
        if !working_copy {
            canvas.draw_armed = false;
        }
        if (revision_changed || target_changed)
            && let Some(pan) = canvas.pan.take()
        {
            canvas.camera_override = pan.original_override;
            canvas.navigation = None;
        }
        if revision_changed || (target_changed && canvas.camera_override.is_none()) {
            canvas.navigation = None;
        }
        canvas.revision = revision.clone();
        canvas.target = target;
        canvas.artboards.clear();
        canvas.snapping_document = None;
        canvas.applied_snapping_preferences = None;
        canvas.can_draw_artboard = false;
        let camera = canvas.camera_override;
        if let SavedSceneFrame::Ready(image) =
            std::mem::replace(&mut canvas.frame, SavedSceneFrame::Empty)
        {
            cx.drop_image(image, Some(window));
        }
        let Some((target, contents)) = target.zip(contents) else {
            cx.notify();
            return;
        };
        if target.resolution.max_element() > 8192
            || u64::from(target.resolution.x) * u64::from(target.resolution.y)
                > u64::from(SAVED_SCENE_PIXEL_BUDGET)
        {
            canvas.frame = SavedSceneFrame::Failed(
                "The Canvas exceeds its pixel budget. Resize the pane to try again.".into(),
            );
            cx.notify();
            return;
        }
        if blank_working_graph {
            let document = contents.canvas_snapping_defaults();
            canvas.source_grid_visible = document.snapping_state.grid_snapping;
            canvas.snapping_document = Some(document);
            if let Some(preferences) = &snapping_preferences {
                canvas.apply_snapping_preferences(preferences);
            }
            canvas.frame = SavedSceneFrame::Blank;
            cx.notify();
            return;
        }
        let renderer = cx.svg_renderer();
        let render = cx.background_spawn(async move {
            let output = futures::executor::block_on(contents.render_document_canvas_svg(
                target.resolution,
                target.scale,
                camera,
                CpuSvgEvaluationBudget {
                    max_viewport_pixels: SAVED_SCENE_PIXEL_BUDGET,
                    max_output_bytes: 64 * 1024,
                },
            ))
            .map_err(|error| match error {
                GddArchiveError::UnsupportedGraphLayerPreview => {
                    "Connect supported artwork or an Artboard to the document output to view the Canvas."
                        .to_owned()
                }
                error => error.to_string(),
            })?;
            let navigation = DocumentNavigation {
                camera: output.camera,
                document_size: output.document_size,
            };
            let image = target.render_image(output.render, &renderer)?;
            let artboards = output
                .artboards
                .into_iter()
                .map(SavedSceneArtboard::from)
                .collect();
            Ok::<_, String>((image, navigation, artboards, output.can_draw_artboard, output.snapping_document))
        });
        let request = Arc::new(());
        canvas.frame = SavedSceneFrame::Pending {
            request: request.clone(),
            _task: cx.spawn(async move |item, cx| {
                let result = render.await;
                if let Err(error) = item.update(cx, |item, cx| {
                    let Some(canvas) = &mut item.scene_canvas else {
                        return;
                    };
                    let model = item.model.read(cx);
                    if !Arc::ptr_eq(&revision, &model.revision)
                        || !matches!(&canvas.frame, SavedSceneFrame::Pending { request: pending, .. } if Arc::ptr_eq(pending, &request))
                    {
                        return;
                    }
                    canvas.frame = match result {
                        Ok((image, navigation, artboards, can_draw_artboard, snapping_document)) => {
                            canvas.navigation = Some(navigation);
                            canvas.artboards = artboards;
                            canvas.can_draw_artboard = can_draw_artboard;
                            canvas.source_grid_visible = snapping_document.as_ref()
                                .is_some_and(|document| document.snapping_state.grid_snapping);
                            canvas.snapping_document = snapping_document;
                            if let Some(preferences) = &model.snapping_preferences {
                                canvas.apply_snapping_preferences(preferences);
                            }
                            SavedSceneFrame::Ready(image)
                        }
                        Err(error) => SavedSceneFrame::Failed(error.into()),
                    };
                    cx.notify();
                }) {
                    log::debug!("Graphite Canvas closed: {error:#}");
                }
            }),
        };
        cx.notify();
    }

    fn scroll_scene_canvas(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        event: &gpui::ScrollWheelEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some((target, navigation)) = canvas.target.zip(canvas.navigation.as_mut()) else {
            return;
        };
        if !Arc::ptr_eq(instance, &canvas.instance)
            || !Arc::ptr_eq(&canvas.revision, &self.model.read(cx).revision)
            || SavedSceneTarget::measure(pane.size, window.scale_factor()) != Some(target)
            || canvas.pan.is_some()
            || canvas.artboard_draw.is_some()
            || canvas.artboard_drag.is_some()
        {
            return;
        }
        let Some(camera) =
            navigation.scroll(target.logical_size(), pane, event, window.line_height())
        else {
            return;
        };
        navigation.camera = camera;
        canvas.camera_override = Some(camera);
        cx.stop_propagation();
        self.update_scene_canvas(Some(target), true, window, cx);
    }

    fn start_scene_pan(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        event: &gpui::MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some(navigation) = canvas.navigation else {
            return;
        };
        if !Arc::ptr_eq(instance, &canvas.instance)
            || !Arc::ptr_eq(&canvas.revision, &self.model.read(cx).revision)
            || SavedSceneTarget::measure(pane.size, window.scale_factor()) != canvas.target
            || !pane.contains(&event.position)
            || event.modifiers.alt
            || event.modifiers.control
            || canvas.pan.is_some()
            || canvas.artboard_draw.is_some()
            || canvas.artboard_drag.is_some()
        {
            return;
        }
        canvas.pan = Some(SavedScenePan {
            start: event.position,
            original_camera: navigation,
            original_override: canvas.camera_override,
        });
        self.focus_handle.focus(window, cx);
        cx.stop_propagation();
        cx.notify();
    }

    fn move_scene_pan(
        &mut self,
        instance: &Arc<()>,
        event: &gpui::MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some((target, pan)) = canvas.target.zip(canvas.pan.as_ref()) else {
            return;
        };
        if !Arc::ptr_eq(instance, &canvas.instance)
            || !Arc::ptr_eq(&canvas.revision, &self.model.read(cx).revision)
        {
            return;
        }
        if event.pressed_button != Some(gpui::MouseButton::Middle) {
            self.finish_scene_pan(true, window, cx);
            return;
        }
        let Some(camera) = pan
            .original_camera
            .pan(target.logical_size(), event.position - pan.start)
        else {
            return;
        };
        let camera_override = if camera == pan.original_camera.camera {
            pan.original_override
        } else {
            Some(camera)
        };
        cx.stop_propagation();
        if camera_override != canvas.camera_override {
            canvas.camera_override = camera_override;
            if let Some(navigation) = &mut canvas.navigation {
                navigation.camera = camera;
            }
            self.update_scene_canvas(Some(target), true, window, cx);
        }
    }

    fn finish_scene_pan(&mut self, abort: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some(pan) = canvas.pan.take() else {
            return;
        };
        if abort && canvas.camera_override != pan.original_override {
            canvas.camera_override = pan.original_override;
            canvas.navigation = Some(pan.original_camera);
            let target = canvas.target;
            self.update_scene_canvas(target, true, window, cx);
        } else {
            cx.notify();
        }
    }

    fn fit_scene_canvas(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(canvas) = &mut self.scene_canvas
            && canvas.camera_override.take().is_some()
        {
            canvas.pan = None;
            canvas.clear_artboard_edit(Some(window), cx);
            canvas.navigation = None;
            let target = canvas.target;
            self.update_scene_canvas(target, true, window, cx);
        }
    }

    fn start_scene_artboard_draw(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        event: &gpui::MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some((target, navigation)) = canvas.target.zip(canvas.drawing_navigation()) else {
            return;
        };
        let model = self.model.read(cx);
        if !canvas.draw_armed
            || canvas.pan.is_some()
            || canvas.artboard_view.is_some()
            || model.working.is_none()
            || self.copy_task.is_some()
            || !Arc::ptr_eq(&canvas.instance, instance)
            || !Arc::ptr_eq(&canvas.revision, &model.revision)
            || SavedSceneTarget::measure(pane.size, window.scale_factor()) != Some(target)
            || !pane.contains(&event.position)
        {
            return;
        }
        let position = event.position - pane.origin;
        let point = logical_vector(position);
        if canvas
            .artboards
            .iter()
            .filter_map(|artboard| artboard.pick_target.as_ref())
            .any(|pick| pick.contains(point))
        {
            return;
        }
        let Some(document) = &mut canvas.snapping_document else {
            return;
        };
        set_artboard_snap_camera(document, navigation, target.logical_size());
        canvas.artboard_draw = Some(SavedSceneArtboardDraw {
            gesture: ArtboardDraw::new(point, document, target.logical_size()),
            placement: None,
        });
        canvas.artboard_view = Some(SavedSceneArtboardView::new(pane, navigation));
        self.focus_handle.focus(window, cx);
        cx.stop_propagation();
        cx.notify();
    }

    fn finish_scene_artboard_draw(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        event: &gpui::MouseUpEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        if !Arc::ptr_eq(&canvas.instance, instance) {
            return;
        }
        let Some(mut draw) = canvas.artboard_draw.take() else {
            return;
        };
        let Some((view, target)) = canvas
            .take_artboard_view(Some(window), cx)
            .zip(canvas.target)
        else {
            return;
        };
        cx.stop_propagation();
        cx.notify();
        let model = self.model.read(cx);
        if !Arc::ptr_eq(&canvas.revision, &model.revision)
            || SavedSceneTarget::measure(pane.size, window.scale_factor()) != Some(target)
            || pane != view.pane
            || canvas
                .drawing_navigation()
                .is_none_or(|navigation| navigation.camera != view.initial.camera)
            || model.working.is_none()
            || self.copy_task.is_some()
        {
            return;
        }
        let contents = model.gdd_contents().cloned();
        let revision = model.revision.clone();
        let placement = canvas.snapping_document.as_mut().and_then(|document| {
            set_artboard_snap_camera(document, view.navigation, target.logical_size());
            draw.gesture.placement(
                logical_vector(event.position - pane.origin),
                event.modifiers,
                document,
            )
        });
        self.retain_artboard_camera(&view, window, cx);
        let Some((contents, placement)) = contents.zip(placement) else {
            return;
        };
        if let Some(canvas) = &mut self.scene_canvas {
            canvas.camera_override = Some(view.navigation.camera);
        }
        self.stage_working_snapshot(
            revision,
            async move {
                contents
                    .create_document_artboard_in_copy(placement, ARCHIVE_BUDGET)
                    .await
            },
            window,
            cx,
        );
    }

    fn start_scene_artboard_drag(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        event: &gpui::MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let model = self.model.read(cx);
        let Some((target, navigation)) = canvas.target.zip(canvas.navigation) else {
            return;
        };
        if !(canvas.move_armed || canvas.resize_armed)
            || canvas.pan.is_some()
            || canvas.artboard_view.is_some()
            || self.copy_task.is_some()
            || !Arc::ptr_eq(&canvas.instance, instance)
            || !Arc::ptr_eq(&canvas.revision, &model.revision)
            || SavedSceneTarget::measure(pane.size, window.scale_factor()) != Some(target)
            || !pane.contains(&event.position)
        {
            return;
        }
        let Some(document) = &mut canvas.snapping_document else {
            return;
        };
        let viewport_size = target.logical_size();
        set_artboard_snap_camera(document, navigation, viewport_size);
        let pane_point = logical_vector(event.position - pane.origin);
        let document_to_pane = document.metadata().document_to_viewport;
        let Some((artboard_index, node, gesture, pick)) = canvas
            .artboards
            .iter()
            .enumerate()
            .find_map(|(index, board)| {
                let pick = board.pick_target.as_ref()?;
                let (node, geometry) = model.working.as_ref()?.picked_artboard(pick)?;
                let gesture = if canvas.resize_armed {
                    let resize = ArtboardResize::at_edge(
                        pick.resize_bounds()?,
                        pane_point,
                        document_to_pane,
                    )?;
                    SavedSceneArtboardGesture::Resize(resize, Arc::new(pick.clone()))
                } else if pick.contains(pane_point) {
                    SavedSceneArtboardGesture::Move(ArtboardMove::new(
                        pane_point,
                        geometry,
                        document_to_pane,
                    ))
                } else {
                    return None;
                };
                Some((index, node, gesture, pick))
            })
        else {
            return;
        };
        canvas.artboard_drag = Some(SavedSceneArtboardDrag {
            node: node.clone(),
            artboard_index,
            gesture,
            snapping: ArtboardDragSnap::new(
                pick.source_layer(),
                pick.resize_bounds(),
                document,
                viewport_size,
            ),
            outline_offset: DVec2::ZERO,
            placement: None,
        });
        canvas.artboard_view = Some(SavedSceneArtboardView::new(pane, navigation));
        self.focus_handle.focus(window, cx);
        cx.stop_propagation();
        cx.notify();
    }

    fn move_scene_artboard(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        event: &gpui::MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        if !Arc::ptr_eq(&canvas.instance, instance)
            || !Arc::ptr_eq(&canvas.revision, &self.model.read(cx).revision)
        {
            return;
        }
        let Some((view, target)) = canvas.artboard_view.as_mut().zip(canvas.target) else {
            return;
        };
        if event.pressed_button != Some(gpui::MouseButton::Left) || pane != view.pane {
            self.cancel_scene_artboard_edit(window, cx);
            return;
        }
        let point = logical_vector(event.position - pane.origin);
        view.pointer = point
            .is_finite()
            .then_some((event.position, event.modifiers));
        if point.is_finite()
            && (point.min_element() < 0. || point.cmpgt(target.logical_size()).any())
        {
            view.auto_pan_since
                .get_or_insert_with(|| cx.background_executor().now());
        } else {
            view.auto_pan_since = None;
        }
        Self::schedule_scene_artboard_pan(view, window, cx);
        if self.update_scene_artboard_placement(event.position, event.modifiers, window, cx) {
            self.render_scene_artboard_draft(window, cx);
            cx.notify();
        }
        cx.stop_propagation();
    }

    fn schedule_scene_artboard_pan(
        view: &mut SavedSceneArtboardView,
        window: &Window,
        cx: &Context<Self>,
    ) {
        if view.frame_queued || view.auto_pan_since.is_none() {
            return;
        }
        view.frame_queued = true;
        let identity = view.identity.clone();
        let item = cx.weak_entity();
        window.on_next_frame(move |window, cx| {
            if let Err(error) = item.update(cx, |item, cx| {
                item.advance_scene_artboard_pan(&identity, window, cx);
            }) {
                log::debug!("Graphite Canvas closed during Artboard auto-pan: {error:#}");
            }
        });
    }

    fn advance_scene_artboard_pan(
        &mut self,
        identity: &Arc<()>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some((view, target)) = canvas.artboard_view.as_mut().zip(canvas.target) else {
            return;
        };
        if !Arc::ptr_eq(identity, &view.identity)
            || !Arc::ptr_eq(&canvas.revision, &self.model.read(cx).revision)
        {
            return;
        }
        view.frame_queued = false;
        let Some(((position, modifiers), previous)) = view.pointer.zip(view.auto_pan_since) else {
            return;
        };
        if SavedSceneTarget::measure(view.pane.size, window.scale_factor()) != Some(target) {
            self.cancel_scene_artboard_edit(window, cx);
            return;
        }
        let now = cx.background_executor().now();
        view.auto_pan_since = Some(now);
        let viewport = target.logical_size();
        let delta = artboard_auto_pan_delta(
            viewport,
            position - view.pane.origin,
            now.saturating_duration_since(previous),
        );
        Self::schedule_scene_artboard_pan(view, window, cx);
        if let Some(camera) = delta.and_then(|delta| view.navigation.pan_by(viewport, delta)) {
            view.navigation.camera = camera;
            view.draft_error = None;
            self.update_scene_artboard_placement(position, modifiers, window, cx);
            self.render_scene_artboard_draft(window, cx);
            cx.notify();
        }
    }

    fn update_scene_artboard_placement(
        &mut self,
        position: gpui::Point<gpui::Pixels>,
        modifiers: gpui::Modifiers,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(canvas) = &mut self.scene_canvas else {
            return false;
        };
        let Some((view, target)) = canvas.artboard_view.as_mut().zip(canvas.target) else {
            return false;
        };
        let Some(document) = &mut canvas.snapping_document else {
            return false;
        };
        set_artboard_snap_camera(document, view.navigation, target.logical_size());
        let position = logical_vector(position - view.pane.origin);
        let (placement_changed, manager) = if let Some(draw) = &mut canvas.artboard_draw {
            let placement = draw.gesture.placement(position, modifiers, document);
            let changed = draw.placement != placement;
            draw.placement = placement;
            (changed, draw.gesture.snap_manager())
        } else if let Some(drag) = &mut canvas.artboard_drag {
            let placement =
                drag.gesture
                    .placement(position, modifiers, document, &mut drag.snapping);
            drag.outline_offset = placement.map_or(DVec2::ZERO, |(_, offset)| offset);
            let desired = placement
                .map(|(geometry, _)| geometry)
                .filter(|geometry| drag.node.artboard_geometry() != Some(*geometry));
            let changed = desired != drag.placement;
            if changed {
                drag.placement = desired;
                view.draft_error = None;
                if desired.is_none() {
                    view.retire_draft(Some(window), cx);
                }
            }
            (changed, drag.snapping.snap_manager())
        } else {
            return false;
        };
        let (guides, error) = match CanvasOverlay::for_gesture(
            manager,
            document,
            target.logical_size(),
            target.scale,
        ) {
            Ok(guides) => (guides, None),
            Err(error) => (
                None,
                Some(SharedString::from(format!(
                    "Unable to render snap guides. {error:#}"
                ))),
            ),
        };
        let guides_changed =
            guides.as_ref() != view.snap_guides.as_deref() || error != view.snap_guide_error;
        if guides_changed {
            view.snap_guides = guides.map(Arc::new);
            view.snap_guide_error = error;
        }
        placement_changed || guides_changed
    }

    fn render_scene_artboard_draft(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        let Some((target, request)) = canvas.target.zip(canvas.artboard_render_request()) else {
            return;
        };
        let Some(view) = canvas.artboard_view.as_mut() else {
            return;
        };
        if view.draft_task.is_some()
            || view
                .draft_frame
                .as_ref()
                .is_some_and(|frame| frame.request == request)
        {
            return;
        }
        let model = self.model.read(cx);
        let Some(contents) = model.gdd_contents().cloned() else {
            return;
        };
        if !Arc::ptr_eq(&canvas.revision, &model.revision) {
            return;
        }
        let revision = canvas.revision.clone();
        let identity = view.identity.clone();
        let edit = canvas
            .artboard_drag
            .as_ref()
            .zip(request.placement)
            .map(|(drag, placement)| {
                let resize = match &drag.gesture {
                    SavedSceneArtboardGesture::Move(_) => None,
                    SavedSceneArtboardGesture::Resize(_, pick) => Some(pick.clone()),
                };
                (drag.node.clone(), placement, resize)
            });
        let renderer = cx.svg_renderer();
        let render = cx.background_spawn(async move {
            let budget = CpuSvgEvaluationBudget {
                max_viewport_pixels: SAVED_SCENE_PIXEL_BUDGET,
                max_output_bytes: 64 * 1024,
            };
            let viewport = (target.resolution, target.scale);
            let output = futures::executor::block_on(async {
                match edit {
                    Some((node, placement, Some(pick))) => {
                        contents
                            .render_document_artboard_resize_draft_svg(
                                &node,
                                &pick,
                                placement,
                                viewport,
                                request.camera,
                                budget,
                            )
                            .await
                    }
                    Some((node, placement, None)) => {
                        contents
                            .render_document_artboard_move_draft_svg(
                                &node,
                                placement.location,
                                viewport,
                                request.camera,
                                budget,
                            )
                            .await
                    }
                    None => {
                        contents
                            .render_document_canvas_svg(
                                target.resolution,
                                target.scale,
                                Some(request.camera),
                                budget,
                            )
                            .await
                    }
                }
            })
            .map_err(|error| error.to_string())?;
            let image = target.render_image(output.render, &renderer)?;
            Ok::<_, String>(SavedSceneArtboardDraft {
                request,
                image,
                artboards: output
                    .artboards
                    .into_iter()
                    .map(SavedSceneArtboard::from)
                    .collect(),
            })
        });
        view.draft_task = Some(cx.spawn_in(window, async move |item, cx| {
            let result = render.await;
            if let Err(error) = item.update_in(cx, |item, window, cx| {
                let Some(canvas) = &mut item.scene_canvas else {
                    return;
                };
                if !Arc::ptr_eq(&revision, &item.model.read(cx).revision) {
                    return;
                }
                let desired = canvas.artboard_render_request();
                let Some(view) = &mut canvas.artboard_view else {
                    return;
                };
                if !Arc::ptr_eq(&identity, &view.identity) {
                    return;
                }
                view.draft_task = None;
                if desired.is_some_and(|desired| desired.placement == request.placement) {
                    match result {
                        Ok(frame) => {
                            view.retire_draft(Some(window), cx);
                            view.draft_frame = Some(frame);
                        }
                        Err(error) if desired == Some(request) => {
                            view.draft_error = Some(SharedString::from(error));
                        }
                        Err(_) => {}
                    }
                    cx.notify();
                }
                if desired != Some(request) {
                    item.render_scene_artboard_draft(window, cx);
                }
            }) {
                log::debug!("Graphite Canvas closed during draft rendering: {error:#}");
            }
        }));
    }

    fn finish_scene_artboard_drag(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        event: &gpui::MouseUpEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        if !Arc::ptr_eq(&canvas.instance, instance) {
            return;
        }
        let Some(mut artboard_drag) = canvas.artboard_drag.take() else {
            return;
        };
        let Some((view, target)) = canvas
            .take_artboard_view(Some(window), cx)
            .zip(canvas.target)
        else {
            return;
        };
        cx.stop_propagation();
        cx.notify();
        let model = self.model.read(cx);
        if !Arc::ptr_eq(&canvas.revision, &model.revision)
            || view.pane != pane
            || SavedSceneTarget::measure(pane.size, window.scale_factor()) != Some(target)
            || model.working.is_none()
            || self.copy_task.is_some()
        {
            return;
        }
        let contents = model.gdd_contents().cloned();
        let revision = model.revision.clone();
        let placement = canvas.snapping_document.as_mut().and_then(|document| {
            set_artboard_snap_camera(document, view.navigation, target.logical_size());
            artboard_drag.gesture.placement(
                logical_vector(event.position - pane.origin),
                event.modifiers,
                document,
                &mut artboard_drag.snapping,
            )
        });
        self.retain_artboard_camera(&view, window, cx);
        let Some((placement, _)) = placement else {
            return;
        };
        if artboard_drag.node.artboard_geometry() == Some(placement) {
            return;
        }
        if let Some(canvas) = &mut self.scene_canvas {
            canvas.camera_override = Some(view.navigation.camera);
        }
        if let SavedSceneArtboardGesture::Resize(_, pick) = artboard_drag.gesture {
            let Some(contents) = contents else {
                return;
            };
            self.stage_working_snapshot(
                revision,
                async move {
                    contents
                        .resize_document_artboard_in_copy(
                            &artboard_drag.node,
                            &pick,
                            placement,
                            ARCHIVE_BUDGET,
                        )
                        .await
                },
                window,
                cx,
            );
            return;
        }
        self.apply_graph_edit(
            GraphiteCopy::EditNode {
                node: Box::new(artboard_drag.node),
                change: NodeCopyChange::ArtboardGeometry(placement),
            },
            window,
            cx,
        );
    }

    fn cancel_scene_artboard_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if let Some(canvas) = &mut self.scene_canvas
            && let Some(view) = canvas.take_artboard_view(Some(window), cx)
        {
            canvas.artboard_draw = None;
            canvas.artboard_drag = None;
            self.retain_artboard_camera(&view, window, cx);
            cx.notify();
            return true;
        }
        false
    }

    fn retain_artboard_camera(
        &mut self,
        view: &SavedSceneArtboardView,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if view.navigation.camera != view.initial.camera
            && let Some(canvas) = &mut self.scene_canvas
        {
            canvas.camera_override = Some(view.navigation.camera);
            let target = canvas.target;
            self.update_scene_canvas(target, true, window, cx);
        }
    }

    fn pick_scene_artboard(
        &mut self,
        instance: &Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        position: gpui::Point<gpui::Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas) = &mut self.scene_canvas else {
            return;
        };
        if !canvas.pick_armed
            || canvas.pan.is_some()
            || !Arc::ptr_eq(&canvas.instance, instance)
            || !Arc::ptr_eq(&canvas.revision, &self.model.read(cx).revision)
            || canvas.target != SavedSceneTarget::measure(pane.size, window.scale_factor())
            || !pane.contains(&position)
        {
            return;
        }
        let point = position - pane.origin;
        let pane_point = DVec2::new(f64::from(f32::from(point.x)), f64::from(f32::from(point.y)));
        if let Some(pick) = canvas
            .artboards
            .iter_mut()
            .find_map(|board| board.pick_target.take_if(|pick| pick.contains(pane_point)))
        {
            self.close_scene_canvas(window, cx);
            self.show_node_copy(Some(pick), window, cx);
        }
    }

    fn render_scene_view_controls(
        &self,
        canvas: &SavedSceneCanvas,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<gpui::Div> {
        let model = self.model.read(cx);
        if !Arc::ptr_eq(&canvas.revision, &model.revision) {
            return None;
        }
        let document = canvas.snapping_document.as_ref()?;
        let mut state = document.snapping_state.clone();
        let enabled = state.snapping_enabled;
        let instance = canvas.instance.clone();
        let revision = canvas.revision.clone();
        let grid_visible = model.grid_visible.unwrap_or(canvas.source_grid_visible);
        let grid_instance = instance.clone();
        let grid_revision = revision.clone();
        let item = cx.weak_entity();
        let menu = ContextMenu::build(window, cx, |mut menu, _, _| {
            let grid_item = item.clone();
            let options_instance = instance.clone();
            let options_revision = revision.clone();
            menu = menu.entry("Grid Options", None, move |window, cx| {
                if let Err(error) = grid_item.update(cx, |item, cx| {
                    item.open_scene_grid_options(&options_instance, &options_revision, window, cx);
                }) {
                    log::debug!("Graphite Canvas closed before opening grid options: {error:#}");
                }
            });
            for (heading, options) in canvas_snap_groups() {
                menu = menu.header(heading);
                for &(label, flag, _) in options {
                    let item = item.clone();
                    let instance = instance.clone();
                    let revision = revision.clone();
                    menu = menu.toggleable_entry(
                        label,
                        *flag(&mut state),
                        IconPosition::Start,
                        None,
                        move |_, cx| {
                            if let Err(error) = item.update(cx, |item, cx| {
                                item.toggle_scene_snapping(&instance, &revision, flag, cx);
                            }) {
                                log::debug!(
                                    "Graphite Canvas closed before changing snapping: {error:#}"
                                );
                            }
                        },
                    );
                }
            }
            menu
        });
        Some(
            h_flex()
                .gap_1()
                .child(
                    div()
                        .debug_selector(|| "GRAPHITE-GDD-CANVAS-GRID".into())
                        .child(
                            Button::new(
                                "graphite-gdd-canvas-grid",
                                if grid_visible {
                                    "Hide Grid"
                                } else {
                                    "Show Grid"
                                },
                            )
                            .toggle_state(grid_visible)
                            .on_click(cx.listener(
                                move |item, _, _, cx| {
                                    item.toggle_scene_grid(&grid_instance, &grid_revision, cx);
                                },
                            )),
                        ),
                )
                .child(
                    div()
                        .debug_selector(|| "GRAPHITE-GDD-CANVAS-SNAPPING".into())
                        .child(
                            Button::new(
                                "graphite-gdd-canvas-snapping",
                                if enabled {
                                    "Snapping On"
                                } else {
                                    "Snapping Off"
                                },
                            )
                            .toggle_state(enabled)
                            .on_click(cx.listener(
                                move |item, _, _, cx| {
                                    item.toggle_scene_snapping(
                                        &instance,
                                        &revision,
                                        CANVAS_SNAP_ENABLED,
                                        cx,
                                    );
                                },
                            )),
                        ),
                )
                .child(
                    div()
                        .debug_selector(|| "GRAPHITE-GDD-CANVAS-SNAP-OPTIONS".into())
                        .child(DropdownMenu::new(
                            "graphite-gdd-canvas-snap-options",
                            "Snap Options",
                            menu,
                        )),
                ),
        )
    }

    fn render_scene_overlays_control(
        &self,
        canvas: &SavedSceneCanvas,
        mut visibility: OverlaysVisibilitySettings,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let item = cx.weak_entity();
        let menu = ContextMenu::build(window, cx, |mut menu, _, _| {
            for (label, flag) in CANVAS_OVERLAY_OPTIONS {
                let item = item.clone();
                let instance = canvas.instance.clone();
                let revision = canvas.revision.clone();
                menu = menu.toggleable_entry(
                    label,
                    *flag(&mut visibility),
                    IconPosition::Start,
                    None,
                    move |_, cx| {
                        if let Err(error) = item.update(cx, |item, cx| {
                            item.toggle_scene_overlay(&instance, &revision, flag, cx);
                        }) {
                            log::debug!(
                                "Graphite Canvas closed before changing overlays: {error:#}"
                            );
                        }
                    },
                );
            }
            menu
        });
        div()
            .debug_selector(|| "GRAPHITE-GDD-CANVAS-OVERLAYS".into())
            .child(DropdownMenu::new(
                "graphite-gdd-canvas-overlays",
                "Overlays",
                menu,
            ))
    }

    fn render_scene_grid_options(
        &self,
        form: &GridOptionsForm,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let id = form.id();
        let mode = |label, isometric, selector: &'static str, cx: &mut Context<Self>| {
            div().debug_selector(move || selector.into()).child(
                Button::new(selector, label)
                    .toggle_state(form.isometric == isometric)
                    .on_click(cx.listener(move |item, _, _, cx| {
                        if let Some(form) = item
                            .scene_canvas
                            .as_mut()
                            .and_then(|canvas| canvas.grid_options.as_mut())
                            && form.id() == id
                        {
                            form.isometric = isometric;
                            cx.notify();
                        }
                    })),
            )
        };
        v_flex()
            .id("graphite-gdd-grid-options")
            .absolute()
            .top_9()
            .left_0()
            .w_96()
            .max_w_full()
            .bottom_0()
            .overflow_y_scroll()
            .child(
                v_flex()
                    .occlude()
                    .p_2()
                    .gap_2()
                    .border_1()
                    .rounded_md()
                    .border_color(cx.theme().colors().border)
                    .bg(cx.theme().colors().surface_background)
                    .child(
                        h_flex()
                            .gap_1()
                            .child(mode(
                                "Rectangular",
                                false,
                                "GRAPHITE-GDD-GRID-RECTANGULAR",
                                cx,
                            ))
                            .child(mode("Isometric", true, "GRAPHITE-GDD-GRID-ISOMETRIC", cx))
                            .child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-GRID-DOTTED".into())
                                    .child(
                                        Button::new("graphite-gdd-grid-dotted", "Dotted")
                                            .toggle_state(form.dots)
                                            .on_click(cx.listener(move |item, _, _, cx| {
                                                if let Some(form) = item
                                                    .scene_canvas
                                                    .as_mut()
                                                    .and_then(|canvas| canvas.grid_options.as_mut())
                                                    && form.id() == id
                                                {
                                                    form.dots = !form.dots;
                                                    cx.notify();
                                                }
                                            })),
                                    ),
                            ),
                    )
                    .child(form.render_fields())
                    .when_some(form.error.clone(), |panel, error| {
                        panel.child(
                            div()
                                .debug_selector(|| "GRAPHITE-GDD-GRID-ERROR".into())
                                .child(Label::new(error).color(Color::Error)),
                        )
                    })
                    .child(
                        h_flex()
                            .gap_1()
                            .child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-GRID-APPLY".into())
                                    .child(
                                        Button::new("graphite-gdd-grid-apply", "Apply").on_click(
                                            cx.listener(move |item, _, window, cx| {
                                                item.apply_scene_grid_options(id, window, cx);
                                            }),
                                        ),
                                    ),
                            )
                            .child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-GRID-CANCEL".into())
                                    .child(
                                        Button::new("graphite-gdd-grid-cancel", "Cancel").on_click(
                                            cx.listener(move |item, _, window, cx| {
                                                if let Some(canvas) = &mut item.scene_canvas
                                                    && canvas
                                                        .grid_options
                                                        .as_ref()
                                                        .is_some_and(|form| form.id() == id)
                                                {
                                                    canvas.grid_options = None;
                                                    item.focus_handle.focus(window, cx);
                                                    cx.notify();
                                                }
                                            }),
                                        ),
                                    ),
                            ),
                    ),
            )
    }

    fn render_scene_canvas(
        &self,
        canvas: &SavedSceneCanvas,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let item = cx.entity().downgrade();
        let instance = canvas.instance.clone();
        let scroll_instance = instance.clone();
        let pan_instance = instance.clone();
        let move_instance = instance.clone();
        let capture_instance = instance.clone();
        let pick_instance = instance.clone();
        let capture_item = item.clone();
        let pane_bounds = Rc::new(Cell::new(None));
        let scroll_bounds = pane_bounds.clone();
        let pan_bounds = pane_bounds.clone();
        let move_bounds = pane_bounds.clone();
        let pick_bounds = pane_bounds.clone();
        let target = canvas.target;
        let model = self.model.read(cx);
        let current = Arc::ptr_eq(&canvas.revision, &model.revision);
        let visibility = current.then(|| canvas.overlay_visibility(model)).flatten();
        let show_overlays = visibility
            .as_ref()
            .is_some_and(OverlaysVisibilitySettings::all);
        let show_name = visibility
            .as_ref()
            .is_some_and(OverlaysVisibilitySettings::artboard_name);
        let show_cage = visibility
            .as_ref()
            .is_some_and(OverlaysVisibilitySettings::transform_cage);
        let grid_request = current.then(|| canvas.grid_request(model)).flatten();
        let grid_ready = canvas
            .grid_frame
            .as_ref()
            .filter(|frame| Some(&frame.request) == grid_request.as_ref());
        let refresh_grid = current
            && if grid_request.is_some() {
                grid_ready.is_none() && canvas.grid_task.is_none()
            } else {
                canvas.grid_frame.is_some()
            };
        let grid_image = grid_ready
            .and_then(|frame| frame.image.as_ref().ok())
            .and_then(Clone::clone);
        let grid_error = grid_ready
            .and_then(|frame| frame.image.as_ref().err())
            .cloned();
        let transform = canvas
            .snapping_document
            .as_deref()
            .filter(|_| current)
            .map(|document| document.metadata().document_to_viewport);
        let drawing = canvas.artboard_draw.as_ref().filter(|_| current);
        let moving = canvas.artboard_drag.as_ref().filter(|_| current);
        let blank = current && matches!(canvas.frame, SavedSceneFrame::Blank);
        let can_draw = current
            && self.model.read(cx).working.is_some()
            && canvas.drawing_navigation().is_some();
        let view = canvas.artboard_view.as_ref().filter(|_| current);
        let snap_guides = view
            .filter(|_| show_overlays)
            .and_then(|view| view.snap_guides.clone());
        let draft = view
            .zip(canvas.artboard_render_request())
            .and_then(|(view, request)| {
                view.draft_frame
                    .as_ref()
                    .filter(|frame| request.placement == frame.request.placement)
            });
        let image_offset = view.zip(target).map_or(DVec2::ZERO, |(view, target)| {
            view.offset_from(
                target,
                draft.map_or(view.initial.camera, |frame| frame.request.camera),
            )
        });
        let image_offset = gpui::point(px(image_offset.x as f32), px(image_offset.y as f32));
        let gesture_pane = view.map(|view| view.pane);
        let resize_outline = moving
            .filter(|drag| {
                draft.is_none() && matches!(drag.gesture, SavedSceneArtboardGesture::Resize(..))
            })
            .and_then(|drag| drag.placement);
        let edit_outline = drawing
            .and_then(|draw| draw.placement)
            .or(resize_outline.filter(|_| show_cage))
            .filter(|_| show_overlays)
            .zip(transform)
            .map(|(placement, transform)| artboard_outline(placement, transform));
        let image = match (draft, &canvas.frame) {
            (Some(frame), _) => Some(frame.image.clone()),
            (None, SavedSceneFrame::Ready(image)) if current => Some(image.clone()),
            _ => None,
        };
        let pick_targets = || {
            canvas
                .artboards
                .iter()
                .filter_map(|board| board.pick_target.as_ref())
        };
        let can_pick = current && pick_targets().next().is_some();
        let working = self
            .model
            .read(cx)
            .working
            .as_ref()
            .filter(|_| current && self.copy_task.is_none());
        let can_move = working.is_some_and(|working| {
            pick_targets().any(|pick| working.picked_artboard(pick).is_some())
        });
        let mut overlays = Vec::new();
        if image.is_some() {
            for (index, board) in draft
                .map_or(&canvas.artboards, |frame| &frame.artboards)
                .iter()
                .enumerate()
            {
                let outline_offset = if !show_cage {
                    None
                } else if let Some(moving) = moving {
                    (moving.artboard_index == index
                        && (draft.is_some()
                            || matches!(moving.gesture, SavedSceneArtboardGesture::Move(_))))
                    .then_some(if draft.is_some() {
                        DVec2::ZERO
                    } else {
                        moving.outline_offset
                    })
                } else {
                    board
                        .pick_target
                        .as_ref()
                        .filter(|pick| {
                            canvas.pick_armed
                                || ((canvas.move_armed || canvas.resize_armed)
                                    && working.is_some_and(|working| {
                                        working.picked_artboard(pick).is_some()
                                    }))
                        })
                        .map(|_| DVec2::ZERO)
                };
                if let Some(overlay) = &board.overlay
                    && (show_name || outline_offset.is_some())
                {
                    overlays.push((overlay.clone(), outline_offset));
                }
            }
        }
        let message = match &canvas.frame {
            SavedSceneFrame::Pending { .. } => Some(("Rendering Canvas…".into(), Color::Muted)),
            SavedSceneFrame::Failed(error) => Some((error.clone(), Color::Error)),
            _ => None,
        };
        let view_controls = self.render_scene_view_controls(canvas, window, cx);
        let overlay_control = visibility
            .map(|visibility| self.render_scene_overlays_control(canvas, visibility, window, cx));
        v_flex()
            .size_full()
            .relative()
            .child(
                h_flex()
                    .id("graphite-gdd-canvas-toolbar")
                    .h_9()
                    .overflow_x_scroll()
                    .px_2()
                    .gap_2()
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-CANVAS-BACK".into())
                            .child(
                                Button::new("graphite-gdd-canvas-back", "Back to Inspector")
                                    .on_click(cx.listener(|item, _, window, cx| {
                                        item.close_scene_canvas(window, cx)
                                    })),
                            ),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-CANVAS-FIT".into())
                            .child(
                                Button::new(
                                    "graphite-gdd-canvas-fit",
                                    match canvas.artboards.len() {
                                        0 => "Fit Artwork",
                                        1 => "Fit Artboard",
                                        _ => "Fit Artboards",
                                    },
                                )
                                .disabled(blank || canvas.camera_override.is_none())
                                .on_click(cx.listener(
                                    |item, _, window, cx| {
                                        item.fit_scene_canvas(window, cx);
                                    },
                                )),
                            ),
                    )
                    .children(view_controls)
                    .when(can_draw, |toolbar| {
                        toolbar.child(
                            div()
                                .debug_selector(|| "GRAPHITE-GDD-CANVAS-DRAW-ARTBOARD".into())
                                .child(
                                    Button::new(
                                        "graphite-gdd-canvas-draw-artboard",
                                        if canvas.draw_armed {
                                            "Stop Drawing Artboard"
                                        } else {
                                            "Draw Artboard"
                                        },
                                    )
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(
                                        |item, _, window, cx| {
                                            if let Some(canvas) = &mut item.scene_canvas {
                                                canvas.draw_armed = !canvas.draw_armed;
                                                canvas.pick_armed = false;
                                                canvas.move_armed = false;
                                                canvas.resize_armed = false;
                                                cx.notify();
                                            }
                                            item.cancel_scene_artboard_edit(window, cx);
                                        },
                                    )),
                                ),
                        )
                    })
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-CANVAS-PICK".into())
                            .child(
                                Button::new(
                                    "graphite-gdd-canvas-pick",
                                    if canvas.pick_armed {
                                        "Cancel Artboard Pick"
                                    } else {
                                        "Pick Artboard to Inspect"
                                    },
                                )
                                .disabled(!canvas.pick_armed && !can_pick)
                                .on_click(cx.listener(
                                    |item, _, window, cx| {
                                        if let Some(canvas) = &mut item.scene_canvas {
                                            canvas.pick_armed = !canvas.pick_armed;
                                            canvas.move_armed = false;
                                            canvas.resize_armed = false;
                                            canvas.draw_armed = false;
                                            cx.notify();
                                        }
                                        item.cancel_scene_artboard_edit(window, cx);
                                    },
                                )),
                            ),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-CANVAS-MOVE-ARTBOARD".into())
                            .child(
                                Button::new(
                                    "graphite-gdd-canvas-move-artboard",
                                    if canvas.move_armed {
                                        "Stop Moving Artboard"
                                    } else {
                                        "Move Artboard"
                                    },
                                )
                                .disabled(!canvas.move_armed && !can_move)
                                .on_click(cx.listener(
                                    |item, _, window, cx| {
                                        if let Some(canvas) = &mut item.scene_canvas {
                                            canvas.move_armed = !canvas.move_armed;
                                            canvas.resize_armed = false;
                                            canvas.pick_armed = false;
                                            canvas.draw_armed = false;
                                            cx.notify();
                                        }
                                        item.cancel_scene_artboard_edit(window, cx);
                                    },
                                )),
                            ),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-GDD-CANVAS-RESIZE-ARTBOARD".into())
                            .child(
                                Button::new(
                                    "graphite-gdd-canvas-resize-artboard",
                                    if canvas.resize_armed {
                                        "Stop Resizing Artboard"
                                    } else {
                                        "Resize Artboard"
                                    },
                                )
                                .disabled(!canvas.resize_armed && !can_move)
                                .on_click(cx.listener(
                                    |item, _, window, cx| {
                                        if let Some(canvas) = &mut item.scene_canvas {
                                            canvas.resize_armed = !canvas.resize_armed;
                                            canvas.move_armed = false;
                                            canvas.pick_armed = false;
                                            canvas.draw_armed = false;
                                            cx.notify();
                                        }
                                        item.cancel_scene_artboard_edit(window, cx);
                                    },
                                )),
                            ),
                    )
                    .children(overlay_control)
                    .when_some(message, |toolbar, (message, color)| {
                        toolbar.child(Label::new(message).color(color))
                    })
                    .when_some(grid_error, |toolbar, error| {
                        toolbar.child(Label::new(error).color(Color::Error))
                    })
                    .when_some(self.copy_feedback.clone(), |toolbar, (message, color)| {
                        toolbar.child(Label::new(message).color(color))
                    })
                    .when_some(
                        canvas.artboard_view.as_ref().and_then(|draft| {
                            draft
                                .draft_error
                                .clone()
                                .or_else(|| draft.snap_guide_error.clone().filter(|_| show_overlays))
                        }),
                        |toolbar, message| toolbar.child(Label::new(message).color(Color::Error)),
                    )
                    .when(
                        matches!(canvas.frame, SavedSceneFrame::Failed(_)),
                        |toolbar| {
                            toolbar.child(
                                Button::new("graphite-gdd-canvas-retry", "Retry").on_click(
                                    cx.listener(move |item, _, window, cx| {
                                        item.update_scene_canvas(target, true, window, cx);
                                    }),
                                ),
                            )
                        },
                    ),
            )
            .child(
                div()
                    .id("graphite-gdd-document-canvas")
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .overflow_hidden()
                    .bg(cx.theme().colors().editor_background)
                    .aria_label("Graphite Document Canvas")
                    .debug_selector(|| "GRAPHITE-GDD-DOCUMENT-CANVAS".into())
                    .when(
                        canvas.pan.is_some() || canvas.artboard_drag.is_some(),
                        |pane| pane.cursor(gpui::CursorStyle::ClosedHand),
                    )
                    .when(
                        (canvas.pick_armed
                            || canvas.move_armed
                            || canvas.resize_armed
                            || (can_draw && canvas.draw_armed))
                            && canvas.pan.is_none()
                            && canvas.artboard_drag.is_none(),
                        |pane| pane.cursor(gpui::CursorStyle::Crosshair),
                    )
                    .on_click(
                        cx.listener(move |item, event: &gpui::ClickEvent, window, cx| {
                            if let Some(pane) = pick_bounds.get() {
                                item.pick_scene_artboard(
                                    &pick_instance,
                                    pane,
                                    event.position(),
                                    window,
                                    cx,
                                );
                            }
                        }),
                    )
                    .on_mouse_down(
                        gpui::MouseButton::Left,
                        cx.listener(move |item, event, window, cx| {
                            if let Some(pane) = move_bounds.get() {
                                item.start_scene_artboard_draw(
                                    &move_instance,
                                    pane,
                                    event,
                                    window,
                                    cx,
                                );
                                item.start_scene_artboard_drag(
                                    &move_instance,
                                    pane,
                                    event,
                                    window,
                                    cx,
                                );
                            }
                        }),
                    )
                    .on_mouse_down(
                        gpui::MouseButton::Middle,
                        cx.listener(move |item, event, window, cx| {
                            if let Some(pane) = pan_bounds.get() {
                                item.start_scene_pan(&pan_instance, pane, event, window, cx);
                            }
                        }),
                    )
                    .on_scroll_wheel(cx.listener(move |item, event, window, cx| {
                        if let Some(pane) = scroll_bounds.get() {
                            item.scroll_scene_canvas(&scroll_instance, pane, event, window, cx);
                        }
                    }))
                    .child(
                        gpui::canvas(
                            move |bounds, window, cx| {
                                pane_bounds.set(Some(bounds));
                                let measured =
                                    SavedSceneTarget::measure(bounds.size, window.scale_factor());
                                if target == measured
                                    && gesture_pane.is_none_or(|pane| pane == bounds)
                                {
                                    if refresh_grid {
                                        window.defer(cx, move |window, cx| {
                                            if let Err(error) = item.update(cx, |item, cx| {
                                                item.render_scene_grid(&instance, window, cx);
                                            }) {
                                                log::debug!("Graphite Canvas closed before grid paint: {error:#}");
                                            }
                                        });
                                    }
                                    return (image, grid_image, overlays, edit_outline, snap_guides);
                                }
                                window.defer(cx, move |window, cx| {
                                    if let Err(error) = item.update(cx, |item, cx| {
                                        if item.scene_canvas.as_ref().is_some_and(|canvas| {
                                            Arc::ptr_eq(&canvas.instance, &instance)
                                        }) {
                                            if target == measured
                                                && item
                                                    .scene_canvas
                                                    .as_ref()
                                                    .and_then(|canvas| {
                                                        canvas.artboard_view.as_ref()
                                                    })
                                                    .is_some_and(|view| view.pane != bounds)
                                            {
                                                item.cancel_scene_artboard_edit(window, cx);
                                            }
                                            item.update_scene_canvas(measured, false, window, cx);
                                        }
                                    }) {
                                        log::debug!(
                                            "Graphite Canvas closed before layout: {error:#}"
                                        );
                                    }
                                });
                                (None, None, Vec::new(), None, None)
                            },
                            move |bounds,
                                  (image, grid_image, overlays, edit_outline, snap_guides),
                                  window,
                                  _| {
                                Self::capture_scene_gestures(
                                    capture_item,
                                    capture_instance,
                                    bounds,
                                    window,
                                );
                                if let Some(image) = image
                                    && let Err(error) = window.paint_image(
                                        bounds,
                                        gpui::Bounds::new(
                                            bounds.origin + image_offset,
                                            bounds.size,
                                        ),
                                        gpui::Corners::default(),
                                        image,
                                        0,
                                        false,
                                    )
                                {
                                    log::warn!("Graphite Canvas frame was not painted: {error:#}");
                                }
                                if let Some(image) = grid_image
                                    && let Err(error) = window.paint_image(
                                        bounds,
                                        bounds,
                                        gpui::Corners::default(),
                                        image,
                                        0,
                                        false,
                                    )
                                {
                                    log::warn!("Graphite Canvas grid was not painted: {error:#}");
                                }
                                if !overlays.is_empty() {
                                    window.paint_layer(bounds, |window| {
                                        for (overlay, outline_offset) in overlays {
                                            overlay.paint(
                                                bounds.origin + image_offset,
                                                show_name,
                                                outline_offset,
                                                window,
                                            );
                                        }
                                    });
                                }
                                if edit_outline.is_some() || snap_guides.is_some() {
                                    window.with_content_mask(
                                        Some(gpui::ContentMask { bounds }),
                                        |window| {
                                            if let Some(outline) = edit_outline {
                                                paint_artboard_outline(
                                                    outline,
                                                    bounds.origin,
                                                    window,
                                                );
                                            }
                                            if let Some(guides) = snap_guides {
                                                guides.paint(bounds.origin, window);
                                            }
                                        },
                                    );
                                }
                            },
                        )
                        .absolute()
                        .size_full(),
                    ),
            )
            .children(
                canvas
                    .grid_options
                    .as_ref()
                    .map(|form| self.render_scene_grid_options(form, cx)),
            )
    }

    fn capture_scene_gestures(
        item: WeakEntity<Self>,
        instance: Arc<()>,
        pane: gpui::Bounds<gpui::Pixels>,
        window: &mut Window,
    ) {
        let move_item = item.clone();
        let move_instance = instance.clone();
        window.on_mouse_event(move |event: &gpui::MouseMoveEvent, phase, window, cx| {
            if phase == gpui::DispatchPhase::Capture
                && let Err(error) = move_item.update(cx, |item, cx| {
                    item.move_scene_pan(&move_instance, event, window, cx);
                    item.move_scene_artboard(&move_instance, pane, event, window, cx);
                })
            {
                log::debug!("Graphite Canvas closed during pan: {error:#}");
            }
        });
        let release_item = item.clone();
        let release_instance = instance.clone();
        window.on_mouse_event(move |event: &gpui::MouseUpEvent, phase, window, cx| {
            if phase == gpui::DispatchPhase::Capture
                && let Err(error) = release_item.update(cx, |item, cx| match event.button {
                    gpui::MouseButton::Middle => {
                        item.finish_captured_scene_pan(&release_instance, false, window, cx)
                    }
                    gpui::MouseButton::Left => {
                        item.finish_scene_artboard_draw(&release_instance, pane, event, window, cx);
                        item.finish_scene_artboard_drag(&release_instance, pane, event, window, cx);
                    }
                    _ => {}
                })
            {
                log::debug!("Graphite Canvas closed before pan release: {error:#}");
            }
        });
        window.on_mouse_event(move |event: &gpui::MouseDownEvent, phase, window, cx| {
            if phase == gpui::DispatchPhase::Capture
                && event.button == gpui::MouseButton::Right
                && let Err(error) = item.update(cx, |item, cx| {
                    item.finish_captured_scene_pan(&instance, true, window, cx);
                    if item
                        .scene_canvas
                        .as_ref()
                        .is_some_and(|canvas| Arc::ptr_eq(&canvas.instance, &instance))
                        && item.cancel_scene_artboard_edit(window, cx)
                    {
                        cx.stop_propagation();
                    }
                })
            {
                log::debug!("Graphite Canvas closed before pan cancellation: {error:#}");
            }
        });
    }

    fn finish_captured_scene_pan(
        &mut self,
        instance: &Arc<()>,
        abort: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .scene_canvas
            .as_ref()
            .is_some_and(|canvas| Arc::ptr_eq(&canvas.instance, instance) && canvas.pan.is_some())
        {
            self.finish_scene_pan(abort, window, cx);
            cx.stop_propagation();
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
        if let NodeCatalogPanel::Ready { catalog, .. } = &self.node_catalog {
            catalog.update(cx, NodeCatalog::start_working_copy);
        }
        match &mut self.node_copy {
            NodeCopyPanel::Closed => {}
            NodeCopyPanel::Ready(form) if !Arc::ptr_eq(&form.revision, &revision) => {
                let reopen_graph = matches!(
                    form.saved_graph,
                    SavedGraphPanel::Loading { .. } | SavedGraphPanel::Ready(_)
                );
                if form.saved_graph_focus.contains_focused(window, cx)
                    || form.saved_layers_focus.contains_focused(window, cx)
                {
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
                            let model = item.model.read(cx);
                            let admit_insertion = matches!(
                                &model.document,
                                InspectedDocument::Gdd { contents } if contents.hot_log_len() == 0
                            );
                            let working = model.working.is_some();
                            let catalog = cx.new(|cx| {
                                NodeCatalog::new(source, admit_insertion, working, window, cx)
                            });
                            let insert_subscription =
                                cx.subscribe_in(&catalog, window, |item, _, event, window, cx| {
                                    item.insert_catalog_node(event, window, cx);
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

    fn insert_catalog_node(
        &mut self,
        event: &CatalogNodeInsert,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match parse_graph_grid_position(&event.position) {
            Ok(position) => self.apply_graph_edit(
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

    fn insert_graph_node(
        &mut self,
        node: GraphiteInsertNode,
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
            NodeInsertDestination::SeparateFolder => {
                let request = match node {
                    GraphiteInsertNode::Numeric(operation) => GraphiteCopy::InsertNumericNode {
                        operation,
                        position,
                    },
                    GraphiteInsertNode::Custom => GraphiteCopy::InsertCatalogNode {
                        identifier: CUSTOM_NODE_IDENTIFIER,
                        position,
                    },
                };
                self.save_copy(request, window, cx);
            }
            NodeInsertDestination::WorkingCopy => {
                let revision = self.model.read(cx).revision.clone();
                self.insert_working_graph_node(node, position, None, revision, window, cx);
            }
        }
    }

    fn insert_working_graph_node(
        &mut self,
        node: GraphiteInsertNode,
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
                match (node, network.as_deref()) {
                    (GraphiteInsertNode::Numeric(operation), Some(network)) => {
                        contents
                            .insert_numeric_node_in_network_copy(
                                network,
                                operation,
                                position,
                                ARCHIVE_BUDGET,
                            )
                            .await
                    }
                    (GraphiteInsertNode::Numeric(operation), None) => {
                        contents
                            .insert_root_numeric_node_in_copy(operation, position, ARCHIVE_BUDGET)
                            .await
                    }
                    (GraphiteInsertNode::Custom, Some(network)) => {
                        contents
                            .insert_custom_node_in_network_copy(network, position, ARCHIVE_BUDGET)
                            .await
                    }
                    (GraphiteInsertNode::Custom, None) => {
                        contents
                            .insert_root_catalog_node_in_copy(
                                CUSTOM_NODE_IDENTIFIER,
                                position,
                                ARCHIVE_BUDGET,
                            )
                            .await
                    }
                }
            },
            window,
            cx,
        );
    }

    fn graph_node_picker(
        include_custom: bool,
        pick: impl Fn(GraphiteInsertNode, &mut Window, &mut App) + Clone + 'static,
        window: &mut Window,
        cx: &mut App,
    ) -> Entity<ContextMenu> {
        ContextMenu::build(window, cx, |mut menu, _, _| {
            for node in GRAPHITE_NUMERIC_OPERATIONS
                .into_iter()
                .map(GraphiteInsertNode::Numeric)
                .chain(include_custom.then_some(GraphiteInsertNode::Custom))
            {
                let pick = pick.clone();
                menu = menu.custom_entry(
                    move |_, _| {
                        div()
                            .debug_selector(move || format!("GRAPHITE-GDD-ADD-{}", node.name()))
                            .p_2()
                            .child(Label::new(node.name()))
                            .into_any_element()
                    },
                    move |window, cx| pick(node, window, cx),
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
                    let menu = Self::graph_node_picker(
                        true,
                        move |node, window, cx| {
                            if let Some(item) = item.upgrade() {
                                item.update(cx, |item, cx| {
                                    item.insert_graph_node(node, window, cx);
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
                                        "Choose Node",
                                        menu,
                                    )),
                            )
                        })
                },
            )
    }

    fn toggle_node_copy(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match &self.node_copy {
            NodeCopyPanel::Loading { .. } => {}
            NodeCopyPanel::Ready(_) => {
                self.node_copy = NodeCopyPanel::Closed;
                cx.notify();
            }
            NodeCopyPanel::Closed | NodeCopyPanel::Failed(_) => {
                self.show_node_copy(None, window, cx);
            }
        }
    }

    fn show_node_copy(
        &mut self,
        pick: Option<GddArtboardCanvasPick>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let model = self.model.read(cx);
        let Some(contents) = model.gdd_contents().cloned() else {
            return;
        };
        let revision = model.revision.clone();
        if let NodeCopyPanel::Ready(form) = &self.node_copy
            && Arc::ptr_eq(&form.revision, &revision)
        {
            if let Some(pick) = pick {
                self.select_picked_artboard(pick, window, cx);
            }
            return;
        }
        if let Some(working) = &model.working {
            let inspection = working.inspection.clone();
            self.node_copy = NodeCopyPanel::Ready(Box::new(NodeCopyForm::new(
                contents,
                revision,
                &inspection,
                window,
                cx,
            )));
            if let Some(pick) = pick {
                self.select_picked_artboard(pick, window, cx);
            }
        } else {
            let load = cx.background_spawn({
                let contents = contents.clone();
                async move { contents.graph_inspection().await }
            });
            self.node_copy = NodeCopyPanel::Loading {
                _task: cx.spawn_in(window, async move |item, cx| {
                    let loaded = load.await;
                    if let Err(error) = item.update_in(cx, |item, window, cx| {
                        if !matches!(item.node_copy, NodeCopyPanel::Loading { .. })
                            || !Arc::ptr_eq(&revision, &item.model.read(cx).revision)
                        {
                            return;
                        }
                        item.node_copy = match loaded {
                            Ok(graph) => NodeCopyPanel::Ready(Box::new(NodeCopyForm::new(
                                contents, revision, &graph, window, cx,
                            ))),
                            Err(error) => NodeCopyPanel::Failed(error.to_string().into()),
                        };
                        if let Some(pick) = pick {
                            item.select_picked_artboard(pick, window, cx);
                        }
                        cx.notify();
                    }) {
                        log::debug!("Graphite node edit panel closed: {error:#}");
                    }
                }),
            };
        }
        cx.notify();
    }

    fn select_picked_artboard(
        &mut self,
        pick: GddArtboardCanvasPick,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let NodeCopyPanel::Ready(form) = &mut self.node_copy
            && let Some(index) = form.nodes.iter().position(|node| pick.matches_node(node))
        {
            form.select_network(None);
            form.node_selection.insert(index);
            self.select_copy_node(index, window, cx);
        }
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
            if let Some(geometry) = node.artboard_geometry() {
                let [x, y] = geometry.location.to_array();
                let [width, height] = geometry.dimensions.to_array();
                form.position.update(cx, |editor, cx| {
                    editor.set_text(format!("{x}, {y}"), window, cx)
                });
                form.value.update(cx, |editor, cx| {
                    editor.set_text(format!("{width}, {height}"), window, cx)
                });
            }
            if let Some(background) = node.artboard_background() {
                let [red, green, blue, alpha] = background.to_gamma_srgb_channels();
                form.artboard_background.update(cx, |editor, cx| {
                    editor.set_text(format!("{red}, {green}, {blue}, {alpha}"), window, cx)
                });
            }
            if let Some(radii) = node.ellipse_radii() {
                for (field, radius) in form.ellipse_radii.iter().zip(radii) {
                    field.update(cx, |editor, cx| {
                        editor.set_text(radius.to_string(), window, cx)
                    });
                }
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
            if form.saved_graph_focus.contains_focused(window, cx)
                || form.saved_layers_focus.contains_focused(window, cx)
            {
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
            SavedGraphAction::SelectLayer {
                index,
                extend,
                range,
            } => {
                self.select_saved_layer(index, extend, range, window, cx);
            }
            SavedGraphAction::MoveNode { index, position } => {
                self.move_canvas_nodes(index, position, window, cx);
            }
            SavedGraphAction::SelectAllLayers => self.set_saved_layer_selection(true, window, cx),
            SavedGraphAction::DeselectLayers => self.set_saved_layer_selection(false, window, cx),
            SavedGraphAction::SelectParentLayers => self.select_saved_layer_parents(window, cx),
            SavedGraphAction::ToggleLayerExpansion { index, recursive } => {
                self.toggle_saved_layer_expansion(index, recursive, cx);
            }
            SavedGraphAction::EditLayer { index, change } => {
                self.edit_canvas_layer(index, change, window, cx);
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

    fn edit_canvas_layer(
        &mut self,
        index: usize,
        change: GddRootLayerChange,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() || self.model.read(cx).working.is_none() {
            return;
        }
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let SavedGraphPanel::Ready(graph) = &form.saved_graph else {
            return;
        };
        let Some(node) = form
            .nodes
            .get(index)
            .filter(|node| {
                form.pending_canvas_source.is_none()
                    && graph.root_layers.as_ref().is_ok_and(|layers| {
                        layers
                            .iter()
                            .any(|layer| layer.storage_id == node.storage_id())
                    })
            })
            .cloned()
        else {
            return;
        };
        let snapshot = form.snapshot.clone();
        let revision = form.revision.clone();
        self.stage_working_snapshot(
            revision,
            async move {
                snapshot
                    .edit_root_layer_in_copy(&node, change, ARCHIVE_BUDGET)
                    .await
            },
            window,
            cx,
        );
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
            GraphiteCopy::InsertCatalogNode {
                identifier,
                position,
            } => {
                self.stage_working_snapshot(
                    revision,
                    async move {
                        contents
                            .insert_root_catalog_node_in_copy(identifier, position, ARCHIVE_BUDGET)
                            .await
                    },
                    window,
                    cx,
                );
            }
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

    fn select_saved_layer(
        &mut self,
        index: usize,
        extend: bool,
        range: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if form.pending_canvas_source.is_some() {
            return;
        }
        let Some(scope) = form.clipboard_scope().map(|network| network.id) else {
            return;
        };
        let Some(rows) = form.saved_layer_rows() else {
            return;
        };
        let order = rows.into_iter().map(|(index, _)| index).collect::<Vec<_>>();
        if !order.contains(&index) {
            return;
        }
        let in_scope = |index| {
            form.nodes
                .get(index)
                .is_some_and(|node: &GddGraphNode| node.network_id() == scope)
        };
        form.node_selection.retain(|&index| in_scope(index));
        if form.node_selection.is_empty() {
            form.node_selection
                .extend(form.selected.filter(|&index| in_scope(index)));
        }
        crate::layers::update_layer_selection(
            &order,
            index,
            &mut form.layer_selection_anchor,
            &mut form.node_selection,
            extend,
            range,
        );
        let active = if form.node_selection.contains(&index) {
            Some(index)
        } else {
            form.selected
                .filter(|index| form.node_selection.contains(index))
                .or_else(|| form.node_selection.first().copied())
        };
        self.set_active_saved_layer(active, window, cx);
    }

    fn set_saved_layer_selection(
        &mut self,
        select_all: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if form.pending_canvas_source.is_some() {
            return;
        }
        let Some(rows) = form.saved_layer_rows() else {
            return;
        };
        if select_all {
            form.node_selection = rows
                .into_iter()
                .filter(|(_, layer)| layer.select_all_eligible)
                .map(|(index, _)| index)
                .collect();
        } else {
            form.node_selection.clear();
            form.layer_selection_anchor = None;
        }
        let active = form
            .selected
            .filter(|index| form.node_selection.contains(index))
            .or_else(|| form.node_selection.first().copied());
        self.set_active_saved_layer(active, window, cx);
    }

    fn select_saved_layer_parents(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if form.pending_canvas_source.is_some() {
            return;
        }
        let Some(rows) = form.saved_layer_rows() else {
            return;
        };
        let nodes_by_id = rows
            .iter()
            .map(|&(index, layer)| (layer.storage_id, index))
            .collect::<BTreeMap<_, _>>();
        let mut parents = BTreeSet::new();
        let mut active_parent = None;
        for (index, layer) in rows {
            if form.selected != Some(index) && !form.node_selection.contains(&index) {
                continue;
            }
            let parent = match layer.parent {
                Some(id) => {
                    let Some(&parent) = nodes_by_id.get(&id) else {
                        continue;
                    };
                    parent
                }
                None => index,
            };
            if form.selected == Some(index) {
                active_parent = Some(parent);
            }
            parents.insert(parent);
        }
        if parents.is_empty() {
            return;
        }
        let active = active_parent.or_else(|| parents.first().copied());
        form.node_selection = parents;
        self.set_active_saved_layer(active, window, cx);
    }

    fn set_active_saved_layer(
        &mut self,
        active: Option<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if form.node_selection.len() == 1 {
            form.node_selection.clear();
        }
        if let Some(active) = active {
            self.select_copy_node(active, window, cx);
        } else {
            form.selected = None;
            form.output_preview = NodeOutputPreview::Hidden;
            cx.notify();
        }
    }

    fn toggle_saved_layer_expansion(
        &mut self,
        index: usize,
        recursive: bool,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        let Some(rows) = form.saved_layer_rows() else {
            return;
        };
        let order = rows
            .into_iter()
            .map(|(index, layer)| (index, layer.depth))
            .collect::<Vec<_>>();
        let mut subtree = order.into_iter().skip_while(|(row, _)| *row != index);
        let Some((_, depth)) = subtree.next() else {
            return;
        };
        let mut descendants = subtree
            .take_while(|(_, row_depth)| *row_depth > depth)
            .map(|(row, _)| row)
            .peekable();
        if descendants.peek().is_none() {
            return;
        }
        let expanding = form.collapsed_layers.remove(&index);
        if !expanding {
            form.collapsed_layers.insert(index);
        }
        if recursive {
            if expanding {
                for row in descendants {
                    form.collapsed_layers.remove(&row);
                }
            } else {
                form.collapsed_layers.extend(descendants);
            }
        }
        cx.notify();
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
            network.can_append_ports_in_copy() && (!with_hidden_primary || kind.count(network) == 0)
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

    fn append_numeric_import(
        &mut self,
        kind: GddNumericImportKind,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(network) = form
            .active_network()
            .filter(|network| network.can_append_ports_in_copy())
        else {
            return;
        };
        self.save_copy(
            GraphiteCopy::AppendNumericImport {
                network: Box::new(network.clone()),
                kind,
            },
            window,
            cx,
        );
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
            NodeOutputPreview::Pending { .. }
                | NodeOutputPreview::Ready(_)
                | NodeOutputPreview::ReadyImage(_)
        ) {
            form.output_preview = NodeOutputPreview::Hidden;
            cx.notify();
            return;
        }
        let Some(node) = form
            .selected_node()
            .filter(|node| node.can_preview_numeric_output() || node.layer_preview().is_some())
        else {
            return;
        };
        let model = self.model.read(cx);
        if !Arc::ptr_eq(&form.revision, &model.revision) {
            return;
        }
        let InspectedDocument::Gdd { contents } = &model.document else {
            return;
        };
        let selected = form.selected;
        let node = node.clone();
        let contents = contents.clone();
        let revision = form.revision.clone();
        let layer = node
            .layer_preview()
            .map(|preview| (preview, cx.svg_renderer()));
        let evaluation = cx.background_spawn(async move {
            if let Some((preview, renderer)) = layer {
                futures::executor::block_on(preview_layer_image(
                    &contents, &node, preview, &renderer,
                ))
                .map(NodeOutputPreview::ReadyImage)
            } else {
                futures::executor::block_on(contents.preview_graph_numeric_output(&node))
                    .map(NodeOutputPreview::Ready)
                    .map_err(|error| error.to_string())
            }
        });
        let request = Arc::new(());
        form.output_preview = NodeOutputPreview::Pending {
            request: request.clone(),
            _task: cx.spawn(async move |item, cx| {
                let result = evaluation.await;
                if let Err(error) = item.update(cx, |item, cx| {
                    let NodeCopyPanel::Ready(form) = &mut item.node_copy else {
                        return;
                    };
                    if form.selected != selected
                        || !Arc::ptr_eq(&revision, &item.model.read(cx).revision)
                        || !form.output_preview.is_pending_request(&request)
                    {
                        return;
                    }
                    form.output_preview = match result {
                        Ok(output) => output,
                        Err(error) => NodeOutputPreview::Failed(error.into()),
                    };
                    cx.notify();
                }) {
                    log::debug!("Graphite node preview closed: {error:#}");
                }
            }),
        };
        cx.notify();
    }

    fn run_document_graph(&mut self, force: bool, cx: &mut Context<Self>) {
        let NodeCopyPanel::Ready(form) = &mut self.node_copy else {
            return;
        };
        if !Arc::ptr_eq(&form.revision, &self.model.read(cx).revision) {
            return;
        }
        if matches!(form.document_run, NodeOutputPreview::Pending { .. }) {
            if !force {
                form.document_run = NodeOutputPreview::Cancelled;
                cx.notify();
            }
            return;
        }
        let Some((node, output_index)) = form
            .networks
            .iter()
            .find(|network| network.parent.is_none())
            .and_then(|network| network.connected_export(&form.nodes))
            .map(|(node, output_index)| (node.clone(), output_index))
        else {
            form.document_run = NodeOutputPreview::Failed(
                "Run Graph requires a single connected document export.".into(),
            );
            cx.notify();
            return;
        };
        let contents = form.snapshot.clone();
        let revision = form.revision.clone();
        let cached = form.document_run_evaluator.clone().filter(|_| !force);
        let evaluation = cx.background_spawn(async move {
            futures::executor::block_on(async {
                let evaluator = match cached {
                    Some(evaluator) => evaluator,
                    None => Arc::new(
                        contents
                            .prepare_document_numeric_output(&node, output_index)
                            .await?,
                    ),
                };
                let value = evaluator.evaluate().await?;
                Ok::<_, GddArchiveError>((evaluator, value))
            })
            .map_err(|error| format!("{}: {error}", node.label()))
        });
        let request = Arc::new(());
        form.document_run = NodeOutputPreview::Pending {
            request: request.clone(),
            _task: cx.spawn(async move |item, cx| {
                let result = evaluation.await;
                if let Err(error) = item.update(cx, |item, cx| {
                    let NodeCopyPanel::Ready(form) = &mut item.node_copy else {
                        return;
                    };
                    if !Arc::ptr_eq(&form.revision, &revision)
                        || !Arc::ptr_eq(&item.model.read(cx).revision, &revision)
                        || !form.document_run.is_pending_request(&request)
                    {
                        return;
                    }
                    form.document_run = match result {
                        Ok((evaluator, value)) => {
                            form.document_run_evaluator = Some(evaluator);
                            NodeOutputPreview::Ready(value)
                        }
                        Err(error) => NodeOutputPreview::Failed(error.into()),
                    };
                    cx.notify();
                }) {
                    log::debug!("Graphite document run closed: {error:#}");
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
        layer: Option<GddGraphLayerPreview>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let (button_id, selector, result_selector, image_selector, name) = match layer {
            Some(GddGraphLayerPreview::Merge) => (
                "graphite-gdd-preview-merge",
                "GRAPHITE-GDD-PREVIEW-MERGE",
                "GRAPHITE-GDD-MERGE-RESULT",
                "GRAPHITE-GDD-MERGE-IMAGE",
                "Merge",
            ),
            Some(GddGraphLayerPreview::Artboard { .. }) => (
                "graphite-gdd-preview-artboard",
                "GRAPHITE-GDD-PREVIEW-ARTBOARD",
                "GRAPHITE-GDD-ARTBOARD-RESULT",
                "GRAPHITE-GDD-ARTBOARD-IMAGE",
                "Artboard",
            ),
            None => (
                "graphite-gdd-preview-numeric",
                "GRAPHITE-GDD-PREVIEW-NUMERIC",
                "GRAPHITE-GDD-NUMERIC-RESULT",
                "GRAPHITE-GDD-NUMERIC-IMAGE",
                "Numeric",
            ),
        };
        let label = match preview {
            NodeOutputPreview::Pending { .. } => format!("Cancel {name} Preview"),
            NodeOutputPreview::Ready(_) | NodeOutputPreview::ReadyImage(_) => {
                format!("Hide {name} Preview")
            }
            NodeOutputPreview::Failed(_) => format!("Retry {name} Preview"),
            _ if layer.is_some() => format!("Preview Saved {name}"),
            _ => "Preview Saved Output".to_owned(),
        };
        v_flex()
            .gap_2()
            .child(
                div().debug_selector(move || selector.into()).child(
                    Button::new(button_id, label).on_click(
                        cx.listener(|item, _, _, cx| item.toggle_node_output_preview(cx)),
                    ),
                ),
            )
            .when_some(preview.feedback(), |panel, (message, color)| {
                panel.child(
                    div()
                        .debug_selector(move || result_selector.into())
                        .child(Label::new(message).color(color)),
                )
            })
            .when_some(
                match preview {
                    NodeOutputPreview::ReadyImage(image) => Some(image.clone()),
                    _ => None,
                },
                |panel, image| {
                    panel.child(
                        div()
                            .debug_selector(move || image_selector.into())
                            .size(px(LAYER_PREVIEW_SIZE as f32))
                            .child(img(ImageSource::Render(image)).size_full()),
                    )
                },
            )
    }

    fn render_document_run(run: &NodeOutputPreview, cx: &mut Context<Self>) -> impl IntoElement {
        let pending = matches!(run, NodeOutputPreview::Pending { .. });
        let label = if pending { "Cancel Run" } else { "Run Graph" };
        h_flex()
            .gap_2()
            .flex_wrap()
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-GDD-RUN-GRAPH".into())
                    .child(Button::new("graphite-gdd-run-graph", label).on_click(
                        cx.listener(|item, _, _, cx| item.run_document_graph(false, cx)),
                    )),
            )
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-GDD-FORCE-RUN-GRAPH".into())
                    .child(
                        Button::new("graphite-gdd-force-run-graph", "Force Run")
                            .disabled(pending)
                            .on_click(
                                cx.listener(|item, _, _, cx| item.run_document_graph(true, cx)),
                            ),
                    ),
            )
            .when_some(run.feedback(), |row, (message, color)| {
                row.child(
                    div()
                        .debug_selector(|| "GRAPHITE-GDD-GRAPH-RUN-RESULT".into())
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

    fn change_artboard_geometry(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.copy_task.is_some() {
            return;
        }
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(node) = form
            .selected_node()
            .filter(|node| node.artboard_geometry().is_some())
        else {
            return;
        };
        let geometry = GddArtboardGeometry::parse(
            &form.position.read(cx).text(cx),
            &form.value.read(cx).text(cx),
        );
        match geometry {
            Ok(geometry) => self.apply_graph_edit(
                GraphiteCopy::EditNode {
                    node: Box::new(node.clone()),
                    change: NodeCopyChange::ArtboardGeometry(geometry),
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

    fn change_artboard_background(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.copy_task.is_some() {
            return;
        }
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(node) = form
            .selected_node()
            .filter(|node| node.artboard_background().is_some())
        else {
            return;
        };
        match crate::session::parse_color(
            &form.artboard_background.read(cx).text(cx),
            "Artboard Background",
        ) {
            Ok(channels) => self.apply_graph_edit(
                GraphiteCopy::EditNode {
                    node: Box::new(node.clone()),
                    change: NodeCopyChange::ArtboardBackground(channels),
                },
                window,
                cx,
            ),
            Err(error) => {
                self.copy_feedback = Some((error.into(), Color::Error));
                cx.notify();
            }
        }
    }

    fn change_ellipse_radius(
        &mut self,
        axis: GddEllipseAxis,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.copy_task.is_some() {
            return;
        }
        let NodeCopyPanel::Ready(form) = &self.node_copy else {
            return;
        };
        let Some(node) = form
            .selected_node()
            .filter(|node| node.ellipse_radii().is_some())
        else {
            return;
        };
        let [radius_x, radius_y] = &form.ellipse_radii;
        let field = match axis {
            GddEllipseAxis::X => radius_x,
            GddEllipseAxis::Y => radius_y,
        };
        match field.read(cx).text(cx).trim().parse::<f64>() {
            Ok(radius) => self.apply_graph_edit(
                GraphiteCopy::EditNode {
                    node: Box::new(node.clone()),
                    change: NodeCopyChange::EllipseRadius(axis, radius),
                },
                window,
                cx,
            ),
            Err(_) => {
                self.copy_feedback = Some((
                    GddArchiveError::InvalidEllipseRadius.to_string().into(),
                    Color::Error,
                ));
                cx.notify();
            }
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
                [&form.name, &form.value, &form.position, &form.artboard_background].into_iter().any(editor_focused)
                    || form.ellipse_radii.iter().any(editor_focused)
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

    fn render_ellipse_radii(
        &self,
        form: &NodeCopyForm,
        working: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        v_flex().gap_2().children(
            [
                (
                    GddEllipseAxis::X,
                    "Radius X (px)",
                    "GRAPHITE-GDD-ELLIPSE-RADIUS-X",
                    "GRAPHITE-GDD-APPLY-ELLIPSE-RADIUS-X",
                    "Apply Radius X",
                    "Save Radius X in a Copy",
                ),
                (
                    GddEllipseAxis::Y,
                    "Radius Y (px)",
                    "GRAPHITE-GDD-ELLIPSE-RADIUS-Y",
                    "GRAPHITE-GDD-APPLY-ELLIPSE-RADIUS-Y",
                    "Apply Radius Y",
                    "Save Radius Y in a Copy",
                ),
            ]
            .into_iter()
            .zip(&form.ellipse_radii)
            .map(|((axis, label, field_id, button_id, apply, save), field)| {
                v_flex()
                    .gap_1()
                    .child(Label::new(label))
                    .child(
                        div()
                            .id(field_id)
                            .aria_label(label)
                            .debug_selector(move || field_id.into())
                            .child(field.clone()),
                    )
                    .child(
                        div().debug_selector(move || button_id.into()).child(
                            Button::new(button_id, if working { apply } else { save })
                                .disabled(self.copy_task.is_some())
                                .on_click(cx.listener(move |item, _, window, cx| {
                                    item.change_ellipse_radius(axis, window, cx)
                                })),
                        ),
                    )
            }),
        )
    }

    fn render_ellipse_radius_source(
        &self,
        form: &NodeCopyForm,
        target: &GddGraphNode,
        axis: GddEllipseAxis,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement + use<>> {
        let port = target.ellipse_radius_port(axis)?;
        let (name, axis_name) = match axis {
            GddEllipseAxis::X => ("Radius X", "X"),
            GddEllipseAxis::Y => ("Radius Y", "Y"),
        };
        let connected = match port.source {
            Some(GddGraphPortSource::Node {
                storage_id,
                output_index: 0,
            }) => Some(storage_id),
            _ => None,
        };
        let sources: Vec<_> = form
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, source)| target.can_connect_ellipse_radius_from(axis, source))
            .collect();
        if sources.is_empty() && connected.is_none() {
            return None;
        }
        let menu_id = format!("GRAPHITE-GDD-CONNECT-ELLIPSE-RADIUS-{axis_name}");
        let disconnect_id = format!("GRAPHITE-GDD-DISCONNECT-ELLIPSE-RADIUS-{axis_name}");
        let source_label = connected
            .and_then(|id| form.nodes.iter().find(|node| node.storage_id() == id))
            .map(|source| format!("{name} from {}", source.label()));
        let menu = (!sources.is_empty() && self.copy_task.is_none()).then(|| {
            let item = cx.entity().downgrade();
            ContextMenu::build(window, cx, |mut menu, _, _| {
                for (index, source) in sources {
                    let item = item.clone();
                    let target = target.clone();
                    let source = source.clone();
                    let label: SharedString = source.label().to_owned().into();
                    menu = menu.custom_entry(
                        move |_, _| {
                            div()
                                .debug_selector(move || {
                                    format!(
                                        "GRAPHITE-GDD-ELLIPSE-RADIUS-{axis_name}-SOURCE-{index}"
                                    )
                                })
                                .p_2()
                                .child(Label::new(label.clone()))
                                .into_any_element()
                        },
                        move |window, cx| {
                            if let Some(item) = item.upgrade() {
                                item.update(cx, |item, cx| {
                                    item.apply_graph_edit(
                                        GraphiteCopy::EditNode {
                                            node: Box::new(target.clone()),
                                            change: NodeCopyChange::EllipseRadiusSource(
                                                axis,
                                                Some(Box::new(source.clone())),
                                            ),
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
            })
        });
        let target = target.clone();
        Some(
            v_flex()
                .gap_1()
                .when_some(source_label, |row, label| row.child(Label::new(label)))
                .when_some(menu, |row, menu| {
                    let verb = if connected.is_some() {
                        "Replace"
                    } else {
                        "Connect"
                    };
                    row.child(
                        div()
                            .debug_selector({
                                let id = menu_id.clone();
                                move || id.clone()
                            })
                            .child(DropdownMenu::new(menu_id, format!("{verb} {name}"), menu)),
                    )
                })
                .when(connected.is_some(), |row| {
                    row.child(
                        div()
                            .debug_selector({
                                let id = disconnect_id.clone();
                                move || id.clone()
                            })
                            .child(
                                Button::new(disconnect_id, format!("Disconnect {name}"))
                                    .disabled(self.copy_task.is_some())
                                    .on_click(cx.listener(move |item, _, window, cx| {
                                        item.apply_graph_edit(
                                            GraphiteCopy::EditNode {
                                                node: Box::new(target.clone()),
                                                change: NodeCopyChange::EllipseRadiusSource(
                                                    axis, None,
                                                ),
                                            },
                                            window,
                                            cx,
                                        );
                                    })),
                            ),
                    )
                }),
        )
    }

    fn render_artboard_geometry(
        &self,
        form: &NodeCopyForm,
        working: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        v_flex()
            .gap_2()
            .children(
                [
                    (
                        "Location (x, y) in px",
                        "GRAPHITE-GDD-ARTBOARD-LOCATION",
                        &form.position,
                    ),
                    (
                        "Dimensions (w, h) in px",
                        "GRAPHITE-GDD-ARTBOARD-DIMENSIONS",
                        &form.value,
                    ),
                ]
                .map(|(label, selector, field)| {
                    v_flex().gap_1().child(Label::new(label)).child(
                        div()
                            .id(selector)
                            .aria_label(label)
                            .debug_selector(move || selector.into())
                            .child(field.clone()),
                    )
                }),
            )
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-GDD-APPLY-ARTBOARD-GEOMETRY".into())
                    .child(
                        Button::new(
                            "graphite-gdd-apply-artboard-geometry",
                            if working {
                                "Apply Geometry"
                            } else {
                                "Save Geometry in a Copy"
                            },
                        )
                        .disabled(self.copy_task.is_some())
                        .on_click(cx.listener(|item, _, window, cx| {
                            item.change_artboard_geometry(window, cx)
                        })),
                    ),
            )
    }

    fn render_artboard_background(
        &self,
        form: &NodeCopyForm,
        working: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let label = "Background (sRGB RGBA, 0–1)";
        v_flex()
            .gap_1()
            .child(Label::new(label))
            .child(
                div()
                    .id("graphite-gdd-artboard-background")
                    .aria_label(label)
                    .debug_selector(|| "GRAPHITE-GDD-ARTBOARD-BACKGROUND".into())
                    .child(form.artboard_background.clone()),
            )
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-GDD-APPLY-ARTBOARD-BACKGROUND".into())
                    .child(
                        Button::new(
                            "graphite-gdd-apply-artboard-background",
                            if working {
                                "Apply Background"
                            } else {
                                "Save Background in a Copy"
                            },
                        )
                        .disabled(self.copy_task.is_some())
                        .on_click(cx.listener(|item, _, window, cx| {
                            item.change_artboard_background(window, cx)
                        })),
                    ),
            )
    }

    fn render_graphic_content_menu(
        &self,
        form: &NodeCopyForm,
        target: &GddGraphNode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        let sources = form
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, source)| target.graphic_content_input_from(source).is_some())
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
                let label = SharedString::from(source.label().to_owned());
                menu = menu.custom_entry(
                    move |_, _| {
                        div()
                            .debug_selector(move || {
                                format!("GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-{index}")
                            })
                            .p_2()
                            .child(Label::new(label.clone()))
                            .into_any_element()
                    },
                    move |window, cx| {
                        if let Some(item) = item.upgrade() {
                            item.update(cx, |item, cx| {
                                item.apply_graph_edit(
                                    GraphiteCopy::EditNode {
                                        node: Box::new(target.clone()),
                                        change: NodeCopyChange::ConnectGraphicContent(Box::new(
                                            source.clone(),
                                        )),
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
        let label = if self.model.read(cx).working.is_some() {
            "Attach Content"
        } else {
            "Attach Content in a Copy"
        };
        Some(
            div()
                .debug_selector(|| "GRAPHITE-GDD-CONNECT-GRAPHIC-CONTENT".into())
                .child(DropdownMenu::new(
                    "graphite-gdd-connect-graphic-content",
                    label,
                    menu,
                )),
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
                                    .when(network.can_append_ports_in_copy(), |bar| {
                                        bar.children([
                                            (GddNumericImportKind::Number, "GRAPHITE-GDD-ADD-NUMBER-IMPORT", "Add Number Import in a Copy"),
                                            (GddNumericImportKind::NumberList, "GRAPHITE-GDD-ADD-NUMBER-LIST-IMPORT", "Add Number List Import in a Copy"),
                                        ].map(|(kind, selector, label)| {
                                            div().debug_selector(move || selector.into())
                                                .child(Button::new(selector, label)
                                                    .disabled(self.copy_task.is_some())
                                                    .on_click(cx.listener(move |item, _, window, cx| {
                                                        item.append_numeric_import(kind, window, cx);
                                                    })))
                                        })).children([
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
                            let resolved = form.document_run_evaluator.as_ref().zip(selected)
                                .and_then(|(evaluator, node)| evaluator.resolved_ports(node));
                            panel.child(crate::graph::render_saved_graph_ports(
                                network, parent, selected, &form.nodes, resolved.as_ref(),
                            ))
                        })
                    })
                    .when_some(form.clipboard_scope(), |panel, network| {
                        let label = match &form.saved_graph {
                            SavedGraphPanel::Closed | SavedGraphPanel::Failed(_) => "Show Saved Graph",
                            SavedGraphPanel::Loading { .. } => "Cancel Graph Loading",
                            SavedGraphPanel::Ready(_) => "Hide Saved Graph",
                        };
                        let panel = panel.child(h_flex().gap_2().flex_wrap().child(
                            div().debug_selector(|| "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH".into()).child(
                                Button::new("graphite-gdd-toggle-saved-graph", label)
                                    .on_click(cx.listener(|item, _, window, cx| item.toggle_saved_graph(window, cx))),
                            ),
                        ).when(matches!(form.saved_graph, SavedGraphPanel::Ready(_)), |row| {
                            row.child(Self::render_document_run(&form.document_run, cx))
                        }));
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
                                        viewport: &form.saved_graph_viewport,
                                        focus: &form.saved_graph_focus,
                                        layer_focus: &form.saved_layers_focus,
                                        selected: form.selected,
                                        selection: &form.node_selection,
                                        collapsed_layers: &form.collapsed_layers,
                                        pending_source: form.pending_canvas_source,
                                        can_start_connection: self.copy_task.is_none(),
                                        working,
                                        revision: &form.revision,
                                        drag_start: working.then_some(&form.saved_graph_drag_start),
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
                                                Some(Self::graph_node_picker(
                                                    network.can_insert_custom_node_in_copy(),
                                                    move |node, window, cx| {
                                                    if let Err(error) = item.update(cx, |item, cx| {
                                                        if matches!(&item.node_copy, NodeCopyPanel::Ready(form) if
                                                            matches!(form.saved_graph, SavedGraphPanel::Ready(_))
                                                                && form.pending_canvas_source.is_none()
                                                                && form.clipboard_scope().is_some_and(|network| network.id == scope))
                                                        {
                                                            item.insert_working_graph_node(node, position, Some(network.clone()), revision.clone(), window, cx);
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
                            .when_some(selected.boolean_input(), |panel, property| {
                                let node = Box::new(selected.clone());
                                panel.child(
                                    div().debug_selector(|| "GRAPHITE-GDD-BOOLEAN-INPUT".into()).child(
                                        Switch::new("graphite-gdd-boolean-input", ToggleState::from(property.value))
                                            .label(property.name)
                                            .label_position(SwitchLabelPosition::End)
                                            .tab_index(0isize)
                                            .disabled(!working || self.copy_task.is_some())
                                            .on_click(cx.listener(move |item, choice: &ToggleState, window, cx| {
                                                if item.model.read(cx).working.is_some() {
                                                    item.apply_graph_edit(GraphiteCopy::EditNode {
                                                        node: node.clone(),
                                                        change: NodeCopyChange::BooleanInput(choice.selected()),
                                                    }, window, cx);
                                                }
                                            })),
                                    ),
                                )
                            })
                            .when(selected.artboard_geometry().is_some(), |panel| {
                                panel.child(self.render_artboard_geometry(form, working, cx))
                            })
                            .when(selected.artboard_background().is_some(), |panel| {
                                panel.child(self.render_artboard_background(form, working, cx))
                            })
                            .when(selected.ellipse_radii().is_some(), |panel| {
                                panel.child(self.render_ellipse_radii(form, working, cx))
                            })
                            .children([GddEllipseAxis::X, GddEllipseAxis::Y].into_iter().filter_map(|axis| {
                                self.render_ellipse_radius_source(form, selected, axis, window, cx)
                            }))
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
                            .children(self.render_graphic_content_menu(form, selected, window, cx))
                            .when(selected.can_set_document_artboard(), |panel| {
                                let node = Box::new(selected.clone());
                                panel.child(
                                    div().debug_selector(|| "GRAPHITE-GDD-SET-DOCUMENT-ARTBOARD".into()).child(
                                        Button::new(
                                            "graphite-gdd-set-document-artboard",
                                            if working { "Use as Document Artboard" } else { "Set Artboard in a Copy" },
                                        )
                                        .disabled(self.copy_task.is_some())
                                        .on_click(cx.listener(move |item, _, window, cx| {
                                            item.apply_graph_edit(GraphiteCopy::EditNode {
                                                node: node.clone(),
                                                change: NodeCopyChange::SetDocumentArtboard,
                                            }, window, cx);
                                        })),
                                    ),
                                )
                            })
                            .when(selected.can_preview_numeric_output() || selected.layer_preview().is_some(), |panel| {
                                panel.child(Self::render_node_output_preview(&form.output_preview, selected.layer_preview(), cx))
                            })
                            .when(matches!(selected.layer_preview(), Some(GddGraphLayerPreview::Artboard { .. })), |panel| {
                                let node = Box::new(selected.clone());
                                panel.child(
                                    div().debug_selector(|| "GRAPHITE-GDD-EXPORT-ARTBOARD-SVG".into()).child(
                                        Button::new("graphite-gdd-export-artboard-svg", "Export Artboard SVG")
                                            .disabled(self.copy_task.is_some())
                                            .on_click(cx.listener(move |item, _, window, cx| {
                                                item.save_copy(GraphiteCopy::ExportArtboardSvg(node.clone()), window, cx);
                                            })),
                                    ),
                                )
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
        let grid = model
            .snapping_preferences
            .as_ref()
            .filter(|_| model.grid_options_edited)
            .map(|preferences| preferences.grid.clone());
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
                    async move {
                        match grid {
                            Some(grid) => Ok(snapshot
                                .folder_files_with_grid_options(&grid, ARCHIVE_BUDGET)
                                .await?),
                            None => anyhow::Ok(snapshot.folder_files()?),
                        }
                    }
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
        let artboard_svg = matches!(&request, GraphiteCopy::ExportArtboardSvg(_));
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
        let artboard_revision = artboard_svg.then(|| model.revision.clone());
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
                let suffix = if artboard_svg {
                    ".svg"
                } else {
                    match format {
                        GddCopyFormat::Archive(ArchiveFormat::Zip) => ".gdd.zip",
                        GddCopyFormat::Archive(ArchiveFormat::Xz) => ".gdd.tar.xz",
                        GddCopyFormat::Folder => ".gdd",
                    }
                };
                let prefix = match &request {
                    GraphiteCopy::Original | GraphiteCopy::Export(_) => "Copy of",
                    GraphiteCopy::ExportArtboardSvg(_) => "Artboard from",
                    GraphiteCopy::RecoverFolder => "Recovered",
                    GraphiteCopy::AppendNetworkPort {
                        kind: NetworkPortKind::Import,
                        ..
                    }
                    | GraphiteCopy::AppendNumericImport { .. } => "Added import in",
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
                        change:
                            NodeCopyChange::NumericInput(_)
                            | NodeCopyChange::BooleanInput(_)
                            | NodeCopyChange::ArtboardGeometry(_)
                            | NodeCopyChange::ArtboardBackground(_)
                            | NodeCopyChange::EllipseRadius(_, _)
                            | NodeCopyChange::EllipseRadiusSource(_, _),
                        ..
                    } => "Edited",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::SetDocumentArtboard,
                        ..
                    } => "Set Artboard in",
                    GraphiteCopy::EditNode {
                        change: NodeCopyChange::ConnectGraphicContent(_),
                        ..
                    } => "Connected graphic content in",
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
                            GraphiteCopy::ExportArtboardSvg(node) => {
                                let output = futures::executor::block_on(
                                    archive.export_document_artboard_svg(
                                        &node,
                                        CpuSvgEvaluationBudget {
                                            max_viewport_pixels: 16 * 1024 * 1024,
                                            max_output_bytes: 64 * 1024 * 1024,
                                        },
                                    ),
                                )?;
                                let RenderOutputType::Svg { svg, .. } = output.data else {
                                    anyhow::bail!("Graphite did not return an Artboard SVG.");
                                };
                                return Ok(PreparedCopy::File(svg.into_bytes().into()));
                            }
                            GraphiteCopy::RecoverFolder => {
                                Some(archive.recover_pending_edits(ARCHIVE_BUDGET).await?)
                            }
                            GraphiteCopy::MoveHistory(direction) => Some(
                                archive
                                    .move_history_in_copy(direction, ARCHIVE_BUDGET)
                                    .await?,
                            ),
                            GraphiteCopy::AppendNumericImport { network, kind } => Some(
                                archive
                                    .append_nested_numeric_import_in_copy(
                                        &network,
                                        kind,
                                        ARCHIVE_BUDGET,
                                    )
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
                ensure!(
                    item.read_with(cx, |item, cx| {
                        artboard_revision.as_ref().is_none_or(|revision| {
                            Arc::ptr_eq(revision, &item.model.read(cx).revision)
                        })
                    })?,
                    "The Graphite Artboard changed during export. Try again."
                );
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
                        let message = if artboard_svg {
                            format!("Artboard SVG saved to {}", path.display())
                        } else {
                            format!("Copy saved to {}", path.display())
                        };
                        Some((message.into(), Color::Default))
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
    fn deactivated(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.finish_scene_pan(true, window, cx);
    }

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
            .key_context(match self.scene_canvas.as_ref() {
                Some(canvas) if canvas.pan.is_some() => {
                    "GraphiteGdd GraphiteCanvas GraphiteCanvasPan"
                }
                Some(canvas)
                    if canvas.artboard_drag.is_some() || canvas.artboard_draw.is_some() =>
                {
                    "GraphiteGdd GraphiteCanvas GraphiteCanvasArtboardEdit"
                }
                Some(_) => "GraphiteGdd GraphiteCanvas",
                _ => "GraphiteGdd",
            })
            .on_action(cx.listener(|item, _: &ToggleCanvasSnapping, _, cx| {
                if let Some(canvas) = &item.scene_canvas
                    && item.toggle_scene_snapping(
                        &canvas.instance,
                        &canvas.revision,
                        CANVAS_SNAP_ENABLED,
                        cx,
                    )
                {
                    cx.stop_propagation();
                } else {
                    cx.propagate();
                }
            }))
            .on_action(cx.listener(|item, _: &ToggleCanvasGrid, _, cx| {
                if let Some(canvas) = &item.scene_canvas
                    && item.toggle_scene_grid(&canvas.instance, &canvas.revision, cx)
                {
                    cx.stop_propagation();
                } else {
                    cx.propagate();
                }
            }))
            .on_action(cx.listener(|item, _: &ToggleCanvasOverlays, _, cx| {
                if let Some(canvas) = &item.scene_canvas
                    && item.toggle_scene_overlay(
                        &canvas.instance,
                        &canvas.revision,
                        CANVAS_OVERLAYS_ALL,
                        cx,
                    )
                {
                    cx.stop_propagation();
                } else {
                    cx.propagate();
                }
            }))
            .on_action(
                cx.listener(|item, _: &CancelArtboardCanvasEdit, window, cx| {
                    if item.cancel_scene_artboard_edit(window, cx) {
                        cx.stop_propagation();
                    } else {
                        cx.propagate();
                    }
                }),
            )
            .on_action(
                cx.listener(|item, _: &CancelArtboardCanvasPan, window, cx| {
                    if item
                        .scene_canvas
                        .as_ref()
                        .is_some_and(|canvas| canvas.pan.is_some())
                    {
                        item.finish_scene_pan(true, window, cx);
                    } else {
                        cx.propagate();
                    }
                }),
            )
            .on_action(cx.listener(|item, _: &editor::actions::Undo, window, cx| {
                item.route_working_history(GddHistoryDirection::Undo, window, cx);
            }))
            .on_action(cx.listener(|item, _: &editor::actions::Redo, window, cx| {
                item.route_working_history(GddHistoryDirection::Redo, window, cx);
            }))
            .size_full();
        if let Some(canvas) = &self.scene_canvas {
            return view
                .child(self.render_scene_canvas(canvas, window, cx))
                .into_any_element();
        }
        let view = view
            .p_6()
            .gap_2()
            .overflow_y_scroll()
            .child(Label::new(model.name.clone()).size(LabelSize::Large))
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        div().debug_selector(|| "GRAPHITE-SAVE-COPY".into()).child(
                            Button::new("graphite-save-copy", "Save a Copy")
                                .disabled(self.copy_task.is_some())
                                .on_click(cx.listener(|item, _, window, cx| {
                                    item.save_copy(GraphiteCopy::Original, window, cx)
                                })),
                        ),
                    )
                    .when(
                        model
                            .gdd_contents()
                            .is_some_and(|contents| contents.hot_log_len() == 0),
                        |toolbar| {
                            toolbar.child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-OPEN-CANVAS".into())
                                    .child(
                                        Button::new("graphite-gdd-open-canvas", "Open Canvas")
                                            .on_click(cx.listener(|item, _, window, cx| {
                                                item.open_scene_canvas(window, cx)
                                            })),
                                    ),
                            )
                        },
                    ),
            )
            .when_some(self.copy_feedback.clone(), |view, (message, color)| {
                view.child(Label::new(message).color(color))
            });
        let view = match &model.document {
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
        };
        view.into_any_element()
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
        dispatch_graph_click(window, position, button, modifiers);
        window.run_until_parked();
        paint(window);
    }

    fn dispatch_graph_click(
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
        RootExport,
        WideRootExport,
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

        struct NumericPositions {
            positioned: bool,
            wide: bool,
        }

        impl NodeMetadataSource for NumericPositions {
            fn position(&self, _: &[NodeId], node: NodeId) -> Option<Position> {
                if !self.positioned {
                    return None;
                }
                let grid = match node {
                    NodeId(7) => [0, 0],
                    NodeId(8) if self.wide => [42, 2],
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
            | NumericGraphFixture::NestedConnection
            | NumericGraphFixture::RootExport
            | NumericGraphFixture::WideRootExport => Some(NodeInput::node(NodeId(7), 0)),
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
            NumericGraphFixture::Reconnection
                | NumericGraphFixture::ExposedInputWithAlternative
                | NumericGraphFixture::RootExport
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
        if matches!(
            fixture,
            NumericGraphFixture::NestedConnection
                | NumericGraphFixture::RootExport
                | NumericGraphFixture::WideRootExport
        ) {
            graph.exports = vec![NodeInput::node(NodeId(8), 0)];
        }
        if matches!(fixture, NumericGraphFixture::NestedConnection) {
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
                &NumericPositions {
                    positioned,
                    wide: matches!(fixture, NumericGraphFixture::WideRootExport),
                },
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
        graph_navigation_archive_with_options(false, 0).await
    }

    async fn graph_layer_archive() -> (Vec<u8>, usize) {
        graph_layer_archive_with_grandchild(false).await
    }

    async fn graph_layer_archive_with_grandchild(with_grandchild: bool) -> (Vec<u8>, usize) {
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
                    NodeId(35) => Some("Grandchild"),
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
                    NodeId(35) => [-8, 0],
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
        if with_grandchild {
            *hidden
                .inputs
                .get_mut(1)
                .expect("pinned Merge content input") = NodeInput::node(NodeId(35), 0);
        }
        let mut graph = NodeNetwork {
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
        if with_grandchild {
            graph
                .nodes
                .insert(NodeId(35), merge.node_template.clone().into_parts().0);
        }
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

    async fn graph_navigation_archive_with_options(
        number_target: bool,
        root_output: usize,
    ) -> (Vec<u8>, usize) {
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
            exports: vec![NodeInput::node(NodeId(24), root_output)],
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
            graph_navigation_archive_with_options(!reconnect, 0).await;
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

        let (bytes, _) = graph_navigation_archive_with_options(true, 0).await;
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
    async fn at_gra_layer_002_2_workspace_extends_and_range_selects_saved_layers(
        cx: &mut TestAppContext,
    ) {
        let (original, history) = graph_layer_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/saved-layer-selection.gdd.zip");
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
            .expect("open the layer graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let revision = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        let choose = |window: &mut gpui::VisualTestContext,
                      selector,
                      modifiers,
                      expected: &[usize],
                      anchor| {
            click_saved_graph(window, selector, gpui::MouseButton::Left, modifiers);
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Inspector")
                };
                let mut selection = form.node_selection.clone();
                selection.extend(form.selected);
                assert!(selection.iter().copied().eq(expected.iter().copied()));
                assert_eq!(form.layer_selection_anchor, Some(anchor));
                assert!(!item.is_dirty(cx));
                let model = item.model.read(cx);
                assert!(Arc::ptr_eq(&model.revision, &revision));
                assert_eq!(model.gdd_contents().expect("GDD").history_len(), history);
            });
        };
        let parent = "GRAPHITE-GDD-SAVED-LAYER-1";
        let child = "GRAPHITE-GDD-SAVED-LAYER-2";
        let sibling = "GRAPHITE-GDD-SAVED-LAYER-0";
        let accelerator = gpui::Modifiers {
            platform: cfg!(target_os = "macos"),
            control: !cfg!(target_os = "macos"),
            ..Default::default()
        };
        let shift = gpui::Modifiers {
            shift: true,
            ..Default::default()
        };
        choose(window, parent, Default::default(), &[1], 1);
        choose(window, sibling, accelerator, &[0, 1], 0);
        choose(window, child, shift, &[0, 2], 0);
        choose(
            window,
            parent,
            gpui::Modifiers {
                shift: true,
                ..accelerator
            },
            &[0, 1, 2],
            0,
        );
        choose(window, child, accelerator, &[0, 1], 2);
        choose(
            window,
            parent,
            gpui::Modifiers {
                alt: true,
                ..Default::default()
            },
            &[0, 1],
            2,
        );
        choose(window, sibling, shift, &[0, 2], 2);
        choose(window, child, accelerator, &[0], 2);
        choose(window, sibling, accelerator, &[], 0);
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_layer_005_2_workspace_selects_saved_layer_parents(cx: &mut TestAppContext) {
        let (original, history) = graph_layer_archive_with_grandchild(true).await;
        let (workspace, window) = native_workspace(cx).await;
        window.update(|_, cx| {
            let bindings =
                settings::KeymapFile::load_asset(crate::DEFAULT_GRAPHITE_KEYMAP_PATH, None, cx)
                    .expect("Graphite keymap");
            cx.bind_keys(bindings);
        });
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/saved-layer-parents.gdd.zip");
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
            .expect("open the layer graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        let revision = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready layer Inspector")
            };
            let SavedGraphPanel::Ready(graph) = &form.saved_graph else {
                panic!("ready graph canvas")
            };
            let [parent, child, grandchild, sibling] = graph
                .root_layers
                .as_ref()
                .expect("saved hierarchy")
                .as_slice()
            else {
                panic!("parent, child, grandchild and top-level sibling")
            };
            assert_eq!(parent.parent, None);
            assert_eq!(child.parent, Some(parent.storage_id));
            assert_eq!(grandchild.parent, Some(child.storage_id));
            assert_eq!(sibling.parent, None);
            item.model.read(cx).revision.clone()
        });
        let check = |window: &gpui::VisualTestContext, expected: &[usize], active| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready layer Inspector")
                };
                let mut selection = form.node_selection.clone();
                selection.extend(form.selected);
                assert_eq!(selection.into_iter().collect::<Vec<_>>(), expected);
                assert_eq!(form.selected, Some(active));
                assert_eq!(form.layer_selection_anchor, Some(3));
                assert!(!item.is_dirty(cx));
                assert!(Arc::ptr_eq(&item.model.read(cx).revision, &revision));
                assert_eq!(
                    item.model
                        .read(cx)
                        .gdd_contents()
                        .expect("GDD")
                        .history_len(),
                    history
                );
            });
        };
        click(window, "GRAPHITE-GDD-SAVED-LAYER-3");
        paint(window);
        check(window, &[3], 3);
        for parent in [2, 1, 1] {
            window.simulate_keystrokes("shift-escape");
            window.run_until_parked();
            paint(window);
            check(window, &[parent], parent);
        }
        click(window, "GRAPHITE-GDD-SAVED-LAYER-0");
        paint(window);
        click_saved_graph(
            window,
            "GRAPHITE-GDD-SAVED-LAYER-3",
            gpui::MouseButton::Left,
            gpui::Modifiers {
                platform: cfg!(target_os = "macos"),
                control: !cfg!(target_os = "macos"),
                ..Default::default()
            },
        );
        check(window, &[0, 3], 3);
        click(window, "GRAPHITE-GDD-SELECT-LAYER-PARENT");
        paint(window);
        check(window, &[0, 2], 2);
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_layer_003_004_2_workspace_selects_eligible_layers_and_deselects_locally(
        cx: &mut TestAppContext,
    ) {
        let (original, history) = graph_layer_archive_with_grandchild(true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let primary_path = PathBuf::from("/project/select-all-layers.gdd.zip");
        let other_path = PathBuf::from("/project/other-layer-document.gdd.zip");
        for path in [&primary_path, &other_path] {
            filesystem
                .write(path, &original)
                .await
                .expect("source archive");
        }
        window.run_until_parked();
        let open = |path: PathBuf, window: &mut gpui::VisualTestContext| {
            workspace.update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path, Default::default(), window, cx)
            })
        };
        let open_layers = |window: &mut gpui::VisualTestContext| {
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
            click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-GDD-SAVED-LAYER-2");
            paint(window);
        };
        let check = |window: &gpui::VisualTestContext,
                     item: &Entity<GraphiteDocumentItem>,
                     expected: &[usize],
                     anchor,
                     revision: &Arc<()>| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready layer Inspector")
                };
                let SavedGraphPanel::Ready(graph) = &form.saved_graph else {
                    panic!("ready layer graph")
                };
                let layers = graph.root_layers.as_ref().expect("pinned layer projection");
                assert_eq!(
                    layers
                        .iter()
                        .map(|layer| layer.select_all_eligible)
                        .collect::<Vec<_>>(),
                    [true, false, false, true]
                );
                let mut selection = form.node_selection.clone();
                selection.extend(form.selected);
                assert!(selection.iter().copied().eq(expected.iter().copied()));
                assert_eq!(form.layer_selection_anchor, anchor);
                assert!(!item.is_dirty(cx));
                assert!(Arc::ptr_eq(&item.model.read(cx).revision, revision));
                assert_eq!(
                    item.model
                        .read(cx)
                        .gdd_contents()
                        .expect("GDD")
                        .history_len(),
                    history
                );
            });
        };
        let first = open(primary_path.clone(), window)
            .await
            .expect("open first GDD")
            .downcast::<GraphiteDocumentItem>()
            .expect("first native Inspector");
        open_layers(window);
        let first_revision =
            first.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        let platform = if cfg!(target_os = "macos") {
            "cmd"
        } else {
            "ctrl"
        };
        window.simulate_keystrokes(&format!("{platform}-a"));
        window.run_until_parked();
        paint(window);
        check(window, &first, &[0, 1], Some(2), &first_revision);

        let other = open(other_path.clone(), window)
            .await
            .expect("open independent GDD")
            .downcast::<GraphiteDocumentItem>()
            .expect("other native Inspector");
        assert_ne!(first, other);
        open_layers(window);
        let other_revision =
            other.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        check(window, &other, &[2], Some(2), &other_revision);
        window.simulate_keystrokes(&format!("{platform}-shift-a"));
        window.run_until_parked();
        paint(window);
        check(window, &other, &[], None, &other_revision);
        check(window, &first, &[0, 1], Some(2), &first_revision);
        for path in [&primary_path, &other_path] {
            assert_eq!(
                filesystem.load_bytes(path).await.expect("original archive"),
                original
            );
        }
    }

    #[gpui::test]
    async fn at_gra_layer_018_2_workspace_collapses_and_expands_saved_layer_subtrees(
        cx: &mut TestAppContext,
    ) {
        let (original, history) = graph_layer_archive_with_grandchild(true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/saved-layer-expansion.gdd.zip");
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
            .expect("open the saved layer hierarchy")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-SAVED-LAYER-2");
        paint(window);
        let (revision, selected) = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready Inspector")
            };
            assert_eq!(form.layer_selection_anchor, Some(2));
            (
                item.model.read(cx).revision.clone(),
                form.selected_node()
                    .expect("selected hidden child")
                    .storage_id(),
            )
        });
        let check = |window: &mut gpui::VisualTestContext, child, grandchild| {
            assert!(window.debug_bounds("GRAPHITE-GDD-SAVED-LAYER-0").is_some());
            assert!(window.debug_bounds("GRAPHITE-GDD-SAVED-LAYER-1").is_some());
            assert_eq!(
                window.debug_bounds("GRAPHITE-GDD-SAVED-LAYER-2").is_some(),
                child
            );
            assert_eq!(
                window.debug_bounds("GRAPHITE-GDD-SAVED-LAYER-3").is_some(),
                grandchild
            );
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Inspector")
                };
                assert_eq!(
                    form.selected_node().map(GddGraphNode::storage_id),
                    Some(selected)
                );
                assert_eq!(form.layer_selection_anchor, Some(2));
                assert!(!item.is_dirty(cx));
                let model = item.model.read(cx);
                assert!(Arc::ptr_eq(&model.revision, &revision));
                assert_eq!(model.gdd_contents().expect("GDD").history_len(), history);
            });
        };
        check(window, true, true);
        let recursive = gpui::Modifiers {
            alt: true,
            ..Default::default()
        };
        let accelerator = gpui::Modifiers {
            platform: cfg!(target_os = "macos"),
            control: !cfg!(target_os = "macos"),
            ..Default::default()
        };
        for (modifiers, child, grandchild) in [
            (recursive, false, false),
            (Default::default(), true, false),
            (Default::default(), false, false),
            (accelerator, true, true),
        ] {
            click_saved_graph(
                window,
                "GRAPHITE-GDD-LAYER-EXPANSION-1",
                gpui::MouseButton::Left,
                modifiers,
            );
            check(window, child, grandchild);
        }
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_layer_019_020_2_workspace_edits_working_layer_visibility_and_locking(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = graph_layer_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/working-layer-settings.gdd.zip");
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
            .expect("open the layer graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        let invoke = |window: &mut gpui::VisualTestContext, selector| {
            paint(window);
            click(window, selector);
            window.run_until_parked();
            paint(window);
        };
        invoke(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-LAYER-VISIBILITY-2")
                .is_none()
        );
        assert!(window.debug_bounds("GRAPHITE-GDD-LAYER-LOCK-2").is_none());
        let (initial_revision, parent_id, child_id, sibling_id) =
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Inspector")
                };
                let [sibling, parent, child, ..] = form.nodes.as_slice() else {
                    panic!("root layers")
                };
                assert_eq!(
                    form.selected_node().map(GddGraphNode::storage_id),
                    Some(parent.storage_id())
                );
                (
                    item.model.read(cx).revision.clone(),
                    parent.storage_id(),
                    child.storage_id(),
                    sibling.storage_id(),
                )
            });
        let check = |graph: &GddGraphGeometry, visible, locked| {
            let layers = graph.root_layers.as_ref().expect("saved layer tree");
            assert_eq!(
                layers
                    .iter()
                    .map(|layer| (layer.storage_id, layer.depth, layer.visible, layer.locked))
                    .collect::<Vec<_>>(),
                [
                    (parent_id, 0, true, false),
                    (child_id, 1, visible, locked),
                    (sibling_id, 0, true, false)
                ]
            );
        };
        let check_view = |window: &gpui::VisualTestContext, visible, locked, edited: bool| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Inspector")
                };
                let SavedGraphPanel::Ready(graph) = &form.saved_graph else {
                    panic!("working graph")
                };
                assert_eq!(
                    form.selected_node().map(GddGraphNode::storage_id),
                    Some(parent_id)
                );
                assert_eq!(
                    Arc::ptr_eq(&item.model.read(cx).revision, &initial_revision),
                    !edited
                );
                assert_eq!(item.is_dirty(cx), edited);
                check(graph, visible, locked);
            });
        };
        invoke(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        check_view(window, false, true, false);
        for (selector, visible, locked) in [
            ("GRAPHITE-GDD-LAYER-VISIBILITY-2", true, true),
            ("GRAPHITE-GDD-LAYER-LOCK-2", true, false),
            ("GRAPHITE-GDD-UNDO", true, true),
            ("GRAPHITE-GDD-UNDO", false, true),
            ("GRAPHITE-GDD-REDO", true, true),
            ("GRAPHITE-GDD-REDO", true, false),
        ] {
            invoke(window, selector);
            check_view(window, visible, locked, true);
        }
        let destination = PathBuf::from("/project/saved-working-layer-settings.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen working layer settings");
        let inspection = saved.graph_inspection().await.expect("saved graph");
        let root = inspection
            .networks
            .iter()
            .find(|network| network.parent.is_none())
            .expect("saved root scope");
        check(
            &saved.graph_geometry(root).await.expect("saved layers"),
            true,
            false,
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
    async fn at_gra_graph_037_2_workspace_runs_the_current_document_output(
        cx: &mut TestAppContext,
    ) {
        let (original, history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::RootExport, true).await;
        let (unsupported, _) = graph_layer_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/manual-graph-run.gdd.zip");
        let unsupported_path = PathBuf::from("/project/manual-unsupported-graph-run.gdd.zip");
        for (source, bytes) in [(&path, &original), (&unsupported_path, &unsupported)] {
            filesystem.write(source, bytes).await.expect("source GDD");
        }
        window.run_until_parked();
        let open = |path: PathBuf, window: &mut gpui::VisualTestContext| {
            workspace.update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path, Default::default(), window, cx)
            })
        };
        let invoke = |window: &mut gpui::VisualTestContext, selector| {
            paint(window);
            click(window, selector);
            window.run_until_parked();
            paint(window);
        };
        let item = open(path.clone(), window)
            .await
            .expect("open the saved output")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
        invoke(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        let (original_revision, exported_id) = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            assert_eq!(form.selected, Some(2));
            let root = form
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("document network");
            let (node, output_index) = root
                .connected_export(&form.nodes)
                .expect("linked document output");
            assert_eq!(output_index, 0);
            assert_ne!(
                form.selected_node()
                    .expect("unrelated selection")
                    .storage_id(),
                node.storage_id()
            );
            (item.model.read(cx).revision.clone(), node.storage_id())
        });
        let check = |window: &gpui::VisualTestContext, expected, revision: &Arc<()>| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("ready graph") };
                assert!(matches!(&form.document_run, NodeOutputPreview::Ready(GddNumericValue::Integer(value)) if *value == expected));
                assert!(matches!(&form.output_preview, NodeOutputPreview::Hidden));
                assert!(Arc::ptr_eq(&item.model.read(cx).revision, revision));
                let root = form.networks.iter().find(|network| network.parent.is_none()).expect("document network");
                assert_eq!(root.connected_export(&form.nodes).map(|(node, index)| (node.storage_id(), index)), Some((exported_id, 0)));
            });
        };
        invoke(window, "GRAPHITE-GDD-RUN-GRAPH");
        check(window, 17, &original_revision);
        item.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("GDD")
                    .history_len(),
                history
            );
        });

        invoke(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        invoke(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        invoke(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let field = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            form.value.clone()
        });
        field.update_in(window, |editor, window, cx| {
            editor.set_text("-29", window, cx)
        });
        invoke(window, "GRAPHITE-GDD-APPLY-NUMERIC-INPUT");
        let working_revision = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            assert!(matches!(&form.document_run, NodeOutputPreview::Hidden));
            assert!(item.is_dirty(cx));
            item.model.read(cx).revision.clone()
        });
        assert!(!Arc::ptr_eq(&working_revision, &original_revision));
        invoke(window, "GRAPHITE-GDD-RUN-GRAPH");
        check(window, 29, &working_revision);
        invoke(window, "GRAPHITE-GDD-UNDO");
        let undo_revision = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        invoke(window, "GRAPHITE-GDD-RUN-GRAPH");
        check(window, 17, &undo_revision);

        let unsupported_item = open(unsupported_path.clone(), window)
            .await
            .expect("open unsupported output")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        invoke(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        let revision =
            unsupported_item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        invoke(window, "GRAPHITE-GDD-RUN-GRAPH");
        unsupported_item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            let NodeOutputPreview::Failed(error) = &form.document_run else {
                panic!("node-addressed output error")
            };
            assert!(error.contains("Artwork [20]"));
            assert!(Arc::ptr_eq(&item.model.read(cx).revision, &revision));
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("original numeric archive"),
            original
        );
        assert_eq!(
            filesystem
                .load_bytes(&unsupported_path)
                .await
                .expect("original layer archive"),
            unsupported
        );
    }

    #[gpui::test]
    async fn at_gra_graph_038_2_workspace_forces_a_fresh_numeric_graph_evaluator(
        cx: &mut TestAppContext,
    ) {
        let (original, history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::RootExport, true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/force-numeric-graph-run.gdd.zip");
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
            .expect("open the document")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
        let invoke = |window: &mut gpui::VisualTestContext, selector| {
            paint(window);
            click(window, selector);
            window.run_until_parked();
            paint(window);
        };
        invoke(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        let original_revision =
            item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        let evaluator = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready graph")
                };
                assert!(matches!(
                    form.document_run,
                    NodeOutputPreview::Ready(GddNumericValue::Integer(17))
                ));
                assert!(Arc::ptr_eq(
                    &item.model.read(cx).revision,
                    &original_revision
                ));
                assert_eq!(
                    item.model
                        .read(cx)
                        .gdd_contents()
                        .expect("GDD")
                        .history_len(),
                    history
                );
                assert!(!item.is_dirty(cx));
                form.document_run_evaluator
                    .clone()
                    .expect("current typed evaluator")
            })
        };
        invoke(window, "GRAPHITE-GDD-RUN-GRAPH");
        let initial = evaluator(window);
        invoke(window, "GRAPHITE-GDD-RUN-GRAPH");
        assert!(Arc::ptr_eq(&initial, &evaluator(window)));

        let click_before_completion = |window: &mut gpui::VisualTestContext, selector| {
            paint(window);
            let position = window
                .debug_bounds(selector)
                .expect("visible run button")
                .center();
            dispatch_graph_click(
                window,
                position,
                gpui::MouseButton::Left,
                gpui::Modifiers::default(),
            );
        };
        click_before_completion(window, "GRAPHITE-GDD-FORCE-RUN-GRAPH");
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            assert!(matches!(
                form.document_run,
                NodeOutputPreview::Pending { .. }
            ));
        });
        click_before_completion(window, "GRAPHITE-GDD-RUN-GRAPH");
        window.run_until_parked();
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-GRAPH-RUN-RESULT")
                .is_some()
        );
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            assert!(matches!(form.document_run, NodeOutputPreview::Cancelled));
            assert!(Arc::ptr_eq(
                form.document_run_evaluator
                    .as_ref()
                    .expect("last successful evaluator"),
                &initial
            ));
            assert!(Arc::ptr_eq(
                &item.model.read(cx).revision,
                &original_revision
            ));
            assert!(!item.is_dirty(cx));
        });

        invoke(window, "GRAPHITE-GDD-FORCE-RUN-GRAPH");
        let forced = evaluator(window);
        assert!(!Arc::ptr_eq(&initial, &forced));
        invoke(window, "GRAPHITE-GDD-RUN-GRAPH");
        assert!(Arc::ptr_eq(&forced, &evaluator(window)));
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("unchanged source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_037_2_workspace_runs_the_wired_nested_export(cx: &mut TestAppContext) {
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let invoke = |window: &mut gpui::VisualTestContext, selector| {
            paint(window);
            click(window, selector);
            window.run_until_parked();
            paint(window);
        };
        for (ordinal, value) in [(0, 17.), (1, 23.)] {
            let (original, history) = graph_navigation_archive_with_options(false, ordinal).await;
            let path = PathBuf::from(format!("/project/nested-export-{ordinal}.gdd.zip"));
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
                .expect("open the document")
                .downcast::<GraphiteDocumentItem>()
                .expect("native Inspector");
            choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-7");
            invoke(window, "GRAPHITE-GDD-ENTER-NETWORK");
            let revision = item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready graph")
                };
                let scope = form.active_network().expect("unrelated nested scope");
                assert_eq!((scope.import_count, scope.export_count), (0, 0));
                item.model.read(cx).revision.clone()
            });
            invoke(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
            for selector in ["GRAPHITE-GDD-RUN-GRAPH", "GRAPHITE-GDD-FORCE-RUN-GRAPH"] {
                invoke(window, selector);
                item.read_with(window, |item, cx| {
                    let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("ready graph") };
                    assert!(matches!(form.document_run, NodeOutputPreview::Ready(GddNumericValue::Number(result)) if result == value),
                        "expected the stored document export {ordinal}, resolved as {:?}", form.document_run.feedback());
                    assert!(Arc::ptr_eq(&item.model.read(cx).revision, &revision));
                    assert!(!item.is_dirty(cx));
                    assert_eq!(item.model.read(cx).gdd_contents().expect("GDD").history_len(), history);
                    assert!(form.selected.is_none());
                });
                assert!(
                    window
                        .debug_bounds("GRAPHITE-GDD-GRAPH-RUN-RESULT")
                        .is_some()
                );
            }
            assert_eq!(
                filesystem
                    .load_bytes(&path)
                    .await
                    .expect("unchanged source ZIP"),
                original
            );
        }
    }

    #[gpui::test]
    async fn at_gra_graph_037_2_workspace_reports_the_failed_nested_node(cx: &mut TestAppContext) {
        use graphite_document_graph_storage::NodeMetadataSource;
        use graphite_glam::DVec2;
        use graphite_graph_craft::document::{
            DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
            value::TaggedValue,
        };
        use graphite_resource::ResourceRegistry;
        use graphite_std::math_nodes::{absolute_value, add};

        struct ArithmeticNames;

        impl NodeMetadataSource for ArithmeticNames {
            fn display_name(&self, _: &[NodeId], node: NodeId) -> Option<&str> {
                (node == NodeId(7)).then_some("Arithmetic")
            }
        }

        let literal = |value| NodeInput::value(value, false);
        let operation = |identifier, inputs| DocumentNode {
            implementation: DocumentNodeImplementation::ProtoNode(identifier),
            inputs,
            ..Default::default()
        };
        let nested = NodeNetwork {
            nodes: [
                (
                    NodeId(7),
                    operation(
                        add::IDENTIFIER,
                        vec![
                            literal(TaggedValue::Integer(3)),
                            literal(TaggedValue::DVec2(DVec2::new(1., 2.))),
                        ],
                    ),
                ),
                (
                    NodeId(8),
                    operation(
                        absolute_value::IDENTIFIER,
                        vec![NodeInput::node(NodeId(7), 0)],
                    ),
                ),
            ]
            .into_iter()
            .collect(),
            exports: vec![NodeInput::node(NodeId(8), 0)],
            ..Default::default()
        };
        let root = NodeNetwork {
            nodes: [
                (
                    NodeId(7),
                    operation(
                        absolute_value::IDENTIFIER,
                        vec![literal(TaggedValue::Number(-17.))],
                    ),
                ),
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
            exports: vec![NodeInput::node(NodeId(24), 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            690,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("source GDD");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(
                &root,
                &ArithmeticNames,
                &ResourceRegistry::new(),
                &resources,
            )
            .expect("saved typed inputs with an incompatible combination");
        let history = owner.session().history().count();
        let original = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/nested-graph-node-diagnostic.gdd.zip");
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
            .expect("open the document")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let revision = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        for selector in ["GRAPHITE-GDD-TOGGLE-SAVED-GRAPH", "GRAPHITE-GDD-RUN-GRAPH"] {
            paint(window);
            click(window, selector);
            window.run_until_parked();
        }
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-GRAPH-RUN-RESULT")
                .is_some()
        );
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            let NodeOutputPreview::Failed(error) = &form.document_run else {
                panic!("expected a pinned typing failure")
            };
            assert!(error.contains("Network 24 · Arithmetic [7]"), "{error}");
            assert!(error.contains("incompatible"), "{error}");
            assert!(!error.contains("Root graph · Arithmetic [7]"), "{error}");
            assert!(Arc::ptr_eq(&item.model.read(cx).revision, &revision));
            assert!(!item.is_dirty(cx));
            assert!(form.document_run_evaluator.is_none());
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("GDD")
                    .history_len(),
                history
            );
        });
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("unchanged source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_039_2_workspace_shows_compiler_resolved_numeric_ports(
        cx: &mut TestAppContext,
    ) {
        use graphite_graph_craft::{Type, concrete};

        let (original, history) = graph_navigation_archive().await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/resolved-numeric-ports.gdd.zip");
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
            .expect("open the document")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_scoped_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
        let revision = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-PORTS");
        assert!(window.debug_bounds("GRAPHITE-GDD-SAVED-Input-1").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-RESOLVED-OUTPUT")
                .is_none()
        );
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        click(window, "GRAPHITE-GDD-RUN-GRAPH");
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-RESOLVED-OUTPUT")
                .is_some()
        );
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            let evaluator = form
                .document_run_evaluator
                .as_ref()
                .expect("compiled output");
            let selected = form.selected_node().expect("nested numeric node");
            assert_eq!(selected.name(), "Repeated value");
            let resolved = evaluator
                .resolved_ports(selected)
                .expect("resolved selected ports");
            let [input] = resolved.inputs.as_slice() else {
                panic!("only the saved input slot")
            };
            for ty in [input, &resolved.output] {
                assert!(
                    matches!(ty.nested_type(), Type::Item(element) if **element == concrete!(f64)),
                    "expected the pinned Item<f64> type, got {ty}"
                );
            }
            let same_name = form
                .nodes
                .iter()
                .find(|node| {
                    node.name() == selected.name() && node.network_id() != selected.network_id()
                })
                .expect("same name in a different saved scope");
            assert!(evaluator.resolved_ports(same_name).is_none());
            assert!(form.nodes.iter().any(|node| {
                node.network_id() == selected.network_id()
                    && node.storage_id() != selected.storage_id()
                    && evaluator.resolved_ports(node).is_none()
            }));
            assert!(Arc::ptr_eq(&item.model.read(cx).revision, &revision));
            assert!(!item.is_dirty(cx));
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("GDD")
                    .history_len(),
                history
            );
        });
        assert_eq!(
            filesystem
                .load_bytes(&path)
                .await
                .expect("unchanged source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_graph_040_2_workspace_culls_scrolled_graph_targets(cx: &mut TestAppContext) {
        let (original, history) =
            numeric_graph_archive_with_layout(NumericGraphFixture::WideRootExport, true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let graph_path = PathBuf::from("/project/wide-graph.gdd.zip");
        let adjacent_path = PathBuf::from("/project/adjacent.txt");
        filesystem
            .write(&graph_path, &original)
            .await
            .expect("source ZIP");
        filesystem
            .write(&adjacent_path, b"Adjacent pane")
            .await
            .expect("adjacent file");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(graph_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("wide saved graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        paint(window);
        let initial = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            let SavedGraphPanel::Ready(graph) = &form.saved_graph else {
                panic!("projected graph")
            };
            assert_eq!(graph.nodes.len(), 2);
            assert!(!graph.wires.is_empty());
            (
                form.selected_node().expect("selected source").storage_id(),
                graph.wires.clone(),
                item.model.read(cx).revision.clone(),
            )
        });
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_some());
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-1").is_some());
        workspace
            .update_in(window, |workspace, window, cx| {
                workspace.split_abs_path(adjacent_path, true, window, cx)
            })
            .await
            .expect("narrow adjacent pane");
        window.simulate_resize(gpui::size(gpui::px(1160.), gpui::px(1400.)));
        window.run_until_parked();
        paint(window);
        window.update(|window, cx| window.simulate_next_frame(cx));
        paint(window);

        let viewport = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH-VIEWPORT")
            .expect("measured graph viewport");
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_some());
        let right_target = window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-1");
        assert!(
            right_target.is_none(),
            "right target {right_target:?} in {viewport:?}"
        );
        let scroll = |window: &mut gpui::VisualTestContext, delta| {
            window.simulate_event(gpui::ScrollWheelEvent {
                position: viewport.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(gpui::px(delta), gpui::px(0.))),
                ..Default::default()
            });
            paint(window);
        };
        scroll(window, -800.);
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_none());
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-1").is_some());
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            let SavedGraphPanel::Ready(graph) = &form.saved_graph else {
                panic!("projected graph")
            };
            assert_eq!(graph.nodes.len(), 2);
            assert_eq!(graph.wires, initial.1);
            assert_eq!(
                form.selected_node().expect("same selection").storage_id(),
                initial.0
            );
            assert!(Arc::ptr_eq(&item.model.read(cx).revision, &initial.2));
            assert!(!item.is_dirty(cx));
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("GDD")
                    .history_len(),
                history
            );
        });
        scroll(window, 800.);
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-0").is_some());
        assert!(window.debug_bounds("GRAPHITE-GDD-GRAPH-NODE-1").is_none());
        assert_eq!(
            filesystem
                .load_bytes(&graph_path)
                .await
                .expect("unchanged source ZIP"),
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
    async fn at_gra_node_280_3_workspace_creates_and_enters_a_saved_custom_node(
        cx: &mut TestAppContext,
    ) {
        let blank = create_gdd_folder(623, 189, "Made".into(), ARCHIVE_BUDGET)
            .await
            .expect("source folder");
        let original = blank
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/create-custom-node.gdd.zip");
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
        click(window, "GRAPHITE-NODE-CATALOG-SEARCH");
        window.simulate_input("Custom Node");
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-POSITION");
        window.simulate_keystrokes("ctrl-a");
        window.simulate_input("6, -9");
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-ADD-Custom Node");
        let target = PathBuf::from("/project/created-custom-node.gdd");
        window.simulate_new_path_selection(|_| Some(target.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![target]));
        window.run_until_parked();
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened custom node copy");
        assert_ne!(reopened, item);
        let contents = reopened.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            item.model
                .read(cx)
                .gdd_contents()
                .expect("saved GDD")
                .clone()
        });
        let inspection = contents
            .graph_inspection()
            .await
            .expect("exact saved graph");
        let [root, nested] = inspection.networks.as_slice() else {
            panic!("root and custom network")
        };
        let [node] = inspection.nodes.as_slice() else {
            panic!("one saved custom node")
        };
        assert_eq!(root.parent, None);
        assert_eq!(node.network_id(), root.id);
        assert_eq!(node.nested_network(), Some(nested.id));
        assert_eq!(node.display_name(), "Custom Node");
        assert_eq!(nested.parent, Some(root.id));
        assert_eq!(nested.import_count, 0);
        assert_eq!(nested.export_count, 0);
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        reopened.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            assert_eq!(
                form.active_network().expect("entered custom network").id,
                nested.id
            );
            assert!(!item.is_dirty(cx));
        });
        for (selector, destination, counts) in [
            (
                "GRAPHITE-GDD-ADD-NETWORK-IMPORT",
                "/project/custom-import.gdd",
                (1, 0),
            ),
            (
                "GRAPHITE-GDD-ADD-NETWORK-EXPORT",
                "/project/custom-export.gdd",
                (1, 1),
            ),
            (
                "GRAPHITE-GDD-ADD-NUMBER-IMPORT",
                "/project/custom-number-import.gdd",
                (2, 1),
            ),
            (
                "GRAPHITE-GDD-ADD-NUMBER-LIST-IMPORT",
                "/project/custom-number-list-import.gdd",
                (3, 1),
            ),
        ] {
            click(window, selector);
            window.simulate_new_path_selection(|_| Some(PathBuf::from(destination)));
            window.run_until_parked();
            window.dispatch_action(OpenGraphiteFolder);
            window.simulate_path_prompt_response(|_| Some(vec![PathBuf::from(destination)]));
            window.run_until_parked();
            let edited = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteDocumentItem>(cx)
                })
                .expect("reopened custom boundary edit");
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
            click(window, "GRAPHITE-GDD-ENTER-NETWORK");
            edited.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready custom graph")
                };
                let network = form.active_network().expect("entered custom network");
                assert_eq!(network.id, nested.id);
                assert_eq!((network.import_count, network.export_count), counts);
                assert!(network.can_extend_ports_in_copy());
                assert_eq!(network.can_start_numeric_import(1), counts.0 >= 2);
                assert_eq!(network.can_start_numeric_import(2), counts.0 == 3);
                assert!(!item.is_dirty(cx));
            });
        }
        let working = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("saved typed boundary");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        let bounds = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH")
            .expect("editable custom canvas");
        click_graph_point(
            window,
            bounds.origin + gpui::point(px(250.), px(177.)),
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        click(window, "GRAPHITE-GDD-ADD-Absolute Value");
        let (internal, before_wire) = working.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready working graph")
            };
            let internal = form
                .nodes
                .iter()
                .find(|node| node.network_id() == nested.id)
                .expect("pinned internal node");
            assert_eq!(internal.numeric_input_exposed(), Some(true));
            (internal.storage_id(), item.model.read(cx).revision.clone())
        });
        click(window, "GRAPHITE-GDD-GRAPH-Import-3");
        click(window, "GRAPHITE-GDD-CANCEL-GRAPH-CONNECTION");
        click(window, "GRAPHITE-GDD-GRAPH-Import-2");
        click(window, "GRAPHITE-GDD-GRAPH-NODE-1");
        working.read_with(window, |item, cx| {
            assert!(Arc::ptr_eq(&item.model.read(cx).revision, &before_wire));
            assert!(matches!(&item.copy_feedback, Some((message, Color::Error))
                if message.as_ref() == "Choose an exposed numeric input that accepts this import."));
        });
        click(window, "GRAPHITE-GDD-GRAPH-Import-3");
        click(window, "GRAPHITE-GDD-GRAPH-NODE-1");
        click(window, "GRAPHITE-GDD-GRAPH-Export-1");
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        working.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("ready working graph") };
            assert!(matches!(&form.output_preview,
                NodeOutputPreview::Ready(GddNumericValue::NumberList(values)) if values.is_empty()));
            assert!(item.is_dirty(cx));
        });
        let wired_destination = PathBuf::from("/project/wired-custom-node.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(wired_destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &wired_destination)
            .await
            .expect("reopened native wired custom graph");
        let graph = saved.graph_inspection().await.expect("saved typed ports");
        let network = graph
            .networks
            .iter()
            .find(|network| network.id == nested.id)
            .expect("same child network");
        let node = graph
            .nodes
            .iter()
            .find(|node| node.storage_id() == internal)
            .expect("same internal node");
        assert_eq!(node.network_id(), nested.id);
        assert_eq!(
            node.input_ports()
                .first()
                .and_then(|port| port.source.as_ref()),
            Some(&graphics_core::formats::gdd::GddGraphPortSource::Import(2)),
        );
        assert_eq!(
            network
                .exports()
                .first()
                .and_then(|port| port.source.as_ref()),
            Some(&graphics_core::formats::gdd::GddGraphPortSource::Node {
                storage_id: internal,
                output_index: 0
            }),
        );
        assert!(matches!(saved.preview_graph_numeric_output(node).await,
            Ok(GddNumericValue::NumberList(values)) if values.is_empty()));
        assert_eq!(
            filesystem
                .load_bytes(&source_path)
                .await
                .expect("original archive"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_prop_002_2_workspace_edits_and_reopens_radians_from_native_input(
        cx: &mut TestAppContext,
    ) {
        let source = create_gdd_folder(624, 220, "Pinned Graphite".into(), ARCHIVE_BUDGET)
            .await
            .expect("source folder")
            .insert_root_catalog_node_in_copy(
                "PROTONODE:math_nodes::CosineInverseNode",
                [4, 6],
                ARCHIVE_BUDGET,
            )
            .await
            .expect("pinned Cosine Inverse");
        let original = source
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/cosine-inverse.gdd.zip");
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
            .expect("open source graph")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let check = |window: &gpui::VisualTestContext, radians, result, dirty| {
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("ready trigonometric node") };
                let node = form.selected_node().expect("selected trigonometric node");
                assert_eq!(node.boolean_input().map(|property| property.value), Some(radians));
                assert!(matches!(form.output_preview, NodeOutputPreview::Ready(GddNumericValue::Number(value)) if value == result));
                assert_eq!(item.is_dirty(cx), dirty);
                node.storage_id()
            })
        };
        click(window, "NATIVE-SWITCH-CONTROL");
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        let node_id = check(window, false, 90., false);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        window.update(|window, _| window.activate_window());
        let handle = window.window_handle();
        assert!(
            window
                .cx
                .simulate_accessibility_activation(handle)
                .is_some()
        );
        window.run_until_parked();
        paint(window);
        let mut switch_focused = false;
        for _ in 0..32 {
            window.update(|window, cx| window.focus_next(cx));
            window.run_until_parked();
            paint(window);
            let tree = window
                .update(|window, _| window.debug_a11y_tree_json())
                .expect("native accessibility tree");
            let tree: serde_json::Value =
                serde_json::from_str(&tree).expect("native accessibility JSON");
            let focused = tree["gpui_focus"]
                .as_str()
                .and_then(|id| tree["nodes"].get(id));
            if focused.is_some_and(|node| {
                node["aria"]["role"] == "Switch" && node["aria"]["label"] == "Radians"
            }) {
                switch_focused = true;
                break;
            }
        }
        assert!(switch_focused, "Radians is keyboard reachable");
        window.update(|window, cx| {
            let keystroke = gpui::Keystroke::parse("space").expect("space keystroke");
            window.dispatch_event(
                gpui::PlatformInput::KeyDown(gpui::KeyDownEvent {
                    keystroke: keystroke.clone(),
                    is_held: false,
                    prefer_character_input: false,
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::KeyUp(gpui::KeyUpEvent { keystroke }),
                cx,
            );
        });
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        let radians = std::f64::consts::FRAC_PI_2;
        assert_eq!(check(window, true, radians, true), node_id);
        click(window, "GRAPHITE-GDD-UNDO");
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        assert_eq!(check(window, false, 90., true), node_id);
        click(window, "GRAPHITE-GDD-REDO");
        click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
        assert_eq!(check(window, true, radians, true), node_id);

        let destination = PathBuf::from("/project/cosine-inverse-radians.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        assert!(!item.read_with(window, |item, cx| item.is_dirty(cx)));
        let reopened = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("saved folder");
        let nodes = reopened.graph_nodes().await.expect("reopened node");
        let [node] = nodes.as_slice() else {
            panic!("one saved node")
        };
        assert_eq!(node.storage_id(), node_id);
        assert_eq!(
            node.boolean_input().map(|property| property.value),
            Some(true)
        );
        assert!(
            node.input_ports()
                .get(1)
                .is_some_and(|port| !port.exposed && port.source.is_none())
        );
        assert!(
            matches!(reopened.preview_graph_numeric_output(node).await, Ok(GddNumericValue::Number(value)) if value == radians)
        );
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_prop_002_2_workspace_edits_modulo_always_positive(cx: &mut TestAppContext) {
        use graphite_document_graph_storage::NoMetadata;
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_proto_node_type;
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork, value::TaggedValue};
        use graphite_resource::ResourceRegistry;
        use graphite_std::math_nodes::modulo;

        let mut template = resolve_proto_node_type(modulo::IDENTIFIER)
            .expect("pinned Modulo definition")
            .default_node_template();
        let [numerator, modulus, _] = template.inputs.as_mut_slice() else {
            panic!("three ordered Modulo inputs");
        };
        *numerator = NodeInput::value(TaggedValue::Number(-11.), true);
        *modulus = NodeInput::value(TaggedValue::Number(5.), false);
        let graph = NodeNetwork {
            nodes: [(NodeId(7), template.into_parts().0)].into_iter().collect(),
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            627,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("actual saved Graphite document");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&graph, &NoMetadata, &ResourceRegistry::new(), &resources)
            .expect("write pinned Modulo");
        let original = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source_path = PathBuf::from("/project/modulo.gdd.zip");
        filesystem
            .write(&source_path, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source_path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open Modulo")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        let assert_preview = |window: &mut gpui::VisualTestContext, flag, expected| {
            click(window, "GRAPHITE-GDD-PREVIEW-NUMERIC");
            item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else { panic!("ready Modulo node") };
                let node = form.selected_node().expect("selected pinned Modulo");
                assert_eq!(node.boolean_input().map(|property| (property.name, property.value)), Some(("Always Positive", flag)));
                assert!(matches!(form.output_preview, NodeOutputPreview::Ready(GddNumericValue::Number(result)) if result == expected));
                node.storage_id()
            })
        };
        let node_id = assert_preview(window, true, 4.);
        click(window, "NATIVE-SWITCH-CONTROL");
        assert_eq!(assert_preview(window, false, -1.), node_id);
        click(window, "GRAPHITE-GDD-UNDO");
        assert_eq!(assert_preview(window, true, 4.), node_id);
        click(window, "GRAPHITE-GDD-REDO");
        assert_eq!(assert_preview(window, false, -1.), node_id);
        let destination = PathBuf::from("/project/modulo-signed.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("saved Modulo folder");
        let nodes = saved.graph_nodes().await.expect("reopened Modulo");
        let [node] = nodes.as_slice() else {
            panic!("one saved Modulo node")
        };
        assert_eq!(node.storage_id(), node_id);
        assert_eq!(
            node.boolean_input().map(|property| property.value),
            Some(false)
        );
        assert!(
            node.input_ports()
                .get(2)
                .is_some_and(|port| !port.exposed && port.source.is_none())
        );
        assert!(matches!(
            saved.preview_graph_numeric_output(node).await,
            Ok(GddNumericValue::Number(-1.))
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
    async fn at_gra_node_281_2_workspace_adds_the_pinned_merge_from_the_catalog(
        cx: &mut TestAppContext,
    ) {
        assert_workspace_catalog_layer(
            cx,
            "Merge",
            "GRAPHITE-NODE-CATALOG-ADD-Merge",
            &[("Base", true), ("Content", true)],
            6,
        )
        .await;
    }

    #[gpui::test]
    async fn at_gra_node_282_2_workspace_adds_the_pinned_artboard_from_the_catalog(
        cx: &mut TestAppContext,
    ) {
        assert_workspace_catalog_layer(
            cx,
            "Artboard",
            "GRAPHITE-NODE-CATALOG-ADD-Artboard",
            &[
                ("Base", true),
                ("Content", true),
                ("Location", false),
                ("Dimensions", false),
                ("Background", false),
                ("Clip", false),
            ],
            5,
        )
        .await;
    }

    async fn assert_workspace_catalog_layer(
        cx: &mut TestAppContext,
        name: &'static str,
        selector: &'static str,
        expected_ports: &[(&str, bool)],
        expected_children: usize,
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
        let source = PathBuf::from("/project/layer-catalog.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("original graph");
        window.run_until_parked();
        workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open native source");
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG");
        click(window, "GRAPHITE-NODE-CATALOG-SEARCH");
        window.simulate_input(name);
        window.run_until_parked();
        paint(window);
        click(window, selector);
        let destination = PathBuf::from("/project/catalog-layer.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        window.dispatch_action(OpenGraphiteFolder);
        window.simulate_path_prompt_response(|_| Some(vec![destination]));
        window.run_until_parked();
        let item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("reopened native layer");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let child = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("reopened layer graph")
            };
            let owner = form.selected_node().expect("root layer owner");
            assert_eq!(
                owner
                    .input_ports()
                    .iter()
                    .map(|port| (port.name.as_deref().unwrap_or_default(), port.exposed))
                    .collect::<Vec<_>>(),
                expected_ports
            );
            assert!(owner.input_ports().iter().all(|port| port.source.is_none()));
            assert!(!item.is_dirty(cx));
            owner.nested_network().expect("exact layer child network")
        });
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("entered layer graph")
            };
            let network = form.active_network().expect("active layer child");
            assert_eq!(network.id, child);
            assert_eq!(
                (network.import_count, network.export_count),
                (expected_ports.len(), 1)
            );
            assert_eq!(
                form.nodes
                    .iter()
                    .filter(|node| node.network_id() == child)
                    .count(),
                expected_children
            );
            assert!(!network.can_extend_ports_in_copy());
        });
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_node_282_2_workspace_sets_the_document_artboard(cx: &mut TestAppContext) {
        let blank = create_gdd_folder(621, 187, "Made".into(), ARCHIVE_BUDGET)
            .await
            .expect("source folder");
        let initial = blank
            .insert_root_catalog_node_in_copy("NETWORK:Artboard", [4, -3], ARCHIVE_BUDGET)
            .await
            .expect("unconnected catalog Artboard");
        let original = initial
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/unconnected-artboard.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("input ZIP");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open ZIP")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-0");
        let connect = "GRAPHITE-GDD-SET-DOCUMENT-ARTBOARD";
        let preview = "GRAPHITE-GDD-PREVIEW-ARTBOARD";
        assert!(window.debug_bounds(preview).is_none());
        let address = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working Artboard")
            };
            form.selected_node().expect("catalog Artboard").storage_id()
        });
        click(window, connect);
        assert!(window.debug_bounds(connect).is_none());
        assert!(window.debug_bounds(preview).is_some());
        click(window, preview);
        assert!(window.debug_bounds("GRAPHITE-GDD-ARTBOARD-IMAGE").is_some());
        let image = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("working Artboard")
            };
            let NodeOutputPreview::ReadyImage(image) = &form.output_preview else {
                panic!("native Artboard image")
            };
            image.clone()
        });
        assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
        click(window, "GRAPHITE-GDD-UNDO");
        assert!(window.debug_bounds(connect).is_some());
        assert!(window.debug_bounds(preview).is_none());
        click(window, "GRAPHITE-GDD-REDO");
        assert!(window.debug_bounds(preview).is_some());
        assert!(item.read_with(window, |item, cx| item.is_dirty(cx)));
        let destination = PathBuf::from("/project/document-artboard.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("published Artboard");
        let graph = saved.graph_inspection().await.expect("reopened graph");
        let board = graph
            .nodes
            .iter()
            .find(|node| node.storage_id() == address)
            .expect("original Artboard address");
        assert!(matches!(
            board.layer_preview(),
            Some(GddGraphLayerPreview::Artboard { .. })
        ));
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("input ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_node_282_2_workspace_attaches_saved_merge_content(cx: &mut TestAppContext) {
        use graphics_core::formats::gdd::GddGraphPortSource;
        use graphite_core_types::{Color, NodeParameter};
        use graphite_document_graph_storage::{NodeMetadataSource, Position};
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::{resolve_network_node_type, resolve_proto_node_type};
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork, value::TaggedValue};
        use graphite_resource::ResourceRegistry;
        use graphite_std::vector_nodes::fill;

        struct SavedLayerLayout;
        impl NodeMetadataSource for SavedLayerLayout {
            fn is_layer(&self, path: &[NodeId], node: NodeId) -> bool {
                path.is_empty() && matches!(node, NodeId(14 | 15))
            }

            fn position(&self, path: &[NodeId], node: NodeId) -> Option<Position> {
                let position = match node {
                    NodeId(11) => [-18, 0],
                    NodeId(12) => [-12, 0],
                    NodeId(14) => [-6, 0],
                    NodeId(15) => [2, 0],
                    _ => return None,
                };
                path.is_empty().then_some(Position::Absolute(position))
            }
        }

        let mut artboard = resolve_network_node_type("Artboard")
            .expect("pinned Artboard")
            .default_node_template();
        *artboard.inputs.get_mut(3).expect("Artboard dimensions") =
            NodeInput::value(TaggedValue::DVec2(DVec2::new(128., 64.)), false);
        let merge = resolve_network_node_type("Merge")
            .expect("pinned Merge")
            .default_node_template();
        let [shape, (paint_id, mut fill_node)] =
            preview_rectangle(NodeId(11), NodeId(12), 24., 24., Color::BLUE);
        let default_fill = resolve_proto_node_type(fill::IDENTIFIER)
            .expect("pinned Fill")
            .default_node_template();
        *fill_node
            .inputs
            .get_mut(fill::ContentInput::INDEX)
            .expect("Fill Content") = default_fill
            .inputs
            .get(fill::ContentInput::INDEX)
            .expect("exact disconnected Content")
            .clone();
        let graph = NodeNetwork {
            nodes: [shape, (paint_id, fill_node)]
                .into_iter()
                .chain([
                    (NodeId(14), merge.into_parts().0),
                    (NodeId(15), artboard.into_parts().0),
                ])
                .collect(),
            exports: vec![NodeInput::node(NodeId(15), 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            655,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("saved GDD");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(
                &graph,
                &SavedLayerLayout,
                &ResourceRegistry::new(),
                &resources,
            )
            .expect("save the Artboard and unfinished painted branch");
        let original = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/detached-artboard-content.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open ZIP")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        let attach = "GRAPHITE-GDD-CONNECT-GRAPHIC-CONTENT";
        let (fill_address, rectangle_address) = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("saved Fill projection");
            };
            (
                form.selected_node().expect("Fill").storage_id(),
                form.nodes.first().expect("Rectangle").storage_id(),
            )
        });
        click(window, attach);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-2")
                .is_none()
        );
        click(window, "GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-0");
        assert!(window.debug_bounds(attach).is_none());
        click(window, "GRAPHITE-GDD-UNDO");
        assert!(window.debug_bounds(attach).is_some());
        click(window, "GRAPHITE-GDD-REDO");
        assert!(window.debug_bounds(attach).is_none());
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
        let merge_address = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("saved Merge projection");
            };
            form.selected_node().expect("Merge").storage_id()
        });
        click(window, attach);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-0")
                .is_none()
        );
        click(window, "GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-1");
        assert!(window.debug_bounds(attach).is_none());
        click(window, "GRAPHITE-GDD-UNDO");
        assert!(window.debug_bounds(attach).is_some());
        click(window, "GRAPHITE-GDD-REDO");
        assert!(window.debug_bounds(attach).is_none());
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        click(window, "GRAPHITE-GDD-NODE-COPY-ITEM-3");
        let preview = "GRAPHITE-GDD-PREVIEW-ARTBOARD";
        let preview_pixel = |window: &mut gpui::VisualTestContext| {
            click(window, preview);
            let color = item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("working Artboard")
                };
                let NodeOutputPreview::ReadyImage(image) = &form.output_preview else {
                    panic!(
                        "native Artboard preview: {:?}",
                        form.output_preview.feedback()
                    )
                };
                let offset = (37 * LAYER_PREVIEW_SIZE as usize + 5) * 4;
                image
                    .as_bytes(0)
                    .expect("BGRA preview")
                    .get(offset..offset + 4)
                    .expect("sample inside preview")
                    .to_vec()
            });
            click(window, preview);
            color
        };
        assert_eq!(preview_pixel(window), [255, 255, 255, 255]);
        click(window, attach);
        assert!(
            window
                .debug_bounds("GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-1")
                .is_none()
        );
        click(window, "GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-2");
        assert!(window.debug_bounds(attach).is_none());
        assert_eq!(preview_pixel(window), [255, 0, 0, 255]);
        click(window, "GRAPHITE-GDD-UNDO");
        assert!(window.debug_bounds(attach).is_some());
        assert_eq!(preview_pixel(window), [255, 255, 255, 255]);
        click(window, "GRAPHITE-GDD-REDO");
        assert!(window.debug_bounds(attach).is_none());
        let painted_pixel = preview_pixel(window);
        let export_button = "GRAPHITE-GDD-EXPORT-ARTBOARD-SVG";
        let svg_path = PathBuf::from("/project/artboard-with-content.svg");
        click(window, export_button);
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        assert!(filesystem.load_bytes(&svg_path).await.is_err());
        click(window, export_button);
        window.simulate_new_path_selection(|_| Some(svg_path.clone()));
        window.run_until_parked();
        paint(window);
        let svg = filesystem
            .load_bytes(&svg_path)
            .await
            .expect("native Artboard SVG");
        let text = std::str::from_utf8(&svg).expect("UTF-8 SVG");
        assert!(text.contains(r#"viewBox="0 0 128 64" width="128" height="64""#));
        let renderer = window.update(|_, cx| cx.svg_renderer());
        let exported =
            crate::render::prepare_svg_bytes_at_target(&svg, 128, 64, 1., &renderer, 128 * 64 * 4)
                .expect("rasterize exported source SVG");
        let offset = (5 * 128 + 5) * 4;
        let pixel = exported
            .as_bytes(0)
            .expect("exported BGRA pixels")
            .get(offset..offset + 4)
            .expect("sample inside the exported bounds");
        assert_eq!(pixel, painted_pixel);
        click(window, export_button);
        window.simulate_new_path_selection(|_| Some(svg_path.clone()));
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            assert!(matches!(item.copy_feedback, Some((_, ui::Color::Error))));
        });
        assert_eq!(
            filesystem
                .load_bytes(&svg_path)
                .await
                .expect("retained SVG"),
            svg
        );
        let destination = PathBuf::from("/project/artboard-with-content.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("published Artboard");
        let nodes = saved.graph_nodes().await.expect("reopened graph");
        let painted = nodes
            .iter()
            .find(|node| node.storage_id() == fill_address)
            .expect("same Fill address after reopening");
        assert!(matches!(
            painted.input_ports().get(fill::ContentInput::INDEX).and_then(|port| port.source.as_ref()),
            Some(GddGraphPortSource::Node { storage_id, output_index: 0 })
                if *storage_id == rectangle_address
        ));
        let merged = nodes
            .iter()
            .find(|node| node.storage_id() == merge_address)
            .expect("same Merge address after reopening");
        assert!(matches!(
            merged.input_ports().get(1).and_then(|port| port.source.as_ref()),
            Some(GddGraphPortSource::Node { storage_id, output_index: 0 })
                if *storage_id == fill_address
        ));
        let board = nodes
            .iter()
            .find(|node| {
                matches!(
                    node.layer_preview(),
                    Some(GddGraphLayerPreview::Artboard { .. })
                )
            })
            .expect("reopened Artboard export");
        assert!(matches!(
            board
                .input_ports()
                .get(1)
                .and_then(|port| port.source.as_ref()),
            Some(GddGraphPortSource::Node {
                output_index: 0,
                ..
            })
        ));
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("input ZIP"),
            original
        );
    }

    fn preview_rectangle(
        rectangle_id: graphite_graph_craft::document::NodeId,
        paint_id: graphite_graph_craft::document::NodeId,
        width: f64,
        height: f64,
        color: graphite_core_types::Color,
    ) -> [(
        graphite_graph_craft::document::NodeId,
        graphite_graph_craft::document::DocumentNode,
    ); 2] {
        use graphite_core_types::NodeParameter;
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_proto_node_type;
        use graphite_graph_craft::document::{NodeInput, value::TaggedValue};
        use graphite_std::vector_nodes::{fill, generator_nodes::rectangle};

        let mut shape = resolve_proto_node_type(rectangle::IDENTIFIER)
            .expect("pinned Rectangle")
            .default_node_template();
        *shape
            .inputs
            .get_mut(rectangle::WidthInput::INDEX)
            .expect("width") = NodeInput::value(TaggedValue::Number(width), false);
        *shape
            .inputs
            .get_mut(rectangle::HeightInput::INDEX)
            .expect("height") = NodeInput::value(TaggedValue::Number(height), false);
        let mut paint = resolve_proto_node_type(fill::IDENTIFIER)
            .expect("pinned Fill")
            .default_node_template();
        *paint
            .inputs
            .get_mut(fill::ContentInput::INDEX)
            .expect("paint input") = NodeInput::node(rectangle_id, 0);
        *paint
            .inputs
            .get_mut(fill::PaintInput::INDEX)
            .expect("paint color") = NodeInput::value(TaggedValue::Color(color), false);
        [
            (rectangle_id, shape.into_parts().0),
            (paint_id, paint.into_parts().0),
        ]
    }

    #[gpui::test]
    async fn at_gra_node_281_3_workspace_previews_the_saved_merge_layers(cx: &mut TestAppContext) {
        use graphite_core_types::Color;
        use graphite_document_graph_storage::NoMetadata;
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_network_node_type;
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork};
        use graphite_resource::ResourceRegistry;

        let layer = |rectangle_id, fill_id, layer_id, base, width, height, color| {
            let [shape, paint] = preview_rectangle(rectangle_id, fill_id, width, height, color);
            let mut merge = resolve_network_node_type("Merge")
                .expect("pinned Merge")
                .default_node_template();
            let [base_port, content] = merge.inputs.as_mut_slice() else {
                panic!("two Merge inputs")
            };
            if let Some(base) = base {
                *base_port = NodeInput::node(base, 0);
            }
            *content = NodeInput::node(fill_id, 0);
            [shape, paint, (layer_id, merge.into_parts().0)]
        };
        let bottom = NodeId(14);
        let top = NodeId(25);
        let graph = NodeNetwork {
            nodes: layer(NodeId(11), NodeId(12), bottom, None, 60., 18., Color::RED)
                .into_iter()
                .chain(layer(
                    NodeId(21),
                    NodeId(22),
                    top,
                    Some(bottom),
                    14.,
                    44.,
                    Color::BLUE,
                ))
                .collect(),
            exports: vec![NodeInput::node(top, 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            655,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("saved GDD");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&graph, &NoMetadata, &ResourceRegistry::new(), &resources)
            .expect("save exact layered source");
        let history = owner.session().history().count();
        let original = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/pinned-merge-preview.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("original ZIP");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open saved Merge")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-5");
        let revision = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        let preview = "GRAPHITE-GDD-PREVIEW-MERGE";
        let click_before_completion = |window: &mut gpui::VisualTestContext| {
            paint(window);
            let position = window
                .debug_bounds(preview)
                .expect("visible Merge preview control")
                .center();
            dispatch_graph_click(
                window,
                position,
                gpui::MouseButton::Left,
                gpui::Modifiers::default(),
            );
        };
        click_before_completion(window);
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready Merge form")
            };
            assert!(matches!(
                form.output_preview,
                NodeOutputPreview::Pending { .. }
            ));
        });
        click_before_completion(window);
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-GDD-MERGE-IMAGE").is_none());
        click(window, preview);
        let image = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready Merge form")
            };
            let NodeOutputPreview::ReadyImage(image) = &form.output_preview else {
                panic!(
                    "expected Merge pixels, got {:?}",
                    form.output_preview.feedback()
                );
            };
            assert!(Arc::ptr_eq(&revision, &item.model.read(cx).revision));
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("GDD")
                    .history_len(),
                history
            );
            assert!(!item.is_dirty(cx));
            image.clone()
        });
        let bytes = image.as_bytes(0).expect("native BGRA pixels");
        let pixel = |x, y| {
            let offset = (y * LAYER_PREVIEW_SIZE as usize + x) * 4;
            bytes
                .get(offset..offset + 4)
                .expect("sample inside preview")
        };
        assert_eq!(pixel(40, 64), [0, 0, 255, 255]);
        assert_eq!(pixel(64, 64), [255, 0, 0, 255]);
        assert_eq!(pixel(8, 8), [0, 0, 0, 0]);
        assert!(window.debug_bounds("GRAPHITE-GDD-MERGE-IMAGE").is_some());
        assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
        click(window, preview);
        assert!(window.debug_bounds("GRAPHITE-GDD-MERGE-IMAGE").is_none());
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_node_282_3_workspace_fits_and_clips_the_saved_artboard(
        cx: &mut TestAppContext,
    ) {
        use graphite_core_types::Color;
        use graphite_document_graph_storage::NoMetadata;
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_network_node_type;
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork, value::TaggedValue};
        use graphite_resource::ResourceRegistry;

        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        for clip in [true, false] {
            let board_id = NodeId(15);
            let rectangle_id = NodeId(11);
            let paint_id = NodeId(12);
            let mut board = resolve_network_node_type("Artboard")
                .expect("pinned Artboard")
                .default_node_template();
            let [_, content, location, dimensions, background, clipping] =
                board.inputs.as_mut_slice()
            else {
                panic!("six pinned Artboard inputs")
            };
            *content = NodeInput::node(paint_id, 0);
            *location = NodeInput::value(TaggedValue::DVec2(DVec2::new(80., -16.)), false);
            *dimensions = NodeInput::value(TaggedValue::DVec2(DVec2::new(-64., 32.)), false);
            *background = NodeInput::value(TaggedValue::Color(Color::WHITE), false);
            *clipping = NodeInput::value(TaggedValue::Bool(clip), false);
            let graph = NodeNetwork {
                nodes: preview_rectangle(rectangle_id, paint_id, 24., 24., Color::BLUE)
                    .into_iter()
                    .chain([(board_id, board.into_parts().0)])
                    .collect(),
                exports: vec![NodeInput::node(board_id, 0)],
                ..Default::default()
            };
            let mut owner = GddV1::create_in(
                AnyContainer::Memory(MemoryBackend::new()),
                GddV1Layout,
                Default::default(),
                655,
                "Pinned Graphite".into(),
                "Pinned library".into(),
            )
            .expect("saved GDD");
            let resources = HashMapResourceStorage::new();
            owner
                .commit_from_runtime(&graph, &NoMetadata, &ResourceRegistry::new(), &resources)
                .expect("save pinned Artboard and painted content");
            let original = owner
                .export_to_bytes(
                    ExportFormat::Zip,
                    ExportOptions::default(),
                    &resources,
                    None,
                )
                .await
                .expect("source ZIP");
            let path = PathBuf::from(format!("/project/pinned-artboard-clip-{clip}.gdd.zip"));
            filesystem
                .write(&path, &original)
                .await
                .expect("original ZIP");
            window.run_until_parked();
            let item = workspace
                .update_in(window, |workspace, window, cx| {
                    workspace.open_abs_path(path.clone(), Default::default(), window, cx)
                })
                .await
                .expect("open saved Artboard")
                .downcast::<GraphiteDocumentItem>()
                .expect("native Inspector");
            choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-2");
            click(window, "GRAPHITE-GDD-PREVIEW-ARTBOARD");
            let image = item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Artboard form")
                };
                let NodeOutputPreview::ReadyImage(image) = &form.output_preview else {
                    panic!(
                        "expected Artboard pixels, got {:?}",
                        form.output_preview.feedback()
                    );
                };
                assert!(!item.is_dirty(cx));
                image.clone()
            });
            let bytes = image.as_bytes(0).expect("native BGRA pixels");
            let pixel = |x, y| {
                let offset = (y * LAYER_PREVIEW_SIZE as usize + x) * 4;
                bytes
                    .get(offset..offset + 4)
                    .expect("sample inside preview")
            };
            assert_eq!(pixel(64, 30), [0, 0, 0, 0]);
            assert_eq!(pixel(64, 34), [255, 255, 255, 255]);
            assert_eq!(pixel(64, 94), [255, 255, 255, 255]);
            assert_eq!(pixel(64, 98), [0, 0, 0, 0]);
            assert_eq!(pixel(10, 40), [255, 0, 0, 255]);
            assert_eq!(
                pixel(10, 20),
                if clip { [0, 0, 0, 0] } else { [255, 0, 0, 255] }
            );
            assert!(window.debug_bounds("GRAPHITE-GDD-ARTBOARD-IMAGE").is_some());
            assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
            if clip {
                click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
                let property = "GRAPHITE-GDD-BOOLEAN-INPUT";
                let preview = "GRAPHITE-GDD-PREVIEW-ARTBOARD";
                let read_clip = |window: &gpui::VisualTestContext| {
                    item.read_with(window, |item, _| {
                        let NodeCopyPanel::Ready(form) = &item.node_copy else {
                            panic!("working Artboard");
                        };
                        let node = form.selected_node().expect("selected Artboard");
                        let clip = node.boolean_input().expect("original Clip property");
                        assert_eq!(clip.name, "Clip");
                        assert!(
                            node.input_ports()
                                .get(5)
                                .is_some_and(|port| { !port.exposed && port.source.is_none() })
                        );
                        (node.storage_id(), clip.value)
                    })
                };
                let (address, initial) = read_clip(window);
                assert!(initial);
                click(window, property);
                assert_eq!(read_clip(window), (address, false));
                assert!(window.debug_bounds("GRAPHITE-GDD-ARTBOARD-IMAGE").is_none());
                click(window, preview);
                item.read_with(window, |item, _| {
                    let NodeCopyPanel::Ready(form) = &item.node_copy else {
                        panic!("working Artboard");
                    };
                    let NodeOutputPreview::ReadyImage(image) = &form.output_preview else {
                        panic!("updated Artboard preview");
                    };
                    let offset = (20 * LAYER_PREVIEW_SIZE as usize + 10) * 4;
                    let bytes = image.as_bytes(0).expect("native BGRA pixels");
                    assert_eq!(
                        bytes.get(offset..offset + 4),
                        Some([255, 0, 0, 255].as_slice())
                    );
                });
                click(window, "GRAPHITE-GDD-UNDO");
                assert_eq!(read_clip(window), (address, true));
                click(window, "GRAPHITE-GDD-REDO");
                assert_eq!(read_clip(window), (address, false));
                let destination = PathBuf::from("/project/artboard-edited-clip.gdd");
                click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
                window.simulate_new_path_selection(|_| Some(destination.clone()));
                window.run_until_parked();
                let saved = read_graphite_folder(filesystem.as_ref(), &destination)
                    .await
                    .expect("saved Clip edit");
                let nodes = saved.graph_nodes().await.expect("saved Artboard graph");
                let board = nodes
                    .iter()
                    .find(|node| node.storage_id() == address)
                    .expect("same saved Artboard");
                assert_eq!(
                    board.boolean_input().map(|property| property.value),
                    Some(false)
                );
            }
            assert_eq!(
                filesystem.load_bytes(&path).await.expect("original ZIP"),
                original
            );
        }
    }

    #[gpui::test]
    async fn at_gra_tool_008_2_workspace_draws_an_artboard_around_saved_layers(
        cx: &mut TestAppContext,
    ) {
        use graphite_core_types::Color;
        use graphite_document_graph_storage::{NodeMetadataSource, Position};
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::{
            resolve_network_node_type, resolve_proto_node_type,
        };
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork, value::TaggedValue};
        use graphite_resource::ResourceRegistry;
        use graphite_std::transform_nodes::transform;

        struct LayerMetadata;
        impl NodeMetadataSource for LayerMetadata {
            fn is_layer(&self, path: &[NodeId], id: NodeId) -> bool {
                path.is_empty() && matches!(id, NodeId(1 | 2))
            }
            fn reference(&self, path: &[NodeId]) -> Option<&str> {
                matches!(path, [NodeId(1 | 2)]).then_some("Merge")
            }
            fn position(&self, path: &[NodeId], id: NodeId) -> Option<Position> {
                match (path, id) {
                    ([], NodeId(1)) => Some(Position::Absolute([4, -1])),
                    ([], NodeId(2)) => Some(Position::Stack(1)),
                    _ => None,
                }
            }
        }

        let merge = resolve_network_node_type("Merge").expect("pinned Merge");
        let front = merge
            .node_template_input_override([
                Some(NodeInput::node(NodeId(2), 0)),
                Some(NodeInput::node(NodeId(7), 0)),
            ])
            .into_parts()
            .0;
        let back = merge
            .node_template_input_override([None, Some(NodeInput::node(NodeId(6), 0))])
            .into_parts()
            .0;
        let offset = DVec2::new(35., 17.);
        let transform = resolve_proto_node_type(transform::IDENTIFIER)
            .expect("pinned Transform")
            .node_template_input_override([
                Some(NodeInput::node(NodeId(4), 0)),
                Some(NodeInput::value(TaggedValue::DVec2(offset), false)),
            ])
            .into_parts()
            .0;
        let graph = NodeNetwork {
            nodes: [
                (NodeId(1), front),
                (NodeId(2), back),
                (NodeId(7), transform),
            ]
            .into_iter()
            .chain(preview_rectangle(
                NodeId(3),
                NodeId(4),
                34.,
                26.,
                Color::RED,
            ))
            .chain(preview_rectangle(
                NodeId(5),
                NodeId(6),
                16.,
                16.,
                Color::BLUE,
            ))
            .collect(),
            exports: vec![NodeInput::node(NodeId(1), 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            663,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("saved document");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&graph, &LayerMetadata, &ResourceRegistry::new(), &resources)
            .expect("saved unframed layers");
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
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1050.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/unframed-layers.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("source ZIP");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open saved layers")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        let canvas_button = "GRAPHITE-GDD-OPEN-CANVAS";
        let draw_button = "GRAPHITE-GDD-CANVAS-DRAW-ARTBOARD";
        let assert_artwork = |canvas: &SavedSceneCanvas| {
            let image = match &canvas.frame {
                SavedSceneFrame::Ready(image) => image,
                SavedSceneFrame::Failed(error) => panic!("Canvas failed: {error}"),
                _ => panic!("expected the rendered document"),
            };
            let target = canvas.target.expect("measured target");
            let viewport = canvas
                .navigation
                .expect("accepted camera")
                .document_to_pane(target.logical_size());
            let sample = |point| {
                let position = (viewport.transform_point2(point) * target.scale).round();
                let index = (position.y as usize * image.size(0).width.0 as usize
                    + position.x as usize)
                    * 4;
                image
                    .as_bytes(0)
                    .and_then(|bytes| bytes.get(index..index + 4))
                    .expect("sample inside Canvas")
            };
            assert_eq!(sample(offset), [0, 0, 255, 255]);
            assert_eq!(sample(DVec2::ZERO), [255, 0, 0, 255]);
            viewport
        };
        click(window, canvas_button);
        assert!(window.debug_bounds(draw_button).is_none());
        item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("read-only Canvas");
            assert!(canvas.artboards.is_empty());
            assert_artwork(canvas);
            assert!(!item.is_dirty(cx));
        });
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, canvas_button);
        click(window, draw_button);
        let pane = window
            .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
            .expect("Canvas pane");
        let (viewport, front_address, back_address) = item.read_with(window, |item, cx| {
            let graph = &item
                .model
                .read(cx)
                .working
                .as_ref()
                .expect("working document")
                .inspection;
            let root = graph
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("document root");
            let (front, _) = root.connected_export(&graph.nodes).expect("top Merge");
            let Some(GddGraphPortSource::Node {
                storage_id: back,
                output_index: 0,
            }) = front
                .input_ports()
                .first()
                .and_then(|port| port.source.as_ref())
            else {
                panic!("back Merge connected to Base");
            };
            (
                assert_artwork(item.scene_canvas.as_ref().expect("working Canvas")),
                front.storage_id(),
                *back,
            )
        });
        let at = |point| {
            let point = viewport.transform_point2(point);
            pane.origin + gpui::point(px(point.x as f32), px(point.y as f32))
        };
        let geometry = GddArtboardGeometry {
            location: DVec2::new(-9., -9.),
            dimensions: DVec2::new(62., 40.),
        };
        let start = at(geometry.location);
        let end = at(geometry.location + geometry.dimensions);
        assert!(pane.contains(&start));
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position: start,
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                    position: end,
                    pressed_button: Some(gpui::MouseButton::Left),
                    modifiers: Default::default(),
                }),
                cx,
            );
        });
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let draft = item
                .scene_canvas
                .as_ref()
                .and_then(|canvas| canvas.artboard_draw.as_ref())
                .expect("draw gesture");
            assert_eq!(draft.placement, Some(geometry));
            assert!(!item.is_dirty(cx));
        });
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position: end,
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                }),
                cx,
            );
        });
        let read_root = |window: &mut gpui::VisualTestContext| {
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, cx| {
                assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
                assert_eq!(
                    assert_artwork(item.scene_canvas.as_ref().expect("Canvas")),
                    viewport
                );
                let graph = &item
                    .model
                    .read(cx)
                    .working
                    .as_ref()
                    .expect("working document")
                    .inspection;
                let root = graph
                    .networks
                    .iter()
                    .find(|network| network.parent.is_none())
                    .expect("document root");
                let (root, _) = root
                    .connected_export(&graph.nodes)
                    .expect("saved document output");
                (root.storage_id(), root.artboard_geometry())
            })
        };
        let (board_address, placement) = read_root(window);
        assert_ne!(board_address, front_address);
        assert_eq!(placement, Some(geometry));
        window.dispatch_action(editor::actions::Undo);
        assert_eq!(read_root(window), (front_address, None));
        window.dispatch_action(editor::actions::Redo);
        assert_eq!(read_root(window), (board_address, Some(geometry)));
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        let destination = PathBuf::from("/project/framed-layers.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("saved framed scene");
        let graph = saved.graph_inspection().await.expect("reopened graph");
        let root = graph
            .networks
            .iter()
            .find(|network| network.parent.is_none())
            .expect("document root");
        let layers = saved
            .graph_geometry(root)
            .await
            .expect("saved geometry")
            .root_layers
            .expect("root layers");
        let [board, front, back] = layers.as_slice() else {
            panic!("Artboard and two children")
        };
        assert_eq!(board.storage_id, board_address);
        assert_eq!(
            (front.storage_id, front.parent, back.storage_id, back.parent),
            (
                front_address,
                Some(board_address),
                back_address,
                Some(board_address)
            )
        );
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("original ZIP")
                .as_slice(),
            original.as_ref()
        );
    }

    struct SavedVectorFixture {
        original: Arc<[u8]>,
        origin: DVec2,
        curve_to_artboard: DAffine2,
        rectangle_to_artboard: DAffine2,
    }

    async fn saved_vector_fixture(
        visibility: Option<OverlaysVisibilitySettings>,
    ) -> SavedVectorFixture {
        use graphite_core_types::Color;
        use graphite_document_graph_storage::NodeMetadataSource;
        use graphite_editor::messages::portfolio::document::{
            graph_operation::transform_utils::get_current_transform,
            node_graph::document_node_definitions::{
                resolve_network_node_type, resolve_proto_node_type,
            },
        };
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork, value::TaggedValue};
        use graphite_resource::ResourceRegistry;
        use graphite_std::{
            Vector, transform_nodes::transform, vector::VectorModification, vector_nodes::stroke,
            vector_types::kurbo::BezPath,
        };

        struct VectorLayers;
        impl NodeMetadataSource for VectorLayers {
            fn is_layer(&self, path: &[NodeId], id: NodeId) -> bool {
                path.is_empty() && matches!(id, NodeId(1..=4))
            }
            fn reference(&self, path: &[NodeId]) -> Option<&str> {
                match path {
                    [NodeId(1)] => Some("Artboard"),
                    [NodeId(2..=4)] => Some("Merge"),
                    [NodeId(10)] => Some("Path"),
                    _ => None,
                }
            }
        }

        let mut curve = BezPath::new();
        curve.move_to((0., 30.));
        curve.curve_to((14., 0.), (56., 60.), (70., 30.));
        let curve = VectorModification::create_from_vector(&Vector::from_bezpath(curve));
        let path = resolve_network_node_type("Path")
            .expect("pinned Path")
            .node_template_input_override([
                None,
                Some(NodeInput::value(
                    TaggedValue::VectorModification(Box::new(curve)),
                    false,
                )),
            ])
            .into_parts()
            .0;
        let stroke = resolve_proto_node_type(stroke::IDENTIFIER)
            .expect("pinned Stroke")
            .node_template_input_override([
                Some(NodeInput::node(NodeId(10), 0)),
                Some(NodeInput::value(TaggedValue::Color(Color::BLUE), false)),
                Some(NodeInput::value(TaggedValue::Number(6.), false)),
            ])
            .into_parts()
            .0;
        let transform = |source, offset, rotation, scale, skew| {
            resolve_proto_node_type(transform::IDENTIFIER)
                .expect("pinned Transform")
                .node_template_input_override([
                    Some(NodeInput::node(source, 0)),
                    Some(NodeInput::value(TaggedValue::DVec2(offset), false)),
                    Some(NodeInput::value(TaggedValue::Number(rotation), false)),
                    Some(NodeInput::value(TaggedValue::DVec2(scale), false)),
                    Some(NodeInput::value(TaggedValue::DVec2(skew), false)),
                ])
                .into_parts()
                .0
        };
        let curve_transform = transform(
            NodeId(11),
            DVec2::new(28., 36.),
            -15.,
            DVec2::new(1., 0.75),
            DVec2::new(8., 0.),
        );
        let rectangle_transform = transform(
            NodeId(13),
            DVec2::new(6., 8.),
            10.,
            DVec2::new(0.8, 1.2),
            DVec2::ZERO,
        );
        let group_transform = transform(
            NodeId(3),
            DVec2::new(52., 48.),
            20.,
            DVec2::new(1.15, 0.9),
            DVec2::new(-6., 5.),
        );
        let group_to_artboard = get_current_transform(&group_transform.inputs);
        let curve_to_artboard = group_to_artboard * get_current_transform(&curve_transform.inputs);
        let rectangle_to_artboard =
            group_to_artboard * get_current_transform(&rectangle_transform.inputs);
        let merge = resolve_network_node_type("Merge").expect("pinned Merge");
        let group = |base: Option<NodeId>, content| {
            merge
                .node_template_input_override([
                    base.map(|id| NodeInput::node(id, 0)),
                    Some(NodeInput::node(content, 0)),
                ])
                .into_parts()
                .0
        };
        let origin = DVec2::new(40., -20.);
        let board = resolve_network_node_type("Artboard")
            .expect("pinned Artboard")
            .node_template_input_override([
                None,
                Some(NodeInput::node(NodeId(2), 0)),
                Some(NodeInput::value(TaggedValue::DVec2(origin), false)),
                Some(NodeInput::value(
                    TaggedValue::DVec2(DVec2::new(220., 160.)),
                    false,
                )),
            ])
            .into_parts()
            .0;
        let graph = NodeNetwork {
            nodes: [
                (NodeId(1), board),
                (NodeId(2), group(None, NodeId(16))),
                (NodeId(3), group(Some(NodeId(4)), NodeId(14))),
                (NodeId(4), group(None, NodeId(15))),
                (NodeId(10), path),
                (NodeId(11), stroke),
                (NodeId(14), curve_transform),
                (NodeId(15), rectangle_transform),
                (NodeId(16), group_transform),
            ]
            .into_iter()
            .chain(preview_rectangle(
                NodeId(12),
                NodeId(13),
                28.,
                18.,
                Color::RED,
            ))
            .collect(),
            exports: vec![NodeInput::node(NodeId(1), 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            662,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("saved GDD");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(&graph, &VectorLayers, &ResourceRegistry::new(), &resources)
            .expect("F-VECTOR source document");
        if let Some(visibility) = visibility {
            use graphite_document_graph_storage::{attr::session::doc::OVERLAYS, to_value};
            owner
                .set_view_settings(BTreeMap::from([(
                    OVERLAYS.into(),
                    to_value(&visibility).expect("pinned overlay settings"),
                )]))
                .expect("persist source overlay settings");
        }
        let original = owner
            .export_to_bytes(
                ExportFormat::Zip,
                ExportOptions::default(),
                &resources,
                None,
            )
            .await
            .expect("source ZIP")
            .into();
        SavedVectorFixture {
            original,
            origin,
            curve_to_artboard,
            rectangle_to_artboard,
        }
    }

    #[gpui::test]
    async fn at_gra_view_009_2_workspace_changes_f_vector_overlays_without_losing_selection(
        cx: &mut TestAppContext,
    ) {
        let SavedVectorFixture {
            original, origin, ..
        } = saved_vector_fixture(Some(OverlaysVisibilitySettings {
            all: false,
            ..Default::default()
        }))
        .await;
        let (workspace, window) = native_workspace(cx).await;
        window.update(|_, cx| {
            let bindings =
                settings::KeymapFile::load_asset(crate::DEFAULT_GRAPHITE_KEYMAP_PATH, None, cx)
                    .expect("Graphite keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(px(1160.), px(1050.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/f-vector-overlays.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source ZIP");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        let open = "GRAPHITE-GDD-OPEN-CANVAS";
        let pick = "GRAPHITE-GDD-CANVAS-PICK";
        let artboard_center = |window: &mut gpui::VisualTestContext| {
            let pane = window
                .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
                .expect("Canvas pane");
            let center = item.read_with(window, |item, _| {
                let canvas = item.scene_canvas.as_ref().expect("ready canvas");
                let target = canvas.target.expect("measured viewport");
                canvas
                    .navigation
                    .expect("accepted camera")
                    .document_to_pane(target.logical_size())
                    .transform_point2(origin + DVec2::new(110., 80.))
            });
            pane.origin + gpui::point(px(center.x as f32), px(center.y as f32))
        };
        click(window, open);
        let first_instance = item.read_with(window, |item, _| {
            item.scene_canvas.as_ref().expect("Canvas").instance.clone()
        });
        click(window, pick);
        let center = artboard_center(window);
        window.simulate_click(center, Default::default());
        window.run_until_parked();
        paint(window);
        let selected_address = item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("source inspector selection");
            };
            form.selected_node()
                .expect("picked source Artboard")
                .storage_id()
        });
        click(window, open);
        click(window, pick);
        click(window, "GRAPHITE-GDD-CANVAS-GRID");
        item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas after selection");
            assert!(
                !canvas
                    .overlay_visibility(item.model.read(cx))
                    .expect("loaded source overlay settings")
                    .all()
            );
            assert!(canvas.grid_frame.is_none());
        });
        window.simulate_keystrokes("alt-o");
        window.run_until_parked();
        paint(window);
        let (base, revision, artwork, grid) = item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            let canvas = item.scene_canvas.as_ref().expect("Canvas after selection");
            let SavedSceneFrame::Ready(artwork) = &canvas.frame else {
                panic!("accepted F-VECTOR");
            };
            let frame = canvas.grid_frame.as_ref().expect("visible grid");
            let grid = frame
                .image
                .as_ref()
                .expect("grid did render")
                .as_ref()
                .expect("grid image");
            (
                model.gdd_contents().expect("saved F-VECTOR").clone(),
                model.revision.clone(),
                artwork.clone(),
                grid.clone(),
            )
        });
        let check_view = |window: &gpui::VisualTestContext, master, name, cage| {
            item.read_with(window, |item, cx| {
                let model = item.model.read(cx);
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let settings = canvas.overlay_visibility(model).expect("accepted overlays");
                assert_eq!(settings.all(), master);
                assert_eq!(settings.artboard_name(), name);
                assert_eq!(settings.transform_cage(), cage);
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("selection retained");
                };
                assert_eq!(form.selected_node().map(GddGraphNode::storage_id), Some(selected_address));
                assert!(canvas.pick_armed);
                assert!(matches!(&canvas.frame, SavedSceneFrame::Ready(frame) if Arc::ptr_eq(frame, &artwork)));
                assert!(Arc::ptr_eq(&model.revision, &revision));
                assert!(Arc::ptr_eq(model.gdd_contents().expect("F-VECTOR snapshot"), &base));
                assert!(!item.is_dirty(cx));
                assert!(!item.can_save(cx));
            });
        };
        let toggle_overlay = |window: &mut gpui::VisualTestContext, selector| {
            click(window, "GRAPHITE-GDD-CANVAS-OVERLAYS");
            click(window, selector);
        };
        check_view(window, true, true, true);
        toggle_overlay(window, "MENU_ITEM-Artboard Name");
        check_view(window, true, false, true);
        toggle_overlay(window, "MENU_ITEM-Transform Cage");
        check_view(window, true, false, false);
        toggle_overlay(window, "MENU_ITEM-Artboard Name");
        check_view(window, true, true, false);
        assert!(window.update(|window, _| window.has_image_atlas_entry(&grid)));
        toggle_overlay(window, "MENU_ITEM-All Overlays");
        window.run_until_parked();
        paint(window);
        check_view(window, false, false, false);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&grid)));
        window.simulate_keystrokes("alt-o");
        window.run_until_parked();
        paint(window);
        check_view(window, true, true, false);
        item.update_in(window, |item, _, cx| {
            assert!(!item.toggle_scene_overlay(
                &first_instance,
                &revision,
                CANVAS_OVERLAYS_ALL,
                cx
            ));
        });
        check_view(window, true, true, false);
        toggle_overlay(window, "MENU_ITEM-Artboard Name");
        let center = artboard_center(window);
        window.simulate_click(center, Default::default());
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("Artboard can still be picked with hidden overlays");
            };
            assert!(item.scene_canvas.is_none());
            assert_eq!(
                form.selected_node().map(GddGraphNode::storage_id),
                Some(selected_address)
            );
            assert!(!item.is_dirty(cx));
        });
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("source ZIP"),
            original.as_ref()
        );
    }

    #[gpui::test]
    async fn at_gra_view_007_2_workspace_applies_and_saves_f_vector_grid_options(
        cx: &mut TestAppContext,
    ) {
        use graphite_editor::messages::portfolio::document::utility_types::misc::GridType;

        let SavedVectorFixture { original, .. } = saved_vector_fixture(None).await;
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1050.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/f-vector-grid.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source ZIP");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        let open_options = |window: &mut gpui::VisualTestContext| {
            click(window, "GRAPHITE-GDD-CANVAS-SNAP-OPTIONS");
            click(window, "MENU_ITEM-Grid Options");
        };
        open_options(window);
        click(window, "GRAPHITE-GDD-GRID-ISOMETRIC");
        click(window, "GRAPHITE-GDD-GRID-CANCEL");
        item.read_with(window, |item, cx| {
            assert!(item.model.read(cx).snapping_preferences.is_none());
            assert!(!item.can_save(cx));
            assert!(!item.is_dirty(cx));
        });
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        let (base, revision, artwork) = item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let SavedSceneFrame::Ready(artwork) = &canvas.frame else {
                panic!("accepted artwork frame");
            };
            (
                model.gdd_contents().expect("F-VECTOR snapshot").clone(),
                model.revision.clone(),
                artwork.clone(),
            )
        });
        open_options(window);
        let enter = |window: &mut gpui::VisualTestContext, label, text| {
            click(window, label);
            window.dispatch_action(editor::actions::SelectAll);
            window.simulate_input(text);
            paint(window);
        };
        enter(window, "Spacing X", "16");
        enter(window, "Spacing Y", "20");
        click(window, "GRAPHITE-GDD-GRID-ISOMETRIC");
        enter(window, "Display color", "wrong");
        click(window, "GRAPHITE-GDD-GRID-APPLY");
        assert!(window.debug_bounds("GRAPHITE-GDD-GRID-ERROR").is_some());
        item.read_with(window, |item, cx| {
            assert!(item.model.read(cx).snapping_preferences.is_none());
            assert!(!item.is_dirty(cx));
        });
        enter(window, "Display color", "aa444480");
        enter(window, "Origin X", "12");
        enter(window, "Origin Y", "-8");
        enter(window, "Isometric spacing Y", "23");
        enter(window, "Angle A (°)", "35");
        enter(window, "Angle B (°)", "40");
        click(window, "GRAPHITE-GDD-GRID-DOTTED");
        click(window, "GRAPHITE-GDD-GRID-APPLY");
        assert!(window.debug_bounds("GRAPHITE-GDD-GRID-APPLY").is_none());
        let expected = item.read_with(window, |item, cx| {
            let grid = &item
                .model
                .read(cx)
                .snapping_preferences
                .as_ref()
                .expect("applied Grid Options")
                .grid;
            assert_eq!(grid.color, "#aa444480");
            assert_eq!(grid.origin, DVec2::new(12., -8.));
            assert_eq!(grid.rectangular_spacing, DVec2::new(16., 20.));
            assert_eq!(
                grid.grid_type,
                GridType::Isometric {
                    y_axis_spacing: 23.,
                    angle_a: 35.,
                    angle_b: 40.,
                }
            );
            assert!(grid.dot_display);
            grid.clone()
        });
        open_options(window);
        click(window, "GRAPHITE-GDD-GRID-RECTANGULAR");
        item.update_in(window, |item, _, _| {
            item.scene_canvas
                .as_mut()
                .expect("Canvas")
                .grid_options
                .as_mut()
                .expect("open form")
                .revision = Arc::new(());
        });
        click(window, "GRAPHITE-GDD-GRID-APPLY");
        assert!(window.debug_bounds("GRAPHITE-GDD-GRID-ERROR").is_some());
        click(window, "GRAPHITE-GDD-GRID-CANCEL");
        click(window, "GRAPHITE-GDD-CANVAS-GRID");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let grid = canvas.grid_frame.as_ref().expect("visible grid");
            assert_eq!(grid.request.options, expected);
            assert!(matches!(grid.image, Ok(Some(_))));
            let snapping = &canvas.snapping_document.as_ref().expect("accepted scene").snapping_state;
            assert!(!snapping.grid_snapping);
            assert!(Arc::ptr_eq(&model.revision, &revision));
            assert!(Arc::ptr_eq(model.gdd_contents().expect("F-VECTOR snapshot"), &base));
            assert!(matches!(&canvas.frame, SavedSceneFrame::Ready(image) if Arc::ptr_eq(image, &artwork)));
            assert!(!item.is_dirty(cx));
        });
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| None);
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        let destination = PathBuf::from("/project/f-vector-grid-working.gdd");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen saved F-VECTOR");
        assert!(saved.matches_saved_history(&base));
        let restored = saved.canvas_snapping_defaults().snapping_state;
        assert_eq!(restored.grid, expected);
        assert!(!restored.grid_snapping);
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("source ZIP"),
            original.as_ref()
        );
    }

    #[gpui::test]
    async fn at_gra_tool_008_2_workspace_moves_and_resizes_the_saved_f_vector_scene(
        cx: &mut TestAppContext,
    ) {
        let SavedVectorFixture {
            original,
            origin,
            curve_to_artboard,
            rectangle_to_artboard,
        } = saved_vector_fixture(None).await;
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1050.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/f-vector.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("source ZIP");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open saved vector scene")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        for enabled in [false, true] {
            click(window, "GRAPHITE-GDD-CANVAS-SNAP-OPTIONS");
            click(window, "MENU_ITEM-Anchor Points");
            item.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("F-VECTOR Canvas");
                let document = canvas
                    .snapping_document
                    .as_ref()
                    .expect("F-VECTOR snapping");
                assert_eq!(document.snapping_state.path.anchor_point, enabled);
                assert!(!item.is_dirty(cx));
            });
        }
        let assert_pixels = |navigation: DocumentNavigation,
                             target: SavedSceneTarget,
                             image: &RenderImage,
                             origin: DVec2| {
            let pane = navigation.document_to_pane(target.logical_size());
            let sample = |point| {
                let position = (pane.transform_point2(origin + point) * target.scale).round();
                let index = (position.y as usize * image.size(0).width.0 as usize
                    + position.x as usize)
                    * 4;
                image
                    .as_bytes(0)
                    .and_then(|bytes| bytes.get(index..index + 4))
                    .expect("sample inside Canvas")
            };
            assert_eq!(
                sample(curve_to_artboard.transform_point2(DVec2::new(35., 30.))),
                [255, 0, 0, 255]
            );
            assert_eq!(
                sample(rectangle_to_artboard.transform_point2(DVec2::ZERO)),
                [0, 0, 255, 255]
            );
            assert_eq!(
                sample(curve_to_artboard.transform_point2(DVec2::new(35., 42.))),
                [255, 255, 255, 255]
            );
        };
        let assert_scene = |canvas: &SavedSceneCanvas, image: &RenderImage, origin: DVec2| {
            assert_pixels(
                canvas.navigation.expect("accepted Canvas camera"),
                canvas.target.expect("measured Canvas"),
                image,
                origin,
            );
        };
        item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let image = match &canvas.frame {
                SavedSceneFrame::Ready(image) => image,
                SavedSceneFrame::Failed(error) => panic!("F-VECTOR renderer failed: {error}"),
                _ => panic!("rendered F-VECTOR Canvas"),
            };
            assert_scene(canvas, image, origin);
            assert!(!item.is_dirty(cx));
        });
        click(window, "GRAPHITE-GDD-CANVAS-GRID");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("F-VECTOR Canvas");
            let grid = canvas.grid_frame.as_ref().expect("F-VECTOR grid");
            assert!(matches!(&grid.image, Ok(Some(_))));
            let SavedSceneFrame::Ready(image) = &canvas.frame else {
                panic!("accepted F-VECTOR frame")
            };
            assert_scene(canvas, image, origin);
            assert!(!item.is_dirty(cx));
        });
        click(window, "GRAPHITE-GDD-CANVAS-GRID");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        click(window, "GRAPHITE-GDD-CANVAS-MOVE-ARTBOARD");
        let bounds = window
            .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
            .expect("Canvas bounds");
        let viewport = item.read_with(window, |item, _| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            canvas
                .navigation
                .expect("accepted viewport")
                .document_to_pane(canvas.target.expect("measured Canvas").logical_size())
        });
        let at = |point| {
            let point = viewport.transform_point2(point);
            bounds.origin + gpui::point(px(point.x as f32), px(point.y as f32))
        };
        let outside = gpui::point(bounds.left() - px(70.), at(origin).y);
        for resize in [false, true] {
            if resize {
                click(window, "GRAPHITE-GDD-CANVAS-RESIZE-ARTBOARD");
            }
            let start = if resize {
                origin
            } else {
                origin + DVec2::splat(10.)
            };
            window.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                        position: at(start),
                        button: gpui::MouseButton::Left,
                        modifiers: Default::default(),
                        click_count: 1,
                        first_mouse: false,
                    }),
                    cx,
                );
                window.dispatch_event(
                    gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                        position: outside,
                        pressed_button: Some(gpui::MouseButton::Left),
                        modifiers: Default::default(),
                    }),
                    cx,
                );
            });
            window.run_until_parked();
            let preview = |window: &gpui::VisualTestContext| {
                item.read_with(window, |item, cx| {
                    let canvas = item.scene_canvas.as_ref().expect("Canvas");
                    let view = canvas
                        .artboard_view
                        .as_ref()
                        .expect("active Artboard viewport");
                    let target = canvas.target.expect("measured Canvas");
                    let drag = canvas
                        .artboard_drag
                        .as_ref()
                        .expect("active Artboard gesture");
                    let geometry = drag.placement.expect("draft placement");
                    assert!(view.draft_error.is_none(), "{:?}", view.draft_error);
                    let draft = view.draft_frame.as_ref().expect("rendered panning draft");
                    assert_eq!(draft.request.placement, Some(geometry));
                    assert_eq!(draft.request.camera, view.navigation.camera);
                    assert_eq!(view.pane, bounds);
                    assert!(!item.is_dirty(cx));
                    assert_pixels(
                        view.navigation,
                        target,
                        &draft.image,
                        if resize { origin } else { geometry.location },
                    );
                    (
                        geometry,
                        view.offset_from(target, view.initial.camera),
                        view.navigation.camera,
                    )
                })
            };
            let (before, offset, _) = preview(window);
            assert_eq!(offset, DVec2::ZERO);
            window
                .executor()
                .advance_clock(std::time::Duration::from_millis(16));
            window.update(|window, cx| window.simulate_next_frame(cx));
            window.run_until_parked();
            let (after, offset, camera) = preview(window);
            assert!(offset.abs_diff_eq(DVec2::new(16., 0.), 1e-9));
            assert!(after.location.x < before.location.x);
            if resize {
                assert_eq!(
                    after.location + after.dimensions,
                    before.location + before.dimensions
                );
            } else {
                assert_eq!(after.dimensions, before.dimensions);
            }
            window.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                        position: at(start),
                        button: gpui::MouseButton::Right,
                        modifiers: Default::default(),
                        click_count: 1,
                        first_mouse: false,
                    }),
                    cx,
                );
            });
            window
                .executor()
                .advance_clock(std::time::Duration::from_millis(16));
            window.update(|window, cx| window.simulate_next_frame(cx));
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                assert!(canvas.artboard_view.is_none());
                assert!(canvas.artboard_drag.is_none());
                assert_eq!(canvas.camera_override, Some(camera));
                let SavedSceneFrame::Ready(image) = &canvas.frame else {
                    panic!("committed vector canvas after cancel");
                };
                assert_scene(canvas, image, origin);
                assert!(!item.is_dirty(cx));
            });
            click(window, "GRAPHITE-GDD-CANVAS-FIT");
        }
        click(window, "GRAPHITE-GDD-CANVAS-MOVE-ARTBOARD");
        let displacement = DVec2::new(19., -14.);
        let start = origin + DVec2::new(100., 70.);
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position: at(start),
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                    position: at(start + displacement),
                    pressed_button: Some(gpui::MouseButton::Left),
                    modifiers: Default::default(),
                }),
                cx,
            );
        });
        window.run_until_parked();
        paint(window);
        let address = item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let drag = canvas.artboard_drag.as_ref().expect("F-VECTOR gesture");
            let draft = canvas
                .artboard_view
                .as_ref()
                .expect("Artboard view")
                .draft_frame
                .as_ref()
                .expect("moved curve and rectangle preview");
            assert_eq!(
                draft.request.placement.expect("move placement").location,
                origin + displacement
            );
            assert_scene(canvas, &draft.image, origin + displacement);
            assert!(!item.is_dirty(cx));
            drag.node.storage_id()
        });
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position: at(start + displacement),
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                }),
                cx,
            );
        });
        for (action, expected) in [
            (None, origin + displacement),
            (Some(true), origin),
            (Some(false), origin + displacement),
        ] {
            match action {
                Some(true) => window.dispatch_action(editor::actions::Undo),
                Some(false) => window.dispatch_action(editor::actions::Redo),
                None => {}
            }
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, _| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let SavedSceneFrame::Ready(image) = &canvas.frame else {
                    panic!("committed F-VECTOR scene");
                };
                assert_scene(canvas, image, expected);
            });
        }
        let moved_origin = origin + displacement;
        let resized = GddArtboardGeometry {
            location: moved_origin - DVec2::new(15., 10.),
            dimensions: DVec2::new(235., 170.),
        };
        click(window, "GRAPHITE-GDD-CANVAS-RESIZE-ARTBOARD");
        let resize = |window: &mut gpui::VisualTestContext| {
            window.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                        position: at(moved_origin),
                        button: gpui::MouseButton::Left,
                        modifiers: Default::default(),
                        click_count: 1,
                        first_mouse: false,
                    }),
                    cx,
                );
                window.dispatch_event(
                    gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                        position: at(resized.location),
                        pressed_button: Some(gpui::MouseButton::Left),
                        modifiers: Default::default(),
                    }),
                    cx,
                );
            });
            window.run_until_parked();
            paint(window);
        };
        let release_resize = |window: &mut gpui::VisualTestContext| {
            window.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                        position: at(resized.location),
                        button: gpui::MouseButton::Left,
                        modifiers: Default::default(),
                        click_count: 1,
                    }),
                    cx,
                );
            });
            window.run_until_parked();
            paint(window);
        };
        resize(window);
        let draft_image = item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let drag = canvas.artboard_drag.as_ref().expect("captured resize");
            assert!(matches!(
                drag.gesture,
                SavedSceneArtboardGesture::Resize(..)
            ));
            let view = canvas.artboard_view.as_ref().expect("Artboard view");
            assert!(view.draft_error.is_none(), "{:?}", view.draft_error);
            let draft = view.draft_frame.as_ref().expect("resized Canvas draft");
            assert_eq!(draft.request.placement, Some(resized));
            assert_scene(canvas, &draft.image, moved_origin);
            let model = item.model.read(cx);
            let board = model
                .working
                .as_ref()
                .expect("working copy")
                .inspection
                .nodes
                .iter()
                .find(|node| node.storage_id() == address)
                .expect("same Artboard");
            assert_eq!(
                board
                    .artboard_geometry()
                    .expect("committed geometry")
                    .location,
                moved_origin
            );
            draft.image.clone()
        });
        assert!(window.update(|window, _| window.has_image_atlas_entry(&draft_image)));
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position: at(resized.location),
                    button: gpui::MouseButton::Right,
                    modifiers: Default::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            );
        });
        release_resize(window);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&draft_image)));
        resize(window);
        release_resize(window);
        for (action, expected) in [
            (None, resized),
            (
                Some(true),
                GddArtboardGeometry {
                    location: moved_origin,
                    dimensions: DVec2::new(220., 160.),
                },
            ),
            (Some(false), resized),
        ] {
            match action {
                Some(true) => window.dispatch_action(editor::actions::Undo),
                Some(false) => window.dispatch_action(editor::actions::Redo),
                None => {}
            }
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, cx| {
                assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let SavedSceneFrame::Ready(image) = &canvas.frame else {
                    panic!("rendered resized F-VECTOR");
                };
                assert_scene(canvas, image, moved_origin);
                let model = item.model.read(cx);
                let board = model
                    .working
                    .as_ref()
                    .expect("working copy")
                    .inspection
                    .nodes
                    .iter()
                    .find(|node| node.storage_id() == address)
                    .expect("same Artboard");
                assert_eq!(board.artboard_geometry(), Some(expected));
            });
        }
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        let destination = PathBuf::from("/project/f-vector-resized.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopened resized F-VECTOR");
        let nodes = saved.graph_nodes().await.expect("reopened source nodes");
        let board = nodes
            .iter()
            .find(|node| node.storage_id() == address)
            .expect("same source Artboard");
        assert_eq!(board.artboard_geometry(), Some(resized));
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("original archive")
                .as_slice(),
            original.as_ref()
        );
    }

    #[gpui::test]
    async fn at_gra_tool_008_2_workspace_picks_and_moves_artboards_in_stack_order(
        cx: &mut TestAppContext,
    ) {
        use graphite_core_types::Color;
        use graphite_document_graph_storage::NodeMetadataSource;
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_network_node_type;
        use graphite_graph_craft::document::{NodeId, NodeInput, NodeNetwork, value::TaggedValue};
        use graphite_resource::ResourceRegistry;

        struct ArtboardMetadata;

        impl NodeMetadataSource for ArtboardMetadata {
            fn is_layer(&self, path: &[NodeId], _: NodeId) -> bool {
                path.is_empty()
            }

            fn reference(&self, path: &[NodeId]) -> Option<&str> {
                matches!(path, [NodeId(15 | 25 | 35)]).then_some("Artboard")
            }

            fn display_name(&self, path: &[NodeId], node: NodeId) -> Option<&str> {
                match (path, node) {
                    ([], NodeId(15)) => Some("Front"),
                    ([], NodeId(25)) => Some("Back"),
                    ([], NodeId(35)) => Some("Unconnected"),
                    _ => None,
                }
            }
        }

        let front = NodeId(15);
        let back = NodeId(25);
        let unconnected = NodeId(35);
        let definition = resolve_network_node_type("Artboard").expect("pinned Artboard");
        let board = |base: Option<NodeId>, x, width, color| {
            definition
                .node_template_input_override([
                    base.map(|id| NodeInput::node(id, 0)),
                    None,
                    Some(NodeInput::value(
                        TaggedValue::DVec2(DVec2::new(x, -40.)),
                        false,
                    )),
                    Some(NodeInput::value(
                        TaggedValue::DVec2(DVec2::new(width, 80.)),
                        false,
                    )),
                    Some(NodeInput::value(TaggedValue::Color(color), false)),
                ])
                .into_parts()
                .0
        };
        let graph = NodeNetwork {
            nodes: [
                (front, board(Some(back), -60., 120., Color::RED)),
                (back, board(None, 10., 150., Color::BLUE)),
                (unconnected, board(None, 1000., 200., Color::GREEN)),
            ]
            .into_iter()
            .collect(),
            exports: vec![NodeInput::node(front, 0)],
            ..Default::default()
        };
        let mut owner = GddV1::create_in(
            AnyContainer::Memory(MemoryBackend::new()),
            GddV1Layout,
            Default::default(),
            655,
            "Pinned Graphite".into(),
            "Pinned library".into(),
        )
        .expect("saved GDD");
        let resources = HashMapResourceStorage::new();
        owner
            .commit_from_runtime(
                &graph,
                &ArtboardMetadata,
                &ResourceRegistry::new(),
                &resources,
            )
            .expect("pinned Artboard stack");
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
        let original_graph = open_gdd_archive(&original, ARCHIVE_BUDGET)
            .await
            .expect("original graph");
        let nodes = original_graph
            .graph_nodes()
            .await
            .expect("original addresses");
        let find = |name| {
            nodes
                .iter()
                .find(|node| node.display_name() == name)
                .expect("saved Artboard")
        };
        let front_node = find("Front");
        let back_node = find("Back");
        let back_address = back_node.storage_id();
        let viewport = UVec2::new(300, 180);
        let limits = CpuSvgEvaluationBudget {
            max_viewport_pixels: viewport.x * viewport.y,
            max_output_bytes: 64 * 1024,
        };
        for (target, change, other, visible, size) in [
            (
                front_node,
                GddRootLayerChange::Visibility(false),
                back_node,
                1,
                DVec2::new(150., 80.),
            ),
            (
                back_node,
                GddRootLayerChange::Visibility(false),
                front_node,
                1,
                DVec2::new(120., 80.),
            ),
            (
                front_node,
                GddRootLayerChange::Locked(true),
                back_node,
                2,
                DVec2::new(220., 80.),
            ),
        ] {
            let contents = original_graph
                .edit_root_layer_in_copy(target, change, ARCHIVE_BUDGET)
                .await
                .expect("root Artboard state");
            let output = contents
                .render_document_canvas_svg(viewport, 1., None, limits)
                .await
                .expect("render with changed layer");
            assert_eq!(output.artboards.len(), visible);
            assert_eq!(output.document_size, size);
            let mut picks = output
                .artboards
                .iter()
                .filter_map(|layer| layer.pick_target.as_ref());
            let pick = picks.next().expect("a visible unlocked Artboard");
            assert!(picks.next().is_none());
            let nodes = contents.graph_nodes().await.expect("variant addresses");
            let current = |original: &GddGraphNode| {
                nodes
                    .iter()
                    .find(|node| node.storage_id() == original.storage_id())
                    .expect("same Artboard")
            };
            assert!(pick.matches_node(current(other)));
            assert!(matches!(
                contents
                    .render_document_artboard_move_draft_svg(
                        current(target),
                        DVec2::ZERO,
                        (viewport, 1.),
                        output.camera,
                        limits
                    )
                    .await,
                Err(GddArchiveError::UnsupportedArtboardGeometry),
            ));
        }

        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1050.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/artboard-stack.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("original ZIP");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open Artboard stack")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        let open = "GRAPHITE-GDD-OPEN-CANVAS";
        click(window, open);
        let pose = |window: &mut gpui::VisualTestContext| {
            let bounds = window
                .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
                .expect("Canvas bounds");
            let transform = item.read_with(window, |item, _| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let navigation = canvas.navigation.expect("fitted stack");
                assert_eq!(navigation.document_size, DVec2::new(220., 80.));
                navigation.document_to_pane(canvas.target.expect("Canvas target").logical_size())
            });
            (bounds, transform)
        };
        let at = |bounds: gpui::Bounds<gpui::Pixels>, transform: DAffine2, position: DVec2| {
            let position = transform.transform_point2(position);
            bounds.origin + gpui::point(px(position.x as f32), px(position.y as f32))
        };
        let sample = |image: &RenderImage, position: DVec2| {
            let position = position.round();
            let index =
                (position.y as usize * image.size(0).width.0 as usize + position.x as usize) * 4;
            image
                .as_bytes(0)
                .and_then(|bytes| bytes.get(index..index + 4))
                .expect("sample inside Canvas rendering")
                .to_vec()
        };
        let (bounds, transform) = pose(window);
        item.read_with(window, |item, _| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            assert_eq!(canvas.artboards.len(), 2);
            assert!(canvas.artboards.iter().all(|board| board.overlay.is_some()));
            let SavedSceneFrame::Ready(image) = &canvas.frame else {
                panic!("stack image")
            };
            let scale = canvas.target.expect("Canvas target").scale;
            assert_eq!(
                sample(
                    image,
                    transform.transform_point2(DVec2::new(30., 0.)) * scale
                ),
                [0, 0, 255, 255]
            );
            assert_eq!(
                sample(
                    image,
                    transform.transform_point2(DVec2::new(110., 0.)) * scale
                ),
                [255, 0, 0, 255]
            );
        });
        for (sample, address) in [
            (DVec2::new(30., 0.), front_node.storage_id()),
            (DVec2::new(110., 0.), back_address),
        ] {
            click(window, "GRAPHITE-GDD-CANVAS-PICK");
            window.simulate_click(at(bounds, transform, sample), gpui::Modifiers::default());
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, cx| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("picked Artboard Inspector")
                };
                assert_eq!(
                    form.selected_node().map(GddGraphNode::storage_id),
                    Some(address)
                );
                assert!(!item.is_dirty(cx));
            });
            click(window, open);
        }
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, open);
        click(window, "GRAPHITE-GDD-CANVAS-MOVE-ARTBOARD");
        let (bounds, transform) = pose(window);
        let start = at(bounds, transform, DVec2::new(110., -10.));
        let end = at(bounds, transform, DVec2::new(111., 69.));
        let constrained = gpui::Modifiers {
            shift: true,
            ..Default::default()
        };
        let pointer = |window: &mut gpui::VisualTestContext, event| {
            window.update(|window, cx| window.dispatch_event(event, cx));
            window.run_until_parked();
            paint(window);
        };
        pointer(
            window,
            gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                position: start,
                button: gpui::MouseButton::Left,
                modifiers: Default::default(),
                click_count: 1,
                first_mouse: false,
            }),
        );
        pointer(
            window,
            gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                position: end,
                pressed_button: Some(gpui::MouseButton::Left),
                modifiers: constrained,
            }),
        );
        item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let moving = canvas.artboard_drag.as_ref().expect("back Artboard move");
            assert_eq!(moving.node.storage_id(), back_address);
            let placement = moving.placement.expect("snapped move placement");
            assert_eq!(placement.location, DVec2::new(10., 40.));
            assert_eq!(placement.dimensions, DVec2::new(150., 80.));
            assert!(moving.snapping.snap_point().is_some());
            let view = canvas.artboard_view.as_ref().expect("Artboard view");
            assert!(
                view.snap_guide_error.is_none(),
                "{:?}",
                view.snap_guide_error
            );
            assert!(view.snap_guides.is_some());
            let draft = canvas
                .artboard_view
                .as_ref()
                .expect("Artboard view")
                .draft_frame
                .as_ref()
                .expect("rendered stack draft");
            assert!(
                draft
                    .artboards
                    .iter()
                    .all(|board| board.pick_target.is_none())
            );
            let scale = canvas.target.expect("Canvas target").scale;
            let sample = |x, y| {
                sample(
                    &draft.image,
                    transform.transform_point2(DVec2::new(x, y)) * scale,
                )
            };
            assert_eq!(sample(30., 0.), [0, 0, 255, 255]);
            assert_eq!(sample(110., -10.), [0; 4]);
            assert_eq!(sample(110., 70.), [255, 0, 0, 255]);
            assert!(!item.is_dirty(cx));
        });
        pointer(
            window,
            gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                position: end,
                button: gpui::MouseButton::Left,
                modifiers: constrained,
                click_count: 1,
            }),
        );
        let check_stack = |inspection: &GddGraphInspection, back_y, back_width| {
            let root = inspection
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("root network");
            let (front, _) = root
                .connected_export(&inspection.nodes)
                .expect("original export");
            assert_eq!(front.storage_id(), front_node.storage_id());
            assert_eq!(
                front.artboard_geometry().expect("front geometry").location,
                DVec2::new(-60., -40.)
            );
            let back = inspection
                .nodes
                .iter()
                .find(|node| node.storage_id() == back_address)
                .expect("same back Artboard");
            assert_eq!(
                back.artboard_geometry(),
                Some(GddArtboardGeometry {
                    location: DVec2::new(10., back_y),
                    dimensions: DVec2::new(back_width, 80.),
                }),
            );
        };
        for (action, expected) in [(None, 40.), (Some(true), -40.), (Some(false), 40.)] {
            match action {
                Some(true) => window.dispatch_action(editor::actions::Undo),
                Some(false) => window.dispatch_action(editor::actions::Redo),
                None => {}
            }
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, cx| {
                let working = item.model.read(cx).working.as_ref().expect("working stack");
                check_stack(&working.inspection, expected, 150.);
            });
        }
        click(window, "GRAPHITE-GDD-CANVAS-RESIZE-ARTBOARD");
        let edge = at(bounds, transform, DVec2::new(160., 80.));
        let near_neighbor = at(bounds, transform, DVec2::new(61., 80.));
        let snapped_resize = GddArtboardGeometry {
            location: DVec2::new(10., 40.),
            dimensions: DVec2::new(50., 80.),
        };
        let resize_history = item.read_with(window, |item, cx| {
            item.model
                .read(cx)
                .gdd_contents()
                .expect("working GDD")
                .history_len()
        });
        let draft_resize = |window: &mut gpui::VisualTestContext| {
            pointer(
                window,
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position: edge,
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
            );
            pointer(
                window,
                gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                    position: near_neighbor,
                    pressed_button: Some(gpui::MouseButton::Left),
                    modifiers: Default::default(),
                }),
            );
            item.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let drag = canvas.artboard_drag.as_ref().expect("back Artboard resize");
                assert_eq!(drag.node.storage_id(), back_address);
                assert_eq!(drag.placement, Some(snapped_resize));
                assert!(drag.snapping.snap_point().is_some());
                let view = canvas.artboard_view.as_ref().expect("Artboard view");
                assert!(
                    view.snap_guide_error.is_none(),
                    "{:?}",
                    view.snap_guide_error
                );
                assert!(view.snap_guides.is_some());
                assert!(view.draft_error.is_none(), "{:?}", view.draft_error);
                let draft = view.draft_frame.as_ref().expect("source resize draft");
                assert_eq!(draft.request.placement, Some(snapped_resize));
                let scale = canvas.target.expect("Canvas target").scale;
                assert_eq!(
                    sample(
                        &draft.image,
                        transform.transform_point2(DVec2::new(35., 80.)) * scale
                    ),
                    [255, 0, 0, 255]
                );
                assert_eq!(
                    sample(
                        &draft.image,
                        transform.transform_point2(DVec2::new(80., 80.)) * scale
                    ),
                    [0; 4]
                );
                assert_eq!(
                    item.model
                        .read(cx)
                        .gdd_contents()
                        .expect("working GDD")
                        .history_len(),
                    resize_history
                );
            });
        };
        let release_resize = |window: &mut gpui::VisualTestContext| {
            pointer(
                window,
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position: near_neighbor,
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                }),
            );
        };
        draft_resize(window);
        pointer(
            window,
            gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                position: near_neighbor,
                button: gpui::MouseButton::Right,
                modifiers: Default::default(),
                click_count: 1,
                first_mouse: false,
            }),
        );
        release_resize(window);
        item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            check_stack(
                &model.working.as_ref().expect("working stack").inspection,
                40.,
                150.,
            );
            assert_eq!(
                model.gdd_contents().expect("working GDD").history_len(),
                resize_history
            );
        });
        draft_resize(window);
        release_resize(window);
        for (action, width) in [(None, 50.), (Some(true), 150.), (Some(false), 50.)] {
            match action {
                Some(true) => window.dispatch_action(editor::actions::Undo),
                Some(false) => window.dispatch_action(editor::actions::Redo),
                None => {}
            }
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, cx| {
                assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
                let working = item.model.read(cx).working.as_ref().expect("working stack");
                check_stack(&working.inspection, 40., width);
            });
        }
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        let destination = PathBuf::from("/project/artboard-stack-edited.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let reopened = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen edited stack");
        check_stack(
            &reopened.graph_inspection().await.expect("saved stack"),
            40.,
            50.,
        );
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("original ZIP")
                .as_slice(),
            original.as_ref()
        );
    }

    #[gpui::test]
    async fn at_gra_view_006_2_workspace_grid_visibility_is_independent_of_snapping(
        cx: &mut TestAppContext,
    ) {
        let source = create_gdd_folder(667, 187, "Canvas Grid".into(), ARCHIVE_BUDGET)
            .await
            .expect("new Graphite document")
            .create_document_artboard_in_copy(
                GddArtboardGeometry {
                    location: DVec2::new(-50., -40.),
                    dimensions: DVec2::new(100., 80.),
                },
                ARCHIVE_BUDGET,
            )
            .await
            .expect("pinned Artboard");
        let history = source.history_len();
        let original = source
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.update(|_, cx| {
            let bindings =
                settings::KeymapFile::load_asset(crate::DEFAULT_GRAPHITE_KEYMAP_PATH, None, cx)
                    .expect("Graphite keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(px(1160.), px(1000.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/grid-visibility.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source ZIP");
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        let (revision, instance, content_frame) = source.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            assert!(canvas.grid_frame.is_none());
            assert!(!canvas.source_grid_visible);
            let SavedSceneFrame::Ready(image) = &canvas.frame else {
                panic!("accepted Canvas frame")
            };
            (
                item.model.read(cx).revision.clone(),
                canvas.instance.clone(),
                image.clone(),
            )
        });
        let read_grid = |window: &gpui::VisualTestContext, snapping: bool| {
            source.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let model = item.model.read(cx);
                let state = &canvas
                    .snapping_document
                    .as_ref()
                    .expect("source snapping")
                    .snapping_state;
                assert_eq!(state.grid_snapping, snapping);
                assert!(Arc::ptr_eq(&model.revision, &revision));
                assert_eq!(
                    model.gdd_contents().expect("source snapshot").history_len(),
                    history
                );
                assert!(!item.is_dirty(cx));
                assert!(!item.can_save(cx));
                canvas.grid_frame.as_ref().map(|frame| {
                    let image = frame
                        .image
                        .as_ref()
                        .expect("source grid rendered")
                        .as_ref()
                        .expect("non-empty grid")
                        .clone();
                    assert!(
                        image
                            .as_bytes(0)
                            .expect("grid pixels")
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .any(|pixel| pixel[3] > 0)
                    );
                    (frame.request.clone(), image)
                })
            })
        };
        let settle_grid = |window: &mut gpui::VisualTestContext| {
            for _ in 0..2 {
                window.run_until_parked();
                paint(window);
            }
        };
        click(window, "GRAPHITE-GDD-CANVAS-GRID");
        settle_grid(window);
        let (request, grid) =
            read_grid(window, false).expect("visible grid with snapping disabled");
        source.read_with(window, |item, _| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            assert!(matches!(&canvas.frame, SavedSceneFrame::Ready(image) if Arc::ptr_eq(image, &content_frame)));
        });
        assert!(window.update(|window, _| window.has_image_atlas_entry(&grid)));

        click(window, "GRAPHITE-GDD-CANVAS-SNAP-OPTIONS");
        click(window, "MENU_ITEM-Grid");
        let (_, unchanged_grid) =
            read_grid(window, true).expect("visible grid after changing snapping");
        assert!(Arc::ptr_eq(&grid, &unchanged_grid));
        window.simulate_keystrokes("alt-g");
        settle_grid(window);
        assert!(read_grid(window, true).is_none());
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&grid)));

        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        assert!(read_grid(window, true).is_none());
        source.update_in(window, |item, _, cx| {
            assert!(!item.toggle_scene_grid(&instance, &revision, cx));
        });
        click(window, "GRAPHITE-GDD-CANVAS-GRID");
        settle_grid(window);
        let (reopened, _) = read_grid(window, true).expect("newly visible grid");
        assert_eq!(reopened.resolution, request.resolution);
        window.simulate_resize(gpui::size(px(1280.), px(1020.)));
        settle_grid(window);
        let (resized, before_zoom) = read_grid(window, true).expect("grid fitted to new bounds");
        assert_ne!(resized.resolution, reopened.resolution);
        let pane = window
            .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
            .expect("Canvas pane");
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::ScrollWheel(gpui::ScrollWheelEvent {
                    position: pane.center(),
                    delta: gpui::ScrollDelta::Pixels(gpui::point(px(0.), px(120.))),
                    modifiers: gpui::Modifiers {
                        control: true,
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                cx,
            );
        });
        settle_grid(window);
        let (zoomed, after_zoom) = read_grid(window, true).expect("grid follows accepted zoom");
        assert!(zoomed.camera.zoom() > resized.camera.zoom());
        assert_eq!(zoomed.resolution, resized.resolution);
        assert!(!Arc::ptr_eq(&before_zoom, &after_zoom));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&before_zoom)));
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_view_008_2_workspace_shares_snap_controls_without_editing_artwork(
        cx: &mut TestAppContext,
    ) {
        let source = create_gdd_folder(664, 187, "Snap Controls".into(), ARCHIVE_BUDGET)
            .await
            .expect("new Graphite document")
            .create_document_artboard_in_copy(
                GddArtboardGeometry {
                    location: DVec2::new(-50., -40.),
                    dimensions: DVec2::new(100., 80.),
                },
                ARCHIVE_BUDGET,
            )
            .await
            .expect("pinned Artboard");
        let history = source.history_len();
        let original = source
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.update(|_, cx| {
            let bindings =
                settings::KeymapFile::load_asset(crate::DEFAULT_GRAPHITE_KEYMAP_PATH, None, cx)
                    .expect("Graphite keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(px(1320.), px(1000.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let path = PathBuf::from("/project/snap-controls.gdd.zip");
        filesystem
            .write(&path, &original)
            .await
            .expect("source ZIP");
        let source = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(path.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open source")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        let (revision, base, instance, frame) = source.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let SavedSceneFrame::Ready(frame) = &canvas.frame else {
                panic!("accepted Canvas frame")
            };
            (
                model.revision.clone(),
                model.gdd_contents().expect("saved snapshot").clone(),
                canvas.instance.clone(),
                frame.clone(),
            )
        });
        let toggle_option = |window: &mut gpui::VisualTestContext, selector| {
            click(window, "GRAPHITE-GDD-CANVAS-SNAP-OPTIONS");
            click(window, selector);
        };
        toggle_option(window, "MENU_ITEM-Corner Points");
        toggle_option(window, "MENU_ITEM-Anchor Points");
        toggle_option(window, "MENU_ITEM-Grid");
        click(window, "GRAPHITE-GDD-CANVAS-SNAPPING");
        let check_view =
            |view: &Entity<GraphiteDocumentItem>, window: &gpui::VisualTestContext, enabled| {
                view.read_with(window, |item, cx| {
                    let model = item.model.read(cx);
                    let canvas = item.scene_canvas.as_ref().expect("accepted Canvas");
                    let state = &canvas
                        .snapping_document
                        .as_ref()
                        .expect("source snapping context")
                        .snapping_state;
                    assert_eq!(state.snapping_enabled, enabled);
                    assert!(!state.bounding_box.corner_point);
                    assert!(!state.path.anchor_point);
                    assert!(state.grid_snapping);
                    assert!(state.artboards);
                    assert!(Arc::ptr_eq(&model.revision, &revision));
                    assert!(Arc::ptr_eq(
                        model.gdd_contents().expect("saved snapshot"),
                        &base
                    ));
                    assert_eq!(base.history_len(), history);
                    assert!(!item.is_dirty(cx));
                    assert!(!item.can_save(cx));
                });
            };
        check_view(&source, window, false);
        source.read_with(window, |item, _| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            assert!(matches!(&canvas.frame, SavedSceneFrame::Ready(current) if Arc::ptr_eq(current, &frame)));
        });
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        check_view(&source, window, false);
        source.update_in(window, |item, _, cx| {
            assert!(!item.toggle_scene_snapping(&instance, &revision, CANVAS_SNAP_ENABLED, cx));
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
                .expect("split native Inspector")
        });
        assert_ne!(source, split);
        split.update_in(window, |item, window, cx| {
            item.open_scene_canvas(window, cx)
        });
        window.run_until_parked();
        paint(window);
        check_view(&source, window, false);
        check_view(&split, window, false);
        window.simulate_keystrokes("alt-s");
        window.run_until_parked();
        paint(window);
        check_view(&source, window, true);
        check_view(&split, window, true);
        assert_eq!(
            filesystem.load_bytes(&path).await.expect("original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_tool_008_2_workspace_draws_and_stacks_artboards_on_canvas(
        cx: &mut TestAppContext,
    ) {
        let blank = create_gdd_folder(621, 187, "Blank Artboard Canvas".into(), ARCHIVE_BUDGET)
            .await
            .expect("blank Graphite document");
        let original = blank
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.update(|_, cx| {
            let bindings =
                settings::KeymapFile::load_asset(crate::DEFAULT_GRAPHITE_KEYMAP_PATH, None, cx)
                    .expect("Graphite keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(px(1160.), px(1050.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/blank-artboard.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("source ZIP");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the blank document")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        let canvas_button = "GRAPHITE-GDD-OPEN-CANVAS";
        let tool = "GRAPHITE-GDD-CANVAS-DRAW-ARTBOARD";
        click(window, canvas_button);
        assert!(window.debug_bounds(tool).is_none());
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, canvas_button);
        click(window, tool);

        let pane = window
            .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
            .expect("Canvas pane");
        let center = item.read_with(window, |item, _| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            assert!(matches!(canvas.frame, SavedSceneFrame::Blank));
            let navigation = DocumentNavigation {
                camera: PTZ::default(),
                document_size: DVec2::ZERO,
            };
            navigation
                .document_to_pane(canvas.target.expect("measured Canvas").logical_size())
                .transform_point2(DVec2::ZERO)
        });
        let start = pane.origin + gpui::point(px(center.x as f32 + 40.), px(center.y as f32 - 35.));
        let end = start + gpui::point(px(-70.), px(30.));
        let modifiers = gpui::Modifiers {
            shift: true,
            alt: true,
            ..Default::default()
        };
        let pointer = |window: &mut gpui::VisualTestContext, event| {
            window.update(|window, cx| window.dispatch_event(event, cx));
            window.run_until_parked();
            paint(window);
        };
        let down = || {
            gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                position: start,
                button: gpui::MouseButton::Left,
                modifiers: gpui::Modifiers::default(),
                click_count: 1,
                first_mouse: false,
            })
        };
        let drag = || {
            gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                position: end,
                pressed_button: Some(gpui::MouseButton::Left),
                modifiers,
            })
        };
        let up = |position| {
            gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                position,
                button: gpui::MouseButton::Left,
                modifiers,
                click_count: 1,
            })
        };
        let expected = GddArtboardGeometry {
            location: DVec2::new(-30., -105.),
            dimensions: DVec2::splat(140.),
        };
        let assert_blank = |window: &gpui::VisualTestContext, before_first_edit: bool| {
            item.read_with(window, |item, cx| {
                let model = item.model.read(cx);
                let working = model.working.as_ref().expect("working document");
                assert!(working.inspection.document_output_is_empty());
                assert!(working.inspection.nodes.is_empty());
                if before_first_edit {
                    assert!(!item.is_dirty(cx));
                }
            });
        };
        pointer(window, down());
        pointer(window, up(start));
        assert_blank(window, true);
        pointer(window, down());
        pointer(window, drag());
        item.read_with(window, |item, _| {
            let draft = item
                .scene_canvas
                .as_ref()
                .and_then(|canvas| canvas.artboard_draw.as_ref())
                .expect("native draw gesture");
            assert_eq!(draft.placement, Some(expected));
        });
        assert_blank(window, true);
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        pointer(window, up(end));
        assert_blank(window, true);
        pointer(window, down());
        let outside = gpui::point(pane.left() - px(70.), end.y);
        pointer(
            window,
            gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                position: outside,
                pressed_button: Some(gpui::MouseButton::Left),
                modifiers,
            }),
        );
        let draw_state = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, _| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let view = canvas.artboard_view.as_ref().expect("Artboard viewport");
                let placement = canvas
                    .artboard_draw
                    .as_ref()
                    .expect("draw gesture")
                    .placement;
                let target = canvas.target.expect("measured Canvas");
                (placement, view.offset_from(target, view.initial.camera))
            })
        };
        let (before, _) = draw_state(window);
        for _ in 0..2 {
            window
                .executor()
                .advance_clock(std::time::Duration::from_millis(16));
            window.update(|window, cx| window.simulate_next_frame(cx));
        }
        let (after, offset) = draw_state(window);
        assert_eq!(offset, DVec2::new(32., 0.));
        assert!(
            after.expect("panned draw").dimensions.x > before.expect("initial draw").dimensions.x
        );
        assert_blank(window, true);
        let shifted_end = end + gpui::point(px(32.), px(0.));
        pointer(
            window,
            gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                position: shifted_end,
                pressed_button: Some(gpui::MouseButton::Left),
                modifiers,
            }),
        );
        window
            .executor()
            .advance_clock(std::time::Duration::from_millis(16));
        window.update(|window, cx| window.simulate_next_frame(cx));
        assert_eq!(draw_state(window), (Some(expected), offset));
        pointer(window, up(shifted_end));
        window.run_until_parked();
        paint(window);
        let address = item.read_with(window, |item, cx| {
            assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
            assert!(item.is_dirty(cx));
            let model = item.model.read(cx);
            let graph = &model.working.as_ref().expect("working document").inspection;
            let root = graph
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("root");
            let (board, outlet) = root
                .connected_export(&graph.nodes)
                .expect("Artboard document output");
            assert_eq!(outlet, 0);
            assert_eq!(board.artboard_geometry(), Some(expected));
            assert_eq!(
                board.artboard_background(),
                Some(graphite_std::Color::WHITE)
            );
            assert_eq!(
                board
                    .boolean_input()
                    .map(|property| (property.name, property.value)),
                Some(("Clip", true))
            );
            let canvas = item.scene_canvas.as_ref().expect("updated Canvas");
            let [artboard] = canvas.artboards.as_slice() else {
                panic!("one Artboard on Canvas");
            };
            let pick = artboard
                .pick_target
                .as_ref()
                .expect("source-rendered Canvas");
            assert!(pick.matches_node(board) && pick.contains(center + offset));
            assert!(matches!(canvas.frame, SavedSceneFrame::Ready(_)));
            board.storage_id()
        });
        window.dispatch_action(editor::actions::Undo);
        window.run_until_parked();
        paint(window);
        assert_blank(window, false);
        window.dispatch_action(editor::actions::Redo);
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let graph = &item
                .model
                .read(cx)
                .working
                .as_ref()
                .expect("working document")
                .inspection;
            let root = graph
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("root");
            let (board, _) = root
                .connected_export(&graph.nodes)
                .expect("redone Artboard output");
            assert_eq!(board.storage_id(), address);
            assert_eq!(board.artboard_geometry(), Some(expected));
        });
        let (pane, transform, history) = {
            let pane = window
                .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
                .expect("Canvas pane after redo");
            let (transform, history) = item.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let target = canvas.target.expect("measured Canvas");
                let navigation = canvas.drawing_navigation().expect("ready Canvas");
                let history = item
                    .model
                    .read(cx)
                    .gdd_contents()
                    .expect("working GDD")
                    .history_len();
                (navigation.document_to_pane(target.logical_size()), history)
            });
            (pane, transform, history)
        };
        let move_pointer = |window: &mut gpui::VisualTestContext, position| {
            pointer(
                window,
                gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                    position,
                    pressed_button: Some(gpui::MouseButton::Left),
                    modifiers: Default::default(),
                }),
            );
        };
        let at = |position| {
            let point = transform.transform_point2(position);
            pane.origin + gpui::point(px(point.x as f32), px(point.y as f32))
        };
        let draft_again = |window: &mut gpui::VisualTestContext, start, end| {
            pointer(
                window,
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position: at(start),
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
            );
            move_pointer(window, at(end));
        };
        let release_again = |window: &mut gpui::VisualTestContext, end| {
            pointer(
                window,
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position: at(end),
                    button: gpui::MouseButton::Left,
                    modifiers: Default::default(),
                    click_count: 1,
                }),
            );
            window.run_until_parked();
            paint(window);
        };
        let second_geometry = GddArtboardGeometry {
            location: DVec2::new(-30., -70.),
            dimensions: DVec2::new(140., 140.),
        };
        let assert_stack = |graph: &GddGraphInspection| {
            let root = graph
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("root");
            let (new_front, output) = root
                .connected_export(&graph.nodes)
                .expect("new root Artboard");
            assert_eq!(output, 0);
            assert_ne!(new_front.storage_id(), address);
            assert_eq!(new_front.artboard_geometry(), Some(second_geometry));
            assert_eq!(
                new_front
                    .input_ports()
                    .first()
                    .and_then(|input| input.source.as_ref()),
                Some(&GddGraphPortSource::Node {
                    storage_id: address,
                    output_index: 0
                })
            );
            let original = graph
                .nodes
                .iter()
                .find(|node| node.storage_id() == address)
                .expect("original Artboard");
            assert_eq!(original.artboard_geometry(), Some(expected));
            new_front.storage_id()
        };
        draft_again(window, DVec2::ZERO, DVec2::new(160., 70.));
        release_again(window, DVec2::new(160., 70.));
        item.read_with(window, |item, cx| {
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("working GDD")
                    .history_len(),
                history
            );
        });
        let start = DVec2::new(111., -70.);
        let end = DVec2::new(-29., 70.);
        draft_again(window, start, end);
        let assert_snapped = |window: &gpui::VisualTestContext, expected_pan| {
            item.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                let draw = canvas
                    .artboard_draw
                    .as_ref()
                    .expect("snapping draw gesture");
                let view = canvas.artboard_view.as_ref().expect("Artboard viewport");
                let target = canvas.target.expect("measured Canvas");
                assert_eq!(view.offset_from(target, view.initial.camera), expected_pan);
                assert_eq!(draw.placement, Some(second_geometry));
                assert!(
                    view.snap_guide_error.is_none(),
                    "{:?}",
                    view.snap_guide_error
                );
                assert!(view.snap_guides.is_some());
                let snapped = draw.gesture.snap_point().expect("Graphite snap indicator");
                assert!(snapped.abs_diff_eq(DVec2::new(-30., 70.), 1e-9));
                assert_eq!(
                    item.model
                        .read(cx)
                        .gdd_contents()
                        .expect("working GDD")
                        .history_len(),
                    history,
                );
            });
        };
        assert_snapped(window, DVec2::ZERO);
        window.simulate_keystrokes("alt-s");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let draw = canvas.artboard_draw.as_ref().expect("active draw gesture");
            let unsnapped = draw.placement.expect("unsnapped draft");
            assert_eq!(unsnapped.dimensions.x, 139.);
            assert_eq!(unsnapped.location.x, -29.);
            assert!(draw.gesture.snap_point().is_none());
            assert!(
                canvas
                    .artboard_view
                    .as_ref()
                    .expect("Artboard view")
                    .snap_guides
                    .is_none()
            );
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("working GDD")
                    .history_len(),
                history
            );
        });
        window.simulate_keystrokes("alt-s");
        window.run_until_parked();
        paint(window);
        assert_snapped(window, DVec2::ZERO);

        let pivot = at(end);
        for (outside, offset) in [(pane.left() - px(70.), 16.), (pane.right() + px(70.), 0.)] {
            move_pointer(window, gpui::point(outside, pivot.y));
            window
                .executor()
                .advance_clock(std::time::Duration::from_millis(16));
            window.update(|window, cx| window.simulate_next_frame(cx));
            move_pointer(window, pivot + gpui::point(px(offset as f32), px(0.)));
            assert_snapped(window, DVec2::new(offset, 0.));
        }
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        release_again(window, end);
        item.read_with(window, |item, cx| {
            assert!(
                item.scene_canvas
                    .as_ref()
                    .expect("Canvas")
                    .artboard_draw
                    .is_none()
            );
            assert_eq!(
                item.model
                    .read(cx)
                    .gdd_contents()
                    .expect("working GDD")
                    .history_len(),
                history,
            );
        });
        draft_again(window, start, end);
        release_again(window, end);
        let second_address = item.read_with(window, |item, cx| {
            assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
            let model = item.model.read(cx);
            let graph = &model.working.as_ref().expect("working GDD").inspection;
            assert!(model.gdd_contents().expect("working GDD").history_len() > history);
            let second_address = assert_stack(graph);
            let canvas = item.scene_canvas.as_ref().expect("Canvas after creation");
            let navigation = canvas.navigation.expect("accepted Canvas render");
            assert_eq!(
                navigation.document_to_pane(canvas.target.expect("measured Canvas").logical_size()),
                transform
            );
            let overlap = transform.transform_point2(DVec2::ZERO);
            let pick = canvas
                .artboards
                .iter()
                .filter_map(|board| board.pick_target.as_ref())
                .find(|pick| pick.contains(overlap))
                .expect("pick in overlap");
            assert!(
                pick.matches_node(
                    graph
                        .nodes
                        .iter()
                        .find(|node| node.storage_id() == second_address)
                        .expect("new front")
                )
            );
            second_address
        });
        window.dispatch_action(editor::actions::Undo);
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            let graph = &item
                .model
                .read(cx)
                .working
                .as_ref()
                .expect("working GDD")
                .inspection;
            let root = graph
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("root");
            let (board, _) = root
                .connected_export(&graph.nodes)
                .expect("original export restored");
            assert_eq!(board.storage_id(), address);
            assert!(
                !graph
                    .nodes
                    .iter()
                    .any(|node| node.storage_id() == second_address)
            );
        });
        window.dispatch_action(editor::actions::Redo);
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        let destination = PathBuf::from("/project/drawn-artboard.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let reopened = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen newly drawn Artboard");
        let graph = reopened.graph_inspection().await.expect("saved graph");
        assert_eq!(assert_stack(&graph), second_address);
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_tool_008_2_workspace_moves_the_saved_artboard_on_canvas(
        cx: &mut TestAppContext,
    ) {
        let mut stage = create_gdd_folder(621, 187, "Artboard move".into(), ARCHIVE_BUDGET)
            .await
            .expect("native source");
        let mut addresses = BTreeMap::new();
        for (name, identifier, position) in [
            ("Artboard", "NETWORK:Artboard", [2, 0]),
            ("Merge", "NETWORK:Merge", [-6, 0]),
            (
                "Fill",
                "PROTONODE:graphene_core::vector::FillNode",
                [-12, 0],
            ),
            (
                "Rectangle",
                "PROTONODE:vector_nodes::generator_nodes::RectangleNode",
                [-18, 0],
            ),
        ] {
            stage = stage
                .insert_root_catalog_node_in_copy(identifier, position, ARCHIVE_BUDGET)
                .await
                .expect("pinned catalog node");
            let graph = stage
                .graph_inspection()
                .await
                .expect("root scene addresses");
            let root = graph
                .networks
                .iter()
                .find(|network| network.parent.is_none())
                .expect("root network");
            let node = graph
                .nodes
                .iter()
                .find(|node| {
                    node.network_id() == root.id
                        && !addresses
                            .values()
                            .any(|address| *address == node.storage_id())
                })
                .expect("new root node");
            addresses.insert(name, node.storage_id());
        }
        let nodes = stage.graph_nodes().await.expect("root nodes");
        let root = nodes
            .iter()
            .find(|node| node.can_set_document_artboard())
            .expect("root Artboard");
        let address = root.storage_id();
        stage = stage
            .set_document_artboard_in_copy(root, ARCHIVE_BUDGET)
            .await
            .expect("document output");
        for (target, source) in [
            ("Fill", "Rectangle"),
            ("Merge", "Fill"),
            ("Artboard", "Merge"),
        ] {
            let nodes = stage.graph_nodes().await.expect("pinned scene nodes");
            let find = |name| {
                nodes
                    .iter()
                    .find(|node| addresses.get(name) == Some(&node.storage_id()))
                    .expect("root scene node")
            };
            stage = stage
                .connect_graphic_content_in_copy(find(target), find(source), ARCHIVE_BUDGET)
                .await
                .expect("pinned painted Artboard graph");
        }
        let original = stage
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.update(|_, cx| {
            let bindings =
                settings::KeymapFile::load_asset(crate::DEFAULT_GRAPHITE_KEYMAP_PATH, None, cx)
                    .expect("Graphite keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(px(1160.), px(1050.)));
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/artboard-move-source.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("original source");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the original Artboard")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        let canvas_button = "GRAPHITE-GDD-OPEN-CANVAS";
        let tool = "GRAPHITE-GDD-CANVAS-MOVE-ARTBOARD";
        click(window, canvas_button);
        click(window, tool);
        item.read_with(window, |item, cx| {
            let canvas = item.scene_canvas.as_ref().expect("read-only Canvas");
            assert!(!canvas.move_armed);
            assert!(!item.is_dirty(cx));
        });
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, canvas_button);
        click(window, tool);

        let state = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, cx| {
                assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
                let model = item.model.read(cx);
                let working = model.working.as_ref().expect("working copy");
                let node = working
                    .inspection
                    .nodes
                    .iter()
                    .find(|node| node.storage_id() == address)
                    .expect("same Artboard");
                (
                    node.artboard_geometry().expect("exact source geometry"),
                    model
                        .gdd_contents()
                        .expect("working snapshot")
                        .history_len(),
                )
            })
        };
        let pose = |window: &mut gpui::VisualTestContext| {
            let pane = window
                .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
                .expect("Canvas pane");
            let transform = item.read_with(window, |item, _| {
                let canvas = item.scene_canvas.as_ref().expect("Canvas");
                canvas
                    .navigation
                    .expect("accepted renderer camera")
                    .document_to_pane(canvas.target.expect("measured view").logical_size())
            });
            (pane, transform)
        };
        let pointer = |window: &mut gpui::VisualTestContext, event| {
            window.update(|window, cx| window.dispatch_event(event, cx));
            window.run_until_parked();
            paint(window);
        };
        let down = |position| {
            gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                position,
                button: gpui::MouseButton::Left,
                modifiers: gpui::Modifiers::default(),
                click_count: 1,
                first_mouse: false,
            })
        };
        let drag = |position, shift| {
            gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                position,
                pressed_button: Some(gpui::MouseButton::Left),
                modifiers: gpui::Modifiers {
                    shift,
                    ..Default::default()
                },
            })
        };
        let up = |position, shift| {
            gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                position,
                button: gpui::MouseButton::Left,
                modifiers: gpui::Modifiers {
                    shift,
                    ..Default::default()
                },
                click_count: 1,
            })
        };

        let initial = state(window);
        let (pane, fitted) = pose(window);
        pointer(window, down(pane.origin + gpui::point(px(2.), px(2.))));
        item.read_with(window, |item, _| {
            assert!(
                item.scene_canvas
                    .as_ref()
                    .expect("Canvas")
                    .artboard_drag
                    .is_none()
            )
        });
        pointer(window, up(pane.origin + gpui::point(px(2.), px(2.)), false));
        pointer(window, down(pane.center()));
        window.update(|window, cx| {
            window.dispatch_event(
                drag(pane.center() + gpui::point(px(15.), px(-10.)), false),
                cx,
            );
            window.dispatch_event(
                drag(pane.center() + gpui::point(px(45.), px(-30.)), false),
                cx,
            );
        });
        window.run_until_parked();
        paint(window);
        let draft_image = item.read_with(window, |item, _| {
            let canvas = item.scene_canvas.as_ref().expect("Canvas");
            let gesture = canvas.artboard_drag.as_ref().expect("live draft");
            assert_ne!(gesture.outline_offset, DVec2::ZERO);
            let view = canvas.artboard_view.as_ref().expect("Artboard view");
            assert!(view.draft_error.is_none(), "{:?}", view.draft_error);
            let draft = view
                .draft_frame
                .as_ref()
                .expect("source-rendered live artwork");
            let displacement = fitted
                .inverse()
                .transform_vector2(DVec2::new(45., -30.))
                .round();
            assert_eq!(
                draft.request.placement.expect("move placement").location,
                initial.0.location + displacement
            );
            assert_eq!(gesture.placement, draft.request.placement);
            let SavedSceneFrame::Ready(original) = &canvas.frame else {
                panic!("committed Canvas artwork")
            };
            let sample = |image: &RenderImage, document_position: DVec2| {
                let position = (fitted.transform_point2(document_position)
                    * canvas.target.expect("measured view").scale)
                    .round();
                let size = image.size(0);
                let index = (position.y as usize * size.width.0 as usize + position.x as usize) * 4;
                image
                    .as_bytes(0)
                    .and_then(|bytes| bytes.get(index..index + 4))
                    .expect("sample in Canvas artwork")
                    .to_vec()
            };
            let painted = DVec2::splat(20.);
            assert_eq!(sample(original, painted), [0, 0, 0, 255]);
            assert_eq!(sample(&draft.image, painted + displacement), [0, 0, 0, 255]);
            assert_eq!(sample(&draft.image, painted), [0; 4]);
            draft.image.clone()
        });
        assert!(window.update(|window, _| window.has_image_atlas_entry(&draft_image)));
        assert_eq!(state(window), initial);
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&draft_image)));
        pointer(window, up(pane.center(), false));
        assert_eq!(state(window), initial);
        pointer(window, down(pane.center()));
        pointer(
            window,
            drag(pane.center() + gpui::point(px(25.), px(15.)), false),
        );
        window.simulate_resize(gpui::size(px(1040.), px(1050.)));
        paint(window);
        window.run_until_parked();
        paint(window);
        pointer(
            window,
            up(pane.center() + gpui::point(px(25.), px(15.)), false),
        );
        assert_eq!(state(window), initial);

        let (pane, transform) = pose(window);
        let near_left = transform.transform_point2(DVec2::new(24., 540.));
        let start = pane.origin + gpui::point(px(near_left.x as f32), px(near_left.y as f32));
        let end = gpui::point(pane.origin.x - px(20.), start.y + px(4.));
        assert!(!pane.contains(&end));
        let travel = f64::from(f32::from(end.x - start.x));
        let expected = GddArtboardGeometry {
            location: transform
                .inverse()
                .transform_vector2(DVec2::new(travel, 0.))
                .round(),
            ..initial.0
        };
        pointer(window, down(start));
        pointer(window, drag(end, true));
        assert_eq!(state(window), initial);
        pointer(window, up(end, true));
        assert_eq!(state(window), (expected, initial.1 + 1));
        assert_eq!(pose(window).1, transform);
        for (edit, placement) in [(true, initial.0), (false, expected)] {
            if edit {
                window.dispatch_action(editor::actions::Undo);
            } else {
                window.dispatch_action(editor::actions::Redo);
            }
            window.run_until_parked();
            paint(window);
            window.run_until_parked();
            assert_eq!(state(window).0, placement);
        }
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        let destination = PathBuf::from("/project/artboard-moved.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let reopened = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopen moved document");
        let nodes = reopened.graph_nodes().await.expect("reopened Artboard");
        let artboard = nodes
            .iter()
            .find(|node| node.storage_id() == address)
            .expect("same saved Artboard");
        assert_eq!(artboard.artboard_geometry(), Some(expected));
        assert_eq!(
            filesystem
                .load_bytes(&source)
                .await
                .expect("unchanged source ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_node_282_2_workspace_edits_saved_artboard_geometry(cx: &mut TestAppContext) {
        let stage = create_gdd_folder(621, 187, "Made".into(), ARCHIVE_BUDGET)
            .await
            .expect("native source")
            .insert_root_catalog_node_in_copy("NETWORK:Artboard", [2, 0], ARCHIVE_BUDGET)
            .await
            .expect("pinned catalog Artboard");
        let nodes = stage.graph_nodes().await.expect("working nodes");
        let root = nodes
            .iter()
            .find(|node| node.can_set_document_artboard())
            .expect("root Artboard");
        let stage = stage
            .set_document_artboard_in_copy(root, ARCHIVE_BUDGET)
            .await
            .expect("set document Artboard");
        let original = stage
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("source ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.update(|_, cx| {
            let bindings =
                settings::KeymapFile::load_asset(crate::DEFAULT_GRAPHITE_KEYMAP_PATH, None, cx)
                    .expect("Graphite keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/geometry-artboard.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("original source");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the source Artboard")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        let canvas_frame = |window: &mut gpui::VisualTestContext, color: [u8; 4]| {
            window.run_until_parked();
            paint(window);
            let (target, image) = item.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("open document Canvas");
                assert!(Arc::ptr_eq(&canvas.revision, &item.model.read(cx).revision));
                let SavedSceneFrame::Ready(image) = &canvas.frame else {
                    panic!("the measured Canvas has no frame");
                };
                (canvas.target.expect("measured pane"), image.clone())
            });
            let bounds = window
                .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
                .expect("native Canvas bounds");
            let scale = window.update(|window, _| window.scale_factor());
            assert!(SavedSceneTarget::measure(bounds.size, scale) == Some(target));
            let width = target.resolution.x as usize;
            let height = target.resolution.y as usize;
            let bytes = image.as_bytes(0).expect("native BGRA frame");
            let center = (height / 2 * width + width / 2) * 4;
            assert_eq!(bytes.get(center..center + 4), Some(color.as_slice()));
            assert_eq!(bytes.get(..4), Some([0; 4].as_slice()));
            assert_eq!(image.size(0).width.0, target.resolution.x as i32);
            assert_eq!(image.size(0).height.0, target.resolution.y as i32);
            assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
            (target.resolution, image)
        };
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        let (initial_viewport, initial_frame) = canvas_frame(window, [255; 4]);
        assert!(initial_viewport.min_element() > LAYER_PREVIEW_SIZE);
        item.read_with(window, |item, cx| {
            assert!(matches!(item.node_copy, NodeCopyPanel::Closed));
            assert!(!item.is_dirty(cx));
        });
        for visible in [false, true] {
            click(window, "GRAPHITE-GDD-CANVAS-OVERLAYS");
            click(window, "MENU_ITEM-Artboard Name");
            item.read_with(window, |item, cx| {
                let canvas = item.scene_canvas.as_ref().expect("Artboard Canvas");
                let visibility = canvas
                    .overlay_visibility(item.model.read(cx))
                    .expect("overlay settings");
                assert_eq!(visibility.artboard_name(), visible);
                assert!(visibility.transform_cage());
                let [artboard] = canvas.artboards.as_slice() else {
                    panic!("one Artboard on Canvas");
                };
                assert!(artboard.overlay.is_some());
                let SavedSceneFrame::Ready(image) = &canvas.frame else {
                    panic!("the artwork frame is retained during overlay changes");
                };
                assert!(Arc::ptr_eq(image, &initial_frame));
                assert!(!item.is_dirty(cx));
            });
        }
        window.simulate_resize(gpui::size(px(1030.), px(1400.)));
        paint(window);
        let (resized_viewport, resized_frame) = canvas_frame(window, [255; 4]);
        assert!(resized_viewport.x < initial_viewport.x);
        assert!(!Arc::ptr_eq(&initial_frame, &resized_frame));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&initial_frame)));
        let pane = window
            .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
            .expect("visible Canvas");
        let cursor = pane.origin + gpui::point(pane.size.width * 0.35, pane.size.height * 0.42);
        let coordinate = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, _| {
                let canvas = item.scene_canvas.as_ref().expect("open Canvas");
                let target = canvas.target.expect("current viewport");
                canvas
                    .navigation
                    .expect("current evaluated Artboard camera")
                    .document_to_pane(target.logical_size())
            })
        };
        let send_scroll = |window: &mut gpui::VisualTestContext, delta, modifiers| {
            window.update(|window, cx| {
                window.dispatch_event(
                    gpui::PlatformInput::ScrollWheel(gpui::ScrollWheelEvent {
                        position: cursor,
                        delta: gpui::ScrollDelta::Pixels(delta),
                        modifiers,
                        ..Default::default()
                    }),
                    cx,
                );
            });
            window.run_until_parked();
            paint(window);
        };
        let fitted = coordinate(window);
        send_scroll(
            window,
            gpui::point(px(80.), px(-20.)),
            gpui::Modifiers::default(),
        );
        let panned = coordinate(window);
        assert_eq!(panned.matrix2, fitted.matrix2);
        let expected_pan = DVec2::new(80., -20.) * graphite_editor::consts::VIEWPORT_SCROLL_RATE;
        assert!(
            (panned.translation - fitted.translation - expected_pan)
                .abs()
                .max_element()
                <= 1.
        );
        let (_, panned_frame) = canvas_frame(window, [255; 4]);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&resized_frame)));
        let cursor_in_pane = cursor - pane.origin;
        let anchor = DVec2::new(
            f64::from(f32::from(cursor_in_pane.x)),
            f64::from(f32::from(cursor_in_pane.y)),
        );
        let addressed_point = panned.inverse().transform_point2(anchor);
        send_scroll(
            window,
            gpui::point(px(0.), px(120.)),
            gpui::Modifiers {
                control: true,
                ..Default::default()
            },
        );
        let zoomed = coordinate(window);
        assert!(zoomed.matrix2.x_axis.x > panned.matrix2.x_axis.x);
        assert!(
            (zoomed.transform_point2(addressed_point) - anchor)
                .abs()
                .max_element()
                <= 2.
        );
        let (_, zoomed_frame) = canvas_frame(window, [255; 4]);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&panned_frame)));
        send_scroll(
            window,
            gpui::point(px(120.), px(0.)),
            gpui::Modifiers {
                control: true,
                ..Default::default()
            },
        );
        assert_eq!(coordinate(window), zoomed);
        assert!(window.update(|window, _| window.has_image_atlas_entry(&zoomed_frame)));
        click(window, "GRAPHITE-GDD-CANVAS-FIT");
        window.run_until_parked();
        assert_eq!(coordinate(window), fitted);
        let dispatch_pointer = |window: &mut gpui::VisualTestContext, event| {
            window.update(|window, cx| {
                window.dispatch_event(event, cx);
            });
            window.run_until_parked();
            paint(window);
        };
        let press = |position, button| {
            gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                position,
                button,
                modifiers: gpui::Modifiers::default(),
                click_count: 1,
                first_mouse: false,
            })
        };
        let drag = |position| {
            gpui::PlatformInput::MouseMove(gpui::MouseMoveEvent {
                position,
                pressed_button: Some(gpui::MouseButton::Middle),
                modifiers: gpui::Modifiers::default(),
            })
        };
        let outside = gpui::point(cursor.x + px(25.), pane.origin.y - px(8.));
        assert!(!pane.contains(&outside));
        dispatch_pointer(window, press(cursor, gpui::MouseButton::Middle));
        dispatch_pointer(window, drag(outside));
        let dragged = coordinate(window);
        let displacement = DVec2::new(
            f64::from(f32::from(outside.x - cursor.x)),
            f64::from(f32::from(outside.y - cursor.y)),
        );
        assert!(
            (dragged.translation - fitted.translation - displacement)
                .abs()
                .max_element()
                <= 1.
        );
        dispatch_pointer(
            window,
            gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                position: outside,
                button: gpui::MouseButton::Middle,
                modifiers: gpui::Modifiers::default(),
                click_count: 1,
            }),
        );
        item.read_with(window, |item, _| {
            assert!(item.scene_canvas.as_ref().expect("Canvas").pan.is_none());
        });
        dispatch_pointer(window, drag(cursor));
        assert_eq!(coordinate(window), dragged);
        for abort_with_escape in [true, false] {
            dispatch_pointer(window, press(cursor, gpui::MouseButton::Middle));
            dispatch_pointer(window, drag(cursor + gpui::point(px(-60.), px(30.))));
            assert_ne!(coordinate(window), dragged);
            if abort_with_escape {
                window.simulate_keystrokes("escape");
                window.run_until_parked();
                paint(window);
            } else {
                dispatch_pointer(window, press(outside, gpui::MouseButton::Right));
            }
            assert_eq!(coordinate(window), dragged);
            item.read_with(window, |item, _| {
                assert!(item.scene_canvas.as_ref().expect("Canvas").pan.is_none());
            });
        }
        dispatch_pointer(window, press(cursor, gpui::MouseButton::Middle));
        dispatch_pointer(window, drag(cursor + gpui::point(px(-60.), px(30.))));
        assert_ne!(coordinate(window), dragged);
        for width in [1070., 1030.] {
            window.simulate_resize(gpui::size(px(width), px(1400.)));
            paint(window);
            window.run_until_parked();
            paint(window);
            item.read_with(window, |item, _| {
                assert!(item.scene_canvas.as_ref().expect("Canvas").pan.is_none());
            });
        }
        assert_eq!(coordinate(window), dragged);
        click(window, "GRAPHITE-GDD-CANVAS-FIT");
        window.run_until_parked();
        assert_eq!(coordinate(window), fitted);
        item.read_with(window, |item, cx| assert!(!item.is_dirty(cx)));
        let pane = window
            .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
            .expect("fitted Canvas");
        window.simulate_click(pane.center(), gpui::Modifiers::default());
        item.read_with(window, |item, _| assert!(item.scene_canvas.is_some()));
        click(window, "GRAPHITE-GDD-CANVAS-PICK");
        window.simulate_click(
            pane.origin + gpui::point(px(5.), px(5.)),
            gpui::Modifiers::default(),
        );
        item.read_with(window, |item, _| {
            assert!(
                item.scene_canvas
                    .as_ref()
                    .expect("miss stays in Canvas")
                    .pick_armed
            );
            assert!(matches!(item.node_copy, NodeCopyPanel::Closed));
        });
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::ScrollWheel(gpui::ScrollWheelEvent {
                    position: pane.center(),
                    delta: gpui::ScrollDelta::Pixels(gpui::point(px(60.), px(-20.))),
                    ..Default::default()
                }),
                cx,
            );
            assert!(
                item.read(cx)
                    .scene_canvas
                    .as_ref()
                    .expect("Canvas after pan request")
                    .artboards
                    .is_empty()
            );
            window.dispatch_event(press(pane.center(), gpui::MouseButton::Left), cx);
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position: pane.center(),
                    button: gpui::MouseButton::Left,
                    modifiers: gpui::Modifiers::default(),
                    click_count: 1,
                }),
                cx,
            );
            assert!(item.read(cx).scene_canvas.is_some());
        });
        window.run_until_parked();
        paint(window);
        let pane = window
            .debug_bounds("GRAPHITE-GDD-DOCUMENT-CANVAS")
            .expect("Artboard Canvas after pan");
        let point = coordinate(window).transform_point2(DVec2::new(960., 540.));
        window.simulate_click(
            pane.origin + gpui::point(px(point.x as f32), px(point.y as f32)),
            gpui::Modifiers::default(),
        );
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, cx| {
            assert!(item.scene_canvas.is_none());
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("picked Artboard Inspector");
            };
            assert_eq!(
                form.selected_node().map(GddGraphNode::storage_id),
                Some(root.storage_id())
            );
            assert!(!item.is_dirty(cx));
        });
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&resized_frame)));
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        let read_geometry = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Artboard geometry");
                };
                let node = form.selected_node().expect("selected Artboard");
                (
                    node.storage_id(),
                    node.artboard_geometry().expect("native geometry fields"),
                )
            })
        };
        let (address, initial) = read_geometry(window);
        assert_eq!(initial.location, DVec2::ZERO);
        assert_eq!(initial.dimensions, DVec2::new(1920., 1080.));
        let set_field = |window: &mut gpui::VisualTestContext, selector, value: &str| {
            click(window, selector);
            window.dispatch_action(editor::actions::SelectAll);
            window.simulate_input(value);
            window.run_until_parked();
            paint(window);
        };
        let apply_geometry = "GRAPHITE-GDD-APPLY-ARTBOARD-GEOMETRY";
        set_field(window, "GRAPHITE-GDD-ARTBOARD-LOCATION", "invalid");
        click(window, apply_geometry);
        item.read_with(window, |item, cx| {
            assert!(!item.is_dirty(cx));
            assert!(matches!(item.copy_feedback, Some((_, Color::Error))));
        });
        assert_eq!(read_geometry(window), (address, initial));
        set_field(window, "GRAPHITE-GDD-ARTBOARD-LOCATION", "85.5, -18.5");
        set_field(window, "GRAPHITE-GDD-ARTBOARD-DIMENSIONS", "-64.5, 31.5");
        click(window, apply_geometry);
        let expected = GddArtboardGeometry {
            location: DVec2::new(86., -19.),
            dimensions: DVec2::new(-65., 32.),
        };
        assert_eq!(read_geometry(window), (address, expected));
        click(window, "GRAPHITE-GDD-UNDO");
        assert_eq!(read_geometry(window), (address, initial));
        click(window, "GRAPHITE-GDD-REDO");
        assert_eq!(read_geometry(window), (address, expected));
        let read_background = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Artboard properties");
                };
                let node = form.selected_node().expect("selected Artboard");
                node.artboard_background().expect("native Background")
            })
        };
        let white = read_background(window);
        assert_eq!([white.r(), white.g(), white.b(), white.a()], [1.; 4]);
        set_field(window, "GRAPHITE-GDD-ARTBOARD-BACKGROUND", "1, 0, 0, 1");
        click(window, "GRAPHITE-GDD-APPLY-ARTBOARD-BACKGROUND");
        let red = read_background(window);
        assert_eq!([red.r(), red.g(), red.b(), red.a()], [1., 0., 0., 1.]);
        click(window, "GRAPHITE-GDD-UNDO");
        assert_eq!(read_background(window), white);
        assert_eq!(read_geometry(window), (address, expected));
        click(window, "GRAPHITE-GDD-REDO");
        assert_eq!(read_background(window), red);
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        let (_, red_frame) = canvas_frame(window, [0, 0, 255, 255]);
        window.dispatch_action(editor::actions::Undo);
        let (_, white_frame) = canvas_frame(window, [255; 4]);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&red_frame)));
        window.dispatch_action(editor::actions::Redo);
        canvas_frame(window, [0, 0, 255, 255]);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&white_frame)));
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        let exported = PathBuf::from("/project/changed-artboard-bounds.svg");
        click(window, "GRAPHITE-GDD-EXPORT-ARTBOARD-SVG");
        window.simulate_new_path_selection(|_| Some(exported.clone()));
        window.run_until_parked();
        paint(window);
        let bytes = filesystem
            .load_bytes(&exported)
            .await
            .expect("exported document SVG");
        let svg = std::str::from_utf8(&bytes).expect("SVG text");
        assert!(svg.contains(r#"viewBox="0 0 65 32" width="65" height="32""#));
        assert!(svg.contains(r##"fill="#ff0000""##));
        assert!(!svg.contains(graphite_editor::consts::COLOR_OVERLAY_GRAY));
        assert!(!svg.contains(graphite_editor::consts::COLOR_OVERLAY_BLUE));
        let saved = PathBuf::from("/project/changed-artboard-bounds.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(saved.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &saved)
            .await
            .expect("reopened geometry");
        let nodes = saved.graph_nodes().await.expect("saved GDD nodes");
        let board = nodes
            .iter()
            .find(|node| node.storage_id() == address)
            .expect("same Artboard address");
        assert_eq!(board.artboard_geometry(), Some(expected));
        assert_eq!(board.artboard_background(), Some(red));
        assert!(matches!(
            board.layer_preview(),
            Some(GddGraphLayerPreview::Artboard { origin, size })
                if origin == DVec2::new(21., -19.) && size == DVec2::new(65., 32.)
        ));
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("original ZIP"),
            original
        );
    }

    #[gpui::test]
    async fn at_gra_node_282_3_workspace_builds_and_exports_the_default_artboard(
        cx: &mut TestAppContext,
    ) {
        verify_painted_artboard(
            cx,
            "Rectangle",
            "GRAPHITE-NODE-CATALOG-ADD-Rectangle",
            [0, 0, 0, 255],
            false,
        )
        .await;
    }

    #[gpui::test]
    async fn at_gra_node_262_3_workspace_paints_and_reopens_an_ellipse(cx: &mut TestAppContext) {
        verify_painted_artboard(
            cx,
            "Ellipse",
            "GRAPHITE-NODE-CATALOG-ADD-Ellipse",
            [255; 4],
            false,
        )
        .await;
    }

    #[gpui::test]
    async fn at_gra_node_262_3_workspace_edits_and_reopens_ellipse_radii(cx: &mut TestAppContext) {
        verify_painted_artboard(
            cx,
            "Ellipse",
            "GRAPHITE-NODE-CATALOG-ADD-Ellipse",
            [0, 0, 0, 255],
            true,
        )
        .await;
    }

    #[gpui::test]
    async fn at_gra_node_262_3_workspace_authors_ellipse_radius_wires(cx: &mut TestAppContext) {
        let catalog = GraphiteNodeCatalog::load().expect("pinned catalog");
        let mut initial =
            create_gdd_folder(621, 187, "Ellipse radius wires".into(), ARCHIVE_BUDGET)
                .await
                .expect("empty source folder");
        for (name, position) in [
            ("Ellipse", [4, 0]),
            ("Absolute Value", [-10, -5]),
            ("Sign", [-12, 5]),
        ] {
            let definition = catalog
                .search(name)
                .into_iter()
                .flat_map(|group| group.nodes)
                .find(|entry| entry.name == name)
                .expect("pinned node definition");
            initial = initial
                .insert_root_catalog_node_in_copy(&definition.identifier, position, ARCHIVE_BUDGET)
                .await
                .expect("insert pinned node");
        }
        let original = initial
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("original ZIP");
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/ellipse-wire-source.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("source ZIP");
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the original archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, "GRAPHITE-GDD-NODE-COPY");
        let find = |window: &gpui::VisualTestContext, name: &str| {
            item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Inspector")
                };
                form.nodes
                    .iter()
                    .enumerate()
                    .find(|(_, node)| node.display_name() == name)
                    .map(|(index, node)| (index, node.storage_id()))
                    .expect("saved root node")
            })
        };
        let (ellipse_index, ellipse_id) = find(window, "Ellipse");
        let (_, first) = find(window, "Absolute Value");
        let (_, second) = find(window, "Sign");
        let node_selectors = [
            "GRAPHITE-GDD-NODE-COPY-ITEM-0",
            "GRAPHITE-GDD-NODE-COPY-ITEM-1",
            "GRAPHITE-GDD-NODE-COPY-ITEM-2",
        ];
        click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
        click(
            window,
            node_selectors.get(ellipse_index).expect("Ellipse selector"),
        );
        let verify = |window: &gpui::VisualTestContext, expected| {
            item.read_with(window, |item, _| {
                assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Ellipse")
                };
                let node = form.selected_node().expect("selected Ellipse");
                assert_eq!(node.storage_id(), ellipse_id);
                let radii = [GddEllipseAxis::X, GddEllipseAxis::Y].map(|axis| {
                    let port = node
                        .ellipse_radius_port(axis)
                        .expect("original radius port");
                    let source = match port.source {
                        Some(GddGraphPortSource::Node {
                            storage_id,
                            output_index: 0,
                        }) => Some(storage_id),
                        None => None,
                        _ => panic!("unexpected radius source"),
                    };
                    (source, port.exposed)
                });
                assert_eq!(radii, expected);
            });
        };
        let connect = |window: &mut gpui::VisualTestContext, axis, source: &str| {
            let (index, _) = find(window, source);
            let (menu, sources) = match axis {
                GddEllipseAxis::X => (
                    "GRAPHITE-GDD-CONNECT-ELLIPSE-RADIUS-X",
                    [
                        "GRAPHITE-GDD-ELLIPSE-RADIUS-X-SOURCE-0",
                        "GRAPHITE-GDD-ELLIPSE-RADIUS-X-SOURCE-1",
                        "GRAPHITE-GDD-ELLIPSE-RADIUS-X-SOURCE-2",
                    ],
                ),
                GddEllipseAxis::Y => (
                    "GRAPHITE-GDD-CONNECT-ELLIPSE-RADIUS-Y",
                    [
                        "GRAPHITE-GDD-ELLIPSE-RADIUS-Y-SOURCE-0",
                        "GRAPHITE-GDD-ELLIPSE-RADIUS-Y-SOURCE-1",
                        "GRAPHITE-GDD-ELLIPSE-RADIUS-Y-SOURCE-2",
                    ],
                ),
            };
            click(window, menu);
            click(window, sources.get(index).expect("radius source selector"));
        };
        verify(window, [(None, false), (None, false)]);
        connect(window, GddEllipseAxis::X, "Absolute Value");
        verify(window, [(Some(first), true), (None, false)]);
        connect(window, GddEllipseAxis::Y, "Sign");
        verify(window, [(Some(first), true), (Some(second), true)]);
        connect(window, GddEllipseAxis::X, "Sign");
        verify(window, [(Some(second), true), (Some(second), true)]);
        click(window, "GRAPHITE-GDD-UNDO");
        verify(window, [(Some(first), true), (Some(second), true)]);
        click(window, "GRAPHITE-GDD-REDO");
        verify(window, [(Some(second), true), (Some(second), true)]);
        click(window, "GRAPHITE-GDD-DISCONNECT-ELLIPSE-RADIUS-Y");
        verify(window, [(Some(second), true), (None, true)]);
        click(window, "GRAPHITE-GDD-UNDO");
        verify(window, [(Some(second), true), (Some(second), true)]);
        click(window, "GRAPHITE-GDD-REDO");
        verify(window, [(Some(second), true), (None, true)]);
        click(window, "GRAPHITE-GDD-DISCONNECT-ELLIPSE-RADIUS-X");
        verify(window, [(None, true), (None, true)]);
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready Ellipse");
            };
            let ellipse = form.selected_node().expect("selected Ellipse");
            for axis in [GddEllipseAxis::X, GddEllipseAxis::Y] {
                let port = ellipse.ellipse_radius_port(axis).expect("exposed fallback");
                assert_eq!(port.stored_type, Some(graphite_graph_craft::list!(f64)));
            }
        });
        click(window, "GRAPHITE-GDD-UNDO");
        verify(window, [(Some(second), true), (None, true)]);

        let destination = PathBuf::from("/project/ellipse-wire-result.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let reopened = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("saved graph folder");
        let nodes = reopened.graph_nodes().await.expect("reopened graph");
        let ellipse = nodes
            .iter()
            .find(|node| node.storage_id() == ellipse_id)
            .expect("same Ellipse");
        let horizontal = ellipse
            .ellipse_radius_port(GddEllipseAxis::X)
            .expect("horizontal input");
        assert!(
            matches!(horizontal.source, Some(GddGraphPortSource::Node { storage_id, output_index: 0 }) if storage_id == second)
        );
        let vertical = ellipse
            .ellipse_radius_port(GddEllipseAxis::Y)
            .expect("vertical input");
        assert!(vertical.source.is_none() && vertical.exposed);
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("original ZIP"),
            original
        );
    }

    async fn verify_painted_artboard(
        cx: &mut TestAppContext,
        shape: &'static str,
        add_shape: &'static str,
        expected_lower_sample: [u8; 4],
        edit_ellipse: bool,
    ) {
        use graphics_core::formats::gdd::GddGraphPortSource;

        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/default-artboard-source.gdd");
        window.dispatch_action(NewGraphiteFolder);
        window.run_until_parked();
        window.simulate_new_path_selection(|_| Some(source.clone()));
        window.run_until_parked();
        let item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteDocumentItem>(cx)
            })
            .expect("new native working document");
        let original = read_graphite_folder(filesystem.as_ref(), &source)
            .await
            .expect("new folder")
            .folder_files()
            .expect("original folder entries");

        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        assert!(item.read_with(window, |item, cx| item.model.read(cx).working.is_some()));
        click(window, "GRAPHITE-NODE-CATALOG");
        let mut addresses = BTreeMap::new();
        for (name, position, add) in [
            ("Artboard", "2, 0", "GRAPHITE-NODE-CATALOG-ADD-Artboard"),
            ("Merge", "-6, 0", "GRAPHITE-NODE-CATALOG-ADD-Merge"),
            ("Fill", "-12, 0", "GRAPHITE-NODE-CATALOG-ADD-Fill"),
            (shape, "-18, 0", add_shape),
        ] {
            click(window, "GRAPHITE-NODE-CATALOG-SEARCH");
            window.dispatch_action(editor::actions::SelectAll);
            window.simulate_input(name);
            window.run_until_parked();
            paint(window);
            click(window, "GRAPHITE-NODE-CATALOG-POSITION");
            window.dispatch_action(editor::actions::SelectAll);
            window.simulate_input(position);
            window.run_until_parked();
            paint(window);
            click(window, add);
            let address = item.read_with(window, |item, cx| {
                assert!(
                    item.copy_feedback.is_none(),
                    "{name}: {:?}",
                    item.copy_feedback
                );
                let working = item.model.read(cx).working.as_ref().expect("working copy");
                let root = working
                    .inspection
                    .networks
                    .iter()
                    .find(|network| network.parent.is_none())
                    .expect("native root");
                let mut new_nodes = working.inspection.nodes.iter().filter(|node| {
                    node.network_id() == root.id
                        && !addresses.values().any(|&id| id == node.storage_id())
                });
                let added = new_nodes.next().expect("one added root node").storage_id();
                assert!(new_nodes.next().is_none());
                added
            });
            addresses.insert(name, address);
        }
        click(window, "GRAPHITE-NODE-CATALOG");
        click(window, "GRAPHITE-GDD-NODE-COPY");
        let root_index = |window: &gpui::VisualTestContext, name: &str| {
            item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("open working graph");
                };
                let root = form
                    .networks
                    .iter()
                    .find(|network| network.parent.is_none())
                    .expect("native root");
                form.nodes
                    .iter()
                    .position(|node| {
                        node.network_id() == root.id
                            && Some(&node.storage_id()) == addresses.get(name)
                    })
                    .expect("inserted root definition")
            })
        };
        let root_buttons = [
            "GRAPHITE-GDD-NODE-COPY-ITEM-0",
            "GRAPHITE-GDD-NODE-COPY-ITEM-1",
            "GRAPHITE-GDD-NODE-COPY-ITEM-2",
            "GRAPHITE-GDD-NODE-COPY-ITEM-3",
        ];
        let source_buttons = [
            "GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-0",
            "GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-1",
            "GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-2",
            "GRAPHITE-GDD-GRAPHIC-CONTENT-SOURCE-3",
        ];
        let select = |window: &mut gpui::VisualTestContext, name| {
            let index = root_index(window, name);
            click(window, "GRAPHITE-GDD-NODE-COPY-CHOOSE");
            click(window, root_buttons.get(index).expect("root node selector"));
        };
        select(window, "Artboard");
        click(window, "GRAPHITE-GDD-SET-DOCUMENT-ARTBOARD");
        for (target, source) in [("Fill", shape), ("Merge", "Fill"), ("Artboard", "Merge")] {
            select(window, target);
            let index = root_index(window, source);
            click(window, "GRAPHITE-GDD-CONNECT-GRAPHIC-CONTENT");
            click(
                window,
                source_buttons
                    .get(index)
                    .expect("source connector selector"),
            );
            item.read_with(window, |item, _| {
                assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback)
            });
        }
        if edit_ellipse {
            select(window, shape);
            verify_ellipse_radius_edits(&item, window);
            select(window, "Artboard");
        }
        let preview = "GRAPHITE-GDD-PREVIEW-ARTBOARD";
        click(window, preview);
        let painting = [0, 0, 0, 255];
        let background = [255; 4];
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("saved Artboard selection");
            };
            let NodeOutputPreview::ReadyImage(image) = &form.output_preview else {
                panic!(
                    "source Artboard preview: {:?}",
                    form.output_preview.feedback()
                );
            };
            let pixels = image.as_bytes(0).expect("preview pixels");
            let sample = |x, y| {
                let offset = (y * LAYER_PREVIEW_SIZE as usize + x) * 4;
                pixels
                    .get(offset..offset + 4)
                    .expect("pixel within Artboard preview")
                    .to_vec()
            };
            assert_eq!(sample(8, 8), [0, 0, 0, 0]);
            let pixel = sample(1, 29);
            assert!(matches!(pixel.as_slice(), [red, green, blue, 255] if red == green && green == blue && *red < 255));
            assert_eq!(sample(64, 64), background);
        });
        click(window, preview);
        click(window, "GRAPHITE-GDD-OPEN-CANVAS");
        window.run_until_parked();
        paint(window);
        item.read_with(window, |item, _| {
            let Some(SavedSceneCanvas {
                frame: SavedSceneFrame::Ready(image),
                ..
            }) = &item.scene_canvas
            else {
                panic!("fitted document Canvas");
            };
            let size = image.size(0);
            let width = size.width.0 as usize;
            let height = size.height.0 as usize;
            let pixels = image.as_bytes(0).expect("Canvas pixels");
            let sample = |x, y| {
                let offset = (y * width + x) * 4;
                pixels.get(offset..offset + 4).expect("pixel in Canvas")
            };
            let left = (0..width)
                .find(|&column| sample(column, height / 2)[3] != 0)
                .expect("left Artboard edge");
            let right = (0..width)
                .rfind(|&column| sample(column, height / 2)[3] != 0)
                .expect("right Artboard edge");
            let top = (0..height)
                .find(|&row| sample(width / 2, row)[3] != 0)
                .expect("top Artboard edge");
            let bottom = (0..height)
                .rfind(|&row| sample(width / 2, row)[3] != 0)
                .expect("bottom Artboard edge");
            let canvas_scale = (right - left + 1) as f64 / 1920.;
            assert!(((bottom - top + 1) as f64 - canvas_scale * 1080.).abs() <= 2.);
            let inset = (20. * canvas_scale).round() as usize;
            assert_eq!(sample(left + inset, top + inset), painting.as_slice());
            let below = (35. * canvas_scale).round() as usize;
            assert_eq!(
                sample(left + inset, top + below),
                expected_lower_sample.as_slice()
            );
            if edit_ellipse {
                let right = (60. * canvas_scale).round() as usize;
                assert_eq!(sample(left + right, top + inset), painting.as_slice());
            }
            assert_eq!(sample(width / 2, height / 2), background.as_slice());
            assert_eq!(sample(0, 0), [0; 4]);
        });
        click(window, "GRAPHITE-GDD-CANVAS-BACK");
        let svg_path = PathBuf::from("/project/default-artboard.svg");
        click(window, "GRAPHITE-GDD-EXPORT-ARTBOARD-SVG");
        window.simulate_new_path_selection(|_| Some(svg_path.clone()));
        window.run_until_parked();
        paint(window);
        let svg = filesystem
            .load_bytes(&svg_path)
            .await
            .expect("exported native SVG");
        let text = std::str::from_utf8(&svg).expect("SVG text");
        assert!(text.contains(r#"viewBox="0 0 1920 1080" width="1920" height="1080""#));
        assert!(text.contains("<clipPath"));
        let renderer = window.update(|_, cx| cx.svg_renderer());
        let exported = crate::render::prepare_svg_bytes_at_target(
            &svg,
            1920,
            1080,
            1.,
            &renderer,
            1920 * 1080 * 4,
        )
        .expect("rasterize source export");
        let pixels = exported.as_bytes(0).expect("export pixels");
        for (x, y, expected) in [(20, 20, painting), (960, 540, background)] {
            let offset = (y * 1920 + x) * 4;
            assert_eq!(pixels.get(offset..offset + 4), Some(expected.as_slice()));
        }
        let offset = (35 * 1920 + 20) * 4;
        assert_eq!(
            pixels.get(offset..offset + 4),
            Some(expected_lower_sample.as_slice())
        );
        if edit_ellipse {
            let offset = (20 * 1920 + 60) * 4;
            assert_eq!(pixels.get(offset..offset + 4), Some(painting.as_slice()));
        }
        let published = PathBuf::from("/project/default-artboard-result.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(published.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &published)
            .await
            .expect("reopen native GDD");
        let graph = saved.graph_inspection().await.expect("persisted graph");
        let root = graph
            .networks
            .iter()
            .find(|network| network.parent.is_none())
            .expect("root");
        let find = |name| {
            graph
                .nodes
                .iter()
                .find(|node| {
                    node.network_id() == root.id && Some(&node.storage_id()) == addresses.get(name)
                })
                .expect("original catalog node")
        };
        let artboard = find("Artboard");
        assert!(matches!(
            artboard.layer_preview(),
            Some(GddGraphLayerPreview::Artboard { origin, size })
                if origin == DVec2::ZERO && size == DVec2::new(1920.,1080.)
        ));
        if edit_ellipse {
            assert_eq!(find(shape).ellipse_radii(), Some([72.5, 52.5]));
        }
        for (target, index, source) in [
            ("Artboard", 1, "Merge"),
            ("Merge", 1, "Fill"),
            ("Fill", 0, shape),
        ] {
            assert!(matches!(
                find(target).input_ports().get(index).and_then(|port| port.source.as_ref()),
                Some(GddGraphPortSource::Node { storage_id, output_index: 0 }) if *storage_id == find(source).storage_id()
            ));
        }
        assert_eq!(
            read_graphite_folder(filesystem.as_ref(), &source)
                .await
                .expect("retained input folder")
                .folder_files()
                .expect("original entries"),
            original
        );
    }

    fn verify_ellipse_radius_edits(
        item: &Entity<GraphiteDocumentItem>,
        window: &mut gpui::VisualTestContext,
    ) {
        let radii = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, _| {
                let NodeCopyPanel::Ready(form) = &item.node_copy else {
                    panic!("ready Ellipse Inspector");
                };
                form.selected_node().and_then(GddGraphNode::ellipse_radii)
            })
        };
        let apply_radius = |window: &mut gpui::VisualTestContext, field, apply, value: &str| {
            click(window, field);
            window.dispatch_action(editor::actions::SelectAll);
            window.simulate_input(value);
            window.run_until_parked();
            paint(window);
            click(window, apply);
        };
        let radius_x = "GRAPHITE-GDD-ELLIPSE-RADIUS-X";
        let apply_x = "GRAPHITE-GDD-APPLY-ELLIPSE-RADIUS-X";
        let radius_y = "GRAPHITE-GDD-ELLIPSE-RADIUS-Y";
        let apply_y = "GRAPHITE-GDD-APPLY-ELLIPSE-RADIUS-Y";
        assert_eq!(radii(window), Some([50., 25.]));
        apply_radius(window, radius_x, apply_x, "72.5");
        assert_eq!(radii(window), Some([72.5, 25.]));
        let revision = item.read_with(window, |item, cx| item.model.read(cx).revision.clone());
        for invalid in ["invalid", "1.7976931348623157e308"] {
            apply_radius(window, radius_y, apply_y, invalid);
            item.read_with(window, |item, cx| {
                assert!(Arc::ptr_eq(&revision, &item.model.read(cx).revision));
                assert!(
                    matches!(item.copy_feedback.as_ref(), Some((message, Color::Error))
                    if message.as_ref() == GddArchiveError::InvalidEllipseRadius.to_string())
                );
            });
            assert_eq!(radii(window), Some([72.5, 25.]));
        }
        apply_radius(window, radius_y, apply_y, "52.5");
        assert_eq!(radii(window), Some([72.5, 52.5]));
        click(window, "GRAPHITE-GDD-UNDO");
        assert_eq!(radii(window), Some([72.5, 25.]));
        click(window, "GRAPHITE-GDD-UNDO");
        assert_eq!(radii(window), Some([50., 25.]));
        click(window, "GRAPHITE-GDD-REDO");
        assert_eq!(radii(window), Some([72.5, 25.]));
        click(window, "GRAPHITE-GDD-REDO");
        assert_eq!(radii(window), Some([72.5, 52.5]));
    }

    #[gpui::test]
    async fn at_gra_graph_005_2_workspace_adds_catalog_layers_to_the_open_working_copy(
        cx: &mut TestAppContext,
    ) {
        let blank = create_gdd_folder(621, 187, "Made".into(), ARCHIVE_BUDGET)
            .await
            .expect("blank source folder");
        let original = blank
            .export_archive(ArchiveFormat::Zip, ARCHIVE_BUDGET.max_archive_bytes)
            .expect("blank original archive");
        let (workspace, window) = native_workspace(cx).await;
        window.simulate_resize(gpui::size(px(1160.), px(1400.)));
        window.run_until_parked();
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/catalog-working-source.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("source ZIP");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open the original archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");

        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG");
        click(window, "GRAPHITE-NODE-CATALOG-SEARCH");
        window.simulate_input("Artboard");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-GDD-EDIT-WORKING-COPY");
        click(window, "GRAPHITE-NODE-CATALOG-POSITION");
        window.dispatch_action(editor::actions::SelectAll);
        window.simulate_input("12, -4");
        window.run_until_parked();
        paint(window);
        click(window, "GRAPHITE-NODE-CATALOG-ADD-Artboard");
        item.read_with(window, |item, _| {
            assert!(item.copy_feedback.is_none(), "{:?}", item.copy_feedback);
        });
        let inspect_working = |window: &gpui::VisualTestContext| {
            item.read_with(window, |item, cx| {
                let model = item.model.read(cx);
                let working = model.working.as_ref().expect("active working copy");
                let root = working
                    .inspection
                    .networks
                    .iter()
                    .find(|network| network.parent.is_none())
                    .expect("document network");
                let layer = working
                    .inspection
                    .nodes
                    .iter()
                    .find(|node| node.network_id() == root.id);
                layer.map(GddGraphNode::storage_id)
            })
        };
        let address = inspect_working(window).expect("inserted on the currently open document");
        assert!(item.read_with(window, |item, cx| item.is_dirty(cx)));
        click(window, "GRAPHITE-GDD-UNDO");
        assert!(inspect_working(window).is_none());
        click(window, "GRAPHITE-GDD-REDO");
        assert_eq!(inspect_working(window), Some(address));
        click(window, "GRAPHITE-NODE-CATALOG");
        let destination = PathBuf::from("/project/catalog-working-result.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("published working graph");
        let graph = saved.graph_inspection().await.expect("reopened Artboard");
        let layer = graph
            .nodes
            .iter()
            .find(|node| node.storage_id() == address)
            .expect("same stored layer address");
        assert!(layer.can_set_document_artboard());
        let root = graph
            .networks
            .iter()
            .find(|network| network.parent.is_none())
            .expect("root network");
        let geometry = saved.graph_geometry(root).await.expect("native layout");
        let placed = geometry
            .nodes
            .iter()
            .find(|node| node.storage_id == address)
            .expect("placed layer");
        let grid = f64::from(graphite_editor::consts::GRID_SIZE);
        assert!(placed.is_layer);
        // The pinned drawing bounds include the exposed Content slot's chain gutter.
        assert_eq!(placed.bounds[0], [11. * grid, -4. * grid]);
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("input ZIP"),
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
    async fn at_gra_node_280_2_workspace_inserts_custom_node_in_the_working_scope(
        cx: &mut TestAppContext,
    ) {
        let (original, _) = nested_graph_archive_with_sibling(true).await;
        let (workspace, window) = native_workspace(cx).await;
        let filesystem = workspace.read_with(window, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        let source = PathBuf::from("/project/working-custom-node.gdd.zip");
        filesystem
            .write(&source, &original)
            .await
            .expect("source archive");
        window.run_until_parked();
        let item = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.open_abs_path(source.clone(), Default::default(), window, cx)
            })
            .await
            .expect("open nested archive")
            .downcast::<GraphiteDocumentItem>()
            .expect("native Inspector");
        choose_copy_node(window, "GRAPHITE-GDD-NODE-COPY-ITEM-1");
        click(window, "GRAPHITE-GDD-ENTER-NETWORK");
        click(window, "GRAPHITE-GDD-TOGGLE-SAVED-GRAPH");
        let (network, original_nodes, input_revision) = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready graph")
            };
            (
                form.active_network().expect("editable nested scope").id,
                form.nodes
                    .iter()
                    .map(|node| (node.storage_id(), node.network_id()))
                    .collect::<BTreeSet<_>>(),
                item.model.read(cx).revision.clone(),
            )
        });
        let bounds = window
            .debug_bounds("GRAPHITE-GDD-SAVED-GRAPH")
            .expect("nested canvas");
        click_graph_point(
            window,
            bounds.origin + gpui::point(px(250.), px(177.)),
            gpui::MouseButton::Right,
            gpui::Modifiers::default(),
        );
        click(window, "GRAPHITE-GDD-ADD-Custom Node");
        let (node, child) = item.read_with(window, |item, cx| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready working graph")
            };
            assert_eq!(
                form.active_network().expect("same visible scope").id,
                network
            );
            assert!(item.is_dirty(cx));
            assert!(!Arc::ptr_eq(&item.model.read(cx).revision, &input_revision));
            let added = form
                .nodes
                .iter()
                .filter(|node| !original_nodes.contains(&(node.storage_id(), node.network_id())))
                .collect::<Vec<_>>();
            let [added] = added.as_slice() else {
                panic!("one scoped addition")
            };
            let child = added.nested_network().expect("owned nested network");
            assert_eq!(added.display_name(), "Custom Node");
            assert_eq!(added.network_id(), network);
            assert!(form.networks.iter().any(|entry| entry.id == child
                && entry.parent == Some(network)
                && entry.import_count == 0
                && entry.export_count == 0));
            (added.storage_id(), child)
        });
        click(window, "GRAPHITE-GDD-UNDO");
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready history")
            };
            assert!(form.nodes.iter().all(|entry| entry.storage_id() != node));
            assert!(form.networks.iter().all(|entry| entry.id != child));
        });
        click(window, "GRAPHITE-GDD-REDO");
        item.read_with(window, |item, _| {
            let NodeCopyPanel::Ready(form) = &item.node_copy else {
                panic!("ready history")
            };
            assert!(
                form.nodes.iter().any(
                    |entry| entry.storage_id() == node && entry.nested_network() == Some(child)
                )
            );
        });
        let destination = PathBuf::from("/project/saved-working-custom-node.gdd");
        click(window, "GRAPHITE-GDD-SAVE-WORKING-FOLDER");
        window.simulate_new_path_selection(|_| Some(destination.clone()));
        window.run_until_parked();
        let saved = read_graphite_folder(filesystem.as_ref(), &destination)
            .await
            .expect("reopened working custom node");
        let graph = saved.graph_inspection().await.expect("reopened graph");
        assert!(original_nodes.iter().all(|address| {
            graph
                .nodes
                .iter()
                .any(|entry| (entry.storage_id(), entry.network_id()) == *address)
        }));
        assert!(graph.nodes.iter().any(|entry| entry.storage_id() == node
            && entry.network_id() == network
            && entry.nested_network() == Some(child)));
        assert_eq!(
            filesystem.load_bytes(&source).await.expect("original ZIP"),
            original
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
