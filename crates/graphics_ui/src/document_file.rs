use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, anyhow, ensure};
use gpui::{
    App, AppContext as _, AsyncWindowContext, Context, Entity, EntityId, EventEmitter, FocusHandle,
    Focusable, PathPromptOptions, Render, SharedString, Task, WeakEntity, Window, actions,
};
use graphics_core::formats::{
    gdd::{
        ArchiveFormat, GddArchiveBudget, GddExportContent, GddRecoverySource, GddSavedView,
        InspectedGddV1, create_gdd_folder, inspect_gdd_folder_files, open_gdd_archive,
    },
    legacy::{LegacyDocumentSchema, PreservedLegacyDocument, inspect_legacy_document},
};
use project::{DirectoryLister, Project, ProjectEntryId, ProjectPath, Worktree};
use ui::{Button, Checkbox, Color, Label, LabelSize, ToggleState, prelude::*};
use workspace::{
    Item, Pane, Workspace, item::ItemBufferKind, notifications::DetachAndPromptErr as _,
};

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
}

enum PreparedCopy {
    File(Arc<[u8]>),
    Folder(Vec<(PathBuf, Vec<u8>)>),
}

impl GraphiteDocumentItem {
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
        let (format, recover_pending) = match request {
            GraphiteCopy::Original => (None, false),
            GraphiteCopy::Export(format) => (Some(format), false),
            GraphiteCopy::RecoverFolder => (None, true),
        };
        let content = if format.is_some() {
            self.export_content
        } else {
            GddExportContent::default()
        };
        let embed_linked_resources = format.is_some() && self.embed_linked_resources;
        let embed_legacy_document = format.is_some() && self.embed_legacy_document;
        if format.is_some()
            && let Err(error) = content.validate()
        {
            self.copy_feedback = Some((error.to_string().into(), Color::Error));
            cx.notify();
            return;
        }
        let model = self.model.read(cx);
        let source = model.path.clone();
        let project = self.project.clone();
        let format = format.or_else(|| {
            (recover_pending || model.original.is_none()).then_some(GddCopyFormat::Folder)
        });
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
                if recover_pending {
                    format!("Recovered {stem}{suffix}")
                } else {
                    format!("Copy of {stem}{suffix}")
                }
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
                        if recover_pending {
                            let recovered = archive.recover_pending_edits(ARCHIVE_BUDGET).await?;
                            return Ok(PreparedCopy::Folder(recovered.folder_files()?));
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
        })))
    }
}

impl Render for GraphiteDocumentItem {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
                        .when(contents.has_saved_view(), |view| {
                            view.child(self.render_saved_view(cx))
                        })
                        .when(recovery.hot_log_len > 0, |view| {
                            view.child(
                                div()
                                    .debug_selector(|| "GRAPHITE-GDD-SAVE-RECOVERED".into())
                                    .child(
                                        Button::new(
                                            "graphite-gdd-save-recovered",
                                            "Save Recovered Folder",
                                        )
                                        .disabled(self.copy_task.is_some())
                                        .on_click(
                                            cx.listener(|item, _, window, cx| {
                                                item.save_copy(
                                                    GraphiteCopy::RecoverFolder,
                                                    window,
                                                    cx,
                                                )
                                            }),
                                        ),
                                    ),
                            )
                        })
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
