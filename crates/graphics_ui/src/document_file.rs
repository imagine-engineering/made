use std::{path::PathBuf, sync::Arc};

use anyhow::{Context as _, anyhow, ensure};
use gpui::{
    App, AppContext as _, Context, Entity, EntityId, EventEmitter, FocusHandle, Focusable, Render,
    SharedString, Task, WeakEntity, Window,
};
use graphics_core::formats::{
    gdd::{GddArchiveBudget, PreservedGddV1, open_gdd_archive},
    legacy::{LegacyDocumentSchema, PreservedLegacyDocument, inspect_legacy_document},
};
use project::{Project, ProjectEntryId, ProjectPath};
use ui::{Button, Color, Label, LabelSize, prelude::*};
use workspace::{Item, Pane, item::ItemBufferKind};

const ARCHIVE_BUDGET: GddArchiveBudget = GddArchiveBudget {
    max_archive_bytes: 64 * 1024 * 1024,
    max_entry_bytes: 64 * 1024 * 1024,
    max_total_entry_bytes: 128 * 1024 * 1024,
    max_entries: 8192,
};

enum InspectedDocument {
    Archive {
        archive: Box<PreservedGddV1>,
        history_len: usize,
        hot_log_len: usize,
    },
    Legacy(PreservedLegacyDocument),
}

pub(crate) struct GraphiteDocumentProjectItem {
    name: SharedString,
    path: ProjectPath,
    entry_id: Option<ProjectEntryId>,
    original: Arc<[u8]>,
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
                Ok((
                    bytes,
                    InspectedDocument::Archive {
                        history_len: archive.history_len(),
                        hot_log_len: archive.hot_log_len(),
                        archive: Box::new(archive),
                    },
                ))
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
                original,
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

pub(crate) struct GraphiteDocumentItem {
    model: Entity<GraphiteDocumentProjectItem>,
    project: WeakEntity<Project>,
    focus_handle: FocusHandle,
    copy_task: Option<Task<()>>,
    copy_feedback: Option<(SharedString, Color)>,
}

impl GraphiteDocumentItem {
    fn save_copy(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.copy_task.is_some() {
            return;
        }
        let model = self.model.read(cx);
        let original = model.original.clone();
        let suggested_name = format!("Copy of {}", model.name);
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
                item.read_with(cx, |_, _| ())?;
                ensure!(
                    path.is_absolute(),
                    "Choose an absolute path for the Graphite copy."
                );
                filesystem
                    .create_new_file_atomically(path.clone(), original)
                    .await
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
                        .on_click(cx.listener(|item, _, window, cx| item.save_copy(window, cx))),
                ),
            )
            .when_some(self.copy_feedback.clone(), |view, (message, color)| {
                view.child(Label::new(message).color(color))
            });
        match &model.document {
            InspectedDocument::Archive {
                archive,
                history_len,
                hot_log_len,
            } => {
                let manifest = archive.manifest();
                view.child(Label::new(format!(
                    "Graphite GDD v{} · Read-only",
                    manifest.format_version
                )))
                .child(Label::new(format!("Document ID {}", manifest.document_id)))
                .child(Label::new(format!(
                    "Recorded editor version {}",
                    manifest.editor_version
                )))
                .child(Label::new(format!(
                    "{history_len} history entries · {hot_log_len} pending operations"
                )))
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
            let InspectedDocument::Archive { archive, .. } = &model.document else {
                panic!("expected the Graphite archive model");
            };
            assert_eq!(archive.manifest().document_id, 171);
            assert_eq!(archive.original_bytes(), original);
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
            let InspectedDocument::Archive { archive, .. } = &item.model.read(cx).document else {
                panic!("expected the Graphite archive model");
            };
            assert_eq!(archive.original_bytes(), bytes);
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
