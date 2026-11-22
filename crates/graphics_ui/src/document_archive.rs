use std::sync::Arc;

use anyhow::{Context as _, anyhow, ensure};
use gpui::{
    App, AppContext as _, Context, Entity, EntityId, EventEmitter, FocusHandle, Focusable, Render,
    SharedString, Task, Window,
};
use graphics_core::formats::gdd::{GddArchiveBudget, PreservedGddV1, open_gdd_zip};
use project::{Project, ProjectEntryId, ProjectPath};
use ui::{Label, LabelSize, prelude::*};
use workspace::{Item, Pane, item::ItemBufferKind};

const ARCHIVE_BUDGET: GddArchiveBudget = GddArchiveBudget {
    max_archive_bytes: 64 * 1024 * 1024,
    max_entry_bytes: 64 * 1024 * 1024,
    max_total_entry_bytes: 128 * 1024 * 1024,
    max_entries: 8192,
};

pub(crate) struct GraphiteArchiveProjectItem {
    name: SharedString,
    path: ProjectPath,
    entry_id: Option<ProjectEntryId>,
    archive: PreservedGddV1,
    history_len: usize,
    hot_log_len: usize,
}

impl project::ProjectItem for GraphiteArchiveProjectItem {
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
        if !extension.ends_with(".gdd.zip") && !extension.ends_with(".gdd") {
            return None;
        }
        if !project_model.is_local() {
            return Some(Task::ready(Err(anyhow!(
                "Open this Graphite archive from a local workspace."
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
                let archive = futures::executor::block_on(open_gdd_zip(&bytes, ARCHIVE_BUDGET))?;
                let history_len = archive.history_len();
                let hot_log_len = archive.hot_log_len();
                Ok::<_, anyhow::Error>((archive, history_len, hot_log_len))
            }
        });
        Some(cx.spawn(async move |cx| {
            let (archive, history_len, hot_log_len) = load
                .await
                .with_context(|| format!("Opening Graphite archive {name}"))?;
            project.read_with(cx, |project, cx| {
                let current_worktree = project.worktree_for_id(path.worktree_id, cx);
                ensure!(
                    project.is_local()
                        && current_worktree.is_some_and(|current| {
                            current == worktree && current.read(cx).abs_path() == root
                        })
                        && project.entry_for_path(&path, cx).map(|entry| entry.id) == entry_id,
                    "The Graphite archive path changed while it was opening. Reopen the file."
                );
                Ok(())
            })??;
            Ok(cx.new(|_| Self {
                name,
                path,
                entry_id,
                archive,
                history_len,
                hot_log_len,
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

pub(crate) struct GraphiteArchiveItem {
    model: Entity<GraphiteArchiveProjectItem>,
    focus_handle: FocusHandle,
}

impl workspace::ProjectItem for GraphiteArchiveItem {
    type Item = GraphiteArchiveProjectItem;

    fn for_project_item(
        _project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            model: item,
            focus_handle: cx.focus_handle(),
        }
    }
}

impl Focusable for GraphiteArchiveItem {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<()> for GraphiteArchiveItem {}

impl Item for GraphiteArchiveItem {
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
        Task::ready(Some(cx.new(|cx| Self {
            model,
            focus_handle: cx.focus_handle(),
        })))
    }
}

impl Render for GraphiteArchiveItem {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let model = self.model.read(cx);
        let manifest = model.archive.manifest();
        v_flex()
            .id("graphite-archive-inspector")
            .track_focus(&self.focus_handle)
            .size_full()
            .p_6()
            .gap_2()
            .overflow_y_scroll()
            .child(Label::new(model.name.clone()).size(LabelSize::Large))
            .child(Label::new(format!(
                "Graphite GDD v{} · Read-only",
                manifest.format_version
            )))
            .child(Label::new(format!("Document ID {}", manifest.document_id)))
            .child(Label::new(format!(
                "Recorded editor version {}",
                manifest.editor_version
            )))
            .child(Label::new(format!(
                "{} history entries · {} pending operations",
                model.history_len, model.hot_log_len
            )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png_workflow::tests::native_workspace;
    use gpui::TestAppContext;
    use graphite_document_container::{AnyContainer, backends::memory::MemoryBackend};
    use graphite_document_format::{ExportFormat, ExportOptions, GddV1, GddV1Layout};
    use graphite_graph_craft::application_io::resource::HashMapResourceStorage;
    use std::path::PathBuf;

    async fn archive_bytes(version: u32) -> Vec<u8> {
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
                ExportFormat::Zip,
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
        let original = archive_bytes(1).await;
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
            .downcast::<GraphiteArchiveItem>()
            .expect("native Graphite archive view");
        item.read_with(window, |item, cx| {
            let model = item.model.read(cx);
            assert_eq!(item.tab_content_text(0, cx).as_ref(), "original.gdd.zip");
            assert_eq!(model.archive.manifest().document_id, 171);
            assert_eq!(model.archive.original_bytes(), original);
            assert!(!item.is_dirty(cx));
            assert!(!item.can_save(cx));
            assert!(!item.can_save_as(cx));
        });

        let future_path = PathBuf::from("/project/future.gdd.zip");
        let future = archive_bytes(2).await;
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
                .active_item_as::<GraphiteArchiveItem>(cx)),
            Some(item)
        );
    }
}
