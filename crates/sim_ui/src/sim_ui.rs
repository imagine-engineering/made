mod actions;
mod context_menu;
#[cfg(all(test, feature = "test-support"))]
mod context_menu_tests;
mod execution_catalog;
mod execution_model;
mod execution_panel;
mod execution_surfaces;
#[cfg(all(test, feature = "test-support"))]
mod execution_tests;
mod generated_command_catalog;
mod generated_execution_catalog;
mod generated_frontend_extension_catalog;
mod generated_keybinding_catalog;
mod generated_menu_catalog;
mod graph;
mod graph_commands;
mod graph_render;
#[cfg(all(test, feature = "test-support"))]
mod graph_tests;
mod history_panel;
mod image_import;
mod legacy_extension_placeholder;
mod output_view;
#[cfg(all(test, feature = "test-support"))]
mod plugin_contribution_tests;
mod plugin_contributions;
mod properties_panel;
#[cfg(all(test, feature = "test-support"))]
mod properties_panel_tests;
mod queue_panel;
mod shell;
mod workflow_item;

use anyhow::{Context as _, anyhow};
use gpui::{App, AppContext as _, Context, Entity, Global, Task, WeakEntity, Window};
use project::{Project, ProjectEntryId, ProjectPath};
use sim_runtime::{
    ProfileId, SharedAssetService, SubgraphBlueprintCatalog, SubgraphBlueprintLibrary,
    WorkflowStorageProvider, authorize_native_subgraph_library,
};
use sim_types::CancellationToken;
use workspace::{Pane, item::ProjectItemKind};

pub use actions::*;
pub use context_menu::*;
pub use execution_catalog::*;
pub use execution_model::*;
pub use execution_panel::*;
pub use graph::*;
pub use graph_commands::*;
pub use history_panel::*;
pub use image_import::*;
pub use legacy_extension_placeholder::*;
pub use output_view::*;
pub use plugin_contributions::*;
pub use properties_panel::*;
pub use queue_panel::*;
pub use shell::*;
pub use workflow_item::*;

pub use generated_frontend_extension_catalog::{
    GENERATED_FRONTEND_EXTENSION_DISPOSITIONS, GeneratedFrontendExtensionDisposition,
    GeneratedFrontendExtensionDispositionKind,
};

struct SimUiRegistration {
    initialization_error: Option<String>,
    #[cfg(feature = "test-support")]
    completed_passes: u8,
}

impl Global for SimUiRegistration {}

#[derive(Clone)]
pub struct NativeAssetServices {
    profile_id: String,
    assets: SharedAssetService,
    subgraph_blueprints: SubgraphBlueprintLibrary,
    image_importer: std::sync::Arc<image_import::NativeImageImportService>,
}

impl NativeAssetServices {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn assets(&self) -> SharedAssetService {
        self.assets.clone()
    }

    pub fn subgraph_blueprints(&self) -> SubgraphBlueprintLibrary {
        self.subgraph_blueprints.clone()
    }

    pub fn image_importer(&self) -> Option<std::sync::Arc<dyn NativeImageImporter>> {
        Some(self.image_importer.clone())
    }

    pub fn image_output_reader(&self) -> Option<std::sync::Arc<dyn NativeImageOutputReader>> {
        Some(self.image_importer.clone())
    }
}

pub(crate) fn subgraph_catalog_diagnostic_message(
    catalog: &SubgraphBlueprintCatalog,
) -> Option<String> {
    let first = catalog.diagnostics().first()?;
    Some(format!(
        "isolated {} invalid subgraph blueprint asset(s); first was {}: {}",
        catalog.diagnostics().len(),
        first.identity.relative_path.display(),
        first.message
    ))
}

pub(crate) fn subgraph_catalog_node_library_message(catalog: &SubgraphBlueprintCatalog) -> String {
    let entry_count = catalog.entries().len();
    let names = catalog
        .entries()
        .values()
        .take(3)
        .map(|entry| entry.descriptor.display_name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    match (entry_count, names.is_empty()) {
        (0, _) => "Native node library updated: no published subgraphs".to_owned(),
        (1, false) => format!("Native node library updated: 1 published subgraph ({names})"),
        (_, false) => format!(
            "Native node library updated: {entry_count} published subgraphs (including {names})"
        ),
        _ => format!("Native node library updated: {entry_count} published subgraphs"),
    }
}

mod native_asset_projection {
    use super::*;

    struct GlobalNativeAssetServices {
        services: NativeAssetServices,
        subgraph_catalog: SubgraphBlueprintCatalog,
    }

    impl Global for GlobalNativeAssetServices {}

    #[derive(Default)]
    pub(crate) struct GlobalNativeSubgraphCatalogRevision(u64);

    impl Global for GlobalNativeSubgraphCatalogRevision {}

    pub fn register_native_asset_services(
        assets: SharedAssetService,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        let profile_id = assets
            .lock()
            .map_err(|error| anyhow!("native asset service is unavailable: {error}"))?
            .roots()
            .profile_id
            .clone();
        let authorization = authorize_native_subgraph_library(profile_id.clone())?;
        let subgraph_blueprints = SubgraphBlueprintLibrary::new(assets.clone(), authorization);
        let subgraph_catalog = subgraph_blueprints.reload(&CancellationToken::default())?;
        let image_importer = image_import::NativeImageImportService::new(assets.clone())?;
        if let Some(previous) = cx.try_global::<GlobalNativeAssetServices>() {
            previous.services.image_importer.revoke();
        }
        cx.set_global(GlobalNativeAssetServices {
            services: NativeAssetServices {
                profile_id,
                assets,
                subgraph_blueprints,
                image_importer,
            },
            subgraph_catalog,
        });
        cx.set_global(GlobalNativeSubgraphCatalogRevision::default());
        Ok(())
    }

    pub fn native_asset_services(cx: &App) -> Option<NativeAssetServices> {
        cx.try_global::<GlobalNativeAssetServices>()
            .map(|global| global.services.clone())
    }

    pub(crate) fn native_subgraph_catalog(cx: &App) -> Option<&SubgraphBlueprintCatalog> {
        cx.try_global::<GlobalNativeAssetServices>()
            .map(|global| &global.subgraph_catalog)
    }

    pub(crate) fn replace_native_subgraph_catalog(
        catalog: SubgraphBlueprintCatalog,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        let Some(global) = cx.try_global::<GlobalNativeAssetServices>() else {
            return Err(anyhow!("native subgraph catalog is unavailable"));
        };
        if !catalog_profile_matches(&catalog, &global.services.profile_id) {
            return Err(anyhow!(
                "native subgraph catalog profile does not match the active profile"
            ));
        }
        cx.global_mut::<GlobalNativeAssetServices>()
            .subgraph_catalog = catalog;
        cx.defer(|cx| {
            if cx.has_global::<GlobalNativeSubgraphCatalogRevision>() {
                let revision = cx.global_mut::<GlobalNativeSubgraphCatalogRevision>();
                revision.0 = revision.0.wrapping_add(1);
            }
        });
        Ok(())
    }

    #[cfg(all(test, feature = "test-support"))]
    pub(crate) fn remove_native_asset_services_for_test(cx: &mut App) {
        if let Some(previous) = cx.try_global::<GlobalNativeAssetServices>() {
            previous.services.image_importer.revoke();
        }
        let _removed_services = cx.remove_global::<GlobalNativeAssetServices>();
        let _removed_revision = cx.remove_global::<GlobalNativeSubgraphCatalogRevision>();
    }

    fn catalog_profile_matches(catalog: &SubgraphBlueprintCatalog, profile_id: &str) -> bool {
        catalog
            .entries()
            .values()
            .map(|entry| entry.asset.identity.profile_id.as_str())
            .chain(
                catalog
                    .diagnostics()
                    .iter()
                    .map(|diagnostic| diagnostic.identity.profile_id.as_str()),
            )
            .all(|candidate| candidate == profile_id)
    }
}

#[cfg(all(test, feature = "test-support"))]
pub(crate) use native_asset_projection::remove_native_asset_services_for_test;
pub(crate) use native_asset_projection::{
    GlobalNativeSubgraphCatalogRevision, native_subgraph_catalog, replace_native_subgraph_catalog,
};
pub use native_asset_projection::{native_asset_services, register_native_asset_services};

pub struct NativeWorkflowProjectItem {
    model: GraphWorkspaceModel,
    entry_id: Option<ProjectEntryId>,
    project_path: ProjectPath,
}

fn is_native_workflow_file_name(file_name: &str) -> bool {
    let file_name = file_name.to_ascii_lowercase();
    file_name == "workflow.json"
        || file_name.ends_with("-workflow.json")
        || file_name.ends_with("_workflow.json")
        || file_name.ends_with(".workflow.json")
        || file_name.ends_with(".sim.json")
        || file_name.ends_with(".simworkflow")
        || file_name.ends_with(".comfy.json")
        || file_name.ends_with(".comfyworkflow")
}

impl project::ProjectItem for NativeWorkflowProjectItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        let root_file_path = if path.path.is_empty() {
            project.read(cx).absolute_path(path, cx)
        } else {
            None
        };
        let file_name = path
            .path
            .file_name()
            .or_else(|| root_file_path.as_deref()?.file_name()?.to_str());
        let file_name = file_name.filter(|file_name| is_native_workflow_file_name(file_name))?;
        let title = file_name.to_owned();

        let absolute_path =
            match root_file_path.or_else(|| project.read(cx).absolute_path(path, cx)) {
                Some(path) => path,
                None => {
                    return Some(Task::ready(Err(anyhow!(
                        "native workflow path cannot be resolved locally: {:?}",
                        path.path
                    ))));
                }
            };
        let document_identity = match absolute_path.to_str() {
            Some(path) => path.to_owned(),
            None => {
                return Some(Task::ready(Err(anyhow!(
                    "native workflow path is not valid UTF-8: {}",
                    absolute_path.display()
                ))));
            }
        };
        let entry_id = project
            .read(cx)
            .entry_for_path(path, cx)
            .map(|entry| entry.id);
        let project_path = path.clone();
        let fs = project.read(cx).fs().clone();
        let load_task = cx.background_spawn(async move {
            let bytes = fs.load_bytes(&absolute_path).await.with_context(|| {
                format!("failed to read native workflow {}", absolute_path.display())
            })?;
            GraphWorkspaceModel::open(
                title,
                document_identity,
                WorkflowStorageProvider::LocalFile,
                bytes,
            )
            .map_err(anyhow::Error::from)
        });

        Some(cx.spawn(async move |cx| {
            let model = load_task.await?;
            Ok(cx.new(|_| Self {
                model,
                entry_id,
                project_path,
            }))
        }))
    }

    fn entry_id(&self, _cx: &App) -> Option<ProjectEntryId> {
        self.entry_id
    }

    fn project_path(&self, _cx: &App) -> Option<ProjectPath> {
        Some(self.project_path.clone())
    }

    fn is_dirty(&self) -> bool {
        false
    }
}

impl workspace::ProjectItem for GraphWorkspaceItem {
    type Item = NativeWorkflowProjectItem;

    fn project_item_kind() -> Option<ProjectItemKind> {
        Some(ProjectItemKind("ComfyNativeGraph"))
    }

    fn for_project_item(
        _project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(item.read(cx).model.clone(), WeakEntity::new_invalid(), cx)
    }
}

pub fn register_native_workflow_project_item(cx: &mut App) {
    workspace::register_project_item::<GraphWorkspaceItem>(cx);
}

pub fn init(cx: &mut App) {
    init_for_profile(LOCAL_EXECUTION_PROFILE_ID, cx);
}

pub fn init_for_profile(profile_id: ProfileId, cx: &mut App) {
    if cx.try_global::<SimUiRegistration>().is_some() {
        if let Err(error) = init_execution_ui_model_for_profile(profile_id, cx) {
            set_initialization_error(error.to_string(), cx);
        }
        return;
    }

    workspace::register_serializable_item::<GraphWorkspaceItem>(cx);
    register_native_workflow_project_item(cx);
    let initialization_error = init_execution_ui_model_for_profile(profile_id, cx)
        .err()
        .map(|error| error.to_string());
    init_execution_panel(cx);
    init_graph_properties_panel(cx);
    cx.set_global(SimUiRegistration {
        initialization_error,
        #[cfg(feature = "test-support")]
        completed_passes: 1,
    });
}

pub fn initialized_profile_id(cx: &App) -> Option<ProfileId> {
    execution_ui_model(cx).and_then(|model| model.read(cx).active_profile_id())
}

pub fn initialization_error(cx: &App) -> Option<&str> {
    cx.try_global::<SimUiRegistration>()
        .and_then(|registration| registration.initialization_error.as_deref())
}

pub fn set_initialization_error(message: impl Into<String>, cx: &mut App) {
    let message = message.into();
    if cx.try_global::<SimUiRegistration>().is_none() {
        return;
    }
    let registration = cx.global_mut::<SimUiRegistration>();
    registration.initialization_error = Some(match registration.initialization_error.take() {
        Some(existing) => format!("{existing}; {message}"),
        None => message,
    });
}

pub fn clear_initialization_error(cx: &mut App) {
    if cx.try_global::<SimUiRegistration>().is_some() {
        cx.global_mut::<SimUiRegistration>().initialization_error = None;
    }
}

#[cfg(feature = "test-support")]
pub fn initialization_passes_for_test(cx: &App) -> Option<u8> {
    cx.try_global::<SimUiRegistration>()
        .map(|registration| registration.completed_passes)
}

#[cfg(all(test, feature = "test-support"))]
mod registration_tests {
    use super::*;
    use gpui::TestAppContext;
    use project::FakeFs;
    use serde_json::json;
    use std::path::Path;
    use uuid::Uuid;
    use workspace::ItemHandle as _;

    #[gpui::test(seed = 417)]
    fn initialization_error_notifications_preserve_append_and_recovery(cx: &mut TestAppContext) {
        use std::{cell::Cell, rc::Rc};

        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            init(cx);
        });
        cx.run_until_parked();
        let notifications = Rc::new(Cell::new(0));
        let _observation = cx.update(|cx| {
            let notifications = notifications.clone();
            cx.observe_global::<SimUiRegistration>(move |_| {
                notifications.set(notifications.get() + 1);
            })
        });
        cx.run_until_parked();
        let initial_notifications = notifications.get();
        cx.update(|cx| set_initialization_error("worker initialization failed", cx));
        cx.run_until_parked();
        assert!(notifications.get() > initial_notifications);
        assert_eq!(
            cx.read(|cx| initialization_error(cx).map(str::to_owned)),
            Some("worker initialization failed".to_owned())
        );
        let first_notifications = notifications.get();
        cx.update(|cx| set_initialization_error("shutdown failed", cx));
        cx.run_until_parked();
        assert!(notifications.get() > first_notifications);
        assert_eq!(
            cx.read(|cx| initialization_error(cx).map(str::to_owned)),
            Some("worker initialization failed; shutdown failed".to_owned())
        );
        let update_notifications = notifications.get();
        cx.update(|cx| clear_initialization_error(cx));
        cx.run_until_parked();
        assert!(notifications.get() > update_notifications);
        assert!(cx.read(|cx| initialization_error(cx).is_none()));
        cx.update(|cx| {
            init_for_profile(LOCAL_EXECUTION_PROFILE_ID, cx);
            assert!(initialization_error(cx).is_none());
            assert_eq!(initialization_passes_for_test(cx), Some(1));
        });
        let clear_notifications = notifications.get();
        cx.update(|cx| set_initialization_error("new activation failed", cx));
        cx.run_until_parked();
        assert!(notifications.get() > clear_notifications);
        assert_eq!(
            cx.read(|cx| initialization_error(cx).map(str::to_owned)),
            Some("new activation failed".to_owned())
        );
    }

    const WORKFLOW: &str = r#"{"version":0.4,"last_node_id":0,"last_link_id":0,"nodes":[],"links":[],"groups":[],"config":{},"extra":{}}"#;

    #[gpui::test(seed = 16016)]
    async fn init_is_idempotent_and_registers_open_and_restoration_handlers(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            workspace::AppState::test(cx);
            let selected_profile_id = ProfileId(Uuid::from_u128(16_016));
            let next_profile_id = ProfileId(Uuid::from_u128(16_017));
            init_for_profile(selected_profile_id, cx);
            init_for_profile(next_profile_id, cx);

            let registration = cx
                .try_global::<SimUiRegistration>()
                .expect("native graph registration should be present");
            assert_eq!(registration.completed_passes, 1);
            assert_eq!(initialized_profile_id(cx), Some(next_profile_id));
            let execution_model = execution_ui_model(cx)
                .expect("native execution model should be registered exactly once");
            assert!(
                execution_model
                    .read(cx)
                    .snapshot(selected_profile_id)
                    .is_ok()
            );
            assert!(execution_model.read(cx).snapshot(next_profile_id).is_ok());

            let draft = cx.new(|cx| {
                GraphWorkspaceItem::new_draft(
                    "Restorable native workflow",
                    WeakEntity::new_invalid(),
                    cx,
                )
                .expect("create restorable native workflow")
            });
            assert!(draft.to_serializable_item_handle(cx).is_some());
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            Path::new("/project"),
            json!({"native-workflow.json": WORKFLOW, "ordinary.json": WORKFLOW}),
        )
        .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let workflow_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("project/native-workflow.json", cx)
            })
            .expect("resolve native workflow path");
        let ordinary_json_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("project/ordinary.json", cx)
            })
            .expect("resolve ordinary JSON path");
        let (workspace, cx) =
            cx.add_window_view(|window, cx| workspace::Workspace::test_new(project, window, cx));

        let item = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(workflow_path, None, true, window, cx)
            })
            .await
            .expect("open registered native workflow")
            .downcast::<GraphWorkspaceItem>()
            .expect("native workflow should build a graph workspace item");
        item.read_with(cx, |item, _cx| {
            assert_eq!(item.model().title, "native-workflow.json");
            assert_eq!(
                item.model().save_coordinator.provider(),
                &WorkflowStorageProvider::LocalFile
            );
            assert_eq!(
                item.model().save_coordinator.document_identity(),
                "/project/native-workflow.json"
            );
            assert!(!item.model().is_read_only());
        });

        let ordinary_open = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(ordinary_json_path, None, true, window, cx)
            })
            .await;
        assert!(
            ordinary_open.is_err(),
            "native registration must not take ownership of unrelated JSON files"
        );
    }

    #[gpui::test(seed = 417)]
    async fn late_native_registration_preserves_editor_fallback_and_claims_workflows(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            init(cx);
            editor::init(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            Path::new("/project"),
            json!({
                "before.workflow.json": WORKFLOW,
                "after.workflow.json": WORKFLOW,
                "ordinary.json": WORKFLOW,
                "broken.workflow.json": "not JSON"
            }),
        )
        .await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let resolve = |name: &str, cx: &TestAppContext| {
            project
                .read_with(cx, |project, cx| {
                    project.find_project_path(&format!("project/{name}"), cx)
                })
                .expect("resolve project file")
        };
        let before = resolve("before.workflow.json", cx);
        let after = resolve("after.workflow.json", cx);
        let ordinary = resolve("ordinary.json", cx);
        let broken = resolve("broken.workflow.json", cx);
        let (workspace, cx) =
            cx.add_window_view(|window, cx| workspace::Workspace::test_new(project, window, cx));
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(before, None, true, window, cx)
            })
            .await
            .expect("original production order opens buffer")
            .downcast::<editor::Editor>()
            .expect("editor wins before late registration");
        cx.update(|_, cx| register_native_workflow_project_item(cx));
        let graph = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(after, None, true, window, cx)
            })
            .await
            .expect("open native workflow")
            .downcast::<GraphWorkspaceItem>()
            .expect("native opener wins after late registration");
        graph.read_with(cx, |item, _| {
            assert_eq!(
                item.model().save_coordinator.document_identity(),
                "/project/after.workflow.json"
            );
            assert!(!item.model().is_read_only());
        });
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(ordinary, None, true, window, cx)
            })
            .await
            .expect("ordinary JSON remains editable")
            .downcast::<editor::Editor>()
            .expect("ordinary JSON still uses Editor");
        let broken_graph = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(broken, None, true, window, cx)
            })
            .await
            .expect("malformed workflow opens with its canonical diagnostic")
            .downcast::<GraphWorkspaceItem>()
            .expect("malformed claimed workflow must not fall back to Editor");
        broken_graph.read_with(cx, |item, _| {
            let model = item.model();
            assert!(model.is_read_only());
            assert_eq!(model.original_bytes(), b"not JSON");
            assert_eq!(
                model.save_coordinator.document_identity(),
                "/project/broken.workflow.json"
            );
            assert!(matches!(
                &model.open_state,
                WorkflowOpenState::ReadOnly { diagnostic, .. } if !diagnostic.is_empty()
            ));
        });
    }

    #[gpui::test(seed = 417)]
    async fn single_file_worktrees_resolve_native_suffix_without_claiming_ordinary_json(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let app_state = workspace::AppState::test(cx);
            workspace::AppState::set_global(app_state, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            Path::new("/project"),
            json!({
                "shader.workflow.json": WORKFLOW,
                "ordinary.json": WORKFLOW,
                "broken.workflow.json": "not JSON"
            }),
        )
        .await;
        let project = Project::test(
            fs,
            [
                Path::new("/project/shader.workflow.json"),
                Path::new("/project/ordinary.json"),
                Path::new("/project/broken.workflow.json"),
            ],
            cx,
        )
        .await;
        let resolve = |absolute_path: &str, cx: &TestAppContext| {
            let path = project
                .read_with(cx, |project, cx| {
                    project.find_project_path(absolute_path, cx)
                })
                .expect("resolve single-file root");
            assert!(
                path.path.is_empty(),
                "single-file root must exercise empty relative path"
            );
            path
        };
        let native = resolve("/project/shader.workflow.json", cx);
        let ordinary = resolve("/project/ordinary.json", cx);
        let broken = resolve("/project/broken.workflow.json", cx);
        let (workspace, cx) =
            cx.add_window_view(|window, cx| workspace::Workspace::test_new(project, window, cx));
        let graph = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(native, None, true, window, cx)
            })
            .await
            .expect("open single-file native workflow")
            .downcast::<GraphWorkspaceItem>()
            .expect("single-file suffix must select Graph");
        graph.read_with(cx, |item, _| {
            assert_eq!(item.model().title, "shader.workflow.json");
            assert_eq!(
                item.model().save_coordinator.document_identity(),
                "/project/shader.workflow.json"
            );
        });
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(ordinary, None, true, window, cx)
            })
            .await
            .expect("open ordinary single-file JSON")
            .downcast::<editor::Editor>()
            .expect("ordinary single-file JSON must remain Editor");
        let broken_graph = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(broken, None, true, window, cx)
            })
            .await
            .expect("malformed workflow opens with its canonical diagnostic")
            .downcast::<GraphWorkspaceItem>()
            .expect("malformed claimed workflow must not fall back to Editor");
        broken_graph.read_with(cx, |item, _| {
            let model = item.model();
            assert!(model.is_read_only());
            assert_eq!(model.original_bytes(), b"not JSON");
            assert_eq!(
                model.save_coordinator.document_identity(),
                "/project/broken.workflow.json"
            );
            assert!(matches!(
                &model.open_state,
                WorkflowOpenState::ReadOnly { diagnostic, .. } if !diagnostic.is_empty()
            ));
        });
    }
}
