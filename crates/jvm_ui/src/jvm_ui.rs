mod dependencies_panel;

pub use dependencies_panel::{DependenciesPanel, ShowGradle, ShowMaven, ToggleDependenciesPanel};

use gpui::{App, KeyBinding};
use language_tools::language_tool_tree::{
    ActivateSelected, CollapseAll, ExpandAll, Refresh, SelectFirst, SelectFirstChild, SelectLast,
    SelectNext, SelectParent, SelectPrevious, ToggleExpanded,
};
use workspace::Workspace;

pub fn init(cx: &mut App) {
    init_for_product(product_flavor::ID, cx);
}

fn init_for_product(product_id: &'static str, cx: &mut App) {
    let default_jvm = product_id == "jvm";
    cx.observe_new(move |workspace: &mut Workspace, _, cx| {
        workspace.register_action(DependenciesPanel::toggle_focus);
        workspace.register_action(DependenciesPanel::show_gradle);
        workspace.register_action(DependenciesPanel::show_maven);
        if default_jvm {
            let project = workspace.project().clone();
            if !project.read(cx).is_via_collab() {
                let store = project.read(cx).lsp_store();
                if let Err(error) = store.update(cx, |store, cx| {
                    store.set_default_jvm_project_languages_enabled(true, cx)
                }) {
                    log::error!("Could not enable Groovy project language tooling: {error:#}");
                }
            }
        }
    })
    .detach();
    bind_keys(cx);
}

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new(
            "up",
            SelectPrevious,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new(
            "down",
            SelectNext,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new(
            "home",
            SelectFirst,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new(
            "end",
            SelectLast,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new(
            "left",
            SelectParent,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new(
            "right",
            SelectFirstChild,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new(
            "space",
            ToggleExpanded,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new(
            "enter",
            ActivateSelected,
            Some("GroovyDependenciesPanel && !GroovyDependenciesPanelControls"),
        ),
        KeyBinding::new("cmd-shift-e", ExpandAll, Some("GroovyDependenciesPanel")),
        KeyBinding::new("cmd-shift-c", CollapseAll, Some("GroovyDependenciesPanel")),
        KeyBinding::new("cmd-r", Refresh, Some("GroovyDependenciesPanel")),
    ]);
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use futures::{FutureExt as _, StreamExt as _};
    use gpui::{AppContext as _, Entity, TestAppContext};
    use language::{FakeLspAdapter, LanguageConfig, LanguageMatcher, LoadedLanguage};
    use project::{
        Project,
        jvm_project::JdtlsImportedBuildKind,
        trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees},
    };
    use settings::{SettingsStore, WorktreeId};
    use theme::LoadThemes;
    use workspace::MultiWorkspace;

    use super::*;

    struct WorkspaceLanguageFixture {
        project: Entity<Project>,
        worktree_id: WorktreeId,
        loads: Arc<AtomicUsize>,
        queries: Arc<AtomicUsize>,
        servers: futures::channel::mpsc::UnboundedReceiver<lsp::FakeLanguageServer>,
    }

    impl WorkspaceLanguageFixture {
        fn initialize(product_id: Option<&'static str>, cx: &mut TestAppContext) {
            cx.update(|cx| {
                let settings = SettingsStore::test(cx);
                cx.set_global(settings);
                trusted_worktrees::init(DbTrustedPaths::default(), cx);
                theme_settings::init(LoadThemes::JustBase, cx);
                editor::init(cx);
                match product_id {
                    Some(product_id) => init_for_product(product_id, cx),
                    None => init(cx),
                }
            });
        }

        async fn new(maven: bool, cx: &mut TestAppContext) -> Self {
            let root = if maven {
                "/groovy-default-maven"
            } else {
                "/groovy-default-gradle"
            };
            let root_uri = format!("file://{root}/");
            let module_uri = format!("{root_uri}application/");
            let source_path = format!("{root}/application/src/main/kotlin");
            let fs = project::FakeFs::new(cx.executor());
            let mut tree = serde_json::json!({"application": {"src": {"main": {"kotlin": {"Main.kt": "fun main() {}"}}}}});
            if maven {
                tree["pom.xml"] = "<project/>".into();
                tree["application"]["pom.xml"] = "<project/>".into();
            } else {
                tree["settings.gradle.kts"] = "include(\":application\")".into();
                tree["application"]["build.gradle.kts"] = "plugins { kotlin(\"jvm\") }".into();
            }
            fs.insert_tree(root, tree).await;
            let project = Project::test_with_worktree_trust(fs, [Path::new(root)], cx).await;
            let (languages, worktree_id) = project.read_with(cx, |project, cx| {
                (
                    project.languages().clone(),
                    project
                        .worktrees(cx)
                        .next()
                        .expect("fixture folder")
                        .read(cx)
                        .id(),
                )
            });
            let loads = Arc::new(AtomicUsize::new(0));
            let config = LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            };
            languages.register_language(
                config.name.clone(),
                None,
                config.matcher.clone(),
                false,
                None,
                {
                    let loads = loads.clone();
                    Arc::new(move || {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(LoadedLanguage {
                            config: config.clone(),
                            queries: Default::default(),
                            context_provider: None,
                            toolchain_provider: None,
                            manifest_name: None,
                        })
                    })
                },
            );
            let queries = Arc::new(AtomicUsize::new(0));
            let servers = languages.register_fake_lsp("Java", FakeLspAdapter {
                name: "jdtls",
                capabilities: lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions { commands: vec!["java.project.getAll".into(), "java.project.getSettings".into(), "groovy.maven.importedProjects".into(), "groovy.gradle.importedProjects".into()], ..Default::default() }),
                    ..Default::default()
                },
                initializer: Some(Box::new({
                    let queries = queries.clone();
                    move |server| {
                        server.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                            let queries = queries.clone();
                            let root_uri = root_uri.clone();
                            let module_uri = module_uri.clone();
                            let source_path = source_path.clone();
                            move |params, _| {
                                let queries = queries.clone();
                                let root_uri = root_uri.clone();
                                let module_uri = module_uri.clone();
                                let source_path = source_path.clone();
                                async move {
                                    match params.command.as_str() {
                                        "java.project.getAll" => {
                                            queries.fetch_add(1, Ordering::SeqCst);
                                            Ok(Some(serde_json::json!([module_uri])))
                                        }
                                        "java.project.getSettings" => Ok(Some(serde_json::json!({"org.eclipse.jdt.ls.core.sourcePaths": [source_path], "org.eclipse.jdt.ls.core.classpathEntries": []}))),
                                        "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                            let argument = params.arguments.first().expect("provider scope");
                                            assert_eq!(argument["rootUri"], root_uri);
                                            assert_eq!(argument["cachedOnly"], true);
                                            let selected = (params.command == "groovy.maven.importedProjects") == maven;
                                            Ok(Some(serde_json::json!({"version": 1, "rootUri": root_uri, "projects": if selected { vec![serde_json::json!({"projectUri":module_uri,"javaNature":true})] } else { vec![] }, "partial": false, "fullSynchronizationKnown": false})))
                                        }
                                        _ => anyhow::bail!("Unexpected fixture command"),
                                    }
                                }
                            }
                        });
                    }
                })),
                ..Default::default()
            });
            Self {
                project,
                worktree_id,
                loads,
                queries,
                servers,
            }
        }

        fn mount(&self, cx: &mut TestAppContext) -> Entity<MultiWorkspace> {
            cx.add_window_view(|window, cx| {
                MultiWorkspace::test_new(self.project.clone(), window, cx)
            })
            .0
        }

        fn mount_again(
            &self,
            first: &Entity<MultiWorkspace>,
            cx: &mut TestAppContext,
        ) -> Entity<MultiWorkspace> {
            let workspace = first.read_with(cx, |container, _| container.workspace().clone());
            let app_state = workspace.read_with(cx, |workspace, _| workspace.app_state().clone());
            cx.add_window_view(|window, cx| {
                let workspace =
                    cx.new(|cx| Workspace::new(None, self.project.clone(), app_state, window, cx));
                MultiWorkspace::new(workspace, window, cx)
            })
            .0
        }

        fn trust(&self, cx: &mut TestAppContext) {
            let worktrees = self
                .project
                .read_with(cx, |project, _| project.worktree_store());
            cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"))
                .update(cx, |trusted, cx| {
                    trusted.trust(
                        &worktrees,
                        HashSet::from_iter([PathTrust::Worktree(self.worktree_id)]),
                        cx,
                    )
                });
        }

        fn pump(cx: &mut TestAppContext) {
            for _ in 0..2 {
                cx.run_until_parked();
                cx.executor().advance_clock(Duration::from_millis(300));
            }
            cx.run_until_parked();
        }

        fn server_ids(&self, cx: &TestAppContext) -> Vec<lsp::LanguageServerId> {
            self.project.read_with(cx, |project, cx| {
                project
                    .lsp_store()
                    .read(cx)
                    .language_server_statuses
                    .iter()
                    .filter_map(|(id, status)| {
                        (status.name == lsp::LanguageServerName::new_static("jdtls")).then_some(*id)
                    })
                    .collect()
            })
        }
    }

    #[gpui::test]
    async fn groovy_workspace_attachment_starts_kotlin_only_build_providers(
        cx: &mut TestAppContext,
    ) {
        WorkspaceLanguageFixture::initialize(Some("jvm"), cx);
        for maven in [true, false] {
            let mut fixture = WorkspaceLanguageFixture::new(maven, cx).await;
            let first = fixture.mount(cx);
            WorkspaceLanguageFixture::pump(cx);
            assert!(
                fixture.servers.next().now_or_never().is_none(),
                "no provider binary before trust"
            );
            fixture.trust(cx);
            let server = fixture
                .servers
                .next()
                .await
                .expect("default build provider");
            WorkspaceLanguageFixture::pump(cx);
            let expected_id = server.server.server_id();
            assert_eq!(fixture.server_ids(cx), [expected_id]);
            assert_eq!(fixture.loads.load(Ordering::SeqCst), 1);
            assert_eq!(
                fixture.queries.load(Ordering::SeqCst),
                2,
                "provider metadata without a Java buffer or panel"
            );
            fixture.project.read_with(cx, |project, cx| {
                assert!(project.buffer_store().read(cx).buffers().next().is_none());
                let store = project.lsp_store();
                let (snapshot, current) = store
                    .read(cx)
                    .jdtls_project_snapshot(fixture.worktree_id)
                    .expect("current provider metadata");
                assert!(current);
                assert_eq!(snapshot.projects.len(), 1);
                let module = snapshot.projects.first().expect("provider module");
                assert_eq!(
                    module.imported_build_kind,
                    if maven {
                        JdtlsImportedBuildKind::Maven
                    } else {
                        JdtlsImportedBuildKind::Gradle
                    }
                );
                assert_eq!(module.module_root.path.as_unix_str(), "application");
                assert_eq!(module.source_roots.len(), 1);
            });
            let workspace = first.read_with(cx, |container, _| container.workspace().clone());
            assert!(workspace.read_with(cx, |workspace, cx| {
                workspace.panel::<DependenciesPanel>(cx).is_none()
            }));
            let _second = fixture.mount_again(&first, cx);
            WorkspaceLanguageFixture::pump(cx);
            assert_eq!(fixture.server_ids(cx), [expected_id]);
            assert_eq!(
                fixture.queries.load(Ordering::SeqCst),
                2,
                "repeat attachment retains current import"
            );
            assert!(fixture.servers.next().now_or_never().is_none());
        }
    }

    #[gpui::test]
    async fn copper_workspace_and_panel_actions_do_not_load_java(cx: &mut TestAppContext) {
        WorkspaceLanguageFixture::initialize(Some("rust"), cx);
        let mut fixture = WorkspaceLanguageFixture::new(false, cx).await;
        fixture.trust(cx);
        let (container, cx) = cx.add_window_view(|window, cx| {
            MultiWorkspace::test_new(fixture.project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("mounted shared panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            DependenciesPanel::show_gradle(workspace, &ShowGradle, window, cx);
            DependenciesPanel::show_maven(workspace, &ShowMaven, window, cx);
        });
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_secs(1));
        cx.run_until_parked();
        assert_eq!(fixture.loads.load(Ordering::SeqCst), 0);
        assert_eq!(fixture.queries.load(Ordering::SeqCst), 0);
        assert!(fixture.servers.next().now_or_never().is_none());
        assert!(fixture.server_ids(cx).is_empty());
    }

    #[gpui::test]
    async fn compiled_workspace_identity_uses_only_its_catalog_default(cx: &mut TestAppContext) {
        WorkspaceLanguageFixture::initialize(None, cx);
        let mut fixture = WorkspaceLanguageFixture::new(true, cx).await;
        fixture.trust(cx);
        let _workspace = fixture.mount(cx);
        WorkspaceLanguageFixture::pump(cx);
        if product_flavor::ID == "jvm" {
            let server = fixture
                .servers
                .next()
                .await
                .expect("compiled Groovy startup");
            WorkspaceLanguageFixture::pump(cx);
            assert_eq!(fixture.server_ids(cx), [server.server.server_id()]);
            assert_eq!(fixture.loads.load(Ordering::SeqCst), 1);
            assert_eq!(fixture.queries.load(Ordering::SeqCst), 2);
        } else {
            assert_eq!(fixture.loads.load(Ordering::SeqCst), 0);
            assert!(fixture.server_ids(cx).is_empty());
            assert!(fixture.servers.next().now_or_never().is_none());
        }
    }

    #[gpui::test]
    async fn groovy_collaborative_workspace_cannot_start_client_local_java(
        cx: &mut TestAppContext,
    ) {
        WorkspaceLanguageFixture::initialize(Some("jvm"), cx);
        let mut fixture = WorkspaceLanguageFixture::new(true, cx).await;
        fixture.trust(cx);
        fixture
            .project
            .update(cx, |project, _| project.mark_as_collab_for_testing());
        let _workspace = fixture.mount(cx);
        WorkspaceLanguageFixture::pump(cx);
        assert_eq!(fixture.loads.load(Ordering::SeqCst), 0);
        assert_eq!(fixture.queries.load(Ordering::SeqCst), 0);
        assert!(fixture.servers.next().now_or_never().is_none());
        assert!(fixture.server_ids(cx).is_empty());
    }
}
