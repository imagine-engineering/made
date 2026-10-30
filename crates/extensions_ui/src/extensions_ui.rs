mod components;
mod extension_suggest;
mod extension_version_selector;

use std::sync::OnceLock;
use std::time::Duration;
use std::{any::TypeId, ops::Range, sync::Arc};

use anyhow::Context as _;
use cloud_api_types::{ExtensionMetadata, ExtensionProvides};
use collections::{BTreeMap, BTreeSet};
use command_palette_hooks::CommandPaletteFilter;
use editor::{Editor, EditorElement, EditorStyle};
use extension_host::{ExtensionManifest, ExtensionSettings, ExtensionStore};
use fuzzy::{StringMatch, StringMatchCandidate, match_strings};
use git::{GitHostingProviderRegistry, parse_git_remote_url};
use gpui::{
    Action, App, ClipboardItem, Context, DismissEvent, Entity, EventEmitter, Focusable,
    InteractiveElement, KeyContext, ParentElement, Render, StatefulInteractiveElement, Styled,
    Task, TaskExt, TextStyle, UniformListScrollHandle, WeakEntity, Window, actions, point,
    uniform_list,
};

use picker::{Picker, PickerDelegate};
use project::DirectoryLister;

use schemars::JsonSchema;
use serde::Deserialize;
use settings::{Settings, SettingsContent};
use strum::IntoEnumIterator as _;
use theme_settings::ThemeSettings;
use ui::{
    Banner, ContextMenu, Divider, ListItem, ListItemSpacing, ScrollableHandle, Switch,
    ToggleButtonGroup, ToggleButtonGroupSize, ToggleButtonGroupStyle, ToggleButtonSimple, Tooltip,
    WithScrollbar, prelude::*,
};
use util::ResultExt;
use vim_mode_setting::VimModeSetting;
use workspace::{
    Workspace,
    item::{Item, ItemEvent},
    workspace_error::{ErrorAction, ErrorSeverity, WorkspaceError},
};
use zed_actions::ExtensionCategoryFilter;

use crate::components::{ExtensionCard, extension_provides_label, remote_extension_status};
use crate::extension_version_selector::{
    ExtensionVersionSelector, ExtensionVersionSelectorDelegate,
};

actions!(
    zed,
    [
        /// Installs an extension from a local directory for development.
        InstallDevExtension,
    ]
);

/// Rebuilds an installed dev extension.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, JsonSchema, gpui::Action)]
#[action(namespace = zed)]
#[serde(deny_unknown_fields)]
pub struct RebuildDevExtension {
    /// The ID of the dev extension to rebuild.
    ///
    /// Default: opens a picker if multiple dev extensions are installed.
    #[serde(default)]
    pub extension_id: Option<String>,
}

#[derive(Default)]
struct DevExtensionNotInstalledError {
    extension_id: Option<SharedString>,
}

impl WorkspaceError for DevExtensionNotInstalledError {
    fn primary_message(&self) -> SharedString {
        match &self.extension_id {
            Some(extension_id) => {
                format!("Dev extension '{extension_id}' is not installed.").into()
            }
            None => "No dev extensions are installed.".into(),
        }
    }

    fn primary_action(&self) -> ErrorAction {
        ErrorAction::new("Install Dev Extension", InstallDevExtension)
    }

    fn severity(&self) -> ErrorSeverity {
        ErrorSeverity::Warning
    }
}

fn update_rebuild_dev_extension_visibility(store: &Entity<ExtensionStore>, cx: &mut App) {
    let has_dev_extensions = store.read(cx).dev_extensions().next().is_some();
    CommandPaletteFilter::update_global(cx, |filter, _cx| {
        if has_dev_extensions {
            filter.show_action_types(&[TypeId::of::<RebuildDevExtension>()]);
        } else {
            filter.hide_action_types(&[TypeId::of::<RebuildDevExtension>()]);
        }
    });
}

pub fn init(cx: &mut App) {
    let store = ExtensionStore::global(cx);
    update_rebuild_dev_extension_visibility(&store, cx);
    cx.observe(&store, |store, cx| {
        update_rebuild_dev_extension_visibility(&store, cx);
    })
    .detach();

    cx.observe_new(move |workspace: &mut Workspace, window, cx| {
        let Some(window) = window else {
            return;
        };
        workspace
            .register_action(
                move |workspace, action: &zed_actions::Extensions, window, cx| {
                    let provides_filter = action.category_filter.map(|category| match category {
                        ExtensionCategoryFilter::Themes => ExtensionProvides::Themes,
                        ExtensionCategoryFilter::IconThemes => ExtensionProvides::IconThemes,
                        ExtensionCategoryFilter::Languages => ExtensionProvides::Languages,
                        ExtensionCategoryFilter::Grammars => ExtensionProvides::Grammars,
                        ExtensionCategoryFilter::LanguageServers => {
                            ExtensionProvides::LanguageServers
                        }
                        ExtensionCategoryFilter::ContextServers => {
                            ExtensionProvides::ContextServers
                        }
                        ExtensionCategoryFilter::Snippets => ExtensionProvides::Snippets,
                        ExtensionCategoryFilter::DebugAdapters => ExtensionProvides::DebugAdapters,
                    });

                    let existing = workspace
                        .active_pane()
                        .read(cx)
                        .items()
                        .find_map(|item| item.downcast::<ExtensionsPage>());

                    if let Some(existing) = existing {
                        existing.update(cx, |extensions_page, cx| {
                            if provides_filter.is_some() {
                                extensions_page.change_provides_filter(provides_filter, cx);
                            }
                            if let Some(id) = action.id.as_ref() {
                                extensions_page.focus_extension(id, window, cx);
                            }
                        });

                        workspace.activate_item(&existing, true, true, window, cx);
                    } else {
                        let extensions_page = ExtensionsPage::new(
                            workspace,
                            provides_filter,
                            action.id.as_deref(),
                            window,
                            cx,
                        );
                        workspace.add_item_to_active_pane(
                            Box::new(extensions_page),
                            None,
                            true,
                            window,
                            cx,
                        )
                    }
                },
            )
            .register_action(move |workspace, _: &InstallDevExtension, window, cx| {
                let store = ExtensionStore::global(cx);
                let prompt = workspace.prompt_for_open_path(
                    gpui::PathPromptOptions {
                        files: false,
                        directories: true,
                        multiple: false,
                        prompt: None,
                    },
                    DirectoryLister::Local(
                        workspace.project().clone(),
                        workspace.app_state().fs.clone(),
                    ),
                    window,
                    cx,
                );

                let workspace_handle = cx.entity().downgrade();
                window
                    .spawn(cx, async move |cx| {
                        let extension_path = match prompt.await.map_err(anyhow::Error::from) {
                            Ok(Some(mut paths)) => paths.pop()?,
                            Ok(None) => return None,
                            Err(err) => {
                                workspace_handle
                                    .update(cx, |workspace, cx| {
                                        workspace.show_error(
                                            workspace::workspace_error::PortalError::new(
                                                err.to_string(),
                                            ),
                                            cx,
                                        );
                                    })
                                    .ok();
                                return None;
                            }
                        };

                        let install_task = store.update(cx, |store, cx| {
                            store.install_dev_extension(extension_path, cx)
                        });

                        match install_task.await {
                            Ok(_) => {}
                            Err(err) => {
                                log::error!("Failed to install dev extension: {:?}", err);
                                workspace_handle
                                    .update(cx, |workspace, cx| {
                                        // NOTE: using `anyhow::context` here ends up not printing
                                        // the error
                                        workspace.show_error(
                                            format!("Failed to install dev extension: {}", err),
                                            cx,
                                        );
                                    })
                                    .ok();
                            }
                        }

                        Some(())
                    })
                    .detach();
            })
            .register_action(move |workspace, action: &RebuildDevExtension, window, cx| {
                if let Some(target_id) = action.extension_id.as_deref() {
                    let extension_id = ExtensionStore::global(cx)
                        .read(cx)
                        .dev_extensions()
                        .find_map(|m| {
                            if m.id.as_ref() == target_id {
                                Some(m.id.clone())
                            } else {
                                None
                            }
                        });
                    if let Some(extension_id) = extension_id {
                        ExtensionStore::global(cx).update(cx, |store, cx| {
                            store.rebuild_dev_extension(extension_id, cx);
                        });
                    } else {
                        workspace.show_error(
                            DevExtensionNotInstalledError {
                                extension_id: Some(SharedString::from(target_id.to_owned())),
                            },
                            cx,
                        );
                    }
                    return;
                }

                let dev_extensions = ExtensionStore::global(cx)
                    .read(cx)
                    .dev_extensions()
                    .cloned()
                    .collect::<Vec<_>>();

                match dev_extensions.len() {
                    0 => {
                        workspace.show_error(DevExtensionNotInstalledError::default(), cx);
                    }
                    1 => {
                        let extension_id = dev_extensions[0].id.clone();
                        ExtensionStore::global(cx).update(cx, |store, cx| {
                            store.rebuild_dev_extension(extension_id, cx);
                        });
                    }
                    _ => {
                        workspace.toggle_modal(window, cx, |window, cx| {
                            let delegate = DevExtensionRebuildPickerDelegate::new(dev_extensions);
                            Picker::uniform_list(delegate, window, cx)
                        });
                    }
                }
            });

        cx.subscribe_in(workspace.project(), window, |_, _, event, window, cx| {
            if let project::Event::LanguageNotFound(buffer) = event {
                extension_suggest::suggest(buffer.clone(), window, cx);
            }
        })
        .detach();
    })
    .detach();
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
enum ExtensionFilter {
    All,
    Installed,
    NotInstalled,
}

impl ExtensionFilter {
    pub fn include_dev_extensions(&self) -> bool {
        match self {
            Self::All | Self::Installed => true,
            Self::NotInstalled => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
enum Feature {
    AgentClaude,
    AgentCodex,
    AgentGemini,
    ExtensionBasedpyright,
    ExtensionRuff,
    ExtensionTailwind,
    ExtensionTy,
    Git,
    LanguageBash,
    LanguageC,
    LanguageCpp,
    LanguageGo,
    LanguagePython,
    LanguageReact,
    LanguageRust,
    LanguageTypescript,
    OpenIn,
    Vim,
}

fn keywords_by_feature() -> &'static BTreeMap<Feature, Vec<&'static str>> {
    static KEYWORDS_BY_FEATURE: OnceLock<BTreeMap<Feature, Vec<&'static str>>> = OnceLock::new();
    KEYWORDS_BY_FEATURE.get_or_init(|| {
        BTreeMap::from_iter([
            (
                Feature::AgentClaude,
                vec!["claude", "claude code", "claude agent"],
            ),
            (Feature::AgentCodex, vec!["codex", "codex cli"]),
            (Feature::AgentGemini, vec!["gemini", "gemini cli"]),
            (
                Feature::ExtensionBasedpyright,
                vec!["basedpyright", "pyright"],
            ),
            (Feature::ExtensionRuff, vec!["ruff"]),
            (Feature::ExtensionTailwind, vec!["tail", "tailwind"]),
            (Feature::ExtensionTy, vec!["ty"]),
            (Feature::Git, vec!["git"]),
            (Feature::LanguageBash, vec!["sh", "bash"]),
            (Feature::LanguageC, vec!["c", "clang"]),
            (Feature::LanguageCpp, vec!["c++", "cpp", "clang"]),
            (Feature::LanguageGo, vec!["go", "golang"]),
            (Feature::LanguagePython, vec!["python", "py"]),
            (Feature::LanguageReact, vec!["react"]),
            (Feature::LanguageRust, vec!["rust", "rs"]),
            (
                Feature::LanguageTypescript,
                vec!["type", "typescript", "ts"],
            ),
            (
                Feature::OpenIn,
                vec![
                    "github",
                    "gitlab",
                    "bitbucket",
                    "codeberg",
                    "sourcehut",
                    "permalink",
                    "link",
                    "open in",
                ],
            ),
            (Feature::Vim, vec!["vim"]),
        ])
    })
}

#[cfg(feature = "agentic")]
#[derive(Default)]
struct PackageManagement {
    expanded: bool,
    preview: Option<(
        Option<String>,
        extension_host::agent_plugins::PackageSnapshot,
    )>,
    removal: Option<String>,
    error: Option<String>,
    task: Option<Task<()>>,
}

#[cfg(all(test, feature = "agentic"))]
mod package_tests {
    use super::*;
    use extension_host::agent_plugins::{MANIFEST_SCHEMA, PackageChange};
    use fs::Fs as _;
    use gpui::{TestAppContext, UpdateGlobal as _};
    use project::{FakeFs, Project};
    use std::path::Path;

    #[gpui::test]
    async fn plugin_desktop_inspection_cancel_install_update_and_uninstall(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            settings::SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |content| {
                    content
                        .extension
                        .auto_install_extensions
                        .insert("html".into(), false);
                });
            });
            cx.set_global(db::AppDatabase::test_new());
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
            editor::init(cx);
            extension::init(cx);
            gpui_tokio::init(cx);
        });
        let filesystem = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(filesystem.clone(), cx));
        filesystem
            .insert_tree("/project", serde_json::json!({}))
            .await;
        filesystem.insert_tree("/package", serde_json::json!({"plugin.json": serde_json::json!({"$schema": MANIFEST_SCHEMA, "name": "desktop-package"}).to_string()})).await;
        filesystem
            .insert_tree("/invalid", serde_json::json!({"plugin.json": "not json"}))
            .await;
        let project = Project::test(filesystem.clone(), [Path::new("/project")], cx).await;
        let (workspace, cx) =
            cx.add_window_view(|window, cx| Workspace::test_new(project.clone(), window, cx));
        workspace.update(cx, |workspace, cx| {
            let state = workspace.app_state();
            extension_host::init(
                Arc::new(extension::ExtensionHostProxy::new()),
                filesystem.clone(),
                state.client.clone(),
                state.node_runtime.clone(),
                cx,
            );
        });
        cx.executor().advance_clock(Duration::from_millis(200));
        cx.run_until_parked();
        let store = cx.update(|_, cx| ExtensionStore::global(cx));
        let page = workspace.update_in(cx, |workspace, window, cx| {
            let page = ExtensionsPage::new(workspace, None, None, window, cx);
            workspace.add_item_to_active_pane(Box::new(page.clone()), None, true, window, cx);
            page
        });
        cx.simulate_resize(gpui::size(px(1400.), px(1100.)));
        page.update_in(cx, |page, window, cx| {
            page.inspect_package(None, window, cx)
        });
        cx.simulate_path_prompt_response(|_| None);
        cx.run_until_parked();
        assert!(page.read_with(cx, |page, _| page.packages.preview.is_none()
            && page.packages.error.is_none()));
        page.update_in(cx, |page, window, cx| {
            page.inspect_package(None, window, cx)
        });
        cx.simulate_path_prompt_response(|_| Some(vec!["/invalid".into()]));
        cx.run_until_parked();
        assert!(page.read_with(cx, |page, _| page.packages.error.is_some()));
        page.update_in(cx, |page, window, cx| {
            page.inspect_package(None, window, cx)
        });
        cx.simulate_path_prompt_response(|_| Some(vec!["/package".into()]));
        cx.run_until_parked();
        assert!(store.read_with(cx, |store, _| store.agent_plugins.is_empty()));
        let cancel = cx
            .debug_bounds("agent-plugin-cancel")
            .expect("review cancel button");
        cx.simulate_click(cancel.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        assert!(page.read_with(cx, |page, _| page.packages.preview.is_none()));
        assert!(store.read_with(cx, |store, _| store.agent_plugins.is_empty()));
        page.update_in(cx, |page, window, cx| {
            page.inspect_package(None, window, cx)
        });
        cx.simulate_path_prompt_response(|_| Some(vec!["/package".into()]));
        cx.run_until_parked();
        let confirm = cx
            .debug_bounds("agent-plugin-confirm")
            .expect("review install button");
        cx.simulate_click(confirm.center(), gpui::Modifiers::default());
        page.update(cx, |page, cx| page.confirm_package(cx));
        cx.run_until_parked();
        let record = store.read_with(cx, |store, _| {
            assert_eq!(store.agent_plugins.len(), 1);
            store
                .agent_plugins
                .values()
                .next()
                .expect("installed")
                .clone()
        });
        assert!(record.approved_servers.is_empty());
        filesystem
            .write(Path::new("/package/asset"), b"update")
            .await
            .expect("change source");
        page.update_in(cx, |page, window, cx| {
            page.inspect_package(Some(record.id.clone()), window, cx)
        });
        cx.run_until_parked();
        page.update(cx, |page, cx| page.confirm_package(cx));
        cx.run_until_parked();
        assert_ne!(
            record.revision,
            store.read_with(cx, |store, _| store
                .agent_plugins
                .get(&record.id)
                .expect("updated")
                .revision
                .clone())
        );
        page.update(cx, |page, cx| {
            page.apply_package_change(record.id.clone(), PackageChange::Enabled(false), cx)
        });
        cx.run_until_parked();
        assert!(!store.read_with(cx, |store, _| {
            store
                .agent_plugins
                .get(&record.id)
                .expect("disabled")
                .enabled
        }));
        let data = store.read_with(cx, |store, _| {
            store.agent_plugins_directory.join("data").join(&record.id)
        });
        filesystem.create_dir(&data).await.expect("data directory");
        filesystem
            .write(&data.join("retained"), b"data")
            .await
            .expect("package data");
        page.update(cx, |page, cx| {
            page.apply_package_change(
                record.id.clone(),
                PackageChange::Uninstall { delete_data: false },
                cx,
            )
        });
        cx.run_until_parked();
        assert!(store.read_with(cx, |store, _| store.agent_plugins.is_empty()));
        assert_eq!(
            filesystem
                .load(&data.join("retained"))
                .await
                .expect("retained data"),
            "data"
        );
    }
}

pub struct ExtensionsPage {
    workspace: WeakEntity<Workspace>,
    provider_registry: Arc<GitHostingProviderRegistry>,
    list: UniformListScrollHandle,
    is_fetching_extensions: bool,
    fetch_failed: bool,
    filter: ExtensionFilter,
    remote_extension_entries: Vec<ExtensionMetadata>,
    dev_extension_entries: Vec<Arc<ExtensionManifest>>,
    filtered_remote_extension_indices: Vec<usize>,
    filtered_dev_extension_indices: Vec<usize>,
    query_editor: Entity<Editor>,
    query_contains_error: bool,
    provides_filter: Option<ExtensionProvides>,
    _subscriptions: Vec<gpui::Subscription>,
    extension_fetch_task: Option<Task<()>>,
    upsells: BTreeSet<Feature>,
    #[cfg(feature = "agentic")]
    packages: PackageManagement,
}

#[cfg(feature = "agentic")]
impl ExtensionsPage {
    fn inspect_package(
        &mut self,
        update: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.packages.task.is_some() {
            return;
        }
        let store = ExtensionStore::global(cx);
        let filesystem = store.read(cx).fs.clone();
        let source = update.as_ref().and_then(|id| {
            store
                .read(cx)
                .agent_plugins
                .get(id)
                .map(|record| record.source.clone())
        });
        let prompt = if source.is_none() {
            let Some(workspace) = self.workspace.upgrade() else {
                return;
            };
            Some(workspace.update(cx, |workspace, cx| {
                workspace.prompt_for_open_path(
                    gpui::PathPromptOptions {
                        files: false,
                        directories: true,
                        multiple: false,
                        prompt: Some("Inspect Agent Plugin".into()),
                    },
                    DirectoryLister::Local(workspace.project().clone(), filesystem.clone()),
                    window,
                    cx,
                )
            }))
        } else {
            None
        };
        self.packages.expanded = true;
        self.packages.error = None;
        self.packages.preview = None;
        self.packages.task = Some(cx.spawn(async move |this, cx| {
            let result = async {
                let source = match source {
                    Some(source) => Some(source),
                    None => match prompt.context("Missing directory prompt")?.await? {
                        Some(mut paths) => paths.pop(),
                        None => None,
                    },
                };
                let Some(source) = source else {
                    return anyhow::Ok(None);
                };
                let snapshot = extension_host::agent_plugins::capture_package(
                    filesystem.as_ref(),
                    &source,
                    Default::default(),
                )
                .await?;
                Ok(Some((update, snapshot)))
            }
            .await;
            this.update(cx, |this, cx| {
                this.packages.task = None;
                match result {
                    Ok(preview) => this.packages.preview = preview,
                    Err(error) => this.packages.error = Some(format!("{error:#}")),
                }
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn apply_package_change(
        &mut self,
        id: String,
        change: extension_host::agent_plugins::PackageChange,
        cx: &mut Context<Self>,
    ) {
        if self.packages.task.is_some() {
            return;
        }
        let task = ExtensionStore::global(cx)
            .update(cx, |store, cx| store.change_agent_plugin(&id, change, cx));
        self.await_package_change(task, cx);
    }

    fn confirm_package(&mut self, cx: &mut Context<Self>) {
        if self.packages.task.is_some() {
            return;
        }
        let Some((id, snapshot)) = self.packages.preview.take() else {
            return;
        };
        let task = ExtensionStore::global(cx).update(cx, |store, cx| match id {
            Some(id) => store.change_agent_plugin(
                &id,
                extension_host::agent_plugins::PackageChange::Update(snapshot),
                cx,
            ),
            None => store.install_agent_plugin(snapshot, cx),
        });
        self.await_package_change(task, cx);
    }

    fn await_package_change(&mut self, task: Task<anyhow::Result<()>>, cx: &mut Context<Self>) {
        self.packages.error = None;
        self.packages.removal = None;
        self.packages.task = Some(cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| {
                this.packages.task = None;
                if let Err(error) = result {
                    this.packages.error = Some(format!("{error:#}"));
                }
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn render_packages(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        use extension_host::agent_plugins::{PackageChange, ServerDeclaration};
        let store = ExtensionStore::global(cx);
        let owner = store.read(cx);
        let busy = self.packages.task.is_some() || owner.agent_plugins_busy;
        let records = owner.agent_plugins.values().cloned().collect::<Vec<_>>();
        let diagnostics = owner.agent_plugins_diagnostics.clone();
        let error = self
            .packages
            .error
            .clone()
            .or_else(|| owner.agent_plugins_error.clone());
        let connections = self
            .workspace
            .read_with(cx, |workspace, cx| {
                workspace.project().read(cx).context_server_store()
            })
            .ok();
        v_flex().px_4().py_2().gap_2()
            .child(h_flex().gap_2()
                .child(Button::new("agent-plugins-toggle", "Agent Plugins").tab_index(0_isize)
                    .on_click(cx.listener(|this, _, _, cx| { this.packages.expanded = !this.packages.expanded; cx.notify(); })))
                .child(Button::new("agent-plugin-inspect", "Install Local Package…").tab_index(0_isize).disabled(busy)
                    .on_click(cx.listener(|this, _, window, cx| this.inspect_package(None, window, cx)))))
            .when(self.packages.expanded, |element| element.child(
                v_flex().id("agent-plugins-list").max_h(rems(24.)).overflow_y_scroll().gap_3()
                    .child(Label::new("Portable skills and MCP servers. Installing a package does not approve server execution.").color(Color::Muted))
                    .when(busy, |element| element.child(Label::new("Processing package…")))
                    .when_some(error, |element, error| element.child(Label::new(error).color(Color::Error)))
                    .when_some(self.packages.preview.as_ref(), |element, (id, snapshot)| {
                        element.child(v_flex().gap_1()
                            .child(Label::new(format!("Review {} {}", snapshot.manifest.name, snapshot.manifest.version.as_deref().unwrap_or("(unversioned)"))))
                            .child(Label::new(format!("Source: {}", snapshot.source.display())))
                            .child(Label::new(format!("Revision: {}", snapshot.revision)))
                            .children(snapshot.manifest.description.clone().map(Label::new))
                            .children(snapshot.files.keys().filter(|path| path.starts_with("skills") && path.file_name().is_some_and(|name| name == "SKILL.md")).map(|path| Label::new(format!("Skill: {}", path.display()))))
                            .children(snapshot.servers.iter().map(|(name, server)| Label::new(format!("MCP: {name} ({}) — requires approval", match server { ServerDeclaration::Stdio {..} => "local process", ServerDeclaration::StreamableHttp {..} => "Streamable HTTP", ServerDeclaration::Sse {..} => "unsupported SSE" }))))
                            .children(snapshot.diagnostics.iter().cloned().map(|message| Label::new(message).color(Color::Warning)))
                            .child(h_flex().gap_2()
                                .child(div().debug_selector(|| "agent-plugin-confirm".into()).child(Button::new("agent-plugin-confirm", if id.is_some() { "Apply Reviewed Update" } else { "Install Reviewed Package" }).tab_index(0_isize).disabled(busy).on_click(cx.listener(|this, _, _, cx| this.confirm_package(cx)))))
                                .child(div().debug_selector(|| "agent-plugin-cancel".into()).child(Button::new("agent-plugin-cancel", "Cancel").tab_index(0_isize).disabled(busy).on_click(cx.listener(|this, _, _, cx| { this.packages.preview = None; cx.notify(); }))))))
                    })
                    .children(records.into_iter().map(|record| {
                        let id = record.id.clone();
                        let update_id = id.clone();
                        let remove_id = id.clone();
                        let enabled = record.enabled;
                        v_flex().gap_1().border_b_1().border_color(cx.theme().colors().border_variant).pb_2()
                            .child(Label::new(format!("{} {} — {}", record.manifest.name, record.manifest.version.as_deref().unwrap_or(""), if enabled { "Enabled" } else { "Disabled" })))
                            .child(Label::new(format!("Revision: {}", record.revision)).color(Color::Muted))
                            .children(record.diagnostics.iter().chain(diagnostics.get(&record.id).into_iter().flatten()).cloned().map(|message| Label::new(message).color(Color::Warning)))
                            .children(record.servers.iter().map(|(name, server)| {
                                let action = zed_actions::agent::ConfigurePluginServer { package_id: id.clone(), server_name: name.clone() };
                                let unsupported = matches!(server, ServerDeclaration::Sse { .. });
                                let status = if unsupported { "Unsupported SSE".to_owned() }
                                    else if !record.approved_servers.contains(name) { "Disabled — approval required".to_owned() }
                                    else {
                                        use project::context_server_store::ContextServerStatus;
                                        match connections.as_ref().and_then(|store| store.read(cx).status_for_server(&context_server::ContextServerId(record.server_id(name).into()))) {
                                            Some(ContextServerStatus::Running) => "Connected".into(),
                                            Some(ContextServerStatus::Starting) => "Connecting".into(),
                                            Some(ContextServerStatus::AuthRequired) => "Sign in required".into(),
                                            Some(ContextServerStatus::ClientSecretRequired { .. }) => "Client secret required".into(),
                                            Some(ContextServerStatus::Authenticating) => "Signing in".into(),
                                            Some(ContextServerStatus::Error(error)) => format!("Error: {error}"),
                                            Some(ContextServerStatus::Stopped) => "Stopped".into(),
                                            None => "Not connected — review server settings".into(),
                                        }
                                    };
                                h_flex().gap_2()
                                    .child(Label::new(format!("{name}: {status}")))
                                    .child(Button::new(SharedString::from(format!("package-server-{}-{name}", id)), "Review Server").tab_index(0_isize).disabled(busy || unsupported)
                                        .on_click(move |_, window, cx| window.dispatch_action(Box::new(action.clone()), cx)))
                            }))
                            .child(h_flex().gap_2()
                                .child(Button::new(SharedString::from(format!("package-update-{id}")), "Inspect Update").tab_index(0_isize).disabled(busy).on_click(cx.listener(move |this, _, window, cx| this.inspect_package(Some(update_id.clone()), window, cx))))
                                .child(Button::new(SharedString::from(format!("package-enable-{id}")), if enabled { "Disable" } else { "Enable Skills" }).tab_index(0_isize).disabled(busy).on_click(cx.listener(move |this, _, _, cx| this.apply_package_change(id.clone(), PackageChange::Enabled(!enabled), cx))))
                                .child(Button::new(SharedString::from(format!("package-remove-{remove_id}")), "Uninstall…").tab_index(0_isize).disabled(busy).on_click(cx.listener(move |this, _, _, cx| { this.packages.removal = Some(remove_id.clone()); cx.notify(); }))))
                    }))
                    .when_some(self.packages.removal.clone(), |element, id| {
                        let delete_id = id.clone();
                        element.child(v_flex().gap_1()
                            .child(Label::new("Uninstall this package? Retain its data unless you explicitly delete it. Shared account credentials are preserved."))
                            .child(h_flex().gap_2()
                                .child(Button::new("package-remove-keep", "Uninstall and Keep Data").tab_index(0_isize).disabled(busy).on_click(cx.listener(move |this, _, _, cx| this.apply_package_change(id.clone(), PackageChange::Uninstall { delete_data: false }, cx))))
                                .child(Button::new("package-remove-delete", "Uninstall and Delete Data").tab_index(0_isize).disabled(busy).on_click(cx.listener(move |this, _, _, cx| this.apply_package_change(delete_id.clone(), PackageChange::Uninstall { delete_data: true }, cx))))
                                .child(Button::new("package-remove-cancel", "Cancel").tab_index(0_isize).on_click(cx.listener(|this, _, _, cx| { this.packages.removal = None; cx.notify(); })))))
                    })
            ))
    }
}

impl ExtensionsPage {
    pub fn new(
        workspace: &Workspace,
        provides_filter: Option<ExtensionProvides>,
        focus_extension_id: Option<&str>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        cx.new(|cx| {
            let store = ExtensionStore::global(cx);
            let workspace_handle = workspace.weak_handle();
            let subscriptions = vec![
                cx.observe(&store, |_: &mut Self, _, cx| cx.notify()),
                cx.subscribe_in(
                    &store,
                    window,
                    move |this, _, event, window, cx| match event {
                        extension_host::Event::ExtensionsUpdated => {
                            this.fetch_extensions_debounced(None, cx)
                        }
                        extension_host::Event::ExtensionInstalled(extension_id) => this
                            .on_extension_installed(
                                workspace_handle.clone(),
                                extension_id,
                                window,
                                cx,
                            ),
                        _ => {}
                    },
                ),
            ];

            #[cfg(feature = "agentic")]
            let mut subscriptions = subscriptions;
            #[cfg(feature = "agentic")]
            subscriptions.push(cx.observe(
                &workspace.project().read(cx).context_server_store(),
                |_: &mut Self, _, cx| cx.notify(),
            ));

            let query_editor = cx.new(|cx| {
                let mut input = Editor::single_line(window, cx);
                input.set_placeholder_text("Search extensions...", window, cx);
                if let Some(id) = focus_extension_id {
                    input.set_text(format!("id:{id}"), window, cx);
                }
                input
            });
            cx.subscribe(&query_editor, Self::on_query_change).detach();

            let scroll_handle = UniformListScrollHandle::new();
            let provider_registry = GitHostingProviderRegistry::default_global(cx);

            let mut this = Self {
                workspace: workspace.weak_handle(),
                provider_registry,
                list: scroll_handle,
                is_fetching_extensions: false,
                fetch_failed: false,
                filter: ExtensionFilter::All,
                dev_extension_entries: Vec::new(),
                filtered_remote_extension_indices: Vec::new(),
                filtered_dev_extension_indices: Vec::new(),
                remote_extension_entries: Vec::new(),
                query_contains_error: false,
                provides_filter,
                extension_fetch_task: None,
                _subscriptions: subscriptions,
                query_editor,
                upsells: BTreeSet::default(),
                #[cfg(feature = "agentic")]
                packages: PackageManagement::default(),
            };
            this.fetch_extensions(
                this.search_query(cx),
                Some(BTreeSet::from_iter(this.provides_filter)),
                None,
                cx,
            );
            this
        })
    }

    fn get_repository_icon(&self, repository_url: &str) -> IconName {
        parse_git_remote_url(Arc::clone(&self.provider_registry), repository_url)
            .map(|(provider, _)| ui::git_hosting_provider_icon(provider.name().as_str()))
            .unwrap_or(IconName::Link)
    }

    fn on_extension_installed(
        &mut self,
        workspace: WeakEntity<Workspace>,
        extension_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let extension_store = ExtensionStore::global(cx).read(cx);
        let themes = extension_store
            .extension_themes(extension_id)
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        if !themes.is_empty() {
            workspace
                .update(cx, |_workspace, cx| {
                    window.dispatch_action(
                        zed_actions::theme_selector::Toggle {
                            themes_filter: Some(themes),
                        }
                        .boxed_clone(),
                        cx,
                    );
                })
                .ok();
            return;
        }

        let icon_themes = extension_store
            .extension_icon_themes(extension_id)
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        if !icon_themes.is_empty() {
            workspace
                .update(cx, |_workspace, cx| {
                    window.dispatch_action(
                        zed_actions::icon_theme_selector::Toggle {
                            themes_filter: Some(icon_themes),
                        }
                        .boxed_clone(),
                        cx,
                    );
                })
                .ok();
        }
    }

    fn filter_extension_entries(&mut self, cx: &mut Context<Self>) {
        self.filtered_remote_extension_indices.clear();
        self.filtered_remote_extension_indices.extend(
            self.remote_extension_entries
                .iter()
                .enumerate()
                .filter(|(_, extension)| {
                    if self.filter == ExtensionFilter::All {
                        return true;
                    }

                    let status = remote_extension_status(&extension.id, cx);
                    match self.filter {
                        ExtensionFilter::Installed => status.is_installed(),
                        ExtensionFilter::NotInstalled => !status.is_installed(),
                        ExtensionFilter::All => true,
                    }
                })
                .filter(|(_, extension)| match self.provides_filter {
                    Some(provides) => extension.manifest.provides.contains(&provides),
                    None => true,
                })
                .map(|(ix, _)| ix),
        );

        self.filtered_dev_extension_indices.clear();
        self.filtered_dev_extension_indices.extend(
            self.dev_extension_entries
                .iter()
                .enumerate()
                .filter(|(_, manifest)| match self.provides_filter {
                    Some(provides) => manifest.provides().contains(&provides),
                    None => true,
                })
                .map(|(ix, _)| ix),
        );

        cx.notify();
    }

    fn scroll_to_top(&mut self, cx: &mut Context<Self>) {
        self.list.set_offset(point(px(0.), px(0.)));
        cx.notify();
    }

    fn fetch_extensions(
        &mut self,
        search: Option<String>,
        provides_filter: Option<BTreeSet<ExtensionProvides>>,
        on_complete: Option<Box<dyn FnOnce(&mut Self, &mut Context<Self>) + Send>>,
        cx: &mut Context<Self>,
    ) {
        self.is_fetching_extensions = true;
        self.fetch_failed = false;
        cx.notify();

        let extension_store = ExtensionStore::global(cx);

        let dev_extensions = extension_store
            .read(cx)
            .dev_extensions()
            .cloned()
            .collect::<Vec<_>>();

        let remote_extensions =
            if let Some(id) = search.as_ref().and_then(|s| s.strip_prefix("id:")) {
                let versions =
                    extension_store.update(cx, |store, cx| store.fetch_extension_versions(id, cx));
                cx.foreground_executor().spawn(async move {
                    let versions = versions.await?;
                    let latest = versions
                        .into_iter()
                        .max_by_key(|v| v.published_at)
                        .context("no extension found")?;
                    Ok(vec![latest])
                })
            } else {
                extension_store.update(cx, |store, cx| {
                    store.fetch_extensions(search.as_deref(), provides_filter.as_ref(), cx)
                })
            };

        cx.spawn(async move |this, cx| {
            let dev_extensions = if let Some(search) = search {
                let match_candidates = dev_extensions
                    .iter()
                    .enumerate()
                    .map(|(ix, manifest)| StringMatchCandidate::new(ix, &manifest.name))
                    .collect::<Vec<_>>();

                let matches = match_strings(
                    &match_candidates,
                    &search,
                    false,
                    true,
                    match_candidates.len(),
                    &Default::default(),
                    cx.background_executor().clone(),
                )
                .await;
                matches
                    .into_iter()
                    .map(|mat| dev_extensions[mat.candidate_id].clone())
                    .collect()
            } else {
                dev_extensions
            };

            let fetch_result = remote_extensions.await;

            let result = this.update(cx, |this, cx| {
                cx.notify();
                this.dev_extension_entries = dev_extensions;
                this.is_fetching_extensions = false;

                match fetch_result {
                    Ok(extensions) => {
                        this.fetch_failed = false;
                        this.remote_extension_entries = extensions;
                        this.filter_extension_entries(cx);
                        if let Some(callback) = on_complete {
                            callback(this, cx);
                        }
                        Ok(())
                    }
                    Err(err) => {
                        this.fetch_failed = true;
                        this.filter_extension_entries(cx);
                        Err(err)
                    }
                }
            });

            result?
        })
        .detach_and_log_err(cx);
    }

    fn render_extensions(
        &mut self,
        range: Range<usize>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<ExtensionCard> {
        let dev_extension_entries_len = if self.filter.include_dev_extensions() {
            self.filtered_dev_extension_indices.len()
        } else {
            0
        };
        range
            .map(|ix| {
                if ix < dev_extension_entries_len {
                    let dev_ix = self.filtered_dev_extension_indices[ix];
                    let extension = &self.dev_extension_entries[dev_ix];
                    let repository_icon = extension
                        .repository
                        .as_deref()
                        .map(|url| self.get_repository_icon(url));
                    let card = ExtensionCard::for_dev(extension.clone(), cx);
                    if let Some(icon) = repository_icon {
                        card.repository_icon(icon)
                    } else {
                        card
                    }
                } else {
                    let extension_ix =
                        self.filtered_remote_extension_indices[ix - dev_extension_entries_len];
                    let extension = &self.remote_extension_entries[extension_ix];
                    self.render_remote_extension(extension, cx)
                }
            })
            .collect()
    }

    fn render_remote_extension(
        &self,
        extension: &ExtensionMetadata,
        cx: &mut Context<Self>,
    ) -> ExtensionCard {
        let repository_icon = self.get_repository_icon(&extension.manifest.repository);
        let card = ExtensionCard::for_remote(extension, cx).repository_icon(repository_icon);
        let this = cx.weak_entity();

        card.context_menu(move |extension_id, authors, window, cx| {
            let this = this.upgrade()?;
            Some(ContextMenu::build(window, cx, |context_menu, window, _| {
                context_menu
                    .entry(
                        "Install Another Version...",
                        None,
                        window.handler_for(&this, {
                            let extension_id = extension_id.clone();
                            move |this, window, cx| {
                                this.show_extension_version_list(extension_id.clone(), window, cx)
                            }
                        }),
                    )
                    .entry("Copy Extension ID", None, {
                        let extension_id = extension_id.clone();
                        move |_, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(
                                extension_id.to_string(),
                            ));
                        }
                    })
                    .entry("Copy Author Info", None, move |_, cx| {
                        cx.write_to_clipboard(ClipboardItem::new_string(authors.to_string()));
                    })
            }))
        })
    }

    fn show_extension_version_list(
        &mut self,
        extension_id: Arc<str>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };

        cx.spawn_in(window, async move |this, cx| {
            let extension_versions_task = this.update(cx, |_, cx| {
                let extension_store = ExtensionStore::global(cx);

                extension_store.update(cx, |store, cx| {
                    store.fetch_extension_versions(&extension_id, cx)
                })
            })?;

            let extension_versions = extension_versions_task.await?;

            workspace.update_in(cx, |workspace, window, cx| {
                let fs = workspace.project().read(cx).fs().clone();
                workspace.toggle_modal(window, cx, |window, cx| {
                    let delegate = ExtensionVersionSelectorDelegate::new(
                        fs,
                        cx.entity().downgrade(),
                        extension_versions,
                    );

                    ExtensionVersionSelector::new(delegate, window, cx)
                });
            })?;

            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn render_search(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut key_context = KeyContext::new_with_defaults();
        key_context.add("BufferSearchBar");

        let editor_border = if self.query_contains_error {
            Color::Error.color(cx)
        } else {
            cx.theme().colors().border
        };

        h_flex()
            .id("extensions-search")
            .role(gpui::Role::SearchInput)
            .aria_label("Search extensions...")
            .aria_value(self.query_editor.read(cx).text(cx))
            .track_focus(&self.query_editor.read(cx).focus_handle(cx))
            .key_context(key_context)
            .h_8()
            .min_w(rems_from_px(384_f32))
            .flex_1()
            .pl_1p5()
            .pr_2()
            .gap_2()
            .border_1()
            .border_color(editor_border)
            .rounded_md()
            .child(Icon::new(IconName::MagnifyingGlass).color(Color::Muted))
            .child(self.render_text_input(&self.query_editor, cx))
    }

    fn render_text_input(
        &self,
        editor: &Entity<Editor>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let settings = ThemeSettings::get_global(cx);
        let text_style = TextStyle {
            color: if editor.read(cx).read_only(cx) {
                cx.theme().colors().text_disabled
            } else {
                cx.theme().colors().text
            },
            font_family: settings.ui_font.family.clone(),
            font_features: settings.ui_font.features.clone(),
            font_fallbacks: settings.ui_font.fallbacks.clone(),
            font_size: rems(0.875).into(),
            font_weight: settings.ui_font.weight,
            line_height: relative(1.3),
            ..Default::default()
        };

        EditorElement::new(
            editor,
            EditorStyle {
                background: cx.theme().colors().editor_background,
                local_player: cx.theme().players().local(),
                text: text_style,
                ..Default::default()
            },
        )
    }

    fn on_query_change(
        &mut self,
        _: Entity<Editor>,
        event: &editor::EditorEvent,
        cx: &mut Context<Self>,
    ) {
        if let editor::EditorEvent::Edited { .. } = event {
            self.query_contains_error = false;
            self.refresh_search(cx);
        }
    }

    fn refresh_search(&mut self, cx: &mut Context<Self>) {
        self.fetch_extensions_debounced(
            Some(Box::new(|this, cx| {
                this.scroll_to_top(cx);
            })),
            cx,
        );
        self.refresh_feature_upsells(cx);
    }

    pub fn focus_extension(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.query_editor.update(cx, |editor, cx| {
            editor.set_text(format!("id:{id}"), window, cx)
        });
        self.refresh_search(cx);
    }

    pub fn change_provides_filter(
        &mut self,
        provides_filter: Option<ExtensionProvides>,
        cx: &mut Context<Self>,
    ) {
        self.provides_filter = provides_filter;
        self.refresh_search(cx);
    }

    fn fetch_extensions_debounced(
        &mut self,
        on_complete: Option<Box<dyn FnOnce(&mut Self, &mut Context<Self>) + Send>>,
        cx: &mut Context<ExtensionsPage>,
    ) {
        self.extension_fetch_task = Some(cx.spawn(async move |this, cx| {
            let search = this
                .update(cx, |this, cx| this.search_query(cx))
                .ok()
                .flatten();

            // Only debounce the fetching of extensions if we have a search
            // query.
            //
            // If the search was just cleared then we can just reload the list
            // of extensions without a debounce, which allows us to avoid seeing
            // an intermittent flash of a "no extensions" state.
            if search.is_some() {
                cx.background_executor()
                    .timer(Duration::from_millis(250))
                    .await;
            };

            this.update(cx, |this, cx| {
                this.fetch_extensions(
                    search,
                    Some(BTreeSet::from_iter(this.provides_filter)),
                    on_complete,
                    cx,
                );
            })
            .ok();
        }));
    }

    pub fn search_query(&self, cx: &mut App) -> Option<String> {
        let search = self.query_editor.read(cx).text(cx);
        if search.trim().is_empty() {
            None
        } else {
            Some(search)
        }
    }

    fn render_empty_state(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let has_search = self.search_query(cx).is_some();

        let message = if self.is_fetching_extensions {
            "Loading extensions…"
        } else if self.fetch_failed {
            "Failed to load extensions. Please check your connection and try again."
        } else {
            match self.filter {
                ExtensionFilter::All => {
                    if has_search {
                        "No extensions that match your search."
                    } else {
                        "No extensions."
                    }
                }
                ExtensionFilter::Installed => {
                    if has_search {
                        "No installed extensions that match your search."
                    } else {
                        "No installed extensions."
                    }
                }
                ExtensionFilter::NotInstalled => {
                    if has_search {
                        "No not installed extensions that match your search."
                    } else {
                        "No not installed extensions."
                    }
                }
            }
        };

        h_flex()
            .py_4()
            .gap_1p5()
            .when(self.fetch_failed, |this| {
                this.child(
                    Icon::new(IconName::Warning)
                        .size(IconSize::Small)
                        .color(Color::Warning),
                )
            })
            .child(Label::new(message))
    }

    fn update_settings(
        &mut self,
        selection: &ToggleState,

        cx: &mut Context<Self>,
        callback: impl 'static + Send + Fn(&mut SettingsContent, bool),
    ) {
        if let Some(workspace) = self.workspace.upgrade() {
            let fs = workspace.read(cx).app_state().fs.clone();
            let selection = *selection;
            settings::update_settings_file(fs, cx, move |settings, _| {
                let value = match selection {
                    ToggleState::Unselected => false,
                    ToggleState::Selected => true,
                    _ => return,
                };

                callback(settings, value)
            });
        }
    }

    fn refresh_feature_upsells(&mut self, cx: &mut Context<Self>) {
        let Some(search) = self.search_query(cx) else {
            self.upsells.clear();
            return;
        };

        if let Some(id) = search.strip_prefix("id:") {
            self.upsells.clear();

            let upsell = match id.to_lowercase().as_str() {
                "ruff" => Some(Feature::ExtensionRuff),
                "basedpyright" => Some(Feature::ExtensionBasedpyright),
                "ty" => Some(Feature::ExtensionTy),
                _ => None,
            };

            if let Some(upsell) = upsell {
                self.upsells.insert(upsell);
            }

            return;
        }

        let search = search.to_lowercase();
        let search_terms = search
            .split_whitespace()
            .map(|term| term.trim())
            .collect::<Vec<_>>();

        for (feature, keywords) in keywords_by_feature() {
            if keywords
                .iter()
                .any(|keyword| search_terms.contains(keyword))
            {
                self.upsells.insert(*feature);
            } else {
                self.upsells.remove(feature);
            }
        }
    }

    fn render_feature_upsell_banner(
        &self,
        label: SharedString,
        docs_url: SharedString,
        vim: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let docs_url_button = Button::new("open_docs", "View Documentation")
            .end_icon(Icon::new(IconName::ArrowUpRight).size(IconSize::Small))
            .on_click({
                move |_event, _window, cx| {
                    telemetry::event!(
                        "Documentation Viewed",
                        source = "Feature Upsell",
                        url = docs_url,
                    );
                    cx.open_url(&docs_url)
                }
            });

        div()
            .pt_4()
            .px_4()
            .child(
                Banner::new()
                    .severity(Severity::Success)
                    .child(Label::new(label).mt_0p5())
                    .map(|this| {
                        if vim {
                            this.action_slot(
                                h_flex()
                                    .gap_1()
                                    .child(docs_url_button)
                                    .child(Divider::vertical().color(ui::DividerColor::Border))
                                    .child(
                                        h_flex()
                                            .pl_1()
                                            .gap_1()
                                            .child(Label::new("Enable Vim mode"))
                                            .child(
                                                Switch::new(
                                                    "enable-vim",
                                                    if VimModeSetting::get_global(cx).0 {
                                                        ui::ToggleState::Selected
                                                    } else {
                                                        ui::ToggleState::Unselected
                                                    },
                                                )
                                                .on_click(cx.listener(
                                                    move |this, selection, _, cx| {
                                                        telemetry::event!(
                                                            "Vim Mode Toggled",
                                                            source = "Feature Upsell"
                                                        );
                                                        this.update_settings(
                                                            selection,
                                                            cx,
                                                            |setting, value| {
                                                                setting.vim_mode = Some(value)
                                                            },
                                                        );
                                                    },
                                                )),
                                            ),
                                    ),
                            )
                        } else {
                            this.action_slot(docs_url_button)
                        }
                    }),
            )
            .into_any_element()
    }

    fn render_feature_upsells(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut container = v_flex();

        for feature in &self.upsells {
            let banner = match feature {
                Feature::AgentClaude => self.render_feature_upsell_banner(
                    "Claude Agent support is built-in to Zed!".into(),
                    "https://zed.dev/docs/ai/external-agents#claude-agent".into(),
                    false,
                    cx,
                ),
                Feature::AgentCodex => self.render_feature_upsell_banner(
                    "Codex CLI support is built-in to Zed!".into(),
                    "https://zed.dev/docs/ai/external-agents#codex-cli".into(),
                    false,
                    cx,
                ),
                Feature::AgentGemini => self.render_feature_upsell_banner(
                    "Gemini CLI support is built-in to Zed!".into(),
                    "https://zed.dev/docs/ai/external-agents#gemini-cli".into(),
                    false,
                    cx,
                ),
                Feature::ExtensionBasedpyright => self.render_feature_upsell_banner(
                    "Basedpyright (Python language server) support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/python#basedpyright".into(),
                    false,
                    cx,
                ),
                Feature::ExtensionRuff => self.render_feature_upsell_banner(
                    "Ruff (linter for Python) support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/python#code-formatting--linting".into(),
                    false,
                    cx,
                ),
                Feature::ExtensionTailwind => self.render_feature_upsell_banner(
                    "Tailwind CSS support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/tailwindcss".into(),
                    false,
                    cx,
                ),
                Feature::ExtensionTy => self.render_feature_upsell_banner(
                    "Ty (Python language server) support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/python".into(),
                    false,
                    cx,
                ),
                Feature::Git => self.render_feature_upsell_banner(
                    "Zed comes with basic Git support—more features are coming in the future."
                        .into(),
                    "https://zed.dev/docs/git".into(),
                    false,
                    cx,
                ),
                Feature::LanguageBash => self.render_feature_upsell_banner(
                    "Shell support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/bash".into(),
                    false,
                    cx,
                ),
                Feature::LanguageC => self.render_feature_upsell_banner(
                    "C support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/c".into(),
                    false,
                    cx,
                ),
                Feature::LanguageCpp => self.render_feature_upsell_banner(
                    "C++ support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/cpp".into(),
                    false,
                    cx,
                ),
                Feature::LanguageGo => self.render_feature_upsell_banner(
                    "Go support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/go".into(),
                    false,
                    cx,
                ),
                Feature::LanguagePython => self.render_feature_upsell_banner(
                    "Python support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/python".into(),
                    false,
                    cx,
                ),
                Feature::LanguageReact => self.render_feature_upsell_banner(
                    "React support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/typescript".into(),
                    false,
                    cx,
                ),
                Feature::LanguageRust => self.render_feature_upsell_banner(
                    "Rust support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/rust".into(),
                    false,
                    cx,
                ),
                Feature::LanguageTypescript => self.render_feature_upsell_banner(
                    "Typescript support is built-in to Zed!".into(),
                    "https://zed.dev/docs/languages/typescript".into(),
                    false,
                    cx,
                ),
                Feature::OpenIn => self.render_feature_upsell_banner(
                    "Zed supports linking to a source line on GitHub and others.".into(),
                    "https://zed.dev/docs/git#git-integrations".into(),
                    false,
                    cx,
                ),
                Feature::Vim => self.render_feature_upsell_banner(
                    "Vim support is built-in to Zed!".into(),
                    "https://zed.dev/docs/vim".into(),
                    true,
                    cx,
                ),
            };
            container = container.child(banner);
        }

        container
    }
}

struct DevExtensionRebuildPickerDelegate {
    entries: Vec<Arc<ExtensionManifest>>,
    matches: Vec<StringMatch>,
    selected_index: usize,
}

impl DevExtensionRebuildPickerDelegate {
    fn new(manifests: Vec<Arc<ExtensionManifest>>) -> Self {
        let matches = manifests
            .iter()
            .enumerate()
            .map(|(ix, manifest)| StringMatch {
                candidate_id: ix,
                score: 0.0,
                positions: Vec::new(),
                string: manifest.name.clone(),
            })
            .collect();

        Self {
            entries: manifests,
            matches,
            selected_index: 0,
        }
    }
}

impl PickerDelegate for DevExtensionRebuildPickerDelegate {
    type ListItem = ListItem;

    fn name() -> &'static str {
        "dev-extension-rebuild"
    }

    fn match_count(&self) -> usize {
        self.matches.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(
        &mut self,
        ix: usize,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) {
        self.selected_index = ix;
    }

    fn selected_index_changed(
        &self,
        _ix: usize,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Option<Box<dyn Fn(&mut Window, &mut App) + 'static>> {
        None
    }

    fn update_matches(
        &mut self,
        query: String,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        let background = cx.background_executor().clone();
        let candidates = self
            .entries
            .iter()
            .enumerate()
            .map(|(ix, manifest)| StringMatchCandidate::new(ix, manifest.name.as_ref()))
            .collect::<Vec<_>>();

        cx.spawn_in(window, async move |this, cx| {
            let matches = if query.is_empty() {
                candidates
                    .into_iter()
                    .enumerate()
                    .map(|(index, candidate)| StringMatch {
                        candidate_id: index,
                        string: candidate.string,
                        positions: Vec::new(),
                        score: 0.0,
                    })
                    .collect()
            } else {
                match_strings(
                    &candidates,
                    &query,
                    false,
                    true,
                    100,
                    &Default::default(),
                    background,
                )
                .await
            };

            this.update(cx, |this, _cx| {
                this.delegate.matches = matches;
                this.delegate.selected_index = this
                    .delegate
                    .selected_index
                    .min(this.delegate.matches.len().saturating_sub(1));
            })
            .log_err();
        })
    }

    fn confirm(&mut self, _secondary: bool, _window: &mut Window, cx: &mut Context<Picker<Self>>) {
        let Some(mat) = self.matches.get(self.selected_index) else {
            return;
        };

        let extension_id = self.entries[mat.candidate_id].id.clone();
        ExtensionStore::global(cx).update(cx, |store, cx| {
            store.rebuild_dev_extension(extension_id, cx);
        });

        cx.emit(DismissEvent);
    }

    fn dismissed(&mut self, _window: &mut Window, _cx: &mut Context<Picker<Self>>) {}

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        Arc::from("Rebuild dev extension…")
    }

    fn render_match(
        &self,
        ix: usize,
        selected: bool,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let mat = self.matches.get(ix)?;
        let entry = self.entries.get(mat.candidate_id)?;

        let item = ListItem::new(("dev-extension-list-item", mat.candidate_id))
            .inset(true)
            .spacing(ListItemSpacing::Sparse)
            .toggle_state(selected)
            .child(
                h_flex()
                    .w_full()
                    .py_px()
                    .justify_between()
                    .gap_2()
                    .child(Label::new(entry.name.clone()))
                    .child(
                        Label::new(format!("{} • v{}", entry.id, entry.version))
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    ),
            );

        Some(item)
    }

    fn no_matches_text(&self, _window: &mut Window, _cx: &mut App) -> Option<SharedString> {
        Some("No dev extensions found".into())
    }
}

fn visible_product_default_install_failures(
    settings: &ExtensionSettings,
    default_extensions: &[&str],
) -> Vec<(Arc<str>, String)> {
    let mut failures = settings
        .auto_install_extension_failures
        .iter()
        .filter(|(extension_id, _)| {
            default_extensions.contains(&extension_id.as_ref())
                && settings
                    .auto_install_extensions
                    .get(extension_id.as_ref())
                    .copied()
                    .unwrap_or(true)
        })
        .map(|(extension_id, reason)| (extension_id.clone(), reason.clone()))
        .collect::<Vec<_>>();
    failures.sort_by(|left, right| left.0.cmp(&right.0));
    failures
}

#[cfg(test)]
mod product_default_tests {
    use super::{ExtensionSettings, visible_product_default_install_failures};

    #[test]
    fn unavailable_default_rows_follow_effective_disposition() {
        let mut settings = ExtensionSettings::default();
        settings
            .auto_install_extension_failures
            .insert("kotlin".into(), "offline".into());
        settings
            .auto_install_extension_failures
            .insert("java".into(), "download denied".into());
        settings
            .auto_install_extension_failures
            .insert("unrelated".into(), "offline".into());
        assert!(visible_product_default_install_failures(&settings, &[]).is_empty());
        let defaults = &["java", "kotlin"];
        let failures = visible_product_default_install_failures(&settings, defaults);
        assert_eq!(
            failures
                .iter()
                .map(|(extension_id, _)| extension_id.as_ref())
                .collect::<Vec<_>>(),
            ["java", "kotlin"]
        );

        settings
            .auto_install_extensions
            .insert("java".into(), false);
        let failures = visible_product_default_install_failures(&settings, defaults);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0.as_ref(), "kotlin");
        assert_eq!(failures[0].1, "offline");
    }
}

impl Render for ExtensionsPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let default_install_failures = visible_product_default_install_failures(
            ExtensionSettings::get_global(cx),
            product_flavor::DEFAULT_EXTENSIONS,
        );
        let default_installs_in_progress = {
            let store = ExtensionStore::global(cx);
            let store = store.read(cx);
            product_flavor::DEFAULT_EXTENSIONS
                .iter()
                .copied()
                .filter(|id| store.product_default_install_in_progress(id))
                .collect::<Vec<_>>()
        };
        let page = cx.entity().downgrade();
        v_flex()
            .size_full()
            .bg(cx.theme().colors().editor_background)
            .child(
                v_flex()
                    .gap_4()
                    .pt_4()
                    .px_4()
                    .bg(cx.theme().colors().editor_background)
                    .child(
                        h_flex()
                            .w_full()
                            .gap_1p5()
                            .justify_between()
                            .child(Headline::new("Extensions").size(HeadlineSize::Large))
                            .child(
                                Button::new("install-dev-extension", "Install Dev Extension")
                                    .style(ButtonStyle::Outlined)
                                    .size(ButtonSize::Medium)
                                    .on_click(|_event, window, cx| {
                                        window.dispatch_action(Box::new(InstallDevExtension), cx)
                                    }),
                            ),
                    )
                    .children(
                        default_install_failures
                            .into_iter()
                            .map(|(extension_id, reason)| {
                                let page = page.clone();
                                let failure_message =
                                    format!("{extension_id} extension unavailable: {reason}");
                                h_flex()
                                    .id(format!("default-extension-failure-{extension_id}"))
                                    .role(gpui::Role::Group)
                                    .aria_label(failure_message.clone())
                                    .w_full()
                                    .min_w_0()
                                    .gap_2()
                                    .tooltip(Tooltip::text(failure_message.clone()))
                                    .child(
                                        div().flex_1().min_w_0().overflow_hidden().child(
                                            Label::new(failure_message)
                                                .line_clamp(3)
                                                .color(Color::Warning),
                                        ),
                                    )
                                    .child(
                                        div().flex_none().child(
                                            Button::new(
                                                format!("retry-default-{extension_id}"),
                                                "Retry install",
                                            )
                                            .aria_label(format!(
                                                "Retry installing {extension_id} extension"
                                            ))
                                            .style(ButtonStyle::Outlined)
                                            .disabled(
                                                default_installs_in_progress
                                                    .contains(&extension_id.as_ref()),
                                            )
                                            .on_click(
                                                move |_, _, cx| {
                                                    let install = ExtensionStore::global(cx)
                                                        .update(cx, |store, cx| {
                                                            store.install_product_default_extension(
                                                                extension_id.clone(),
                                                                cx,
                                                            )
                                                        });
                                                    let page = page.clone();
                                                    cx.spawn(async move |cx| {
                                                        let result = install.await;
                                                        page.update(cx, |_, cx| cx.notify())
                                                            .log_err();
                                                        result
                                                    })
                                                    .detach_and_log_err(cx);
                                                },
                                            ),
                                        ),
                                    )
                            }),
                    )
                    .child(
                        h_flex()
                            .w_full()
                            .flex_wrap()
                            .gap_2()
                            .child(self.render_search(cx))
                            .child(
                                div().child(
                                    ToggleButtonGroup::single_row(
                                        "filter-buttons",
                                        [
                                            ToggleButtonSimple::new(
                                                "All",
                                                cx.listener(|this, _event, _, cx| {
                                                    this.filter = ExtensionFilter::All;
                                                    this.filter_extension_entries(cx);
                                                    this.scroll_to_top(cx);
                                                }),
                                            ),
                                            ToggleButtonSimple::new(
                                                "Installed",
                                                cx.listener(|this, _event, _, cx| {
                                                    this.filter = ExtensionFilter::Installed;
                                                    this.filter_extension_entries(cx);
                                                    this.scroll_to_top(cx);
                                                }),
                                            ),
                                            ToggleButtonSimple::new(
                                                "Not Installed",
                                                cx.listener(|this, _event, _, cx| {
                                                    this.filter = ExtensionFilter::NotInstalled;
                                                    this.filter_extension_entries(cx);
                                                    this.scroll_to_top(cx);
                                                }),
                                            ),
                                        ],
                                    )
                                    .style(ToggleButtonGroupStyle::Outlined)
                                    .size(ToggleButtonGroupSize::Custom(rems_from_px(30_f32))) // Perfectly matches the input
                                    .label_size(LabelSize::Default)
                                    .auto_width()
                                    .selected_index(match self.filter {
                                        ExtensionFilter::All => 0,
                                        ExtensionFilter::Installed => 1,
                                        ExtensionFilter::NotInstalled => 2,
                                    })
                                    .into_any_element(),
                                ),
                            ),
                    ),
            )
            .child(
                h_flex()
                    .id("filter-row")
                    .gap_2()
                    .py_2p5()
                    .px_4()
                    .border_b_1()
                    .border_color(cx.theme().colors().border_variant)
                    .overflow_x_scroll()
                    .child(
                        Button::new("filter-all-categories", "All")
                            .when(self.provides_filter.is_none(), |button| {
                                button.style(ButtonStyle::Filled)
                            })
                            .when(self.provides_filter.is_some(), |button| {
                                button.style(ButtonStyle::Subtle)
                            })
                            .toggle_state(self.provides_filter.is_none())
                            .on_click(cx.listener(|this, _event, _, cx| {
                                this.change_provides_filter(None, cx);
                            })),
                    )
                    .children(
                        ExtensionProvides::iter()
                            .filter(|provides| match provides {
                                ExtensionProvides::AgentServers
                                | ExtensionProvides::Grammars // grammars do not add anything of value to users currently
                                | ExtensionProvides::IndexedDocsProviders
                                | ExtensionProvides::SlashCommands => false,
                                _ => true,
                            })
                            .map(|provides| {
                                let label = extension_provides_label(provides);
                                let button_id =
                                    SharedString::from(format!("filter-category-{}", label));

                                Button::new(button_id, label)
                                    .style(if self.provides_filter == Some(provides) {
                                        ButtonStyle::Filled
                                    } else {
                                        ButtonStyle::Subtle
                                    })
                                    .toggle_state(self.provides_filter == Some(provides))
                                    .on_click({
                                        cx.listener(move |this, _event, _, cx| {
                                            this.change_provides_filter(Some(provides), cx);
                                        })
                                    })
                            }),
                    ),
            )
            .map(|element| {
                #[cfg(feature = "agentic")]
                let element = element.child(self.render_packages(cx));
                element
            })
            .child(self.render_feature_upsells(cx))
            .child(v_flex().px_4().size_full().overflow_y_hidden().map(|this| {
                let mut count = self.filtered_remote_extension_indices.len();
                if self.filter.include_dev_extensions() {
                    count += self.filtered_dev_extension_indices.len();
                }

                if count == 0 {
                    this.child(self.render_empty_state(cx)).into_any_element()
                } else {
                    let scroll_handle = &self.list;
                    this.child(
                        uniform_list("entries", count, cx.processor(Self::render_extensions))
                            .flex_grow_1()
                            .pb_4()
                            .track_scroll(scroll_handle),
                    )
                    .vertical_scrollbar_for(scroll_handle, window, cx)
                    .into_any_element()
                }
            }))
    }
}

impl EventEmitter<ItemEvent> for ExtensionsPage {}

impl Focusable for ExtensionsPage {
    fn focus_handle(&self, cx: &App) -> gpui::FocusHandle {
        self.query_editor.read(cx).focus_handle(cx)
    }
}

impl Item for ExtensionsPage {
    type Event = ItemEvent;

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Extensions".into()
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        Some("Extensions Page Opened")
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    fn to_item_events(event: &Self::Event, f: &mut dyn FnMut(workspace::item::ItemEvent)) {
        f(*event)
    }
}
