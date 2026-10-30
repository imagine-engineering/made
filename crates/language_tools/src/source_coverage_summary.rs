use crate::language_tool_tree::{LanguageToolNode, LanguageToolNodeId};
use collections::HashMap;
use editor::Editor;
use gpui::{App, AppContext as _, Context, EventEmitter, Render, Subscription, WeakEntity, Window};
use project::{
    Project, ProjectPath,
    source_coverage::{SourceCoverageSnapshot, SourceCoverageStatus},
};
use ui::{ContextMenu, PopoverMenu, PopoverMenuHandle, Tooltip, prelude::*};
use util::ResultExt as _;
use workspace::{ItemHandle, Toolbar, ToolbarItemEvent, ToolbarItemLocation, ToolbarItemView};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceCoverageFilter {
    All,
    Covered,
    Uncovered,
    Partial,
}

#[derive(Clone, Eq, PartialEq)]
struct SourceCoverageReportIdentity {
    project_generation: u64,
    provider_id: project::source_coverage::SourceCoverageProviderId,
    generation: u64,
}

impl From<&SourceCoverageSnapshot> for SourceCoverageReportIdentity {
    fn from(snapshot: &SourceCoverageSnapshot) -> Self {
        Self {
            project_generation: snapshot.project_generation,
            provider_id: snapshot.provider_id.clone(),
            generation: snapshot.generation,
        }
    }
}

pub struct SourceCoverageSummaryProjection {
    identity: SourceCoverageReportIdentity,
    pub roots: Vec<LanguageToolNode>,
    navigation: HashMap<LanguageToolNodeId, ProjectPath>,
}

impl SourceCoverageSummaryProjection {
    pub fn project(snapshot: &SourceCoverageSnapshot, filter: SourceCoverageFilter) -> Self {
        Self::project_visible(snapshot, filter, snapshot.status, |_| true)
    }

    fn project_visible(
        snapshot: &SourceCoverageSnapshot,
        filter: SourceCoverageFilter,
        status: SourceCoverageStatus,
        permitted: impl Fn(&ProjectPath) -> bool,
    ) -> Self {
        let mut navigation = HashMap::default();
        let mut files = snapshot
            .files
            .iter()
            .filter(|file| permitted(&file.path))
            .filter(|file| match filter {
                SourceCoverageFilter::All => true,
                SourceCoverageFilter::Covered => file.covered_lines > 0,
                SourceCoverageFilter::Uncovered => file.uncovered_lines > 0,
                SourceCoverageFilter::Partial => file.truncated || snapshot.truncated,
            })
            .map(|file| {
                let id = LanguageToolNodeId(format!(
                    "source-coverage:{}:{}:{}:{}",
                    snapshot.provider_id.0,
                    snapshot.generation,
                    file.path.worktree_id.to_proto(),
                    file.path.path.as_unix_str()
                ));
                navigation.insert(id.clone(), file.path.clone());
                let total = file.covered_lines.saturating_add(file.uncovered_lines);
                let percentage =
                    (total > 0).then(|| u64::from(file.covered_lines) * 100 / u64::from(total));
                LanguageToolNode {
                    id,
                    label: file.path.path.as_unix_str().to_string(),
                    secondary_label: Some(match percentage {
                        Some(percentage) => format!(
                            "{percentage}% · {} covered · {} uncovered{}",
                            file.covered_lines,
                            file.uncovered_lines,
                            if file.truncated { " · truncated" } else { "" }
                        ),
                        None => "no executable lines".to_string(),
                    }),
                    icon: None,
                    accessibility_label: format!(
                        "Coverage for {}, {} covered lines, {} uncovered lines{}",
                        file.path.path.as_unix_str(),
                        file.covered_lines,
                        file.uncovered_lines,
                        if file.truncated { ", truncated" } else { "" }
                    ),
                    children: Vec::new(),
                    enabled: true,
                    activation_label: Some("Open covered source file".to_string()),
                }
            })
            .collect::<Vec<_>>();
        files.sort_by(|left, right| left.label.cmp(&right.label));
        let status = status_label(status);
        let root = LanguageToolNode {
            id: LanguageToolNodeId(format!(
                "source-coverage:{}:{}",
                snapshot.provider_id.0, snapshot.generation
            )),
            label: format!("Coverage — {}", snapshot.provider_id.0),
            secondary_label: Some(format!(
                "{status} · {} file(s){}",
                files.len(),
                if snapshot.truncated {
                    " · truncated"
                } else {
                    ""
                }
            )),
            icon: None,
            accessibility_label: format!(
                "Source coverage provider {}, {status}, {} files{}",
                snapshot.provider_id.0,
                files.len(),
                if snapshot.truncated {
                    ", truncated"
                } else {
                    ""
                }
            ),
            children: files,
            enabled: false,
            activation_label: None,
        };
        Self {
            identity: snapshot.into(),
            roots: vec![root],
            navigation,
        }
    }

    pub fn navigation(&self, id: &LanguageToolNodeId) -> Option<&ProjectPath> {
        self.navigation.get(id)
    }
}

pub fn init(cx: &mut App) {
    cx.observe_new(|toolbar: &mut Toolbar, window, cx| {
        if let Some(window) = window {
            let summary = cx.new(|_| SourceCoverageToolbar::default());
            toolbar.add_item(summary, window, cx);
        }
    })
    .detach();
}

#[cfg(any(test, feature = "test-support"))]
pub fn toolbar_snapshot_for_test(
    workspace: &gpui::Entity<workspace::Workspace>,
    cx: &App,
) -> Option<(String, usize)> {
    let toolbar = workspace.read(cx).active_pane().read(cx).toolbar().clone();
    toolbar
        .read(cx)
        .item_of_type::<SourceCoverageToolbar>()
        .map(|summary| {
            summary.read_with(cx, |summary, _| {
                (
                    summary.label.clone(),
                    summary
                        .projection
                        .as_ref()
                        .map_or(0, |projection| projection.navigation.len()),
                )
            })
        })
}

struct SourceCoverageToolbar {
    editor: Option<WeakEntity<Editor>>,
    provider: Option<project::source_coverage::SourceCoverageProviderId>,
    providers: Vec<project::source_coverage::SourceCoverageProviderId>,
    filter: SourceCoverageFilter,
    projection: Option<SourceCoverageSummaryProjection>,
    label: String,
    menu: PopoverMenuHandle<ContextMenu>,
    subscriptions: Vec<Subscription>,
}

impl Default for SourceCoverageToolbar {
    fn default() -> Self {
        Self {
            editor: None,
            provider: None,
            providers: Vec::new(),
            filter: SourceCoverageFilter::All,
            projection: None,
            label: String::new(),
            menu: PopoverMenuHandle::default(),
            subscriptions: Vec::new(),
        }
    }
}

fn coverage_path_permitted(project: &Project, path: &ProjectPath, cx: &App) -> bool {
    !project.is_disconnected(cx)
        && !project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
            &project.worktree_store(),
            cx,
        )
        && project
            .visible_worktrees(cx)
            .any(|tree| tree.read(cx).id() == path.worktree_id)
        && project.entry_for_path(path, cx).is_some_and(|entry| {
            entry.is_file() && !entry.is_private && !entry.is_fifo && !entry.is_external
        })
}

impl SourceCoverageToolbar {
    fn refresh(&mut self, cx: &mut Context<Self>) {
        self.menu.hide(cx);
        self.projection = None;
        self.providers.clear();
        self.label.clear();
        if let Some(editor) = self.editor.as_ref().and_then(WeakEntity::upgrade)
            && let Some(project) = editor.read(cx).project()
        {
            let project = project.read(cx);
            let store = project.source_coverage_store().read(cx);
            self.providers = store
                .cached_snapshots()
                .map(|snapshot| snapshot.provider_id.clone())
                .collect();
            self.provider = store
                .selected_snapshot()
                .map(|snapshot| snapshot.provider_id.clone());
            if let Some(snapshot) = store.selected_snapshot() {
                let status = if project.is_disconnected(cx) {
                    SourceCoverageStatus::Disconnected
                } else if project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &project.worktree_store(),
                    cx,
                ) {
                    SourceCoverageStatus::Restricted
                } else {
                    snapshot.status
                };
                self.label = format!(
                    "Coverage: {} · {}{}",
                    snapshot.provider_id.0,
                    status_label(status),
                    if snapshot.truncated {
                        " · truncated"
                    } else {
                        ""
                    }
                );
                self.projection = Some(SourceCoverageSummaryProjection::project_visible(
                    snapshot,
                    self.filter,
                    status,
                    |path| {
                        matches!(
                            status,
                            SourceCoverageStatus::Current | SourceCoverageStatus::Partial
                        ) && coverage_path_permitted(project, path, cx)
                    },
                ));
            }
        }
        cx.emit(ToolbarItemEvent::ChangeLocation(if self.label.is_empty() {
            ToolbarItemLocation::Hidden
        } else {
            ToolbarItemLocation::PrimaryRight
        }));
        cx.notify();
    }

    fn select_provider(
        &mut self,
        provider: project::source_coverage::SourceCoverageProviderId,
        cx: &mut Context<Self>,
    ) {
        if let Some(editor) = self.editor.as_ref().and_then(WeakEntity::upgrade)
            && let Some(project) = editor.read(cx).project()
        {
            let store = project.read(cx).source_coverage_store().clone();
            store
                .update(cx, |store, cx| store.select_provider(provider, cx))
                .log_err();
        }
        self.refresh(cx);
    }

    fn open_source(
        &self,
        identity: &SourceCoverageReportIdentity,
        path: ProjectPath,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(editor) = self.editor.as_ref().and_then(WeakEntity::upgrade) else {
            return;
        };
        let Some(project) = editor.read(cx).project() else {
            return;
        };
        let current_source = project.read_with(cx, |project, cx| {
            project
                .source_coverage_store()
                .read(cx)
                .selected_snapshot()
                .is_some_and(|snapshot| {
                    SourceCoverageReportIdentity::from(snapshot) == *identity
                        && matches!(
                            snapshot.status,
                            SourceCoverageStatus::Current | SourceCoverageStatus::Partial
                        )
                        && snapshot.files.iter().any(|file| file.path == path)
                })
                && coverage_path_permitted(project, &path, cx)
        });
        if !current_source {
            return;
        }
        if let Some(workspace) = editor.read(cx).workspace() {
            workspace
                .update(cx, |workspace, cx| {
                    workspace.open_path(path, None, true, window, cx)
                })
                .detach_and_log_err(cx);
        }
    }
}

impl EventEmitter<ToolbarItemEvent> for SourceCoverageToolbar {}

impl ToolbarItemView for SourceCoverageToolbar {
    fn set_active_pane_item(
        &mut self,
        item: Option<&dyn ItemHandle>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> ToolbarItemLocation {
        self.subscriptions.clear();
        self.editor = item
            .and_then(|item| item.downcast::<Editor>())
            .map(|editor| editor.downgrade());
        if let Some(editor) = self.editor.as_ref().and_then(WeakEntity::upgrade)
            && let Some(project) = editor.read(cx).project().cloned()
        {
            let store = project.read(cx).source_coverage_store().clone();
            self.subscriptions
                .push(cx.subscribe(&store, |this, _, _, cx| this.refresh(cx)));
            self.subscriptions
                .push(cx.subscribe(&project, |this, _, event, cx| {
                    if matches!(
                        event,
                        project::Event::WorktreeAdded(_)
                            | project::Event::WorktreeRemoved(_)
                            | project::Event::WorktreeOrderChanged
                            | project::Event::WorktreeUpdatedEntries(..)
                            | project::Event::WorktreePathsChanged { .. }
                            | project::Event::RemoteIdChanged(_)
                            | project::Event::DisconnectedFromHost
                            | project::Event::DisconnectedFromRemote { .. }
                            | project::Event::Closed
                            | project::Event::Rejoined
                    ) {
                        this.refresh(cx);
                    }
                }));
            if let Some(trust) = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx) {
                self.subscriptions
                    .push(cx.subscribe(&trust, |this, _, _, cx| this.refresh(cx)));
            }
        }
        self.refresh(cx);
        if self.label.is_empty() {
            ToolbarItemLocation::Hidden
        } else {
            ToolbarItemLocation::PrimaryRight
        }
    }
}

impl Render for SourceCoverageToolbar {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let this = cx.entity().downgrade();
        let providers = self.providers.clone();
        let selected = self.provider.clone();
        let filter = self.filter;
        let files = self
            .projection
            .as_ref()
            .map(|projection| {
                projection
                    .roots
                    .iter()
                    .flat_map(|root| &root.children)
                    .filter_map(|node| {
                        projection.navigation(&node.id).map(|path| {
                            (
                                projection.identity.clone(),
                                node.accessibility_label.clone(),
                                path.clone(),
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let menu = PopoverMenu::new("source-coverage-summary")
            .with_handle(self.menu.clone())
            .trigger_with_tooltip(
                Button::new("source-coverage-summary-trigger", self.label.clone()),
                Tooltip::text("Source coverage summary and filters"),
            )
            .menu(move |window, cx| {
                let this = this.clone();
                let providers = providers.clone();
                let selected = selected.clone();
                let files = files.clone();
                Some(ContextMenu::build(window, cx, move |mut menu, _, _| {
                    for provider in providers {
                        let this = this.clone();
                        menu = menu.toggleable_entry(
                            provider.0.clone(),
                            selected.as_ref() == Some(&provider),
                            IconPosition::Start,
                            None,
                            move |_, cx| {
                                if let Some(this) = this.upgrade() {
                                    this.update(cx, |this, cx| {
                                        this.select_provider(provider.clone(), cx);
                                    });
                                }
                            },
                        );
                    }
                    menu = menu.separator();
                    for (label, selected_filter) in [
                        ("All files", SourceCoverageFilter::All),
                        ("Covered files", SourceCoverageFilter::Covered),
                        ("Uncovered files", SourceCoverageFilter::Uncovered),
                        ("Partial files", SourceCoverageFilter::Partial),
                    ] {
                        let this = this.clone();
                        menu = menu.toggleable_entry(
                            label,
                            filter == selected_filter,
                            IconPosition::Start,
                            None,
                            move |_, cx| {
                                if let Some(this) = this.upgrade() {
                                    this.update(cx, |this, cx| {
                                        this.filter = selected_filter;
                                        this.refresh(cx);
                                    });
                                }
                            },
                        );
                    }
                    menu = menu.separator();
                    for (identity, label, path) in files {
                        let this = this.clone();
                        menu = menu.entry(label, None, move |window, cx| {
                            if let Some(this) = this.upgrade() {
                                this.update(cx, |this, cx| {
                                    this.open_source(&identity, path.clone(), window, cx)
                                });
                            }
                        });
                    }
                    menu
                }))
            });
        div()
            .id("source-coverage-status")
            .role(gpui::Role::Status)
            .aria_label("Source coverage status")
            .aria_value(self.label.clone())
            .aria_live(gpui::accesskit::Live::Polite)
            .child(menu)
    }
}

fn status_label(status: SourceCoverageStatus) -> &'static str {
    match status {
        SourceCoverageStatus::Loading => "loading",
        SourceCoverageStatus::Current => "current",
        SourceCoverageStatus::Empty => "empty",
        SourceCoverageStatus::Partial => "partial",
        SourceCoverageStatus::Stale => "stale",
        SourceCoverageStatus::Error => "error",
        SourceCoverageStatus::Restricted => "restricted",
        SourceCoverageStatus::Disconnected => "disconnected",
        SourceCoverageStatus::Mismatch => "host mismatch",
        SourceCoverageStatus::Cancelled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Arc};

    use project::ProjectItem as _;
    use project::rust_coverage_provider::RustCoverageArtifactProvider;
    use project::source_coverage::{
        SourceCoverageFile, SourceCoveragePoint, SourceCoverageProviderId, SourceCoverageRange,
        SourceCoverageSnapshot,
    };
    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    fn path(value: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(1),
            path: Arc::from(RelPath::from_unix_str(value).expect("valid fixture path")),
        }
    }

    #[gpui::test]
    async fn source_coverage_summary_production_presents_real_llvm3_report(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::UpdateGlobal as _;

        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            settings::SettingsStore::update_global(cx, |store, _| {
                store.register_setting::<feature_flags::FeatureFlagsSettings>()
            });
            feature_flags::FeatureFlagStore::init(cx);
            settings::SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |content| {
                    content
                        .feature_flags
                        .get_or_insert_default()
                        .insert("rust-coverage-experimental".into(), "on".into());
                })
            });
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/coverage",
            serde_json::json!({
                "src": {
                    "lib.rs": include_str!("../../project/test_data/rust_test_provider/physical-workspace/src/lib.rs")
                }
            }),
        )
        .await;
        let project = Project::test(fs, ["/coverage".as_ref()], cx).await;
        let source = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/coverage/src/lib.rs", cx)
                .expect("fixture source")
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(source.clone(), None, true, window, cx)
            })
            .await
            .expect("open fixture source");
        cx.run_until_parked();
        let summary = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_pane()
                .read(cx)
                .toolbar()
                .read(cx)
                .item_of_type::<SourceCoverageToolbar>()
                .expect("production coverage toolbar")
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let mut provider = RustCoverageArtifactProvider::new(1);

        for (report, expected_covered) in [
            (
                include_bytes!("../../project/test_data/source_coverage/llvm3-unit.json")
                    .as_slice(),
                10,
            ),
            (
                include_bytes!("../../project/test_data/source_coverage/llvm3-ignored.json")
                    .as_slice(),
                6,
            ),
        ] {
            let token = provider.begin_run();
            let snapshot = provider
                .parse_artifact(token, report, |absolute_path| {
                    (absolute_path == Path::new("/workspace/src/lib.rs")).then(|| source.clone())
                })
                .expect("real LLVM 3 report should parse");
            assert_eq!(
                snapshot
                    .files
                    .first()
                    .expect("mapped fixture file")
                    .covered_lines,
                expected_covered
            );
            store
                .update(cx, |store, cx| store.publish(snapshot, cx))
                .expect("publish parsed report");
            cx.run_until_parked();
            let gutter_counts = workspace.read_with(cx, |workspace, cx| {
                workspace
                    .active_item_as::<Editor>(cx)
                    .expect("fixture editor")
                    .read(cx)
                    .source_coverage_gutter_counts()
            });
            assert!(
                gutter_counts.0 > 0 && gutter_counts.1 > 0,
                "{gutter_counts:?}"
            );
            summary.read_with(cx, |summary, _| {
                assert!(summary.label.contains("current"), "{}", summary.label);
                let projection = summary.projection.as_ref().expect("report projection");
                assert_eq!(projection.navigation.len(), 1);
                assert!(
                    projection
                        .roots
                        .first()
                        .and_then(|root| root.children.first())
                        .expect("fixture coverage row")
                        .accessibility_label
                        .contains(&format!("{expected_covered} covered lines"))
                );
            });
        }
    }

    #[gpui::test]
    async fn source_coverage_summary_production_large_report_foreground_budget(
        cx: &mut gpui::TestAppContext,
    ) {
        use project::source_coverage::{
            MAX_SOURCE_COVERAGE_RANGES, MAX_SOURCE_COVERAGE_RANGES_PER_FILE,
        };

        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        let file_count = MAX_SOURCE_COVERAGE_RANGES / MAX_SOURCE_COVERAGE_RANGES_PER_FILE;
        let files = (0..file_count)
            .map(|index| {
                (
                    format!("file-{index}.py"),
                    serde_json::Value::String("x\n".repeat(MAX_SOURCE_COVERAGE_RANGES_PER_FILE)),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        fs.insert_tree("/coverage", serde_json::Value::Object(files))
            .await;
        let project = Project::test(fs, ["/coverage".as_ref()], cx).await;
        let sources = project.read_with(cx, |project, cx| {
            (0..file_count)
                .map(|index| {
                    project
                        .find_project_path(&format!("/coverage/file-{index}.py"), cx)
                        .expect("large covered source")
                })
                .collect::<Vec<_>>()
        });
        let source = sources.first().expect("first covered source").clone();
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(source.clone(), None, true, window, cx)
            })
            .await
            .expect("open large covered source");
        cx.simulate_resize(gpui::size(gpui::px(1600.0), gpui::px(720.0)));
        cx.run_until_parked();
        let summary = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_pane()
                .read(cx)
                .toolbar()
                .read(cx)
                .item_of_type::<SourceCoverageToolbar>()
                .expect("production coverage toolbar")
        });
        let snapshot = SourceCoverageSnapshot {
            project_generation: 1,
            provider_id: SourceCoverageProviderId("large-generic".into()),
            generation: 1,
            status: SourceCoverageStatus::Current,
            truncated: false,
            diagnostic: None,
            files: sources
                .into_iter()
                .map(|path| SourceCoverageFile {
                    path,
                    ranges: (0..MAX_SOURCE_COVERAGE_RANGES_PER_FILE as u32)
                        .map(|line| SourceCoverageRange {
                            start: SourceCoveragePoint { line, column: 0 },
                            end: SourceCoveragePoint { line, column: 1 },
                            hit_count: u64::from(line % 2),
                        })
                        .collect(),
                    covered_lines: (MAX_SOURCE_COVERAGE_RANGES_PER_FILE / 2) as u32,
                    uncovered_lines: (MAX_SOURCE_COVERAGE_RANGES_PER_FILE / 2) as u32,
                    truncated: false,
                })
                .collect(),
        };
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let started = std::time::Instant::now();
        store
            .update(cx, |store, cx| store.publish(snapshot, cx))
            .expect("publish maximum-size coverage report");
        store.read_with(cx, |store, _| {
            let selected = store.selected_snapshot().expect("selected coverage report");
            assert!(!selected.truncated);
            assert_eq!(
                selected
                    .files
                    .iter()
                    .map(|file| file.ranges.len())
                    .sum::<usize>(),
                MAX_SOURCE_COVERAGE_RANGES
            );
        });
        let immediate_counts = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_item_as::<Editor>(cx)
                .expect("covered editor after publication")
                .read(cx)
                .source_coverage_gutter_counts()
        });
        let immediate_current =
            summary.read_with(cx, |summary, _| summary.label.contains("current"));
        let publication = started.elapsed();
        cx.run_until_parked();
        let application = started.elapsed();
        let started = std::time::Instant::now();
        cx.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
        let paint = started.elapsed();
        let counts = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_item_as::<Editor>(cx)
                .expect("large covered editor")
                .read(cx)
                .source_coverage_gutter_counts()
        });
        assert_eq!(counts, (5_000, 5_000));
        assert_eq!(immediate_counts, counts);
        assert!(immediate_current);
        summary.read_with(cx, |summary, _| {
            assert!(summary.label.contains("current"));
            assert_eq!(
                summary
                    .projection
                    .as_ref()
                    .expect("live summary")
                    .navigation
                    .len(),
                file_count
            );
        });
        eprintln!(
            "source-coverage-full-window ranges={MAX_SOURCE_COVERAGE_RANGES} publication_us={} application_us={} paint_us={} immediate_counts={immediate_counts:?} immediate_current={immediate_current}",
            publication.as_micros(),
            application.as_micros(),
            paint.as_micros()
        );
        assert!(
            application + paint <= std::time::Duration::from_millis(250),
            "large coverage application {application:?} plus paint {paint:?} exceeded foreground budget"
        );
    }

    #[gpui::test]
    async fn source_coverage_summary_production_tracks_store_and_navigation(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/coverage",
            serde_json::json!({"a.py": "a = 1\n", "b.py": "b = 2\n", ".env": "secret"}),
        )
        .await;
        let project = Project::test(fs, ["/coverage".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let (first, second, private) = project.read_with(cx, |project, cx| {
            (
                project
                    .find_project_path("/coverage/a.py", cx)
                    .expect("first source"),
                project
                    .find_project_path("/coverage/b.py", cx)
                    .expect("second source"),
                project
                    .find_project_path("/coverage/.env", cx)
                    .expect("private source"),
            )
        });
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(first.clone(), None, true, window, cx)
            })
            .await
            .expect("open coverage source");
        cx.run_until_parked();
        let summary = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_pane()
                .read(cx)
                .toolbar()
                .read(cx)
                .item_of_type::<SourceCoverageToolbar>()
                .expect("production toolbar registration")
        });
        assert!(summary.read_with(cx, |summary, _| summary.label.is_empty()));
        cx.update(|window, _| window.activate_a11y_for_test());
        let mut previous_status_id = None;
        let mut assert_accessibility_status = |cx: &mut gpui::VisualTestContext| {
            let label = summary.read_with(cx, |summary, _| summary.label.clone());
            for _ in 0..2 {
                cx.update(|window, cx| {
                    let focus = window.focused(cx).expect("focused source editor");
                    window.draw(cx).clear(cx);
                    assert_eq!(window.focused(cx).as_ref(), Some(&focus));
                    let data = window.debug_a11y_tree_json().expect("active coverage tree");
                    let data: serde_json::Value =
                        serde_json::from_str(&data).expect("valid coverage tree");
                    let status = data["nodes"]
                        .as_object()
                        .expect("coverage nodes")
                        .values()
                        .find(|node| node["aria"]["role"] == "Status");
                    if label.is_empty() {
                        assert!(
                            status.is_none(),
                            "hidden summary must not expose old status"
                        );
                    } else {
                        let status = status.expect("production coverage status");
                        assert_eq!(status["aria"]["value"], label, "{status}");
                        assert_eq!(status["aria"]["label"], "Source coverage status");
                        assert_eq!(status["aria"]["live"], "Polite");
                        if let Some(previous) = &previous_status_id {
                            assert_eq!(&status["accesskit_id"], previous);
                        }
                        previous_status_id = Some(status["accesskit_id"].clone());
                    }
                });
            }
        };
        assert_accessibility_status(cx);
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let snapshot = SourceCoverageSnapshot {
            project_generation: 1,
            provider_id: SourceCoverageProviderId("generic-python".into()),
            generation: 1,
            status: SourceCoverageStatus::Current,
            truncated: false,
            diagnostic: None,
            files: vec![
                (first.clone(), 1, 0),
                (second.clone(), 0, 1),
                (private, 1, 0),
            ]
            .into_iter()
            .map(
                |(path, covered_lines, uncovered_lines)| SourceCoverageFile {
                    path,
                    ranges: vec![SourceCoverageRange {
                        start: SourceCoveragePoint { line: 0, column: 0 },
                        end: SourceCoveragePoint { line: 0, column: 1 },
                        hit_count: u64::from(covered_lines),
                    }],
                    covered_lines,
                    uncovered_lines,
                    truncated: false,
                },
            )
            .collect(),
        };
        store
            .update(cx, |store, cx| store.publish(snapshot.clone(), cx))
            .expect("publish generic coverage");
        cx.run_until_parked();
        assert_accessibility_status(cx);
        workspace.read_with(cx, |workspace, cx| {
            assert_eq!(
                workspace
                    .active_item_as::<Editor>(cx)
                    .expect("first editor")
                    .read(cx)
                    .source_coverage_gutter_counts(),
                (1, 0)
            );
        });
        summary.read_with(cx, |summary, _| {
            assert!(summary.label.contains("current"));
            assert_eq!(
                summary
                    .projection
                    .as_ref()
                    .expect("live projection")
                    .navigation
                    .len(),
                2,
                "private source is excluded"
            );
        });
        summary.update(cx, |summary, cx| {
            summary.filter = SourceCoverageFilter::Uncovered;
            summary.refresh(cx);
        });
        summary.read_with(cx, |summary, _| {
            let projection = summary.projection.as_ref().expect("filtered projection");
            assert_eq!(projection.navigation.len(), 1);
            assert_eq!(projection.navigation.values().next(), Some(&second));
        });
        summary.update_in(cx, |summary, window, cx| {
            summary.open_source(&(&snapshot).into(), second.clone(), window, cx)
        });
        cx.run_until_parked();
        assert_accessibility_status(cx);
        let editor = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_item_as::<Editor>(cx)
                .expect("navigated editor")
        });
        editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.source_coverage_gutter_counts(), (0, 1));
            assert_eq!(
                editor
                    .buffer()
                    .read(cx)
                    .as_singleton()
                    .expect("single source")
                    .read(cx)
                    .project_path(cx),
                Some(second.clone())
            );
        });
        let alternate = SourceCoverageProviderId("alternate".into());
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        provider_id: alternate.clone(),
                        files: vec![SourceCoverageFile {
                            path: second.clone(),
                            ranges: vec![SourceCoverageRange {
                                start: SourceCoveragePoint { line: 0, column: 0 },
                                end: SourceCoveragePoint { line: 0, column: 1 },
                                hit_count: 1,
                            }],
                            covered_lines: 1,
                            uncovered_lines: 0,
                            truncated: false,
                        }],
                        ..snapshot.clone()
                    },
                    cx,
                )
            })
            .expect("publish another generic provider");
        cx.run_until_parked();
        assert_accessibility_status(cx);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1),
            "new providers do not silently change selection"
        );
        summary.update(cx, |summary, cx| summary.select_provider(alternate, cx));
        cx.run_until_parked();
        assert_accessibility_status(cx);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0)
        );
        summary.update_in(cx, |summary, window, cx| {
            let identity = summary
                .projection
                .as_ref()
                .expect("selected provider projection")
                .identity
                .clone();
            summary.open_source(&identity, first.clone(), window, cx)
        });
        cx.run_until_parked();
        assert_accessibility_status(cx);
        workspace.read_with(cx, |workspace, cx| {
            assert_eq!(
                workspace
                    .active_item_as::<Editor>(cx)
                    .expect("selected provider source remains active")
                    .entity_id(),
                editor.entity_id(),
                "a source absent from the selected provider must not open"
            );
        });
        summary.update(cx, |summary, cx| {
            summary.select_provider(snapshot.provider_id.clone(), cx)
        });
        cx.run_until_parked();
        assert_accessibility_status(cx);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        project.update(cx, |_, cx| cx.emit(project::Event::DisconnectedFromHost));
        cx.run_until_parked();
        assert_accessibility_status(cx);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        summary.read_with(cx, |summary, _| {
            assert!(summary.label.contains("disconnected"));
            let root = summary
                .projection
                .as_ref()
                .expect("disconnected projection")
                .roots
                .first()
                .expect("coverage root");
            assert!(root.accessibility_label.contains("disconnected"));
            assert!(root.children.is_empty());
        });
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        generation: 2,
                        ..snapshot.clone()
                    },
                    cx,
                )
            })
            .expect("new coverage after disconnection");
        cx.run_until_parked();
        assert_accessibility_status(cx);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        let current_identity = summary.read_with(cx, |summary, _| {
            summary
                .projection
                .as_ref()
                .expect("current report projection")
                .identity
                .clone()
        });
        editor.update_in(cx, |editor, window, cx| {
            editor.insert("changed", window, cx)
        });
        cx.run_until_parked();
        assert_accessibility_status(cx);
        assert!(summary.read_with(cx, |summary, _| summary.label.contains("stale")));
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        summary.update_in(cx, |summary, window, cx| {
            summary.open_source(&current_identity, first.clone(), window, cx)
        });
        cx.run_until_parked();
        assert_accessibility_status(cx);
        workspace.read_with(cx, |workspace, cx| {
            let active = workspace
                .active_item_as::<Editor>(cx)
                .expect("the selected coverage source remains active");
            assert_eq!(
                active.read_with(cx, |editor, cx| {
                    editor
                        .buffer()
                        .read(cx)
                        .as_singleton()
                        .expect("single source")
                        .read(cx)
                        .project_path(cx)
                }),
                Some(second.clone()),
                "a former coverage link must not open after the report becomes stale"
            );
        });
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        generation: 3,
                        status: SourceCoverageStatus::Empty,
                        files: Vec::new(),
                        ..snapshot.clone()
                    },
                    cx,
                )
            })
            .expect("replace coverage");
        cx.run_until_parked();
        assert_accessibility_status(cx);
        summary.read_with(cx, |summary, _| {
            assert!(summary.label.contains("empty"));
            assert!(
                summary
                    .projection
                    .as_ref()
                    .expect("empty projection")
                    .navigation
                    .is_empty()
            );
        });
        for (generation, status, label, retains_files) in [
            (4, SourceCoverageStatus::Partial, "partial", false),
            (5, SourceCoverageStatus::Loading, "loading", true),
            (6, SourceCoverageStatus::Error, "error", true),
            (7, SourceCoverageStatus::Mismatch, "host mismatch", true),
            (8, SourceCoverageStatus::Restricted, "restricted", true),
            (9, SourceCoverageStatus::Cancelled, "cancelled", true),
        ] {
            store
                .update(cx, |store, cx| {
                    store.publish(
                        SourceCoverageSnapshot {
                            generation,
                            status,
                            files: if retains_files {
                                snapshot.files.clone()
                            } else {
                                Vec::new()
                            },
                            diagnostic: Some("/private/host TOKEN=secret".into()),
                            ..snapshot.clone()
                        },
                        cx,
                    )
                })
                .expect("publish coverage status");
            cx.run_until_parked();
            assert_accessibility_status(cx);
            summary.read_with(cx, |summary, _| {
                assert!(summary.label.contains(label));
                assert!(!summary.label.contains("/private/host"));
                assert!(!summary.label.contains("TOKEN=secret"));
                let projection = summary.projection.as_ref().expect("status projection");
                assert!(
                    projection
                        .roots
                        .first()
                        .expect("coverage root")
                        .accessibility_label
                        .contains(label)
                );
                assert!(
                    projection.navigation.is_empty(),
                    "{label} must not expose source links from cached report files"
                );
            });
            assert_eq!(
                editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
                (0, 0),
                "{label} must not expose cached gutter coverage"
            );
        }
        let source_buffer = editor.read_with(cx, |editor, cx| {
            editor
                .buffer()
                .read(cx)
                .as_singleton()
                .expect("edited source buffer")
        });
        assert!(source_buffer.read_with(cx, |buffer, _| buffer.is_dirty()));
        editor.update_in(cx, |editor, window, cx| {
            editor.undo(&Default::default(), window, cx)
        });
        cx.run_until_parked();
        source_buffer.read_with(cx, |buffer, _| {
            assert!(!buffer.is_dirty());
            assert_eq!(buffer.text(), "b = 2\n");
        });
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        generation: 10,
                        ..snapshot.clone()
                    },
                    cx,
                )
            })
            .expect("fresh report after cancellation");
        cx.run_until_parked();
        assert_accessibility_status(cx);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        summary.read_with(cx, |summary, _| {
            assert_eq!(summary.filter, SourceCoverageFilter::Uncovered);
            let navigation = &summary
                .projection
                .as_ref()
                .expect("fresh projection")
                .navigation;
            assert_eq!(navigation.len(), 1);
            assert_eq!(navigation.values().next(), Some(&second));
        });
        summary.update(cx, |summary, cx| {
            summary.filter = SourceCoverageFilter::All;
            summary.refresh(cx);
        });
        cx.run_until_parked();
        assert_accessibility_status(cx);
        summary.read_with(cx, |summary, _| {
            assert!(summary.label.contains("current"));
            assert_eq!(
                summary
                    .projection
                    .as_ref()
                    .expect("fresh projection")
                    .navigation
                    .len(),
                2
            );
        });
        summary.update_in(cx, |summary, window, cx| {
            summary.set_active_pane_item(None, window, cx);
        });
        assert!(
            summary.read_with(cx, |summary, _| summary.subscriptions.is_empty()
                && summary.label.is_empty())
        );
        assert_accessibility_status(cx);
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn source_coverage_summary_production_rejects_obsolete_same_path_navigation(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/coverage",
            serde_json::json!({"a.py": "a = 1\n", "b.py": "b = 2\n"}),
        )
        .await;
        let project = Project::test(fs, ["/coverage".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let (first, second) = project.read_with(cx, |project, cx| {
            (
                project
                    .find_project_path("/coverage/a.py", cx)
                    .expect("initial source"),
                project
                    .find_project_path("/coverage/b.py", cx)
                    .expect("covered source"),
            )
        });
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(first.clone(), None, true, window, cx)
            })
            .await
            .expect("open initial source");
        cx.run_until_parked();
        let summary = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_pane()
                .read(cx)
                .toolbar()
                .read(cx)
                .item_of_type::<SourceCoverageToolbar>()
                .expect("production coverage toolbar")
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let snapshot = SourceCoverageSnapshot {
            project_generation: 1,
            provider_id: SourceCoverageProviderId("generic-python".into()),
            generation: 1,
            status: SourceCoverageStatus::Current,
            truncated: false,
            diagnostic: None,
            files: vec![SourceCoverageFile {
                path: second.clone(),
                ranges: vec![SourceCoverageRange {
                    start: SourceCoveragePoint { line: 0, column: 0 },
                    end: SourceCoveragePoint { line: 0, column: 1 },
                    hit_count: 1,
                }],
                covered_lines: 1,
                uncovered_lines: 0,
                truncated: false,
            }],
        };
        store
            .update(cx, |store, cx| store.publish(snapshot.clone(), cx))
            .expect("publish initial report");
        cx.run_until_parked();
        for generation in [1, 2] {
            let obsolete = summary.read_with(cx, |summary, _| {
                summary
                    .projection
                    .as_ref()
                    .expect("original report projection")
                    .identity
                    .clone()
            });
            let replacement = SourceCoverageSnapshot {
                provider_id: SourceCoverageProviderId("alternate".into()),
                generation,
                ..snapshot.clone()
            };
            store
                .update(cx, |store, cx| store.publish(replacement.clone(), cx))
                .expect("publish replacement report containing the same source");
            summary.update(cx, |summary, cx| {
                summary.select_provider(replacement.provider_id.clone(), cx)
            });
            cx.run_until_parked();
            summary.update_in(cx, |summary, window, cx| {
                summary.open_source(&obsolete, second.clone(), window, cx)
            });
            cx.run_until_parked();
            workspace.read_with(cx, |workspace, cx| {
                let editor = workspace
                    .active_item_as::<Editor>(cx)
                    .expect("initial editor remains active");
                assert_eq!(
                    editor
                        .read(cx)
                        .buffer()
                        .read(cx)
                        .as_singleton()
                        .expect("initial source buffer")
                        .read(cx)
                        .project_path(cx),
                    Some(first.clone()),
                    "a former provider or run must not navigate even when its path remains covered"
                );
            });
            summary.update_in(cx, |summary, window, cx| {
                summary.open_source(&(&replacement).into(), second.clone(), window, cx)
            });
            cx.run_until_parked();
            workspace.read_with(cx, |workspace, cx| {
                let editor = workspace
                    .active_item_as::<Editor>(cx)
                    .expect("current report opens the covered source");
                assert_eq!(
                    editor
                        .read(cx)
                        .buffer()
                        .read(cx)
                        .as_singleton()
                        .expect("covered source buffer")
                        .read(cx)
                        .project_path(cx),
                    Some(second.clone())
                );
            });
            workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.open_path(first.clone(), None, true, window, cx)
                })
                .await
                .expect("return to the initial source");
            cx.run_until_parked();
        }
        let actionable = SourceCoverageSnapshot {
            generation: 3,
            ..snapshot.clone()
        };
        store
            .update(cx, |store, cx| store.publish(actionable.clone(), cx))
            .expect("current report before cancellation");
        summary.update(cx, |summary, cx| {
            summary.select_provider(actionable.provider_id.clone(), cx)
        });
        cx.run_until_parked();
        let action_identity = summary.read_with(cx, |summary, _| {
            summary
                .projection
                .as_ref()
                .expect("current action")
                .identity
                .clone()
        });
        store
            .update(cx, |store, cx| {
                store.mark_provider_status(
                    actionable.provider_id.clone(),
                    SourceCoverageStatus::Cancelled,
                    Some("host-private detail".into()),
                    cx,
                )
            })
            .expect("cancel the same report");
        cx.run_until_parked();
        summary.update_in(cx, |summary, window, cx| {
            summary.open_source(&action_identity, second.clone(), window, cx)
        });
        cx.run_until_parked();
        summary.read_with(cx, |summary, _| {
            assert!(summary.label.contains("cancelled"));
            assert!(!summary.label.contains("host-private detail"));
            assert!(
                summary
                    .projection
                    .as_ref()
                    .expect("cancelled projection")
                    .navigation
                    .is_empty()
            );
        });
        workspace.read_with(cx, |workspace, cx| {
            let editor = workspace
                .active_item_as::<Editor>(cx)
                .expect("original editor");
            assert_eq!(
                editor
                    .read(cx)
                    .buffer()
                    .read(cx)
                    .as_singleton()
                    .expect("single source")
                    .read(cx)
                    .project_path(cx),
                Some(first.clone()),
                "a cancelled report must reject its formerly current source action"
            );
            assert_eq!(editor.read(cx).source_coverage_gutter_counts(), (0, 0));
        });
        let recovered = SourceCoverageSnapshot {
            generation: 4,
            ..actionable
        };
        store
            .update(cx, |store, cx| store.publish(recovered.clone(), cx))
            .expect("fresh current report after cancellation");
        cx.run_until_parked();
        summary.update_in(cx, |summary, window, cx| {
            summary.open_source(&(&recovered).into(), second.clone(), window, cx)
        });
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            let editor = workspace
                .active_item_as::<Editor>(cx)
                .expect("recovered editor");
            assert_eq!(
                editor
                    .read(cx)
                    .buffer()
                    .read(cx)
                    .as_singleton()
                    .expect("single source")
                    .read(cx)
                    .project_path(cx),
                Some(second.clone())
            );
            assert_eq!(editor.read(cx).source_coverage_gutter_counts(), (1, 0));
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn source_coverage_summary_production_excludes_invisible_worktree(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/coverage",
            serde_json::json!({"visible.py": "visible = 1\n"}),
        )
        .await;
        fs.insert_tree("/hidden", serde_json::json!({"hidden.py": "hidden = 1\n"}))
            .await;
        let project = Project::test(fs, ["/coverage".as_ref()], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let hidden_worktree = worktree_store
            .update(cx, |store, cx| store.create_worktree("/hidden", false, cx))
            .await
            .expect("create invisible worktree");
        cx.run_until_parked();
        let visible = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/coverage/visible.py", cx)
                .expect("visible source")
        });
        let hidden = ProjectPath {
            worktree_id: hidden_worktree.read_with(cx, |worktree, _| worktree.id()),
            path: Arc::from(RelPath::from_unix_str("hidden.py").expect("hidden source path")),
        };
        project.read_with(cx, |project, cx| {
            assert!(project.entry_for_path(&hidden, cx).is_some());
            assert!(
                !project
                    .visible_worktrees(cx)
                    .any(|worktree| worktree.read(cx).id() == hidden.worktree_id)
            );
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let editor = workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(visible.clone(), None, true, window, cx)
            })
            .await
            .expect("open visible source")
            .downcast::<Editor>()
            .expect("source editor");
        cx.run_until_parked();
        let summary = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_pane()
                .read(cx)
                .toolbar()
                .read(cx)
                .item_of_type::<SourceCoverageToolbar>()
                .expect("production toolbar")
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        project_generation: 1,
                        provider_id: SourceCoverageProviderId("generic-python".into()),
                        generation: 1,
                        status: SourceCoverageStatus::Current,
                        truncated: false,
                        diagnostic: None,
                        files: [visible.clone(), hidden.clone()]
                            .into_iter()
                            .map(|path| SourceCoverageFile {
                                path,
                                ranges: vec![SourceCoverageRange {
                                    start: SourceCoveragePoint { line: 0, column: 0 },
                                    end: SourceCoveragePoint { line: 0, column: 1 },
                                    hit_count: 1,
                                }],
                                covered_lines: 1,
                                uncovered_lines: 0,
                                truncated: false,
                            })
                            .collect(),
                    },
                    cx,
                )
            })
            .expect("publish coverage containing an invisible worktree source");
        cx.run_until_parked();
        summary.read_with(cx, |summary, _| {
            let projection = summary.projection.as_ref().expect("production projection");
            assert!(summary.label.contains("current"));
            assert_eq!(projection.navigation.len(), 1);
            assert_eq!(projection.navigation.values().next(), Some(&visible));
        });
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0)
        );
        summary.update_in(cx, |summary, window, cx| {
            let identity = summary
                .projection
                .as_ref()
                .expect("production projection")
                .identity
                .clone();
            summary.open_source(&identity, hidden, window, cx)
        });
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            assert_eq!(
                workspace
                    .active_item_as::<Editor>(cx)
                    .expect("visible editor remains active")
                    .entity_id(),
                editor.entity_id(),
                "invisible source action must not open an editor"
            );
        });
    }

    #[gpui::test]
    async fn source_coverage_summary_production_revokes_trust_and_private_source(
        cx: &mut gpui::TestAppContext,
    ) {
        use collections::HashSet;
        use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            trusted_worktrees::init(DbTrustedPaths::default(), cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/coverage", serde_json::json!({"b.py": "b = 2\n"}))
            .await;
        let project = Project::test_with_worktree_trust(fs, ["/coverage".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let source = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/coverage/b.py", cx)
                .expect("coverage source")
        });
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let worktree_id = source.worktree_id;
        let trusted_worktrees =
            cx.update(|_, cx| TrustedWorktrees::try_get_global(cx).expect("trusted worktrees"));
        trusted_worktrees.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(source.clone(), None, true, window, cx)
            })
            .await
            .expect("open coverage source");
        cx.run_until_parked();
        let editor = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_item_as::<Editor>(cx)
                .expect("source editor")
        });
        let summary = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_pane()
                .read(cx)
                .toolbar()
                .read(cx)
                .item_of_type::<SourceCoverageToolbar>()
                .expect("production coverage toolbar")
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let snapshot = SourceCoverageSnapshot {
            project_generation: 1,
            provider_id: SourceCoverageProviderId("generic-python".into()),
            generation: 1,
            status: SourceCoverageStatus::Current,
            truncated: false,
            diagnostic: None,
            files: vec![SourceCoverageFile {
                path: source.clone(),
                ranges: vec![SourceCoverageRange {
                    start: SourceCoveragePoint { line: 0, column: 0 },
                    end: SourceCoveragePoint { line: 0, column: 1 },
                    hit_count: 0,
                }],
                covered_lines: 0,
                uncovered_lines: 1,
                truncated: false,
            }],
        };
        store
            .update(cx, |store, cx| store.publish(snapshot.clone(), cx))
            .expect("publish coverage");
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        assert!(summary.read_with(cx, |summary, _| {
            summary.projection.as_ref().is_some_and(|projection| {
                projection.navigation.values().any(|path| path == &source)
            })
        }));

        trusted_worktrees.update(cx, |trust, cx| {
            trust.restrict(
                worktree_store.downgrade(),
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
        cx.run_until_parked();
        summary.read_with(cx, |summary, _| {
            assert!(summary.label.contains("restricted"));
            let projection = summary.projection.as_ref().expect("restricted projection");
            assert!(!projection.navigation.values().any(|path| path == &source));
            let root = projection.roots.first().expect("coverage root");
            assert!(root.accessibility_label.contains("restricted"));
            assert!(root.children.is_empty());
        });
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        trusted_worktrees.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            );
        });
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        generation: 2,
                        ..snapshot.clone()
                    },
                    cx,
                )
            })
            .expect("publish coverage after restoring trust");
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        assert!(summary.read_with(cx, |summary, _| {
            summary.label.contains("current")
                && summary.projection.as_ref().is_some_and(|projection| {
                    projection.navigation.values().any(|path| path == &source)
                })
        }));

        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/b.py".into()]));
                });
            });
        });
        cx.run_until_parked();
        assert!(project.read_with(cx, |project, cx| {
            project
                .entry_for_path(&source, cx)
                .is_some_and(|entry| entry.is_private)
        }));
        assert!(summary.read_with(cx, |summary, _| {
            summary.projection.as_ref().is_some_and(|projection| {
                !projection.navigation.values().any(|path| path == &source)
            })
        }));
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );

        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = None;
                });
            });
        });
        cx.run_until_parked();
        assert!(project.read_with(cx, |project, cx| {
            project
                .entry_for_path(&source, cx)
                .is_some_and(|entry| !entry.is_private)
        }));
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        generation: 3,
                        ..snapshot
                    },
                    cx,
                )
            })
            .expect("publish coverage before worktree removal");
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        assert!(summary.read_with(cx, |summary, _| {
            summary.projection.as_ref().is_some_and(|projection| {
                projection.navigation.values().any(|path| path == &source)
            })
        }));
        project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        assert!(summary.read_with(cx, |summary, _| {
            summary.label.contains("stale")
                && summary
                    .projection
                    .as_ref()
                    .is_some_and(|projection| projection.navigation.is_empty())
        }));
    }

    #[gpui::test]
    async fn source_coverage_summary_production_maps_multiple_excerpts(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/coverage",
            serde_json::json!({"a.py": "skip\nhit\nskip\n", "b.py": "skip\nmiss\nskip\n"}),
        )
        .await;
        let project = Project::test(fs, ["/coverage".as_ref()], cx).await;
        let (first_path, second_path) = project.read_with(cx, |project, cx| {
            (
                project
                    .find_project_path("/coverage/a.py", cx)
                    .expect("first path"),
                project
                    .find_project_path("/coverage/b.py", cx)
                    .expect("second path"),
            )
        });
        let first = project
            .update(cx, |project, cx| {
                project.open_buffer(first_path.clone(), cx)
            })
            .await
            .expect("first buffer");
        let second = project
            .update(cx, |project, cx| {
                project.open_buffer(second_path.clone(), cx)
            })
            .await
            .expect("second buffer");
        let multibuffer = cx.new(|cx| {
            let mut buffer = editor::MultiBuffer::new(language::Capability::ReadWrite);
            buffer.set_excerpts_for_path(
                editor::PathKey::sorted(0),
                first.clone(),
                [language::Point::new(1, 0)..language::Point::new(1, 3)],
                0,
                cx,
            );
            buffer.set_excerpts_for_path(
                editor::PathKey::sorted(1),
                second.clone(),
                [language::Point::new(1, 0)..language::Point::new(1, 4)],
                0,
                cx,
            );
            buffer
        });
        let (editor, cx) = cx.add_window_view(|window, cx| {
            Editor::new(
                editor::EditorMode::full(),
                multibuffer.clone(),
                Some(project.clone()),
                window,
                cx,
            )
        });
        cx.run_until_parked();
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        project_generation: 1,
                        provider_id: SourceCoverageProviderId("generic-excerpts".into()),
                        generation: 1,
                        status: SourceCoverageStatus::Current,
                        truncated: false,
                        diagnostic: None,
                        files: vec![(first_path, 1), (second_path, 0)]
                            .into_iter()
                            .map(|(path, hit_count)| SourceCoverageFile {
                                path,
                                ranges: vec![SourceCoverageRange {
                                    start: SourceCoveragePoint { line: 0, column: 0 },
                                    end: SourceCoveragePoint { line: 2, column: 0 },
                                    hit_count,
                                }],
                                covered_lines: u32::from(hit_count > 0),
                                uncovered_lines: u32::from(hit_count == 0),
                                truncated: false,
                            })
                            .collect(),
                    },
                    cx,
                )
            })
            .expect("publish multi-file coverage");
        cx.run_until_parked();
        editor.read_with(cx, |editor, cx| {
            let (covered, uncovered) = editor.source_coverage_gutter_points(cx);
            assert_eq!((covered.len(), uncovered.len()), (1, 1));
            let text = editor.buffer().read(cx).snapshot(cx).text();
            let hit_row = text
                .lines()
                .position(|line| line == "hit")
                .expect("first excerpt") as u32;
            let miss_row = text
                .lines()
                .position(|line| line == "miss")
                .expect("second excerpt") as u32;
            assert_eq!(
                covered.first().expect("covered range").start,
                language::Point::new(hit_row, 0)
            );
            assert_eq!(
                uncovered.first().expect("uncovered range").start,
                language::Point::new(miss_row, 0)
            );
        });
        multibuffer.update(cx, |buffer, cx| {
            buffer.remove_excerpts(editor::PathKey::sorted(0), cx)
        });
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        multibuffer.update(cx, |buffer, cx| {
            buffer.set_excerpts_for_path(
                editor::PathKey::sorted(0),
                first.clone(),
                [language::Point::new(1, 0)..language::Point::new(1, 3)],
                0,
                cx,
            );
        });
        store.update(cx, |store, cx| {
            let mut snapshot = store.selected_snapshot().expect("retained report").clone();
            snapshot.generation += 1;
            for file in &mut snapshot.files {
                for range in &mut file.ranges {
                    range.hit_count = 1;
                }
                file.ranges.extend(file.ranges.clone());
                file.covered_lines = 1;
                file.uncovered_lines = 0;
            }
            store.publish(snapshot, cx).expect("covered sources");
        });
        cx.run_until_parked();
        editor.read_with(cx, |editor, cx| {
            let (covered, uncovered) = editor.source_coverage_gutter_points(cx);
            assert_eq!(covered.len(), 2);
            assert!(uncovered.is_empty());
            assert!(covered.windows(2).all(|pair| pair[0].end < pair[1].start));
        });
    }

    #[test]
    fn source_coverage_summary_filters_navigates_and_exposes_partial_state() {
        let snapshot = SourceCoverageSnapshot {
            project_generation: 1,
            provider_id: SourceCoverageProviderId("fake".to_string()),
            generation: 2,
            status: SourceCoverageStatus::Partial,
            files: vec![
                SourceCoverageFile {
                    path: path("src/covered.rs"),
                    ranges: Vec::new(),
                    covered_lines: 9,
                    uncovered_lines: 1,
                    truncated: false,
                },
                SourceCoverageFile {
                    path: path("src/uncovered.rs"),
                    ranges: Vec::new(),
                    covered_lines: 0,
                    uncovered_lines: 4,
                    truncated: true,
                },
            ],
            truncated: true,
            diagnostic: Some("partial report".to_string()),
        };
        let projection =
            SourceCoverageSummaryProjection::project(&snapshot, SourceCoverageFilter::Uncovered);
        assert_eq!(projection.roots[0].children.len(), 2);
        assert!(
            projection.roots[0]
                .secondary_label
                .as_deref()
                .is_some_and(|label| label.contains("partial"))
        );
        let file = &projection.roots[0].children[0];
        assert!(projection.navigation(&file.id).is_some());
        assert!(file.children.is_empty());
    }

    #[test]
    fn source_coverage_summary_projects_only_the_visible_window_at_scale() {
        let ranges = (0..100_000)
            .map(|line| SourceCoverageRange {
                start: SourceCoveragePoint { line, column: 0 },
                end: SourceCoveragePoint { line, column: 1 },
                hit_count: u64::from(line % 2),
            })
            .collect::<Vec<_>>();
        let summary =
            editor::source_coverage::summarize_visible_source_coverage(&ranges, 50_000..50_100);
        assert_eq!(summary.covered, 50);
        assert_eq!(summary.uncovered, 50);
    }

    #[gpui::test]
    fn source_coverage_summary_applies_replaces_and_clears_editor_gutter_markers(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            settings::init(cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
            editor::init(cx);
        });
        let editor = cx.add_window(|window, cx| {
            let buffer = editor::MultiBuffer::build_simple("covered\nuncovered\n", cx);
            editor::Editor::new(editor::EditorMode::full(), buffer, None, window, cx)
        });
        let file = SourceCoverageFile {
            path: path("src/lib.rs"),
            ranges: vec![
                SourceCoverageRange {
                    start: SourceCoveragePoint { line: 0, column: 0 },
                    end: SourceCoveragePoint { line: 0, column: 1 },
                    hit_count: 1,
                },
                SourceCoverageRange {
                    start: SourceCoveragePoint { line: 1, column: 0 },
                    end: SourceCoveragePoint { line: 1, column: 1 },
                    hit_count: 0,
                },
            ],
            covered_lines: 1,
            uncovered_lines: 1,
            truncated: false,
        };
        editor
            .update(cx, |editor, _window, cx| {
                editor.apply_source_coverage(Some(&file), cx);
                assert_eq!(editor.source_coverage_gutter_counts(), (1, 1));
                editor.apply_source_coverage(
                    Some(&SourceCoverageFile {
                        ranges: vec![file.ranges[0].clone()],
                        covered_lines: 1,
                        uncovered_lines: 0,
                        ..file.clone()
                    }),
                    cx,
                );
                assert_eq!(editor.source_coverage_gutter_counts(), (1, 0));
                editor.apply_source_coverage(None, cx);
                assert_eq!(editor.source_coverage_gutter_counts(), (0, 0));
            })
            .expect("coverage editor window should remain open");
    }
}
