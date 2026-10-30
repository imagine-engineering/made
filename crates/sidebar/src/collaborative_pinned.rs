use std::{cmp::Reverse, collections::HashMap};

use agent_ui::thread_metadata_store::{ThreadMetadata, ThreadMetadataStore};
use channel::{Channel, ChannelStore};
use chrono::{DateTime, Utc};
use gpui::{
    Action, Context, InteractiveElement, KeyDownEvent, Render, SharedString, Task, WeakEntity,
};
use ui::{Window, prelude::*};
use workspace::{
    MultiWorkspace, RecentWorkspace, WorkspaceDb,
    collaborative_navigation::CollaborativeNavigationTarget,
};

use crate::collaborative_navigation::{
    CollaborativeNavigationBadge, CollaborativeNavigationProjection, CollaborativeNavigationRow,
    CollaborativeNavigationSourceId, navigation_action, navigation_icon, navigation_label,
    navigation_row, render_collaborative_navigation_badges,
};
use crate::{
    collaborative_projects::activate_project_target, collaborative_tasks::activate_thread_target,
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum CollaborativePinnedState {
    Loading,
    Ready,
    Unavailable(SharedString),
}

#[derive(Clone)]
struct RecentNavigationCandidate {
    updated_at: DateTime<Utc>,
    canonical_row: CollaborativeNavigationRow,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CollaborativePinnedProjection {
    navigation: CollaborativeNavigationProjection,
    missing_targets: Vec<CollaborativeNavigationTarget>,
    pinned_sources: std::collections::HashSet<CollaborativeNavigationSourceId>,
}

impl CollaborativePinnedProjection {
    fn from_sources(
        pinned_targets: &[CollaborativeNavigationTarget],
        recent_projects: &[RecentWorkspace],
        recent_threads: impl IntoIterator<Item = ThreadMetadata>,
        channels: impl IntoIterator<Item = (Channel, bool)>,
    ) -> Result<Self, SharedString> {
        let mut recent_candidates = Vec::new();
        for workspace in recent_projects {
            let project_row = CollaborativeNavigationRow::from_project_group(
                workspace.project_group_key(),
                workspace.project_group_key().display_name(&HashMap::new()),
                Vec::new(),
            );
            recent_candidates.push(RecentNavigationCandidate {
                updated_at: workspace.timestamp,
                canonical_row: project_row,
            });
        }
        for thread in recent_threads {
            if !thread.archived {
                recent_candidates.push(RecentNavigationCandidate {
                    updated_at: thread.updated_at,
                    canonical_row: CollaborativeNavigationRow::from_thread(&thread, Vec::new()),
                });
            }
        }
        recent_candidates.sort_by_key(|candidate| Reverse(candidate.updated_at));

        let mut candidates_by_source = HashMap::new();
        for candidate in &recent_candidates {
            let source_id = candidate.canonical_row.source_id().clone();
            if candidates_by_source
                .insert(source_id.clone(), candidate.canonical_row.clone())
                .is_some()
            {
                return Err(format!("duplicate recent source: {source_id:?}").into());
            }
        }
        for (channel, unread) in channels {
            let badges = unread
                .then_some(CollaborativeNavigationBadge::NotesChanged)
                .into_iter()
                .collect();
            let row = CollaborativeNavigationRow::from_channel(&channel, badges);
            let source_id = row.source_id().clone();
            if candidates_by_source
                .insert(source_id.clone(), row)
                .is_some()
            {
                return Err(format!("duplicate channel source: {source_id:?}").into());
            }
        }
        let mut rows = Vec::new();
        let mut missing_targets = Vec::new();
        let mut included_sources = std::collections::HashSet::new();
        let mut pinned_sources = std::collections::HashSet::new();
        for target in pinned_targets {
            let candidate = candidates_by_source
                .values()
                .find(|row| CollaborativePinned::target(row).as_ref() == Some(target));
            let Some(candidate) = candidate else {
                missing_targets.push(target.clone());
                continue;
            };
            let source_id = candidate.source_id().clone();
            if !included_sources.insert(source_id) {
                return Err(format!("duplicate pinned target: {target:?}").into());
            }
            pinned_sources.insert(candidate.source_id().clone());
            rows.push(CollaborativeNavigationRow::pinned(candidate, Vec::new()));
        }

        let navigation = CollaborativeNavigationProjection::try_from_rows(rows)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            navigation,
            missing_targets,
            pinned_sources,
        })
    }
}

pub(crate) struct CollaborativePinned {
    multi_workspace: Option<WeakEntity<MultiWorkspace>>,
    state: CollaborativePinnedState,
    activation_error: Option<SharedString>,
    recent_projects: Vec<RecentWorkspace>,
    _load_task: Task<()>,
}

impl CollaborativePinned {
    pub(crate) fn new(multi_workspace: WeakEntity<MultiWorkspace>, cx: &mut Context<Self>) -> Self {
        if let Some(thread_store) = ThreadMetadataStore::try_global(cx) {
            cx.observe(&thread_store, |_, _, cx| cx.notify()).detach();
        }
        if let Some(channel_store) = ChannelStore::try_global(cx) {
            cx.observe(&channel_store, |_, _, cx| cx.notify()).detach();
        }

        let multi_workspace = multi_workspace.upgrade();
        let load_task = if let Some(multi_workspace) = multi_workspace.as_ref() {
            let fs = multi_workspace
                .read(cx)
                .workspace()
                .read(cx)
                .app_state()
                .fs
                .clone();
            let database = WorkspaceDb::global(cx);
            cx.spawn(async move |this, cx| {
                let result = database.recent_project_workspaces(fs.as_ref()).await;
                if let Err(error) = this.update(cx, |this, cx| match result {
                    Ok(recent_projects) => {
                        this.recent_projects = recent_projects;
                        this.state = CollaborativePinnedState::Ready;
                        cx.notify();
                    }
                    Err(error) => {
                        this.state = CollaborativePinnedState::Unavailable(
                            format!("Recent work is unavailable: {error}").into(),
                        );
                        cx.notify();
                    }
                }) {
                    log::error!("failed to update collaborative recent work: {error:#}");
                }
            })
        } else {
            Task::ready(())
        };

        let has_multi_workspace = multi_workspace.is_some();
        let multi_workspace = multi_workspace
            .as_ref()
            .map(|multi_workspace| multi_workspace.downgrade());
        Self {
            multi_workspace,
            state: if has_multi_workspace {
                CollaborativePinnedState::Loading
            } else {
                CollaborativePinnedState::Unavailable(
                    "Recent work is unavailable without an active workspace".into(),
                )
            },
            activation_error: None,
            recent_projects: Vec::new(),
            _load_task: load_task,
        }
    }

    fn projection(&self, cx: &gpui::App) -> Result<CollaborativePinnedProjection, SharedString> {
        let recent_threads = ThreadMetadataStore::try_global(cx)
            .map(|store| store.read(cx).entries().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let channels = ChannelStore::try_global(cx)
            .map(|store| {
                let store = store.read(cx);
                store
                    .channels()
                    .map(|channel| {
                        (
                            channel.as_ref().clone(),
                            store.has_channel_buffer_changed(channel.id),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut pinned_targets = self
            .multi_workspace
            .as_ref()
            .and_then(WeakEntity::upgrade)
            .map(|multi_workspace| {
                let workspace = multi_workspace.read(cx).workspace().clone();
                workspace
                    .read(cx)
                    .collaborative_navigation()
                    .pinned()
                    .to_vec()
            })
            .unwrap_or_default();
        if let Some(channel_store) = ChannelStore::try_global(cx) {
            pinned_targets.extend(channel_store.read(cx).favorite_channel_ids().iter().map(
                |channel_id| CollaborativeNavigationTarget::channel(channel_id.0.to_string()),
            ));
        }
        CollaborativePinnedProjection::from_sources(
            &pinned_targets,
            &self.recent_projects,
            recent_threads,
            channels,
        )
    }

    fn target(row: &CollaborativeNavigationRow) -> Option<CollaborativeNavigationTarget> {
        match row.source_id() {
            CollaborativeNavigationSourceId::Project(project) => {
                Some(CollaborativeNavigationTarget::project(project))
            }
            CollaborativeNavigationSourceId::Thread(thread_id) => Some(
                CollaborativeNavigationTarget::thread(thread_id.to_key_string()),
            ),
            CollaborativeNavigationSourceId::Channel(channel_id) => Some(
                CollaborativeNavigationTarget::channel(channel_id.to_string()),
            ),
            _ => None,
        }
    }

    fn selected_target(&self, cx: &gpui::App) -> Option<CollaborativeNavigationTarget> {
        let multi_workspace = self.multi_workspace.as_ref()?.upgrade()?;
        let workspace = multi_workspace.read(cx).workspace().clone();
        workspace
            .read(cx)
            .collaborative_navigation()
            .current()
            .cloned()
    }

    fn unpin(
        &mut self,
        target: CollaborativeNavigationTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(multi_workspace) = self.multi_workspace.as_ref().and_then(WeakEntity::upgrade)
        else {
            self.activation_error = Some("Pinning is unavailable".into());
            cx.notify();
            return;
        };
        match &target {
            CollaborativeNavigationTarget::Channel { channel_id } => {
                let result = channel_id
                    .parse::<u64>()
                    .map_err(|_| SharedString::from("The pinned channel is invalid"))
                    .and_then(|channel_id| {
                        let channel_store = ChannelStore::try_global(cx).ok_or_else(|| {
                            SharedString::from("Channel favorites are unavailable")
                        })?;
                        channel_store.update(cx, |store, cx| {
                            let channel_id = {
                                store
                                    .channels()
                                    .find(|channel| channel.id.0 == channel_id)
                                    .map(|channel| channel.id)
                            };
                            if let Some(channel_id) = channel_id {
                                store.toggle_favorite_channel(channel_id, cx);
                            }
                        });
                        Ok(())
                    });
                self.activation_error = result.err();
            }
            _ => {
                let workspace = multi_workspace.read(cx).workspace().clone();
                self.activation_error = workspace
                    .update(cx, |workspace, cx| {
                        workspace.toggle_collaborative_pin(target, window, cx)
                    })
                    .err()
                    .map(|error| error.to_string().into());
            }
        }
        cx.notify();
    }

    fn activate(
        &mut self,
        target: CollaborativeNavigationTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(multi_workspace) = self.multi_workspace.as_ref().and_then(WeakEntity::upgrade)
        else {
            self.activation_error = Some("Navigation is unavailable".into());
            cx.notify();
            return;
        };
        let result = match &target {
            CollaborativeNavigationTarget::Project { .. }
            | CollaborativeNavigationTarget::Repository { .. }
            | CollaborativeNavigationTarget::Worktree { .. } => {
                activate_project_target(&multi_workspace, target, window, cx)
            }
            CollaborativeNavigationTarget::Thread { thread_id } => {
                let metadata = ThreadMetadataStore::try_global(cx).and_then(|store| {
                    store
                        .read(cx)
                        .entries()
                        .find(|metadata| metadata.thread_id.to_key_string() == *thread_id)
                        .cloned()
                });
                metadata.map_or_else(
                    || {
                        Err(SharedString::from(
                            "The selected thread is no longer available",
                        ))
                    },
                    |metadata| activate_thread_target(&multi_workspace, metadata, window, cx),
                )
            }
            CollaborativeNavigationTarget::Channel { channel_id } => channel_id
                .parse::<u64>()
                .map_err(|_| SharedString::from("The selected channel is invalid"))
                .and_then(|channel_id| {
                    let available = ChannelStore::try_global(cx).is_some_and(|store| {
                        store
                            .read(cx)
                            .channels()
                            .any(|channel| channel.id.0 == channel_id)
                    });
                    if !available {
                        return Err(SharedString::from(
                            "The selected channel is no longer available",
                        ));
                    }
                    let workspace = multi_workspace.read(cx).workspace().clone();
                    workspace
                        .update(cx, |workspace, cx| {
                            workspace.navigate_collaborative_to(
                                CollaborativeNavigationTarget::channel(channel_id.to_string()),
                                |_| true,
                                window,
                                cx,
                            )
                        })
                        .map_err(|error| SharedString::from(error.to_string()))?;
                    window.dispatch_action(
                        workspace::OpenChannelMessagesById { channel_id }.boxed_clone(),
                        cx,
                    );
                    Ok(())
                }),
            _ => Err("The selected item is unsupported".into()),
        };
        self.activation_error = result.err();
        cx.notify();
    }
}

impl Render for CollaborativePinned {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selected_target = self.selected_target(cx);
        let contents = match &self.state {
            CollaborativePinnedState::Loading => v_flex()
                .debug_selector(|| "COLLABORATIVE-PINNED-LOADING".to_owned())
                .child(
                    Label::new("Loading recent work…")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .into_any_element(),
            CollaborativePinnedState::Unavailable(message) => v_flex()
                .debug_selector(|| "COLLABORATIVE-PINNED-UNAVAILABLE".to_owned())
                .child(
                    Label::new(message.clone())
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .into_any_element(),
            CollaborativePinnedState::Ready => match self.projection(cx) {
                Ok(projection)
                    if projection.navigation.rows().is_empty()
                        && projection.missing_targets.is_empty() =>
                {
                    v_flex()
                        .debug_selector(|| "COLLABORATIVE-PINNED-EMPTY".to_owned())
                        .child(
                            Label::new("Pin a task from its actions menu, or use a project pin or channel star.")
                                .size(LabelSize::Small)
                                .color(Color::Muted),
                        )
                        .child(Button::new("pinned-open-project", "Open a project")
                            .size(ButtonSize::Compact)
                            .disabled(self.multi_workspace.as_ref().and_then(WeakEntity::upgrade).is_none())
                            .on_click(|_, window, cx| window.dispatch_action(zed_actions::OpenRecent::default().boxed_clone(), cx)))
                        .into_any_element()
                }
                Ok(projection) => {
                    let missing_target_count = projection.missing_targets.len();
                    v_flex()
                        .when_some(self.activation_error.clone(), |this, error| {
                            this.child(
                                Label::new(error)
                                    .size(LabelSize::XSmall)
                                    .color(Color::Error),
                            )
                        })
                        .when(missing_target_count > 0, |this| {
                            this.child(
                                Label::new(format!(
                                    "{missing_target_count} pinned item unavailable"
                                ))
                                .size(LabelSize::XSmall)
                                .color(Color::Muted),
                            )
                        })
                        .children(projection.navigation.rows().iter().map(|row| {
                            let badges = render_collaborative_navigation_badges(row);
                            let Some(target) = Self::target(row) else {
                                return Label::new("Pinned target is unavailable")
                                    .size(LabelSize::Small)
                                    .color(Color::Error)
                                    .into_any_element();
                            };
                            let keyboard_target = target.clone();
                            let unpin_target = target.clone();
                            let is_pinned = projection.pinned_sources.contains(row.source_id());
                            let selected = selected_target.as_ref() == Some(&target);
                            let unread = match row.source_id() {
                                CollaborativeNavigationSourceId::Thread(thread_id) => {
                                    ThreadMetadataStore::global(cx)
                                        .read(cx)
                                        .task_organization(*thread_id)
                                        .unread
                                }
                                _ => false,
                            };
                            let icon = match row.source_id() {
                                CollaborativeNavigationSourceId::Project(_) => IconName::Folder,
                                CollaborativeNavigationSourceId::Thread(_) => IconName::Thread,
                                CollaborativeNavigationSourceId::Channel(_) => IconName::Hash,
                                CollaborativeNavigationSourceId::Worktree { .. } => {
                                    IconName::Folder
                                }
                                CollaborativeNavigationSourceId::Repository { .. } => {
                                    IconName::GitBranch
                                }
                            };
                            let menu_thread = match row.source_id() {
                                CollaborativeNavigationSourceId::Thread(thread_id) => {
                                    Some(*thread_id)
                                }
                                _ => None,
                            };
                            let navigation_row = navigation_row(
                                SharedString::from(format!(
                                    "collaborative-pinned-target-{target:?}"
                                )),
                                selected,
                                0,
                                cx,
                            )
                            .aria_label(format!(
                                "Open pinned item {}{}",
                                row.label(),
                                if unread { ", unread" } else { "" }
                            ))
                            .child(navigation_icon(
                                icon,
                                if unread { Color::Info } else { Color::Muted },
                            ))
                            .child(navigation_label(row.label().clone()))
                            .when_some(badges, |this, badges| this.child(badges))
                            .when(is_pinned && menu_thread.is_none(), |this| {
                                this.child(
                                    navigation_action(
                                        SharedString::from(format!(
                                            "unpin-collaborative-target-{unpin_target:?}"
                                        )),
                                        IconName::Unpin,
                                        "Unpin",
                                    )
                                    .on_click(cx.listener(
                                        move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.unpin(unpin_target.clone(), window, cx);
                                        },
                                    )),
                                )
                            })
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.activate(target.clone(), window, cx);
                            }))
                            .on_key_down(cx.listener(
                                move |this, event: &KeyDownEvent, window, cx| {
                                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                        cx.stop_propagation();
                                        this.activate(keyboard_target.clone(), window, cx);
                                    }
                                },
                            ));
                            if let Some(thread_id) = menu_thread
                                && let Some(multi_workspace) = self.multi_workspace.clone()
                            {
                                crate::collaborative_task_menu::task_row_menu(
                                    navigation_row,
                                    thread_id,
                                    multi_workspace,
                                    cx,
                                )
                            } else {
                                navigation_row.into_any_element()
                            }
                        }))
                        .into_any_element()
                }
                Err(error) => v_flex()
                    .debug_selector(|| "COLLABORATIVE-PINNED-UNAVAILABLE".to_owned())
                    .child(Label::new(error).size(LabelSize::Small).color(Color::Muted))
                    .into_any_element(),
            },
        };

        v_flex()
            .debug_selector(|| "COLLABORATIVE-PINNED".to_owned())
            .child(contents)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use agent_ui::ThreadId;
    use agent_ui::thread_metadata_store::WorktreePaths;
    use chrono::TimeZone as _;
    use project::{AgentId, ProjectGroupKey};
    use util::path_list::PathList;
    use workspace::{SerializedWorkspaceLocation, WorkspaceId};

    use super::*;

    fn recent_project(path: &str, timestamp: DateTime<Utc>) -> RecentWorkspace {
        let paths = PathList::new(&[PathBuf::from(path)]);
        RecentWorkspace {
            workspace_id: WorkspaceId::from_i64(timestamp.timestamp()),
            location: SerializedWorkspaceLocation::Local,
            paths: paths.clone(),
            identity_paths: paths,
            timestamp,
        }
    }

    fn recent_thread(title: &str, timestamp: DateTime<Utc>, archived: bool) -> ThreadMetadata {
        ThreadMetadata {
            execution_outcome: acp_thread::ThreadExecutionOutcome::Unknown,
            thread_id: ThreadId::new(),
            session_id: None,
            agent_id: AgentId("test-agent".into()),
            title: Some(title.into()),
            title_override: None,
            updated_at: timestamp,
            created_at: None,
            interacted_at: None,
            worktree_paths: WorktreePaths::default(),
            remote_connection: None,
            archived,
        }
    }

    #[test]
    fn collaborative_pinned_does_not_include_unpinned_recent_work() {
        let older = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let newer = Utc.with_ymd_and_hms(2026, 1, 2, 0, 0, 0).unwrap();
        let project = recent_project("/workspace/project", older);
        let thread = recent_thread("newer thread", newer, false);
        let project_source = CollaborativeNavigationSourceId::Project(ProjectGroupKey::new(
            None,
            project.identity_paths.clone(),
        ));
        let project_target = CollaborativeNavigationTarget::project(match &project_source {
            CollaborativeNavigationSourceId::Project(project) => project,
            _ => unreachable!(),
        });

        let projection = CollaborativePinnedProjection::from_sources(
            std::slice::from_ref(&project_target),
            std::slice::from_ref(&project),
            [thread],
            [],
        )
        .expect("valid recent work should project");
        assert_eq!(projection.navigation.rows().len(), 1);
        assert_eq!(projection.navigation.rows()[0].source_id(), &project_source);
    }

    #[test]
    fn collaborative_pinned_removes_missing_and_archived_work() {
        let timestamp = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let project = recent_project("/workspace/project", timestamp);
        let project_source = CollaborativeNavigationSourceId::Project(ProjectGroupKey::new(
            None,
            project.identity_paths.clone(),
        ));
        let project_target = CollaborativeNavigationTarget::project(match &project_source {
            CollaborativeNavigationSourceId::Project(project) => project,
            _ => unreachable!(),
        });
        let archived = recent_thread("archived", timestamp, true);

        let present = CollaborativePinnedProjection::from_sources(
            std::slice::from_ref(&project_target),
            std::slice::from_ref(&project),
            [archived.clone()],
            [],
        )
        .expect("present target should project");
        assert_eq!(present.navigation.rows().len(), 1);

        let removed = CollaborativePinnedProjection::from_sources(
            std::slice::from_ref(&project_target),
            &[],
            [archived],
            [],
        )
        .expect("missing targets should remain observable");
        assert!(removed.navigation.rows().is_empty());
        assert_eq!(removed.missing_targets, vec![project_target]);
    }

    #[test]
    fn collaborative_pinned_rejects_duplicate_target_records() {
        let timestamp = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let project = recent_project("/workspace/project", timestamp);
        let source = CollaborativeNavigationSourceId::Project(ProjectGroupKey::new(
            None,
            project.identity_paths.clone(),
        ));
        let target = CollaborativeNavigationTarget::project(match &source {
            CollaborativeNavigationSourceId::Project(project) => project,
            _ => unreachable!(),
        });
        let error = CollaborativePinnedProjection::from_sources(
            &[target.clone(), target],
            std::slice::from_ref(&project),
            [],
            [],
        )
        .expect_err("duplicate target records should not silently collapse");
        assert!(error.contains("duplicate pinned target"));

        let duplicate_recent =
            CollaborativePinnedProjection::from_sources(&[], &[project.clone(), project], [], [])
                .expect_err("duplicate recent records should not silently overwrite one another");
        assert!(duplicate_recent.contains("duplicate recent source"));
    }

    #[test]
    fn collaborative_navigation_activation_projects_pinned_targets() {
        let timestamp = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let project = recent_project("/workspace/project", timestamp);
        let thread = recent_thread("thread", timestamp, false);
        let project_row = CollaborativeNavigationRow::from_project_group(
            project.project_group_key(),
            "project",
            Vec::new(),
        );
        let thread_row = CollaborativeNavigationRow::from_thread(&thread, Vec::new());

        assert_eq!(
            CollaborativePinned::target(&project_row),
            Some(CollaborativeNavigationTarget::project(
                &project.project_group_key()
            ))
        );
        assert_eq!(
            CollaborativePinned::target(&thread_row),
            Some(CollaborativeNavigationTarget::thread(
                thread.thread_id.to_key_string()
            ))
        );
    }

    #[gpui::test]
    async fn collaborative_pinned_renders_empty_and_unavailable_states(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let (view, cx) = cx.add_window_view(|_, _| CollaborativePinned {
            multi_workspace: None,
            state: CollaborativePinnedState::Ready,
            activation_error: None,
            recent_projects: Vec::new(),
            _load_task: Task::ready(()),
        });
        cx.run_until_parked();
        assert!(cx.debug_bounds("COLLABORATIVE-PINNED-EMPTY").is_some());

        view.update(cx, |view, cx| {
            view.state = CollaborativePinnedState::Unavailable("Recent work unavailable".into());
            cx.notify();
        });
        cx.run_until_parked();
        assert!(
            cx.debug_bounds("COLLABORATIVE-PINNED-UNAVAILABLE")
                .is_some()
        );
        assert!(cx.debug_bounds("COLLABORATIVE-PINNED-EMPTY").is_none());
    }
}
