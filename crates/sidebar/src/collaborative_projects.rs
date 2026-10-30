use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use crate::collaborative_tasks::CollaborativeTasks;

use channel::ChannelStore;
use gpui::{
    Action, Context, Entity, InteractiveElement, KeyDownEvent, Render, SharedString, WeakEntity,
};
use project::{ProjectGroupKey, WorktreeId};
use ui::{Window, prelude::*};
use workspace::{
    MultiWorkspace,
    collaborative_navigation::{CollaborativeNavigationError, CollaborativeNavigationTarget},
};

use crate::collaborative_navigation::{
    CollaborativeNavigationBadge, CollaborativeNavigationRow, CollaborativeNavigationRowId,
    CollaborativeNavigationSourceId, navigation_action, navigation_icon, navigation_label,
    navigation_row, render_collaborative_navigation_badges,
};

#[derive(Clone, Debug)]
struct CollaborativeProjectSource {
    key: ProjectGroupKey,
    label: SharedString,
    expanded: bool,
    repositories: Vec<(PathBuf, SharedString)>,
    worktrees: Vec<(WorktreeId, PathBuf, SharedString)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CollaborativeProjectGroupProjection {
    project: CollaborativeNavigationRow,
    expanded: bool,
    repositories: Vec<CollaborativeNavigationRow>,
    worktrees: Vec<CollaborativeNavigationRow>,
}

fn project_hierarchy(
    sources: impl IntoIterator<Item = CollaborativeProjectSource>,
) -> Result<Vec<CollaborativeProjectGroupProjection>, SharedString> {
    let mut row_ids = HashSet::<CollaborativeNavigationRowId>::new();
    let mut groups = Vec::new();
    for mut source in sources {
        source
            .repositories
            .sort_by(|left, right| left.0.cmp(&right.0));
        source.worktrees.sort_by(|left, right| left.1.cmp(&right.1));

        let project = CollaborativeNavigationRow::from_project_group(
            source.key.clone(),
            source.label,
            Vec::new(),
        );
        if !row_ids.insert(project.id().clone()) {
            return Err("duplicate project group in collaborative hierarchy".into());
        }

        let mut repositories = Vec::new();
        for (work_directory, label) in source.repositories {
            let row = CollaborativeNavigationRow::from_repository(
                source.key.clone(),
                work_directory,
                label,
                Vec::new(),
            );
            if row_ids.insert(row.id().clone()) {
                repositories.push(row);
            }
        }

        let mut worktrees = Vec::new();
        for (worktree_id, path, label) in source.worktrees {
            let row = CollaborativeNavigationRow::from_worktree(
                source.key.clone(),
                worktree_id,
                path,
                label,
                Vec::new(),
            );
            if row_ids.insert(row.id().clone()) {
                worktrees.push(row);
            }
        }
        groups.push(CollaborativeProjectGroupProjection {
            project,
            expanded: source.expanded,
            repositories,
            worktrees,
        });
    }
    Ok(groups)
}

pub(crate) struct CollaborativeProjects {
    multi_workspace: WeakEntity<MultiWorkspace>,
    activation_error: Option<SharedString>,
    task_views: HashMap<ProjectGroupKey, Entity<CollaborativeTasks>>,
}

impl CollaborativeProjects {
    pub(crate) fn new(multi_workspace: WeakEntity<MultiWorkspace>, cx: &mut Context<Self>) -> Self {
        if let Some(multi_workspace) = multi_workspace.upgrade() {
            cx.observe(&multi_workspace, |_, _, cx| cx.notify())
                .detach();
        }
        if let Some(channel_store) = ChannelStore::try_global(cx) {
            cx.observe(&channel_store, |_, _, cx| cx.notify()).detach();
        }
        Self {
            multi_workspace,
            activation_error: None,
            task_views: HashMap::new(),
        }
    }

    fn sources(&self, cx: &gpui::App) -> Option<Vec<CollaborativeProjectSource>> {
        let multi_workspace = self.multi_workspace.upgrade()?;
        Some(
            multi_workspace
                .read(cx)
                .project_groups(cx)
                .into_iter()
                .map(|group| {
                    let mut repositories = Vec::new();
                    let mut worktrees = Vec::new();
                    for workspace in group.workspaces {
                        let project = workspace.read(cx).project().clone();
                        let project = project.read(cx);
                        repositories.extend(project.repositories(cx).values().map(|repository| {
                            let repository = repository.read(cx);
                            let path = repository.work_directory_abs_path.as_ref().to_path_buf();
                            let label = path
                                .file_name()
                                .map(|name| name.to_string_lossy().to_string())
                                .unwrap_or_else(|| path.display().to_string());
                            (path, label.into())
                        }));
                        worktrees.extend(project.visible_worktrees(cx).map(|worktree| {
                            let worktree = worktree.read(cx);
                            (
                                worktree.id(),
                                worktree.abs_path().as_ref().to_path_buf(),
                                SharedString::from(worktree.root_name_str().to_owned()),
                            )
                        }));
                    }
                    CollaborativeProjectSource {
                        label: group.key.display_name(&Default::default()),
                        key: group.key,
                        expanded: group.expanded,
                        repositories,
                        worktrees,
                    }
                })
                .collect(),
        )
    }

    fn target(row: &CollaborativeNavigationRow) -> Option<CollaborativeNavigationTarget> {
        match row.source_id() {
            crate::collaborative_navigation::CollaborativeNavigationSourceId::Project(project) => {
                Some(CollaborativeNavigationTarget::project(project))
            }
            crate::collaborative_navigation::CollaborativeNavigationSourceId::Repository {
                project,
                work_directory,
            } => Some(CollaborativeNavigationTarget::repository(
                project,
                work_directory.clone(),
            )),
            crate::collaborative_navigation::CollaborativeNavigationSourceId::Worktree {
                project,
                worktree_id,
                path,
            } => Some(CollaborativeNavigationTarget::worktree(
                project,
                worktree_id.to_proto(),
                path.clone(),
            )),
            _ => None,
        }
    }

    pub(crate) fn render_channels(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let selected_target = self.selected_target(cx);
        match self.channel_rows(cx) {
            None => v_flex()
                .debug_selector(|| "COLLABORATIVE-COMMUNITIES-UNAVAILABLE".to_owned())
                .child(
                    Label::new("Communities require a signed-in collaboration service")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .into_any_element(),
            Some(rows) if rows.is_empty() => div().into_any_element(),
            Some(rows) => v_flex()
                .debug_selector(|| "COLLABORATIVE-COMMUNITIES".to_owned())
                .gap(DynamicSpacing::Base02.rems(cx))
                .id("collaborative-channels-scroll")
                .max_h(rems_from_px(ui::fractal::CONTROL_SMALL * 6.0))
                .overflow_y_scroll()
                .children(rows.into_iter().map(|(depth, row)| {
                    let CollaborativeNavigationSourceId::Channel(channel_id) = row.source_id()
                    else {
                        return unavailable_projects("Channel target is unavailable");
                    };
                    let channel_id = *channel_id;
                    let is_favorite = ChannelStore::try_global(cx).is_some_and(|store| {
                        let store = store.read(cx);
                        store
                            .channels()
                            .find(|channel| channel.id.0 == channel_id)
                            .is_some_and(|channel| store.is_channel_favorited(channel.id))
                    });
                    let selected = selected_target.as_ref()
                        == Some(&CollaborativeNavigationTarget::channel(
                            channel_id.to_string(),
                        ));
                    let badges = render_collaborative_navigation_badges(&row);
                    navigation_row(
                        SharedString::from(format!("collaborative-channel-target-{channel_id}")),
                        selected,
                        depth,
                        cx,
                    )
                    .aria_label(format!("Open channel messages: {}", row.label()))
                    .child(navigation_icon(IconName::Hash, Color::Muted))
                    .child(navigation_label(row.label().clone()))
                    .when_some(badges, |this, badges| this.child(badges))
                    .child(
                        navigation_action(
                            ("collaborative-channel-favorite", channel_id as usize),
                            if is_favorite {
                                IconName::StarFilled
                            } else {
                                IconName::Star
                            },
                            if is_favorite {
                                "Remove from pinned"
                            } else {
                                "Add to pinned"
                            },
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            cx.stop_propagation();
                            this.toggle_channel_favorite(channel_id, cx);
                        })),
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.activate_channel(channel_id, window, cx);
                    }))
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                        if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                            cx.stop_propagation();
                            this.activate_channel(channel_id, window, cx);
                        }
                    }))
                    .into_any_element()
                }))
                .into_any_element(),
        }
    }

    fn channel_rows(&self, cx: &gpui::App) -> Option<Vec<(usize, CollaborativeNavigationRow)>> {
        let channel_store = ChannelStore::try_global(cx)?;
        let channel_store = channel_store.read(cx);
        Some(
            channel_store
                .ordered_channels()
                .map(|(depth, channel)| {
                    let badges = channel_store
                        .has_channel_buffer_changed(channel.id)
                        .then_some(CollaborativeNavigationBadge::NotesChanged)
                        .into_iter()
                        .collect();
                    (
                        depth,
                        CollaborativeNavigationRow::from_channel(channel, badges),
                    )
                })
                .collect(),
        )
    }

    fn set_expanded(&self, key: &ProjectGroupKey, expanded: bool, cx: &mut Context<Self>) {
        if let Some(multi_workspace) = self.multi_workspace.upgrade() {
            multi_workspace.update(cx, |multi_workspace, cx| {
                if let Some(state) = multi_workspace.group_state_by_key_mut(key) {
                    state.expanded = expanded;
                    multi_workspace.serialize(cx);
                    cx.notify();
                }
            });
        }
    }

    fn selected_target(&self, cx: &gpui::App) -> Option<CollaborativeNavigationTarget> {
        let multi_workspace = self.multi_workspace.upgrade()?;
        let workspace = multi_workspace.read(cx).workspace().clone();
        workspace
            .read(cx)
            .collaborative_navigation()
            .current()
            .cloned()
    }

    fn is_pinned(&self, target: &CollaborativeNavigationTarget, cx: &gpui::App) -> bool {
        let Some(multi_workspace) = self.multi_workspace.upgrade() else {
            return false;
        };
        let workspace = multi_workspace.read(cx).workspace().clone();
        workspace
            .read(cx)
            .collaborative_navigation()
            .is_pinned(target)
    }

    fn toggle_pin(
        &mut self,
        target: CollaborativeNavigationTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(multi_workspace) = self.multi_workspace.upgrade() else {
            self.activation_error = Some("Pinning is unavailable".into());
            cx.notify();
            return;
        };
        let workspace = multi_workspace.read(cx).workspace().clone();
        self.activation_error = workspace
            .update(cx, |workspace, cx| {
                workspace.toggle_collaborative_pin(target, window, cx)
            })
            .err()
            .map(|error| error.to_string().into());
        cx.notify();
    }

    fn toggle_channel_favorite(&mut self, channel_id: u64, cx: &mut Context<Self>) {
        let Some(channel_store) = ChannelStore::try_global(cx) else {
            self.activation_error = Some("Channel favorites are unavailable".into());
            cx.notify();
            return;
        };
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
        self.activation_error = None;
        cx.notify();
    }

    fn activate_channel(&mut self, channel_id: u64, window: &mut Window, cx: &mut Context<Self>) {
        let Some(multi_workspace) = self.multi_workspace.upgrade() else {
            self.activation_error = Some("Navigation is unavailable".into());
            cx.notify();
            return;
        };
        let available = ChannelStore::try_global(cx).is_some_and(|store| {
            store
                .read(cx)
                .channels()
                .any(|channel| channel.id.0 == channel_id)
        });
        if !available {
            self.activation_error = Some("The selected channel is no longer available".into());
            cx.notify();
            return;
        }
        let workspace = multi_workspace.read(cx).workspace().clone();
        let target = CollaborativeNavigationTarget::channel(channel_id.to_string());
        self.activation_error = workspace
            .update(cx, |workspace, cx| {
                workspace.navigate_collaborative_to(target, |_| true, window, cx)
            })
            .err()
            .map(|error| error.to_string().into());
        if self.activation_error.is_none() {
            window.dispatch_action(
                workspace::OpenChannelMessagesById { channel_id }.boxed_clone(),
                cx,
            );
        }
        cx.notify();
    }

    fn activate(
        &mut self,
        target: CollaborativeNavigationTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(multi_workspace) = self.multi_workspace.upgrade() else {
            self.activation_error = Some("Navigation is unavailable".into());
            cx.notify();
            return;
        };
        self.activation_error = activate_project_target(&multi_workspace, target, window, cx).err();
        cx.notify();
    }
}

pub(super) fn activate_project_target(
    multi_workspace: &Entity<MultiWorkspace>,
    target: CollaborativeNavigationTarget,
    window: &mut Window,
    cx: &mut gpui::App,
) -> Result<(), SharedString> {
    let selected_workspace = multi_workspace
        .read(cx)
        .project_groups(cx)
        .into_iter()
        .find_map(|group| {
            group.workspaces.into_iter().find(|workspace| {
                let project = workspace.read(cx).project().read(cx);
                match &target {
                    CollaborativeNavigationTarget::Project {
                        project: target_project,
                    } => target_project.matches(&group.key),
                    CollaborativeNavigationTarget::Repository {
                        project: target_project,
                        work_directory,
                    } => {
                        target_project.matches(&group.key)
                            && project.repositories(cx).values().any(|repository| {
                                repository.read(cx).work_directory_abs_path.as_ref()
                                    == work_directory
                            })
                    }
                    CollaborativeNavigationTarget::Worktree {
                        project: target_project,
                        worktree_id,
                        path,
                    } => {
                        target_project.matches(&group.key)
                            && project.visible_worktrees(cx).any(|worktree| {
                                let worktree = worktree.read(cx);
                                worktree.id().to_proto() == *worktree_id
                                    && worktree.abs_path().as_ref() == path
                            })
                    }
                    _ => false,
                }
            })
        })
        .ok_or_else(|| SharedString::from("The selected project item is no longer available"))?;

    multi_workspace.update(cx, |multi_workspace, cx| {
        multi_workspace.activate(selected_workspace.clone(), None, window, cx);
    });
    selected_workspace
        .update(cx, |workspace, cx| {
            workspace.navigate_collaborative_to(target, |_| true, window, cx)
        })
        .map_err(navigation_error_message)?;
    Ok(())
}

fn navigation_error_message(error: CollaborativeNavigationError) -> SharedString {
    match error {
        CollaborativeNavigationError::MissingTarget(_) => "The selected item is unavailable".into(),
        _ => error.to_string().into(),
    }
}

impl Render for CollaborativeProjects {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selected_target = self.selected_target(cx);
        let contents = match self.sources(cx) {
            None => unavailable_projects("Projects are unavailable"),
            Some(sources) => match project_hierarchy(sources) {
                Ok(groups) if groups.is_empty() => v_flex()
                    .debug_selector(|| "COLLABORATIVE-PROJECTS-EMPTY".to_owned())
                    .child(
                        Label::new("No communities or projects")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    )
                    .into_any_element(),
                Ok(groups) => {
                    self.task_views.retain(|key, _| groups.iter().any(|group| {
                        matches!(group.project.source_id(), CollaborativeNavigationSourceId::Project(group_key) if group_key == key)
                    }));
                    let groups = groups
                        .into_iter()
                        .map(|group| {
                            let is_pinned = Self::target(&group.project)
                                .is_some_and(|target| self.is_pinned(&target, cx));
                            let task_view = match group.project.source_id() {
                                CollaborativeNavigationSourceId::Project(key) => {
                                    let multi_workspace = self.multi_workspace.clone();
                                    let key = key.clone();
                                    Some(
                                        self.task_views
                                            .entry(key.clone())
                                            .or_insert_with(|| {
                                                cx.new(|cx| {
                                                    CollaborativeTasks::for_project(
                                                        multi_workspace,
                                                        key,
                                                        cx,
                                                    )
                                                })
                                            })
                                            .clone(),
                                    )
                                }
                                _ => None,
                            };
                            (group, is_pinned, task_view)
                        })
                        .collect::<Vec<_>>();
                    v_flex()
                        .gap(DynamicSpacing::Base04.rems(cx))
                        .when_some(self.activation_error.clone(), |this, error| {
                            this.child(Label::new(error).size(LabelSize::Small).color(Color::Error))
                        })
                        .children(groups.into_iter().map(|(group, is_pinned, task_view)| {
                            render_project_group(
                                group,
                                task_view,
                                is_pinned,
                                selected_target.as_ref(),
                                cx,
                            )
                        }))
                        .into_any_element()
                }
                Err(error) => unavailable_projects(error),
            },
        };
        v_flex()
            .debug_selector(|| "COLLABORATIVE-PROJECTS".to_owned())
            .gap(DynamicSpacing::Base04.rems(cx))
            .child(contents)
    }
}

fn unavailable_projects(message: impl Into<SharedString>) -> AnyElement {
    v_flex()
        .debug_selector(|| "COLLABORATIVE-PROJECTS-UNAVAILABLE".to_owned())
        .child(
            Label::new(message.into())
                .size(LabelSize::Small)
                .color(Color::Muted),
        )
        .into_any_element()
}

fn render_project_group(
    group: CollaborativeProjectGroupProjection,
    tasks: Option<Entity<CollaborativeTasks>>,
    is_pinned: bool,
    selected_target: Option<&CollaborativeNavigationTarget>,
    cx: &mut Context<CollaborativeProjects>,
) -> AnyElement {
    let badges = render_collaborative_navigation_badges(&group.project);
    let Some(project_target) = CollaborativeProjects::target(&group.project) else {
        return unavailable_projects("Project target is unavailable");
    };
    let CollaborativeNavigationSourceId::Project(key) = group.project.source_id() else {
        return unavailable_projects("Project group is unavailable");
    };
    let expansion_key = key.clone();
    let keyboard_expansion_key = key.clone();
    let expanded = group.expanded;
    let selected = selected_target == Some(&project_target);
    let keyboard_target = project_target.clone();
    let pin_target = project_target.clone();
    v_flex()
        .gap(DynamicSpacing::Base02.rems(cx))
        .child(
            navigation_row(
                SharedString::from(format!("collaborative-project-target-{project_target:?}")),
                selected,
                0,
                cx,
            )
            .debug_selector(|| "COLLABORATIVE-NAVIGATION-PROJECT-ROW".to_owned())
            .aria_label(format!("Open project {}", group.project.label()))
            .child(
                div()
                    .id(SharedString::from(format!(
                        "project-expansion-{expansion_key:?}"
                    )))
                    .debug_selector(|| "COLLABORATIVE-PROJECT-EXPANSION".to_owned())
                    .tooltip(ui::Tooltip::text(if expanded {
                        "Collapse project"
                    } else {
                        "Expand project"
                    }))
                    .child(navigation_icon(IconName::Folder, Color::Muted))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        cx.stop_propagation();
                        this.set_expanded(&expansion_key, !expanded, cx);
                    })),
            )
            .child(navigation_label(group.project.label().clone()))
            .when_some(badges, |this, badges| this.child(badges))
            .child(
                navigation_action(
                    SharedString::from(format!("pin-project-{pin_target:?}")),
                    if is_pinned {
                        IconName::Unpin
                    } else {
                        IconName::Pin
                    },
                    if is_pinned { "Unpin" } else { "Pin" },
                )
                .on_click(cx.listener(move |this, _, window, cx| {
                    cx.stop_propagation();
                    this.toggle_pin(pin_target.clone(), window, cx);
                })),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.activate(project_target.clone(), window, cx);
            }))
            .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                if matches!(event.keystroke.key.as_str(), "left" | "right") {
                    cx.stop_propagation();
                    this.set_expanded(&keyboard_expansion_key, event.keystroke.key == "right", cx);
                }
                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                    cx.stop_propagation();
                    this.activate(keyboard_target.clone(), window, cx);
                }
            })),
        )
        .when(expanded, |this| {
            this.children(tasks)
                .children(
                    group
                        .repositories
                        .into_iter()
                        .map(|row| render_project_child("Repository", row, selected_target, cx)),
                )
                .children(
                    group
                        .worktrees
                        .into_iter()
                        .map(|row| render_project_child("Worktree", row, selected_target, cx)),
                )
        })
        .into_any_element()
}

fn render_project_child(
    kind: &'static str,
    row: CollaborativeNavigationRow,
    selected_target: Option<&CollaborativeNavigationTarget>,
    cx: &mut Context<CollaborativeProjects>,
) -> AnyElement {
    let badges = render_collaborative_navigation_badges(&row);
    let Some(target) = CollaborativeProjects::target(&row) else {
        return unavailable_projects(format!("{kind} target is unavailable"));
    };
    let selected = selected_target == Some(&target);
    let keyboard_target = target.clone();
    navigation_row(
        SharedString::from(format!(
            "collaborative-{}-target-{target:?}",
            kind.to_lowercase()
        )),
        selected,
        1,
        cx,
    )
    .debug_selector(|| "COLLABORATIVE-NAVIGATION-CHILD-ROW".to_owned())
    .aria_label(format!("Open {kind} {}", row.label()))
    .child(navigation_icon(
        if kind == "Repository" {
            IconName::GitBranch
        } else {
            IconName::Folder
        },
        Color::Muted,
    ))
    .child(navigation_label(format!("{kind} · {}", row.label())))
    .when_some(badges, |this, badges| this.child(badges))
    .on_click(cx.listener(move |this, _, window, cx| {
        this.activate(target.clone(), window, cx);
    }))
    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
        if matches!(event.keystroke.key.as_str(), "enter" | "space") {
            cx.stop_propagation();
            this.activate(keyboard_target.clone(), window, cx);
        }
    }))
    .into_any_element()
}

#[cfg(test)]
mod tests {
    use util::path_list::PathList;

    use super::*;

    fn source(worktrees: &[(&str, usize)], repositories: &[&str]) -> CollaborativeProjectSource {
        let key = ProjectGroupKey::new(None, PathList::new(&[PathBuf::from("/project")]));
        CollaborativeProjectSource {
            key,
            label: "project".into(),
            expanded: true,
            repositories: repositories
                .iter()
                .map(|path| (PathBuf::from(path), SharedString::from(*path)))
                .collect(),
            worktrees: worktrees
                .iter()
                .map(|(path, id)| {
                    (
                        WorktreeId::from_usize(*id),
                        PathBuf::from(path),
                        SharedString::from(*path),
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn collaborative_projects_projects_multiple_repositories_and_worktrees() {
        let groups = project_hierarchy([source(
            &[("/project", 1), ("/worktrees/feature", 1)],
            &["/project/nested", "/project"],
        )])
        .expect("valid hierarchy should project");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].repositories.len(), 2);
        assert_eq!(groups[0].repositories[0].label().as_ref(), "/project");
        assert_eq!(groups[0].worktrees.len(), 2);
        assert_eq!(groups[0].worktrees[0].label().as_ref(), "/project");
        assert_eq!(
            groups[0].worktrees[1].label().as_ref(),
            "/worktrees/feature"
        );
        assert_ne!(groups[0].worktrees[0].id(), groups[0].worktrees[1].id());
    }

    #[test]
    fn collaborative_projects_removes_deleted_worktrees() {
        let before =
            project_hierarchy([source(&[("/project", 1), ("/worktrees/feature", 2)], &[])])
                .expect("initial hierarchy should project");
        let after = project_hierarchy([source(&[("/project", 1)], &[])])
            .expect("updated hierarchy should project");

        assert_eq!(before[0].worktrees.len(), 2);
        assert_eq!(after[0].worktrees.len(), 1);
        assert!(
            after[0]
                .worktrees
                .iter()
                .all(|row| row.label().as_ref() != "/worktrees/feature")
        );
    }

    #[test]
    fn collaborative_projects_rejects_duplicate_groups() {
        let source = source(&[], &[]);
        let error = project_hierarchy([source.clone(), source])
            .expect_err("duplicate project groups should fail explicitly");
        assert!(error.contains("duplicate project group"));
    }

    #[test]
    fn collaborative_navigation_activation_projects_hierarchy_targets() {
        let groups = project_hierarchy([source(&[("/project", 1)], &["/project/repository"])])
            .expect("valid hierarchy should project");
        let group = &groups[0];
        let project_key = ProjectGroupKey::new(None, PathList::new(&[PathBuf::from("/project")]));

        assert_eq!(
            CollaborativeProjects::target(&group.project),
            Some(CollaborativeNavigationTarget::project(&project_key))
        );
        assert_eq!(
            CollaborativeProjects::target(&group.repositories[0]),
            Some(CollaborativeNavigationTarget::repository(
                &project_key,
                PathBuf::from("/project/repository")
            ))
        );
        assert_eq!(
            CollaborativeProjects::target(&group.worktrees[0]),
            Some(CollaborativeNavigationTarget::worktree(
                &project_key,
                1,
                PathBuf::from("/project")
            ))
        );
    }
    #[gpui::test]
    async fn channel_rows_preserve_depth_and_notes_changed_semantics(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::sidebar_tests::init_test(cx);
        let store = cx.update(|cx| {
            let app_state = workspace::AppState::test(cx);
            channel::init(&app_state.client, app_state.user_store.clone(), cx);
            ChannelStore::global(cx)
        });
        store.update(cx, |store, cx| {
            store.set_channels_for_tests(
                [
                    (client::ChannelId(301), "Community".into(), None),
                    (
                        client::ChannelId(302),
                        "Design".into(),
                        Some(client::ChannelId(301)),
                    ),
                ],
                cx,
            );
            store.update_latest_notes_version(client::ChannelId(302), 1, &clock::Global::new(), cx);
        });
        let rows = cx.new(|cx| CollaborativeProjects::new(gpui::WeakEntity::new_invalid(), cx));
        rows.read_with(cx, |view, cx| {
            let projected = view.channel_rows(cx).expect("channel owner");
            assert_eq!(
                projected
                    .iter()
                    .map(|(depth, _)| *depth)
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            assert_eq!(
                projected[1].1.badges(),
                &[CollaborativeNavigationBadge::NotesChanged]
            );
            assert!(projected[0].1.badges().is_empty());
        });
        store.update(cx, |store, cx| {
            store.acknowledge_notes_version(client::ChannelId(302), 1, &clock::Global::new(), cx)
        });
        rows.read_with(cx, |view, cx| {
            assert!(
                view.channel_rows(cx).expect("channels")[1]
                    .1
                    .badges()
                    .is_empty()
            )
        });
    }
}
