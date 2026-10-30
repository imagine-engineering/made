use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

use acp_thread::{AgentThreadEntry, ToolCallStatus};
use agent_client_protocol::schema::v1 as acp;
use agent_ui::thread_metadata_store::{ThreadMetadata, ThreadMetadataStore};
use agent_ui::{Agent, AgentPanel, AgentThreadSource, ConversationView, ThreadId};
use gpui::{Context, Entity, InteractiveElement, KeyDownEvent, Render, SharedString, WeakEntity};
use ui::{AgentThreadStatus, Window, prelude::*};
use util::ResultExt;
use workspace::{
    MultiWorkspace, ProjectGroupKey,
    collaborative_navigation::{CollaborativeNavigationError, CollaborativeNavigationTarget},
};

use crate::{
    collaborative_live_thread_statuses,
    collaborative_navigation::{
        CollaborativeNavigationBadge, CollaborativeNavigationRow, navigation_action,
        navigation_icon, navigation_label, navigation_row, render_collaborative_navigation_badges,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CollaborativeTaskState {
    Running,
    WaitingForUser,
    Failed,
    Draft,
    Completed,
    Unknown,
    Cancelled,
    Interrupted,
    Archived,
}

impl CollaborativeTaskState {
    fn badge(self) -> Option<CollaborativeNavigationBadge> {
        match self {
            Self::Running => Some(CollaborativeNavigationBadge::Running),
            Self::WaitingForUser => Some(CollaborativeNavigationBadge::WaitingForUser),
            Self::Failed => Some(CollaborativeNavigationBadge::Failed),
            Self::Draft => None,
            Self::Completed => Some(CollaborativeNavigationBadge::Completed),
            Self::Unknown => Some(CollaborativeNavigationBadge::Unknown),
            Self::Cancelled => Some(CollaborativeNavigationBadge::Cancelled),
            Self::Interrupted => Some(CollaborativeNavigationBadge::Interrupted),
            Self::Archived => Some(CollaborativeNavigationBadge::Archived),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Running => "Running",
            Self::WaitingForUser => "Waiting for user",
            Self::Failed => "Failed",
            Self::Draft => "Draft",
            Self::Completed => "Completed",
            Self::Unknown => "Unknown",
            Self::Cancelled => "Cancelled",
            Self::Interrupted => "Interrupted",
            Self::Archived => "Archived",
        }
    }

    fn priority(self) -> u8 {
        match self {
            Self::Running => 0,
            Self::WaitingForUser => 1,
            Self::Failed => 2,
            Self::Draft => 3,
            Self::Completed => 4,
            Self::Unknown => 5,
            Self::Cancelled => 6,
            Self::Interrupted => 7,
            Self::Archived => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CollaborativeTaskRow {
    navigation: CollaborativeNavigationRow,
    metadata: ThreadMetadata,
    state: CollaborativeTaskState,
    updated_at: chrono::DateTime<chrono::Utc>,
}

fn task_rows(
    metadata: impl IntoIterator<Item = ThreadMetadata>,
    live_statuses: &HashMap<acp::SessionId, AgentThreadStatus>,
) -> Vec<CollaborativeTaskRow> {
    let mut rows = metadata
        .into_iter()
        .map(|thread| {
            let state = if thread.archived {
                CollaborativeTaskState::Archived
            } else if thread.is_draft() {
                CollaborativeTaskState::Draft
            } else {
                match thread
                    .session_id
                    .as_ref()
                    .and_then(|session_id| live_statuses.get(session_id))
                    .copied()
                    .unwrap_or_else(|| thread.execution_status())
                {
                    AgentThreadStatus::Running => CollaborativeTaskState::Running,
                    AgentThreadStatus::WaitingForConfirmation => {
                        CollaborativeTaskState::WaitingForUser
                    }
                    AgentThreadStatus::Error => CollaborativeTaskState::Failed,
                    AgentThreadStatus::Completed => CollaborativeTaskState::Completed,
                    AgentThreadStatus::Unknown => CollaborativeTaskState::Unknown,
                    AgentThreadStatus::Cancelled => CollaborativeTaskState::Cancelled,
                    AgentThreadStatus::Interrupted => CollaborativeTaskState::Interrupted,
                }
            };
            let badges = state.badge().into_iter().collect();
            let updated_at = thread.updated_at;
            CollaborativeTaskRow {
                navigation: CollaborativeNavigationRow::from_thread(&thread, badges),
                metadata: thread,
                state,
                updated_at,
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| (row.state.priority(), Reverse(row.updated_at)));
    rows
}

#[cfg(test)]
fn task_belongs_to_project(metadata: &ThreadMetadata, project: &ProjectGroupKey) -> bool {
    !metadata.worktree_paths.is_empty()
        && project.matches(&ProjectGroupKey::from_worktree_paths(
            &metadata.worktree_paths,
            metadata.remote_connection.clone(),
        ))
}

fn task_target(metadata: &ThreadMetadata) -> CollaborativeNavigationTarget {
    CollaborativeNavigationTarget::thread(metadata.thread_id.to_key_string())
}

#[derive(Clone)]
struct CollaborativeSubtask {
    session_id: acp::SessionId,
    label: SharedString,
    depth: usize,
    color: Color,
}

fn conversation_subtasks(
    conversation: &ConversationView,
    cx: &gpui::App,
) -> Vec<CollaborativeSubtask> {
    let Some(root) = conversation.root_thread_view() else {
        return Vec::new();
    };
    let root_session = root.read(cx).thread.read(cx).session_id().clone();
    let mut pending = vec![(root_session, None, Some(root), 0usize, Color::Muted)];
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    while let Some((session_id, label, view, depth, color)) = pending.pop() {
        if !seen.insert(session_id.clone()) {
            continue;
        }
        if let Some(label) = label {
            rows.push(CollaborativeSubtask {
                session_id,
                label,
                depth: depth.saturating_sub(1),
                color,
            });
        }
        let Some(view) = view else {
            continue;
        };
        let thread = view.read(cx).thread.read(cx);
        let mut children = Vec::new();
        for entry in thread.entries() {
            let AgentThreadEntry::ToolCall(call) = entry else {
                continue;
            };
            let Some(info) = &call.subagent_session_info else {
                continue;
            };
            let child = conversation.thread_view(&info.session_id);
            let color = if child
                .as_ref()
                .is_some_and(|view| view.read(cx).thread.read(cx).is_waiting_for_confirmation())
            {
                Color::Warning
            } else {
                match call.status {
                    ToolCallStatus::Pending | ToolCallStatus::InProgress => Color::Info,
                    ToolCallStatus::WaitingForConfirmation { .. } => Color::Warning,
                    ToolCallStatus::Failed | ToolCallStatus::Rejected => Color::Error,
                    ToolCallStatus::Completed | ToolCallStatus::Canceled => Color::Muted,
                }
            };
            children.push((
                info.session_id.clone(),
                Some(call.label.read(cx).source().clone()),
                child,
                depth + 1,
                color,
            ));
        }
        pending.extend(children.into_iter().rev());
    }
    rows
}

pub(crate) struct CollaborativeTasks {
    multi_workspace: WeakEntity<MultiWorkspace>,
    activation_error: Option<SharedString>,
    project_group: Option<ProjectGroupKey>,
    observed_conversations: HashMap<gpui::EntityId, gpui::Subscription>,
}

impl CollaborativeTasks {
    pub(crate) fn new(multi_workspace: WeakEntity<MultiWorkspace>, cx: &mut Context<Self>) -> Self {
        if let Some(thread_store) = ThreadMetadataStore::try_global(cx) {
            cx.observe(&thread_store, |_, _, cx| cx.notify()).detach();
        }
        if let Some(multi_workspace) = multi_workspace.upgrade() {
            cx.observe(&multi_workspace, |_, _, cx| cx.notify())
                .detach();
        }
        Self {
            multi_workspace,
            activation_error: None,
            project_group: None,
            observed_conversations: HashMap::new(),
        }
    }

    pub(crate) fn for_project(
        multi_workspace: WeakEntity<MultiWorkspace>,
        project_group: ProjectGroupKey,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut tasks = Self::new(multi_workspace, cx);
        tasks.project_group = Some(project_group);
        tasks
    }

    fn rows(&self, cx: &gpui::App) -> Option<Vec<CollaborativeTaskRow>> {
        let multi_workspace = self.multi_workspace.upgrade()?;
        let metadata_store = ThreadMetadataStore::try_global(cx)?;
        let live_statuses = collaborative_live_thread_statuses(multi_workspace.read(cx), cx);
        let open_groups = multi_workspace.read(cx).project_groups(cx);
        let workspace = multi_workspace.read(cx).workspace().clone();
        let navigation = workspace.read(cx).collaborative_navigation();
        let mut rows = task_rows(
            metadata_store
                .read(cx)
                .entries()
                .filter(|metadata| match &self.project_group {
                    Some(project) => metadata_store
                        .read(cx)
                        .task_projects(metadata.thread_id)
                        .iter()
                        .any(|association| association.matches(project)),
                    None => {
                        !navigation.is_pinned(&task_target(metadata))
                            && !open_groups.iter().any(|project| {
                                metadata_store
                                    .read(cx)
                                    .task_projects(metadata.thread_id)
                                    .iter()
                                    .any(|association| association.matches(&project.key))
                            })
                    }
                })
                .cloned(),
            &live_statuses,
        );
        for row in &mut rows {
            if metadata_store
                .read(cx)
                .task_organization(row.metadata.thread_id)
                .unread
            {
                let mut badges = row.navigation.badges().to_vec();
                badges.push(CollaborativeNavigationBadge::Unread(1));
                row.navigation = CollaborativeNavigationRow::from_thread(&row.metadata, badges);
            }
        }
        Some(rows)
    }

    fn conversation(
        &mut self,
        thread_id: ThreadId,
        cx: &mut Context<Self>,
    ) -> Option<Entity<ConversationView>> {
        let multi_workspace = self.multi_workspace.upgrade()?;
        let conversation = multi_workspace
            .read(cx)
            .workspaces()
            .find_map(|workspace| {
                workspace
                    .read(cx)
                    .panel::<AgentPanel>(cx)?
                    .read(cx)
                    .conversation_views()
                    .into_iter()
                    .find(|conversation| conversation.read(cx).parent_id() == thread_id)
            })?;
        self.observed_conversations
            .entry(conversation.entity_id())
            .or_insert_with(|| cx.observe(&conversation, |_, _, cx| cx.notify()));
        let views = conversation
            .read(cx)
            .root_thread_view()
            .into_iter()
            .chain(
                conversation_subtasks(conversation.read(cx), cx)
                    .into_iter()
                    .filter_map(|row| conversation.read(cx).thread_view(&row.session_id)),
            )
            .collect::<Vec<_>>();
        for view in views {
            let thread = view.read(cx).thread.clone();
            self.observed_conversations
                .entry(thread.entity_id())
                .or_insert_with(|| cx.observe(&thread, |_, _, cx| cx.notify()));
        }
        Some(conversation)
    }

    fn activate_subtask(
        &mut self,
        thread_id: ThreadId,
        session_id: &acp::SessionId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(metadata) = row_metadata(&thread_id, cx) else {
            self.activation_error = Some("The parent task is no longer available".into());
            cx.notify();
            return;
        };
        let Some(conversation) = self
            .conversation(thread_id, cx)
            .filter(|conversation| conversation.read(cx).thread_view(session_id).is_some())
        else {
            self.activation_error = Some("The subtask session is not available yet".into());
            cx.notify();
            return;
        };
        self.activate(metadata, window, cx);
        if self.activation_error.is_none() {
            conversation.update(cx, |conversation, cx| {
                conversation.navigate_to_thread(session_id.clone(), window, cx)
            });
        }
    }

    fn toggle_subtasks(&mut self, thread_id: ThreadId, cx: &mut Context<Self>) {
        let store = ThreadMetadataStore::global(cx);
        let save = store.update(cx, |store, cx| {
            store.update_task_organization(
                thread_id,
                |state| {
                    state.subtasks_collapsed = !state.subtasks_collapsed;
                },
                cx,
            )
        });
        cx.spawn(async move |this, cx| {
            if let Err(error) = save.await {
                this.update(cx, |this, cx| {
                    this.activation_error = Some(error.to_string().into());
                    cx.notify();
                })
                .log_err();
            }
        })
        .detach();
    }

    fn selected_thread_id(&self, cx: &gpui::App) -> Option<String> {
        let multi_workspace = self.multi_workspace.upgrade()?;
        let workspace = multi_workspace.read(cx).workspace().clone();
        match workspace.read(cx).collaborative_navigation().current()? {
            CollaborativeNavigationTarget::Thread { thread_id } => Some(thread_id.clone()),
            _ => None,
        }
    }

    fn activate(&mut self, metadata: ThreadMetadata, window: &mut Window, cx: &mut Context<Self>) {
        let Some(multi_workspace) = self.multi_workspace.upgrade() else {
            self.activation_error = Some("Navigation is unavailable".into());
            cx.notify();
            return;
        };
        let is_available = ThreadMetadataStore::try_global(cx)
            .is_some_and(|store| store.read(cx).entry(metadata.thread_id).is_some());
        if !is_available {
            self.activation_error = Some("The selected thread is no longer available".into());
            cx.notify();
            return;
        }

        self.activation_error =
            activate_thread_target(&multi_workspace, metadata, window, cx).err();
        cx.notify();
    }
}

pub(super) fn activate_thread_target(
    multi_workspace: &Entity<MultiWorkspace>,
    metadata: ThreadMetadata,
    window: &mut Window,
    cx: &mut gpui::App,
) -> Result<(), SharedString> {
    if ThreadMetadataStore::global(cx)
        .read(cx)
        .task_deletion_pending(metadata.thread_id)
    {
        return Err("Task deletion is in progress".into());
    }
    let project_group = ProjectGroupKey::from_worktree_paths(
        &metadata.worktree_paths,
        metadata.remote_connection.clone(),
    );
    let workspace = if metadata.worktree_paths.is_empty() {
        Some(multi_workspace.read(cx).workspace().clone())
    } else {
        multi_workspace
            .read(cx)
            .last_active_workspace_for_group(&project_group, cx)
            .or_else(|| {
                multi_workspace
                    .read(cx)
                    .workspaces_for_project_group(&project_group, cx)
                    .into_iter()
                    .next()
            })
    };
    let Some(workspace) = workspace else {
        let sidebar = multi_workspace
            .read(cx)
            .sidebar()
            .and_then(|sidebar| sidebar.to_any().downcast::<crate::Sidebar>().ok())
            .ok_or_else(|| SharedString::from("The project navigation surface is unavailable"))?;
        let folder_paths = metadata.folder_paths().clone();
        sidebar.update(cx, |sidebar, cx| {
            sidebar.open_workspace_and_activate_thread(
                metadata,
                folder_paths,
                &project_group,
                window,
                cx,
            );
        });
        return Ok(());
    };
    let agent_panel = workspace
        .read(cx)
        .panel::<AgentPanel>(cx)
        .ok_or_else(|| SharedString::from("The thread surface is unavailable"))?;

    multi_workspace.update(cx, |multi_workspace, cx| {
        multi_workspace.activate(workspace.clone(), None, window, cx);
    });
    agent_panel.update(cx, |panel, cx| {
        panel.load_agent_thread(
            Agent::from(metadata.agent_id.clone()),
            metadata.thread_id,
            Some(metadata.folder_paths().clone()),
            metadata.title.clone(),
            true,
            AgentThreadSource::Sidebar,
            window,
            cx,
        );
    });
    workspace.update(cx, |workspace, cx| {
        AgentPanel::reveal_conversation(workspace, true, window, cx);
    });
    crate::collaborative_task_menu::mark_task_read(metadata.thread_id, workspace.clone(), cx);
    let target = task_target(&metadata);
    workspace
        .update(cx, |workspace, cx| {
            workspace.navigate_collaborative_to(target, |_| true, window, cx)
        })
        .map_err(navigation_error_message)?;
    Ok(())
}

fn row_metadata(thread_id: &ThreadId, cx: &gpui::App) -> Option<ThreadMetadata> {
    ThreadMetadataStore::global(cx)
        .read(cx)
        .entry(*thread_id)
        .cloned()
}

fn navigation_error_message(error: CollaborativeNavigationError) -> SharedString {
    match error {
        CollaborativeNavigationError::MissingTarget(_) => {
            "The selected thread is unavailable".into()
        }
        _ => error.to_string().into(),
    }
}

impl Render for CollaborativeTasks {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let selected_thread_id = self.selected_thread_id(cx);
        let contents = match self.rows(cx) {
            None => v_flex()
                .debug_selector(|| "COLLABORATIVE-TASKS-UNAVAILABLE".to_owned())
                .child(
                    Label::new("Tasks and threads are unavailable")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .into_any_element(),
            Some(rows) if rows.is_empty() && self.project_group.is_some() => {
                div().into_any_element()
            }
            Some(rows) if rows.is_empty() => v_flex()
                .debug_selector(|| "COLLABORATIVE-TASKS-EMPTY".to_owned())
                .child(
                    Label::new("No tasks or threads")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .into_any_element(),
            Some(rows) => v_flex()
                .when_some(self.activation_error.clone(), |this, error| {
                    this.child(
                        Label::new(error)
                            .size(LabelSize::XSmall)
                            .color(Color::Error),
                    )
                })
                .children(rows.into_iter().map(|row| {
                    let metadata = row.metadata.clone();
                    let conversation = self.conversation(metadata.thread_id, cx);
                    let subtasks = conversation
                        .as_ref()
                        .map(|conversation| conversation_subtasks(conversation.read(cx), cx))
                        .unwrap_or_default();
                    let collapsed = ThreadMetadataStore::global(cx)
                        .read(cx)
                        .task_organization(metadata.thread_id)
                        .subtasks_collapsed;
                    let has_subtasks = !subtasks.is_empty();
                    let keyboard_metadata = metadata.clone();
                    let selected = selected_thread_id
                        .as_ref()
                        .is_some_and(|thread_id| *thread_id == metadata.thread_id.to_key_string());
                    let badges = render_collaborative_navigation_badges(&row.navigation);
                    let state_color = match row.state {
                        CollaborativeTaskState::Running => Color::Info,
                        CollaborativeTaskState::WaitingForUser => Color::Warning,
                        CollaborativeTaskState::Failed => Color::Error,
                        CollaborativeTaskState::Draft
                        | CollaborativeTaskState::Completed
                        | CollaborativeTaskState::Unknown
                        | CollaborativeTaskState::Cancelled
                        | CollaborativeTaskState::Interrupted
                        | CollaborativeTaskState::Archived => Color::Muted,
                    };
                    let thread_id = metadata.thread_id;
                    let row = navigation_row(
                        SharedString::from(format!(
                            "collaborative-thread-target-{}",
                            metadata.thread_id.to_key_string()
                        )),
                        selected,
                        usize::from(self.project_group.is_some()),
                        cx,
                    )
                    .aria_label(format!(
                        "{} · {}",
                        row.navigation.label(),
                        row.state.label()
                    ))
                    .when(has_subtasks, |row| {
                        row.child(
                            navigation_action(
                                SharedString::from(format!("subtask-disclosure-{thread_id:?}")),
                                if collapsed {
                                    IconName::ChevronRight
                                } else {
                                    IconName::ChevronDown
                                },
                                if collapsed {
                                    "Expand subtasks"
                                } else {
                                    "Collapse subtasks"
                                },
                            )
                            .on_click(cx.listener(
                                move |this, _, _, cx| {
                                    cx.stop_propagation();
                                    this.toggle_subtasks(thread_id, cx);
                                },
                            )),
                        )
                    })
                    .when(!has_subtasks, |row| {
                        row.child(navigation_icon(IconName::Thread, state_color))
                    })
                    .child(navigation_label(row.navigation.label().clone()))
                    .when_some(badges, |this, badges| this.child(badges))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.activate(metadata.clone(), window, cx);
                    }))
                    .on_key_down(cx.listener(
                        move |this, event: &KeyDownEvent, window, cx| {
                            if has_subtasks
                                && ((event.keystroke.key == "right" && collapsed)
                                    || (event.keystroke.key == "left" && !collapsed))
                            {
                                cx.stop_propagation();
                                this.toggle_subtasks(thread_id, cx);
                            } else if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                cx.stop_propagation();
                                this.activate(keyboard_metadata.clone(), window, cx);
                            }
                        },
                    ));
                    let parent = crate::collaborative_task_menu::task_row_menu(
                        row,
                        thread_id,
                        self.multi_workspace.clone(),
                        cx,
                    );
                    v_flex()
                        .child(parent)
                        .when(!collapsed, |column| {
                            column.children(subtasks.into_iter().map(|subtask| {
                                let session_id = subtask.session_id.clone();
                                let conversation = conversation.clone();
                                let selected = conversation
                                    .as_ref()
                                    .and_then(|conversation| {
                                        conversation.read(cx).active_thread().map(|view| {
                                            view.read(cx).thread.read(cx).session_id()
                                                == &session_id
                                        })
                                    })
                                    .unwrap_or(false)
                                    && selected;
                                let keyboard_session = session_id.clone();
                                let activate =
                                    cx.listener(move |this: &mut Self, _, window, cx| {
                                        this.activate_subtask(thread_id, &session_id, window, cx);
                                    });
                                navigation_row(
                                    SharedString::from(format!(
                                        "collaborative-subtask-{thread_id:?}-{}",
                                        subtask.session_id
                                    )),
                                    selected,
                                    usize::from(self.project_group.is_some()) + subtask.depth + 1,
                                    cx,
                                )
                                .debug_selector(move || {
                                    format!("COLLABORATIVE-SUBTASK-{}", subtask.session_id)
                                })
                                .aria_label(subtask.label.clone())
                                .child(navigation_icon(IconName::ZedAgent, subtask.color))
                                .child(navigation_label(subtask.label))
                                .on_click(activate)
                                .on_key_down(cx.listener(
                                    move |this, event: &KeyDownEvent, window, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            cx.stop_propagation();
                                            this.activate_subtask(
                                                thread_id,
                                                &keyboard_session,
                                                window,
                                                cx,
                                            );
                                        }
                                    },
                                ))
                            }))
                        })
                        .into_any_element()
                }))
                .into_any_element(),
        };
        v_flex()
            .debug_selector(|| "COLLABORATIVE-TASKS".to_owned())
            .child(contents)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use agent_ui::{ThreadId, thread_metadata_store::WorktreePaths};
    use chrono::{TimeZone as _, Utc};
    use project::AgentId;

    use super::*;

    fn metadata(
        title: &str,
        session_id: Option<&str>,
        hour: u32,
        archived: bool,
    ) -> ThreadMetadata {
        ThreadMetadata {
            execution_outcome: acp_thread::ThreadExecutionOutcome::Unknown,
            thread_id: ThreadId::new(),
            session_id: session_id.map(|id| acp::SessionId::new(Arc::<str>::from(id))),
            agent_id: AgentId("test-agent".into()),
            title: Some(title.into()),
            title_override: None,
            updated_at: Utc.with_ymd_and_hms(2026, 1, 1, hour, 0, 0).unwrap(),
            created_at: None,
            interacted_at: None,
            worktree_paths: WorktreePaths::default(),
            remote_connection: None,
            archived,
        }
    }

    #[test]
    fn collaborative_tasks_group_linked_worktrees_by_the_canonical_main_project() {
        let main_paths = workspace::PathList::new(&[std::path::PathBuf::from("/project")]);
        let linked_paths = workspace::PathList::new(&[std::path::PathBuf::from("/linked")]);
        let project = ProjectGroupKey::new(None, main_paths.clone());
        let other_project = ProjectGroupKey::new(None, linked_paths.clone());
        let mut task = metadata("linked task", Some("session"), 1, false);
        assert!(!task_belongs_to_project(&task, &ProjectGroupKey::default()));
        task.worktree_paths = WorktreePaths::from_path_lists(main_paths, linked_paths)
            .expect("paired worktree paths");
        assert!(task_belongs_to_project(&task, &project));
        assert!(!task_belongs_to_project(&task, &other_project));
    }

    #[test]
    fn collaborative_tasks_projects_state_transitions() {
        let thread = metadata("task", Some("session"), 1, false);
        let session_id = thread
            .session_id
            .clone()
            .expect("test session should exist");
        for (status, expected) in [
            (AgentThreadStatus::Running, CollaborativeTaskState::Running),
            (
                AgentThreadStatus::WaitingForConfirmation,
                CollaborativeTaskState::WaitingForUser,
            ),
            (AgentThreadStatus::Error, CollaborativeTaskState::Failed),
            (
                AgentThreadStatus::Completed,
                CollaborativeTaskState::Completed,
            ),
        ] {
            let rows = task_rows(
                [thread.clone()],
                &HashMap::from([(session_id.clone(), status)]),
            );
            assert_eq!(rows[0].state, expected);
            assert_eq!(rows[0].navigation.badges(), expected.badge().as_slice());
        }
    }

    #[test]
    fn collaborative_tasks_preserve_recorded_outcomes_when_sessions_are_unloaded() {
        for (outcome, expected) in [
            (
                acp_thread::ThreadExecutionOutcome::Unknown,
                CollaborativeTaskState::Unknown,
            ),
            (
                acp_thread::ThreadExecutionOutcome::Running,
                CollaborativeTaskState::Interrupted,
            ),
            (
                acp_thread::ThreadExecutionOutcome::Completed,
                CollaborativeTaskState::Completed,
            ),
            (
                acp_thread::ThreadExecutionOutcome::Cancelled,
                CollaborativeTaskState::Cancelled,
            ),
            (
                acp_thread::ThreadExecutionOutcome::Failed,
                CollaborativeTaskState::Failed,
            ),
            (
                acp_thread::ThreadExecutionOutcome::Interrupted,
                CollaborativeTaskState::Interrupted,
            ),
        ] {
            let mut thread = metadata("restored task", Some("restored"), 1, false);
            thread.execution_outcome = outcome;
            let rows = task_rows([thread.clone()], &HashMap::default());
            assert_eq!(rows[0].state, expected);
            assert_eq!(rows[0].navigation.badges(), expected.badge().as_slice());
            let live = HashMap::from([(
                thread.session_id.clone().expect("session"),
                AgentThreadStatus::Running,
            )]);
            assert_eq!(
                task_rows([thread], &live)[0].state,
                CollaborativeTaskState::Running
            );
        }
    }

    #[test]
    fn collaborative_tasks_orders_active_history_and_archive() {
        let running = metadata("running", Some("running"), 1, false);
        let waiting = metadata("waiting", Some("waiting"), 2, false);
        let failed = metadata("failed", Some("failed"), 3, false);
        let mut newer_completed = metadata("newer completed", Some("newer"), 5, false);
        newer_completed.execution_outcome = acp_thread::ThreadExecutionOutcome::Completed;
        let mut older_completed = metadata("older completed", Some("older"), 4, false);
        older_completed.execution_outcome = acp_thread::ThreadExecutionOutcome::Completed;
        let archived = metadata("archived", Some("archived"), 6, true);
        let statuses = HashMap::from([
            (
                running.session_id.clone().expect("running session"),
                AgentThreadStatus::Running,
            ),
            (
                waiting.session_id.clone().expect("waiting session"),
                AgentThreadStatus::WaitingForConfirmation,
            ),
            (
                failed.session_id.clone().expect("failed session"),
                AgentThreadStatus::Error,
            ),
        ]);
        let rows = task_rows(
            [
                archived,
                older_completed,
                newer_completed,
                failed,
                waiting,
                running,
            ],
            &statuses,
        );

        assert_eq!(
            rows.iter()
                .map(|row| row.navigation.label().as_ref())
                .collect::<Vec<_>>(),
            [
                "running",
                "waiting",
                "failed",
                "newer completed",
                "older completed",
                "archived",
            ]
        );
    }

    #[test]
    fn collaborative_tasks_keeps_drafts_out_of_completed_history() {
        let rows = task_rows([metadata("draft", None, 1, false)], &HashMap::new());
        assert_eq!(rows[0].state, CollaborativeTaskState::Draft);
        assert!(rows[0].navigation.badges().is_empty());
    }

    #[test]
    fn collaborative_navigation_activation_projects_task_thread_target() {
        let thread = metadata("task", Some("session"), 1, false);

        assert_eq!(
            task_target(&thread),
            CollaborativeNavigationTarget::thread(thread.thread_id.to_key_string())
        );
    }
}
