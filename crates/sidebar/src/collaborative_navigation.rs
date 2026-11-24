#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "navigation bindings are introduced by the following collaborative rail tasks"
    )
)]

use std::{collections::HashSet, fmt, path::PathBuf};

use agent_ui::{ThreadId, thread_metadata_store::ThreadMetadata};
use channel::Channel;
use gpui::{AnyElement, App, Div, ElementId, Role, SharedString, Stateful, relative};
use project::{Project, ProjectGroupKey, WorktreeId};
use ui::{Color, IconButtonShape, Label, LabelSize, prelude::*};

pub(crate) fn navigation_row(
    id: impl Into<ElementId>,
    selected: bool,
    depth: usize,
    cx: &App,
) -> Stateful<Div> {
    let colors = cx.theme().colors();
    h_flex()
        .id(id)
        .min_w_0()
        // Keep the compact action target and focus border inside the same row
        // minimum even when peers have no trailing actions.
        .min_h(ButtonSize::Default.rems())
        .px(DynamicSpacing::Base08.rems(cx))
        .pl(DynamicSpacing::Base08.rems(cx) + DynamicSpacing::Base12.rems(cx) * depth as f32)
        .gap(DynamicSpacing::Base06.rems(cx))
        .font(theme::theme_settings(cx).ui_font(cx).clone())
        .text_ui(cx)
        .line_height(relative(ui::fractal::BODY_LINE_HEIGHT))
        .rounded(rems_from_px(ui::fractal::RADIUS_INNER))
        .border_1()
        .border_color(gpui::transparent_black())
        .tab_index(0)
        .role(Role::Link)
        .aria_selected(selected)
        .cursor_pointer()
        .when(selected, |this| this.bg(colors.ghost_element_selected))
        .hover(|style| style.bg(colors.ghost_element_hover))
        .active(|style| style.bg(colors.ghost_element_active))
        .focus_visible(|style| style.border_color(colors.border_focused))
}

pub(crate) fn navigation_label(label: impl Into<SharedString>) -> Label {
    Label::new(label.into())
        .size(LabelSize::Default)
        .flex_1()
        .truncate()
}

pub(crate) fn navigation_icon(icon: IconName, color: Color) -> impl IntoElement {
    // Fractal's folder fills 22/24 of its viewport; native outline fallbacks
    // fill about 12/16. Adjacent shared sizes balance their painted silhouettes.
    let size = if icon == IconName::Folder {
        IconSize::Small
    } else {
        IconSize::Medium
    };
    h_flex()
        .size(IconSize::Medium.rems())
        .flex_none()
        .justify_center()
        .child(Icon::new(icon).size(size).color(color))
}

pub(crate) fn navigation_action(
    id: impl Into<ElementId>,
    icon: IconName,
    label: &'static str,
) -> IconButton {
    IconButton::new(id, icon)
        .shape(IconButtonShape::Square)
        .size(ButtonSize::Compact)
        .icon_size(if matches!(icon, IconName::Pin | IconName::Unpin) {
            IconSize::Medium
        } else {
            IconSize::Small
        })
        .style(ButtonStyle::Subtle)
        .aria_label(label)
        .tooltip(ui::Tooltip::text(label))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CollaborativeNavigationGroup {
    Pinned,
    CommunitiesAndProjects,
    TasksAndThreads,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CollaborativeNavigationSourceId {
    Project(ProjectGroupKey),
    Worktree {
        project: ProjectGroupKey,
        worktree_id: WorktreeId,
        path: PathBuf,
    },
    Repository {
        project: ProjectGroupKey,
        work_directory: PathBuf,
    },
    Channel(u64),
    Thread(ThreadId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CollaborativeNavigationRowId {
    group: CollaborativeNavigationGroup,
    source: CollaborativeNavigationSourceId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollaborativeNavigationBadge {
    Unread(u32),
    NotesChanged,
    Running,
    WaitingForUser,
    Failed,
    Archived,
    Completed,
    Unknown,
    Cancelled,
    Interrupted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CollaborativeNavigationRow {
    id: CollaborativeNavigationRowId,
    label: SharedString,
    badges: Vec<CollaborativeNavigationBadge>,
}

impl CollaborativeNavigationRow {
    pub(crate) fn from_project(
        project: &Project,
        label: impl Into<SharedString>,
        badges: Vec<CollaborativeNavigationBadge>,
        cx: &App,
    ) -> Self {
        Self::from_project_group(project.project_group_key(cx), label, badges)
    }

    pub(crate) fn from_project_group(
        project: ProjectGroupKey,
        label: impl Into<SharedString>,
        badges: Vec<CollaborativeNavigationBadge>,
    ) -> Self {
        Self {
            id: CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::CommunitiesAndProjects,
                source: CollaborativeNavigationSourceId::Project(project),
            },
            label: label.into(),
            badges,
        }
    }

    pub(crate) fn from_worktree(
        project: ProjectGroupKey,
        worktree_id: WorktreeId,
        path: PathBuf,
        label: impl Into<SharedString>,
        badges: Vec<CollaborativeNavigationBadge>,
    ) -> Self {
        Self {
            id: CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::CommunitiesAndProjects,
                source: CollaborativeNavigationSourceId::Worktree {
                    project,
                    worktree_id,
                    path,
                },
            },
            label: label.into(),
            badges,
        }
    }

    pub(crate) fn from_repository(
        project: ProjectGroupKey,
        work_directory: PathBuf,
        label: impl Into<SharedString>,
        badges: Vec<CollaborativeNavigationBadge>,
    ) -> Self {
        Self {
            id: CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::CommunitiesAndProjects,
                source: CollaborativeNavigationSourceId::Repository {
                    project,
                    work_directory,
                },
            },
            label: label.into(),
            badges,
        }
    }

    pub(crate) fn from_channel(
        channel: &Channel,
        badges: Vec<CollaborativeNavigationBadge>,
    ) -> Self {
        Self {
            id: CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::CommunitiesAndProjects,
                source: CollaborativeNavigationSourceId::Channel(channel.id.0),
            },
            label: channel.name.clone(),
            badges,
        }
    }

    pub(crate) fn from_thread(
        thread: &ThreadMetadata,
        badges: Vec<CollaborativeNavigationBadge>,
    ) -> Self {
        Self {
            id: CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::TasksAndThreads,
                source: CollaborativeNavigationSourceId::Thread(thread.thread_id),
            },
            label: thread.display_title(),
            badges,
        }
    }

    pub(crate) fn pinned(source: &Self, badges: Vec<CollaborativeNavigationBadge>) -> Self {
        Self {
            id: CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::Pinned,
                source: source.id.source.clone(),
            },
            label: source.label.clone(),
            badges,
        }
    }

    pub(crate) fn id(&self) -> &CollaborativeNavigationRowId {
        &self.id
    }

    pub(crate) fn group(&self) -> CollaborativeNavigationGroup {
        self.id.group
    }

    pub(crate) fn source_id(&self) -> &CollaborativeNavigationSourceId {
        &self.id.source
    }

    pub(crate) fn label(&self) -> &SharedString {
        &self.label
    }

    pub(crate) fn badges(&self) -> &[CollaborativeNavigationBadge] {
        &self.badges
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DuplicateCollaborativeNavigationRow {
    id: CollaborativeNavigationRowId,
}

impl fmt::Display for DuplicateCollaborativeNavigationRow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "duplicate collaborative navigation row: {:?}",
            self.id
        )
    }
}

impl std::error::Error for DuplicateCollaborativeNavigationRow {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CollaborativeNavigationProjection {
    rows: Vec<CollaborativeNavigationRow>,
}

impl CollaborativeNavigationProjection {
    pub(crate) fn try_from_rows(
        rows: impl IntoIterator<Item = CollaborativeNavigationRow>,
    ) -> Result<Self, Box<DuplicateCollaborativeNavigationRow>> {
        let mut unique_ids = HashSet::new();
        let mut projected_rows = Vec::new();
        for row in rows {
            if !unique_ids.insert(row.id.clone()) {
                return Err(Box::new(DuplicateCollaborativeNavigationRow { id: row.id }));
            }
            projected_rows.push(row);
        }
        Ok(Self {
            rows: projected_rows,
        })
    }

    pub(crate) fn rows(&self) -> &[CollaborativeNavigationRow] {
        &self.rows
    }
}

pub(crate) fn render_collaborative_navigation_badges(
    row: &CollaborativeNavigationRow,
) -> Option<AnyElement> {
    if row.badges().is_empty() {
        return None;
    }

    let accessibility_label = row
        .badges()
        .iter()
        .map(|badge| match badge {
            CollaborativeNavigationBadge::Unread(count) => format!("{count} unread"),
            CollaborativeNavigationBadge::NotesChanged => "Channel notes changed".to_owned(),
            CollaborativeNavigationBadge::Running => "Running".to_owned(),
            CollaborativeNavigationBadge::WaitingForUser => "Waiting for user".to_owned(),
            CollaborativeNavigationBadge::Failed => "Failed".to_owned(),
            CollaborativeNavigationBadge::Archived => "Archived".to_owned(),
            CollaborativeNavigationBadge::Completed => "Completed".to_owned(),
            CollaborativeNavigationBadge::Unknown => "Execution outcome not recorded".to_owned(),
            CollaborativeNavigationBadge::Cancelled => "Cancelled".to_owned(),
            CollaborativeNavigationBadge::Interrupted => {
                "Interrupted; no live process restored".to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    Some(
        h_flex()
            .id(SharedString::from(format!(
                "collaborative-navigation-badges-{:?}",
                row.id()
            )))
            .gap_1()
            .role(Role::Status)
            .aria_label(accessibility_label)
            .children(row.badges().iter().map(|badge| {
                let (icon, color, label) = match badge {
                    CollaborativeNavigationBadge::Unread(count) => {
                        return Label::new(count.to_string())
                            .size(LabelSize::XSmall)
                            .color(Color::Accent)
                            .into_any_element();
                    }
                    CollaborativeNavigationBadge::NotesChanged => {
                        (IconName::File, Color::Accent, "Channel notes changed")
                    }
                    CollaborativeNavigationBadge::Running => {
                        (IconName::LoadCircle, Color::Info, "Running")
                    }
                    CollaborativeNavigationBadge::WaitingForUser => {
                        (IconName::Bell, Color::Warning, "Waiting for user")
                    }
                    CollaborativeNavigationBadge::Failed => {
                        (IconName::XCircle, Color::Error, "Failed")
                    }
                    CollaborativeNavigationBadge::Archived => {
                        (IconName::Archive, Color::Muted, "Archived")
                    }
                    CollaborativeNavigationBadge::Completed => {
                        (IconName::Check, Color::Muted, "Completed")
                    }
                    CollaborativeNavigationBadge::Unknown => (
                        IconName::Info,
                        Color::Muted,
                        "Execution outcome not recorded",
                    ),
                    CollaborativeNavigationBadge::Cancelled => {
                        (IconName::Stop, Color::Muted, "Cancelled")
                    }
                    CollaborativeNavigationBadge::Interrupted => (
                        IconName::Warning,
                        Color::Muted,
                        "Interrupted; no live process restored",
                    ),
                };
                div()
                    .id(SharedString::from(format!("navigation-status-{label}")))
                    .tooltip(ui::Tooltip::text(label))
                    .child(Icon::new(icon).size(IconSize::Small).color(color))
                    .into_any_element()
            }))
            .into_any_element(),
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use agent_ui::thread_metadata_store::WorktreePaths;
    use chrono::Utc;
    use project::AgentId;
    use rpc::proto::ChannelVisibility;
    use util::path_list::PathList;

    use super::*;

    #[gpui::test]
    async fn collaborative_navigation_projection_maps_project_entity(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = fs::FakeFs::new(cx.executor());
        let project_path = PathBuf::from("/workspace/project");
        let project = Project::test(fs, [project_path.as_path()], cx).await;
        let row = project.read_with(cx, |project, cx| {
            CollaborativeNavigationRow::from_project(project, "project", Vec::new(), cx)
        });
        let expected_key = project.read_with(cx, Project::project_group_key);

        assert_eq!(
            row.source_id(),
            &CollaborativeNavigationSourceId::Project(expected_key)
        );
    }

    #[test]
    fn collaborative_navigation_projection_uses_stable_source_ids() {
        let project_key =
            ProjectGroupKey::new(None, PathList::new(&[PathBuf::from("/workspace/project")]));
        let project_row = CollaborativeNavigationRow::from_project_group(
            project_key.clone(),
            "project",
            Vec::new(),
        );
        let worktree_row = CollaborativeNavigationRow::from_worktree(
            project_key,
            WorktreeId::from_usize(7),
            PathBuf::from("/workspace/project-feature"),
            "feature",
            vec![CollaborativeNavigationBadge::Running],
        );
        let channel = Channel {
            id: client::ChannelId(42),
            name: "community".into(),
            visibility: ChannelVisibility::Members,
            parent_path: Vec::new(),
            channel_order: 0,
        };
        let channel_row = CollaborativeNavigationRow::from_channel(
            &channel,
            vec![CollaborativeNavigationBadge::Unread(3)],
        );
        let thread_id = ThreadId::new();
        let thread = ThreadMetadata {
            execution_outcome: acp_thread::ThreadExecutionOutcome::Unknown,
            thread_id,
            session_id: None,
            agent_id: AgentId("test-agent".into()),
            title: Some("thread".into()),
            title_override: None,
            updated_at: Utc::now(),
            created_at: None,
            interacted_at: None,
            worktree_paths: WorktreePaths::default(),
            remote_connection: None,
            archived: false,
        };
        let thread_row = CollaborativeNavigationRow::from_thread(
            &thread,
            vec![CollaborativeNavigationBadge::WaitingForUser],
        );

        let projection = CollaborativeNavigationProjection::try_from_rows([
            project_row,
            worktree_row,
            channel_row.clone(),
            thread_row.clone(),
        ])
        .expect("canonical sources should project once");
        assert_eq!(projection.rows().len(), 4);
        assert_eq!(
            projection.rows()[0].group(),
            CollaborativeNavigationGroup::CommunitiesAndProjects
        );
        assert_eq!(projection.rows()[2].label().as_ref(), "community");
        assert_eq!(
            projection.rows()[2].badges(),
            &[CollaborativeNavigationBadge::Unread(3)]
        );
        assert_eq!(
            projection.rows()[3].group(),
            CollaborativeNavigationGroup::TasksAndThreads
        );
        assert_eq!(
            projection.rows()[3].id(),
            &CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::TasksAndThreads,
                source: CollaborativeNavigationSourceId::Thread(thread_id),
            }
        );

        let renamed_channel = Channel {
            name: "renamed-community".into(),
            ..channel
        };
        let renamed_channel_row =
            CollaborativeNavigationRow::from_channel(&renamed_channel, Vec::new());
        assert_eq!(channel_row.id(), renamed_channel_row.id());

        let mut renamed_thread = thread;
        renamed_thread.title_override = Some("renamed-thread".into());
        let renamed_thread_row =
            CollaborativeNavigationRow::from_thread(&renamed_thread, Vec::new());
        assert_eq!(thread_row.id(), renamed_thread_row.id());
    }

    #[test]
    fn collaborative_navigation_projection_rejects_duplicate_sources() {
        let channel = Channel {
            id: client::ChannelId(42),
            name: "community".into(),
            visibility: ChannelVisibility::Members,
            parent_path: Vec::new(),
            channel_order: 0,
        };
        let row = CollaborativeNavigationRow::from_channel(&channel, Vec::new());
        let error = CollaborativeNavigationProjection::try_from_rows([row.clone(), row])
            .expect_err("a canonical source cannot produce two rows");
        assert_eq!(
            error.id,
            CollaborativeNavigationRowId {
                group: CollaborativeNavigationGroup::CommunitiesAndProjects,
                source: CollaborativeNavigationSourceId::Channel(42),
            }
        );
    }

    #[test]
    fn collaborative_navigation_projection_allows_a_pinned_source_reference() {
        let channel = Channel {
            id: client::ChannelId(42),
            name: "community".into(),
            visibility: ChannelVisibility::Members,
            parent_path: Vec::new(),
            channel_order: 0,
        };
        let channel_row = CollaborativeNavigationRow::from_channel(&channel, Vec::new());
        let badges = vec![
            CollaborativeNavigationBadge::Unread(1),
            CollaborativeNavigationBadge::Running,
            CollaborativeNavigationBadge::WaitingForUser,
            CollaborativeNavigationBadge::Failed,
            CollaborativeNavigationBadge::Archived,
            CollaborativeNavigationBadge::Completed,
        ];
        let pinned_row = CollaborativeNavigationRow::pinned(&channel_row, badges.clone());
        let projection =
            CollaborativeNavigationProjection::try_from_rows([channel_row, pinned_row])
                .expect("the same source may appear once in distinct groups");

        assert_eq!(projection.rows().len(), 2);
        assert_eq!(
            projection.rows()[0].source_id(),
            projection.rows()[1].source_id()
        );
        assert_eq!(
            projection.rows()[1].group(),
            CollaborativeNavigationGroup::Pinned
        );
        assert_eq!(projection.rows()[1].badges(), badges);
    }
}
