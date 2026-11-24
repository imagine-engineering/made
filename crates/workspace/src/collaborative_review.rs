use std::{error::Error, fmt};

use gpui::{AnyView, Entity, EntityId, actions};
use project::Project;
use serde::{Deserialize, Serialize};

actions!(
    collaborative_workspace,
    [
        ShowCollaborativeReview,
        SelectAgentReview,
        SelectProjectReview,
        ToggleCollaborativeReview
    ]
);

pub(crate) struct CollaborativeEditorSurface {
    workspace: gpui::WeakEntity<crate::Workspace>,
}

pub(crate) struct CollaborativeEditorTabs {
    workspace: gpui::WeakEntity<crate::Workspace>,
}

impl CollaborativeEditorTabs {
    pub(crate) fn new(workspace: gpui::WeakEntity<crate::Workspace>) -> Self {
        Self { workspace }
    }
}

impl gpui::Render for CollaborativeEditorTabs {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use gpui::{InteractiveElement as _, IntoElement as _, ParentElement as _, Styled as _};
        let Some(workspace) = self.workspace.upgrade() else {
            return gpui::Empty.into_any_element();
        };
        let pane = workspace.read(cx).active_pane().clone();
        pane.update(cx, |pane, cx| {
            if pane.collaborative_tab_bar_in_header(cx) {
                gpui::div()
                    .debug_selector(|| "COLLABORATIVE-NATIVE-TABS".to_owned())
                    .min_w_0()
                    .child(pane.render_tab_bar(window, cx))
                    .into_any_element()
            } else {
                gpui::Empty.into_any_element()
            }
        })
    }
}

impl CollaborativeEditorSurface {
    pub(crate) fn new(workspace: gpui::WeakEntity<crate::Workspace>) -> Self {
        Self { workspace }
    }
}

impl gpui::Render for CollaborativeEditorSurface {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use gpui::IntoElement as _;
        let Some(workspace) = self.workspace.upgrade() else {
            return gpui::Empty.into_any_element();
        };
        workspace.update(cx, |workspace, cx| {
            let render_context = crate::pane_group::PaneRenderContext {
                follower_states: &workspace.follower_states,
                active_call: workspace.active_call(),
                active_pane: &workspace.active_pane,
                app_state: &workspace.app_state,
                project: &workspace.project,
                workspace: &workspace.weak_self,
            };
            workspace
                .render_center(&render_context, window, cx)
                .into_any_element()
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollaborativeReviewSlot {
    AgentChanges,
    ProjectChanges,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollaborativeReviewRegistration {
    slot: CollaborativeReviewSlot,
    provider_id: EntityId,
}

impl CollaborativeReviewRegistration {
    pub fn slot(self) -> CollaborativeReviewSlot {
        self.slot
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollaborativeReviewRegistrationError {
    ProjectMismatch,
    SlotOccupied(CollaborativeReviewSlot),
}

impl fmt::Display for CollaborativeReviewRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectMismatch => {
                formatter.write_str("review provider belongs to a different project")
            }
            Self::SlotOccupied(slot) => write!(formatter, "review slot {slot:?} is occupied"),
        }
    }
}

impl Error for CollaborativeReviewRegistrationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollaborativeReviewSelectionError {
    SlotUnavailable(CollaborativeReviewSlot),
}

impl fmt::Display for CollaborativeReviewSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlotUnavailable(slot) => {
                write!(formatter, "review slot {slot:?} is unavailable")
            }
        }
    }
}

impl Error for CollaborativeReviewSelectionError {}

pub struct CollaborativeReviewHost {
    project: Entity<Project>,
    agent_changes: Option<AnyView>,
    project_changes: Option<AnyView>,
    selected_slot: Option<CollaborativeReviewSlot>,
}

impl CollaborativeReviewHost {
    pub fn new(project: Entity<Project>) -> Self {
        Self {
            project,
            agent_changes: None,
            project_changes: None,
            selected_slot: None,
        }
    }

    pub fn restore_selected_slot(&mut self, slot: Option<CollaborativeReviewSlot>) {
        self.selected_slot = slot;
    }

    pub fn project(&self) -> &Entity<Project> {
        &self.project
    }

    pub fn register(
        &mut self,
        slot: CollaborativeReviewSlot,
        project: &Entity<Project>,
        provider: AnyView,
    ) -> Result<CollaborativeReviewRegistration, CollaborativeReviewRegistrationError> {
        if project.entity_id() != self.project.entity_id() {
            return Err(CollaborativeReviewRegistrationError::ProjectMismatch);
        }

        let destination = self.provider_mut(slot);
        if destination.is_some() {
            return Err(CollaborativeReviewRegistrationError::SlotOccupied(slot));
        }

        let registration = CollaborativeReviewRegistration {
            slot,
            provider_id: provider.entity_id(),
        };
        *destination = Some(provider);
        Ok(registration)
    }

    pub fn unregister(&mut self, registration: CollaborativeReviewRegistration) -> bool {
        let destination = self.provider_mut(registration.slot);
        let matches_registration = destination
            .as_ref()
            .is_some_and(|provider| provider.entity_id() == registration.provider_id);
        if !matches_registration {
            return false;
        }

        *destination = None;
        true
    }

    pub fn select(
        &mut self,
        slot: CollaborativeReviewSlot,
    ) -> Result<bool, CollaborativeReviewSelectionError> {
        if self.provider(slot).is_none() {
            return Err(CollaborativeReviewSelectionError::SlotUnavailable(slot));
        }

        let changed = self.selected_slot != Some(slot);
        self.selected_slot = Some(slot);
        Ok(changed)
    }

    pub fn selected_slot(&self) -> Option<CollaborativeReviewSlot> {
        self.resolved_slot()
    }

    pub fn preferred_slot(&self) -> Option<CollaborativeReviewSlot> {
        self.selected_slot
    }

    pub fn selected_view(&self) -> Option<AnyView> {
        self.resolved_slot()
            .and_then(|slot| self.provider(slot))
            .cloned()
    }

    pub fn provider_available(&self, slot: CollaborativeReviewSlot) -> bool {
        self.provider(slot).is_some()
    }

    pub fn visible_view(&self, review_requested: bool) -> Option<AnyView> {
        review_requested.then(|| self.selected_view()).flatten()
    }

    fn resolved_slot(&self) -> Option<CollaborativeReviewSlot> {
        self.selected_slot
            .filter(|slot| self.provider(*slot).is_some())
            .or_else(|| {
                self.provider(CollaborativeReviewSlot::AgentChanges)
                    .is_some()
                    .then_some(CollaborativeReviewSlot::AgentChanges)
            })
            .or_else(|| {
                self.provider(CollaborativeReviewSlot::ProjectChanges)
                    .is_some()
                    .then_some(CollaborativeReviewSlot::ProjectChanges)
            })
    }

    fn provider(&self, slot: CollaborativeReviewSlot) -> Option<&AnyView> {
        match slot {
            CollaborativeReviewSlot::AgentChanges => self.agent_changes.as_ref(),
            CollaborativeReviewSlot::ProjectChanges => self.project_changes.as_ref(),
        }
    }

    fn provider_mut(&mut self, slot: CollaborativeReviewSlot) -> &mut Option<AnyView> {
        match slot {
            CollaborativeReviewSlot::AgentChanges => &mut self.agent_changes,
            CollaborativeReviewSlot::ProjectChanges => &mut self.project_changes,
        }
    }
}
