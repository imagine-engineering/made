use gpui::{
    Action, AnyView, App, Context, Entity, FocusHandle, Focusable, KeyDownEvent, Render, Role,
};
use project::Project;
use ui::{Window, prelude::*};

use crate::{
    collaborative_accessibility::WORKSPACE_LABEL,
    collaborative_composer::CollaborativeComposerSurface,
    collaborative_focus::CollaborativeFocusRegion,
    collaborative_layout::CollaborativeLayout,
    collaborative_layout_persistence::CollaborativeLayoutState,
    collaborative_participants::{
        CollaborativeParticipantProvider, CollaborativeParticipantProviderState,
    },
    collaborative_shell_state::{
        CollaborativeShellPhase, CollaborativeShellState, CollaborativeShellStatus,
    },
    collaborative_top_bar::CollaborativeTopBar,
    status_bar::StatusBar,
};

pub(crate) struct CollaborativeWorkspace {
    project: Entity<Project>,
    layout: Entity<CollaborativeLayout>,
    composer: Entity<CollaborativeComposerSurface>,
    participant_provider: Option<CollaborativeParticipantProvider>,
    shell_state: Entity<CollaborativeShellState>,
    focus_handle: FocusHandle,
}

impl CollaborativeWorkspace {
    pub(crate) fn new(
        project: Entity<Project>,
        workspace: gpui::WeakEntity<crate::Workspace>,
        focus_handle: FocusHandle,
        layout_state: CollaborativeLayoutState,
        status_bar: Entity<StatusBar>,
        cx: &mut Context<Self>,
    ) -> Self {
        let composer = cx.new(CollaborativeComposerSurface::new);
        let native_editor = cx.new(|_| {
            crate::collaborative_review::CollaborativeEditorSurface::new(workspace.clone())
        });
        let native_editor_tabs =
            cx.new(|_| crate::collaborative_review::CollaborativeEditorTabs::new(workspace));
        let layout = cx.new(|cx| {
            CollaborativeLayout::new(
                layout_state,
                composer.clone(),
                status_bar,
                native_editor.into(),
                native_editor_tabs.into(),
                cx,
            )
        });
        let shell_state = cx.new(|_| CollaborativeShellState::new());
        cx.observe(&layout, |_, _, cx| cx.notify()).detach();
        cx.observe(&composer, |_, _, cx| cx.notify()).detach();
        cx.observe(&shell_state, |_, _, cx| cx.notify()).detach();
        Self {
            project,
            layout,
            composer,
            participant_provider: None,
            shell_state,
            focus_handle,
        }
    }

    #[cfg(test)]
    pub(crate) fn project_entity_id(&self) -> gpui::EntityId {
        self.project.entity_id()
    }

    pub(crate) fn layout_state(&self, cx: &App) -> CollaborativeLayoutState {
        self.layout.read(cx).state()
    }

    pub(crate) fn request_dock(
        &mut self,
        position: crate::dock::DockPosition,
        cx: &mut Context<Self>,
    ) {
        self.layout
            .update(cx, |layout, cx| layout.request_dock(position, cx));
    }

    pub(crate) fn restore_review_requested(&mut self, requested: bool, cx: &mut Context<Self>) {
        self.layout.update(cx, |layout, cx| {
            layout.restore_review_requested(requested, cx)
        });
    }

    pub(crate) fn set_project_review_base(
        &mut self,
        base: project::git_store::diff_buffer_list::DiffBase,
        cx: &mut Context<Self>,
    ) {
        self.layout
            .update(cx, |layout, cx| layout.set_project_review_base(base, cx));
    }

    pub(crate) fn native_editor_active(&self, cx: &App) -> bool {
        self.layout.read(cx).native_editor_active()
    }

    pub(crate) fn set_native_editor_active(&mut self, active: bool, cx: &mut Context<Self>) {
        self.layout.update(cx, |layout, cx| {
            layout.set_native_editor_active(active, cx);
        });
    }

    pub(crate) fn top_bar(&self, cx: &App) -> CollaborativeTopBar {
        CollaborativeTopBar::new(
            self.project.clone(),
            self.layout.clone(),
            self.participant_provider
                .as_ref()
                .map(|provider| provider.state(cx))
                .unwrap_or(CollaborativeParticipantProviderState::Unavailable),
        )
    }

    pub(crate) fn timeline_width(&self, window: &Window, cx: &App) -> Pixels {
        self.layout.read(cx).geometry(window).timeline_width
    }

    pub(crate) fn rail_width(&self, cx: &App) -> Pixels {
        self.layout.read(cx).rail_width()
    }

    pub(crate) fn rail_requested(&self, cx: &App) -> bool {
        self.layout.read(cx).rail_requested()
    }

    pub(crate) fn set_rail_requested(&mut self, requested: bool, cx: &mut Context<Self>) {
        self.layout
            .update(cx, |layout, cx| layout.set_rail_requested(requested, cx));
    }

    pub(crate) fn set_rail_width(&mut self, width: Pixels, cx: &mut Context<Self>) {
        self.layout
            .update(cx, |layout, cx| layout.set_rail_width(width, cx));
    }

    pub(crate) fn reset_rail_width(&mut self, cx: &mut Context<Self>) {
        self.layout
            .update(cx, CollaborativeLayout::reset_rail_width);
    }

    pub(crate) fn toggle_review(&mut self, cx: &mut Context<Self>) {
        self.layout.update(cx, CollaborativeLayout::toggle_review);
    }

    pub(crate) fn set_review_view(
        &mut self,
        review_view: Option<AnyView>,
        selected_slot: Option<crate::collaborative_review::CollaborativeReviewSlot>,
        preferred_slot: Option<crate::collaborative_review::CollaborativeReviewSlot>,
        agent_available: bool,
        project_available: bool,
        cx: &mut Context<Self>,
    ) {
        self.layout.update(cx, |layout, cx| {
            layout.set_review_view(
                review_view,
                selected_slot,
                preferred_slot,
                agent_available,
                project_available,
                cx,
            )
        });
    }

    pub(crate) fn set_timeline_view(
        &mut self,
        timeline_view: Option<AnyView>,
        cx: &mut Context<Self>,
    ) {
        self.layout
            .update(cx, |layout, cx| layout.set_timeline_view(timeline_view, cx));
    }

    pub(crate) fn set_composer_view(
        &mut self,
        composer_view: Option<AnyView>,
        composer_focus_handle: Option<FocusHandle>,
        unavailable_message: Option<gpui::SharedString>,
        input_in_timeline: bool,
        cx: &mut Context<Self>,
    ) {
        self.composer.update(cx, |composer, cx| {
            composer.set_view(
                composer_view,
                composer_focus_handle,
                unavailable_message,
                input_in_timeline,
                cx,
            )
        });
    }

    pub(crate) fn set_participant_provider(
        &mut self,
        provider: Option<CollaborativeParticipantProvider>,
        cx: &mut Context<Self>,
    ) {
        self.participant_provider = provider;
        cx.notify();
    }

    #[cfg(test)]
    pub(crate) fn shell_state_entity(&self) -> Entity<CollaborativeShellState> {
        self.shell_state.clone()
    }

    #[cfg(test)]
    pub(crate) fn layout_entity(&self) -> Entity<CollaborativeLayout> {
        self.layout.clone()
    }

    pub(crate) fn timeline_visible(&self, cx: &App) -> bool {
        self.layout.read(cx).timeline_visible()
    }

    pub(crate) fn composer_focus_handle(&self, cx: &App) -> FocusHandle {
        self.composer.read(cx).focus_handle()
    }

    pub(crate) fn focus_region_handles(
        &self,
        cx: &App,
    ) -> Vec<(CollaborativeFocusRegion, FocusHandle)> {
        let layout = self.layout.read(cx);
        let mut regions = Vec::new();
        if layout.timeline_visible() {
            regions.extend([
                (
                    CollaborativeFocusRegion::Timeline,
                    layout.timeline_focus_handle(),
                ),
                (
                    CollaborativeFocusRegion::Composer,
                    self.composer_focus_handle(cx),
                ),
            ]);
        }
        if let Some(review) = layout.review_focus_handle() {
            regions.push((CollaborativeFocusRegion::Review, review));
        }
        regions
    }
}

impl Focusable for CollaborativeWorkspace {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for CollaborativeWorkspace {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let shell_ready = matches!(
            self.shell_state.read(cx).phase(),
            CollaborativeShellPhase::Ready
        );
        div()
            .id(("collaborative-workspace", self.project.entity_id()))
            .debug_selector(|| "COLLABORATIVE-WORKSPACE".to_owned())
            .size_full()
            .flex()
            .flex_col()
            .key_context("CollaborativeWorkspace")
            .track_focus(&self.focus_handle)
            .role(Role::Main)
            .aria_label(WORKSPACE_LABEL)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                if event.keystroke.key != "tab"
                    || this.layout.read(cx).native_editor_has_focus(window, cx)
                {
                    return;
                }
                cx.stop_propagation();
                if event.keystroke.modifiers.shift {
                    window.dispatch_action(
                        crate::collaborative_focus::FocusPreviousCollaborativeRegion.boxed_clone(),
                        cx,
                    );
                } else {
                    window.dispatch_action(
                        crate::collaborative_focus::FocusNextCollaborativeRegion.boxed_clone(),
                        cx,
                    );
                }
            }))
            .bg(cx.theme().colors().background)
            .when(!shell_ready, |this| {
                this.child(CollaborativeShellStatus::new(self.shell_state.clone()))
            })
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .overflow_hidden()
                    .child(self.layout.clone()),
            )
    }
}
