use gpui::{
    Action, AnyView, Bounds, Context, DragMoveEvent, Entity, FocusHandle, Hsla, MouseButton, Point,
    Render, Role, canvas, px,
};
use settings::Settings;
use ui::{Window, prelude::*};

use crate::{
    collaborative_accessibility::{REVIEW_LABEL, TIMELINE_LABEL},
    collaborative_composer::CollaborativeComposerSurface,
    collaborative_layout_persistence::{CollaborativeLayoutState, MIN_REVIEW_WIDTH},
    collaborative_review::{
        CollaborativeReviewSlot, SelectAgentReview, SelectProjectReview, ShowCollaborativeReview,
    },
    status_bar::StatusBar,
    workspace_settings::StatusBarSettings,
};

const MIN_TIMELINE_WIDTH: Pixels = px(320.);
pub(crate) const PREFERRED_CENTER_WIDTH: Pixels = px(720.);
pub(crate) const RESIZE_HANDLE_WIDTH: Pixels = px(3.);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CollaborativeLayoutGeometry {
    pub(crate) timeline_width: Pixels,
    pub(crate) review_width: Pixels,
    pub(crate) review_visible: bool,
}

#[derive(Clone)]
struct DraggedCollaborativeReview;

impl Render for DraggedCollaborativeReview {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        gpui::Empty
    }
}

pub(crate) struct CollaborativeLayout {
    bounds: Bounds<Pixels>,
    state: CollaborativeLayoutState,
    timeline_view: Option<AnyView>,
    composer: Entity<CollaborativeComposerSurface>,
    status_bar: Entity<StatusBar>,
    review_view: Option<AnyView>,
    native_editor: AnyView,
    native_editor_tabs: AnyView,
    selected_review_slot: Option<CollaborativeReviewSlot>,
    agent_review_available: bool,
    project_review_available: bool,
    timeline_focus_handle: FocusHandle,
    review_focus_handle: FocusHandle,
}

impl CollaborativeLayout {
    pub(crate) fn new(
        state: CollaborativeLayoutState,
        composer: Entity<CollaborativeComposerSurface>,
        status_bar: Entity<StatusBar>,
        native_editor: AnyView,
        native_editor_tabs: AnyView,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            bounds: Bounds::default(),
            selected_review_slot: state.selected_review_slot(),
            state,
            timeline_view: None,
            composer,
            status_bar,
            review_view: None,
            native_editor,
            native_editor_tabs,
            agent_review_available: false,
            project_review_available: false,
            timeline_focus_handle: cx.focus_handle().tab_stop(true),
            review_focus_handle: cx.focus_handle().tab_stop(true),
        }
    }

    pub(crate) fn set_timeline_view(
        &mut self,
        timeline_view: Option<AnyView>,
        cx: &mut Context<Self>,
    ) {
        let changed = self.timeline_view.as_ref().map(AnyView::entity_id)
            != timeline_view.as_ref().map(AnyView::entity_id);
        if changed {
            self.timeline_view = timeline_view;
            cx.notify();
        }
    }

    pub(crate) fn set_review_view(
        &mut self,
        review_view: Option<AnyView>,
        selected_slot: Option<CollaborativeReviewSlot>,
        preferred_slot: Option<CollaborativeReviewSlot>,
        agent_available: bool,
        project_available: bool,
        cx: &mut Context<Self>,
    ) {
        let changed = self.review_view.as_ref().map(AnyView::entity_id)
            != review_view.as_ref().map(AnyView::entity_id)
            || self.selected_review_slot != selected_slot
            || self.state.selected_review_slot() != preferred_slot
            || self.agent_review_available != agent_available
            || self.project_review_available != project_available;
        if changed {
            self.review_view = review_view;
            self.selected_review_slot = selected_slot;
            self.state = self.state.clone().with_selected_review_slot(preferred_slot);
            self.agent_review_available = agent_available;
            self.project_review_available = project_available;
            cx.notify();
        }
    }

    pub(crate) fn review_requested(&self) -> bool {
        self.state.review_requested()
    }

    pub(crate) fn rail_requested(&self) -> bool {
        self.state.rail_requested()
    }

    pub(crate) fn set_rail_requested(&mut self, requested: bool, cx: &mut Context<Self>) {
        let state = self.state.clone().with_rail_requested(requested);
        if state != self.state {
            self.state = state;
            cx.notify();
        }
    }

    pub(crate) fn request_dock(
        &mut self,
        position: crate::dock::DockPosition,
        cx: &mut Context<Self>,
    ) {
        let state = self.state.clone().with_dock_requested(position);
        if state != self.state {
            self.state = state;
            cx.notify();
        }
    }

    pub(crate) fn restore_review_requested(&mut self, requested: bool, cx: &mut Context<Self>) {
        let state = self.state.clone().with_review_requested(requested);
        if state != self.state {
            self.state = state;
            cx.notify();
        }
    }

    pub(crate) fn native_editor_active(&self) -> bool {
        self.state.native_editor_active()
    }

    pub(crate) fn set_native_editor_active(&mut self, active: bool, cx: &mut Context<Self>) {
        let state = self.state.clone().with_native_editor_active(active);
        let state = if active {
            state.with_review_requested(true)
        } else {
            state
        };
        if state != self.state {
            self.state = state;
            cx.notify();
        }
    }

    pub(crate) fn native_editor_has_focus(&self, window: &Window, cx: &App) -> bool {
        self.state.native_editor_active()
            && !self.review_focus_handle.is_focused(window)
            && self.review_focus_handle.contains_focused(window, cx)
    }

    pub(crate) fn timeline_visible(&self) -> bool {
        self.geometry_for_current_state(self.bounds.size.width)
            .timeline_width
            > px(0.)
    }

    pub(crate) fn timeline_focus_handle(&self) -> FocusHandle {
        self.timeline_focus_handle.clone()
    }

    pub(crate) fn review_focus_handle(&self) -> Option<FocusHandle> {
        let geometry = self.geometry_for_current_state(self.bounds.size.width);
        geometry
            .review_visible
            .then(|| self.review_focus_handle.clone())
    }

    pub(crate) fn state(&self) -> CollaborativeLayoutState {
        self.state.clone()
    }

    pub(crate) fn set_project_review_base(
        &mut self,
        base: project::git_store::diff_buffer_list::DiffBase,
        cx: &mut Context<Self>,
    ) {
        if self.state.project_review_base() != Some(&base) {
            self.state = self.state.clone().with_project_review_base(base);
            cx.notify();
        }
    }

    pub(crate) fn rail_width(&self) -> Pixels {
        px(self.state.rail_width())
    }

    pub(crate) fn set_rail_width(&mut self, width: Pixels, cx: &mut Context<Self>) {
        let state = self.state.clone().with_rail_width(f32::from(width));
        if state != self.state {
            self.state = state;
            cx.notify();
        }
    }

    pub(crate) fn reset_rail_width(&mut self, cx: &mut Context<Self>) {
        let state = self.state.clone().reset_rail_width();
        if state != self.state {
            self.state = state;
            cx.notify();
        }
    }

    pub(crate) fn toggle_review(&mut self, cx: &mut Context<Self>) {
        self.state = self
            .state
            .clone()
            .with_review_requested(!self.state.review_requested());
        cx.notify();
    }

    pub(crate) fn geometry_for(
        available_width: Pixels,
        review_requested: bool,
        requested_review_width: Pixels,
    ) -> CollaborativeLayoutGeometry {
        let available_width = available_width.max(px(0.));
        let minimum_review_width = px(MIN_REVIEW_WIDTH);
        let minimum_expanded_width = Self::minimum_center_width(true);
        if !review_requested || available_width < minimum_expanded_width {
            return CollaborativeLayoutGeometry {
                timeline_width: available_width,
                review_width: px(0.),
                review_visible: false,
            };
        }

        let maximum_review_width = available_width - MIN_TIMELINE_WIDTH - RESIZE_HANDLE_WIDTH;
        let review_width = requested_review_width.clamp(minimum_review_width, maximum_review_width);
        CollaborativeLayoutGeometry {
            timeline_width: available_width - RESIZE_HANDLE_WIDTH - review_width,
            review_width,
            review_visible: true,
        }
    }

    pub(crate) fn minimum_center_width(review_requested: bool) -> Pixels {
        if review_requested {
            MIN_TIMELINE_WIDTH + RESIZE_HANDLE_WIDTH + px(MIN_REVIEW_WIDTH)
        } else {
            MIN_TIMELINE_WIDTH
        }
    }

    fn available_width(&self, window: &Window) -> Pixels {
        if self.bounds.size.width > px(0.) {
            self.bounds
                .size
                .width
                .min((window.viewport_size().width - self.bounds.origin.x).max(px(0.)))
        } else {
            window.viewport_size().width
        }
    }

    fn resize_review(
        &mut self,
        bounds: Bounds<Pixels>,
        pointer: Point<Pixels>,
        cx: &mut Context<Self>,
    ) {
        let geometry = Self::geometry_for_state(bounds.size.width, &self.state);
        if !geometry.review_visible {
            return;
        }

        let maximum_review_width = bounds.size.width - MIN_TIMELINE_WIDTH - RESIZE_HANDLE_WIDTH;
        let review_width =
            (bounds.right() - pointer.x).clamp(px(MIN_REVIEW_WIDTH), maximum_review_width);
        let state = if self.state.review_fraction().is_some() {
            self.state.clone().with_review_fraction(
                f64::from(f32::from(review_width))
                    / f64::from(f32::from(bounds.size.width - RESIZE_HANDLE_WIDTH)),
            )
        } else {
            self.state
                .clone()
                .with_review_width(f32::from(review_width))
        };
        if state != self.state {
            self.state = state;
            cx.notify();
        }
    }

    fn render_timeline(
        geometry: CollaborativeLayoutGeometry,
        background: Hsla,
        focus_border: Hsla,
        timeline_view: Option<AnyView>,
        composer: Entity<CollaborativeComposerSurface>,
        status_bar: Option<Entity<StatusBar>>,
        composer_footer: Option<AnyView>,
        focus_handle: &FocusHandle,
        input_in_timeline: bool,
    ) -> impl IntoElement {
        let timeline_available = timeline_view.is_some();
        v_flex()
            .id("collaborative-timeline-region")
            .debug_selector(|| "COLLABORATIVE-TIMELINE-REGION".to_owned())
            .h_full()
            .w(geometry.timeline_width)
            .flex_none()
            .track_focus(focus_handle)
            .focus_visible(move |style| style.border_1().border_color(focus_border))
            .tab_index(0)
            .role(Role::Document)
            .aria_label(TIMELINE_LABEL)
            .overflow_hidden()
            .bg(background)
            .when_some(timeline_view, |this, timeline_view| {
                this.child(
                    div()
                        .id("collaborative-timeline-content")
                        .debug_selector(|| "COLLABORATIVE-TIMELINE-CONTENT".to_owned())
                        .w_full()
                        .flex_1()
                        .min_h_0()
                        .overflow_hidden()
                        .child(timeline_view),
                )
            })
            .when(!timeline_available, |this| {
                this.child(
                    v_flex()
                        .w_full()
                        .flex_1()
                        .min_h_0()
                        .items_center()
                        .justify_center()
                        .gap_1()
                        .child(
                            Label::new("Select an agent task to view its activity")
                                .size(LabelSize::Small)
                                .color(Color::Muted),
                        )
                        .child(
                            Label::new("Channel activity requires an authorized collaboration service connection")
                                .size(LabelSize::XSmall)
                                .color(Color::Muted),
                        )
                        .map(|empty_state| {
                            #[cfg(feature = "agentic")]
                            let empty_state = empty_state.child(
                                Button::new("collaborative-start-thread", "New agent thread")
                                    .on_click(|_, window, cx| {
                                        window.dispatch_action(crate::NewThread.boxed_clone(), cx);
                                    }),
                            );
                            empty_state
                        }),
                )
            })
            .when(!timeline_available || !input_in_timeline, |this| {
                this.child(composer)
            })
            .when_some(composer_footer, |this, footer| {
                this.child(h_flex().w_full().justify_end().p_1().child(footer))
            })
            .when_some(status_bar, |this, status_bar| {
                this.child(
                    div()
                        .debug_selector(|| "COLLABORATIVE-STATUS-BAR".to_owned())
                        .w_full()
                        .flex_none()
                        .overflow_hidden()
                        .child(status_bar),
                )
            })
    }

    pub(crate) fn geometry(&self, window: &Window) -> CollaborativeLayoutGeometry {
        self.geometry_for_current_state(self.available_width(window))
    }

    fn geometry_for_current_state(&self, available_width: Pixels) -> CollaborativeLayoutGeometry {
        Self::geometry_for_state(available_width, &self.state)
    }

    pub(crate) fn geometry_for_state(
        available_width: Pixels,
        state: &CollaborativeLayoutState,
    ) -> CollaborativeLayoutGeometry {
        let review_fraction = state.review_fraction();
        let review_requested = state.review_requested()
            && (review_fraction.is_none() || available_width >= PREFERRED_CENTER_WIDTH);
        let requested_width = px(state.requested_review_width_for(f32::from(
            (available_width - RESIZE_HANDLE_WIDTH).max(px(0.)),
        )));
        Self::geometry_for(available_width, review_requested, requested_width)
    }

    pub(crate) fn render_review_header(&self) -> impl IntoElement + use<> {
        let selected_slot = self.selected_review_slot;
        let agent_available = self.agent_review_available;
        let project_available = self.project_review_available;
        h_flex()
            .debug_selector(|| "COLLABORATIVE-REVIEW-TABS".to_owned())
            .h_full()
            .min_w_0()
            .flex_1()
            .px_2()
            .gap_1()
            .child(
                div().flex_none().child(
                    Button::new("collaborative-review-tab", "Review Changes")
                        .style(ButtonStyle::Subtle)
                        .toggle_state(!self.state.native_editor_active())
                        .on_click(|_, window, cx| {
                            window.dispatch_action(ShowCollaborativeReview.boxed_clone(), cx);
                        }),
                ),
            )
            .child(
                ui::PopoverMenu::new("collaborative-review-provider")
                    .trigger(
                        IconButton::new(
                            "collaborative-review-provider-menu",
                            IconName::ChevronDown,
                        )
                        .size(ButtonSize::Compact)
                        .aria_label("Choose review provider")
                        .tooltip(ui::Tooltip::text("Choose review provider")),
                    )
                    .menu(move |window, cx| {
                        Some(ui::ContextMenu::build(window, cx, |menu, _, _| {
                            menu.toggleable_entry_disabled_when(
                                "Agent changes",
                                selected_slot == Some(CollaborativeReviewSlot::AgentChanges),
                                !agent_available,
                                ui::IconPosition::Start,
                                Some(SelectAgentReview.boxed_clone()),
                                |window, cx| {
                                    window.dispatch_action(SelectAgentReview.boxed_clone(), cx)
                                },
                            )
                            .toggleable_entry_disabled_when(
                                "Project changes",
                                selected_slot == Some(CollaborativeReviewSlot::ProjectChanges),
                                !project_available,
                                ui::IconPosition::Start,
                                Some(SelectProjectReview.boxed_clone()),
                                |window, cx| {
                                    window.dispatch_action(SelectProjectReview.boxed_clone(), cx)
                                },
                            )
                        }))
                    }),
            )
            .child(
                div()
                    .min_w_0()
                    .flex_1()
                    .child(self.native_editor_tabs.clone()),
            )
    }

    fn render_review(
        geometry: CollaborativeLayoutGeometry,
        background: Hsla,
        focus_border: Hsla,
        review_view: Option<AnyView>,
        focus_handle: &FocusHandle,
    ) -> impl IntoElement {
        let review_available = review_view.is_some();
        v_flex()
            .id("collaborative-review-region")
            .debug_selector(|| "COLLABORATIVE-REVIEW-REGION".to_owned())
            .h_full()
            .w(geometry.review_width)
            .min_w(geometry.review_width.min(px(MIN_REVIEW_WIDTH)))
            .flex_none()
            .track_focus(focus_handle)
            .focus_visible(move |style| style.border_1().border_color(focus_border))
            .tab_index(0)
            .role(Role::Complementary)
            .aria_label(REVIEW_LABEL)
            .overflow_hidden()
            .bg(background)
            .when_some(review_view, |this, review_view| {
                this.child(
                    div()
                        .id("collaborative-review-content")
                        .debug_selector(|| "COLLABORATIVE-REVIEW-CONTENT".to_owned())
                        .flex_1()
                        .min_h_0()
                        .w_full()
                        .overflow_hidden()
                        .child(review_view),
                )
            })
            .when(!review_available, |this| {
                this.child(
                    v_flex().size_full().items_center().justify_center().child(
                        Label::new("Select an agent task or a project to review changes")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    ),
                )
            })
    }

    fn render_resize_handle(border: Hsla) -> impl IntoElement {
        div()
            .id("collaborative-review-resize-handle")
            .debug_selector(|| "COLLABORATIVE-REVIEW-RESIZE-HANDLE".to_owned())
            .h_full()
            .w(RESIZE_HANDLE_WIDTH)
            .flex_none()
            .cursor_col_resize()
            .border_l_1()
            .border_color(border)
            .on_drag(DraggedCollaborativeReview, |dragged, _, _, cx| {
                cx.stop_propagation();
                cx.new(|_| dragged.clone())
            })
            .on_mouse_down(MouseButton::Left, |_, _, cx| {
                cx.stop_propagation();
            })
    }

    #[cfg(test)]
    pub(crate) fn test_review_width(&self) -> Pixels {
        px(self
            .state
            .requested_review_width_for(f32::from(self.bounds.size.width - RESIZE_HANDLE_WIDTH)))
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::collaborative_layout_persistence::DEFAULT_REVIEW_WIDTH;

    const REFERENCE_RAIL_WIDTH: f32 = 226.;

    #[test]
    fn collaborative_default_review_tracks_available_center() {
        let state = CollaborativeLayoutState::default();
        for available in [732., 763., 965., 920.] {
            let geometry = CollaborativeLayout::geometry_for_state(px(available), &state);
            assert!(geometry.review_visible);
            assert_eq!(geometry.timeline_width, geometry.review_width);
            assert_eq!(
                geometry.timeline_width + geometry.review_width + RESIZE_HANDLE_WIDTH,
                px(available)
            );
        }
        let narrow = CollaborativeLayout::geometry_for_state(px(650.), &state);
        assert!(!narrow.review_visible);
        assert_eq!(narrow.timeline_width, px(650.));
        let explicit = state.with_review_width(354.5);
        let expanded = CollaborativeLayout::geometry_for_state(px(739.), &explicit);
        assert_eq!(expanded.review_width, px(354.5));
        assert_eq!(expanded.timeline_width, px(381.5));
    }

    #[test]
    fn collaborative_reference_and_responsive_geometry() {
        let expanded = CollaborativeLayout::geometry_for(
            px(1930. / 2. - REFERENCE_RAIL_WIDTH),
            true,
            px(DEFAULT_REVIEW_WIDTH),
        );
        assert!(expanded.review_visible);
        assert_eq!(expanded.review_width, px(354.5));
        assert_eq!(expanded.timeline_width, px(381.5));
        assert_eq!(
            (px(REFERENCE_RAIL_WIDTH) + expanded.timeline_width + RESIZE_HANDLE_WIDTH) * 2.,
            px(1221.)
        );

        let collapsed = CollaborativeLayout::geometry_for(
            px(1928. / 2. - REFERENCE_RAIL_WIDTH),
            false,
            px(DEFAULT_REVIEW_WIDTH),
        );
        assert!(!collapsed.review_visible);
        assert_eq!(collapsed.timeline_width, px(738.));
        assert_eq!(collapsed.review_width, px(0.));

        let narrow = CollaborativeLayout::geometry_for(px(602.), true, px(DEFAULT_REVIEW_WIDTH));
        assert!(!narrow.review_visible);
        assert_eq!(narrow.timeline_width, px(602.));

        let wider = CollaborativeLayout::geometry_for(px(900.), true, px(DEFAULT_REVIEW_WIDTH));
        assert!(wider.review_visible);
        assert_eq!(wider.review_width, px(DEFAULT_REVIEW_WIDTH));
        assert_eq!(wider.timeline_width, px(542.5));
    }
}

impl Render for CollaborativeLayout {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let geometry = self.geometry(window);
        let colors = cx.theme().colors();
        let timeline_background = colors.background;
        let review_background = colors.panel_background;
        let border = colors.border;
        let focus_border = colors.border_focused;
        let layout = cx.entity();
        let timeline_view = self.timeline_view.clone();
        let review_view = if self.state.native_editor_active() {
            Some(self.native_editor.clone())
        } else {
            self.review_view.clone()
        };

        h_flex()
            .id("collaborative-layout")
            .debug_selector(|| "COLLABORATIVE-LAYOUT".to_owned())
            .relative()
            .size_full()
            .overflow_hidden()
            .on_drag_move(cx.listener(
                |this, event: &DragMoveEvent<DraggedCollaborativeReview>, _, cx| {
                    this.resize_review(event.bounds, event.event.position, cx);
                },
            ))
            .child(
                canvas(
                    move |bounds, window, cx| {
                        if layout.read(cx).bounds != bounds {
                            layout.update(cx, |layout, _| layout.bounds = bounds);
                            // Geometry is measured after render; notifying during prepaint can
                            // be cleared when this frame finishes and leave the old split painted.
                            window.on_next_frame(move |_, cx| {
                                layout.update(cx, |_, cx| cx.notify());
                            });
                        }
                    },
                    |_, _, _, _| {},
                )
                .absolute()
                .size_full(),
            )
            .when(geometry.timeline_width > px(0.), |this| {
                this.child(Self::render_timeline(
                    geometry,
                    timeline_background,
                    focus_border,
                    timeline_view,
                    self.composer.clone(),
                    StatusBarSettings::get_global(cx)
                        .show
                        .then(|| self.status_bar.clone()),
                    (!StatusBarSettings::get_global(cx).show)
                        .then(|| self.status_bar.read(cx).collaborative_composer_footer())
                        .flatten(),
                    &self.timeline_focus_handle,
                    self.composer.read(cx).input_in_timeline(),
                ))
            })
            .when(geometry.review_visible, |this| {
                this.when(geometry.timeline_width > px(0.), |this| {
                    this.child(Self::render_resize_handle(border))
                })
                .child(Self::render_review(
                    geometry,
                    review_background,
                    focus_border,
                    review_view,
                    &self.review_focus_handle,
                ))
            })
    }
}
