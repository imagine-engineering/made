use crate::{ItemHandle, MultiWorkspace, Pane, SidebarSide};
#[cfg(feature = "multiplayer-tools")]
use crate::{
    SwitchToCollaborativeWorkspace,
    collaborative_participants::{
        CollaborativeExecutionPhase, CollaborativeParticipantProvider,
        CollaborativeParticipantProviderState,
    },
    dock::{DockPanelButton, PanelButtons},
};
#[cfg(feature = "agentic")]
use crate::{ToggleWorkspaceSidebar, sidebar_side_context_menu};
#[cfg(feature = "agentic")]
use gpui::Anchor;
#[cfg(feature = "multiplayer-tools")]
use gpui::{Action, EntityId, canvas, px, relative};
use gpui::{
    AnyView, App, Context, Decorations, Entity, FocusHandle, Focusable, IntoElement, ParentElement,
    Render, Role, SharedString, Styled, Subscription, WeakEntity, Window,
};
use project::Project;
use settings::{SettingsContent, update_settings_file};
use std::{any::TypeId, sync::Arc};
use theme::CLIENT_SIDE_DECORATION_ROUNDING;
#[cfg(feature = "agentic")]
use ui::Divider;
use ui::{ContextMenu, IconPosition, prelude::*, right_click_menu};
#[cfg(feature = "agentic")]
use ui::{Indicator, Tooltip};
#[cfg(feature = "multiplayer-tools")]
use util::ResultExt;

/// Describes how a status-bar item can be hidden by the user.
///
/// Every [`StatusItemView`] must either provide this (so that the user gets a
/// "Hide Button" entry in the right-click menu) or explicitly return `None`
/// to opt out. Returning `None` should be reserved for items that are
/// already conditional on some other setting exposed elsewhere (e.g., the
/// activity indicator, which disappears on its own once there's no work to
/// display).
#[derive(Clone)]
pub struct HideStatusItem {
    hide: Arc<dyn Fn(&mut SettingsContent) + Send + Sync>,
}

impl HideStatusItem {
    pub fn new(hide: impl Fn(&mut SettingsContent) + Send + Sync + 'static) -> Self {
        Self {
            hide: Arc::new(hide),
        }
    }

    /// Persists the hide by updating the user settings file.
    pub fn apply(&self, cx: &App) {
        let hide = self.hide.clone();
        let fs = <dyn fs::Fs>::global(cx);
        update_settings_file(fs, cx, move |settings, _cx| (hide)(settings));
    }
}

pub trait StatusItemView: Render {
    #[cfg(feature = "multiplayer-tools")]
    fn is_collaborative_primary(&self) -> bool {
        false
    }

    /// Event callback that is triggered when the active pane item changes.
    fn set_active_pane_item(
        &mut self,
        active_pane_item: Option<&dyn crate::ItemHandle>,
        window: &mut Window,
        cx: &mut Context<Self>,
    );

    /// Returns metadata describing how this item can be hidden from the
    /// status bar by writing to the user settings file.
    ///
    /// Implementors that return `None` must be inherently conditional on
    /// another user-exposed setting; otherwise, they should return `Some` so
    /// that the status bar can show a "Hide Button" entry in its
    /// right-click menu.
    fn hide_setting(&self, cx: &App) -> Option<HideStatusItem>;
}

trait StatusItemViewHandle: Send {
    #[cfg(feature = "multiplayer-tools")]
    fn is_collaborative_primary(&self, cx: &App) -> bool;
    fn to_any(&self) -> AnyView;
    fn set_active_pane_item(
        &self,
        active_pane_item: Option<&dyn ItemHandle>,
        window: &mut Window,
        cx: &mut App,
    );
    fn item_type(&self) -> TypeId;
    fn hide_setting(&self, cx: &App) -> Option<HideStatusItem>;
}

#[derive(Default)]
struct SidebarStatus {
    open: bool,
    side: SidebarSide,
    #[cfg(feature = "agentic")]
    has_notifications: bool,
    #[cfg(feature = "agentic")]
    show_toggle: bool,
}

impl SidebarStatus {
    fn query(multi_workspace: &Option<WeakEntity<MultiWorkspace>>, cx: &App) -> Self {
        multi_workspace
            .as_ref()
            .and_then(|mw| mw.upgrade())
            .map(|mw| {
                let mw = mw.read(cx);
                let enabled = mw.multi_workspace_enabled(cx);
                Self {
                    open: mw.sidebar_open() && enabled,
                    side: mw.sidebar_side(cx),
                    #[cfg(feature = "agentic")]
                    has_notifications: mw.sidebar_has_notifications(cx),
                    #[cfg(feature = "agentic")]
                    show_toggle: enabled,
                }
            })
            .unwrap_or_default()
    }
}

pub struct StatusBar {
    left_items: Vec<Box<dyn StatusItemViewHandle>>,
    right_items: Vec<Box<dyn StatusItemViewHandle>>,
    active_pane: Entity<Pane>,
    multi_workspace: Option<WeakEntity<MultiWorkspace>>,
    #[cfg(feature = "multiplayer-tools")]
    collaborative_participants: Option<CollaborativeParticipantProvider>,
    #[cfg(feature = "multiplayer-tools")]
    collaborative_workspace_active: bool,
    #[cfg(feature = "multiplayer-tools")]
    collaborative_composer_footer: Option<AnyView>,
    #[cfg(feature = "multiplayer-tools")]
    collaborative_add_project_button: Option<AnyView>,
    #[cfg(feature = "multiplayer-tools")]
    collaborative_file_details: Vec<EntityId>,
    #[cfg(feature = "multiplayer-tools")]
    panel_strip_width: Pixels,
    #[cfg(feature = "multiplayer-tools")]
    panel_overflow_focus: FocusHandle,
    #[cfg(feature = "multiplayer-tools")]
    panel_overflow: ui::PopoverMenuHandle<ContextMenu>,
    focus_handle: FocusHandle,
    _observe_active_pane: Subscription,
    _observe_git_store: Subscription,
}

impl Focusable for StatusBar {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for StatusBar {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let sidebar = SidebarStatus::query(&self.multi_workspace, cx);

        h_flex()
            .id("status-bar")
            .debug_selector(|| "WORKSPACE-STATUS-BAR".to_owned())
            .track_focus(&self.focus_handle)
            .key_context("StatusBar")
            // Expose the status bar as an ARIA toolbar so assistive technology
            // announces it as a toolbar and region navigation can reach its
            // controls. The controls inside form a tab group: region navigation
            // lands on the first control (per the ARIA toolbar pattern), Tab
            // steps through them, and arrow keys move between them once focus is
            // inside.
            .role(Role::Toolbar)
            .aria_label("Status bar")
            .tab_group()
            .on_key_down(
                cx.listener(|status_bar, event: &gpui::KeyDownEvent, window, cx| {
                    if event.keystroke.modifiers.modified() {
                        return;
                    }
                    match event.keystroke.key.as_str() {
                        "right" => {
                            status_bar.move_item_focus(true, window, cx);
                            cx.stop_propagation();
                        }
                        "left" => {
                            status_bar.move_item_focus(false, window, cx);
                            cx.stop_propagation();
                        }
                        _ => {}
                    }
                }),
            )
            .w_full()
            .justify_between()
            .gap(DynamicSpacing::Base08.rems(cx))
            .p(DynamicSpacing::Base04.rems(cx))
            .bg(cx.theme().colors().status_bar_background)
            .map(|el| match window.window_decorations() {
                Decorations::Server => el,
                Decorations::Client { tiling, .. } => el
                    .when(
                        !(tiling.bottom || tiling.right)
                            && !(sidebar.open && sidebar.side == SidebarSide::Right),
                        |el| el.rounded_br(CLIENT_SIDE_DECORATION_ROUNDING),
                    )
                    .when(
                        !(tiling.bottom || tiling.left)
                            && !(sidebar.open && sidebar.side == SidebarSide::Left),
                        |el| el.rounded_bl(CLIENT_SIDE_DECORATION_ROUNDING),
                    )
                    // This border is to avoid a transparent gap in the rounded corners
                    .mb(px(-1.))
                    .mt({
                        #[cfg(target_os = "linux")]
                        let needs_gap_fix = {
                            // Running on Wayland and using some scaling levels other than 100% causes a
                            // 1px gap above the status bar; adding a margin avoids this.
                            gpui::guess_compositor() == "Wayland" && window.scale_factor() != 1.0
                        };
                        #[cfg(not(target_os = "linux"))]
                        let needs_gap_fix = false;
                        if needs_gap_fix { px(-1.) } else { px(0.) }
                    })
                    .border_b(px(1.0))
                    .border_color(cx.theme().colors().status_bar_background),
            })
            .map(|this| {
                #[cfg(feature = "multiplayer-tools")]
                if self.collaborative_workspace_active {
                    return this.child(self.render_collaborative_status(window, cx));
                }
                this.child(self.render_left_tools(&sidebar, cx))
                    .child(self.render_right_tools(&sidebar, cx))
            })
    }
}

impl StatusBar {
    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn render_conversation_footer(&self, cx: &App) -> impl IntoElement + use<> {
        let primary = self
            .left_items
            .iter()
            .chain(self.right_items.iter().rev())
            .filter(|item| item.is_collaborative_primary(cx));
        let has_project = self
            .multi_workspace
            .as_ref()
            .and_then(WeakEntity::upgrade)
            .is_some_and(|multi_workspace| {
                let workspace = multi_workspace.read(cx).workspace().read(cx);
                workspace
                    .project()
                    .read(cx)
                    .visible_worktrees(cx)
                    .next()
                    .is_some()
            });
        let state = self
            .collaborative_participants
            .as_ref()
            .map(|provider| provider.state(cx))
            .unwrap_or(CollaborativeParticipantProviderState::Unavailable);
        let primary_controls = h_flex()
            .id("collaborative-project-controls")
            .debug_selector(|| "COLLABORATIVE-PROJECT-CONTROLS".to_owned())
            .min_w_0()
            .gap_0p5()
            .child(
                h_flex()
                    .id("collaborative-project-picker-scroll")
                    .min_w_0()
                    .gap_0p5()
                    .overflow_x_scroll()
                    .children(primary.into_iter().enumerate().map(|(index, item)| {
                        render_hideable_item("collaborative-footer", index, item.as_ref(), cx)
                    })),
            )
            .children(self.collaborative_add_project_button.clone());
        let (project_row, inline_project) = if has_project {
            (Some(primary_controls), None)
        } else {
            (None, Some(primary_controls))
        };
        v_flex()
            .id("collaborative-footer-controls")
            .debug_selector(|| "COLLABORATIVE-FOOTER-CONTROLS".to_owned())
            .bg(cx.theme().colors().status_bar_background)
            .p(DynamicSpacing::Base04.rems(cx))
            .w_full()
            .min_w_0()
            .children(project_row)
            .child(
                h_flex()
                    .id("collaborative-composer-controls")
                    .debug_selector(|| "COLLABORATIVE-COMPOSER-CONTROLS".to_owned())
                    .w_full()
                    .min_w_0()
                    .gap_0p5()
                    .children(inline_project)
                    .child(CollaborativeParticipantStatus { state })
                    .children(self.collaborative_composer_footer.clone()),
            )
    }

    #[cfg(feature = "multiplayer-tools")]
    fn panel_buttons(&self, window: &mut Window, cx: &App) -> Vec<DockPanelButton> {
        self.left_items
            .iter()
            .chain(self.right_items.iter().rev())
            .filter_map(|item| item.to_any().downcast::<PanelButtons>().ok())
            .flat_map(|buttons| buttons.read(cx).buttons(true, window, cx))
            .collect()
    }

    #[cfg(feature = "multiplayer-tools")]
    fn render_collaborative_status(
        &self,
        window: &mut Window,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let has_active_file = self
            .active_pane
            .read(cx)
            .active_item()
            .is_some_and(|item| item.project_path(cx).is_some())
            && self
                .multi_workspace
                .as_ref()
                .and_then(WeakEntity::upgrade)
                .is_some_and(|multi_workspace| {
                    multi_workspace
                        .read(cx)
                        .workspace()
                        .read(cx)
                        .collaborative_status_file_visible(window, cx)
                });
        let tools = h_flex()
            .id("collaborative-status-tools")
            .min_w_0()
            .max_w(relative(0.6))
            .gap_1()
            .overflow_x_scroll()
            .children(
                self.left_items
                    .iter()
                    .chain(self.right_items.iter().rev())
                    .enumerate()
                    .filter(|(_, item)| {
                        item.item_type() != TypeId::of::<PanelButtons>()
                            && !item.is_collaborative_primary(cx)
                            && (has_active_file
                                || !self
                                    .collaborative_file_details
                                    .contains(&item.to_any().entity_id()))
                    })
                    .map(|(index, item)| {
                        render_hideable_item("collaborative-status-tools", index, item.as_ref(), cx)
                    }),
            );
        let panel_buttons = self.panel_buttons(window, cx);
        let visible = visible_panel_count(
            self.panel_strip_width,
            panel_buttons.len(),
            window.rem_size(),
        );
        let has_overflow = visible < panel_buttons.len();
        let status_bar = cx.entity().downgrade();
        let observed_status_bar = status_bar.clone();
        h_flex()
            .debug_selector(|| "COLLABORATIVE-STATUS-BAR".to_owned())
            .min_h(ButtonSize::Default.rems())
            .w_full()
            .gap_1()
            .child(tools)
            .child(
                h_flex()
                    .relative()
                    .flex_1()
                    .min_w_0()
                    .gap_1()
                    .justify_end()
                    .child(
                        canvas(
                            move |bounds, window, cx| {
                                let observed_status_bar = observed_status_bar.clone();
                                if observed_status_bar
                                    .read_with(cx, |status_bar, _| {
                                        status_bar.panel_strip_width == bounds.size.width
                                    })
                                    .unwrap_or(true)
                                {
                                    return;
                                }
                                window.on_next_frame(move |window, cx| {
                                    observed_status_bar
                                        .update(cx, |status_bar, cx| {
                                            let count = status_bar.panel_buttons(window, cx).len();
                                            if visible_panel_count(
                                                status_bar.panel_strip_width,
                                                count,
                                                window.rem_size(),
                                            ) != visible_panel_count(
                                                bounds.size.width,
                                                count,
                                                window.rem_size(),
                                            ) {
                                                status_bar.panel_overflow.hide(cx);
                                            }
                                            status_bar.panel_strip_width = bounds.size.width;
                                            cx.notify();
                                        })
                                        .log_err();
                                });
                            },
                            |_, _, _, _| {},
                        )
                        .absolute()
                        .size_full(),
                    )
                    .children(
                        panel_buttons
                            .into_iter()
                            .take(visible)
                            .map(|button| button.render(true, window, cx)),
                    )
                    .when(has_overflow, |strip| {
                        strip.child(
                            div()
                                .debug_selector(|| "COLLABORATIVE-PANEL-OVERFLOW".to_owned())
                                .child(
                                    ui::PopoverMenu::new("collaborative-panel-overflow")
                                        .anchor(gpui::Anchor::BottomRight)
                                        .with_handle(self.panel_overflow.clone())
                                        .restore_focus_to(&self.panel_overflow_focus)
                                        .menu(move |window, cx| {
                                            let mut buttons = status_bar
                                                .read_with(cx, |status_bar, cx| {
                                                    status_bar.panel_buttons(window, cx)
                                                })
                                                .ok()?;
                                            if visible >= buttons.len() {
                                                return None;
                                            }
                                            let entries = buttons
                                                .drain(visible..)
                                                .map(|button| button.overflow_entry(window, cx))
                                                .collect::<Vec<_>>();
                                            Some(ContextMenu::build(window, cx, |menu, _, _| {
                                                entries.into_iter().fold(menu, ContextMenu::item)
                                            }))
                                        })
                                        .trigger(
                                            IconButton::new("more-panels", IconName::Ellipsis)
                                                .shape(ui::IconButtonShape::Square)
                                                .icon_size(IconSize::Small)
                                                .tab_index(0isize)
                                                .track_focus(&self.panel_overflow_focus)
                                                .aria_label("More panels")
                                                .tooltip(ui::Tooltip::text("More panels")),
                                        ),
                                ),
                        )
                    }),
            )
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn set_collaborative_composer_footer(
        &mut self,
        footer: Option<AnyView>,
        cx: &mut Context<Self>,
    ) {
        if self
            .collaborative_composer_footer
            .as_ref()
            .map(AnyView::entity_id)
            != footer.as_ref().map(AnyView::entity_id)
        {
            self.collaborative_composer_footer = footer;
            cx.notify();
        }
    }

    #[cfg_attr(not(feature = "agentic"), allow(unused_variables))]
    fn render_left_tools(
        &self,
        sidebar: &SidebarStatus,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let tools = h_flex().gap_1().min_w_0().overflow_x_hidden();
        #[cfg(feature = "multiplayer-tools")]
        let tools = tools.when(!self.collaborative_workspace_active, |this| {
            this.child(
                div()
                    .debug_selector(|| "OPEN-COLLABORATIVE-WORKSPACE".to_owned())
                    .child(
                        Button::new("open-collaborative-workspace", "Multiplayer Workspace")
                            .style(ButtonStyle::Subtle)
                            .on_click(|_, window, cx| {
                                window.dispatch_action(
                                    SwitchToCollaborativeWorkspace.boxed_clone(),
                                    cx,
                                );
                            }),
                    ),
            )
        });
        #[cfg(feature = "agentic")]
        let tools = tools.when(
            sidebar.show_toggle && !sidebar.open && sidebar.side == SidebarSide::Left,
            |this| this.child(self.render_sidebar_toggle(sidebar, cx)),
        );
        tools.children(
            self.left_items.iter().enumerate().map(|(index, item)| {
                render_hideable_item("status-bar-left", index, item.as_ref(), cx)
            }),
        )
    }

    #[cfg_attr(not(feature = "agentic"), allow(unused_variables))]
    fn render_right_tools(
        &self,
        sidebar: &SidebarStatus,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let tools = h_flex()
            .flex_shrink_0()
            .gap_1()
            .overflow_x_hidden()
            .children(
                self.right_items
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(index, item)| {
                        render_hideable_item("status-bar-right", index, item.as_ref(), cx)
                    }),
            );
        #[cfg(feature = "agentic")]
        let tools = tools.when(
            sidebar.show_toggle && !sidebar.open && sidebar.side == SidebarSide::Right,
            |this| this.child(self.render_sidebar_toggle(sidebar, cx)),
        );
        tools
    }

    #[cfg(feature = "agentic")]
    fn render_sidebar_toggle(
        &self,
        sidebar: &SidebarStatus,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let on_right = sidebar.side == SidebarSide::Right;
        let has_notifications = sidebar.has_notifications;
        let indicator_border = cx.theme().colors().status_bar_background;

        let toggle = sidebar_side_context_menu("sidebar-status-toggle-menu", cx)
            .anchor(if on_right {
                Anchor::BottomRight
            } else {
                Anchor::BottomLeft
            })
            .attach(if on_right {
                Anchor::TopRight
            } else {
                Anchor::TopLeft
            })
            .trigger(move |_is_active, _window, _cx| {
                IconButton::new(
                    "toggle-workspace-sidebar",
                    if on_right {
                        IconName::ThreadsSidebarRightClosed
                    } else {
                        IconName::ThreadsSidebarLeftClosed
                    },
                )
                .icon_size(IconSize::Small)
                .tab_index(0isize)
                .aria_label("Open threads sidebar")
                .when(has_notifications, |this| {
                    this.indicator(Indicator::dot().color(Color::Accent))
                        .indicator_border_color(Some(indicator_border))
                })
                .tooltip(move |_, cx| {
                    Tooltip::for_action("Open Threads Sidebar", &ToggleWorkspaceSidebar, cx)
                })
                .on_click(move |_, window, cx| {
                    if let Some(multi_workspace) = window.root::<MultiWorkspace>().flatten() {
                        multi_workspace.update(cx, |multi_workspace, cx| {
                            multi_workspace.toggle_sidebar(window, cx);
                        });
                    }
                })
            });

        h_flex()
            .gap_0p5()
            .when(on_right, |this| {
                this.child(Divider::vertical().color(ui::DividerColor::Border))
            })
            .child(toggle)
            .when(!on_right, |this| {
                this.child(Divider::vertical().color(ui::DividerColor::Border))
            })
    }
}

fn render_hideable_item(
    side: &'static str,
    index: usize,
    item: &dyn StatusItemViewHandle,
    cx: &App,
) -> impl IntoElement {
    render_hideable_view(side, index, item.to_any(), item.hide_setting(cx))
}

fn render_hideable_view(
    side: &'static str,
    index: usize,
    view: AnyView,
    hide: Option<HideStatusItem>,
) -> gpui::AnyElement {
    let Some(hide) = hide else {
        return view.into_any_element();
    };

    let menu_id: SharedString = format!("{side}-item-menu-{index}").into();
    right_click_menu(menu_id)
        .trigger(move |_is_active, _window, _cx| view)
        .menu(move |window, cx| {
            let hide = hide.clone();
            ContextMenu::build(window, cx, move |menu, _window, _cx| {
                add_hide_button_entry(menu, hide)
            })
        })
        .into_any_element()
}

/// Appends a "Hide Button" entry aligned with surrounding toggleable entries.
pub fn add_hide_button_entry(menu: ContextMenu, hide: HideStatusItem) -> ContextMenu {
    menu.toggleable_entry(
        "Hide Button",
        false,
        IconPosition::Start,
        None,
        move |_window, cx| hide.apply(cx),
    )
}

#[cfg(feature = "multiplayer-tools")]
fn visible_panel_count(width: Pixels, count: usize, rem_size: Pixels) -> usize {
    let gap = rems(0.25) * rem_size;
    let slot = ButtonSize::Default.rems() * rem_size + gap;
    let capacity = (f32::from(width + gap) / f32::from(slot)).floor() as usize;
    if count <= capacity {
        count
    } else {
        capacity.saturating_sub(1)
    }
}

impl StatusBar {
    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn collaborative_focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    pub fn new(
        active_pane: &Entity<Pane>,
        project: Entity<Project>,
        multi_workspace: Option<WeakEntity<MultiWorkspace>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let git_store = project.read(cx).git_store().clone();
        let mut this = Self {
            left_items: Default::default(),
            right_items: Default::default(),
            active_pane: active_pane.clone(),
            multi_workspace,
            #[cfg(feature = "multiplayer-tools")]
            collaborative_participants: None,
            #[cfg(feature = "multiplayer-tools")]
            collaborative_workspace_active: false,
            #[cfg(feature = "multiplayer-tools")]
            collaborative_composer_footer: None,
            #[cfg(feature = "multiplayer-tools")]
            collaborative_add_project_button: None,
            #[cfg(feature = "multiplayer-tools")]
            collaborative_file_details: Vec::new(),
            #[cfg(feature = "multiplayer-tools")]
            panel_strip_width: px(0.),
            #[cfg(feature = "multiplayer-tools")]
            panel_overflow_focus: cx.focus_handle(),
            #[cfg(feature = "multiplayer-tools")]
            panel_overflow: Default::default(),
            focus_handle: cx.focus_handle(),
            _observe_active_pane: cx.observe_in(active_pane, window, |this, _, window, cx| {
                this.update_active_pane_item(window, cx)
            }),
            _observe_git_store: cx.observe(&git_store, |_, _, cx| cx.notify()),
        };
        this.update_active_pane_item(window, cx);
        this
    }

    pub fn set_multi_workspace(
        &mut self,
        multi_workspace: WeakEntity<MultiWorkspace>,
        cx: &mut Context<Self>,
    ) {
        self.multi_workspace = Some(multi_workspace);
        cx.notify();
    }

    #[cfg(feature = "multiplayer-tools")]
    pub fn set_collaborative_add_project_button(
        &mut self,
        button: AnyView,
        cx: &mut Context<Self>,
    ) {
        self.collaborative_add_project_button = Some(button);
        cx.notify();
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn set_collaborative_participants(
        &mut self,
        collaborative_workspace_active: bool,
        provider: Option<CollaborativeParticipantProvider>,
        cx: &mut Context<Self>,
    ) {
        self.collaborative_workspace_active = collaborative_workspace_active;
        self.collaborative_participants = provider;
        cx.notify();
    }

    #[cfg(feature = "multiplayer-tools")]
    pub fn set_collaborative_file_details(&mut self, items: Vec<EntityId>, cx: &mut Context<Self>) {
        self.collaborative_file_details = items;
        cx.notify();
    }

    pub fn add_left_item<T>(&mut self, item: Entity<T>, window: &mut Window, cx: &mut Context<Self>)
    where
        T: 'static + StatusItemView,
    {
        let active_pane_item = self.active_pane.read(cx).active_item();
        item.set_active_pane_item(active_pane_item.as_deref(), window, cx);

        #[cfg(feature = "multiplayer-tools")]
        if TypeId::of::<T>() == TypeId::of::<PanelButtons>() {
            cx.observe(&item, |_, _, cx| cx.notify()).detach();
        }
        self.left_items.push(Box::new(item));
        cx.notify();
    }

    pub fn item_of_type<T: StatusItemView>(&self) -> Option<Entity<T>> {
        self.left_items
            .iter()
            .chain(self.right_items.iter())
            .find_map(|item| item.to_any().downcast().ok())
    }

    pub fn position_of_item<T>(&self) -> Option<usize>
    where
        T: StatusItemView,
    {
        for (index, item) in self.left_items.iter().enumerate() {
            if item.item_type() == TypeId::of::<T>() {
                return Some(index);
            }
        }
        for (index, item) in self.right_items.iter().enumerate() {
            if item.item_type() == TypeId::of::<T>() {
                return Some(index + self.left_items.len());
            }
        }
        None
    }

    pub fn insert_item_after<T>(
        &mut self,
        position: usize,
        item: Entity<T>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) where
        T: 'static + StatusItemView,
    {
        let active_pane_item = self.active_pane.read(cx).active_item();
        item.set_active_pane_item(active_pane_item.as_deref(), window, cx);

        if position < self.left_items.len() {
            self.left_items.insert(position + 1, Box::new(item))
        } else {
            self.right_items
                .insert(position + 1 - self.left_items.len(), Box::new(item))
        }
        cx.notify()
    }

    pub fn remove_item_at(&mut self, position: usize, cx: &mut Context<Self>) {
        if position < self.left_items.len() {
            self.left_items.remove(position);
        } else {
            self.right_items.remove(position - self.left_items.len());
        }
        cx.notify();
    }

    pub fn add_right_item<T>(
        &mut self,
        item: Entity<T>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) where
        T: 'static + StatusItemView,
    {
        let active_pane_item = self.active_pane.read(cx).active_item();
        item.set_active_pane_item(active_pane_item.as_deref(), window, cx);

        #[cfg(feature = "multiplayer-tools")]
        if TypeId::of::<T>() == TypeId::of::<PanelButtons>() {
            cx.observe(&item, |_, _, cx| cx.notify()).detach();
        }
        self.right_items.push(Box::new(item));
        cx.notify();
    }

    pub fn set_active_pane(
        &mut self,
        active_pane: &Entity<Pane>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.active_pane = active_pane.clone();
        self._observe_active_pane = cx.observe_in(active_pane, window, |this, _, window, cx| {
            this.update_active_pane_item(window, cx)
        });
        self.update_active_pane_item(window, cx);
    }

    fn update_active_pane_item(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let active_pane_item = self.active_pane.read(cx).active_item();
        for item in self.left_items.iter().chain(&self.right_items) {
            item.set_active_pane_item(active_pane_item.as_deref(), window, cx);
        }
    }

    /// Moves focus between the interactive controls within the status bar in
    /// response to arrow keys. Navigation is clamped to the status bar so
    /// arrows move between items and stop at the ends (ARIA toolbar semantics);
    /// Tab is still used to leave the toolbar.
    fn move_item_focus(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        let previous = window.focused(cx);
        if forward {
            window.focus_next(cx);
        } else {
            window.focus_prev(cx);
        }
        let landed_in_status_bar = window
            .focused(cx)
            .is_some_and(|handle| self.focus_handle.contains(&handle, window));
        if !landed_in_status_bar && let Some(previous) = previous {
            window.focus(&previous, cx);
        }
        cx.notify();
    }
}

#[cfg(feature = "multiplayer-tools")]
#[derive(IntoElement)]
struct CollaborativeParticipantStatus {
    state: CollaborativeParticipantProviderState,
}

#[cfg(feature = "multiplayer-tools")]
impl RenderOnce for CollaborativeParticipantStatus {
    fn render(self, _: &mut Window, _: &mut App) -> impl IntoElement {
        let (label, description, color) = match self.state {
            CollaborativeParticipantProviderState::Ready(data) => {
                let Some(execution) = data.execution else {
                    return gpui::Empty.into_any_element();
                };
                let phase = match execution.phase {
                    CollaborativeExecutionPhase::Idle => None,
                    CollaborativeExecutionPhase::Running => Some("Running"),
                    CollaborativeExecutionPhase::WaitingForUser => Some("Waiting"),
                    CollaborativeExecutionPhase::Failed => Some("Failed"),
                    CollaborativeExecutionPhase::Completed => Some("Completed"),
                    CollaborativeExecutionPhase::Unknown => None,
                };
                let location = execution.location_label();
                let description = phase.map_or_else(
                    || location.to_string(),
                    |phase| format!("{phase} · {location}"),
                );
                let color = if execution.phase == CollaborativeExecutionPhase::Failed {
                    Color::Error
                } else {
                    Color::Muted
                };
                (
                    phase.map(SharedString::from).unwrap_or(location),
                    description.into(),
                    color,
                )
            }
            CollaborativeParticipantProviderState::Failed(message) => {
                ("Failed".into(), message, Color::Error)
            }
            CollaborativeParticipantProviderState::Unavailable => {
                return gpui::Empty.into_any_element();
            }
        };
        h_flex()
            .id("collaborative-participant-summary")
            .debug_selector(|| "COLLABORATIVE-PARTICIPANT-STATUS".to_owned())
            .gap_0p5()
            .min_w(IconSize::XSmall.rems())
            .flex_shrink_1()
            .overflow_hidden()
            .aria_label(description.clone())
            .tooltip(ui::Tooltip::text(description))
            .child(
                div()
                    .flex_none()
                    .debug_selector(|| "COLLABORATIVE-PARTICIPANT-STATUS-ICON".to_owned())
                    .child(
                        Icon::new(IconName::ZedAgent)
                            .size(IconSize::XSmall)
                            .color(color),
                    ),
            )
            .child(
                Label::new(label)
                    .size(LabelSize::XSmall)
                    .color(color)
                    .truncate(),
            )
            .into_any_element()
    }
}

impl<T: StatusItemView> StatusItemViewHandle for Entity<T> {
    #[cfg(feature = "multiplayer-tools")]
    fn is_collaborative_primary(&self, cx: &App) -> bool {
        self.read(cx).is_collaborative_primary()
    }
    fn to_any(&self) -> AnyView {
        self.clone().into()
    }

    fn set_active_pane_item(
        &self,
        active_pane_item: Option<&dyn ItemHandle>,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.update(cx, |this, cx| {
            this.set_active_pane_item(active_pane_item, window, cx)
        });
    }

    fn item_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn hide_setting(&self, cx: &App) -> Option<HideStatusItem> {
        self.read(cx).hide_setting(cx)
    }
}

impl From<&dyn StatusItemViewHandle> for AnyView {
    fn from(val: &dyn StatusItemViewHandle) -> Self {
        val.to_any()
    }
}

#[cfg(all(test, feature = "multiplayer-tools"))]
mod tests {
    use super::*;

    #[test]
    fn collaborative_status_bar_reserves_overflow_only_when_panels_do_not_fit() {
        let rem = px(16.);
        let button = ButtonSize::Default.rems() * rem;
        let gap = rems(0.25) * rem;
        let three_buttons = button * 3. + gap * 2.;
        assert_eq!(visible_panel_count(three_buttons, 3, rem), 3);
        assert_eq!(visible_panel_count(three_buttons, 4, rem), 2);
        assert_eq!(visible_panel_count(three_buttons - px(1.), 3, rem), 1);
        assert_eq!(visible_panel_count(button, 3, rem), 0);
        assert_eq!(visible_panel_count(px(0.), 0, rem), 0);
    }
}
