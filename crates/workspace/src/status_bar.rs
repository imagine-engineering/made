use crate::{ItemHandle, MultiWorkspace, Pane, SidebarSide};
#[cfg(feature = "multiplayer-tools")]
use crate::{
    SwitchToCollaborativeWorkspace,
    collaborative_participants::{
        CollaborativeExecutionPhase, CollaborativeParticipantProvider,
        CollaborativeParticipantProviderState,
    },
};
#[cfg(feature = "agentic")]
use crate::{ToggleWorkspaceSidebar, sidebar_side_context_menu};
#[cfg(feature = "multiplayer-tools")]
use gpui::Action;
#[cfg(feature = "agentic")]
use gpui::Anchor;
use gpui::{
    AnyView, App, Context, Decorations, Entity, FocusHandle, Focusable, IntoElement, ParentElement,
    Render, Role, SharedString, Styled, Subscription, WeakEntity, Window,
};
use project::Project;
use settings::{SettingsContent, update_settings_file};
use std::{any::TypeId, sync::Arc};
use theme::CLIENT_SIDE_DECORATION_ROUNDING;
#[cfg(any(feature = "agentic", feature = "multiplayer-tools"))]
use ui::Divider;
use ui::{ContextMenu, IconPosition, prelude::*, right_click_menu};
#[cfg(feature = "agentic")]
use ui::{Indicator, Tooltip};

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

    #[cfg(feature = "multiplayer-tools")]
    fn collaborative_menu_group(&self) -> CollaborativeStatusItemGroup {
        CollaborativeStatusItemGroup::Tools
    }

    #[cfg(feature = "multiplayer-tools")]
    fn render_collaborative_menu(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        self.render(window, cx).into_any_element()
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
    #[cfg(feature = "multiplayer-tools")]
    fn to_collaborative_menu_view(&self, cx: &mut App) -> (CollaborativeStatusItemGroup, AnyView);
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

#[cfg(feature = "multiplayer-tools")]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollaborativeStatusItemGroup {
    Tools,
    Panels,
}

#[cfg(feature = "multiplayer-tools")]
struct CollaborativeStatusItem<T: StatusItemView> {
    item: Entity<T>,
    _observation: Subscription,
}

#[cfg(feature = "multiplayer-tools")]
impl<T: StatusItemView> Render for CollaborativeStatusItem<T> {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.item
            .update(cx, |item, cx| item.render_collaborative_menu(window, cx))
    }
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
                    return this.child(self.render_collaborative_tools(cx));
                }
                this.child(self.render_left_tools(&sidebar, cx))
                    .child(self.render_right_tools(&sidebar, cx))
            })
    }
}

impl StatusBar {
    #[cfg(feature = "multiplayer-tools")]
    fn render_collaborative_tools(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let items = self.left_items.iter().chain(self.right_items.iter().rev());
        let primary = items.filter(|item| item.is_collaborative_primary(cx));
        let status_bar = cx.entity().downgrade();
        let state = self
            .collaborative_participants
            .as_ref()
            .map(|provider| provider.state(cx))
            .unwrap_or(CollaborativeParticipantProviderState::Unavailable);
        let primary_controls = h_flex()
            .id("collaborative-project-controls")
            .debug_selector(|| "COLLABORATIVE-PROJECT-CONTROLS".to_owned())
            .min_w_0()
            .flex_shrink_0()
            .gap_0p5()
            .overflow_x_scroll()
            .children(primary.into_iter().enumerate().map(|(index, item)| {
                render_hideable_item("collaborative-footer", index, item.as_ref(), cx)
            }))
            .child(
                div()
                    .debug_selector(|| "COLLABORATIVE-STATUS-OVERFLOW".to_owned())
                    .child(
                        ui::PopoverMenu::new("collaborative-status-overflow")
                            .anchor(gpui::Anchor::BottomLeft)
                            .trigger(
                                IconButton::new(
                                    "collaborative-status-overflow-trigger",
                                    IconName::Ellipsis,
                                )
                                .icon_size(IconSize::Small)
                                .size(ButtonSize::Compact)
                                .aria_label("Workspace status and agent execution")
                                .tooltip(ui::Tooltip::text("Workspace status and agent execution")),
                            )
                            .menu(move |window, cx| {
                                let status_bar = status_bar.upgrade()?;
                                Some(cx.new(|cx| {
                                    let observation =
                                        cx.observe(&status_bar, |_, _, cx| cx.notify());
                                    let focus_handle = cx.focus_handle();
                                    let focus_out =
                                        cx.on_focus_out(&focus_handle, window, |_, _, _, cx| {
                                            cx.emit(gpui::DismissEvent)
                                        });
                                    CollaborativeStatusMenu {
                                        items: Default::default(),
                                        status_bar: status_bar.downgrade(),
                                        focus_handle,
                                        _observation: observation,
                                        _focus_out: focus_out,
                                    }
                                }))
                            }),
                    ),
            );
        h_flex()
            .id("collaborative-footer-controls")
            .debug_selector(|| "COLLABORATIVE-FOOTER-CONTROLS".to_owned())
            .w_full()
            .min_w_0()
            .gap_0p5()
            .child(primary_controls)
            .child(CollaborativeParticipantStatus::new(state).compact())
            .children(self.collaborative_composer_footer.clone())
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn collaborative_composer_footer(&self) -> Option<AnyView> {
        self.collaborative_composer_footer.clone()
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
        let tools = tools.when(self.collaborative_workspace_active, |this| {
            this.flex_shrink_0().max_w_full().flex_wrap()
        });
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
        #[cfg(feature = "multiplayer-tools")]
        let tools = tools.when(self.collaborative_workspace_active, |this| {
            let this = this.max_w_full().flex_wrap();
            let state = self
                .collaborative_participants
                .as_ref()
                .map(|provider| provider.state(cx))
                .unwrap_or(CollaborativeParticipantProviderState::Unavailable);
            this.child(CollaborativeParticipantStatus::new(state))
        });
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

#[cfg(feature = "multiplayer-tools")]
struct CollaborativeStatusMenu {
    items: std::collections::HashMap<gpui::EntityId, (CollaborativeStatusItemGroup, AnyView)>,
    status_bar: WeakEntity<StatusBar>,
    focus_handle: FocusHandle,
    _observation: Subscription,
    _focus_out: Subscription,
}

#[cfg(feature = "multiplayer-tools")]
impl gpui::EventEmitter<gpui::DismissEvent> for CollaborativeStatusMenu {}

#[cfg(feature = "multiplayer-tools")]
impl Focusable for CollaborativeStatusMenu {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[cfg(feature = "multiplayer-tools")]
impl CollaborativeStatusMenu {
    fn move_focus(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        let previous = window.focused(cx);
        if forward {
            window.focus_next(cx);
        } else {
            window.focus_prev(cx);
        }
        if !window
            .focused(cx)
            .is_some_and(|focus| self.focus_handle.contains(&focus, window))
            && let Some(previous) = previous
        {
            window.focus(&previous, cx);
        }
        cx.notify();
    }
}

#[cfg(feature = "multiplayer-tools")]
impl Render for CollaborativeStatusMenu {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(status_bar) = self.status_bar.upgrade() else {
            return gpui::Empty.into_any_element();
        };
        let (items, state) = status_bar.update(cx, |status_bar, cx| {
            let mut items = Vec::new();
            for item in status_bar
                .left_items
                .iter()
                .chain(status_bar.right_items.iter().rev())
            {
                if item.is_collaborative_primary(cx) {
                    continue;
                }
                let hide = item.hide_setting(cx);
                let (group, view) = self
                    .items
                    .entry(item.to_any().entity_id())
                    .or_insert_with(|| item.to_collaborative_menu_view(cx));
                items.push((*group, view.clone(), hide));
            }
            let state = status_bar
                .collaborative_participants
                .as_ref()
                .map(|provider| provider.state(cx))
                .unwrap_or(CollaborativeParticipantProviderState::Unavailable);
            (items, state)
        });
        let mut tools = Vec::new();
        let mut panels = Vec::new();
        for (index, (group, view, hide)) in items.into_iter().enumerate() {
            let element = div().flex_none().max_w_full().child(render_hideable_view(
                "collaborative-status-overflow",
                index,
                view,
                hide,
            ));
            match group {
                CollaborativeStatusItemGroup::Tools => tools.push(element),
                CollaborativeStatusItemGroup::Panels => panels.push(element),
            }
        }
        v_flex()
            .id("collaborative-status-menu")
            .debug_selector(|| "COLLABORATIVE-STATUS-MENU".to_owned())
            .track_focus(&self.focus_handle)
            .tab_group()
            .role(Role::Toolbar)
            .aria_label("Workspace status and agent execution")
            .on_action(cx.listener(|_, _: &menu::Cancel, _, cx| cx.emit(gpui::DismissEvent)))
            .on_action(cx.listener(|this, _: &menu::SelectNext, window, cx| {
                this.move_focus(true, window, cx);
            }))
            .on_action(cx.listener(|this, _: &menu::SelectPrevious, window, cx| {
                this.move_focus(false, window, cx);
            }))
            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                if event.keystroke.key == "escape" && !event.keystroke.modifiers.modified() {
                    cx.emit(gpui::DismissEvent);
                    cx.stop_propagation();
                    return;
                }
                let mut modifiers = event.keystroke.modifiers;
                modifiers.shift = false;
                if modifiers.modified() {
                    return;
                }
                let forward = match event.keystroke.key.as_str() {
                    "tab" => !event.keystroke.modifiers.shift,
                    "right" | "down" => true,
                    "left" | "up" => false,
                    _ => return,
                };
                this.move_focus(forward, window, cx);
                cx.stop_propagation();
            }))
            .elevation_2(cx)
            .p_2()
            .gap_2()
            .w(px(320.))
            .max_w(_window.viewport_size().width - px(16.))
            .child(
                v_flex()
                    .debug_selector(|| "COLLABORATIVE-STATUS-TOOLS".to_owned())
                    .gap_1()
                    .child(
                        Label::new("Workspace tools")
                            .size(LabelSize::XSmall)
                            .color(Color::Muted),
                    )
                    .child(h_flex().flex_wrap().items_center().gap_1().children(tools)),
            )
            .child(Divider::horizontal())
            .child(
                v_flex()
                    .debug_selector(|| "COLLABORATIVE-STATUS-PANELS".to_owned())
                    .gap_1()
                    .child(
                        Label::new("Panels")
                            .size(LabelSize::XSmall)
                            .color(Color::Muted),
                    )
                    .child(h_flex().flex_wrap().items_center().gap_1().children(panels)),
            )
            .child(Divider::horizontal())
            .child(
                v_flex()
                    .debug_selector(|| "COLLABORATIVE-STATUS-EXECUTION".to_owned())
                    .gap_1()
                    .child(
                        Label::new("Agent execution")
                            .size(LabelSize::XSmall)
                            .color(Color::Muted),
                    )
                    .child(CollaborativeParticipantStatus::new(state)),
            )
            .into_any_element()
    }
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

    pub fn add_left_item<T>(&mut self, item: Entity<T>, window: &mut Window, cx: &mut Context<Self>)
    where
        T: 'static + StatusItemView,
    {
        let active_pane_item = self.active_pane.read(cx).active_item();
        item.set_active_pane_item(active_pane_item.as_deref(), window, cx);

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
    compact: bool,
}

#[cfg(feature = "multiplayer-tools")]
impl CollaborativeParticipantStatus {
    fn new(state: CollaborativeParticipantProviderState) -> Self {
        Self {
            state,
            compact: false,
        }
    }

    fn compact(mut self) -> Self {
        self.compact = true;
        self
    }
}

#[cfg(feature = "multiplayer-tools")]
impl RenderOnce for CollaborativeParticipantStatus {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        if self.compact {
            let (label, detail, color) = match self.state {
                CollaborativeParticipantProviderState::Ready(data) => match data.execution {
                    Some(execution) => {
                        let color = if execution.phase == CollaborativeExecutionPhase::Failed {
                            Color::Error
                        } else {
                            Color::Muted
                        };
                        (
                            Some(execution.location_label()),
                            format!(
                                "{:?} · {} · {} · {}",
                                execution.phase,
                                execution.model_label(),
                                execution.runtime_label(),
                                execution.location_label()
                            ),
                            color,
                        )
                    }
                    None => (None, "Agent execution unavailable".to_owned(), Color::Muted),
                },
                CollaborativeParticipantProviderState::Failed(message) => {
                    (None, message.to_string(), Color::Error)
                }
                CollaborativeParticipantProviderState::Unavailable => {
                    (None, "Agent unavailable".to_owned(), Color::Muted)
                }
            };
            return h_flex()
                .id("collaborative-participant-summary")
                .debug_selector(|| "COLLABORATIVE-PARTICIPANT-STATUS".to_owned())
                .gap_0p5()
                .min_w(IconSize::XSmall.rems())
                .flex_shrink_1()
                .overflow_hidden()
                .aria_label(detail.clone())
                .tooltip(ui::Tooltip::text(detail))
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
                .when_some(label, |this, label| {
                    this.child(
                        Label::new(label)
                            .size(LabelSize::XSmall)
                            .color(color)
                            .truncate(),
                    )
                })
                .into_any_element();
        }
        h_flex()
            .id("collaborative-participant-status")
            .debug_selector(|| "COLLABORATIVE-EXECUTION-DETAILS".to_owned())
            .min_w_0()
            .max_w_full()
            .gap_1()
            .px_1()
            .child(Icon::new(IconName::ZedAgent).size(IconSize::XSmall))
            .child(h_flex().min_w_0().flex_wrap().gap_1().map(|this| {
                match self.state {
                    CollaborativeParticipantProviderState::Ready(view_data) => {
                        let execution_available = view_data.execution.is_some();
                        let execution = view_data.execution;
                        this.debug_selector(|| "COLLABORATIVE-PARTICIPANT-STATUS-READY".to_owned())
                            .when_some(execution, |this, execution| {
                                let phase = match execution.phase {
                                    CollaborativeExecutionPhase::Idle => "Idle",
                                    CollaborativeExecutionPhase::Running => "Running",
                                    CollaborativeExecutionPhase::WaitingForUser => "Waiting",
                                    CollaborativeExecutionPhase::Failed => "Failed",
                                    CollaborativeExecutionPhase::Completed => "Completed",
                                    CollaborativeExecutionPhase::Unknown => "Unknown",
                                };
                                this.child(
                                    Label::new(phase)
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                                .child(Divider::vertical())
                                .child(
                                    Label::new(execution.model_label())
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                                .child(
                                    Label::new(execution.runtime_label())
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                                .child(
                                    Label::new(execution.location_label())
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                            })
                            .when(!execution_available, |this| {
                                this.child(
                                    Label::new("No active execution")
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                            })
                    }
                    CollaborativeParticipantProviderState::Failed(message) => this
                        .debug_selector(|| "COLLABORATIVE-PARTICIPANT-STATUS-FAILED".to_owned())
                        .child(
                            Label::new(message)
                                .size(LabelSize::XSmall)
                                .color(Color::Error),
                        ),
                    CollaborativeParticipantProviderState::Unavailable => this
                        .debug_selector(|| {
                            "COLLABORATIVE-PARTICIPANT-STATUS-UNAVAILABLE".to_owned()
                        })
                        .child(
                            Label::new("Agent unavailable")
                                .size(LabelSize::XSmall)
                                .color(Color::Muted),
                        ),
                }
            }))
            .into_any_element()
    }
}

impl<T: StatusItemView> StatusItemViewHandle for Entity<T> {
    #[cfg(feature = "multiplayer-tools")]
    fn is_collaborative_primary(&self, cx: &App) -> bool {
        self.read(cx).is_collaborative_primary()
    }
    #[cfg(feature = "multiplayer-tools")]
    fn to_collaborative_menu_view(&self, cx: &mut App) -> (CollaborativeStatusItemGroup, AnyView) {
        let group = self.read(cx).collaborative_menu_group();
        let item = self.clone();
        let view = cx.new(|cx| CollaborativeStatusItem {
            _observation: cx.observe(&item, |_, _, cx| cx.notify()),
            item,
        });
        (group, view.into())
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
