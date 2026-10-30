use gpui::{
    Action, Context, Entity, Focusable, InteractiveElement, KeyDownEvent, Render, Role,
    ScrollHandle, SharedString, WeakEntity,
};
use ui::{Avatar, Divider, Window, prelude::*};
use workspace::MultiWorkspace;

use crate::collaborative_pinned::CollaborativePinned;
use crate::collaborative_projects::CollaborativeProjects;
use crate::collaborative_tasks::CollaborativeTasks;

const SECTION_LABELS: [&str; 3] = ["Pinned", "Projects", "Threads"];
#[cfg(test)]
const EMPTY_LABELS: [&str; 3] = [
    "No pinned work",
    "No communities or projects",
    "No tasks or threads",
];

pub(crate) struct CollaborativeRail {
    multi_workspace: WeakEntity<MultiWorkspace>,
    pinned: Entity<CollaborativePinned>,
    projects: Entity<CollaborativeProjects>,
    tasks: Entity<CollaborativeTasks>,
    scroll_handle: ScrollHandle,
}

impl CollaborativeRail {
    pub(crate) fn new(multi_workspace: WeakEntity<MultiWorkspace>, cx: &mut Context<Self>) -> Self {
        if let Some(channel_store) = channel::ChannelStore::try_global(cx) {
            cx.observe(&channel_store, |_, _, cx| cx.notify()).detach();
        }
        Self {
            multi_workspace: multi_workspace.clone(),
            pinned: cx.new(|cx| CollaborativePinned::new(multi_workspace.clone(), cx)),
            projects: cx.new(|cx| CollaborativeProjects::new(multi_workspace.clone(), cx)),
            tasks: cx.new(|cx| CollaborativeTasks::new(multi_workspace, cx)),
            scroll_handle: ScrollHandle::new(),
        }
    }

    fn render_account_surface(&self, cx: &gpui::App) -> impl IntoElement {
        let user = self.multi_workspace.upgrade().and_then(|multi_workspace| {
            let workspace = multi_workspace.read(cx).workspace().clone();
            workspace
                .read(cx)
                .app_state()
                .user_store
                .read(cx)
                .current_user()
        });
        let account_label = user
            .as_ref()
            .and_then(|user| user.name.as_deref())
            .filter(|name| !name.trim().is_empty())
            .map(SharedString::from)
            .or_else(|| user.as_ref().map(|user| user.username.clone()))
            .unwrap_or_else(|| "Sign in".into());
        let product_label = release_channel::ReleaseChannel::try_global(cx)
            .map(|channel| channel.display_name())
            .unwrap_or("Zed");

        h_flex()
            .id("collaborative-account-surface")
            .debug_selector(|| "COLLABORATIVE-ACCOUNT-SURFACE".to_owned())
            .flex_none()
            .px_2()
            .py(DynamicSpacing::Base04.rems(cx))
            .bg(cx.theme().colors().panel_background)
            .gap_2()
            .border_t_1()
            .border_color(cx.theme().colors().border)
            .tab_index(0)
            .role(Role::Button)
            .aria_label(format!("Account: {account_label}, {product_label}"))
            .child(match user {
                Some(user) => Avatar::new(user.avatar_uri.clone())
                    .size(rems(1.5))
                    .into_any_element(),
                None => h_flex()
                    .size_6()
                    .justify_center()
                    .child(
                        Icon::new(IconName::Person)
                            .size(IconSize::Small)
                            .color(Color::Muted),
                    )
                    .into_any_element(),
            })
            .child(Label::new(account_label).size(LabelSize::Small).truncate())
            .child(Divider::vertical())
            .child(
                Label::new(product_label)
                    .size(LabelSize::Small)
                    .color(Color::Muted)
                    .truncate(),
            )
            .child(div().flex_1())
            .child(
                Icon::new(IconName::ChevronDown)
                    .size(IconSize::XSmall)
                    .color(Color::Muted),
            )
            .hover(|style| style.bg(cx.theme().colors().ghost_element_hover))
            .active(|style| style.bg(cx.theme().colors().ghost_element_active))
            .focus_visible(|style| style.border_color(cx.theme().colors().border_focused))
            .on_key_down(|event: &KeyDownEvent, window, cx| {
                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                    cx.stop_propagation();
                    window.dispatch_action(zed_actions::OpenAccountSettings.boxed_clone(), cx);
                }
            })
            .on_click(|_, window, cx| {
                window.dispatch_action(zed_actions::OpenAccountSettings.boxed_clone(), cx);
            })
    }

    fn render_section(
        &self,
        label: &'static str,
        selector: &'static str,
        contents: impl IntoElement,
        cx: &gpui::App,
    ) -> impl IntoElement {
        v_flex()
            .debug_selector(move || selector.to_owned())
            .flex_none()
            .min_w_0()
            .child(
                h_flex()
                    .min_h(rems_from_px(ui::fractal::CONTROL_SMALL))
                    .px(DynamicSpacing::Base08.rems(cx))
                    .child(Label::new(label).size(LabelSize::Small).color(Color::Muted))
                    .child(div().flex_1())
                    .when(label == "Threads", |this| {
                        let workspace = self
                            .multi_workspace
                            .upgrade()
                            .map(|multi_workspace| multi_workspace.read(cx).workspace().clone());
                        let available = workspace.as_ref().is_some_and(|workspace| {
                            workspace
                                .read(cx)
                                .panel::<agent_ui::AgentPanel>(cx)
                                .is_some()
                        });
                        this.child(
                            crate::collaborative_navigation::navigation_action(
                                "collaborative-new-thread",
                                IconName::Plus,
                                "New thread",
                            )
                            .disabled(!available)
                            .on_click(move |_, window, cx| {
                                if let Some(workspace) = &workspace {
                                    workspace.focus_handle(cx).dispatch_action(
                                        &agent_ui::NewThread,
                                        window,
                                        cx,
                                    );
                                }
                            }),
                        )
                    }),
            )
            .child(
                div()
                    .min_w_0()
                    .px(DynamicSpacing::Base08.rems(cx))
                    .child(contents),
            )
    }

    #[cfg(test)]
    pub(crate) fn test_scroll_handle(&self) -> ScrollHandle {
        self.scroll_handle.clone()
    }

    #[cfg(test)]
    pub(crate) fn test_labels(&self) -> ([&'static str; 3], [&'static str; 3]) {
        (SECTION_LABELS, EMPTY_LABELS)
    }
}

impl Render for CollaborativeRail {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let channels = self
            .projects
            .update(cx, |projects, cx| projects.render_channels(cx));
        v_flex()
            .id("collaborative-rail")
            .debug_selector(|| "COLLABORATIVE-RAIL".to_owned())
            .size_full()
            .role(Role::Navigation)
            .aria_label(workspace::collaborative_accessibility::NAVIGATION_LABEL)
            .child(
                v_flex()
                    .id("collaborative-rail-scroll")
                    .debug_selector(|| "COLLABORATIVE-RAIL-SCROLL".to_owned())
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll_handle)
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .pb(DynamicSpacing::Base08.rems(cx))
                    .child(self.render_section(
                        SECTION_LABELS[0],
                        "COLLABORATIVE-RAIL-PINNED",
                        self.pinned.clone(),
                        cx,
                    ))
                    .child(self.render_section(
                        SECTION_LABELS[1],
                        "COLLABORATIVE-RAIL-PROJECTS",
                        self.projects.clone(),
                        cx,
                    ))
                    .child(self.render_section(
                        SECTION_LABELS[2],
                        "COLLABORATIVE-RAIL-TASKS",
                        self.tasks.clone(),
                        cx,
                    ))
                    .child(self.render_section(
                        "Channels",
                        "COLLABORATIVE-RAIL-CHANNELS",
                        channels,
                        cx,
                    )),
            )
            .child(self.render_account_surface(cx))
    }
}
