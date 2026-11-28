pub(crate) mod breakpoint_list;
pub(crate) mod console;
pub(crate) mod loaded_source_list;
pub(crate) mod memory_view;
pub(crate) mod module_list;
pub mod stack_frame_list;
pub mod variable_list;
use std::{
    any::Any,
    path::PathBuf,
    sync::{Arc, LazyLock},
    time::Duration,
};

use crate::{
    ToggleExpandItem,
    attach_modal::{AttachModal, ModalIntent},
    new_process_modal::resolve_path,
    persistence::{self, DebuggerPaneItem, SerializedLayout},
    session::running::memory_view::MemoryView,
};

use anyhow::{Context as _, Result, anyhow, bail};
use breakpoint_list::BreakpointList;
use collections::{HashMap, IndexMap};
use console::Console;
use dap::{
    Capabilities, DapRegistry, RunInTerminalRequestArguments, Thread,
    adapters::{DebugAdapterName, DebugTaskDefinition},
    client::SessionId,
    debugger_settings::DebuggerSettings,
};
use futures::{SinkExt, channel::mpsc};
use gpui::{
    Action as _, AnyView, AppContext, Axis, Entity, EntityId, EventEmitter, FocusHandle, Focusable,
    NoAction, Pixels, Point, Subscription, Task, TaskExt, WeakEntity,
};
use language::Buffer;
use loaded_source_list::LoadedSourceList;
use module_list::ModuleList;
#[cfg(feature = "jvm-tools")]
use project::task_store::{
    JvmTaskTrigger, PreparedSavedJvmTask, SavedJvmTask, prepare_jvm_hook_tasks,
};
use project::{
    DebugScenarioContext, LocalJavaTestDebugTask, Project, RemoteJavaTestDebugTask, WorktreeId,
    debugger::session::{self, Session, SessionEvent, SessionStateEvent, ThreadId, ThreadStatus},
};
use rpc::proto::ViewId;
use serde_json::Value;
use settings::Settings;
use stack_frame_list::StackFrameList;
use task::{
    BuildTaskDefinition, DebugScenario, SharedTaskContext, Shell, ShellBuilder, SpawnInTerminal,
    StructuredTaskHandle, TaskContext, ZedDebugConfig, substitute_variables_in_str,
};
use terminal_view::TerminalView;
use ui::{
    FluentBuilder, IntoElement, Render, StatefulInteractiveElement, Tab, Tooltip, VisibleOnHover,
    VisualContext, prelude::*,
};
use util::ResultExt;
use variable_list::VariableList;
use workspace::{
    ActivePaneDecorator, DraggedTab, Item, ItemHandle, Member, Pane, PaneGroup, SplitDirection,
    Workspace, item::TabContentParams, move_item, pane::Event,
};
#[cfg(feature = "jvm-tools")]
use workspace::{Toast, notifications::NotificationId, tasks::ScheduledTaskResult};

static PROCESS_ID_PLACEHOLDER: LazyLock<String> =
    LazyLock::new(|| task::VariableName::PickProcessId.template_value());

#[cfg(feature = "jvm-tools")]
struct SelectedJavaDebugHooks {
    store: WeakEntity<project::LspStore>,
    worktree_id: WorktreeId,
    project_id: EntityId,
    after: Vec<(SavedJvmTask, PreparedSavedJvmTask)>,
    active_hook: Option<StructuredTaskHandle>,
    terminal: Option<Entity<terminal::Terminal>>,
}

enum SelectedJavaTestBuild {
    Local {
        worktree_id: WorktreeId,
        node_id: String,
        generation: u64,
        port: u16,
        plan: LocalJavaTestDebugTask,
    },
    Remote(RemoteJavaTestDebugTask),
}

async fn wait_for_java_test_debugger(
    terminal: &Entity<terminal::Terminal>,
    port: u16,
    use_terminal_output: bool,
    remote: Option<(&Entity<Project>, &RemoteJavaTestDebugTask)>,
    cx: &mut gpui::AsyncWindowContext,
) -> Result<()> {
    let ready = format!("Listening for transport dt_socket at address: {port}");
    let deadline = cx.background_executor().timer(Duration::from_secs(60));
    let wait = async {
        loop {
            let completed = terminal.read_with(cx, |terminal, cx| {
                let task = terminal
                    .task()
                    .context("The selected Java test Debug terminal no longer has a Task")?;
                if task.structured_task.as_ref().is_some_and(|handle| {
                    matches!(handle.state(), task::StructuredTaskState::Cancelled { .. })
                }) {
                    bail!("The selected Java test Task was cancelled before the JVM suspended.");
                }
                Ok::<_, anyhow::Error>(
                    (task.status != terminal::TaskStatus::Running)
                        .then(|| terminal.wait_for_completed_task(cx)),
                )
            })?;
            if let Some(completion) = completed {
                match completion.await {
                    Some(status) if status.success() => bail!(
                        "The selected Java test Task finished successfully before the JVM suspended. Check the Debug terminal for the build output."
                    ),
                    Some(status) => bail!(
                        "The selected Java test Task failed before the JVM suspended ({status}). Check the Debug terminal for the build output."
                    ),
                    None => bail!(
                        "The selected Java test Task stopped before the JVM suspended. Check the Debug terminal for the build output."
                    ),
                }
            }
            let announced = !use_terminal_output
                || terminal.read_with(cx, |terminal, _| {
                    terminal
                        .last_n_non_empty_lines(32)
                        .iter()
                        .any(|line| line.trim() == ready)
                });
            let is_ready = if !announced {
                false
            } else if let Some((project, plan)) = remote {
                project
                    .update(cx, |project, cx| {
                        project.check_remote_java_test_debug_task(plan, true, cx)
                    })
                    .await?
            } else if use_terminal_output {
                true
            } else {
                // Surefire can buffer the fork's stdout while it is suspended.
                match std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)) {
                    Ok(_) => false,
                    Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => true,
                    Err(error) => {
                        return Err(error)
                            .context("Could not check the selected Maven test Debug port");
                    }
                }
            };
            if is_ready {
                return Ok(());
            }
            cx.background_executor()
                .timer(Duration::from_millis(100))
                .await;
        }
    };
    futures::pin_mut!(wait);
    match futures::future::select(wait, deadline).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(_) => {
            bail!("The selected Java test did not open its Debug port within 60 seconds")
        }
    }
}

pub struct RunningState {
    session: Entity<Session>,
    thread_id: Option<ThreadId>,
    focus_handle: FocusHandle,
    _remote_id: Option<ViewId>,
    workspace: WeakEntity<Workspace>,
    project: WeakEntity<Project>,
    session_id: SessionId,
    variable_list: Entity<variable_list::VariableList>,
    _subscriptions: Vec<Subscription>,
    stack_frame_list: Entity<stack_frame_list::StackFrameList>,
    loaded_sources_list: Entity<LoadedSourceList>,
    pub debug_terminal: Entity<DebugTerminal>,
    selected_java_test_task: Option<StructuredTaskHandle>,
    #[cfg(feature = "jvm-tools")]
    selected_java_debug_hooks: Option<SelectedJavaDebugHooks>,
    module_list: Entity<module_list::ModuleList>,
    console: Entity<Console>,
    breakpoint_list: Entity<BreakpointList>,
    panes: PaneGroup,
    active_pane: Entity<Pane>,
    pane_close_subscriptions: HashMap<EntityId, Subscription>,
    dock_axis: Axis,
    _schedule_serialize: Option<Task<()>>,
    pub(crate) scenario: Option<DebugScenario>,
    pub(crate) scenario_context: Option<DebugScenarioContext>,
    memory_view: Entity<MemoryView>,
}

impl RunningState {
    pub(crate) fn thread_id(&self) -> Option<ThreadId> {
        self.thread_id
    }

    pub(crate) fn active_pane(&self) -> &Entity<Pane> {
        &self.active_pane
    }
}

impl Render for RunningState {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let zoomed_pane = self
            .panes
            .panes()
            .into_iter()
            .find(|pane| pane.read(cx).is_zoomed());

        let active = self.panes.panes().into_iter().next();
        let pane = if let Some(zoomed_pane) = zoomed_pane {
            zoomed_pane.update(cx, |pane, cx| pane.render(window, cx).into_any_element())
        } else if let Some(active) = active {
            self.panes
                .render(
                    None,
                    None,
                    &ActivePaneDecorator::new(active, &self.workspace),
                    window,
                    cx,
                )
                .into_any_element()
        } else {
            div().into_any_element()
        };
        let thread_status = self
            .thread_id
            .map(|thread_id| self.session.read(cx).thread_status(thread_id))
            .unwrap_or(ThreadStatus::Exited);

        self.variable_list.update(cx, |this, cx| {
            this.disabled(thread_status != ThreadStatus::Stopped, cx);
        });
        v_flex()
            .size_full()
            .key_context("DebugSessionItem")
            .track_focus(&self.focus_handle(cx))
            .child(h_flex().flex_1().child(pane))
    }
}

pub(crate) struct SubView {
    inner: AnyView,
    item_focus_handle: FocusHandle,
    kind: DebuggerPaneItem,
    running_state: WeakEntity<RunningState>,
    host_pane: WeakEntity<Pane>,
    show_indicator: Box<dyn Fn(&App) -> bool>,
    actions: Option<Box<dyn FnMut(&mut Window, &mut App) -> AnyElement>>,
    hovered: bool,
}

impl SubView {
    pub(crate) fn new(
        item_focus_handle: FocusHandle,
        view: AnyView,
        kind: DebuggerPaneItem,
        running_state: WeakEntity<RunningState>,
        host_pane: WeakEntity<Pane>,
        cx: &mut App,
    ) -> Entity<Self> {
        cx.new(|_| Self {
            kind,
            inner: view,
            item_focus_handle,
            running_state,
            host_pane,
            show_indicator: Box::new(|_| false),
            actions: None,
            hovered: false,
        })
    }

    pub(crate) fn stack_frame_list(
        stack_frame_list: Entity<StackFrameList>,
        running_state: WeakEntity<RunningState>,
        host_pane: WeakEntity<Pane>,
        cx: &mut App,
    ) -> Entity<Self> {
        let weak_list = stack_frame_list.downgrade();
        let this = Self::new(
            stack_frame_list.focus_handle(cx),
            stack_frame_list.into(),
            DebuggerPaneItem::Frames,
            running_state,
            host_pane,
            cx,
        );

        this.update(cx, |this, _| {
            this.with_actions(Box::new(move |_, cx| {
                weak_list
                    .update(cx, |this, _| this.render_control_strip())
                    .unwrap_or_else(|_| div().into_any_element())
            }));
        });

        this
    }

    pub(crate) fn console(
        console: Entity<Console>,
        running_state: WeakEntity<RunningState>,
        host_pane: WeakEntity<Pane>,
        cx: &mut App,
    ) -> Entity<Self> {
        let weak_console = console.downgrade();
        let this = Self::new(
            console.focus_handle(cx),
            console.into(),
            DebuggerPaneItem::Console,
            running_state,
            host_pane,
            cx,
        );
        this.update(cx, |this, _| {
            this.with_indicator(Box::new(move |cx| {
                weak_console
                    .read_with(cx, |console, cx| console.show_indicator(cx))
                    .unwrap_or_default()
            }))
        });
        this
    }

    pub(crate) fn breakpoint_list(
        list: Entity<BreakpointList>,
        running_state: WeakEntity<RunningState>,
        host_pane: WeakEntity<Pane>,
        cx: &mut App,
    ) -> Entity<Self> {
        let weak_list = list.downgrade();
        let focus_handle = list.focus_handle(cx);
        let this = Self::new(
            focus_handle,
            list.into(),
            DebuggerPaneItem::BreakpointList,
            running_state,
            host_pane,
            cx,
        );

        this.update(cx, |this, _| {
            this.with_actions(Box::new(move |_, cx| {
                weak_list
                    .update(cx, |this, _| this.render_control_strip())
                    .unwrap_or_else(|_| div().into_any_element())
            }));
        });
        this
    }

    pub(crate) fn view_kind(&self) -> DebuggerPaneItem {
        self.kind
    }
    pub(crate) fn with_indicator(&mut self, indicator: Box<dyn Fn(&App) -> bool>) {
        self.show_indicator = indicator;
    }
    pub(crate) fn with_actions(
        &mut self,
        actions: Box<dyn FnMut(&mut Window, &mut App) -> AnyElement>,
    ) {
        self.actions = Some(actions);
    }

    fn set_host_pane(&mut self, host_pane: WeakEntity<Pane>) {
        self.host_pane = host_pane;
    }
}
impl Focusable for SubView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.item_focus_handle.clone()
    }
}
impl EventEmitter<()> for SubView {}
impl Item for SubView {
    type Event = ();

    /// This is used to serialize debugger pane layouts
    /// A SharedString gets converted to a enum and back during serialization/deserialization.
    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        self.kind.to_shared_string()
    }

    fn tab_tooltip_text(&self, _: &App) -> Option<SharedString> {
        Some(self.kind.tab_tooltip())
    }

    fn tab_content(
        &self,
        params: workspace::item::TabContentParams,
        _: &Window,
        cx: &App,
    ) -> AnyElement {
        let label = Label::new(self.kind.to_shared_string())
            .size(ui::LabelSize::Small)
            .color(params.text_color())
            .line_height_style(ui::LineHeightStyle::UiLabel);

        if !params.selected && self.show_indicator.as_ref()(cx) {
            return h_flex()
                .justify_between()
                .child(ui::Indicator::dot())
                .gap_2()
                .child(label)
                .into_any_element();
        }

        label.into_any_element()
    }

    fn handle_drop(
        &self,
        active_pane: &Pane,
        dropped: &dyn Any,
        window: &mut Window,
        cx: &mut App,
    ) -> bool {
        let Some(tab) = dropped.downcast_ref::<DraggedTab>() else {
            return true;
        };
        let Some(this_pane) = self.host_pane.upgrade() else {
            return true;
        };
        if tab.item.downcast::<SubView>().is_none() {
            return true;
        }
        let Some(split_direction) = active_pane.drag_split_direction() else {
            return false;
        };

        let source = tab.pane.clone();
        let item_id_to_move = tab.item.item_id();
        let weak_running = self.running_state.clone();

        // Source pane may be the one currently updated, so defer the move.
        window.defer(cx, move |window, cx| {
            let new_pane = weak_running.update(cx, |running, cx| {
                let Some(project) = running.project.upgrade() else {
                    return Err(anyhow!("Debugger project has been dropped"));
                };

                let new_pane = new_debugger_pane(running.workspace.clone(), project, window, cx);
                let _previous_subscription = running.pane_close_subscriptions.insert(
                    new_pane.entity_id(),
                    cx.subscribe_in(&new_pane, window, RunningState::handle_pane_event),
                );
                debug_assert!(_previous_subscription.is_none());
                running
                    .panes
                    .split(&this_pane, &new_pane, split_direction, cx);
                anyhow::Ok(new_pane)
            });

            match new_pane.and_then(|result| result) {
                Ok(new_pane) => {
                    move_item(
                        &source,
                        &new_pane,
                        item_id_to_move,
                        new_pane.read(cx).active_item_index(),
                        true,
                        window,
                        cx,
                    );
                }
                Err(err) => {
                    log::error!("{err:?}");
                }
            }
        });

        true
    }
}

impl Render for SubView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id(format!(
                "subview-container-{}",
                self.kind.to_shared_string()
            ))
            .size_full()
            .border_1()
            .when(self.item_focus_handle.contains_focused(window, cx), |el| {
                el.border_color(cx.theme().colors().pane_focused_border)
            })
            .child(self.inner.clone())
            .on_hover(cx.listener(|this, hovered, _, cx| {
                this.hovered = *hovered;
                cx.notify();
            }))
    }
}

struct DraggedTabPreview {
    label: SharedString,
}

impl Render for DraggedTabPreview {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let ui_font = theme_settings::ThemeSettings::get_global(cx)
            .ui_font
            .clone();
        let colors = cx.theme().colors();

        h_flex()
            .font(ui_font)
            .h_6()
            .px_1()
            .rounded_sm()
            .shadow_md()
            .border_1()
            .border_color(colors.border)
            .bg(colors.elevated_surface_background)
            .child(Label::new(self.label.clone()).size(LabelSize::Small))
    }
}

fn render_debugger_tab(
    ix: usize,
    item: &dyn ItemHandle,
    selected: bool,
    deemphasized: bool,
    window: &mut Window,
    cx: &mut Context<Pane>,
) -> impl IntoElement + use<> {
    let item_ = item.boxed_clone();
    let colors = cx.theme().colors();

    div()
        .border_l_2()
        .border_color(gpui::transparent_black())
        .drag_over::<DraggedTab>(|wrapper, _, _, cx| wrapper.border_color(cx.theme().colors().text))
        .child(
            div()
                .cursor_pointer()
                .id(format!("debugger_tab_{}", item.item_id().as_u64()))
                .p_1()
                .rounded_sm()
                .map(|s| {
                    if selected {
                        s.bg(colors.text_accent.opacity(0.08))
                            .hover(|s| s.bg(colors.text_accent.opacity(0.25)))
                            .text_color(colors.text_accent)
                    } else {
                        s.hover(|s| s.bg(colors.element_hover))
                    }
                })
                .when(deemphasized, |s| s.opacity(0.8))
                .child(item.tab_content(
                    TabContentParams {
                        selected,
                        deemphasized,
                        ..Default::default()
                    },
                    window,
                    cx,
                ))
                .when_some(item.tab_tooltip_text(cx), |this, tooltip| {
                    this.tooltip(Tooltip::text(tooltip))
                })
                .on_click(cx.listener(move |this, _, window, cx| {
                    let index = this.index_for_item(&*item_);
                    if let Some(index) = index {
                        this.activate_item(index, true, true, window, cx);
                    }
                }))
                .on_drop(
                    cx.listener(move |this, dragged_tab: &DraggedTab, window, cx| {
                        if dragged_tab.item.downcast::<SubView>().is_none() {
                            return;
                        }
                        this.drag_split_direction = None;
                        this.handle_tab_drop(dragged_tab, ix, false, window, cx)
                    }),
                )
                .on_drag(
                    DraggedTab {
                        item: item.boxed_clone(),
                        pane: cx.entity(),
                        detail: 0,
                        is_active: selected,
                        ix,
                    },
                    |tab, _, _, cx| {
                        let label = tab.item.tab_content_text(0, cx);
                        cx.new(|_| DraggedTabPreview { label })
                    },
                ),
        )
}

fn render_debugger_tab_bar(
    pane: &mut Pane,
    focus_handle: &FocusHandle,
    window: &mut Window,
    cx: &mut Context<Pane>,
) -> gpui::AnyElement {
    let active_pane_item = pane.active_item();
    let pane_group_id: SharedString = format!("pane-zoom-button-hover-{}", cx.entity_id()).into();
    let as_subview = active_pane_item
        .as_ref()
        .and_then(|item| item.downcast::<SubView>());

    let is_hovered = as_subview
        .as_ref()
        .is_some_and(|item| item.read(cx).hovered);
    let deemphasized = !pane.has_focus(window, cx);

    let tabs = pane
        .items()
        .enumerate()
        .map(|(ix, item)| {
            let selected = active_pane_item
                .as_ref()
                .is_some_and(|active| active.item_id() == item.item_id());
            render_debugger_tab(ix, item.as_ref(), selected, deemphasized, window, cx)
        })
        .collect::<Vec<_>>();

    h_flex()
        .track_focus(focus_handle)
        .group(pane_group_id.clone())
        .on_action(|_: &menu::Cancel, window, cx| {
            if cx.stop_active_drag(window) {
            } else {
                cx.propagate();
            }
        })
        .pl_1p5()
        .pr_1()
        .justify_between()
        .border_b_1()
        .border_color(cx.theme().colors().border)
        .bg(cx.theme().colors().tab_bar_background)
        .child(
            h_flex()
                .w_full()
                .gap_1()
                .h(Tab::container_height(cx))
                .children(tabs)
                .on_drop(
                    cx.listener(move |this, dragged_tab: &DraggedTab, window, cx| {
                        if dragged_tab.item.downcast::<SubView>().is_none() {
                            return;
                        }
                        this.drag_split_direction = None;
                        this.handle_tab_drop(dragged_tab, this.items_len(), false, window, cx)
                    }),
                )
                .child(
                    div()
                        .flex_1()
                        .h_6()
                        .border_l_2()
                        .border_color(gpui::transparent_black())
                        .drag_over::<DraggedTab>(|spacer, _, _, cx| {
                            spacer.border_color(cx.theme().colors().text)
                        }),
                ),
        )
        .child({
            let zoomed = pane.is_zoomed();

            h_flex()
                .visible_on_hover(pane_group_id)
                .when(is_hovered, |this| this.visible())
                .when_some(as_subview.as_ref(), |this, subview| {
                    subview.update(cx, |view, cx| {
                        let Some(additional_actions) = view.actions.as_mut() else {
                            return this;
                        };
                        this.child(additional_actions(window, cx))
                    })
                })
                .child(
                    IconButton::new(
                        format!("debug-toggle-zoom-{}", cx.entity_id()),
                        if zoomed {
                            IconName::Minimize
                        } else {
                            IconName::Maximize
                        },
                    )
                    .icon_size(IconSize::Small)
                    .on_click(cx.listener(move |pane, _, _, cx| {
                        let is_zoomed = pane.is_zoomed();
                        pane.set_zoomed(!is_zoomed, cx);
                        cx.notify();
                    }))
                    .tooltip({
                        let focus_handle = focus_handle.clone();
                        move |_window, cx| {
                            let zoomed_text = if zoomed { "Minimize" } else { "Expand" };
                            Tooltip::for_action_in(
                                zoomed_text,
                                &ToggleExpandItem,
                                &focus_handle,
                                cx,
                            )
                        }
                    }),
                )
        })
        .into_any_element()
}

pub(crate) fn new_debugger_pane(
    workspace: WeakEntity<Workspace>,
    project: Entity<Project>,
    window: &mut Window,
    cx: &mut Context<RunningState>,
) -> Entity<Pane> {
    let weak_running = cx.weak_entity();

    cx.new(move |cx| {
        let can_drop_predicate: Arc<dyn Fn(&dyn Any, &mut Window, &mut App) -> bool> =
            Arc::new(|any, _window, _cx| {
                any.downcast_ref::<DraggedTab>()
                    .is_some_and(|dragged_tab| dragged_tab.item.downcast::<SubView>().is_some())
            });
        let mut pane = Pane::new(
            workspace.clone(),
            project.clone(),
            Default::default(),
            Some(can_drop_predicate),
            NoAction.boxed_clone(),
            true,
            window,
            cx,
        );
        let focus_handle = pane.focus_handle(cx);
        pane.set_can_split(Some(Arc::new({
            let weak_running = weak_running.clone();
            move |pane, dragged_item, _window, cx| {
                if let Some(tab) = dragged_item.downcast_ref::<DraggedTab>() {
                    let is_current_pane = tab.pane == cx.entity();
                    let Some(can_drag_away) = weak_running
                        .read_with(cx, |running_state, _| {
                            let current_panes = running_state.panes.panes();
                            !current_panes.contains(&&tab.pane)
                                || current_panes.len() > 1
                                || (!is_current_pane || pane.items_len() > 1)
                        })
                        .ok()
                    else {
                        return false;
                    };
                    if can_drag_away {
                        let item = if is_current_pane {
                            pane.item_for_index(tab.ix)
                        } else {
                            tab.pane.read(cx).item_for_index(tab.ix)
                        };
                        if let Some(item) = item {
                            return item.downcast::<SubView>().is_some();
                        }
                    }
                }
                false
            }
        })));
        pane.set_can_toggle_zoom(false, cx);
        pane.display_nav_history_buttons(None);
        pane.set_should_display_tab_bar(|_, _| true);
        pane.set_render_tab_bar_buttons(cx, |_, _, _| (None, None));
        pane.set_render_tab_bar(cx, {
            move |pane, window, cx| render_debugger_tab_bar(pane, &focus_handle, window, cx)
        });
        pane
    })
}

pub struct DebugTerminal {
    pub terminal: Option<Entity<TerminalView>>,
    focus_handle: FocusHandle,
    _subscriptions: [Subscription; 1],
}

impl DebugTerminal {
    fn empty(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        let focus_subscription = cx.on_focus(&focus_handle, window, |this, window, cx| {
            if let Some(terminal) = this.terminal.as_ref() {
                terminal.focus_handle(cx).focus(window, cx);
            }
        });

        Self {
            terminal: None,
            focus_handle,
            _subscriptions: [focus_subscription],
        }
    }
}

impl gpui::Render for DebugTerminal {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .track_focus(&self.focus_handle)
            .size_full()
            .bg(cx.theme().colors().editor_background)
            .children(self.terminal.clone())
    }
}
impl Focusable for DebugTerminal {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl RunningState {
    // todo(debugger) move this to util and make it so you pass a closure to it that converts a string
    pub(crate) fn substitute_variables_in_config(
        config: &mut serde_json::Value,
        context: &TaskContext,
    ) {
        match config {
            serde_json::Value::Object(obj) => {
                obj.values_mut()
                    .for_each(|value| Self::substitute_variables_in_config(value, context));
            }
            serde_json::Value::Array(array) => {
                array
                    .iter_mut()
                    .for_each(|value| Self::substitute_variables_in_config(value, context));
            }
            serde_json::Value::String(s) => {
                // Some built-in zed tasks wrap their arguments in quotes as they might contain spaces.
                if s.starts_with("\"$ZED_") && s.ends_with('"') {
                    *s = s[1..s.len() - 1].to_string();
                }
                if let Some(substituted) = substitute_variables_in_str(s, context) {
                    *s = substituted;
                }
            }
            _ => {}
        }
    }

    pub(crate) fn contains_substring(config: &serde_json::Value, substring: &str) -> bool {
        match config {
            serde_json::Value::Object(obj) => obj
                .values()
                .any(|value| Self::contains_substring(value, substring)),
            serde_json::Value::Array(array) => array
                .iter()
                .any(|value| Self::contains_substring(value, substring)),
            serde_json::Value::String(s) => s.contains(substring),
            _ => false,
        }
    }

    pub(crate) fn substitute_process_id_in_config(config: &mut serde_json::Value, process_id: i32) {
        match config {
            serde_json::Value::Object(obj) => {
                obj.values_mut().for_each(|value| {
                    Self::substitute_process_id_in_config(value, process_id);
                });
            }
            serde_json::Value::Array(array) => {
                array.iter_mut().for_each(|value| {
                    Self::substitute_process_id_in_config(value, process_id);
                });
            }
            serde_json::Value::String(s) => {
                if s.contains(PROCESS_ID_PLACEHOLDER.as_str()) {
                    *s = s.replace(PROCESS_ID_PLACEHOLDER.as_str(), &process_id.to_string());
                }
            }
            _ => {}
        }
    }

    pub(crate) fn relativize_paths(
        key: Option<&str>,
        config: &mut serde_json::Value,
        context: &TaskContext,
    ) {
        match config {
            serde_json::Value::Object(obj) => {
                obj.iter_mut()
                    .for_each(|(key, value)| Self::relativize_paths(Some(key), value, context));
            }
            serde_json::Value::Array(array) => {
                array
                    .iter_mut()
                    .for_each(|value| Self::relativize_paths(None, value, context));
            }
            serde_json::Value::String(s) if key == Some("program") || key == Some("cwd") => {
                // Some built-in zed tasks wrap their arguments in quotes as they might contain spaces.
                if s.starts_with("\"$ZED_") && s.ends_with('"') {
                    *s = s[1..s.len() - 1].to_string();
                }
                resolve_path(s);

                if let Some(substituted) = substitute_variables_in_str(s, context) {
                    *s = substituted;
                }
            }
            _ => {}
        }
    }

    pub(crate) fn new(
        session: Entity<Session>,
        project: Entity<Project>,
        workspace: WeakEntity<Workspace>,
        parent_terminal: Option<Entity<DebugTerminal>>,
        serialized_pane_layout: Option<SerializedLayout>,
        dock_axis: Axis,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        let session_id = session.read(cx).session_id();
        let weak_project = project.downgrade();
        let weak_state = cx.weak_entity();
        let stack_frame_list = cx.new(|cx| {
            StackFrameList::new(
                workspace.clone(),
                session.clone(),
                weak_state.clone(),
                window,
                cx,
            )
        });

        let debug_terminal =
            parent_terminal.unwrap_or_else(|| cx.new(|cx| DebugTerminal::empty(window, cx)));
        let memory_view = cx.new(|cx| {
            MemoryView::new(
                session.clone(),
                workspace.clone(),
                stack_frame_list.downgrade(),
                window,
                cx,
            )
        });
        let variable_list = cx.new(|cx| {
            VariableList::new(
                session.clone(),
                stack_frame_list.clone(),
                memory_view.clone(),
                weak_state.clone(),
                window,
                cx,
            )
        });

        let module_list = cx.new(|cx| ModuleList::new(session.clone(), workspace.clone(), cx));

        let loaded_source_list = cx.new(|cx| LoadedSourceList::new(session.clone(), cx));

        let console = cx.new(|cx| {
            Console::new(
                session.clone(),
                stack_frame_list.clone(),
                variable_list.clone(),
                window,
                cx,
            )
        });

        let breakpoint_list = BreakpointList::new(
            Some(session.clone()),
            workspace.clone(),
            &project,
            window,
            cx,
        );

        let _subscriptions = vec![
            #[cfg(feature = "jvm-tools")]
            cx.on_release(|this, cx| this.cancel_selected_java_debug_hooks(cx)),
            cx.on_app_quit(move |this, cx| {
                #[cfg(feature = "jvm-tools")]
                this.cancel_selected_java_debug_hooks(cx);
                this.stop_selected_java_test(cx);
                let shutdown = this
                    .session
                    .update(cx, |session, cx| session.on_app_quit(cx));
                let terminal = this.debug_terminal.clone();
                async move {
                    shutdown.await;
                    drop(terminal)
                }
            }),
            cx.observe(&module_list, |_, _, cx| cx.notify()),
            cx.subscribe_in(&session, window, |this, _, event, window, cx| {
                match event {
                    SessionEvent::Stopped(thread_id) => {
                        if this.session.read(cx).diagnostic_observation().is_some() {
                            cx.notify();
                            return;
                        }
                        let panel = this
                            .workspace
                            .update(cx, |workspace, cx| {
                                workspace.open_panel::<crate::DebugPanel>(window, cx);
                                workspace.panel::<crate::DebugPanel>(cx)
                            })
                            .log_err()
                            .flatten();

                        if let Some(thread_id) = thread_id {
                            this.select_thread(*thread_id, window, cx);
                        }
                        if let Some(panel) = panel {
                            let id = this.session_id;
                            window.defer(cx, move |window, cx| {
                                panel.update(cx, |this, cx| {
                                    this.activate_session_by_id(id, window, cx);
                                })
                            })
                        }
                    }
                    SessionEvent::Threads => {
                        let threads = this.session.update(cx, |this, cx| this.threads(cx));
                        this.select_current_thread(&threads, window, cx);
                    }
                    SessionEvent::CapabilitiesLoaded => {
                        let capabilities = this.capabilities(cx);
                        if !capabilities.supports_modules_request.unwrap_or(false) {
                            this.remove_pane_item(DebuggerPaneItem::Modules, window, cx);
                        }
                        if !capabilities
                            .supports_loaded_sources_request
                            .unwrap_or(false)
                        {
                            this.remove_pane_item(DebuggerPaneItem::LoadedSources, window, cx);
                        }
                    }
                    SessionEvent::RunInTerminal { request, sender } => this
                        .handle_run_in_terminal(request, sender.clone(), window, cx)
                        .detach_and_log_err(cx),

                    SessionEvent::DebuggeeTerminated { natural: _natural } => {
                        this.selected_java_test_task = None;
                        #[cfg(feature = "jvm-tools")]
                        if *_natural {
                            this.finish_selected_java_debug_hooks(window, cx);
                        }
                    }

                    _ => {}
                }
                cx.notify()
            }),
            cx.on_focus_out(&focus_handle, window, |this, _, window, cx| {
                this.serialize_layout(window, cx);
            }),
            cx.subscribe(
                &session,
                |this, session, event: &SessionStateEvent, cx| match event {
                    SessionStateEvent::Shutdown => {
                        #[cfg(feature = "jvm-tools")]
                        this.cancel_selected_java_debug_hooks(cx);
                        if session.read(cx).is_building() {
                            this.shutdown(cx);
                        } else {
                            this.stop_selected_java_test(cx);
                        }
                    }
                    _ => {}
                },
            ),
        ];

        let mut pane_close_subscriptions = HashMap::default();
        let panes = if let Some(root) = serialized_pane_layout.and_then(|serialized_layout| {
            persistence::deserialize_pane_layout(
                serialized_layout.panes,
                dock_axis != serialized_layout.dock_axis,
                &workspace,
                &project,
                &stack_frame_list,
                &variable_list,
                &module_list,
                &console,
                &breakpoint_list,
                &loaded_source_list,
                &debug_terminal,
                &memory_view,
                &mut pane_close_subscriptions,
                window,
                cx,
            )
        }) {
            workspace::PaneGroup::with_root(root)
        } else {
            pane_close_subscriptions.clear();

            let root = Self::default_pane_layout(
                project,
                &workspace,
                &stack_frame_list,
                &variable_list,
                &console,
                &breakpoint_list,
                &debug_terminal,
                dock_axis,
                &mut pane_close_subscriptions,
                window,
                cx,
            );

            workspace::PaneGroup::with_root(root)
        };
        let active_pane = panes.first_pane();

        Self {
            memory_view,
            session,
            workspace,
            project: weak_project,
            focus_handle,
            variable_list,
            _subscriptions,
            thread_id: None,
            _remote_id: None,
            stack_frame_list,
            session_id,
            panes,
            active_pane,
            module_list,
            console,
            breakpoint_list,
            loaded_sources_list: loaded_source_list,
            pane_close_subscriptions,
            debug_terminal,
            selected_java_test_task: None,
            #[cfg(feature = "jvm-tools")]
            selected_java_debug_hooks: None,
            dock_axis,
            _schedule_serialize: None,
            scenario: None,
            scenario_context: None,
        }
    }

    pub(crate) fn remove_pane_item(
        &mut self,
        item_kind: DebuggerPaneItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some((pane, item_id)) = self.panes.panes().iter().find_map(|pane| {
            Some(pane).zip(
                pane.read(cx)
                    .items()
                    .find(|item| {
                        item.act_as::<SubView>(cx)
                            .is_some_and(|view| view.read(cx).kind == item_kind)
                    })
                    .map(|item| item.item_id()),
            )
        }) {
            pane.update(cx, |pane, cx| {
                pane.remove_item(item_id, false, true, window, cx)
            })
        }
    }

    pub(crate) fn has_pane_at_position(&self, position: Point<Pixels>) -> bool {
        self.panes.pane_at_pixel_position(position).is_some()
    }

    pub(crate) fn ensure_experimental_access(&self, cx: &App) -> Result<()> {
        #[cfg(feature = "jvm-tools")]
        if let Some(hooks) = &self.selected_java_debug_hooks {
            anyhow::ensure!(
                !self.session.read(cx).is_terminated()
                    && hooks.store.read_with(cx, |store, _| {
                        store.jvm_trigger_provider_is_current(
                            hooks.worktree_id,
                            JvmTaskTrigger::BeforeDebug,
                        )
                    })?,
                "The Java provider changed or Debug stopped during its saved Task sequence"
            );
        }
        self.session.read(cx).ensure_experimental_access(cx)
    }

    #[cfg(feature = "jvm-tools")]
    async fn run_before_selected_java_debug(
        running: &WeakEntity<Self>,
        project: &Entity<Project>,
        worktree_id: WorktreeId,
        cx: &mut gpui::AsyncWindowContext,
    ) -> Result<bool> {
        let activation = JvmTaskTrigger::BeforeDebug;
        let (before, has_hooks) = running.update(cx, |running, cx| {
            anyhow::ensure!(
                !running.session.read(cx).is_terminated(),
                "The selected Java Debug session stopped"
            );
            let project_id = project.entity_id();
            let project = project.read(cx);
            let store = project.lsp_store();
            anyhow::ensure!(
                !store.read(cx).jvm_trigger_is_armed(worktree_id, activation),
                "A selected Java Debug Task sequence is already active for this project"
            );
            let before = prepare_jvm_hook_tasks(project, worktree_id, activation, cx)?;
            let after =
                prepare_jvm_hook_tasks(project, worktree_id, JvmTaskTrigger::AfterDebug, cx)?;
            if !before.is_empty() || !after.is_empty() {
                store.update(cx, |store, cx| {
                    store.arm_jvm_trigger(worktree_id, activation, cx)
                })?;
                running.selected_java_debug_hooks = Some(SelectedJavaDebugHooks {
                    store: store.downgrade(),
                    worktree_id,
                    project_id,
                    after,
                    active_hook: None,
                    terminal: None,
                });
            }
            anyhow::Ok((before, running.selected_java_debug_hooks.is_some()))
        })??;
        for (target, prepared) in before {
            let label = target.label.clone();
            let handle = StructuredTaskHandle::new(prepared.task.id.clone());
            let outcome = running
                .update_in(cx, |running, window, cx| {
                    running.ensure_experimental_access(cx)?;
                    let hooks = running
                        .selected_java_debug_hooks
                        .as_mut()
                        .context("The selected Java Debug sequence stopped")?;
                    hooks.active_hook = Some(handle.clone());
                    running.workspace.update(cx, |workspace, cx| {
                        workspace.show_toast(
                            Toast::new(
                                NotificationId::composite::<JvmTaskTrigger>(usize::from(
                                    worktree_id,
                                )),
                                format!("Running before-debug Task '{label}'"),
                            ),
                            cx,
                        );
                        workspace.schedule_saved_jvm_task(
                            hooks.project_id,
                            (target, prepared),
                            Some(activation),
                            true,
                            Some(handle),
                            None,
                            window,
                            cx,
                        )
                    })
                })??
                .await?;
            running.update(cx, |running, _| {
                if let Some(hooks) = &mut running.selected_java_debug_hooks {
                    hooks.active_hook = None;
                }
            })?;
            anyhow::ensure!(
                outcome == ScheduledTaskResult::Success,
                "before-debug Task '{label}' {}. The selected Java Debug session was skipped.",
                match outcome {
                    ScheduledTaskResult::Cancelled => "was cancelled",
                    ScheduledTaskResult::Failure => "failed",
                    _ => "could not start",
                }
            );
        }
        Ok(has_hooks)
    }

    #[cfg(feature = "jvm-tools")]
    fn cancel_selected_java_debug_hooks(&mut self, cx: &mut App) {
        if let Some(hooks) = self.selected_java_debug_hooks.take() {
            if let Some(handle) = hooks.active_hook {
                handle.cancel(cx);
            }
            hooks
                .store
                .update(cx, |store, _| {
                    store.cancel_jvm_trigger(hooks.worktree_id, JvmTaskTrigger::BeforeDebug);
                })
                .log_err();
        }
    }

    #[cfg(feature = "jvm-tools")]
    fn finish_selected_java_debug_hooks(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(hooks) = self.selected_java_debug_hooks.take() else {
            return;
        };
        let workspace = self.workspace.clone();
        window
            .spawn(cx, async move |cx| {
                let activation = JvmTaskTrigger::BeforeDebug;
                let result = async {
                    let terminal = hooks
                        .terminal
                        .context("The selected Java Debug Task is unavailable")?;
                    // The Debug panel may remove the session before the build wrapper exits.
                    let exit = terminal
                        .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                        .await
                        .context("The selected Java Debug Task did not report its exit status")?;
                    drop(terminal);
                    anyhow::ensure!(
                        exit.success(),
                        "The selected Java Debug Task failed. After-debug Tasks were skipped."
                    );
                    Workspace::run_saved_jvm_hook_tasks(
                        &workspace,
                        hooks.after,
                        hooks.project_id,
                        activation,
                        JvmTaskTrigger::AfterDebug,
                        cx,
                    )
                    .await?;
                    hooks.store.update(cx, |store, cx| {
                    anyhow::ensure!(
                        store.complete_jvm_trigger(hooks.worktree_id, activation, cx),
                        "The Java provider or project trust changed during the Debug Task sequence"
                    );
                    Ok(())
                })?
                }
                .await;
                if let Err(error) = result {
                    hooks
                        .store
                        .update(cx, |store, _| {
                            store.cancel_jvm_trigger(hooks.worktree_id, activation);
                        })
                        .log_err();
                    workspace
                        .update(cx, |workspace, cx| {
                            workspace.show_toast(
                                Toast::new(
                                    NotificationId::composite::<JvmTaskTrigger>(usize::from(
                                        hooks.worktree_id,
                                    )),
                                    error.to_string(),
                                ),
                                cx,
                            );
                        })
                        .log_err();
                }
            })
            .detach();
    }

    pub(crate) fn resolve_scenario(
        &self,
        scenario: DebugScenario,
        task_context: SharedTaskContext,
        buffer: Option<Entity<Buffer>>,
        worktree_id: Option<WorktreeId>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<DebugTaskDefinition>> {
        let requirements = match &scenario.build {
            Some(BuildTaskDefinition::Template { task_template, .. }) => {
                task_template.required_experimental_features.as_slice()
            }
            _ => &[],
        };
        if let Err(error) = task::validate_experimental_feature_requirements(requirements) {
            return Task::ready(Err(error));
        }
        if let Err(error) = self.session.update(cx, |session, cx| {
            session.set_required_experimental_features(requirements.to_vec())?;
            session.ensure_experimental_access(cx)
        }) {
            return Task::ready(Err(error));
        }
        let Some(workspace) = self.workspace.upgrade() else {
            return Task::ready(Err(anyhow!("no workspace")));
        };
        let project = workspace.read(cx).project().clone();
        let dap_store = project.read(cx).dap_store().downgrade();
        let dap_registry = cx.global::<DapRegistry>().clone();
        let task_store = project.read(cx).task_store().downgrade();
        let weak_project = project.downgrade();
        let weak_workspace = workspace.downgrade();
        let is_windows = project.read(cx).path_style(cx).is_windows();
        let remote_shell = project
            .read(cx)
            .remote_client()
            .as_ref()
            .and_then(|remote| remote.read(cx).shell());

        cx.spawn_in(window, async move |this, cx| {
            let DebugScenario {
                adapter,
                label,
                build,
                mut config,
                tcp_connection,
            } = scenario;
            Self::relativize_paths(None, &mut config, &task_context);
            Self::substitute_variables_in_config(&mut config, &task_context);

            if Self::contains_substring(&config, PROCESS_ID_PLACEHOLDER.as_str()) || label.as_ref().contains(PROCESS_ID_PLACEHOLDER.as_str()) {
                let (tx, rx) = futures::channel::oneshot::channel::<Option<i32>>();

                let weak_workspace_clone = weak_workspace.clone();
                weak_workspace.update_in(cx, |workspace, window, cx| {
                    let project = workspace.project().clone();
                    workspace.toggle_modal(window, cx, |window, cx| {
                        AttachModal::new(
                            ModalIntent::ResolveProcessId(Some(tx)),
                            weak_workspace_clone,
                            project,
                            true,
                            window,
                            cx,
                        )
                    });
                }).ok();

                let Some(process_id) = rx.await.ok().flatten() else {
                    bail!("No process selected with config that contains {}", PROCESS_ID_PLACEHOLDER.as_str())
                };

                this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
                Self::substitute_process_id_in_config(&mut config, process_id);
            }

            let remote_selected_test = matches!(build, Some(BuildTaskDefinition::SelectedJavaTest { .. }))
                && project.read_with(cx, |project, _| project.is_remote());
            let request_type = if remote_selected_test {
                Ok(dap::StartDebuggingRequestArgumentsRequest::Attach)
            } else {
                match dap_registry.adapter(&adapter)
                    .with_context(|| format!("{adapter}: is not a valid adapter name")) {
                    Ok(adapter) => adapter.request_kind(&config).await,
                    Err(error) => Err(error),
                }
                };


            this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
            let config_is_valid = request_type.is_ok();
            let mut extra_config = Value::Null;
            let build_output = if let Some(build) = build {
                let mut selected_java_test = None;
                let (task_template, locator_name) = match build {
                    BuildTaskDefinition::Template {
                        task_template,
                        locator_name,
                    } => (task_template, locator_name),
                    BuildTaskDefinition::SelectedJavaTest { node_id, discovery_generation } => {
                        let worktree_id = worktree_id.context("Select a Java test worktree")?;
                        anyhow::ensure!(
                            config_is_valid && adapter.as_ref() == "Java"
                                && config["request"] == "attach"
                                && config["hostName"] == "127.0.0.1",
                            "The selected Java test Debug endpoint changed; reselect it in Tests"
                        );
                        let (port, source_root, template, build) = if remote_selected_test {
                            let plan = project.update(cx, |project, cx| {
                                project.prepare_remote_java_test_debug_task(worktree_id, node_id, discovery_generation, cx)
                            }).await?;
                            (plan.port, plan.source_root.clone(), plan.template.clone(), SelectedJavaTestBuild::Remote(plan))
                        } else {
                            let mut plan = project.update(cx, |project, cx| {
                                project.selected_java_test_debug_task(worktree_id, &node_id, discovery_generation, None, cx)
                            })?;
                            let port = match plan.fixed_port {
                                Some(port) => port,
                                None => {
                                    let port = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
                                        .context("Could not allocate a local Maven Debug port")?
                                        .local_addr()?.port();
                                    plan = project.update(cx, |project, cx| {
                                        project.selected_java_test_debug_task(worktree_id, &node_id, discovery_generation, Some(port), cx)
                                    })?;
                                    port
                                }
                            };
                            (port, plan.source_root.clone(), plan.template.clone(), SelectedJavaTestBuild::Local {
                                worktree_id, node_id, generation: discovery_generation, port, plan,
                            })
                        };
                        config["port"] = port.into();
                        config["sourcePaths"] = serde_json::json!([
                            source_root.to_str().context("The Java test source path is not Unicode")?
                        ]);
                        selected_java_test = Some(build);
                        (template, None)
                    }
                    BuildTaskDefinition::ByName(ref label) => {
                        let task = task_store.update(cx, |this, cx| {
                            this.task_inventory().map(|inventory| {
                                inventory.read(cx).task_template_by_label(
                                    buffer,
                                    worktree_id,
                                    label,
                                    cx,
                                )
                            })
                        })?;
                        let task = match task {
                            Some(task) => task.await,
                            None => None,
                        }.with_context(|| format!("Couldn't find task template for {build:?}"))?;
                        (task, None)
                    }
                };
                task::validate_experimental_feature_requirements(&task_template.required_experimental_features)?;
                this.update(cx, |this, cx| {
                    this.session.update(cx, |session, cx| {
                        session.set_required_experimental_features(task_template.required_experimental_features.clone())?;
                        session.ensure_experimental_access(cx)
                    })
                })??;
                let task = if remote_selected_test {
                    let mut template = task_template;
                    template.label = template.label.replace('$', "$$");
                    template.resolve_task_with_literal_arguments("debug-build-task", &task_context)
                } else {
                    task_template.resolve_task("debug-build-task", &task_context)
                };
                let Some(mut task) = task else {
                    anyhow::bail!("Could not resolve task variables within a debug scenario");
                };

                let locator_name = if let Some(locator_name) = locator_name {
                    extra_config = config.clone();
                    debug_assert!(!config_is_valid);
                    Some(locator_name)
                } else if !config_is_valid {
                    let task = dap_store
                        .update(cx, |this, cx| {
                            this.debug_scenario_for_build_task(
                                task.original_task().clone(),
                                adapter.clone().into(),
                                task.display_label().to_owned().into(),
                                cx,
                            )

                        });
                    if let Ok(t) = task {
                        t.await.and_then(|scenario| {
                            extra_config = scenario.config;
                            match scenario.build {
                                Some(BuildTaskDefinition::Template {
                                    locator_name, ..
                                }) => locator_name,
                                _ => None,
                            }
                        })
                    } else {
                        None
                    }

                } else {
                    None
                };

                this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
                if let Some(remote_shell) = remote_shell && task.resolved.shell == Shell::System {
                    task.resolved.shell = Shell::Program(remote_shell);
                }

                let builder = ShellBuilder::new(&task.resolved.shell, is_windows);
                let command_label = builder.command_label(task.resolved.command.as_deref().unwrap_or(""));
                let (command, args) =
                    builder.build(task.resolved.command.clone(), &task.resolved.args);

                let task_with_shell = if remote_selected_test {
                    task.resolved.clone()
                } else { SpawnInTerminal {
                    command_label,
                    command: Some(command),
                    args,
                    ..task.resolved.clone()
                }};

                Workspace::save_for_task(&weak_workspace, task_with_shell.save, cx).await;
                this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
                if let Some(SelectedJavaTestBuild::Local { worktree_id, node_id, generation, port, plan }) = &selected_java_test {
                    let current = project.update(cx, |project, cx| {
                        project.selected_java_test_debug_task(*worktree_id, node_id, *generation, Some(*port), cx)
                    })?;
                    anyhow::ensure!(
                        current == *plan,
                        "The selected Java test changed before Debug could start; refresh Tests"
                    );
                    std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, *port))
                        .with_context(|| format!("Java test Debug port {port} is unavailable"))?;
                    #[cfg(feature = "jvm-tools")]
                    if Self::run_before_selected_java_debug(&this, &project, *worktree_id, cx).await? {
                        this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
                        let current = project.update(cx, |project, cx| {
                            project.selected_java_test_debug_task(*worktree_id, node_id, *generation, Some(*port), cx)
                        })?;
                        anyhow::ensure!(current == *plan, "The selected Java test changed before Debug could start; refresh Tests");
                    }
                }

                let creation = if selected_java_test.is_some() {
                    let preflight = if let Some(SelectedJavaTestBuild::Remote(plan)) = &selected_java_test {
                        let project = project.downgrade();
                        let plan = plan.clone();
                        let check: project::terminals::TaskTerminalCheck = Box::new(move |cx| {
                            cx.spawn(async move |cx| {
                                let verification = project.update(cx, |project, cx| {
                                    project.check_remote_java_test_debug_task(&plan, false, cx)
                                })?;
                                anyhow::ensure!(!verification.await?, "The Java test Debug port opened before the Task started");
                                Ok(())
                            })
                        });
                        Some(check)
                    } else {
                        None
                    };
                    this.update(cx, |this, cx| {
                        this.create_selected_java_test_terminal(&project, task_with_shell, preflight, cx)
                    })?
                } else {
                    project.update(cx, |project, cx| {
                        project.create_terminal_task(task_with_shell, cx)
                    })
                };
                let terminal = creation.await?;

                let terminal_view = cx.new_window_entity(|window, cx| {
                    TerminalView::new(
                        terminal.clone(),
                        weak_workspace,
                        None,
                        weak_project,
                        window,
                        cx,
                    )
                })?;

                #[cfg(feature = "jvm-tools")]
                this.update(cx, |this, _| {
                    if let Some(hooks) = &mut this.selected_java_debug_hooks {
                        hooks.terminal = Some(terminal.clone());
                    }
                })?;
                this.update_in(cx, |this, window, cx| {
                    this.ensure_pane_item(DebuggerPaneItem::Terminal, window, cx);
                    this.debug_terminal.update(cx, |debug_terminal, cx| {
                        debug_terminal.terminal = Some(terminal_view);
                        cx.notify();
                    });
                })?;

                if let Some(SelectedJavaTestBuild::Local { worktree_id, node_id, generation, port, plan }) = selected_java_test {
                    // The selected JVM is suspended until DAP attaches. Waiting for Task exit would deadlock.
                    wait_for_java_test_debugger(&terminal, port, plan.fixed_port.is_some(), None, cx).await?;
                    let current = project.update(cx, |project, cx| {
                        project.selected_java_test_debug_task(worktree_id, &node_id, generation, Some(port), cx)
                    })?;
                    anyhow::ensure!(
                        current == plan,
                        "The selected Java test changed while waiting for Debug; refresh Tests"
                    );
                } else if let Some(SelectedJavaTestBuild::Remote(plan)) = selected_java_test {
                    wait_for_java_test_debugger(&terminal, plan.port, plan.announces_port, Some((&project, &plan)), cx).await?;
                } else {
                    let exit_status = terminal
                        .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                        .await
                        .context("Failed to wait for completed task")?;
                    anyhow::ensure!(exit_status.success(), "Build failed");
                }
                this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
                Some((task.resolved.clone(), locator_name, extra_config))
            } else {
                None
            };

            if config_is_valid {
            } else if let Some((task, locator_name, extra_config)) = build_output {
                let locator_name =
                    locator_name.with_context(|| {
                        format!("Could not find a valid locator for a build task and configure is invalid with error: {}", request_type.err()
                            .map(|err| err.to_string())
                            .unwrap_or_default())
                    })?;
                let request = dap_store
                    .update(cx, |this, cx| {
                        this.run_debug_locator(&locator_name, task, cx)
                    })?
                    .await?;
                this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;

                let zed_config = ZedDebugConfig {
                    label: label.clone(),
                    adapter: adapter.clone(),
                    request,
                    stop_on_entry: None,
                };

                let scenario = dap_registry
                    .adapter(&adapter)
                    .with_context(|| anyhow!("{adapter}: is not a valid adapter name"))?.config_from_zed_format(zed_config)
                    .await?;
                this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
                config = scenario.config;
                util::merge_non_null_json_value_into(extra_config, &mut config);

                Self::substitute_variables_in_config(&mut config, &task_context);
            } else {
                let Err(e) = request_type else {
                    unreachable!();
                };
                anyhow::bail!("Zed cannot determine how to run this debug scenario. `build` field was not provided and Debug Adapter won't accept provided configuration because: {e}");
            };

            this.read_with(cx, |this, cx| this.ensure_experimental_access(cx))??;
            Ok(DebugTaskDefinition {
                label,
                adapter: DebugAdapterName(adapter),
                config,
                tcp_connection,
            })
        })
    }

    fn handle_run_in_terminal(
        &self,
        request: &RunInTerminalRequestArguments,
        mut sender: mpsc::Sender<Result<dap::RunInTerminalResponse>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if let Err(error) = self.ensure_experimental_access(cx) {
            return cx.background_spawn(async move {
                sender
                    .send(Err(error))
                    .await
                    .context("Could not return denied debug terminal request")
            });
        }
        let running = cx.entity();
        let Ok(project) = self
            .workspace
            .read_with(cx, |workspace, _| workspace.project().clone())
        else {
            return Task::ready(Err(anyhow!("no workspace")));
        };
        let session = self.session.read(cx);

        let cwd = (!request.cwd.is_empty())
            .then(|| PathBuf::from(&request.cwd))
            .or_else(|| session.binary().unwrap().cwd.clone());

        let mut envs: HashMap<String, String> =
            self.session.read(cx).task_context().project_env.clone();
        if let Some(Value::Object(env)) = &request.env {
            for (key, value) in env {
                let value_str = match (key.as_str(), value) {
                    (_, Value::String(value)) => value,
                    _ => continue,
                };

                envs.insert(key.clone(), value_str.clone());
            }
        }

        let mut args = request.args.clone();
        let command = if envs.contains_key("VSCODE_INSPECTOR_OPTIONS") {
            // Handle special case for NodeJS debug adapter
            // If the Node binary path is provided (possibly with arguments like --experimental-network-inspection),
            // we set the command to None
            // This prevents the NodeJS REPL from appearing, which is not the desired behavior
            // The expected usage is for users to provide their own Node command, e.g., `node test.js`
            // This allows the NodeJS debug client to attach correctly
            if args
                .iter()
                .filter(|arg| !arg.starts_with("--"))
                .collect::<Vec<_>>()
                .len()
                > 1
            {
                Some(args.remove(0))
            } else {
                None
            }
        } else if !args.is_empty() {
            Some(args.remove(0))
        } else {
            None
        };

        let shell = project.read(cx).terminal_settings(&cwd, cx).shell.clone();
        let title = request
            .title
            .clone()
            .filter(|title| !title.is_empty())
            .or_else(|| command.clone())
            .unwrap_or_else(|| "Debug terminal".to_string());
        let kind = task::SpawnInTerminal {
            required_experimental_features: self
                .session
                .read(cx)
                .required_experimental_features()
                .to_vec(),
            id: task::TaskId("debug".to_string()),
            full_label: title.clone(),
            label: title.clone(),
            command,
            args,
            command_label: title,
            preserve_command_label: false,
            literal_command: false,
            literal_prefix_args: 0,
            windows_direct_program: false,
            cwd,
            env: envs,
            use_new_terminal: true,
            allow_concurrent_runs: true,
            reveal: task::RevealStrategy::NoFocus,
            reveal_target: task::RevealTarget::Dock,
            hide: task::HideStrategy::Never,
            shell,
            show_summary: false,
            show_command: false,
            show_rerun: false,
            save: task::SaveStrategy::default(),
        };

        let workspace = self.workspace.clone();
        let weak_project = project.downgrade();

        let terminal_task =
            project.update(cx, |project, cx| project.create_terminal_task(kind, cx));
        let terminal_task = cx.spawn_in(window, async move |_, cx| {
            let terminal = terminal_task.await?;

            let terminal_view = cx.new_window_entity(|window, cx| {
                TerminalView::new(terminal.clone(), workspace, None, weak_project, window, cx)
            })?;

            running.update_in(cx, |running, window, cx| {
                running.ensure_pane_item(DebuggerPaneItem::Terminal, window, cx);
                running.debug_terminal.update(cx, |debug_terminal, cx| {
                    debug_terminal.terminal = Some(terminal_view);
                    cx.notify();
                });
            })?;

            terminal.read_with(cx, |terminal, _| {
                let process_id = terminal
                    .task_process_id()
                    .map(|process_id| u64::from(process_id.as_u32()))
                    .context(if terminal.is_pty() {
                        "Terminal was spawned but PID was not available"
                    } else {
                        "Headless terminal cannot identify the exact requested local process"
                    })?;
                Ok(dap::RunInTerminalResponse {
                    process_id: (!terminal.is_pty()).then_some(process_id),
                    shell_process_id: terminal.is_pty().then_some(process_id),
                })
            })
        });

        cx.background_spawn(async move { anyhow::Ok(sender.send(terminal_task.await).await?) })
    }

    fn create_sub_view(
        &self,
        item_kind: DebuggerPaneItem,
        pane: &Entity<Pane>,
        cx: &mut Context<Self>,
    ) -> Box<dyn ItemHandle> {
        let running_state = cx.weak_entity();
        let host_pane = pane.downgrade();

        match item_kind {
            DebuggerPaneItem::Console => Box::new(SubView::console(
                self.console.clone(),
                running_state,
                host_pane,
                cx,
            )),
            DebuggerPaneItem::Variables => Box::new(SubView::new(
                self.variable_list.focus_handle(cx),
                self.variable_list.clone().into(),
                item_kind,
                running_state,
                host_pane,
                cx,
            )),
            DebuggerPaneItem::BreakpointList => Box::new(SubView::breakpoint_list(
                self.breakpoint_list.clone(),
                running_state,
                host_pane,
                cx,
            )),
            DebuggerPaneItem::Frames => Box::new(SubView::new(
                self.stack_frame_list.focus_handle(cx),
                self.stack_frame_list.clone().into(),
                item_kind,
                running_state,
                host_pane,
                cx,
            )),
            DebuggerPaneItem::Modules => Box::new(SubView::new(
                self.module_list.focus_handle(cx),
                self.module_list.clone().into(),
                item_kind,
                running_state,
                host_pane,
                cx,
            )),
            DebuggerPaneItem::LoadedSources => Box::new(SubView::new(
                self.loaded_sources_list.focus_handle(cx),
                self.loaded_sources_list.clone().into(),
                item_kind,
                running_state,
                host_pane,
                cx,
            )),
            DebuggerPaneItem::Terminal => Box::new(SubView::new(
                self.debug_terminal.focus_handle(cx),
                self.debug_terminal.clone().into(),
                item_kind,
                running_state,
                host_pane,
                cx,
            )),
            DebuggerPaneItem::MemoryView => Box::new(SubView::new(
                self.memory_view.focus_handle(cx),
                self.memory_view.clone().into(),
                item_kind,
                running_state,
                host_pane,
                cx,
            )),
        }
    }

    pub(crate) fn ensure_pane_item(
        &mut self,
        item_kind: DebuggerPaneItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.pane_items_status(cx).get(&item_kind) == Some(&true) {
            return;
        };
        let pane = self.panes.last_pane();
        let sub_view = self.create_sub_view(item_kind, &pane, cx);

        pane.update(cx, |pane, cx| {
            pane.add_item_inner(sub_view, false, false, false, None, window, cx);
        })
    }

    pub(crate) fn add_pane_item(
        &mut self,
        item_kind: DebuggerPaneItem,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        debug_assert!(
            item_kind.is_supported(self.session.read(cx).capabilities()),
            "We should only allow adding supported item kinds"
        );

        if let Some(pane) = self.panes.pane_at_pixel_position(position) {
            let sub_view = self.create_sub_view(item_kind, pane, cx);

            pane.update(cx, |pane, cx| {
                pane.add_item(sub_view, false, false, None, window, cx);
            })
        }
    }

    pub(crate) fn pane_items_status(&self, cx: &App) -> IndexMap<DebuggerPaneItem, bool> {
        let caps = self.session.read(cx).capabilities();
        let mut pane_item_status = IndexMap::from_iter(
            DebuggerPaneItem::all()
                .iter()
                .filter(|kind| kind.is_supported(caps))
                .map(|kind| (*kind, false)),
        );
        self.panes.panes().iter().for_each(|pane| {
            pane.read(cx)
                .items()
                .filter_map(|item| item.act_as::<SubView>(cx))
                .for_each(|view| {
                    pane_item_status.insert(view.read(cx).kind, true);
                });
        });

        pane_item_status
    }

    pub(crate) fn serialize_layout(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self._schedule_serialize.is_none() {
            self._schedule_serialize = Some(cx.spawn_in(window, async move |this, cx| {
                cx.background_executor()
                    .timer(Duration::from_millis(100))
                    .await;

                let Some((adapter_name, pane_layout)) = this
                    .read_with(cx, |this, cx| {
                        let adapter_name = this.session.read(cx).adapter();
                        (
                            adapter_name,
                            persistence::build_serialized_layout(
                                &this.panes.root,
                                this.dock_axis,
                                cx,
                            ),
                        )
                    })
                    .ok()
                else {
                    return;
                };

                let kvp = this
                    .read_with(cx, |_, cx| db::kvp::KeyValueStore::global(cx))
                    .ok();
                if let Some(kvp) = kvp {
                    persistence::serialize_pane_layout(adapter_name, pane_layout, kvp)
                        .await
                        .log_err();
                }

                this.update(cx, |this, _| {
                    this._schedule_serialize.take();
                })
                .ok();
            }));
        }
    }

    pub(crate) fn handle_pane_event(
        this: &mut RunningState,
        source_pane: &Entity<Pane>,
        event: &Event,
        window: &mut Window,
        cx: &mut Context<RunningState>,
    ) {
        this.serialize_layout(window, cx);
        match event {
            Event::AddItem { item } => {
                if let Some(sub_view) = item.downcast::<SubView>() {
                    sub_view.update(cx, |sub_view, _| {
                        sub_view.set_host_pane(source_pane.downgrade());
                    });
                }
            }
            Event::Remove { .. } => {
                let _did_find_pane = this.panes.remove(source_pane, cx).is_ok();
                debug_assert!(_did_find_pane);
                cx.notify();
            }
            Event::Focus => {
                this.active_pane = source_pane.clone();
            }
            _ => {}
        }
    }

    pub(crate) fn activate_pane_in_direction(
        &mut self,
        direction: SplitDirection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let active_pane = self.active_pane.clone();
        if let Some(pane) = self
            .panes
            .find_pane_in_direction(&active_pane, direction, cx)
        {
            pane.update(cx, |pane, cx| {
                pane.focus_active_item(window, cx);
            })
        } else {
            self.workspace
                .update(cx, |workspace, cx| {
                    workspace.activate_pane_in_direction(direction, window, cx)
                })
                .ok();
        }
    }

    pub(crate) fn go_to_selected_stack_frame(&self, window: &mut Window, cx: &mut Context<Self>) {
        if self.thread_id.is_some() {
            self.stack_frame_list
                .update(cx, |list, cx| {
                    let Some(stack_frame_id) = list.opened_stack_frame_id() else {
                        return Task::ready(Ok(()));
                    };
                    list.go_to_stack_frame(stack_frame_id, window, cx)
                })
                .detach();
        }
    }

    pub(crate) fn has_open_context_menu(&self, cx: &App) -> bool {
        self.variable_list.read(cx).has_open_context_menu()
    }

    pub fn session(&self) -> &Entity<Session> {
        &self.session
    }

    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    pub(crate) fn selected_stack_frame_id(&self, cx: &App) -> Option<dap::StackFrameId> {
        self.stack_frame_list.read(cx).opened_stack_frame_id()
    }

    pub(crate) fn stack_frame_list(&self) -> &Entity<StackFrameList> {
        &self.stack_frame_list
    }

    #[cfg(test)]
    pub fn console(&self) -> &Entity<Console> {
        &self.console
    }

    #[cfg(test)]
    pub(crate) fn module_list(&self) -> &Entity<ModuleList> {
        &self.module_list
    }

    pub(crate) fn activate_item(
        &mut self,
        item: DebuggerPaneItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.ensure_pane_item(item, window, cx);

        let (variable_list_position, pane) = self
            .panes
            .panes()
            .into_iter()
            .find_map(|pane| {
                pane.read(cx)
                    .items_of_type::<SubView>()
                    .position(|view| view.read(cx).view_kind() == item)
                    .map(|view| (view, pane))
            })
            .unwrap();

        pane.update(cx, |this, cx| {
            this.activate_item(variable_list_position, true, true, window, cx);
        });
    }

    #[cfg(test)]
    pub(crate) fn variable_list(&self) -> &Entity<VariableList> {
        &self.variable_list
    }

    #[cfg(test)]
    pub(crate) fn serialized_layout(&self, cx: &App) -> SerializedLayout {
        persistence::build_serialized_layout(&self.panes.root, self.dock_axis, cx)
    }

    pub fn capabilities(&self, cx: &App) -> Capabilities {
        self.session().read(cx).capabilities().clone()
    }

    pub fn select_current_thread(
        &mut self,
        threads: &Vec<(Thread, ThreadStatus)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let selected_thread = self
            .thread_id
            .and_then(|thread_id| threads.iter().find(|(thread, _)| thread.id == thread_id.0))
            .or_else(|| threads.first());

        let Some((selected_thread, _)) = selected_thread else {
            return;
        };

        if Some(ThreadId(selected_thread.id)) != self.thread_id {
            self.select_thread(ThreadId(selected_thread.id), window, cx);
        }
    }

    pub fn selected_thread_id(&self) -> Option<ThreadId> {
        self.thread_id
    }

    pub fn thread_status(&self, cx: &App) -> Option<ThreadStatus> {
        self.thread_id
            .map(|id| self.session().read(cx).thread_status(id))
    }

    pub(crate) fn select_thread(
        &mut self,
        thread_id: ThreadId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.thread_id.is_some_and(|id| id == thread_id) {
            return;
        }

        self.thread_id = Some(thread_id);

        self.stack_frame_list
            .update(cx, |list, cx| list.schedule_refresh(true, window, cx));
    }

    pub fn continue_thread(&mut self, cx: &mut Context<Self>) {
        let Some(thread_id) = self.thread_id else {
            return;
        };

        self.session().update(cx, |state, cx| {
            state.continue_thread(thread_id, cx);
        });
    }

    pub fn step_over(&mut self, cx: &mut Context<Self>) {
        let Some(thread_id) = self.thread_id else {
            return;
        };

        let granularity = DebuggerSettings::get_global(cx).stepping_granularity;

        self.session().update(cx, |state, cx| {
            state.step_over(thread_id, granularity, cx);
        });
    }

    pub(crate) fn step_in(&mut self, cx: &mut Context<Self>) {
        let Some(thread_id) = self.thread_id else {
            return;
        };

        let granularity = DebuggerSettings::get_global(cx).stepping_granularity;

        self.session().update(cx, |state, cx| {
            state.step_in(thread_id, granularity, cx);
        });
    }

    pub(crate) fn step_out(&mut self, cx: &mut Context<Self>) {
        let Some(thread_id) = self.thread_id else {
            return;
        };

        let granularity = DebuggerSettings::get_global(cx).stepping_granularity;

        self.session().update(cx, |state, cx| {
            state.step_out(thread_id, granularity, cx);
        });
    }

    pub(crate) fn step_back(&mut self, cx: &mut Context<Self>) {
        let Some(thread_id) = self.thread_id else {
            return;
        };

        let granularity = DebuggerSettings::get_global(cx).stepping_granularity;

        self.session().update(cx, |state, cx| {
            state.step_back(thread_id, granularity, cx);
        });
    }

    pub fn rerun_session(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.session().read(cx).diagnostic_observation().is_some() {
            if let Some(project) = self.project.upgrade() {
                project.read(cx).dap_store().update(cx, |_, cx| {
                    cx.emit(project::debugger::dap_store::DapStoreEvent::Notification(
                        "Review a new Doctor action to rerun a diagnostic session".into(),
                    ));
                });
            } else {
                log::warn!("Diagnostic project is unavailable; review a new Doctor action");
            }
            return;
        }
        if self
            .scenario
            .as_ref()
            .is_some_and(|scenario| scenario.build.is_some())
        {
            let admission = self
                .scenario_context
                .as_ref()
                .context("Refresh and reselect this debug scenario before replaying legacy history")
                .and_then(DebugScenarioContext::validate_replay_policy);
            if let Err(error) = admission {
                self.session.update(cx, |session, cx| {
                    session
                        .console_output(cx)
                        .unbounded_send(format!("error: {error}"))
                        .log_err();
                });
                return;
            }
        }
        if let Some((scenario, context)) = self.scenario.take().zip(self.scenario_context.take())
            && scenario.build.is_some()
        {
            let DebugScenarioContext {
                task_context,
                active_buffer,
                worktree_id,
                experimental_policy_version: _,
            } = context;
            let active_buffer = active_buffer.and_then(|buffer| buffer.upgrade());

            self.workspace
                .update(cx, |workspace, cx| {
                    workspace.start_debug_session(
                        scenario,
                        task_context,
                        active_buffer,
                        worktree_id,
                        window,
                        cx,
                    )
                })
                .ok();
        } else {
            self.restart_session(cx);
        }
    }

    pub fn restart_session(&self, cx: &mut Context<Self>) {
        if let Err(error) = self.ensure_experimental_access(cx) {
            self.session.update(cx, |session, cx| {
                session
                    .console_output(cx)
                    .unbounded_send(format!("error: {error}"))
                    .log_err();
            });
            return;
        }
        self.session().update(cx, |state, cx| {
            state.restart(None, cx);
        });
    }

    pub fn pause_thread(&self, cx: &mut Context<Self>) {
        let Some(thread_id) = self.thread_id else {
            return;
        };

        self.session().update(cx, |state, cx| {
            state.pause_thread(thread_id, cx);
        });
    }

    pub(crate) fn shutdown(&mut self, cx: &mut Context<Self>) {
        #[cfg(feature = "jvm-tools")]
        self.cancel_selected_java_debug_hooks(cx);
        let owns_selected_java_test = self.selected_java_test_task.is_some();
        self.stop_selected_java_test(cx);
        self.workspace
            .update(cx, |workspace, cx| {
                workspace
                    .project()
                    .read(cx)
                    .breakpoint_store()
                    .update(cx, |store, cx| {
                        store.remove_active_position(Some(self.session_id), cx)
                    })
            })
            .log_err();

        let is_building = self.session.update(cx, |session, cx| {
            session.shutdown(cx).detach();
            matches!(session.state, session::SessionState::Booting(_))
        });

        if is_building && !owns_selected_java_test {
            self.debug_terminal.update(cx, |terminal, cx| {
                if let Some(view) = terminal.terminal.as_ref() {
                    view.update(cx, |view, cx| {
                        view.terminal()
                            .update(cx, |terminal, _| terminal.kill_active_task())
                    })
                }
            })
        }
    }

    fn stop_selected_java_test(&mut self, cx: &mut Context<Self>) {
        if let Some(task) = self.selected_java_test_task.take() {
            task.cancel(cx);
        }
    }

    fn create_selected_java_test_terminal(
        &mut self,
        project: &Entity<Project>,
        spawn_task: SpawnInTerminal,
        preflight: Option<project::terminals::TaskTerminalCheck>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<terminal::Terminal>>> {
        if self.session.read(cx).is_terminated() {
            return Task::ready(Err(anyhow!("The selected Java test Debug session stopped")));
        }
        #[cfg(feature = "jvm-tools")]
        let preflight = if self.selected_java_debug_hooks.is_some() && preflight.is_none() {
            let state = cx.weak_entity();
            let check: project::terminals::TaskTerminalCheck = Box::new(move |cx| {
                Task::ready(
                    state
                        .read_with(cx, |state, cx| {
                            anyhow::ensure!(
                                state.selected_java_debug_hooks.is_some(),
                                "The selected Java Debug sequence stopped"
                            );
                            state.ensure_experimental_access(cx)
                        })
                        .and_then(|result| result),
                )
            });
            Some(check)
        } else {
            preflight
        };
        let task = StructuredTaskHandle::new(spawn_task.id.clone());
        self.selected_java_test_task = Some(task.clone());
        project.update(cx, |project, cx| {
            project.create_terminal_task_with_preflight(
                spawn_task,
                Some(task),
                None,
                None,
                preflight,
                cx,
            )
        })
    }

    pub fn stop_thread(&self, cx: &mut Context<Self>) {
        let Some(thread_id) = self.thread_id else {
            return;
        };

        self.workspace
            .update(cx, |workspace, cx| {
                workspace
                    .project()
                    .read(cx)
                    .breakpoint_store()
                    .update(cx, |store, cx| {
                        store.remove_active_position(Some(self.session_id), cx)
                    })
            })
            .log_err();

        self.session().update(cx, |state, cx| {
            state.terminate_threads(Some(vec![thread_id; 1]), cx);
        });
    }

    pub fn detach_client(&mut self, cx: &mut Context<Self>) {
        #[cfg(feature = "jvm-tools")]
        self.cancel_selected_java_debug_hooks(cx);
        self.session().update(cx, |state, cx| {
            state.disconnect_client(cx);
        });
    }

    pub fn toggle_ignore_breakpoints(&mut self, cx: &mut Context<Self>) {
        self.session.update(cx, |session, cx| {
            session.toggle_ignore_breakpoints(cx).detach();
        });
    }

    fn default_pane_layout(
        project: Entity<Project>,
        workspace: &WeakEntity<Workspace>,
        stack_frame_list: &Entity<StackFrameList>,
        variable_list: &Entity<VariableList>,
        console: &Entity<Console>,
        breakpoints: &Entity<BreakpointList>,
        debug_terminal: &Entity<DebugTerminal>,
        dock_axis: Axis,
        subscriptions: &mut HashMap<EntityId, Subscription>,
        window: &mut Window,
        cx: &mut Context<'_, RunningState>,
    ) -> Member {
        let running_state = cx.weak_entity();

        let leftmost_pane = new_debugger_pane(workspace.clone(), project.clone(), window, cx);
        let leftmost_pane_handle = leftmost_pane.downgrade();
        let leftmost_frames = SubView::new(
            stack_frame_list.focus_handle(cx),
            stack_frame_list.clone().into(),
            DebuggerPaneItem::Frames,
            running_state.clone(),
            leftmost_pane_handle.clone(),
            cx,
        );
        let leftmost_breakpoints = SubView::breakpoint_list(
            breakpoints.clone(),
            running_state.clone(),
            leftmost_pane_handle,
            cx,
        );
        leftmost_pane.update(cx, |this, cx| {
            this.add_item(Box::new(leftmost_frames), true, false, None, window, cx);
            this.add_item(
                Box::new(leftmost_breakpoints),
                true,
                false,
                None,
                window,
                cx,
            );
            this.activate_item(0, false, false, window, cx);
        });

        let center_pane = new_debugger_pane(workspace.clone(), project.clone(), window, cx);
        let center_pane_handle = center_pane.downgrade();
        let center_console = SubView::console(
            console.clone(),
            running_state.clone(),
            center_pane_handle.clone(),
            cx,
        );
        let center_variables = SubView::new(
            variable_list.focus_handle(cx),
            variable_list.clone().into(),
            DebuggerPaneItem::Variables,
            running_state.clone(),
            center_pane_handle,
            cx,
        );

        center_pane.update(cx, |this, cx| {
            this.add_item(Box::new(center_console), true, false, None, window, cx);

            this.add_item(Box::new(center_variables), true, false, None, window, cx);
            this.activate_item(0, false, false, window, cx);
        });

        let rightmost_pane = new_debugger_pane(workspace.clone(), project, window, cx);
        let rightmost_terminal = SubView::new(
            debug_terminal.focus_handle(cx),
            debug_terminal.clone().into(),
            DebuggerPaneItem::Terminal,
            running_state,
            rightmost_pane.downgrade(),
            cx,
        );
        rightmost_pane.update(cx, |this, cx| {
            this.add_item(Box::new(rightmost_terminal), false, false, None, window, cx);
        });

        subscriptions.extend(
            [&leftmost_pane, &center_pane, &rightmost_pane]
                .into_iter()
                .map(|entity| {
                    (
                        entity.entity_id(),
                        cx.subscribe_in(entity, window, Self::handle_pane_event),
                    )
                }),
        );

        let group_root = workspace::PaneAxis::new(
            dock_axis.invert(),
            [leftmost_pane, center_pane, rightmost_pane]
                .into_iter()
                .map(workspace::Member::Pane)
                .collect(),
        );

        Member::Axis(group_root)
    }

    pub(crate) fn invert_axies(&mut self, cx: &mut App) {
        self.dock_axis = self.dock_axis.invert();
        self.panes.invert_axies(cx);
    }
}

impl Focusable for RunningState {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        debugger_panel::DebugPanel,
        tests::{init_test, init_test_workspace, start_debug_session},
    };
    use gpui::{BackgroundExecutor, TestAppContext, VisualTestContext};
    use project::{FakeFs, Project};
    use serde_json::json;
    use util::path;

    #[cfg(unix)]
    #[gpui::test]
    async fn java_test_debug_readiness_precedes_task_exit(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);
        let project = Project::test(FakeFs::new(cx.executor()), [], cx).await;
        let (_, cx) = cx.add_window_view(|_, _| gpui::EmptyView);
        for (script, error_message) in [
            (
                "printf '%s\\n' 'Listening for transport dt_socket at address: 50051' 'Fixture test build failed'; exit 3",
                Some("failed before the JVM suspended (exit status: 3)"),
            ),
            (
                "printf '%s\\n' 'Fixture test completed without suspending'; exit 0",
                Some("finished successfully before the JVM suspended"),
            ),
            (
                "printf '%s\\n' 'Listening for transport dt_socket at address: 5005'; read answer",
                None,
            ),
        ] {
            let spawn_task = SpawnInTerminal {
                command: Some("/bin/sh".into()),
                args: vec!["-c".into(), script.into()],
                ..Default::default()
            };
            let handle = error_message
                .is_none()
                .then(|| StructuredTaskHandle::new(spawn_task.id.clone()));
            let terminal = project
                .update(cx, |project, cx| {
                    project.create_terminal_task_with_structured_handle(
                        spawn_task,
                        handle.clone(),
                        cx,
                    )
                })
                .await
                .expect("test Debug terminal");
            let mut window = cx.update(|window, cx| window.to_async(cx));
            let result =
                wait_for_java_test_debugger(&terminal, 5005, true, None, &mut window).await;
            if let Some(expected) = error_message {
                let error = result.expect_err("only a suspended JVM authorizes attachment");
                let error = error.to_string();
                assert!(error.contains(expected), "{error}");
                assert!(error.contains("Check the Debug terminal"), "{error}");
                terminal.read_with(cx, |terminal, _| {
                    assert!(terminal.get_content().contains("Fixture test"));
                });
            } else {
                result.expect("attached while the Task was still waiting");
                terminal.read_with(cx, |terminal, _| {
                    assert!(
                        terminal
                            .task()
                            .is_some_and(|task| task.status == terminal::TaskStatus::Running)
                    );
                });
                assert!(cx.update(|_, cx| handle.expect("selected test Task handle").cancel(cx)));
                let error = wait_for_java_test_debugger(&terminal, 5005, true, None, &mut window)
                    .await
                    .expect_err("the cancelled Task cannot authorize attachment");
                assert!(
                    error
                        .to_string()
                        .contains("was cancelled before the JVM suspended")
                );
            }
            assert!(
                terminal
                    .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                    .await
                    .is_some()
            );
        }
    }

    #[cfg(all(unix, feature = "jvm-workflow-tests"))]
    #[gpui::test]
    #[expect(clippy::result_large_err)]
    async fn selected_java_debug_hooks_wait_for_natural_termination_and_task_success(
        cx: &mut TestAppContext,
    ) {
        use dap::{
            StartDebuggingRequestArguments, StartDebuggingRequestArgumentsRequest,
            adapters::DebugAdapterBinary,
        };
        use futures::StreamExt as _;
        use language::{FakeLspAdapter, Language, LanguageConfig, LanguageMatcher};
        use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
        use util::rel_path::RelPath;

        cx.executor().allow_parking();
        init_test(cx);
        cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
        let root = tempfile::tempdir().expect("disposable selected Debug worktree");
        let root_path = root.path().canonicalize().expect("canonical project root");
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            &root_path,
            json!({"pom.xml": "<project/>", "Main.java": "class Main {}"}),
        )
        .await;
        let project = Project::test_with_worktree_trust(fs, [root_path.as_path()], cx).await;
        let window = init_test_workspace(&project, cx).await;
        let (dap, lsp, worktrees, worktree, languages, inventory) =
            project.read_with(cx, |project, cx| {
                (
                    project.dap_store(),
                    project.lsp_store(),
                    project.worktree_store(),
                    project.visible_worktrees(cx).next().expect("Java worktree"),
                    project.languages().clone(),
                    project
                        .task_store()
                        .read(cx)
                        .task_inventory()
                        .cloned()
                        .expect("Task inventory"),
                )
            });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let trust = cx.read(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust owner"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktrees,
                collections::HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let mut servers = languages.register_fake_lsp(
            "Java",
            FakeLspAdapter {
                name: "jdtls",
                ..Default::default()
            },
        );
        let (_buffer, _open) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp(root_path.join("Main.java"), cx)
            })
            .await
            .expect("Java source");
        let _server = servers.next().await.expect("live Java provider");
        cx.run_until_parked();
        let save_tasks = |contents: &str, cx: &mut TestAppContext| {
            inventory.update(cx, |inventory, _| {
                inventory
                    .update_file_based_tasks(
                        project::task_store::TaskSettingsLocation::Worktree(
                            settings::SettingsLocation {
                                worktree_id,
                                path: RelPath::from_unix_str(".zed").expect("Task settings folder"),
                            },
                        ),
                        Some(contents),
                    )
                    .expect("saved Task definitions");
            });
        };
        let _dap = project::debugger::test::intercept_debug_sessions(cx, |client| {
            client.on_request::<dap::requests::Attach, _>(|_, _| Ok(()));
        });
        let create_session = |cx: &mut TestAppContext| {
            let session = dap.update(cx, |dap, cx| {
                dap.new_session(
                    Some("Selected Java test".into()),
                    "Java".into(),
                    SharedTaskContext::default(),
                    None,
                    Default::default(),
                    cx,
                )
            });
            let running = window
                .update(cx, |multi, window, cx| {
                    let workspace = multi.workspace().downgrade();
                    cx.new(|cx| {
                        RunningState::new(
                            session.clone(),
                            project.clone(),
                            workspace,
                            None,
                            None,
                            Axis::Horizontal,
                            window,
                            cx,
                        )
                    })
                })
                .expect("Debug window");
            (session, running)
        };
        let armed = |cx: &mut TestAppContext| {
            lsp.read_with(cx, |lsp, _| {
                lsp.jvm_trigger_is_armed(worktree_id, JvmTaskTrigger::BeforeDebug)
            })
        };

        for (attempt, exit_code, natural) in [(0, 0, true), (1, 7, true), (2, 0, false)] {
            let before_marker = format!("before-{attempt}");
            let after_marker = format!("after-{attempt}");
            let release_marker = format!("release-{attempt}");
            save_tasks(&json!([
                {"label":"prepare-test","command":"/usr/bin/touch","args":[before_marker],"tags":["before-debug"]},
                {"label":"finish-test","command":"/usr/bin/touch","args":[after_marker],"tags":["after-debug"]},
                {"label":"unrelated","command":"/usr/bin/touch","args":["unrelated"],"tags":["after-run"]}
            ]).to_string(), cx);
            let (session, running) = create_session(cx);
            let mut async_window = window
                .update(cx, |_, window, cx| window.to_async(cx))
                .expect("async Debug window");
            assert!(
                RunningState::run_before_selected_java_debug(
                    &running.downgrade(),
                    &project,
                    worktree_id,
                    &mut async_window
                )
                .await
                .expect("before-debug Tasks")
            );
            assert!(root_path.join(&before_marker).is_file());
            assert!(armed(cx));
            let terminal = running.update(cx, |running, cx| {
                running.create_selected_java_test_terminal(
                    &project,
                    SpawnInTerminal {
                        cwd: Some(root_path.clone()),
                        command: Some("/bin/sh".into()),
                        args: vec!["-c".into(), format!("while [ ! -f {release_marker} ]; do sleep 0.02; done; exit {exit_code}")],
                        ..Default::default()
                    },
                    None,
                    cx
                )
            }).await.expect("selected Java wrapper terminal");
            running.update(cx, |running, _| {
                running
                    .selected_java_debug_hooks
                    .as_mut()
                    .expect("owned Debug hooks")
                    .terminal = Some(terminal.clone());
            });
            session
                .update(cx, |session, cx| {
                    session.boot(
                        DebugAdapterBinary {
                            command: Some("command".into()),
                            arguments: Vec::new(),
                            envs: Default::default(),
                            cwd: None,
                            connection: None,
                            request_args: StartDebuggingRequestArguments {
                                request: StartDebuggingRequestArgumentsRequest::Attach,
                                configuration: json!({"request":"attach"}),
                            },
                        },
                        worktree.clone(),
                        dap.downgrade(),
                        cx,
                    )
                })
                .await
                .expect("Mock Java adapter attached");
            let client = session
                .read_with(cx, |session, _| session.adapter_client())
                .expect("attached client");
            if natural {
                client
                    .fake_event(dap::messages::Events::Terminated(None))
                    .await;
            } else {
                running.update(cx, |running, cx| running.shutdown(cx));
            }
            cx.run_until_parked();
            assert!(
                !root_path.join(&after_marker).exists(),
                "the after Task must wait for wrapper exit"
            );
            std::fs::write(root_path.join(&release_marker), "")
                .expect("release selected wrapper fixture");
            terminal
                .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                .await;
            let deadline = std::time::Instant::now() + Duration::from_secs(8);
            while armed(cx) && std::time::Instant::now() < deadline {
                cx.background_executor
                    .timer(Duration::from_millis(10))
                    .await;
                cx.run_until_parked();
            }
            assert!(
                !armed(cx),
                "Debug must release the activation after completion or Stop"
            );
            assert_eq!(
                root_path.join(&after_marker).is_file(),
                natural && exit_code == 0
            );
            assert!(!root_path.join("unrelated").exists());
        }

        use std::os::unix::fs::PermissionsExt as _;
        let blocked_setup = root_path.join("block-debug-setup");
        std::fs::write(
            &blocked_setup,
            "#!/bin/sh\n/usr/bin/touch before-blocking\nread resume\n",
        )
        .expect("write the disposable blocking setup Task");
        std::fs::set_permissions(&blocked_setup, std::fs::Permissions::from_mode(0o755))
            .expect("make the fixture Task executable");
        save_tasks(
            r#"[
                {"label":"blocked-setup","command":"./block-debug-setup","tags":["before-debug"]},
                {"label":"finish-test","command":"/usr/bin/touch","args":["after-stop"],"tags":["after-debug"]}
            ]"#,
            cx,
        );
        let (_session, running) = create_session(cx);
        let pending = window
            .update(cx, |_, window, cx| {
                let running = running.downgrade();
                let project = project.clone();
                window.spawn(cx, async move |cx| {
                    RunningState::run_before_selected_java_debug(
                        &running,
                        &project,
                        worktree_id,
                        cx,
                    )
                    .await
                })
            })
            .expect("start the blocking setup Task");
        let deadline = std::time::Instant::now() + Duration::from_secs(8);
        while !root_path.join("before-blocking").exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "the setup Task never started"
            );
            cx.background_executor
                .timer(Duration::from_millis(10))
                .await;
            cx.run_until_parked();
        }
        let handle = running.read_with(cx, |running, _| {
            running
                .selected_java_debug_hooks
                .as_ref()
                .and_then(|hooks| hooks.active_hook.clone())
                .expect("the Debug session owns its active setup Task")
        });
        running.update(cx, |running, cx| running.shutdown(cx));
        let completion = futures::future::select(
            pending,
            cx.background_executor.timer(Duration::from_secs(8)),
        )
        .await;
        assert!(
            matches!(completion, futures::future::Either::Left((Err(_), _))),
            "Stop must cancel the pending Debug"
        );
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::Cancelled { .. }
        ));
        assert!(!armed(cx));
        assert!(!root_path.join("after-stop").exists());

        save_tasks(
            r#"[
            {"label":"failed-setup","command":"/usr/bin/false","tags":["before-debug"]},
            {"label":"finish-test","command":"/usr/bin/touch","args":["after-launch-failure"],"tags":["after-debug"]}
        ]"#,
            cx,
        );
        let (session, running) = create_session(cx);
        let mut async_window = window
            .update(cx, |_, window, cx| window.to_async(cx))
            .expect("async Debug window");
        let error = RunningState::run_before_selected_java_debug(
            &running.downgrade(),
            &project,
            worktree_id,
            &mut async_window,
        )
        .await
        .expect_err("the failed setup must prevent launch");
        assert!(
            error
                .to_string()
                .contains("before-debug Task 'failed-setup' failed")
        );
        session.update(cx, |session, cx| session.shutdown(cx)).await;
        cx.run_until_parked();
        assert!(!armed(cx));
        assert!(!root_path.join("after-launch-failure").exists());
    }

    #[cfg(all(unix, feature = "jvm-workflow-tests"))]
    #[gpui::test]
    #[expect(clippy::result_large_err)]
    async fn restricting_java_debug_cancels_preparing_and_attached_tasks(cx: &mut TestAppContext) {
        use dap::{
            StartDebuggingRequestArguments, StartDebuggingRequestArgumentsRequest,
            adapters::DebugAdapterBinary,
        };
        use futures::channel::oneshot;
        use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};

        cx.executor().allow_parking();
        init_test(cx);
        cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
        let root = std::env::temp_dir();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(&root, json!({"pom.xml": "<project/>"}))
            .await;
        let project = Project::test_with_worktree_trust(fs, [root.as_path()], cx).await;
        let window = init_test_workspace(&project, cx).await;
        let (store, worktrees, worktree, environment) = project.read_with(cx, |project, cx| {
            (
                project.dap_store(),
                project.worktree_store(),
                project.visible_worktrees(cx).next().expect("Java worktree"),
                project.environment().clone(),
            )
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let trust = cx.read(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust owner"));
        let paths = || collections::HashSet::from_iter([PathTrust::Worktree(worktree_id)]);
        trust.update(cx, |trust, cx| trust.trust(&worktrees, paths(), cx));
        let ordinary = start_debug_session(&window, cx, |_| {}).expect("Unrelated Debug session");
        let _intercept = project::debugger::test::intercept_debug_sessions(cx, |client| {
            client.on_request::<dap::requests::Attach, _>(|_, _| Ok(()));
        });

        for hold_preparation in [true, false] {
            trust.update(cx, |trust, cx| trust.trust(&worktrees, paths(), cx));
            let session = store.update(cx, |store, cx| {
                store.new_session(
                    Some("Selected Java test".into()),
                    "Java".into(),
                    SharedTaskContext::default(),
                    None,
                    Default::default(),
                    cx,
                )
            });
            let running = window
                .update(cx, |multi, window, cx| {
                    let workspace = multi.workspace().downgrade();
                    cx.new(|cx| {
                        RunningState::new(
                            session.clone(),
                            project.clone(),
                            workspace,
                            None,
                            None,
                            Axis::Horizontal,
                            window,
                            cx,
                        )
                    })
                })
                .expect("Debug window");
            let held_environment = hold_preparation.then(|| {
                let (entered, entered_receiver) = oneshot::channel();
                let (resume, resume_receiver) = oneshot::channel();
                environment.update(cx, |environment, _| {
                    environment
                        .delay_next_local_directory_environment_for_test(
                            Arc::from(root.as_path()),
                            entered,
                            resume_receiver,
                        )
                        .expect("Hold terminal preparation");
                });
                (entered_receiver, resume)
            });
            let (creation, handle) = running.update(cx, |running, cx| {
                let creation = running.create_selected_java_test_terminal(
                    &project,
                    SpawnInTerminal {
                        cwd: Some(root.clone()),
                        command: Some("/bin/sh".into()),
                        args: vec!["-c".into(), "read answer".into()],
                        ..Default::default()
                    },
                    None,
                    cx,
                );
                let handle = running.selected_java_test_task.clone().expect("Owned Task");
                (creation, handle)
            });

            if let Some((entered, resume)) = held_environment {
                entered.await.expect("Terminal preparation entered");
                trust.update(cx, |trust, cx| {
                    trust.restrict(worktrees.downgrade(), paths(), cx)
                });
                cx.run_until_parked();
                resume.send(None).expect("Resume terminal preparation");
                assert!(
                    creation.await.is_err(),
                    "Restricted preparation spawned a terminal"
                );
            } else {
                let terminal = creation.await.expect("Running selected test terminal");
                session
                    .update(cx, |session, cx| {
                        session.boot(
                            DebugAdapterBinary {
                                command: Some("command".into()),
                                arguments: Vec::new(),
                                envs: Default::default(),
                                cwd: None,
                                connection: None,
                                request_args: StartDebuggingRequestArguments {
                                    request: StartDebuggingRequestArgumentsRequest::Attach,
                                    configuration: json!({"request": "attach"}),
                                },
                            },
                            worktree.clone(),
                            store.downgrade(),
                            cx,
                        )
                    })
                    .await
                    .expect("Mock adapter attached");
                assert!(session.read_with(cx, |session, _| session.is_started()));
                trust.update(cx, |trust, cx| {
                    trust.restrict(worktrees.downgrade(), paths(), cx)
                });
                let status = terminal
                    .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                    .await
                    .expect("Owned terminal process exited");
                assert!(!status.success());
            }
            assert!(matches!(
                handle.state(),
                task::StructuredTaskState::Cancelled {
                    termination_confirmed: true,
                    ..
                }
            ));
            assert!(session.read_with(cx, |session, _| session.is_terminated()));
            assert!(!ordinary.read_with(cx, |session, _| session.is_terminated()));
        }
        store
            .update(cx, |store, cx| store.shutdown_sessions(cx))
            .await;
    }

    struct HeldNamedDebugAdapter {
        resume: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait(?Send)]
    impl dap::adapters::DebugAdapter for HeldNamedDebugAdapter {
        fn name(&self) -> dap::adapters::DebugAdapterName {
            "held-named-debug".into()
        }

        fn dap_schema(&self) -> Value {
            Value::Null
        }

        async fn config_from_zed_format(&self, scenario: ZedDebugConfig) -> Result<DebugScenario> {
            dap::adapters::DebugAdapter::config_from_zed_format(
                &dap::adapters::FakeAdapter::new(),
                scenario,
            )
            .await
        }

        async fn get_binary(
            &self,
            delegate: &Arc<dyn dap::adapters::DapDelegate>,
            definition: &DebugTaskDefinition,
            installed: Option<PathBuf>,
            arguments: Option<Vec<String>>,
            environment: Option<HashMap<String, String>>,
            cx: &mut gpui::AsyncApp,
        ) -> Result<dap::adapters::DebugAdapterBinary> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let resume = self
                .resume
                .lock()
                .map_err(|_| anyhow!("debug fixture gate unavailable"))?
                .take();
            if let Some(resume) = resume {
                resume
                    .await
                    .context("debug fixture continuation was cancelled")?;
            }
            dap::adapters::DebugAdapter::get_binary(
                &dap::adapters::FakeAdapter::new(),
                delegate,
                definition,
                installed,
                arguments,
                environment,
                cx,
            )
            .await
        }
    }

    fn set_named_debug_opt_in(enabled: bool, cx: &mut App) {
        use gpui::UpdateGlobal as _;
        settings::SettingsStore::update_global(cx, |settings, _| {
            settings.register_setting::<feature_flags::FeatureFlagsSettings>();
        });
        feature_flags::FeatureFlagStore::init(cx);
        settings::SettingsStore::update_global(cx, |settings, cx| {
            settings.update_user_settings(cx, |content| {
                content.feature_flags.get_or_insert_default().insert(
                    "rust-named-cargo-presets-experimental".into(),
                    if enabled { "on" } else { "off" }.into(),
                );
            });
        });
    }

    #[gpui::test]
    async fn named_debug_opt_in_revalidates_real_binary_continuation_and_recovers(
        executor: BackgroundExecutor,
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use task::TaskTemplate;

        init_test(cx);
        cx.executor().allow_parking();
        let filesystem = FakeFs::new(executor);
        filesystem
            .insert_tree(path!("/project"), json!({"main.rs": "fn main() {}"}))
            .await;
        let project = Project::test(filesystem, [path!("/project").as_ref()], cx).await;
        let store = project.read_with(cx, |project, _| project.dap_store());
        let worktree = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("original worktree")
        });
        let definition = DebugTaskDefinition {
            label: "named debug continuation".into(),
            adapter: "held-named-debug".into(),
            config: json!({"request":"launch"}),
            tcp_connection: None,
        };
        let template = TaskTemplate {
            required_experimental_features: vec!["rust-named-cargo-presets-experimental".into()],
            label: "original named Build".into(),
            command: "cargo".into(),
            ..Default::default()
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let clients = Arc::new(AtomicUsize::new(0));
        let restarts = Arc::new(AtomicUsize::new(0));
        let _interception = project::debugger::test::intercept_debug_sessions(cx, {
            let clients = clients.clone();
            let restarts = restarts.clone();
            move |client| {
                clients.fetch_add(1, Ordering::SeqCst);
                client.on_request_ext::<dap::requests::Initialize, _>(|_, _| {
                    dap::transport::RequestHandling::Respond(Ok(dap::Capabilities {
                        supports_restart_request: Some(true),
                        ..Default::default()
                    }))
                });
                let restarts = restarts.clone();
                client.on_request_ext::<dap::requests::Restart, _>(move |_, _| {
                    restarts.fetch_add(1, Ordering::SeqCst);
                    dap::transport::RequestHandling::Respond(Ok(()))
                });
            }
        });
        let (resume, receiver) = futures::channel::oneshot::channel();
        cx.update(|cx| {
            use feature_flags::FeatureFlagAppExt as _;
            cx.update_flags(true, vec!["rust-named-cargo-presets-experimental".into()]);
            DapRegistry::global(cx).add_adapter(Arc::new(HeldNamedDebugAdapter {
                resume: std::sync::Mutex::new(Some(receiver)),
                calls: calls.clone(),
            }));
        });
        let new_session = |cx: &mut TestAppContext, marked: bool| {
            store.update(cx, |store, cx| {
                let session = store.new_session(
                    Some(definition.label.clone()),
                    definition.adapter.clone(),
                    SharedTaskContext::default(),
                    None,
                    Default::default(),
                    cx,
                );
                if marked {
                    session
                        .update(cx, |session, _| {
                            session.set_required_experimental_features(
                                template.required_experimental_features.clone(),
                            )
                        })
                        .expect("finite original Task requirement DATA");
                }
                session
            })
        };
        let disabled = new_session(cx, true);
        let result = store
            .update(cx, |store, cx| {
                store.boot_session(disabled.clone(), definition.clone(), worktree.clone(), cx)
            })
            .await;
        assert!(
            result.is_err(),
            "staff/server cannot grant named debug continuation"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(clients.load(Ordering::SeqCst), 0);

        cx.update(|cx| set_named_debug_opt_in(true, cx));
        let expired = new_session(cx, true);
        let pending = store.update(cx, |store, cx| {
            store.boot_session(expired.clone(), definition.clone(), worktree.clone(), cx)
        });
        cx.run_until_parked();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "actual binary preparation has reached held await"
        );
        assert!(expired.read_with(cx, |session, _| session.adapter_client().is_none()));
        cx.update(|cx| set_named_debug_opt_in(false, cx));
        cx.run_until_parked();
        resume
            .send(())
            .expect("release genuine original binary continuation");
        let deadline = cx.background_executor.timer(Duration::from_secs(10));
        match futures::future::select(pending, deadline).await {
            futures::future::Either::Left((result, _)) => assert!(
                result.is_err(),
                "revoked original continuation must refuse before adapter boot"
            ),
            futures::future::Either::Right(_) => panic!(
                "named debug refusal did not complete within original bounded fixture deadline"
            ),
        }
        assert_eq!(clients.load(Ordering::SeqCst), 0);
        assert!(expired.read_with(cx, |session, _| session.adapter_client().is_none()));
        assert!(expired.read_with(cx, |session, _| session.is_terminated()));

        let ordinary = new_session(cx, false);
        store
            .update(cx, |store, cx| {
                store.boot_session(ordinary.clone(), definition.clone(), worktree.clone(), cx)
            })
            .await
            .expect("ordinary authored Debug remains available while named opt-in is off");
        assert_eq!(clients.load(Ordering::SeqCst), 1);
        assert!(ordinary.read_with(cx, |session, _| {
            session.required_experimental_features().is_empty()
        }));

        cx.update(|cx| set_named_debug_opt_in(true, cx));
        let fresh = new_session(cx, true);
        store
            .update(cx, |store, cx| {
                store.boot_session(fresh.clone(), definition, worktree, cx)
            })
            .await
            .expect("fresh explicit local opt-in reaches actual original adapter boot");
        assert_eq!(clients.load(Ordering::SeqCst), 2);
        assert_eq!(
            fresh.read_with(cx, |session, _| session
                .required_experimental_features()
                .to_vec()),
            template.required_experimental_features
        );
        let workspace = init_test_workspace(&project, cx).await;
        let debug_session = workspace
            .update(cx, |multi, window, cx| {
                let panel = multi
                    .workspace()
                    .read(cx)
                    .panel::<DebugPanel>(cx)
                    .expect("actual debug panel");
                let fresh = fresh.clone();
                cx.spawn_in(window, async move |_, cx| {
                    DebugPanel::register_session(panel.downgrade(), fresh, true, cx).await
                })
            })
            .expect("actual UI registration task")
            .await
            .expect("existing running session view");
        let running = debug_session.read_with(cx, |session, _| session.running_state().clone());
        fresh.update(cx, |session, cx| session.restart(None, cx));
        cx.run_until_parked();
        assert_eq!(
            restarts.load(Ordering::SeqCst),
            1,
            "opted-in actual deferred restart reaches adapter receiver"
        );
        fresh.update(cx, |session, cx| session.restart(None, cx));
        cx.update(|cx| set_named_debug_opt_in(false, cx));
        cx.run_until_parked();
        assert_eq!(
            restarts.load(Ordering::SeqCst),
            1,
            "revocation before queued continuation blocks second restart"
        );
        let (sender, mut receiver) = futures::channel::mpsc::channel(1);
        let request = RunInTerminalRequestArguments {
            kind: None,
            title: None,
            cwd: String::new(),
            args: vec!["must-not-launch".into()],
            env: None,
            args_can_be_interpreted_by_shell: None,
        };
        workspace
            .update(cx, |_, window, cx| {
                running.update(cx, |running, cx| {
                    running.handle_run_in_terminal(&request, sender, window, cx)
                })
            })
            .expect("actual terminal request continuation")
            .await
            .expect("denial sent to original requester");
        use futures::StreamExt as _;
        assert!(
            receiver
                .next()
                .await
                .expect("actual RunInTerminal response")
                .is_err()
        );
        assert!(
            running.read_with(cx, |running, cx| running
                .debug_terminal
                .read(cx)
                .terminal
                .is_none()),
            "revoked reverse request creates no terminal"
        );
        assert!(
            ordinary.read_with(cx, |session, _| !session.is_terminated()),
            "ordinary authored session is not retired by named opt-out"
        );
        ordinary
            .update(cx, |session, cx| session.shutdown(cx))
            .await;
        fresh.update(cx, |session, cx| session.shutdown(cx)).await;
    }

    #[gpui::test]
    async fn stale_subview_host_during_tab_drop_does_not_read_updating_source_pane(
        executor: BackgroundExecutor,
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let fs = FakeFs::new(executor);
        fs.insert_tree(
            path!("/project"),
            json!({
                "main.rs": "fn main() {}",
            }),
        )
        .await;

        let project = Project::test(fs, [path!("/project").as_ref()], cx).await;
        let workspace = init_test_workspace(&project, cx).await;
        let cx = &mut VisualTestContext::from_window(*workspace, cx);

        start_debug_session(&workspace, cx, |_| {}).expect("debug session starts");
        cx.run_until_parked();

        let running_state = workspace
            .update(cx, |multi_workspace, _window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let debug_panel = workspace.panel::<DebugPanel>(cx).expect("debug panel");
                    let active_session = debug_panel
                        .read(cx)
                        .active_session()
                        .expect("active debug session");
                    active_session.read(cx).running_state().clone()
                })
            })
            .expect("workspace update succeeds");

        let (source_pane, stale_host_pane) = running_state.read_with(cx, |running_state, _| {
            let panes = running_state.panes.panes();
            let mut panes = panes.into_iter();
            let source_pane = panes.next().expect("source pane").clone();
            let stale_host_pane = panes.next().expect("stale host pane").clone();
            (source_pane, stale_host_pane)
        });

        let dragged_tab = {
            let source_pane_entity = source_pane.clone();
            source_pane.read_with(cx, |source_pane, _| {
                let item = source_pane
                    .item_for_index(0)
                    .expect("source pane contains debugger subview")
                    .boxed_clone();
                DraggedTab {
                    pane: source_pane_entity,
                    item,
                    ix: 0,
                    detail: 0,
                    is_active: true,
                }
            })
        };

        let active_subview = source_pane.read_with(cx, |source_pane, _| {
            source_pane
                .active_item()
                .and_then(|item| item.downcast::<SubView>())
                .expect("active item is a debugger subview")
        });
        active_subview.update(cx, |subview, _| {
            subview.set_host_pane(stale_host_pane.downgrade());
        });

        source_pane.update_in(cx, |source_pane, window, cx| {
            source_pane.handle_tab_drop(
                &dragged_tab,
                source_pane.active_item_index(),
                true,
                window,
                cx,
            );
        });
    }
}
