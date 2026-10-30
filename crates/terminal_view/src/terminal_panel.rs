use std::{cmp, path::PathBuf, process::ExitStatus, sync::Arc, time::Duration};

use crate::{
    TerminalView, default_working_directory,
    persistence::{
        SerializedItems, SerializedTerminalPanel, deserialize_terminal_panel, serialize_pane_group,
    },
};
use breadcrumbs::Breadcrumbs;
use collections::HashMap;
use db::kvp::KeyValueStore;
use futures::{channel::oneshot, future::join_all};
use gpui::{
    Action, Anchor, App, AsyncApp, AsyncWindowContext, Context, Entity, EventEmitter, FocusHandle,
    Focusable, IntoElement, ParentElement, Pixels, Render, Styled, Task, TaskExt, WeakEntity,
    Window, actions,
};
use itertools::Itertools;
use project::{
    Fs, Project, task_store::PendingExperimentalTaskLaunch, terminals::PreparedNativeTerminalLaunch,
};

use settings::{Settings, TerminalDockPosition};
use task::{
    RevealStrategy, RevealTarget, Shell, ShellBuilder, SpawnInTerminal, StructuredTaskHandle,
    TaskId,
};
use terminal::{Terminal, terminal_settings::TerminalSettings};
use ui::{
    ButtonLike, Clickable, ContextMenu, FluentBuilder, PopoverMenu, SplitButton, Toggleable,
    Tooltip, prelude::*,
};
use util::{ResultExt, TryFutureExt};
use workspace::{
    ActivateNextPane, ActivatePane, ActivatePaneDown, ActivatePaneLeft, ActivatePaneRight,
    ActivatePaneUp, ActivatePreviousPane, DraggedTab, ItemId, MoveItemToPane,
    MoveItemToPaneInDirection, MovePaneDown, MovePaneLeft, MovePaneRight, MovePaneUp, Pane,
    PaneGroup, PreparedTaskLaunch, SplitDirection, SplitDown, SplitLeft, SplitMode, SplitRight,
    SplitUp, SwapPaneDown, SwapPaneLeft, SwapPaneRight, SwapPaneUp, TaskLaunchCompletion,
    TaskLaunchPreparation, ToggleZoom, Workspace,
    dock::{DockPosition, Panel, PanelEvent, PanelHandle},
    item::SerializableItem,
    move_active_item, pane,
};

use anyhow::{Result, anyhow};
#[cfg(feature = "agentic")]
use zed_actions::assistant::InlineAssist;

const TERMINAL_PANEL_KEY: &str = "TerminalPanel";

actions!(
    terminal_panel,
    [
        /// Toggles the terminal panel.
        Toggle,
        /// Toggles focus on the terminal panel.
        ToggleFocus
    ]
);

pub fn init(cx: &mut App) {
    cx.observe_new(
        |workspace: &mut Workspace, _window, _: &mut Context<Workspace>| {
            workspace.register_action(TerminalPanel::new_terminal);
            workspace.register_action(TerminalPanel::open_terminal);
            workspace.register_action(|workspace, _: &ToggleFocus, window, cx| {
                if is_enabled_in_workspace(workspace, cx) {
                    workspace.toggle_panel_focus::<TerminalPanel>(window, cx);
                }
            });
            workspace.register_action(|workspace, _: &Toggle, window, cx| {
                if is_enabled_in_workspace(workspace, cx) {
                    if !workspace.toggle_panel_focus::<TerminalPanel>(window, cx) {
                        workspace.close_panel::<TerminalPanel>(window, cx);
                    }
                }
            });
        },
    )
    .detach();
}

pub struct TerminalPanel {
    pub(crate) active_pane: Entity<Pane>,
    pub(crate) center: PaneGroup,
    focus_handle: FocusHandle,
    fs: Arc<dyn Fs>,
    workspace: WeakEntity<Workspace>,
    pending_serialization: Task<Option<()>>,
    pending_terminals_to_add: usize,
    deferred_tasks: HashMap<TaskId, Task<()>>,
    #[cfg(feature = "agentic")]
    assistant_enabled: bool,
    active: bool,
}

struct PendingTerminalCreation {
    panel: Option<WeakEntity<TerminalPanel>>,
    context: AsyncApp,
}

impl Drop for PendingTerminalCreation {
    fn drop(&mut self) {
        let Some(panel) = self.panel.take() else {
            return;
        };
        // A creation future may be destroyed inside an entity update; defer its
        // counter retirement to avoid updating that same entity reentrantly.
        self.context
            .spawn(async move |cx| {
                if let Err(error) = panel.update(cx, |panel, _| {
                    panel.pending_terminals_to_add =
                        panel.pending_terminals_to_add.saturating_sub(1);
                }) {
                    log::debug!("Pending terminal creation owner expired: {error:#}");
                }
            })
            .detach();
    }
}

struct PreparedTerminalTask {
    terminal: Task<Result<WeakEntity<Terminal>>>,
    completion: Option<TaskLaunchCompletion>,
}

enum TaskTerminalDestination {
    New,
    Replace {
        item_index: usize,
        pane: Entity<Pane>,
        terminal: Entity<TerminalView>,
    },
}

impl TerminalPanel {
    pub fn new(workspace: &Workspace, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let project = workspace.project();
        let pane = new_terminal_pane(workspace.weak_handle(), project.clone(), false, window, cx);
        let center = PaneGroup::new(pane.clone());
        let terminal_panel = Self {
            center,
            active_pane: pane,
            focus_handle: cx.focus_handle(),
            fs: workspace.app_state().fs.clone(),
            workspace: workspace.weak_handle(),
            pending_serialization: Task::ready(None),
            pending_terminals_to_add: 0,
            deferred_tasks: HashMap::default(),
            #[cfg(feature = "agentic")]
            assistant_enabled: false,
            active: false,
        };
        terminal_panel.apply_tab_bar_buttons(&terminal_panel.active_pane, cx);
        terminal_panel
    }

    #[cfg(feature = "agentic")]
    pub fn set_assistant_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.assistant_enabled = enabled;
        for pane in self.center.panes() {
            self.apply_tab_bar_buttons(pane, cx);
        }
    }

    pub(crate) fn apply_tab_bar_buttons(
        &self,
        terminal_pane: &Entity<Pane>,
        cx: &mut Context<Self>,
    ) {
        #[cfg(feature = "agentic")]
        let assistant_enabled = self.assistant_enabled;
        terminal_pane.update(cx, |pane, cx| {
            pane.set_render_tab_bar_buttons(cx, move |pane, window, cx| {
                let split_context = pane
                    .active_item()
                    .and_then(|item| item.downcast::<TerminalView>())
                    .map(|terminal_view| terminal_view.read(cx).focus_handle.clone());
                let has_focused_rename_editor = pane
                    .active_item()
                    .and_then(|item| item.downcast::<TerminalView>())
                    .is_some_and(|view| view.read(cx).rename_editor_is_focused(window, cx));
                if !pane.has_focus(window, cx)
                    && !pane.context_menu_focused(window, cx)
                    && !has_focused_rename_editor
                {
                    return (None, None);
                }
                let focus_handle = pane.focus_handle(cx);
                let right_children = h_flex().gap(DynamicSpacing::Base02.rems(cx)).child(
                    PopoverMenu::new("terminal-tab-bar-popover-menu")
                        .trigger_with_tooltip(
                            IconButton::new("plus", IconName::Plus).icon_size(IconSize::Small),
                            Tooltip::text("New…"),
                        )
                        .anchor(Anchor::TopRight)
                        .with_handle(pane.new_item_context_menu_handle.clone())
                        .menu(move |window, cx| {
                            let focus_handle = focus_handle.clone();
                            let menu = ContextMenu::build(window, cx, |menu, _, _| {
                                menu.context(focus_handle.clone())
                                    .action(
                                        "New Terminal",
                                        workspace::NewTerminal::default().boxed_clone(),
                                    )
                                    // We want the focus to go back to terminal panel once task modal is dismissed,
                                    // hence we focus that first. Otherwise, we'd end up without a focused element, as
                                    // context menu will be gone the moment we spawn the modal.
                                    .action("Spawn Task", zed_actions::Spawn::modal().boxed_clone())
                            });

                            Some(menu)
                        }),
                );
                #[cfg(feature = "agentic")]
                let right_children = right_children.when(assistant_enabled, |this| {
                    this.when_some(split_context.clone(), |this, focus_handle| {
                        this.child(InlineAssistTabBarButton { focus_handle })
                    })
                });
                let right_children = right_children
                    .child(
                        PopoverMenu::new("terminal-pane-tab-bar-split")
                            .trigger_with_tooltip(
                                IconButton::new("terminal-pane-split", IconName::Split)
                                    .icon_size(IconSize::Small),
                                Tooltip::text("Split Pane"),
                            )
                            .anchor(Anchor::TopRight)
                            .with_handle(pane.split_item_context_menu_handle.clone())
                            .menu({
                                move |window, cx| {
                                    ContextMenu::build(window, cx, |menu, _, _| {
                                        menu.when_some(
                                            split_context.clone(),
                                            |menu, split_context| menu.context(split_context),
                                        )
                                        .action("Split Right", SplitRight::default().boxed_clone())
                                        .action("Split Left", SplitLeft::default().boxed_clone())
                                        .action("Split Up", SplitUp::default().boxed_clone())
                                        .action("Split Down", SplitDown::default().boxed_clone())
                                    })
                                    .into()
                                }
                            }),
                    )
                    .child({
                        let zoomed = pane.is_zoomed();
                        IconButton::new("toggle_zoom", IconName::Maximize)
                            .icon_size(IconSize::Small)
                            .toggle_state(zoomed)
                            .selected_icon(IconName::Minimize)
                            .on_click(cx.listener(|pane, _, window, cx| {
                                pane.toggle_zoom(&workspace::ToggleZoom, window, cx);
                            }))
                            .tooltip(move |_window, cx| {
                                Tooltip::for_action(
                                    if zoomed { "Zoom Out" } else { "Zoom In" },
                                    &ToggleZoom,
                                    cx,
                                )
                            })
                    })
                    .into_any_element()
                    .into();
                (None, right_children)
            });
        });
    }

    fn serialization_key(workspace: &Workspace) -> Option<String> {
        workspace
            .database_id()
            .map(|id| i64::from(id).to_string())
            .or(workspace.session_id())
            .map(|id| format!("{:?}-{:?}", TERMINAL_PANEL_KEY, id))
    }

    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> Result<Entity<Self>> {
        let mut terminal_panel = None;

        if let Some((database_id, serialization_key, kvp)) = workspace
            .read_with(&cx, |workspace, cx| {
                workspace
                    .database_id()
                    .zip(TerminalPanel::serialization_key(workspace))
                    .map(|(id, key)| (id, key, KeyValueStore::global(cx)))
            })
            .ok()
            .flatten()
            && let Some(serialized_panel) = cx
                .background_spawn(async move { kvp.read_kvp(&serialization_key) })
                .await
                .log_err()
                .flatten()
                .map(|panel| serde_json::from_str::<SerializedTerminalPanel>(&panel))
                .transpose()
                .log_err()
                .flatten()
            && let Ok(serialized) = workspace
                .update_in(&mut cx, |workspace, window, cx| {
                    deserialize_terminal_panel(
                        workspace.weak_handle(),
                        workspace.project().clone(),
                        database_id,
                        serialized_panel,
                        window,
                        cx,
                    )
                })?
                .await
        {
            terminal_panel = Some(serialized);
        }

        let terminal_panel = if let Some(panel) = terminal_panel {
            panel
        } else {
            workspace.update_in(&mut cx, |workspace, window, cx| {
                cx.new(|cx| TerminalPanel::new(workspace, window, cx))
            })?
        };

        if let Some(workspace) = workspace.upgrade() {
            workspace.update(&mut cx, |workspace, _| {
                workspace.set_terminal_provider(TerminalProvider(terminal_panel.clone()))
            });
        }

        // Since panels/docks are loaded outside from the workspace, we cleanup here, instead of through the workspace.
        if let Some(workspace) = workspace.upgrade() {
            let cleanup_task = workspace.update_in(&mut cx, |workspace, window, cx| {
                let alive_item_ids = terminal_panel
                    .read(cx)
                    .center
                    .panes()
                    .into_iter()
                    .flat_map(|pane| pane.read(cx).items())
                    .map(|item| item.item_id().as_u64() as ItemId)
                    .collect();
                workspace.database_id().map(|workspace_id| {
                    TerminalView::cleanup(workspace_id, alive_item_ids, window, cx)
                })
            })?;
            if let Some(task) = cleanup_task {
                task.await.log_err();
            }
        }

        if let Some(workspace) = workspace.upgrade() {
            let should_focus = workspace
                .update_in(&mut cx, |workspace, window, cx| {
                    workspace.active_item(cx).is_none()
                        && workspace
                            .is_dock_at_position_open(terminal_panel.position(window, cx), cx)
                })
                .unwrap_or(false);

            if should_focus {
                terminal_panel
                    .update_in(&mut cx, |panel, window, cx| {
                        panel.active_pane.update(cx, |pane, cx| {
                            pane.focus_active_item(window, cx);
                        });
                    })
                    .ok();
            }
        }
        Ok(terminal_panel)
    }

    fn handle_pane_event(
        &mut self,
        pane: &Entity<Pane>,
        event: &pane::Event,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            pane::Event::ActivateItem { .. } => self.serialize(cx),
            pane::Event::RemovedItem { .. } => self.serialize(cx),
            pane::Event::Remove { focus_on_pane } => {
                let pane_count_before_removal = self.center.panes().len();
                let _removal_result = self.center.remove(pane, cx);
                if pane_count_before_removal == 1 {
                    self.center.first_pane().update(cx, |pane, cx| {
                        pane.set_zoomed(false, cx);
                    });
                    cx.emit(PanelEvent::Close);
                } else if let Some(focus_on_pane) =
                    focus_on_pane.as_ref().or_else(|| self.center.panes().pop())
                {
                    focus_on_pane.focus_handle(cx).focus(window, cx);
                }
            }
            pane::Event::ZoomIn => {
                for pane in self.center.panes() {
                    pane.update(cx, |pane, cx| {
                        pane.set_zoomed(true, cx);
                    })
                }
                cx.emit(PanelEvent::ZoomIn);
                cx.notify();
            }
            pane::Event::ZoomOut => {
                for pane in self.center.panes() {
                    pane.update(cx, |pane, cx| {
                        pane.set_zoomed(false, cx);
                    })
                }
                cx.emit(PanelEvent::ZoomOut);
                cx.notify();
            }
            pane::Event::AddItem { item } => {
                if let Some(workspace) = self.workspace.upgrade() {
                    workspace.update(cx, |workspace, cx| {
                        item.added_to_pane(workspace, pane.clone(), window, cx)
                    })
                }
                self.serialize(cx);
            }
            &pane::Event::Split { direction, mode } => {
                match mode {
                    SplitMode::ClonePane | SplitMode::EmptyPane => {
                        let clone = matches!(mode, SplitMode::ClonePane);
                        let new_pane = self.new_pane_with_active_terminal(clone, window, cx);
                        let pane = pane.clone();
                        cx.spawn_in(window, async move |panel, cx| {
                            let Some(new_pane) = new_pane.await else {
                                return;
                            };
                            panel
                                .update_in(cx, |panel, window, cx| {
                                    panel.center.split(&pane, &new_pane, direction, cx);
                                    window.focus(&new_pane.focus_handle(cx), cx);
                                })
                                .ok();
                        })
                        .detach();
                    }
                    SplitMode::MovePane => {
                        let Some(item) =
                            pane.update(cx, |pane, cx| pane.take_active_item(window, cx))
                        else {
                            return;
                        };
                        let Ok(project) = self
                            .workspace
                            .update(cx, |workspace, _| workspace.project().clone())
                        else {
                            return;
                        };
                        let new_pane =
                            new_terminal_pane(self.workspace.clone(), project, false, window, cx);
                        new_pane.update(cx, |pane, cx| {
                            pane.add_item(item, true, true, None, window, cx);
                        });
                        self.center.split(&pane, &new_pane, direction, cx);
                        window.focus(&new_pane.focus_handle(cx), cx);
                    }
                };
            }
            pane::Event::Focus => {
                self.active_pane = pane.clone();
            }
            pane::Event::ItemPinned | pane::Event::ItemUnpinned => {
                self.serialize(cx);
            }

            _ => {}
        }
    }

    fn new_pane_with_active_terminal(
        &mut self,
        clone: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Pane>>> {
        let Some(workspace) = self.workspace.upgrade() else {
            return Task::ready(None);
        };
        let workspace = workspace.read(cx);
        let database_id = workspace.database_id();
        let weak_workspace = self.workspace.clone();
        let project = workspace.project().clone();
        let active_pane = &self.active_pane;
        let terminal_view = if clone {
            active_pane
                .read(cx)
                .active_item()
                .and_then(|item| item.downcast::<TerminalView>())
        } else {
            None
        };
        let working_directory = if clone {
            terminal_view
                .as_ref()
                .and_then(|terminal_view| {
                    terminal_view
                        .read(cx)
                        .terminal()
                        .read(cx)
                        .working_directory()
                })
                .or_else(|| default_working_directory(workspace, cx))
        } else {
            default_working_directory(workspace, cx)
        };

        let is_zoomed = if clone {
            active_pane.read(cx).is_zoomed()
        } else {
            false
        };
        cx.spawn_in(window, async move |panel, cx| {
            let terminal = project
                .update(cx, |project, cx| match terminal_view {
                    Some(view) => project.clone_terminal(
                        &view.read(cx).terminal.clone(),
                        cx,
                        working_directory,
                    ),
                    None => project.create_terminal_shell(working_directory, cx),
                })
                .await
                .log_err()?;

            panel
                .update_in(cx, move |terminal_panel, window, cx| {
                    let terminal_view = Box::new(cx.new(|cx| {
                        TerminalView::new(
                            terminal.clone(),
                            weak_workspace.clone(),
                            database_id,
                            project.downgrade(),
                            window,
                            cx,
                        )
                    }));
                    let pane = new_terminal_pane(weak_workspace, project, is_zoomed, window, cx);
                    terminal_panel.apply_tab_bar_buttons(&pane, cx);
                    pane.update(cx, |pane, cx| {
                        pane.add_item(terminal_view, true, true, None, window, cx);
                    });
                    Some(pane)
                })
                .ok()
                .flatten()
        })
    }

    pub fn open_terminal(
        workspace: &mut Workspace,
        action: &workspace::OpenTerminal,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let Some(terminal_panel) = workspace.panel::<Self>(cx) else {
            return;
        };

        terminal_panel
            .update(cx, |panel, cx| {
                if action.local {
                    panel.add_local_terminal_shell(RevealStrategy::Always, window, cx)
                } else {
                    panel.add_terminal_shell(
                        Some(action.working_directory.clone()),
                        RevealStrategy::Always,
                        window,
                        cx,
                    )
                }
            })
            .detach_and_log_err(cx);
    }

    pub fn spawn_task(
        &mut self,
        task: &SpawnInTerminal,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        let prepared = self.spawn_task_internal(task, None, None, window, cx);
        cx.spawn(async move |_, _| prepared.await?.terminal.await)
    }

    pub fn spawn_task_with_structured_handle(
        &mut self,
        task: &SpawnInTerminal,
        structured_task: StructuredTaskHandle,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        let prepared = self.spawn_task_internal(task, Some(structured_task), None, window, cx);
        cx.spawn(async move |_, _| prepared.await?.terminal.await)
    }

    fn spawn_task_internal(
        &mut self,
        task: &SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        preparation: Option<TaskLaunchPreparation>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<PreparedTerminalTask>> {
        let Some(workspace) = self.workspace.upgrade() else {
            return Task::ready(Err(anyhow!("failed to read workspace")));
        };
        let project = workspace.read(cx).project().read(cx);
        if project.is_via_collab() {
            return Task::ready(Err(anyhow!("cannot spawn tasks as a guest")));
        }
        let remote_client = project.remote_client();
        let is_windows = project.path_style(cx).is_windows();
        let remote_shell = remote_client
            .as_ref()
            .and_then(|remote_client| remote_client.read(cx).shell());
        let shell = if let Some(remote_shell) = remote_shell
            && task.shell == Shell::System
        {
            Shell::Program(remote_shell)
        } else {
            task.shell.clone()
        };
        let pending_launch = match PendingExperimentalTaskLaunch::new(task, cx) {
            Ok(pending_launch) => pending_launch,
            Err(error) => return Task::ready(Err(error)),
        };
        let task = task.clone();
        if task.allow_concurrent_runs && task.use_new_terminal {
            return self.spawn_eligible_task(
                task,
                structured_task,
                preparation,
                pending_launch,
                shell,
                is_windows,
                TaskTerminalDestination::New,
                window,
                cx,
            );
        }
        let mut terminals_for_task = self.terminals_for_task(&task.full_label, cx);
        let Some((existing_item_index, task_pane, existing_terminal)) = terminals_for_task.pop()
        else {
            return self.spawn_eligible_task(
                task,
                structured_task,
                preparation,
                pending_launch,
                shell,
                is_windows,
                TaskTerminalDestination::New,
                window,
                cx,
            );
        };
        let predecessor = (
            existing_item_index,
            task_pane.clone(),
            existing_terminal.clone(),
        );
        let destination = TaskTerminalDestination::Replace {
            item_index: existing_item_index,
            pane: task_pane,
            terminal: existing_terminal,
        };
        if task.allow_concurrent_runs {
            return self.spawn_eligible_task(
                task,
                structured_task,
                preparation,
                pending_launch,
                shell,
                is_windows,
                destination,
                window,
                cx,
            );
        }
        terminals_for_task.push(predecessor);
        let (sender, receiver) = oneshot::channel();
        self.deferred_tasks.insert(
            task.id.clone(),
            cx.spawn_in(window, async move |terminal_panel, cx| {
                wait_for_terminals_tasks(terminals_for_task, cx).await;
                let destination = if task.use_new_terminal {
                    TaskTerminalDestination::New
                } else {
                    destination
                };
                let prepared = match terminal_panel.update_in(cx, |terminal_panel, window, cx| {
                    terminal_panel.spawn_eligible_task(
                        task,
                        structured_task,
                        preparation,
                        pending_launch,
                        shell,
                        is_windows,
                        destination,
                        window,
                        cx,
                    )
                }) {
                    Ok(prepared) => match prepared.await {
                        Ok(prepared) => {
                            let terminal = prepared.terminal.await;
                            Ok(PreparedTerminalTask {
                                terminal: Task::ready(terminal),
                                completion: prepared.completion,
                            })
                        }
                        Err(error) => Err(error),
                    },
                    Err(error) => Err(error),
                };
                if sender.send(prepared).is_err() {
                    log::debug!(
                        "Deferred Task receiver expired before terminal creation completed"
                    );
                }
            }),
        );
        cx.spawn(async move |_, _| receiver.await?)
    }

    fn spawn_eligible_task(
        &mut self,
        task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        preparation: Option<TaskLaunchPreparation>,
        pending_launch: Option<PendingExperimentalTaskLaunch>,
        shell: Shell,
        is_windows: bool,
        destination: TaskTerminalDestination,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<PreparedTerminalTask>> {
        if let Some(pending_launch) = pending_launch.as_ref()
            && let Err(error) = pending_launch.check(&task, cx)
        {
            return Task::ready(Err(error));
        }
        let Some(prepare) = preparation else {
            let task = prepare_task_for_spawn(&task, &shell, is_windows);
            let terminal = self.spawn_task_at_destination(
                task,
                structured_task,
                None,
                pending_launch,
                destination,
                window,
                cx,
            );
            return Task::ready(Ok(PreparedTerminalTask {
                terminal,
                completion: None,
            }));
        };
        cx.spawn_in(window, async move |terminal_panel, cx| {
            if structured_task
                .as_ref()
                .is_some_and(|handle| handle.state().is_terminal())
            {
                return Err(anyhow::Error::new(workspace::TaskLaunchCancelled));
            }
            let original = task.clone();
            let mut prepared: PreparedTaskLaunch = prepare(task, cx).await?;
            prepared.validate_task_identity(&original)?;
            cx.update(|_, cx| {
                if let Some(pending_launch) = pending_launch.as_ref() {
                    pending_launch.check(&prepared.launch, cx)?;
                }
                Ok::<_, anyhow::Error>(())
            })??;
            if prepared.pending_launch.is_none() {
                prepared.pending_launch = pending_launch;
            }
            if structured_task
                .as_ref()
                .is_some_and(|handle| handle.state().is_terminal())
            {
                return Err(anyhow::Error::new(workspace::TaskLaunchCancelled));
            }
            let terminal = terminal_panel.update_in(cx, |terminal_panel, window, cx| {
                let task = if prepared.native_terminal.is_some() {
                    prepared.launch
                } else {
                    prepare_task_for_spawn(&prepared.launch, &shell, is_windows)
                };
                terminal_panel.spawn_task_at_destination(
                    task,
                    structured_task,
                    prepared.native_terminal,
                    prepared.pending_launch,
                    destination,
                    window,
                    cx,
                )
            })?;
            Ok(PreparedTerminalTask {
                terminal,
                completion: prepared.completion,
            })
        })
    }

    fn spawn_task_at_destination(
        &mut self,
        task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        native_terminal: Option<PreparedNativeTerminalLaunch>,
        pending_launch: Option<PendingExperimentalTaskLaunch>,
        destination: TaskTerminalDestination,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        match destination {
            TaskTerminalDestination::New => self.spawn_in_new_terminal(
                task,
                structured_task,
                native_terminal,
                pending_launch,
                window,
                cx,
            ),
            TaskTerminalDestination::Replace {
                item_index,
                pane,
                terminal,
            } => self.replace_terminal(
                task,
                pane,
                item_index,
                terminal,
                structured_task,
                native_terminal,
                pending_launch,
                window,
                cx,
            ),
        }
    }

    fn spawn_in_new_terminal(
        &mut self,
        spawn_task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        native_terminal: Option<PreparedNativeTerminalLaunch>,
        pending_launch: Option<PendingExperimentalTaskLaunch>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        let reveal = spawn_task.reveal;
        let reveal_target = spawn_task.reveal_target;
        match reveal_target {
            RevealTarget::Center => self
                .workspace
                .update(cx, |workspace, cx| {
                    Self::add_center_terminal(workspace, window, cx, |project, cx| {
                        project.create_terminal_task_with_pending_launch(
                            spawn_task,
                            structured_task,
                            native_terminal,
                            pending_launch,
                            cx,
                        )
                    })
                })
                .unwrap_or_else(|e| Task::ready(Err(e))),
            RevealTarget::Dock => self.add_terminal_task_with_structured_handle(
                spawn_task,
                reveal,
                structured_task,
                native_terminal,
                pending_launch,
                window,
                cx,
            ),
        }
    }

    /// Create a new Terminal in the current working directory or the user's home directory
    fn new_terminal(
        workspace: &mut Workspace,
        action: &workspace::NewTerminal,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let center_pane = workspace.active_pane();
        let center_pane_has_focus = center_pane.focus_handle(cx).contains_focused(window, cx);
        let active_center_item_is_terminal = center_pane
            .read(cx)
            .active_item()
            .is_some_and(|item| item.downcast::<TerminalView>().is_some());

        if center_pane_has_focus && active_center_item_is_terminal {
            let working_directory = default_working_directory(workspace, cx);
            let local = action.local;
            Self::add_center_terminal(workspace, window, cx, move |project, cx| {
                if local {
                    project.create_local_terminal(cx)
                } else {
                    project.create_terminal_shell(working_directory, cx)
                }
            })
            .detach_and_log_err(cx);
            return;
        }

        let Some(terminal_panel) = workspace.panel::<Self>(cx) else {
            return;
        };

        terminal_panel
            .update(cx, |this, cx| {
                if action.local {
                    this.add_local_terminal_shell(RevealStrategy::Always, window, cx)
                } else {
                    this.add_terminal_shell(
                        default_working_directory(workspace, cx),
                        RevealStrategy::Always,
                        window,
                        cx,
                    )
                }
            })
            .detach_and_log_err(cx);
    }

    fn terminals_for_task(
        &self,
        label: &str,
        cx: &mut App,
    ) -> Vec<(usize, Entity<Pane>, Entity<TerminalView>)> {
        let Some(workspace) = self.workspace.upgrade() else {
            return Vec::new();
        };

        let pane_terminal_views = |pane: Entity<Pane>| {
            pane.read(cx)
                .items()
                .enumerate()
                .filter_map(|(index, item)| Some((index, item.act_as::<TerminalView>(cx)?)))
                .filter_map(|(index, terminal_view)| {
                    let task_state = terminal_view.read(cx).terminal().read(cx).task()?;
                    if &task_state.spawned_task.full_label == label {
                        Some((index, terminal_view))
                    } else {
                        None
                    }
                })
                .map(move |(index, terminal_view)| (index, pane.clone(), terminal_view))
        };

        self.center
            .panes()
            .into_iter()
            .cloned()
            .flat_map(pane_terminal_views)
            .chain(
                workspace
                    .read(cx)
                    .panes()
                    .iter()
                    .cloned()
                    .flat_map(pane_terminal_views),
            )
            .sorted_by_key(|(_, _, terminal_view)| terminal_view.entity_id())
            .collect()
    }

    fn activate_terminal_view(
        &self,
        pane: &Entity<Pane>,
        item_index: usize,
        focus: bool,
        window: &mut Window,
        cx: &mut App,
    ) {
        pane.update(cx, |pane, cx| {
            pane.activate_item(item_index, true, focus, window, cx)
        })
    }

    fn reveal_structured_terminal(
        &self,
        terminal_id: task::StructuredTerminalId,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<bool> {
        let Some(workspace) = self.workspace.upgrade() else {
            return None;
        };
        let pane_terminal_views = |pane: Entity<Pane>| {
            pane.read(cx)
                .items()
                .enumerate()
                .filter_map(|(index, item)| Some((index, item.act_as::<TerminalView>(cx)?)))
                .map(move |(index, terminal_view)| (pane.clone(), index, terminal_view))
        };
        let dock_panes = self.center.panes().into_iter().cloned().collect::<Vec<_>>();
        let found = dock_panes
            .iter()
            .cloned()
            .flat_map(&pane_terminal_views)
            .map(|value| (true, value))
            .chain(
                workspace
                    .read(cx)
                    .panes()
                    .iter()
                    .cloned()
                    .flat_map(pane_terminal_views)
                    .map(|value| (false, value)),
            )
            .find(|(_, (_, _, terminal_view))| {
                terminal_view.read(cx).terminal().entity_id().as_u64() == terminal_id.0
            });
        let Some((is_dock, (pane, index, _terminal_view))) = found else {
            return None;
        };
        self.activate_terminal_view(&pane, index, true, window, cx);
        Some(is_dock)
    }

    fn reveal_structured_terminal_handle(
        panel: &Entity<Self>,
        terminal_id: task::StructuredTerminalId,
        window: &mut Window,
        cx: &mut App,
    ) -> bool {
        let is_dock = panel.update(cx, |panel, cx| {
            panel.reveal_structured_terminal(terminal_id, window, cx)
        });
        if is_dock == Some(true) {
            let workspace = panel.read(cx).workspace.clone();
            if let Some(workspace) = workspace.upgrade() {
                workspace.update(cx, |workspace, cx| {
                    workspace.focus_panel::<Self>(window, cx)
                });
            }
        }
        is_dock.is_some()
    }

    pub fn add_center_terminal(
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Workspace>,
        create_terminal: impl FnOnce(
            &mut Project,
            &mut Context<Project>,
        ) -> Task<Result<Entity<Terminal>>>
        + 'static,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        if !is_enabled_in_workspace(workspace, cx) {
            return Task::ready(Err(anyhow!(
                "terminal not yet supported for remote projects"
            )));
        }
        let project = workspace.project().downgrade();
        cx.spawn_in(window, async move |workspace, cx| {
            let terminal = project.update(cx, create_terminal)?.await?;

            workspace.update_in(cx, |workspace, window, cx| {
                let terminal_view = cx.new(|cx| {
                    TerminalView::new(
                        terminal.clone(),
                        workspace.weak_handle(),
                        workspace.database_id(),
                        workspace.project().downgrade(),
                        window,
                        cx,
                    )
                });
                // Don't steal focus from an open modal (e.g. the command palette):
                // a background terminal can finish starting up after the user has
                // moved on, and focusing it would dismiss whatever they opened.
                let focus_item = !workspace.has_active_modal(window, cx);
                workspace.add_item_to_active_pane(
                    Box::new(terminal_view),
                    None,
                    focus_item,
                    window,
                    cx,
                );
            })?;
            Ok(terminal.downgrade())
        })
    }

    pub fn add_terminal_task(
        &mut self,
        task: SpawnInTerminal,
        reveal_strategy: RevealStrategy,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        let pending_launch = match PendingExperimentalTaskLaunch::new(&task, cx) {
            Ok(pending_launch) => pending_launch,
            Err(error) => return Task::ready(Err(error)),
        };
        self.add_terminal_task_with_structured_handle(
            task,
            reveal_strategy,
            None,
            None,
            pending_launch,
            window,
            cx,
        )
    }

    fn add_terminal_task_with_structured_handle(
        &mut self,
        task: SpawnInTerminal,
        reveal_strategy: RevealStrategy,
        structured_task: Option<StructuredTaskHandle>,
        native_terminal: Option<PreparedNativeTerminalLaunch>,
        pending_launch: Option<PendingExperimentalTaskLaunch>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        let workspace = self.workspace.clone();
        cx.spawn_in(window, async move |terminal_panel, cx| {
            if workspace.update(cx, |workspace, cx| !is_enabled_in_workspace(workspace, cx))? {
                anyhow::bail!("terminal not yet supported for remote projects");
            }
            let project = workspace.read_with(cx, |workspace, _| workspace.project().clone())?;
            let pane = terminal_panel.update(cx, |terminal_panel, _| {
                terminal_panel.pending_terminals_to_add += 1;
                terminal_panel.active_pane.clone()
            })?;
            let mut pending_creation = PendingTerminalCreation {
                panel: Some(terminal_panel.clone()),
                context: AsyncApp::clone(cx),
            };
            let created = project
                .update(cx, |project, cx| {
                    project.create_terminal_task_with_pending_launch(
                        task,
                        structured_task,
                        native_terminal,
                        pending_launch,
                        cx,
                    )
                })
                .await;
            terminal_panel.update(cx, |terminal_panel, _| {
                terminal_panel.pending_terminals_to_add =
                    terminal_panel.pending_terminals_to_add.saturating_sub(1);
            })?;
            pending_creation.panel = None;
            let terminal = created?;
            let result = workspace.update_in(cx, |workspace, window, cx| {
                let terminal_view = Box::new(cx.new(|cx| {
                    TerminalView::new(
                        terminal.clone(),
                        workspace.weak_handle(),
                        workspace.database_id(),
                        workspace.project().downgrade(),
                        window,
                        cx,
                    )
                }));

                match reveal_strategy {
                    RevealStrategy::Always => {
                        workspace.focus_panel::<Self>(window, cx);
                    }
                    RevealStrategy::NoFocus => {
                        workspace.open_panel::<Self>(window, cx);
                    }
                    RevealStrategy::Never => {}
                }

                pane.update(cx, |pane, cx| {
                    let focus = matches!(reveal_strategy, RevealStrategy::Always);
                    pane.add_item(terminal_view, true, focus, None, window, cx);
                });

                Ok(terminal.downgrade())
            })?;
            terminal_panel.update(cx, |terminal_panel, cx| terminal_panel.serialize(cx))?;
            result
        })
    }

    fn add_terminal_shell(
        &mut self,
        cwd: Option<PathBuf>,
        reveal_strategy: RevealStrategy,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        self.add_terminal_shell_internal(false, cwd, reveal_strategy, window, cx)
    }

    fn add_local_terminal_shell(
        &mut self,
        reveal_strategy: RevealStrategy,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        self.add_terminal_shell_internal(true, None, reveal_strategy, window, cx)
    }

    fn add_terminal_shell_internal(
        &mut self,
        force_local: bool,
        cwd: Option<PathBuf>,
        reveal_strategy: RevealStrategy,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        let workspace = self.workspace.clone();

        cx.spawn_in(window, async move |terminal_panel, cx| {
            if workspace.update(cx, |workspace, cx| !is_enabled_in_workspace(workspace, cx))? {
                anyhow::bail!("terminal not yet supported for collaborative projects");
            }
            let pane = terminal_panel.update(cx, |terminal_panel, _| {
                terminal_panel.pending_terminals_to_add += 1;
                terminal_panel.active_pane.clone()
            })?;
            let project = workspace.read_with(cx, |workspace, _| workspace.project().clone())?;
            let terminal = if force_local {
                project
                    .update(cx, |project, cx| project.create_local_terminal(cx))
                    .await
            } else {
                project
                    .update(cx, |project, cx| project.create_terminal_shell(cwd, cx))
                    .await
            };

            match terminal {
                Ok(terminal) => {
                    let result = workspace.update_in(cx, |workspace, window, cx| {
                        let terminal_view = Box::new(cx.new(|cx| {
                            TerminalView::new(
                                terminal.clone(),
                                workspace.weak_handle(),
                                workspace.database_id(),
                                workspace.project().downgrade(),
                                window,
                                cx,
                            )
                        }));

                        match reveal_strategy {
                            RevealStrategy::Always => {
                                workspace.focus_panel::<Self>(window, cx);
                            }
                            RevealStrategy::NoFocus => {
                                workspace.open_panel::<Self>(window, cx);
                            }
                            RevealStrategy::Never => {}
                        }

                        pane.update(cx, |pane, cx| {
                            let focus = matches!(reveal_strategy, RevealStrategy::Always);
                            pane.add_item(terminal_view, true, focus, None, window, cx);
                        });

                        Ok(terminal.downgrade())
                    })?;
                    terminal_panel.update(cx, |terminal_panel, cx| {
                        terminal_panel.pending_terminals_to_add =
                            terminal_panel.pending_terminals_to_add.saturating_sub(1);
                        terminal_panel.serialize(cx)
                    })?;
                    result
                }
                Err(error) => {
                    pane.update_in(cx, |pane, window, cx| {
                        let focus = pane.has_focus(window, cx);
                        let failed_to_spawn = cx.new(|cx| FailedToSpawnTerminal {
                            error: error.to_string(),
                            focus_handle: cx.focus_handle(),
                        });
                        pane.add_item(Box::new(failed_to_spawn), true, focus, None, window, cx);
                    })?;
                    Err(error)
                }
            }
        })
    }

    fn serialize(&mut self, cx: &mut Context<Self>) {
        let Some(serialization_key) = self
            .workspace
            .read_with(cx, |workspace, _| {
                TerminalPanel::serialization_key(workspace)
            })
            .ok()
            .flatten()
        else {
            return;
        };
        let kvp = KeyValueStore::global(cx);
        self.pending_serialization = cx.spawn(async move |terminal_panel, cx| {
            cx.background_executor()
                .timer(Duration::from_millis(50))
                .await;
            let terminal_panel = terminal_panel.upgrade()?;
            let items = terminal_panel.update(cx, |terminal_panel, cx| {
                SerializedItems::WithSplits(serialize_pane_group(
                    &terminal_panel.center,
                    &terminal_panel.active_pane,
                    cx,
                ))
            });
            cx.background_spawn(
                async move {
                    kvp.write_kvp(
                        serialization_key,
                        serde_json::to_string(&SerializedTerminalPanel {
                            items,
                            active_item_id: None,
                        })?,
                    )
                    .await?;
                    anyhow::Ok(())
                }
                .log_err(),
            )
            .await;
            Some(())
        });
    }

    fn replace_terminal(
        &self,
        spawn_task: SpawnInTerminal,
        task_pane: Entity<Pane>,
        terminal_item_index: usize,
        terminal_to_replace: Entity<TerminalView>,
        structured_task: Option<StructuredTaskHandle>,
        native_terminal: Option<PreparedNativeTerminalLaunch>,
        pending_launch: Option<PendingExperimentalTaskLaunch>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<WeakEntity<Terminal>>> {
        let reveal = spawn_task.reveal;
        let task_workspace = self.workspace.clone();
        cx.spawn_in(window, async move |terminal_panel, cx| {
            let project = terminal_panel.update(cx, |this, cx| {
                this.workspace
                    .update(cx, |workspace, _| workspace.project().clone())
            })??;
            let new_terminal = project
                .update(cx, |project, cx| {
                    project.create_terminal_task_with_pending_launch(
                        spawn_task,
                        structured_task,
                        native_terminal,
                        pending_launch,
                        cx,
                    )
                })
                .await?;
            terminal_to_replace.update_in(cx, |terminal_to_replace, window, cx| {
                terminal_to_replace.set_terminal(new_terminal.clone(), window, cx);
            })?;

            let reveal_target = terminal_panel.update(cx, |panel, _| {
                if panel.center.panes().iter().any(|p| **p == task_pane) {
                    RevealTarget::Dock
                } else {
                    RevealTarget::Center
                }
            })?;

            match reveal {
                RevealStrategy::Always => match reveal_target {
                    RevealTarget::Center => {
                        task_workspace.update_in(cx, |workspace, window, cx| {
                            let did_activate = workspace.activate_item(
                                &terminal_to_replace,
                                true,
                                true,
                                window,
                                cx,
                            );

                            anyhow::ensure!(did_activate, "Failed to retrieve terminal pane");

                            anyhow::Ok(())
                        })??;
                    }
                    RevealTarget::Dock => {
                        terminal_panel.update_in(cx, |terminal_panel, window, cx| {
                            terminal_panel.activate_terminal_view(
                                &task_pane,
                                terminal_item_index,
                                true,
                                window,
                                cx,
                            )
                        })?;

                        cx.spawn(async move |cx| {
                            task_workspace
                                .update_in(cx, |workspace, window, cx| {
                                    workspace.focus_panel::<Self>(window, cx)
                                })
                                .ok()
                        })
                        .detach();
                    }
                },
                RevealStrategy::NoFocus => match reveal_target {
                    RevealTarget::Center => {
                        task_workspace.update_in(cx, |workspace, window, cx| {
                            workspace.active_pane().focus_handle(cx).focus(window, cx);
                        })?;
                    }
                    RevealTarget::Dock => {
                        terminal_panel.update_in(cx, |terminal_panel, window, cx| {
                            terminal_panel.activate_terminal_view(
                                &task_pane,
                                terminal_item_index,
                                false,
                                window,
                                cx,
                            )
                        })?;

                        cx.spawn(async move |cx| {
                            task_workspace
                                .update_in(cx, |workspace, window, cx| {
                                    workspace.open_panel::<Self>(window, cx)
                                })
                                .ok()
                        })
                        .detach();
                    }
                },
                RevealStrategy::Never => {}
            }

            Ok(new_terminal.downgrade())
        })
    }

    fn has_no_terminals(&self, cx: &App) -> bool {
        self.active_pane.read(cx).items_len() == 0 && self.pending_terminals_to_add == 0
    }

    #[cfg(feature = "agentic")]
    pub fn assistant_enabled(&self) -> bool {
        self.assistant_enabled
    }

    /// Returns all panes in the terminal panel.
    pub fn panes(&self) -> Vec<&Entity<Pane>> {
        self.center.panes()
    }

    /// Returns all non-empty terminal selections from all terminal views in all panes.
    pub fn terminal_selections(&self, cx: &App) -> Vec<String> {
        self.center
            .panes()
            .iter()
            .flat_map(|pane| {
                pane.read(cx).items().filter_map(|item| {
                    let terminal_view = item.downcast::<crate::TerminalView>()?;
                    terminal_view
                        .read(cx)
                        .terminal()
                        .read(cx)
                        .last_content
                        .selection_text
                        .clone()
                        .filter(|text| !text.is_empty())
                })
            })
            .collect()
    }

    fn is_enabled(&self, cx: &App) -> bool {
        self.workspace
            .upgrade()
            .is_some_and(|workspace| is_enabled_in_workspace(workspace.read(cx), cx))
    }

    fn activate_pane_in_direction(
        &mut self,
        direction: SplitDirection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(pane) = self
            .center
            .find_pane_in_direction(&self.active_pane, direction, cx)
        {
            window.focus(&pane.focus_handle(cx), cx);
        } else {
            self.workspace
                .update(cx, |workspace, cx| {
                    workspace.activate_pane_in_direction(direction, window, cx)
                })
                .ok();
        }
    }

    fn swap_pane_in_direction(&mut self, direction: SplitDirection, cx: &mut Context<Self>) {
        if let Some(to) = self
            .center
            .find_pane_in_direction(&self.active_pane, direction, cx)
            .cloned()
        {
            self.center.swap(&self.active_pane, &to, cx);
            cx.notify();
        }
    }

    fn move_pane_to_border(&mut self, direction: SplitDirection, cx: &mut Context<Self>) {
        if self
            .center
            .move_to_border(&self.active_pane, direction, cx)
            .unwrap()
        {
            cx.notify();
        }
    }
}

/// Prepares a `SpawnInTerminal` by computing the command, args, and command_label
/// based on the shell configuration. This is a pure function that can be tested
/// without spawning actual terminals.
pub fn prepare_task_for_spawn(
    task: &SpawnInTerminal,
    shell: &Shell,
    is_windows: bool,
) -> SpawnInTerminal {
    let builder = ShellBuilder::new(shell, is_windows);
    let command_label = if task.preserve_command_label {
        task.command_label.clone()
    } else {
        builder.command_label(task.command.as_deref().unwrap_or(""))
    };
    // Literal JVM batch wrappers still require the selected shell.
    if is_windows
        && task.literal_command
        && task.command.as_deref().is_some_and(|command| {
            let file_name = command.rsplit(['\\', '/']).next().unwrap_or_default();
            !file_name.is_empty()
                && file_name
                    .rsplit_once('.')
                    .is_none_or(|(_, extension)| extension.eq_ignore_ascii_case("exe"))
        })
        && task.literal_prefix_args == task.args.len()
    {
        return SpawnInTerminal {
            command_label,
            shell: shell.clone(),
            windows_direct_program: true,
            ..task.clone()
        };
    }
    let (command, args) = builder.build_with_command_intent(
        task.command.clone(),
        &task.args,
        task.literal_command,
        task.literal_prefix_args,
    );

    SpawnInTerminal {
        command_label,
        command: Some(command),
        args,
        windows_direct_program: false,
        ..task.clone()
    }
}

fn is_enabled_in_workspace(workspace: &Workspace, cx: &App) -> bool {
    workspace.project().read(cx).supports_terminal(cx)
}

pub fn new_terminal_pane(
    workspace: WeakEntity<Workspace>,
    project: Entity<Project>,
    zoomed: bool,
    window: &mut Window,
    cx: &mut Context<TerminalPanel>,
) -> Entity<Pane> {
    let terminal_panel = cx.entity();
    let pane = cx.new(|cx| {
        let mut pane = Pane::new(
            workspace.clone(),
            project.clone(),
            Default::default(),
            None,
            workspace::NewTerminal::default().boxed_clone(),
            false,
            window,
            cx,
        );
        pane.set_zoomed(zoomed, cx);
        pane.set_can_navigate(false, cx);
        pane.display_nav_history_buttons(None);
        pane.set_should_display_tab_bar(|_, _| true);
        pane.set_zoom_out_on_close(false);

        let split_closure_terminal_panel = terminal_panel.downgrade();
        pane.set_can_split(Some(Arc::new(move |pane, dragged_item, _window, cx| {
            if let Some(tab) = dragged_item.downcast_ref::<DraggedTab>() {
                let is_current_pane = tab.pane == cx.entity();
                let Some(can_drag_away) = split_closure_terminal_panel
                    .read_with(cx, |terminal_panel, _| {
                        let current_panes = terminal_panel.center.panes();
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
                        return item.downcast::<TerminalView>().is_some();
                    }
                }
            }
            false
        })));

        let toolbar = pane.toolbar().clone();
        if let Some(callbacks) = cx.try_global::<workspace::PaneSearchBarCallbacks>() {
            let languages = Some(project.read(cx).languages().clone());
            (callbacks.setup_search_bar)(languages, &toolbar, window, cx);
        }
        let breadcrumbs = cx.new(|_| Breadcrumbs::new());
        toolbar.update(cx, |toolbar, cx| {
            toolbar.add_item(breadcrumbs, window, cx);
        });

        pane
    });

    cx.subscribe_in(&pane, window, TerminalPanel::handle_pane_event)
        .detach();
    cx.observe(&pane, |_, _, cx| cx.notify()).detach();

    pane
}

async fn wait_for_terminals_tasks(
    terminals_for_task: Vec<(usize, Entity<Pane>, Entity<TerminalView>)>,
    cx: &mut AsyncApp,
) {
    let pending_tasks = terminals_for_task.iter().map(|(_, _, terminal)| {
        terminal.update(cx, |terminal_view, cx| {
            terminal_view
                .terminal()
                .update(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
        })
    });
    join_all(pending_tasks).await;
}

struct FailedToSpawnTerminal {
    error: String,
    focus_handle: FocusHandle,
}

impl Focusable for FailedToSpawnTerminal {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for FailedToSpawnTerminal {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let popover_menu = PopoverMenu::new("settings-popover")
            .trigger(
                IconButton::new("icon-button-popover", IconName::ChevronDown)
                    .icon_size(IconSize::XSmall),
            )
            .menu(move |window, cx| {
                Some(ContextMenu::build(window, cx, |context_menu, _, _| {
                    context_menu
                        .action("Open Settings", zed_actions::OpenSettings.boxed_clone())
                        .action(
                            "Edit settings.json",
                            zed_actions::OpenSettingsFile.boxed_clone(),
                        )
                }))
            })
            .anchor(Anchor::TopRight)
            .offset(gpui::Point {
                x: px(0.0),
                y: px(2.0),
            });

        v_flex()
            .track_focus(&self.focus_handle)
            .size_full()
            .p_4()
            .items_center()
            .justify_center()
            .bg(cx.theme().colors().editor_background)
            .child(
                v_flex()
                    .max_w_112()
                    .items_center()
                    .justify_center()
                    .text_center()
                    .child(Label::new("Failed to spawn terminal"))
                    .child(
                        Label::new(self.error.to_string())
                            .size(LabelSize::Small)
                            .color(Color::Muted)
                            .mb_4(),
                    )
                    .child(SplitButton::new(
                        ButtonLike::new("open-settings-ui")
                            .child(Label::new("Edit Settings").size(LabelSize::Small))
                            .on_click(|_, window, cx| {
                                window.dispatch_action(zed_actions::OpenSettings.boxed_clone(), cx);
                            }),
                        popover_menu.into_any_element(),
                    )),
            )
    }
}

impl EventEmitter<()> for FailedToSpawnTerminal {}

impl workspace::Item for FailedToSpawnTerminal {
    type Event = ();

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        SharedString::new_static("Failed to spawn terminal")
    }
}

impl EventEmitter<PanelEvent> for TerminalPanel {}

impl Render for TerminalPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let registrar = cx
            .try_global::<workspace::PaneSearchBarCallbacks>()
            .map(|callbacks| {
                (callbacks.wrap_div_with_search_actions)(div(), self.active_pane.clone())
            })
            .unwrap_or_else(div);
        self.workspace
            .update(cx, |workspace, cx| {
                registrar
                    .track_focus(&self.focus_handle)
                    .size_full()
                    .child(self.center.render(
                        workspace.zoomed_item(),
                        None,
                        &workspace::PaneRenderContext {
                            follower_states: &HashMap::default(),
                            active_call: workspace.active_call(),
                            active_pane: &self.active_pane,
                            app_state: workspace.app_state(),
                            project: workspace.project(),
                            workspace: &workspace.weak_handle(),
                        },
                        window,
                        cx,
                    ))
            })
            .ok()
            .map(|div| {
                div.on_action({
                    cx.listener(|terminal_panel, _: &ActivatePaneLeft, window, cx| {
                        terminal_panel.activate_pane_in_direction(SplitDirection::Left, window, cx);
                    })
                })
                .on_action({
                    cx.listener(|terminal_panel, _: &ActivatePaneRight, window, cx| {
                        terminal_panel.activate_pane_in_direction(
                            SplitDirection::Right,
                            window,
                            cx,
                        );
                    })
                })
                .on_action({
                    cx.listener(|terminal_panel, _: &ActivatePaneUp, window, cx| {
                        terminal_panel.activate_pane_in_direction(SplitDirection::Up, window, cx);
                    })
                })
                .on_action({
                    cx.listener(|terminal_panel, _: &ActivatePaneDown, window, cx| {
                        terminal_panel.activate_pane_in_direction(SplitDirection::Down, window, cx);
                    })
                })
                .on_action(
                    cx.listener(|terminal_panel, _action: &ActivateNextPane, window, cx| {
                        let panes = terminal_panel.center.panes();
                        if let Some(ix) = panes
                            .iter()
                            .position(|pane| **pane == terminal_panel.active_pane)
                        {
                            let next_ix = (ix + 1) % panes.len();
                            window.focus(&panes[next_ix].focus_handle(cx), cx);
                        }
                    }),
                )
                .on_action(cx.listener(
                    |terminal_panel, _action: &ActivatePreviousPane, window, cx| {
                        let panes = terminal_panel.center.panes();
                        if let Some(ix) = panes
                            .iter()
                            .position(|pane| **pane == terminal_panel.active_pane)
                        {
                            let prev_ix = cmp::min(ix.wrapping_sub(1), panes.len() - 1);
                            window.focus(&panes[prev_ix].focus_handle(cx), cx);
                        }
                    },
                ))
                .on_action(
                    cx.listener(|terminal_panel, action: &ActivatePane, window, cx| {
                        let panes = terminal_panel.center.panes();
                        if let Some(&pane) = panes.get(action.0) {
                            window.focus(&pane.read(cx).focus_handle(cx), cx);
                        } else {
                            let future =
                                terminal_panel.new_pane_with_active_terminal(true, window, cx);
                            cx.spawn_in(window, async move |terminal_panel, cx| {
                                if let Some(new_pane) = future.await {
                                    _ = terminal_panel.update_in(
                                        cx,
                                        |terminal_panel, window, cx| {
                                            terminal_panel.center.split(
                                                &terminal_panel.active_pane,
                                                &new_pane,
                                                SplitDirection::Right,
                                                cx,
                                            );
                                            let new_pane = new_pane.read(cx);
                                            window.focus(&new_pane.focus_handle(cx), cx);
                                        },
                                    );
                                }
                            })
                            .detach();
                        }
                    }),
                )
                .on_action(cx.listener(|terminal_panel, _: &SwapPaneLeft, _, cx| {
                    terminal_panel.swap_pane_in_direction(SplitDirection::Left, cx);
                }))
                .on_action(cx.listener(|terminal_panel, _: &SwapPaneRight, _, cx| {
                    terminal_panel.swap_pane_in_direction(SplitDirection::Right, cx);
                }))
                .on_action(cx.listener(|terminal_panel, _: &SwapPaneUp, _, cx| {
                    terminal_panel.swap_pane_in_direction(SplitDirection::Up, cx);
                }))
                .on_action(cx.listener(|terminal_panel, _: &SwapPaneDown, _, cx| {
                    terminal_panel.swap_pane_in_direction(SplitDirection::Down, cx);
                }))
                .on_action(cx.listener(|terminal_panel, _: &MovePaneLeft, _, cx| {
                    terminal_panel.move_pane_to_border(SplitDirection::Left, cx);
                }))
                .on_action(cx.listener(|terminal_panel, _: &MovePaneRight, _, cx| {
                    terminal_panel.move_pane_to_border(SplitDirection::Right, cx);
                }))
                .on_action(cx.listener(|terminal_panel, _: &MovePaneUp, _, cx| {
                    terminal_panel.move_pane_to_border(SplitDirection::Up, cx);
                }))
                .on_action(cx.listener(|terminal_panel, _: &MovePaneDown, _, cx| {
                    terminal_panel.move_pane_to_border(SplitDirection::Down, cx);
                }))
                .on_action(
                    cx.listener(|terminal_panel, action: &MoveItemToPane, window, cx| {
                        let Some(&target_pane) =
                            terminal_panel.center.panes().get(action.destination)
                        else {
                            return;
                        };
                        move_active_item(
                            &terminal_panel.active_pane,
                            target_pane,
                            action.focus,
                            true,
                            window,
                            cx,
                        );
                    }),
                )
                .on_action(cx.listener(
                    |terminal_panel, action: &MoveItemToPaneInDirection, window, cx| {
                        let source_pane = &terminal_panel.active_pane;
                        if let Some(destination_pane) = terminal_panel
                            .center
                            .find_pane_in_direction(source_pane, action.direction, cx)
                        {
                            move_active_item(
                                source_pane,
                                destination_pane,
                                action.focus,
                                true,
                                window,
                                cx,
                            );
                        };
                    },
                ))
            })
            .unwrap_or_else(|| div())
    }
}

impl Focusable for TerminalPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Panel for TerminalPanel {
    fn activation_focus_handle(&self, cx: &App) -> FocusHandle {
        self.active_pane.focus_handle(cx)
    }

    fn position(&self, _window: &Window, cx: &App) -> DockPosition {
        TerminalSettings::get_global(cx).dock.into()
    }

    fn position_is_valid(&self, _: DockPosition) -> bool {
        true
    }

    fn starts_open(&self, _: &Window, cx: &App) -> bool {
        TerminalSettings::get_global(cx).starts_open
    }

    fn set_position(
        &mut self,
        position: DockPosition,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        settings::update_settings_file(self.fs.clone(), cx, move |settings, _| {
            let dock = match position {
                DockPosition::Left => TerminalDockPosition::Left,
                DockPosition::Bottom => TerminalDockPosition::Bottom,
                DockPosition::Right => TerminalDockPosition::Right,
            };
            settings.terminal.get_or_insert_default().dock = Some(dock);
        });
    }

    fn default_size(&self, window: &Window, cx: &App) -> Pixels {
        let settings = TerminalSettings::get_global(cx);
        match self.position(window, cx) {
            DockPosition::Left | DockPosition::Right => settings.default_width,
            DockPosition::Bottom => settings.default_height,
        }
    }

    fn supports_flexible_size(&self) -> bool {
        true
    }

    fn has_flexible_size(&self, _window: &Window, cx: &App) -> bool {
        TerminalSettings::get_global(cx).flexible
    }

    fn set_flexible_size(&mut self, flexible: bool, _window: &mut Window, cx: &mut Context<Self>) {
        settings::update_settings_file(self.fs.clone(), cx, move |settings, _| {
            settings.terminal.get_or_insert_default().flexible = Some(flexible);
        });
    }

    fn is_zoomed(&self, _window: &Window, cx: &App) -> bool {
        self.active_pane.read(cx).is_zoomed()
    }

    fn set_zoomed(&mut self, zoomed: bool, _: &mut Window, cx: &mut Context<Self>) {
        for pane in self.center.panes() {
            pane.update(cx, |pane, cx| {
                pane.set_zoomed(zoomed, cx);
            })
        }
        cx.notify();
    }

    fn set_active(&mut self, active: bool, window: &mut Window, cx: &mut Context<Self>) {
        let old_active = self.active;
        self.active = active;
        if !active || old_active == active || !self.has_no_terminals(cx) {
            return;
        }
        cx.defer_in(window, |this, window, cx| {
            let Ok(kind) = this
                .workspace
                .update(cx, |workspace, cx| default_working_directory(workspace, cx))
            else {
                return;
            };

            this.add_terminal_shell(kind, RevealStrategy::Always, window, cx)
                .detach_and_log_err(cx)
        })
    }

    fn icon_label(&self, _window: &Window, cx: &App) -> Option<String> {
        if !TerminalSettings::get_global(cx).show_count_badge {
            return None;
        }
        let count = self
            .center
            .panes()
            .into_iter()
            .map(|pane| pane.read(cx).items_len())
            .sum::<usize>();
        if count == 0 {
            None
        } else {
            Some(count.to_string())
        }
    }

    fn persistent_name() -> &'static str {
        "TerminalPanel"
    }

    fn panel_key() -> &'static str {
        TERMINAL_PANEL_KEY
    }

    fn icon(&self, _window: &Window, cx: &App) -> Option<IconName> {
        if (self.is_enabled(cx) || !self.has_no_terminals(cx))
            && TerminalSettings::get_global(cx).button
        {
            Some(IconName::TerminalAlt)
        } else {
            None
        }
    }

    fn icon_tooltip(&self, _window: &Window, _cx: &App) -> Option<&'static str> {
        Some("Terminal Panel")
    }

    fn toggle_action(&self) -> Box<dyn gpui::Action> {
        Box::new(Toggle)
    }

    fn pane(&self) -> Option<Entity<Pane>> {
        Some(self.active_pane.clone())
    }

    fn activation_priority(&self) -> u32 {
        2
    }

    fn hide_button_setting(&self, _: &App) -> Option<workspace::HideStatusItem> {
        Some(workspace::HideStatusItem::new(|settings| {
            settings.terminal.get_or_insert_default().button = Some(false);
        }))
    }
}

struct TerminalProvider(Entity<TerminalPanel>);

impl TerminalProvider {
    fn spawn_internal(
        &self,
        task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        preparation: Option<TaskLaunchPreparation>,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Option<Result<ExitStatus>>> {
        let terminal_panel = self.0.clone();
        window.spawn(cx, async move |cx| {
            let prepared = match terminal_panel.update_in(cx, |terminal_panel, window, cx| {
                terminal_panel.spawn_task_internal(
                    &task,
                    structured_task.clone(),
                    preparation,
                    window,
                    cx,
                )
            }) {
                Ok(prepared) => match prepared.await {
                    Ok(prepared) => prepared,
                    Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                    Err(error) => return Some(Err(error)),
                },
                Err(_) => return None,
            };
            let completion = prepared.completion;
            let completion_handle = structured_task.clone();
            let terminal_result = cx.spawn(async move |cx| match prepared.terminal.await {
                Ok(terminal) => {
                    if let Some(structured_task) = structured_task {
                        let terminal_panel = terminal_panel.clone();
                        let terminal_id = task::StructuredTerminalId(terminal.entity_id().as_u64());
                        structured_task.set_reveal(move |window, cx| {
                            TerminalPanel::reveal_structured_terminal_handle(
                                &terminal_panel,
                                terminal_id,
                                window,
                                cx,
                            )
                        });
                    }
                    let status = terminal
                        .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                        .ok()?
                        .await?;
                    if completion_handle.as_ref().is_some_and(|handle| {
                        matches!(handle.state(), task::StructuredTaskState::Cancelled { .. })
                    }) {
                        return None;
                    }
                    Some(Ok(status))
                }
                Err(error) => Some(Err(error)),
            });
            match completion {
                Some(completion) => completion(terminal_result, cx).await,
                None => terminal_result.await,
            }
        })
    }
}

impl workspace::TerminalProvider for TerminalProvider {
    fn spawn(
        &self,
        task: SpawnInTerminal,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Option<Result<ExitStatus>>> {
        self.spawn_internal(task, None, None, window, cx)
    }

    fn spawn_structured(
        &self,
        task: SpawnInTerminal,
        handle: StructuredTaskHandle,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Option<Result<ExitStatus>>> {
        self.spawn_internal(task, Some(handle), None, window, cx)
    }

    fn spawn_prepared(
        &self,
        task: SpawnInTerminal,
        handle: Option<StructuredTaskHandle>,
        prepare: TaskLaunchPreparation,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Option<Result<ExitStatus>>> {
        self.spawn_internal(task, handle, Some(prepare), window, cx)
    }
}

#[cfg(feature = "agentic")]
#[derive(IntoElement)]
struct InlineAssistTabBarButton {
    focus_handle: FocusHandle,
}

#[cfg(feature = "agentic")]
impl RenderOnce for InlineAssistTabBarButton {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let focus_handle = self.focus_handle;
        IconButton::new("terminal_inline_assistant", IconName::ZedAssistant)
            .icon_size(IconSize::Small)
            .on_click({
                let focus_handle = focus_handle.clone();
                move |_, window, cx| {
                    focus_handle.dispatch_action(&InlineAssist::default(), window, cx);
                }
            })
            .tooltip(move |_window, cx| {
                Tooltip::for_action_in("Inline Assist", &InlineAssist::default(), &focus_handle, cx)
            })
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use super::*;
    #[cfg(feature = "agentic")]
    use gpui::Modifiers;
    use gpui::{TestAppContext, UpdateGlobal as _, VisualTestContext};
    use pretty_assertions::assert_eq;
    use project::FakeFs;
    use settings::SettingsStore;
    use workspace::MultiWorkspace;

    #[test]
    fn test_prepare_windows_literal_task_preserves_native_arguments() {
        let arguments = vec![
            "".to_owned(),
            "src/lib.rs - example (line 7)".to_owned(),
            "quotes\" and trailing\\".to_owned(),
            "%PATH% !VARIABLE! $(echo expanded) & echo second".to_owned(),
            "λ 日本語".to_owned(),
        ];
        for shell in [
            Shell::Program("cmd.exe".into()),
            Shell::Program("powershell.exe".into()),
            Shell::WithArguments {
                program: "pwsh.exe".into(),
                args: vec!["-NoProfile".into()],
                title_override: None,
            },
        ] {
            for arguments in [arguments.clone(), Vec::new()] {
                let input = SpawnInTerminal {
                    command: Some(r"C:\Program Files\Rust\cargo.exe".into()),
                    literal_command: true,
                    literal_prefix_args: arguments.len(),
                    args: arguments,
                    ..Default::default()
                };
                let prepared = prepare_task_for_spawn(&input, &shell, true);
                assert!(prepared.windows_direct_program);
                assert_eq!(prepared.command, input.command);
                assert_eq!(prepared.args, input.args);
                assert_eq!(prepared.shell, shell);
            }
        }
    }

    #[test]
    fn test_prepare_windows_authored_and_partial_tasks_keep_shell_route() {
        let shell = Shell::Program("cmd.exe".into());
        for (literal_command, literal_prefix_args) in [(false, 0), (true, 1)] {
            let input = SpawnInTerminal {
                command: Some("cargo".into()),
                args: vec!["test".into(), "%USER_TASK_ARGUMENT%".into()],
                literal_command,
                literal_prefix_args,
                windows_direct_program: true,
                ..Default::default()
            };
            let prepared = prepare_task_for_spawn(&input, &shell, true);
            assert!(!prepared.windows_direct_program);
            assert_eq!(prepared.command.as_deref(), Some("cmd.exe"));
            assert!(
                prepared
                    .args
                    .iter()
                    .any(|argument| argument.contains("%USER_TASK_ARGUMENT%"))
            );
        }
        let input = SpawnInTerminal {
            command: Some("cargo".into()),
            literal_command: true,
            windows_direct_program: true,
            ..Default::default()
        };
        let prepared = prepare_task_for_spawn(&input, &Shell::Program("sh".into()), false);
        assert!(!prepared.windows_direct_program);
        assert_eq!(prepared.command.as_deref(), Some("sh"));
    }

    #[test]
    fn test_prepare_windows_literal_program_suffix_preserves_shell_scripts() {
        let shell = Shell::Program("cmd.exe".into());
        for (command, direct) in [
            (r"C:\tool.chains\cargo", true),
            (r"C:\tool.chains\CARGO.EXE", true),
            (r"C:\fixture\mvnw.cmd", false),
            (r"C:\fixture\gradlew.BAT", false),
            (r"C:/fixture/build.ps1", false),
        ] {
            let input = SpawnInTerminal {
                command: Some(command.into()),
                args: vec!["verify".into()],
                literal_command: true,
                literal_prefix_args: 1,
                ..Default::default()
            };
            let prepared = prepare_task_for_spawn(&input, &shell, true);
            assert_eq!(prepared.windows_direct_program, direct, "{command}");
            if direct {
                assert_eq!(prepared.command, input.command);
                assert_eq!(prepared.args, input.args);
            } else {
                assert_eq!(prepared.command.as_deref(), Some("cmd.exe"));
                assert!(
                    prepared
                        .args
                        .last()
                        .expect("shell command")
                        .contains(command)
                );
            }
        }
    }

    #[cfg(feature = "jvm-tools")]
    #[test]
    fn test_prepare_windows_jvm_operation_plans_keep_shell_route() {
        use project::{
            ProjectPath, WorktreeId,
            jvm_project::{
                GradleProject, GradleTask, JdtlsImportedBuildKind, JdtlsProject,
                JdtlsProjectMembership, JdtlsProjectSnapshot, JvmBuildOperation,
                JvmBuildOperationReceipt, MavenLifecycle, MavenProject,
            },
        };
        use task::{TaskContext, TaskTemplate, TaskVariables, VariableName};
        use util::{paths::PathStyle, rel_path::RelPath};

        let module_root = ProjectPath::root_path(WorktreeId::from_usize(1));
        let mut variables = TaskVariables::default();
        variables.insert(VariableName::WorktreeRoot, r"C:\fixture".into());
        let context = TaskContext {
            task_variables: variables,
            ..Default::default()
        };
        for (kind, wrapper, operation) in [
            (
                JdtlsImportedBuildKind::Maven,
                "mvnw.cmd",
                JvmBuildOperation::MavenPhase {
                    lifecycle: "default".into(),
                    phase: "verify".into(),
                },
            ),
            (
                JdtlsImportedBuildKind::Gradle,
                "gradlew.bat",
                JvmBuildOperation::GradleTask {
                    path: ":verify".into(),
                },
            ),
        ] {
            let project = JdtlsProject {
                module_root: module_root.clone(),
                membership: JdtlsProjectMembership::ImportedJava,
                imported_build_kind: kind,
                source_roots: Vec::new(),
                project_references: Vec::new(),
                library_entries: 0,
                library_files: Vec::new(),
                library_artifacts: Vec::new(),
                project_vm_known: false,
                project_vm_java_major: None,
                server_launch_java_major: None,
                java_source_level: None,
                java_target_level: None,
                gradle: (kind == JdtlsImportedBuildKind::Gradle).then(|| GradleProject {
                    dependency_graph: None,
                    build_root: module_root.clone(),
                    project_path: ":".into(),
                    tasks: vec![GradleTask {
                        path: ":verify".into(),
                        name: "verify".into(),
                        group: None,
                        description: None,
                        options_known: false,
                        options: Vec::new(),
                    }],
                    source_sets_known: false,
                    source_sets: Vec::new(),
                    configurations_known: false,
                    configurations: Vec::new(),
                }),
                maven: (kind == JdtlsImportedBuildKind::Maven).then(|| MavenProject {
                    pom: ProjectPath {
                        worktree_id: module_root.worktree_id,
                        path: RelPath::from_unix_str("pom.xml")
                            .expect("fixture POM path")
                            .into(),
                    },
                    modules: Vec::new(),
                    group_id: "fixture".into(),
                    artifact_id: "application".into(),
                    version: "1".into(),
                    packaging: "jar".into(),
                    profiles: Default::default(),
                    profile_definitions_known: false,
                    profile_definitions: Vec::new(),
                    reactor_position: None,
                    reactor_order_problem: None,
                    plugins: Vec::new(),
                    lifecycles: vec![MavenLifecycle {
                        id: "default".into(),
                        phases: vec!["verify".into()],
                    }],
                    dependency_graph: None,
                }),
            };
            let snapshot = JdtlsProjectSnapshot {
                metadata: None,
                server_id: language::LanguageServerId(7),
                worktree_id: module_root.worktree_id,
                generation: 1,
                projects: vec![project],
                partial: false,
            };
            let plan = JvmBuildOperationReceipt::new(
                &snapshot,
                true,
                module_root.clone(),
                vec![operation],
                PathStyle::Windows,
            )
            .expect("current JVM operation receipt")
            .plan(&snapshot, true, PathStyle::Windows)
            .expect("current Windows JVM plan");
            assert_eq!(plan.wrapper.path.as_unix_str(), wrapper);
            let task = plan
                .template
                .resolve_task_with_literal_arguments("windows-jvm-operation", &context)
                .expect("resolve actual JVM operation operands")
                .resolved;
            assert!(task.literal_command);
            assert_eq!(task.literal_prefix_args, task.args.len());
            assert_eq!(
                task.command.as_deref(),
                Some(format!(r"C:\fixture\{wrapper}").as_str())
            );
            for program in ["cmd.exe", "powershell.exe", "pwsh.exe"] {
                let prepared = prepare_task_for_spawn(&task, &Shell::Program(program.into()), true);
                assert!(!prepared.windows_direct_program);
                assert_eq!(prepared.command.as_deref(), Some(program));
                let command = prepared.args.last().expect("existing shell command");
                assert!(command.contains(wrapper));
                for argument in &task.args {
                    assert!(command.contains(argument), "missing operation operand");
                }
                assert_eq!(prepared.cwd, task.cwd);
            }
        }
        let cargo = TaskTemplate {
            label: "Native Cargo control".into(),
            command: r"C:\tool.chains\cargo.exe".into(),
            args: vec!["test".into()],
            ..Default::default()
        }
        .resolve_task_with_literal_arguments("windows-cargo-control", &context)
        .expect("resolve native Cargo control")
        .resolved;
        let prepared = prepare_task_for_spawn(&cargo, &Shell::Program("cmd.exe".into()), true);
        assert!(prepared.windows_direct_program);
        assert_eq!(prepared.command, cargo.command);
        assert_eq!(prepared.args, cargo.args);
    }

    #[test]
    fn test_prepare_empty_task() {
        let input = SpawnInTerminal::default();
        let shell = Shell::System;

        let result = prepare_task_for_spawn(&input, &shell, false);

        let expected_shell = util::get_system_shell();
        assert_eq!(result.env, HashMap::default());
        assert_eq!(result.cwd, None);
        assert_eq!(result.shell, Shell::System);
        assert_eq!(
            result.command,
            Some(expected_shell.clone()),
            "Empty tasks should spawn a -i shell"
        );
        assert_eq!(result.args, Vec::<String>::new());
        assert_eq!(
            result.command_label, expected_shell,
            "We show the shell launch for empty commands"
        );
    }

    #[gpui::test]
    async fn test_bypass_max_tabs_limit(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));

        let terminal_panel = window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    cx.new(|cx| TerminalPanel::new(workspace, window, cx))
                })
            })
            .unwrap();

        set_max_tabs(cx, Some(3));

        for _ in 0..5 {
            let task = window_handle
                .update(cx, |_, window, cx| {
                    terminal_panel.update(cx, |panel, cx| {
                        panel.add_terminal_shell(None, RevealStrategy::Always, window, cx)
                    })
                })
                .unwrap();
            task.await.unwrap();
        }

        cx.run_until_parked();

        let item_count =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());

        assert_eq!(
            item_count, 5,
            "Terminal panel should bypass max_tabs limit and have all 5 terminals"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_prepare_script_like_task() {
        let user_command = r#"REPO_URL=$(git remote get-url origin | sed -e \"s/^git@\\(.*\\):\\(.*\\)\\.git$/https:\\/\\/\\1\\/\\2/\"); COMMIT_SHA=$(git log -1 --format=\"%H\" -- \"${ZED_RELATIVE_FILE}\"); echo \"${REPO_URL}/blob/${COMMIT_SHA}/${ZED_RELATIVE_FILE}#L${ZED_ROW}-$(echo $(($(wc -l <<< \"$ZED_SELECTED_TEXT\") + $ZED_ROW - 1)))\" | xclip -selection clipboard"#.to_string();
        let expected_cwd = PathBuf::from("/some/work");

        let input = SpawnInTerminal {
            command: Some(user_command.clone()),
            cwd: Some(expected_cwd.clone()),
            ..SpawnInTerminal::default()
        };
        let shell = Shell::System;

        let result = prepare_task_for_spawn(&input, &shell, false);

        let system_shell = util::get_system_shell();
        assert_eq!(result.env, HashMap::default());
        assert_eq!(result.cwd, Some(expected_cwd));
        assert_eq!(result.shell, Shell::System);
        assert_eq!(result.command, Some(system_shell.clone()));
        assert_eq!(
            result.args,
            vec!["-i".to_string(), "-c".to_string(), user_command.clone()],
            "User command should have been moved into the arguments, as we're spawning a new -i shell",
        );
        assert_eq!(
            result.command_label,
            format!(
                "{system_shell} {interactive}-c '{user_command}'",
                interactive = if cfg!(windows) { "" } else { "-i " }
            ),
            "We want to show to the user the entire command spawned"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_prepare_literal_zero_argument_installed_program_reaches_actual_receiver() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("owned zero-argument fixture");
        let receiver = directory.path().join("installed profiler $ZED_TEST_DATA");
        let result = directory.path().join("received.txt");
        std::fs::write(
            &receiver,
            "#!/bin/sh\nprintf '%s' \"$#\" > \"$ZED_TEST_RESULT\"\n",
        )
        .expect("write installed-program receiver");
        std::fs::set_permissions(&receiver, std::fs::Permissions::from_mode(0o755))
            .expect("make owned receiver executable");
        let ordinary = SpawnInTerminal {
            command: Some(receiver.to_str().expect("UTF-8 fixture").into()),
            ..Default::default()
        };
        let shell = Shell::Program("/bin/sh".into());
        let run = |task: &SpawnInTerminal| {
            let prepared = prepare_task_for_spawn(task, &shell, false);
            smol::block_on(async {
                smol::process::Command::new(prepared.command.expect("prepared shell"))
                    .args(prepared.args)
                    .env("ZED_TEST_RESULT", &result)
                    .env("ZED_TEST_DATA", "expanded")
                    .output()
                    .await
                    .expect("execute actual prepared shell")
            })
        };
        let negative = run(&ordinary);
        assert!(
            !negative.status.success(),
            "raw zero-argument command must fail on the spaced executable"
        );
        assert!(!result.exists());
        let literal = SpawnInTerminal {
            literal_command: true,
            ..ordinary
        };
        assert!(literal.args.is_empty());
        assert_eq!(literal.literal_prefix_args, 0);
        let positive = run(&literal);
        assert!(
            positive.status.success(),
            "{}",
            String::from_utf8_lossy(&positive.stderr)
        );
        assert_eq!(
            std::fs::read_to_string(&result).expect("actual receiver result"),
            "0"
        );
        std::fs::remove_file(&result).expect("retire owned result");
        let authored = SpawnInTerminal {
            command: Some("printf authored > \"$ZED_TEST_RESULT\"".into()),
            ..Default::default()
        };
        assert!(!authored.literal_command);
        assert!(run(&authored).status.success());
        assert_eq!(
            std::fs::read_to_string(result).expect("ordinary shell fragment"),
            "authored"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_prepare_wrapped_task_preserves_user_command_label() {
        let input = SpawnInTerminal {
            command: Some("/host/zed-remote-server".into()),
            args: vec![
                "observe-task".into(),
                "--socket".into(),
                "/private/task.sock".into(),
                "--".into(),
                "cargo".into(),
                "test".into(),
            ],
            command_label: "cargo test".into(),
            preserve_command_label: true,
            literal_prefix_args: 5,
            ..SpawnInTerminal::default()
        };

        let prepared = prepare_task_for_spawn(&input, &Shell::System, false);

        assert_eq!(prepared.command_label, "cargo test");
        assert!(prepared.args.iter().any(|arg| arg.contains("observe-task")));
        assert!(prepared.preserve_command_label);
    }

    #[cfg(unix)]
    #[test]
    fn test_prepare_wrapped_task_quotes_helper_and_arguments() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary wrapped Task directory");
        let helper = directory.path().join("host helper");
        let output = directory.path().join("arguments.txt");
        std::fs::write(
            &helper,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ZED_TEST_CAPTURE_FILE\"\n",
        )
        .expect("write helper fixture");
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755))
            .expect("make helper executable");
        let expanded_argument = "$ZED_TEST_TASK_VALUE";
        let original = SpawnInTerminal {
            command: Some(helper.to_string_lossy().into_owned()),
            args: vec![
                "observe-task".into(),
                "--socket".into(),
                "/private/socket path".into(),
                "--".into(),
                "cargo".into(),
                expanded_argument.into(),
            ],
            command_label: "cargo test".into(),
            preserve_command_label: true,
            literal_prefix_args: 5,
            ..SpawnInTerminal::default()
        };
        let prepared = prepare_task_for_spawn(&original, &Shell::Program("/bin/sh".into()), false);
        let status = smol::block_on(async {
            smol::process::Command::new(
                prepared.command.as_deref().expect("prepared shell command"),
            )
            .args(&prepared.args)
            .env("ZED_TEST_CAPTURE_FILE", &output)
            .env("ZED_TEST_TASK_VALUE", "expanded")
            .status()
            .await
        })
        .expect("run quoted helper");
        assert!(status.success());
        assert_eq!(prepared.command_label, "cargo test");
        assert_eq!(
            std::fs::read_to_string(output).expect("captured helper arguments"),
            "observe-task\n--socket\n/private/socket path\n--\ncargo\nexpanded\n"
        );
    }

    #[gpui::test]
    async fn renders_error_if_default_shell_fails(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.terminal.get_or_insert_default().project.shell =
                        Some(settings::Shell::Program("__nonexistent_shell__".to_owned()));
                });
            });
        });

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));

        let terminal_panel = window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    cx.new(|cx| TerminalPanel::new(workspace, window, cx))
                })
            })
            .unwrap();

        window_handle
            .update(cx, |_, window, cx| {
                terminal_panel.update(cx, |terminal_panel, cx| {
                    terminal_panel.add_terminal_shell(None, RevealStrategy::Always, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap_err();

        window_handle
            .update(cx, |_, _, cx| {
                terminal_panel.update(cx, |terminal_panel, cx| {
                    assert!(
                        terminal_panel
                            .active_pane
                            .read(cx)
                            .items()
                            .any(|item| item.downcast::<FailedToSpawnTerminal>().is_some()),
                        "should spawn `FailedToSpawnTerminal` pane"
                    );
                })
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_local_terminal_in_local_project(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        #[cfg(unix)]
        cx.update_global(|store: &mut SettingsStore, cx| {
            store.update_user_settings(cx, |settings| {
                settings.terminal.get_or_insert_default().project.shell =
                    Some(settings::Shell::Program("/bin/sh".to_owned()));
            });
        });

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));

        let terminal_panel = window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    cx.new(|cx| TerminalPanel::new(workspace, window, cx))
                })
            })
            .unwrap();

        let result = window_handle
            .update(cx, |_, window, cx| {
                terminal_panel.update(cx, |terminal_panel, cx| {
                    terminal_panel.add_local_terminal_shell(RevealStrategy::Always, window, cx)
                })
            })
            .unwrap()
            .await;

        assert!(
            result.is_ok(),
            "local terminal should successfully create in local project: {result:?}"
        );
    }

    struct FocusOnlyModal {
        focus_handle: gpui::FocusHandle,
    }
    impl gpui::EventEmitter<gpui::DismissEvent> for FocusOnlyModal {}
    impl gpui::Focusable for FocusOnlyModal {
        fn focus_handle(&self, _: &gpui::App) -> gpui::FocusHandle {
            self.focus_handle.clone()
        }
    }
    impl Render for FocusOnlyModal {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            gpui::div().track_focus(&self.focus_handle)
        }
    }
    impl workspace::ModalView for FocusOnlyModal {}

    async fn open_center_display_terminal(
        workspace: &Entity<Workspace>,
        cx: &mut VisualTestContext,
    ) {
        workspace
            .update_in(cx, |workspace, window, cx| {
                TerminalPanel::add_center_terminal(workspace, window, cx, |_, cx| {
                    let terminal = cx.new(|cx| {
                        terminal::TerminalBuilder::new_display_only(
                            terminal::terminal_settings::CursorShape::default(),
                            terminal::terminal_settings::AlternateScroll::On,
                            None,
                            0,
                            cx.background_executor(),
                            util::paths::PathStyle::local(),
                        )
                        .subscribe(cx)
                    });
                    gpui::Task::ready(Ok(terminal))
                })
            })
            .await
            .unwrap();
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn test_center_terminal_keeps_focus_on_active_modal(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window_handle
            .update(cx, |multi_workspace, _, _| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let cx = &mut VisualTestContext::from_window(window_handle.into(), cx);

        let modal_focus_handle = workspace.update_in(cx, |workspace, window, cx| {
            let focus_handle = cx.focus_handle();
            workspace.toggle_modal(window, cx, {
                let focus_handle = focus_handle.clone();
                move |_, _| FocusOnlyModal { focus_handle }
            });
            focus_handle
        });

        workspace.update_in(cx, |workspace, window, cx| {
            assert!(workspace.has_active_modal(window, cx));
            assert!(
                modal_focus_handle.is_focused(window),
                "the modal should hold focus before the terminal is created"
            );
        });

        open_center_display_terminal(&workspace, cx).await;

        workspace.update_in(cx, |_, window, _| {
            assert!(
                modal_focus_handle.is_focused(window),
                "a background center terminal must not steal focus from an active modal"
            );
        });
    }

    #[gpui::test]
    async fn test_center_terminal_takes_focus_without_modal(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window_handle
            .update(cx, |multi_workspace, _, _| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let cx = &mut VisualTestContext::from_window(window_handle.into(), cx);

        open_center_display_terminal(&workspace, cx).await;

        workspace.update_in(cx, |workspace, window, cx| {
            assert!(!workspace.has_active_modal(window, cx));
            let terminal_view = workspace
                .active_pane()
                .read(cx)
                .active_item()
                .and_then(|item| item.downcast::<TerminalView>())
                .expect("the new center terminal should be the active item");
            assert!(
                terminal_view.focus_handle(cx).contains_focused(window, cx),
                "with no modal open, a new center terminal should take focus"
            );
        });
    }

    #[cfg(feature = "agentic")]
    #[gpui::test]
    async fn test_inline_assist_tooltip_shows_keybinding_of_active_terminal(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        init_test(cx);

        cx.update(|cx| {
            cx.bind_keys([gpui::KeyBinding::new(
                "ctrl-enter",
                InlineAssist::default(),
                Some("Terminal"),
            )])
        });

        let (window_handle, terminal_panel) = init_workspace_with_panel(cx).await;
        let cx = &mut VisualTestContext::from_window(window_handle.into(), cx);

        terminal_panel.update(cx, |panel, cx| panel.set_assistant_enabled(true, cx));
        terminal_panel
            .update_in(cx, |panel, window, cx| {
                panel.add_terminal_shell(None, RevealStrategy::Always, window, cx)
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let button_bounds = cx
            .debug_bounds("ICON-ZedAssistant")
            .expect("inline assist button should be rendered in the terminal tab bar");
        cx.simulate_mouse_move(button_bounds.center(), None, Modifiers::default());

        cx.executor().advance_clock(Duration::from_millis(600));
        cx.run_until_parked();

        assert!(
            cx.debug_bounds("KEY_BINDING-enter").is_some(),
            "tooltip should show the InlineAssist keybinding resolved in the terminal's context"
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn terminal_task_creation_drop_retires_pending_count_and_recovers(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        init_test(cx);
        let (window, panel) = init_workspace_with_panel(cx).await;
        let root = tempfile::tempdir().expect("Owned ordinary creation fixture");
        let marker = root.path().join("created");
        let project = window
            .update(cx, |workspace, _, cx| {
                workspace.workspace().read(cx).project().clone()
            })
            .expect("Actual terminal Project owner");
        let environment = project.read_with(cx, |project, _| project.environment().clone());
        let (entered_sender, mut entered_receiver) = oneshot::channel();
        let (resume_sender, resume_receiver) = oneshot::channel();
        environment.update(cx, |environment, _| {
            environment
                .delay_next_local_directory_environment_for_test(
                    Arc::from(root.path()),
                    entered_sender,
                    resume_receiver,
                )
                .expect("Hold the genuine pre-launch environment boundary");
        });
        let task = SpawnInTerminal {
            id: TaskId("creation-drop-recovery".into()),
            cwd: Some(root.path().to_path_buf()),
            full_label: "Creation drop recovery".into(),
            label: "Creation drop recovery".into(),
            command: Some("/bin/sh".into()),
            args: vec![
                "-c".into(),
                "printf created > \"$1\"".into(),
                "fixture".into(),
                marker.to_str().expect("UTF-8 owned marker").into(),
            ],
            ..Default::default()
        };
        let scheduled = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.add_terminal_task(task.clone(), RevealStrategy::Never, window, cx)
                })
            })
            .expect("Schedule genuine ordinary Project creation");
        // A pending panel counter does not establish that the background worker has not
        // launched a child. Hold the actual environment await before terminal preparation.
        cx.run_until_parked();
        assert_eq!(
            entered_receiver
                .try_recv()
                .expect("Environment entry receipt"),
            Some(()),
            "Actual Project creation must enter the held environment"
        );
        panel.update(cx, |panel, cx| {
            assert_eq!(panel.pending_terminals_to_add, 1);
            assert!(panel.terminals_for_task(&task.full_label, cx).is_empty());
        });
        assert!(
            !marker.exists(),
            "The queued creation has not started a child"
        );
        drop(scheduled);
        cx.run_until_parked();
        assert!(
            resume_sender.send(Some(HashMap::default())).is_err(),
            "Dropped creation must release the pending environment receiver"
        );
        cx.run_until_parked();
        panel.update(cx, |panel, cx| {
            assert_eq!(panel.pending_terminals_to_add, 0);
            assert!(panel.terminals_for_task(&task.full_label, cx).is_empty());
        });
        assert!(
            !marker.exists(),
            "Dropped creation cannot execute the child"
        );
        let fresh = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.add_terminal_task(task, RevealStrategy::Never, window, cx)
                })
            })
            .expect("Schedule fresh ordinary child recovery");
        let terminal = fresh.await.expect("Create actual recovery terminal");
        let completed = terminal
            .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
            .expect("Actual recovery completion owner");
        let deadline = cx.background_executor.timer(Duration::from_secs(30));
        futures::pin_mut!(completed, deadline);
        let status = match futures::future::select(completed, deadline).await {
            futures::future::Either::Left((status, _)) => status.expect("Actual child completion"),
            futures::future::Either::Right(_) => panic!("Actual recovery child timed out"),
        };
        assert!(status.success());
        assert_eq!(
            std::fs::read_to_string(marker).expect("Actual child marker"),
            "created"
        );
        panel.update(cx, |panel, _| assert_eq!(panel.pending_terminals_to_add, 0));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn terminal_task_creation_refusal_retires_pending_count_and_recovers(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        init_test(cx);
        let (window, panel) = init_workspace_with_panel(cx).await;
        let task = SpawnInTerminal {
            id: TaskId("creation-refusal-recovery".into()),
            full_label: "Creation refusal recovery".into(),
            label: "Creation refusal recovery".into(),
            command: Some("/bin/sh".into()),
            args: vec!["-c".into(), "exit 0".into()],
            ..Default::default()
        };
        let handle = StructuredTaskHandle::new(task.id.clone());
        cx.update(|cx| assert!(handle.cancel(cx)));
        let refused = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.add_terminal_task_with_structured_handle(
                        task.clone(),
                        RevealStrategy::Never,
                        Some(handle.clone()),
                        None,
                        None,
                        window,
                        cx,
                    )
                })
            })
            .expect("Schedule genuine Project creation refusal");
        match refused.await {
            Err(error) => assert_eq!(
                error.to_string(),
                "The structured Task was cancelled before environment or terminal preparation"
            ),
            Ok(_) => panic!("Cancelled Project creation cannot create a terminal"),
        }
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::Cancelled { .. }
        ));
        panel.update(cx, |panel, cx| {
            assert_eq!(panel.pending_terminals_to_add, 0);
            assert!(panel.terminals_for_task(&task.full_label, cx).is_empty());
        });
        let spawned = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.add_terminal_task(task, RevealStrategy::Never, window, cx)
                })
            })
            .expect("Schedule fresh ordinary recovery");
        let terminal = spawned.await.expect("Create actual recovery terminal");
        let completed = terminal
            .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
            .expect("Read actual recovery completion");
        let deadline = cx.background_executor.timer(Duration::from_secs(30));
        futures::pin_mut!(completed, deadline);
        let status = match futures::future::select(completed, deadline).await {
            futures::future::Either::Left((status, _)) => status.expect("Actual child completion"),
            futures::future::Either::Right(_) => panic!("Actual recovery child timed out"),
        };
        assert!(status.success());
        panel.update(cx, |panel, _| assert_eq!(panel.pending_terminals_to_add, 0));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn terminal_pending_experimental_preparation_and_queue_reject_off_then_on(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        init_test(cx);
        let (window, panel) = init_workspace_with_panel(cx).await;
        let directory = tempfile::tempdir().expect("owned pending terminal fixture");
        for feature in [
            "rust-named-cargo-presets-experimental",
            "rust-exact-doctest-actions-experimental",
            "rust-native-profiling-experimental",
            "rust-coverage-experimental",
        ] {
            let marker = directory.path().join(feature);
            let original = SpawnInTerminal {
                id: TaskId(feature.into()),
                full_label: feature.into(),
                label: feature.into(),
                command: Some("/bin/sh".into()),
                args: vec![
                    "-c".into(),
                    "printf launched > \"$PENDING_TASK_MARKER\"".into(),
                ],
                literal_prefix_args: 2,
                cwd: Some(directory.path().to_path_buf()),
                shell: Shell::Program("/bin/sh".into()),
                env: HashMap::from_iter([(
                    "PENDING_TASK_MARKER".into(),
                    marker.to_str().expect("fixture UTF-8").into(),
                )]),
                required_experimental_features: vec![feature.into()],
                reveal: RevealStrategy::Never,
                use_new_terminal: true,
                ..Default::default()
            };
            cx.update(|cx| {
                project::task_store::set_experimental_feature_for_test(feature, true, cx)
            });
            let (release, held) = oneshot::channel();
            let entered = std::rc::Rc::new(std::cell::Cell::new(false));
            let prepare: TaskLaunchPreparation = Box::new({
                let entered = entered.clone();
                move |launch, cx| {
                    entered.set(true);
                    cx.spawn(async move |_| {
                        held.await
                            .map_err(|_| anyhow!("actual preparation release disappeared"))?;
                        Ok(PreparedTaskLaunch {
                            launch,
                            completion: None,
                            native_terminal: None,
                            pending_launch: None,
                        })
                    })
                }
            });
            let held_launch = window
                .update(cx, |_, window, cx| {
                    panel.update(cx, |panel, cx| {
                        panel.spawn_task_internal(&original, None, Some(prepare), window, cx)
                    })
                })
                .expect("schedule existing preparation callback");
            cx.run_until_parked();
            assert!(entered.get());
            assert!(!marker.exists());
            cx.update(|cx| {
                project::task_store::set_experimental_feature_for_test(feature, false, cx)
            });
            cx.update(|cx| {
                project::task_store::set_experimental_feature_for_test(feature, true, cx)
            });
            release
                .send(())
                .expect("release actual preparation callback");
            match held_launch.await {
                Err(error) => assert!(error.to_string().contains("opt-in was revoked")),
                Ok(_) => panic!("revoked preparation cannot create a terminal"),
            }
            assert!(!marker.exists());
            for reveal_target in [RevealTarget::Dock, RevealTarget::Center] {
                let task = SpawnInTerminal {
                    reveal_target,
                    ..original.clone()
                };
                let queued = window
                    .update(cx, |_, window, cx| {
                        panel.update(cx, |panel, cx| {
                            panel.spawn_task_internal(&task, None, None, window, cx)
                        })
                    })
                    .expect("queue original terminal handoff")
                    .await
                    .expect("prepare queued terminal handoff");
                cx.update(|cx| {
                    project::task_store::set_experimental_feature_for_test(feature, false, cx)
                });
                cx.update(|cx| {
                    project::task_store::set_experimental_feature_for_test(feature, true, cx)
                });
                match queued.terminal.await {
                    Err(error) => assert!(error.to_string().contains("opt-in was revoked")),
                    Ok(_) => panic!("revoked queue cannot start an actual child"),
                }
                assert!(!marker.exists());
            }
            let project = panel.read_with(cx, |panel, cx| {
                panel
                    .workspace
                    .read_with(cx, |workspace, _| workspace.project().clone())
                    .expect("actual Project owner")
            });
            let direct = project.update(cx, |project, cx| {
                project.create_terminal_task(original.clone(), cx)
            });
            cx.update(|cx| {
                project::task_store::set_experimental_feature_for_test(feature, false, cx)
            });
            cx.update(|cx| {
                project::task_store::set_experimental_feature_for_test(feature, true, cx)
            });
            match direct.await {
                Err(error) => assert!(error.to_string().contains("opt-in was revoked")),
                Ok(_) => panic!(
                    "direct Project environment preparation cannot rejuvenate a revoked Task"
                ),
            }
            assert!(!marker.exists());
            let fresh = window
                .update(cx, |_, window, cx| {
                    panel.update(cx, |panel, cx| {
                        panel.spawn_task_internal(&original, None, None, window, cx)
                    })
                })
                .expect("schedule fresh original Task")
                .await
                .expect("fresh preparation");
            let terminal = fresh.terminal.await.expect("actual fresh Terminal");
            let completed = terminal
                .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                .expect("actual kernel completion");
            let deadline = cx.background_executor.timer(Duration::from_secs(30));
            match futures::future::select(Box::pin(completed), Box::pin(deadline)).await {
                futures::future::Either::Left((status, _)) => {
                    assert!(status.expect("actual child status").success())
                }
                futures::future::Either::Right(_) => panic!("fresh actual child timed out"),
            }
            assert_eq!(
                std::fs::read_to_string(marker).expect("actual child marker"),
                "launched"
            );
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn terminal_spawned_named_task_survives_opt_in_revocation(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);
        let (window, panel) = init_workspace_with_panel(cx).await;
        let directory = tempfile::tempdir().expect("owned running terminal fixture");
        let started = directory.path().join("started");
        let finished = directory.path().join("finished");
        let feature = "rust-named-cargo-presets-experimental";
        cx.update(|cx| project::task_store::set_experimental_feature_for_test(feature, true, cx));
        let task = SpawnInTerminal {
            id: TaskId("already-spawned-named".into()),
            full_label: "already-spawned-named".into(),
            command: Some("/bin/sh".into()),
            args: vec!["-c".into(), "printf started > \"$PENDING_TASK_STARTED\"; read release; printf finished > \"$PENDING_TASK_FINISHED\"".into()],
            literal_prefix_args: 2,
            shell: Shell::Program("/bin/sh".into()),
            cwd: Some(directory.path().to_path_buf()),
            env: HashMap::from_iter([("PENDING_TASK_STARTED".into(), started.to_str().expect("fixture UTF-8").into()), ("PENDING_TASK_FINISHED".into(), finished.to_str().expect("fixture UTF-8").into())]),
            required_experimental_features: vec![feature.into()],
            reveal: RevealStrategy::Never,
            use_new_terminal: true,
            ..Default::default()
        };
        let handle = StructuredTaskHandle::new(task.id.clone());
        let prepared = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.spawn_task_internal(&task, Some(handle.clone()), None, window, cx)
                })
            })
            .expect("schedule actual running Task")
            .await
            .expect("prepare actual running Task");
        let terminal = prepared.terminal.await.expect("create actual running Task");
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !started.is_file() {
            assert!(
                std::time::Instant::now() < deadline,
                "actual child never entered stdin wait"
            );
            cx.background_executor
                .timer(Duration::from_millis(10))
                .await;
        }
        assert!(!finished.exists());
        let queued_task = SpawnInTerminal {
            use_new_terminal: false,
            ..task.clone()
        };
        let queued = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.spawn_task_internal(&queued_task, None, None, window, cx)
                })
            })
            .expect("queue actual predecessor replacement");
        cx.run_until_parked();
        assert!(!finished.exists());
        cx.update(|cx| project::task_store::set_experimental_feature_for_test(feature, false, cx));
        cx.run_until_parked();
        cx.update(|cx| project::task_store::set_experimental_feature_for_test(feature, true, cx));
        assert!(
            matches!(handle.state(), task::StructuredTaskState::Running { .. }),
            "pending latch must retire before PTY creation, without cancelling spawned ordinary child"
        );
        terminal
            .update(cx, |terminal, _| terminal.input(b"release\n".to_vec()))
            .expect("release original child stdin");
        let completed = terminal
            .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
            .expect("actual kernel completion");
        let timer = cx
            .background_executor
            .timer(deadline.saturating_duration_since(std::time::Instant::now()));
        match futures::future::select(Box::pin(completed), Box::pin(timer)).await {
            futures::future::Either::Left((status, _)) => {
                assert!(status.expect("actual child completion").success())
            }
            futures::future::Either::Right(_) => {
                panic!("already-spawned Task failed to finish after revocation")
            }
        }
        assert_eq!(
            std::fs::read_to_string(finished).expect("actual child completion marker"),
            "finished"
        );
        let timer = cx
            .background_executor
            .timer(deadline.saturating_duration_since(std::time::Instant::now()));
        match futures::future::select(Box::pin(queued), Box::pin(timer)).await {
            futures::future::Either::Left((Err(error), _)) => {
                assert!(error.to_string().contains("opt-in was revoked"))
            }
            futures::future::Either::Left((Ok(_), _)) => panic!(
                "Off then On cannot rejuvenate deferred replacement after original child finishes"
            ),
            futures::future::Either::Right(_) => {
                panic!("revoked deferred replacement did not retire before original deadline")
            }
        }
    }

    #[gpui::test]
    async fn terminal_task_preparation_holds_logical_launch_before_terminal_creation(
        cx: &mut TestAppContext,
    ) {
        use std::{cell::Cell, rc::Rc};
        cx.executor().allow_parking();
        init_test(cx);
        let (window, panel) = init_workspace_with_panel(cx).await;
        let original = SpawnInTerminal {
            id: TaskId("held-preparation".into()),
            full_label: "Held preparation".into(),
            label: "Held preparation".into(),
            command: Some("/private/fixture collector".into()),
            args: vec!["literal '$HOME $(unselected)'".into(), "".into()],
            literal_prefix_args: 2,
            ..Default::default()
        };
        let calls = Rc::new(Cell::new(0));
        let (sender, receiver) = oneshot::channel();
        let prepare: TaskLaunchPreparation = Box::new({
            let expected = original.clone();
            let calls = calls.clone();
            move |logical, cx| {
                assert_eq!(
                    logical, expected,
                    "Preparation receives logical argv before shell wrapping"
                );
                calls.set(calls.get() + 1);
                cx.spawn(async move |_| {
                    receiver
                        .await
                        .map_err(|_| anyhow!("Held test receiver expired"))?;
                    Err(anyhow!("Controlled preparation rejection"))
                })
            }
        });
        let scheduled = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.spawn_task_internal(&original, None, Some(prepare), window, cx)
                })
            })
            .expect("Schedule actual terminal-panel preparation");
        cx.run_until_parked();
        assert_eq!(calls.get(), 1);
        panel.update(cx, |panel, cx| {
            assert_eq!(panel.pending_terminals_to_add, 0);
            assert!(
                panel
                    .terminals_for_task(&original.full_label, cx)
                    .is_empty()
            );
        });
        sender.send(()).expect("Release controlled rejection");
        match scheduled.await {
            Err(error) => assert_eq!(error.to_string(), "Controlled preparation rejection"),
            Ok(_) => panic!("Rejected preparation cannot create a terminal"),
        }
        panel.update(cx, |panel, cx| {
            assert_eq!(panel.pending_terminals_to_add, 0);
            assert!(
                panel
                    .terminals_for_task(&original.full_label, cx)
                    .is_empty()
            );
        });
    }

    #[gpui::test]
    async fn terminal_task_preparation_rejects_changed_scheduling_identity_and_drops_completion(
        cx: &mut TestAppContext,
    ) {
        use std::{cell::Cell, rc::Rc};
        struct DroppedCompletion(Rc<Cell<usize>>);
        impl Drop for DroppedCompletion {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        cx.executor().allow_parking();
        init_test(cx);
        let (window, panel) = init_workspace_with_panel(cx).await;
        for changed_field in [
            "id",
            "label",
            "concurrency",
            "placement",
            "shell",
            "save",
            "prefix",
        ] {
            let original = SpawnInTerminal {
                id: TaskId(format!("changed-{changed_field}")),
                full_label: format!("Changed {changed_field}"),
                label: "Original".into(),
                command: Some("/private/fixture-collector".into()),
                ..Default::default()
            };
            let dropped = Rc::new(Cell::new(0));
            let completed = Rc::new(Cell::new(false));
            let prepare: TaskLaunchPreparation = Box::new({
                let dropped = dropped.clone();
                let completed = completed.clone();
                move |mut launch, _| {
                    match changed_field {
                        "id" => launch.id = TaskId("other".into()),
                        "label" => launch.full_label = "other".into(),
                        "concurrency" => launch.allow_concurrent_runs = true,
                        "placement" => launch.use_new_terminal = true,
                        "shell" => launch.shell = Shell::Program("/bin/sh".into()),
                        "save" => launch.save = task::SaveStrategy::All,
                        "prefix" => launch.literal_prefix_args = 1,
                        _ => panic!("Unknown controlled field"),
                    }
                    let guard = DroppedCompletion(dropped);
                    Task::ready(Ok(PreparedTaskLaunch {
                        launch,
                        native_terminal: None,
                        pending_launch: None,
                        completion: Some(Box::new(move |result, _| {
                            completed.set(true);
                            drop(guard);
                            result
                        })),
                    }))
                }
            });
            let scheduled = window
                .update(cx, |_, window, cx| {
                    panel.update(cx, |panel, cx| {
                        panel.spawn_task_internal(&original, None, Some(prepare), window, cx)
                    })
                })
                .expect("Actual checked-preparation route");
            match scheduled.await {
                Err(error) => assert_eq!(
                    error.to_string(),
                    "Task preparation changed scheduling identity or supplied an invalid literal prefix"
                ),
                Ok(_) => panic!("Changed identity cannot launch"),
            }
            assert_eq!(
                dropped.get(),
                1,
                "Invalid launch drops its bounded observer ownership"
            );
            assert!(
                !completed.get(),
                "No fabricated terminal completion is supplied"
            );
            panel.update(cx, |panel, cx| {
                assert_eq!(panel.pending_terminals_to_add, 0);
                assert!(
                    panel
                        .terminals_for_task(&original.full_label, cx)
                        .is_empty()
                );
            });
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn terminal_task_preparation_waits_for_selected_reused_predecessor(
        cx: &mut TestAppContext,
    ) {
        use std::{cell::Cell, rc::Rc};
        cx.executor().allow_parking();
        init_test(cx);
        let root = tempfile::tempdir().expect("Owned predecessor fixture");
        let fifo = root.path().join("release.fifo");
        let started = root.path().join("started");
        let finished = root.path().join("finished");
        assert!(
            smol::process::Command::new("/usr/bin/mkfifo")
                .arg(&fifo)
                .status()
                .await
                .expect("Create owned FIFO")
                .success()
        );
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.path()],
            cx,
        )
        .await;
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let panel = window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    panel
                })
            })
            .expect("Actual TerminalPanel");
        let predecessor = SpawnInTerminal {
            id: TaskId("selected-predecessor".into()),
            label: "Selected predecessor".into(),
            full_label: "Selected predecessor".into(),
            command: Some("/bin/sh".into()),
            args: vec![
                "-c".into(),
                r#"printf started > "$1"; /bin/cat "$2" > "$3.pending" && /bin/mv "$3.pending" "$3""#.into(),
                "predecessor".into(),
                started.to_str().expect("Started path").into(),
                fifo.to_str().expect("FIFO path").into(),
                finished.to_str().expect("Finished path").into(),
            ],
            literal_prefix_args: 6,
            cwd: Some(root.path().to_path_buf()),
            shell: Shell::Program("/bin/sh".into()),
            ..Default::default()
        };
        let first = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| panel.spawn_task(&predecessor, window, cx))
            })
            .expect("Spawn actual predecessor")
            .await
            .expect("Predecessor terminal");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !started.is_file() {
            assert!(
                std::time::Instant::now() < deadline,
                "Actual predecessor did not enter its owned FIFO wait"
            );
            cx.background_executor
                .timer(Duration::from_millis(10))
                .await;
        }
        assert!(!finished.exists());
        assert!(
            first
                .read_with(cx, |terminal, _| terminal.task().is_some())
                .expect("Live predecessor")
        );
        let calls = Rc::new(Cell::new(0));
        let prepare: TaskLaunchPreparation = Box::new({
            let calls = calls.clone();
            move |_, _| {
                calls.set(calls.get() + 1);
                Task::ready(Err(anyhow!("Controlled successor rejection")))
            }
        });
        let successor = window
            .update(cx, |_, window, cx| {
                panel.update(cx, |panel, cx| {
                    panel.spawn_task_internal(&predecessor, None, Some(prepare), window, cx)
                })
            })
            .expect("Schedule nonconcurrent reused successor");
        cx.run_until_parked();
        assert_eq!(
            calls.get(),
            0,
            "Selected reused predecessor must remain in eligibility wait"
        );
        let fifo = fifo.clone();
        let release = cx
            .background_executor
            .spawn(async move { std::fs::write(fifo, b"finished") });
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let timer = cx.background_executor.timer(remaining);
        match futures::future::select(Box::pin(release), Box::pin(timer)).await {
            futures::future::Either::Left((result, _)) => {
                result.expect("Release actual owned FIFO predecessor")
            }
            futures::future::Either::Right(_) => {
                panic!("Actual predecessor did not read its release before the original deadline")
            }
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let timer = cx.background_executor.timer(remaining);
        match futures::future::select(Box::pin(successor), Box::pin(timer)).await {
            futures::future::Either::Left((Err(error), _)) => {
                assert_eq!(error.to_string(), "Controlled successor rejection")
            }
            futures::future::Either::Left((Ok(_), _)) => {
                panic!("Rejected successor cannot replace its predecessor")
            }
            futures::future::Either::Right(_) => {
                panic!("Eligible successor did not finish before the original deadline")
            }
        }
        assert_eq!(calls.get(), 1);
        assert_eq!(
            std::fs::read(&finished).expect("Actual predecessor completed"),
            b"finished"
        );
        panel.update(cx, |panel, cx| {
            let remaining = panel.terminals_for_task(&predecessor.full_label, cx);
            assert_eq!(
                remaining.len(),
                1,
                "Rejected preparation preserves predecessor terminal"
            );
            assert_eq!(
                remaining
                    .first()
                    .expect("Original terminal")
                    .2
                    .read(cx)
                    .terminal()
                    .entity_id(),
                first.entity_id()
            );
        });
    }

    async fn init_workspace_with_panel(
        cx: &mut TestAppContext,
    ) -> (gpui::WindowHandle<MultiWorkspace>, Entity<TerminalPanel>) {
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));

        let terminal_panel = window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    panel
                })
            })
            .expect("Failed to initialize workspace with terminal panel");

        (window_handle, terminal_panel)
    }

    #[cfg(all(unix, feature = "rust-coverage"))]
    #[gpui::test]
    async fn coverage_capture_joins_real_terminal_completion_and_rerun_routes(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;

        use collections::{BTreeMap, HashSet};
        use project::{
            TaskSourceKind, source_coverage::SourceCoverageStatus, trusted_worktrees::PathTrust,
        };
        use task::{
            SaveStrategy, TaskArtifact, TaskArtifactGroup, TaskArtifactGroupArgumentBinding,
            TaskArtifactKind, TaskContext, TaskTemplate,
        };
        use workspace::tasks::ScheduledTaskResult;
        use worktree::WorktreeModelHandle as _;

        cx.executor().allow_parking();
        init_test(cx);
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            )
        });
        let root = tempfile::tempdir().expect("real coverage worktree");
        std::fs::create_dir_all(root.path().join("src/bin")).expect("covered source directory");
        std::fs::create_dir(root.path().join("target")).expect("coverage output directory");
        std::fs::write(root.path().join("src/bin/fixture.rs"), "fn main() {}\n")
            .expect("covered source");
        let report = include_str!("../../project/test_data/source_coverage/llvm3-run.json")
            .replace("/workspace/", &format!("{}/", root.path().display()));
        let report_source = root.path().join("fixture-report.json");
        std::fs::write(&report_source, &report).expect("actual LLVM coverage fixture");
        let producer = root.path().join("coverage-producer.sh");
        std::fs::write(&producer,
            "#!/bin/sh\ncp \"$2\" \"$1\" || exit 31\nprintf started > \"$4\"\nwhile [ ! -f \"$5\" ]; do sleep 0.01; done\nexit \"$3\"\n")
            .expect("explicit controlled coverage producer");
        std::fs::set_permissions(&producer, std::fs::Permissions::from_mode(0o700))
            .expect("executable producer");
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.path()],
            cx,
        )
        .await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let (workspace, terminal_panel) = window_handle
            .update(cx, |container, window, cx| {
                let workspace = container.workspace().clone();
                let panel = workspace.update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    workspace.set_terminal_provider(TerminalProvider(panel.clone()));
                    panel
                });
                (workspace, panel)
            })
            .expect("production TerminalPanel provider");
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = project.read_with(cx, |project, cx| {
            let store = project.worktree_store();
            let id = store
                .read(cx)
                .worktrees()
                .next()
                .expect("local worktree")
                .read(cx)
                .id();
            (store, id)
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("trust owner")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let producer_controls = tempfile::tempdir().expect("owned coverage producer controls");
        let started = producer_controls.path().join("producer-started");
        let release = producer_controls.path().join("producer-release");
        std::fs::write(&release, b"release").expect("positive producer admission");
        let template = TaskTemplate {
            label: "coverage terminal producer".into(),
            command: producer.to_string_lossy().into_owned(),
            args: vec![
                "target/coverage.json".into(),
                report_source.to_string_lossy().into_owned(),
                "0".into(),
                started.to_string_lossy().into_owned(),
                release.to_string_lossy().into_owned(),
            ],
            cwd: Some(root.path().to_string_lossy().into_owned()),
            literal_arguments: true,
            save: SaveStrategy::None,
            allow_concurrent_runs: true,
            use_new_terminal: true,
            artifact_group: Some(TaskArtifactGroup {
                format: "llvm-cov-json-v1".into(),
                files: BTreeMap::from_iter([(
                    "report".into(),
                    TaskArtifact {
                        path: "target/coverage.json".into(),
                        kind: TaskArtifactKind::Data,
                        max_bytes: 32 * 1024 * 1024,
                    },
                )]),
            }),
            artifact_group_argument_binding: Some(TaskArtifactGroupArgumentBinding {
                role: "report".into(),
                argument_index: 0,
                related: Vec::new(),
            }),
            ..TaskTemplate::default()
        };
        let id_base = TaskSourceKind::UserInput.to_id_base();
        let initial = template
            .resolve_task(&id_base, &TaskContext::default())
            .expect("coverage Task");
        let stable_id = initial.id.clone();
        let mut cached = initial;
        let mut outputs = HashSet::default();
        let mut generation = 0;
        for route in 0..11 {
            let recovery_timeout = if route == 7 {
                let timeout = Box::pin(cx.background_executor.timer(Duration::from_secs(10)));
                match futures::future::select(
                    Box::pin(async {
                        let absolute_source = root.path().join("src/bin/fixture.rs");
                        let source = project.read_with(cx, |project, cx| {
                            project
                                .find_project_path(&absolute_source, cx)
                                .expect("mapped recovery source")
                        });
                        let buffer = project
                            .update(cx, |project, cx| project.open_buffer(source.clone(), cx))
                            .await
                            .expect("actual dirty recovery Buffer");
                        buffer.read_with(cx, |buffer, _| {
                            assert!(
                                buffer.is_dirty(),
                                "recovery must follow the actual unsaved mapped edit"
                            )
                        });
                        project
                            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
                            .await
                            .expect("actual Project save before fresh recovery");
                        buffer.read_with(cx, |buffer, _| {
                            assert!(
                                !buffer.is_dirty(),
                                "fresh admission requires the actually saved mapped Buffer"
                            )
                        });
                        let (filesystem, worktree) = project.read_with(cx, |project, cx| {
                            (
                                project.fs().clone(),
                                project
                                    .worktrees(cx)
                                    .find(|worktree| worktree.read(cx).id() == source.worktree_id)
                                    .expect("actual recovery Worktree"),
                            )
                        });
                        let metadata = filesystem
                            .metadata(&absolute_source)
                            .await
                            .expect("saved source metadata")
                            .expect("saved source exists");
                        let entry = worktree
                            .update(cx, |worktree, cx| {
                                worktree
                                    .as_local()
                                    .expect("local mapped-source owner")
                                    .refresh_entry(source.path.clone(), None, cx)
                            })
                            .await
                            .expect("actual saved source refresh")
                            .expect("saved source indexed");
                        assert_eq!(
                            (entry.inode, entry.mtime, entry.size),
                            (metadata.inode, Some(metadata.mtime), metadata.len),
                            "saved mapped source is current before fresh admission"
                        );
                        worktree.read_with(cx, |worktree, _| {
                            let indexed = worktree
                                .entry_for_path(&source.path)
                                .expect("current indexed recovery source");
                            assert_eq!(
                                (indexed.inode, indexed.mtime, indexed.size),
                                (metadata.inode, Some(metadata.mtime), metadata.len)
                            );
                        });
                        worktree.flush_fs_events(cx).await;
                    }),
                    timeout,
                )
                .await
                {
                    futures::future::Either::Left(((), timeout)) => Some(timeout),
                    futures::future::Either::Right(_) => {
                        panic!("saved recovery source did not become ready")
                    }
                }
            } else if matches!(route, 0 | 3 | 6 | 8) {
                let worktree = worktree_store.read_with(cx, |store, cx| {
                    store
                        .worktree_for_id(worktree_id, cx)
                        .expect("actual coverage fixture Worktree")
                });
                let timeout = Box::pin(cx.background_executor.timer(Duration::from_secs(10)));
                match futures::future::select(worktree.flush_fs_events(cx), timeout).await {
                    futures::future::Either::Left(((), _)) => None,
                    futures::future::Either::Right(_) => {
                        panic!("prior coverage fixture filesystem events did not settle")
                    }
                }
            } else {
                None
            };
            let mut cancelled_after_exit = None;
            let mut cancelled_while_running = None;
            if started.exists() {
                std::fs::remove_file(&started).expect("reset start evidence");
            }
            let mut task = if route == 1 || route == 10 {
                cached.clone()
            } else {
                template
                    .resolve_task(&id_base, &TaskContext::default())
                    .expect("template coverage Task")
            };
            if route == 3 {
                *task.resolved.args.get_mut(2).expect("exit argument") = "7".into();
            }
            if (4..=6).contains(&route) || route == 8 {
                std::fs::remove_file(&release).expect("hold negative producer");
            }
            let (sender, receiver) = oneshot::channel();
            window_handle
                .update(cx, |_, window, cx| {
                    workspace.update(cx, |workspace, cx| {
                        if route == 6 || route == 8 {
                            workspace.schedule_resolved_task_with_structured_handle_and_completion(
                                TaskSourceKind::UserInput,
                                task,
                                false,
                                move |result, _| {
                                    sender.send(result).expect("Task completion receiver")
                                },
                                window,
                                cx,
                            );
                        } else {
                            workspace.schedule_resolved_task_with_completion(
                                TaskSourceKind::UserInput,
                                task,
                                false,
                                move |result, _| {
                                    sender.send(result).expect("Task completion receiver")
                                },
                                window,
                                cx,
                            );
                        }
                    })
                })
                .expect("schedule actual PTY Task");
            if (4..=6).contains(&route) || route == 8 {
                for _ in 0..100 {
                    if started.exists() {
                        break;
                    }
                    cx.background_executor
                        .timer(Duration::from_millis(50))
                        .await;
                }
                assert!(started.exists(), "negative actual producer started");
                let loading = store.read_with(cx, |store, _| {
                    store.selected_snapshot().expect("live capture").clone()
                });
                assert_eq!(loading.status, SourceCoverageStatus::Loading);
                if route == 4 {
                    let terminal = terminal_panel.update(cx, |panel, cx| {
                        panel
                            .terminals_for_task("coverage terminal producer", cx)
                            .into_iter()
                            .find(|(_, _, view)| {
                                view.read(cx)
                                    .terminal()
                                    .read(cx)
                                    .task()
                                    .is_some_and(|task| {
                                        task.status == terminal::TaskStatus::Running
                                    })
                            })
                            .map(|(_, _, view)| view.read(cx).terminal().clone())
                            .expect("actual running coverage terminal")
                    });
                    terminal.update(cx, |terminal, _| terminal.kill_active_task());
                } else if route == 8 {
                    let terminal = terminal_panel.update(cx, |panel, cx| {
                        panel
                            .terminals_for_task("coverage terminal producer", cx)
                            .into_iter()
                            .find(|(_, _, view)| {
                                view.read(cx)
                                    .terminal()
                                    .read(cx)
                                    .task()
                                    .is_some_and(|task| {
                                        task.status == terminal::TaskStatus::Running
                                    })
                            })
                            .map(|(_, _, view)| view.read(cx).terminal().clone())
                            .expect("actual running user-cancel terminal")
                    });
                    let (process_id, handle) = terminal.read_with(cx, |terminal, _| {
                        (
                            terminal.pid().expect("actual running PTY child"),
                            terminal
                                .task()
                                .expect("actual running coverage Task")
                                .structured_task
                                .clone()
                                .expect("bound user cancellation owner"),
                        )
                    });
                    assert!(
                        !release.exists(),
                        "producer is still held before user cancellation"
                    );
                    assert!(matches!(
                        handle.state(),
                        task::StructuredTaskState::Running { .. }
                    ));
                    let process_id = i32::try_from(process_id.as_u32())
                        .expect("actual Unix PTY child identifier");
                    assert_eq!(
                        unsafe { libc::kill(process_id, 0) },
                        0,
                        "the original PTY child must still be alive before user cancellation"
                    );
                    cx.update(|cx| {
                        assert!(
                            handle.cancel(cx),
                            "the running Task accepts cancellation through its bound Terminal owner"
                        )
                    });
                    assert!(matches!(
                        handle.state(),
                        task::StructuredTaskState::Cancelled { .. }
                    ));
                    cancelled_while_running = Some((terminal, process_id, handle));
                } else if route == 6 {
                    let terminal = terminal_panel.update(cx, |panel, cx| {
                        panel
                            .terminals_for_task("coverage terminal producer", cx)
                            .into_iter()
                            .find(|(_, _, view)| {
                                view.read(cx)
                                    .terminal()
                                    .read(cx)
                                    .task()
                                    .is_some_and(|task| {
                                        task.status == terminal::TaskStatus::Running
                                    })
                            })
                            .map(|(_, _, view)| view.read(cx).terminal().clone())
                            .expect("actual running cancellation-race terminal")
                    });
                    let (process_id, handle) = terminal.read_with(cx, |terminal, _| {
                        (
                            terminal.pid().expect("actual PTY shell process"),
                            terminal
                                .task()
                                .expect("actual Task")
                                .structured_task
                                .clone()
                                .expect("actual structured cancellation owner"),
                        )
                    });
                    std::fs::write(&release, b"release")
                        .expect("release successful child before structured cancellation");
                    // Keep the actual ChildExit foreground event pending while the PTY I/O owner reaps its child.
                    let deadline = std::time::Instant::now() + Duration::from_secs(10);
                    loop {
                        let process_id = i32::try_from(process_id.as_u32())
                            .expect("actual Unix PTY process identifier");
                        let exists = unsafe { libc::kill(process_id, 0) };
                        if exists != 0 {
                            let error = std::io::Error::last_os_error();
                            assert_eq!(
                                error.raw_os_error(),
                                Some(libc::ESRCH),
                                "only the actual child disappearance proves exit before cancellation"
                            );
                            break;
                        }
                        assert!(
                            std::time::Instant::now() < deadline,
                            "actual successful child must exit before cancellation"
                        );
                        cx.dispatcher.tick(true);
                    }
                    terminal.read_with(cx, |terminal, _| {
                        assert_eq!(
                            terminal.task().expect("pending terminal event").status,
                            terminal::TaskStatus::Running
                        );
                    });
                    cx.update(|cx| assert!(handle.cancel(cx),
                        "the actual user cancellation owner is still running before ChildExit dispatch"));
                    assert!(matches!(
                        handle.state(),
                        task::StructuredTaskState::Cancelled { .. }
                    ));
                    cancelled_after_exit = Some(terminal);
                } else {
                    let buffer = project
                        .update(cx, |project, cx| {
                            project.open_buffer(
                                project
                                    .find_project_path(root.path().join("src/bin/fixture.rs"), cx)
                                    .expect("mapped source"),
                                cx,
                            )
                        })
                        .await
                        .expect("actual source buffer");
                    window_handle
                        .update(cx, |_, window, cx| {
                            workspace.update(cx, |workspace, cx| {
                                let editor = cx.new(|cx| {
                                    editor::Editor::for_buffer(
                                        buffer,
                                        Some(project.clone()),
                                        window,
                                        cx,
                                    )
                                });
                                workspace.add_item_to_active_pane(
                                    Box::new(editor.clone()),
                                    None,
                                    true,
                                    window,
                                    cx,
                                );
                                editor.update(cx, |editor, cx| {
                                    editor.edit(
                                        [(
                                            editor::MultiBufferOffset(0)
                                                ..editor::MultiBufferOffset(0),
                                            "// changed during capture\n",
                                        )],
                                        cx,
                                    )
                                });
                            })
                        })
                        .expect("actual Editor dirty-source invalidation");
                    std::fs::write(&release, b"release").expect("release changed-source producer");
                }
            }
            let completion = futures::future::select(
                Box::pin(receiver),
                recovery_timeout.unwrap_or_else(|| {
                    Box::pin(cx.background_executor.timer(Duration::from_secs(10)))
                }),
            )
            .await;
            let result = match completion {
                futures::future::Either::Left((result, _)) => {
                    result.expect("actual terminal completion")
                }
                futures::future::Either::Right(_) => {
                    panic!("coverage terminal completion timed out")
                }
            };
            cx.run_until_parked();
            let (_, latest) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("history owner")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("actual Task history")
            });
            if route < 3 {
                assert_eq!(latest.id, stable_id);
            }
            assert_eq!(latest.original_task(), &template);
            let output = latest
                .resolved_artifact_group()
                .expect("capture intent")
                .files
                .get("report")
                .expect("bound report")
                .path
                .clone();
            assert!(
                outputs.insert(output.clone()),
                "fresh initial/cached/template output"
            );
            assert_eq!(
                std::fs::read_to_string(root.path().join(output)).expect("actual producer report"),
                report
            );
            let snapshot = store.read_with(cx, |store, _| {
                store
                    .selected_snapshot()
                    .expect("capture publication")
                    .clone()
            });
            assert!(snapshot.generation > generation);
            generation = snapshot.generation;
            if route < 3 {
                assert_eq!(result, ScheduledTaskResult::Success);
                assert_eq!(snapshot.status, SourceCoverageStatus::Current);
                assert_eq!(snapshot.files.len(), 1);
                cached = latest;
            } else if route == 3 {
                assert_eq!(result, ScheduledTaskResult::Failure);
                assert_eq!(snapshot.status, SourceCoverageStatus::Error);
            } else if route == 4 {
                assert_ne!(result, ScheduledTaskResult::Success);
                assert_ne!(snapshot.status, SourceCoverageStatus::Current);
                assert_ne!(snapshot.status, SourceCoverageStatus::Loading);
                std::fs::write(&release, b"release").expect("reset release after cancellation");
            } else if route == 5 {
                assert_eq!(result, ScheduledTaskResult::Success);
                assert_eq!(snapshot.status, SourceCoverageStatus::Stale);
            } else if route == 6 {
                assert_eq!(
                    result,
                    ScheduledTaskResult::Cancelled,
                    "a cancelled structured Task cannot report callback success"
                );
                assert_eq!(snapshot.status, SourceCoverageStatus::Cancelled);
                assert!(
                    snapshot.files.is_empty(),
                    "cancelled capture cannot publish Current files"
                );
                let actual_exit = cancelled_after_exit
                    .expect("actual cancellation-race terminal")
                    .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                    .await;
                assert_eq!(
                    actual_exit.and_then(|status| status.code()),
                    Some(0),
                    "core Terminal must retain the actual successful child exit"
                );
            } else if route == 8 {
                assert_eq!(
                    result,
                    ScheduledTaskResult::Cancelled,
                    "user cancellation while the producer runs must reach the ScheduledTask callback"
                );
                assert_eq!(snapshot.status, SourceCoverageStatus::Cancelled);
                assert!(
                    snapshot.files.is_empty(),
                    "cancelled coverage exposes no file/range facts for gutters or source navigation"
                );
                let (terminal, process_id, handle) =
                    cancelled_while_running.expect("actual still-running cancellation owner");
                let completion =
                    terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let actual_exit = match futures::future::select(
                    Box::pin(completion),
                    Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
                )
                .await
                {
                    futures::future::Either::Left((status, _)) => status.expect(
                        "canonical Terminal completion retains the killed child's actual status",
                    ),
                    futures::future::Either::Right(_) => {
                        panic!("user-cancelled PTY child did not complete canonically")
                    }
                };
                assert!(
                    !actual_exit.success(),
                    "the producer cannot exit successfully while its release remains absent"
                );
                cx.run_until_parked();
                assert!(
                    matches!(
                        handle.state(),
                        task::StructuredTaskState::Cancelled {
                            termination_confirmed: true,
                            ..
                        }
                    ),
                    "the real ChildExit confirms termination of the cancelled structured Task"
                );
                terminal.read_with(cx, |terminal, _| {
                    assert_ne!(
                        terminal.task().expect("completed cancelled Task").status,
                        terminal::TaskStatus::Running
                    );
                });
                let exists = unsafe { libc::kill(process_id, 0) };
                assert_ne!(
                    exists, 0,
                    "the original PTY child is absent after canonical completion"
                );
                assert_eq!(
                    std::io::Error::last_os_error().raw_os_error(),
                    Some(libc::ESRCH),
                    "only actual child disappearance confirms the post-completion liveness observation"
                );
                assert!(
                    !release.exists(),
                    "the cancelled producer was never released"
                );
                std::fs::write(&release, b"release").expect(
                    "admit fresh Run and cached-history Rerun after confirmed user cancellation",
                );
            } else {
                assert_eq!(result, ScheduledTaskResult::Success);
                assert_eq!(snapshot.status, SourceCoverageStatus::Current);
                assert_eq!(snapshot.files.len(), 1, "fresh recovery after cancellation");
                if route == 9 {
                    cached = latest;
                }
            }
        }
    }

    #[cfg(all(unix, feature = "rust-coverage"))]
    #[gpui::test]
    async fn coverage_capture_actual_overlapping_tasks_preserve_newest_report(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;

        use collections::{BTreeMap, HashSet};
        use project::{
            TaskSourceKind, source_coverage::SourceCoverageStatus, trusted_worktrees::PathTrust,
        };
        use task::{
            SaveStrategy, TaskArtifact, TaskArtifactGroup, TaskArtifactGroupArgumentBinding,
            TaskArtifactKind, TaskContext, TaskTemplate,
        };
        use workspace::tasks::ScheduledTaskResult;

        cx.executor().allow_parking();
        init_test(cx);
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            )
        });
        let root = tempfile::tempdir().expect("real coverage worktree");
        std::fs::create_dir_all(root.path().join("src/bin")).expect("covered source directory");
        std::fs::create_dir(root.path().join("target")).expect("coverage output directory");
        std::fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
        )
        .expect("manifest");
        std::fs::write(root.path().join("src/bin/fixture.rs"), "fn main() {}\n")
            .expect("covered source");
        let report = include_str!("../../project/test_data/source_coverage/llvm3-run.json")
            .replace("/workspace/", &format!("{}/", root.path().display()));
        let report_source = root.path().join("fixture-report.json");
        std::fs::write(&report_source, &report).expect("actual LLVM coverage fixture");
        let producer = root.path().join("coverage-producer.sh");
        std::fs::write(&producer,
            "#!/bin/sh\ncp \"$2\" \"$1\" || exit 31\ncp \"$1\" \"$1.pending\" || exit 32\nprintf started > \"$1.started\"\nwhile [ ! -f \"$1.release\" ]; do sleep 0.01; done\ncp \"$1.pending\" \"$1\" || exit 33\nrm \"$1.pending\" || exit 34\nexit 0\n")
            .expect("explicit controlled coverage producer");
        std::fs::set_permissions(&producer, std::fs::Permissions::from_mode(0o700))
            .expect("executable producer");
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.path()],
            cx,
        )
        .await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let (workspace, terminal_panel) = window_handle
            .update(cx, |container, window, cx| {
                let workspace = container.workspace().clone();
                let panel = workspace.update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    workspace.set_terminal_provider(TerminalProvider(panel.clone()));
                    panel
                });
                (workspace, panel)
            })
            .expect("production TerminalPanel provider");
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = project.read_with(cx, |project, cx| {
            let store = project.worktree_store();
            let id = store
                .read(cx)
                .worktrees()
                .next()
                .expect("local worktree")
                .read(cx)
                .id();
            (store, id)
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("trust owner")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let set_covered = |covered: u32| {
            let mut value: serde_json::Value =
                serde_json::from_str(&report).expect("actual LLVM report");
            *value
                .get_mut("data")
                .and_then(|data| data.get_mut(0))
                .and_then(|data| data.get_mut("files"))
                .and_then(|files| files.get_mut(0))
                .and_then(|file| file.get_mut("summary"))
                .and_then(|summary| summary.get_mut("lines"))
                .and_then(|lines| lines.get_mut("covered"))
                .expect("LLVM line summary") = covered.into();
            serde_json::to_string(&value).expect("controlled report serialization")
        };
        let old_report = set_covered(2);
        let new_report = set_covered(3);
        std::fs::write(&report_source, &old_report).expect("older report contents");
        let template = TaskTemplate {
            label: "overlapping coverage terminal producer".into(),
            command: producer.to_string_lossy().into_owned(),
            args: vec![
                "target/coverage.json".into(),
                report_source.to_string_lossy().into_owned(),
            ],
            cwd: Some(root.path().to_string_lossy().into_owned()),
            literal_arguments: true,
            save: SaveStrategy::None,
            allow_concurrent_runs: true,
            use_new_terminal: true,
            artifact_group: Some(TaskArtifactGroup {
                format: "llvm-cov-json-v1".into(),
                files: BTreeMap::from_iter([(
                    "report".into(),
                    TaskArtifact {
                        path: "target/coverage.json".into(),
                        kind: TaskArtifactKind::Data,
                        max_bytes: 32 * 1024 * 1024,
                    },
                )]),
            }),
            artifact_group_argument_binding: Some(TaskArtifactGroupArgumentBinding {
                role: "report".into(),
                argument_index: 0,
                related: Vec::new(),
            }),
            ..TaskTemplate::default()
        };
        let id_base = TaskSourceKind::UserInput.to_id_base();
        let mut completions = Vec::new();
        let mut outputs = Vec::new();
        let mut stable_id = None;
        for invocation in 0..2 {
            let resolved = template
                .resolve_task(&id_base, &TaskContext::default())
                .expect("same actual producing Task");
            let (sender, receiver) = oneshot::channel();
            window_handle
                .update(cx, |_, window, cx| {
                    workspace.update(cx, |workspace, cx| {
                        workspace.schedule_resolved_task_with_completion(
                            TaskSourceKind::UserInput,
                            resolved,
                            false,
                            move |result, _| {
                                sender.send(result).expect("overlap completion receiver")
                            },
                            window,
                            cx,
                        );
                    })
                })
                .expect("schedule overlapping actual Task");
            cx.run_until_parked();
            let (_, cached) = project.read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("history owner")
                    .read(cx)
                    .last_scheduled_task(None)
                    .expect("bound actual invocation")
            });
            assert_eq!(cached.original_task(), &template);
            assert_eq!(&cached.id, stable_id.get_or_insert(cached.id.clone()));
            let output = cached
                .resolved_artifact_group()
                .expect("bound group")
                .files
                .get("report")
                .expect("report role")
                .path
                .clone();
            let started = root.path().join(format!("{output}.started"));
            for _ in 0..100 {
                if started.exists() {
                    break;
                }
                cx.background_executor
                    .timer(Duration::from_millis(50))
                    .await;
            }
            assert!(
                started.exists(),
                "each actual child writes before its held completion"
            );
            assert_eq!(
                std::fs::read_to_string(root.path().join(&output)).expect("actual child report"),
                if invocation == 0 {
                    old_report.as_str()
                } else {
                    new_report.as_str()
                }
            );
            let running = terminal_panel.update(cx, |panel, cx| {
                panel
                    .terminals_for_task("overlapping coverage terminal producer", cx)
                    .iter()
                    .filter(|(_, _, view)| {
                        view.read(cx)
                            .terminal()
                            .read(cx)
                            .task()
                            .is_some_and(|task| task.status == terminal::TaskStatus::Running)
                    })
                    .count()
            });
            assert_eq!(
                running,
                invocation + 1,
                "Terminal execution is not globally serialized"
            );
            let loading = store.read_with(cx, |store, _| {
                store.selected_snapshot().expect("held capture").clone()
            });
            assert_eq!(loading.status, SourceCoverageStatus::Loading);
            completions.push(receiver);
            outputs.push(output);
            if invocation == 0 {
                std::fs::write(&report_source, &new_report).expect("newer actual producer input");
            }
        }
        let new_output = outputs.pop().expect("new output");
        let old_output = outputs.pop().expect("old output");
        assert_ne!(old_output, new_output);
        std::fs::write(
            root.path().join(format!("{new_output}.release")),
            b"release",
        )
        .expect("release newer actual child first");
        let newer = completions.pop().expect("newer actual completion");
        let completion = futures::future::select(
            Box::pin(newer),
            Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
        )
        .await;
        match completion {
            futures::future::Either::Left((result, _)) => {
                assert_eq!(result.expect("new result"), ScheduledTaskResult::Success)
            }
            futures::future::Either::Right(_) => panic!("newer actual child did not complete"),
        }
        cx.run_until_parked();
        let published = store.read_with(cx, |store, _| {
            store.selected_snapshot().expect("newest report").clone()
        });
        assert_eq!(published.status, SourceCoverageStatus::Current);
        assert_eq!(published.files.len(), 1);
        assert_eq!(
            published
                .files
                .first()
                .expect("source coverage")
                .covered_lines,
            3
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join(format!("{old_output}.pending")))
                .expect("older process holds bytes for its late output write"),
            old_report
        );
        std::fs::write(
            root.path().join(format!("{old_output}.release")),
            b"release",
        )
        .expect("release older actual child after newer publication");
        let older = completions.pop().expect("older actual completion");
        let completion = futures::future::select(
            Box::pin(older),
            Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
        )
        .await;
        match completion {
            futures::future::Either::Left((result, _)) => {
                assert_eq!(result.expect("old result"), ScheduledTaskResult::Success)
            }
            futures::future::Either::Right(_) => panic!("older actual child did not complete"),
        }
        cx.run_until_parked();
        assert!(
            !root.path().join(format!("{old_output}.pending")).exists(),
            "older success occurs only after its explicit late report write"
        );
        store.read_with(cx, |store, _| {
            assert_eq!(store.selected_snapshot(), Some(&published))
        });
        assert_eq!(
            std::fs::read_to_string(root.path().join(new_output))
                .expect("newer raw report retained"),
            new_report
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join(old_output))
                .expect("older raw report isolated"),
            old_report
        );
    }

    #[cfg(all(unix, feature = "rust-coverage"))]
    #[gpui::test]
    async fn coverage_capture_queued_source_change_prevents_child_and_recovers(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;

        use collections::{BTreeMap, HashSet};
        use project::{
            TaskSourceKind, source_coverage::SourceCoverageStatus, trusted_worktrees::PathTrust,
        };
        use task::{
            SaveStrategy, TaskArtifact, TaskArtifactGroup, TaskArtifactGroupArgumentBinding,
            TaskArtifactKind, TaskContext, TaskTemplate,
        };
        use workspace::tasks::ScheduledTaskResult;

        cx.executor().allow_parking();
        init_test(cx);
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            )
        });
        let root = tempfile::tempdir().expect("real coverage worktree");
        std::fs::create_dir_all(root.path().join("src/bin")).expect("covered source directory");
        std::fs::create_dir(root.path().join("target")).expect("coverage output directory");
        std::fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
        )
        .expect("manifest");
        std::fs::write(root.path().join("src/bin/fixture.rs"), "fn main() {}\n")
            .expect("covered source");
        let report = include_str!("../../project/test_data/source_coverage/llvm3-run.json")
            .replace("/workspace/", &format!("{}/", root.path().display()));
        let report_source = root.path().join("fixture-report.json");
        std::fs::write(&report_source, &report).expect("actual LLVM coverage fixture");
        let controls = tempfile::tempdir().expect("unindexed process controls");
        let producer = controls.path().join("coverage-producer.sh");
        std::fs::write(
            &producer,
            "#!/bin/sh\nprintf started > \"$3\"\ncp \"$2\" \"$1\" || exit 31\nexit 0\n",
        )
        .expect("explicit controlled coverage producer");
        std::fs::set_permissions(&producer, std::fs::Permissions::from_mode(0o700))
            .expect("executable producer");
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.path()],
            cx,
        )
        .await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let (workspace, terminal_panel) = window_handle
            .update(cx, |container, window, cx| {
                let workspace = container.workspace().clone();
                let panel = workspace.update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    workspace.set_terminal_provider(TerminalProvider(panel.clone()));
                    panel
                });
                (workspace, panel)
            })
            .expect("production TerminalPanel provider");
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = project.read_with(cx, |project, cx| {
            let store = project.worktree_store();
            let id = store
                .read(cx)
                .worktrees()
                .next()
                .expect("local worktree")
                .read(cx)
                .id();
            (store, id)
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("trust owner")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let buffer = project
            .update(cx, |project, cx| {
                project.open_buffer(
                    project
                        .find_project_path(root.path().join("src/bin/fixture.rs"), cx)
                        .expect("mapped source"),
                    cx,
                )
            })
            .await
            .expect("actual source buffer");
        let source_editor = window_handle
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    let editor = cx.new(|cx| {
                        editor::Editor::for_buffer(buffer, Some(project.clone()), window, cx)
                    });
                    workspace.add_item_to_active_pane(
                        Box::new(editor.clone()),
                        None,
                        true,
                        window,
                        cx,
                    );
                    editor
                })
            })
            .expect("mounted dirty-source Editor owner");
        let blocker_started = controls.path().join("blocker-started");
        let blocker_release = controls.path().join("blocker-release");
        let child_started = controls.path().join("coverage-started");
        let blocker = TaskTemplate {
            label: "queued coverage producer".into(),
            command: "/bin/sh".into(),
            args: vec![
                "-c".into(),
                "printf started > \"$1\"; while [ ! -f \"$2\" ]; do sleep 0.01; done".into(),
                "blocker".into(),
                blocker_started.to_string_lossy().into_owned(),
                blocker_release.to_string_lossy().into_owned(),
            ],
            cwd: Some(root.path().to_string_lossy().into_owned()),
            literal_arguments: true,
            save: SaveStrategy::None,
            allow_concurrent_runs: false,
            use_new_terminal: true,
            ..TaskTemplate::default()
        };
        let template = TaskTemplate {
            label: blocker.label.clone(),
            command: producer.to_string_lossy().into_owned(),
            args: vec![
                "target/coverage.json".into(),
                report_source.to_string_lossy().into_owned(),
                child_started.to_string_lossy().into_owned(),
            ],
            cwd: blocker.cwd.clone(),
            literal_arguments: true,
            save: SaveStrategy::None,
            allow_concurrent_runs: false,
            use_new_terminal: true,
            artifact_group: Some(TaskArtifactGroup {
                format: "llvm-cov-json-v1".into(),
                files: BTreeMap::from_iter([(
                    "report".into(),
                    TaskArtifact {
                        path: "target/coverage.json".into(),
                        kind: TaskArtifactKind::Data,
                        max_bytes: 32 * 1024 * 1024,
                    },
                )]),
            }),
            artifact_group_argument_binding: Some(TaskArtifactGroupArgumentBinding {
                role: "report".into(),
                argument_index: 0,
                related: Vec::new(),
            }),
            ..TaskTemplate::default()
        };
        let schedule = |template: &TaskTemplate, cx: &mut TestAppContext| {
            let resolved = template
                .resolve_task(
                    &TaskSourceKind::UserInput.to_id_base(),
                    &TaskContext::default(),
                )
                .expect("actual queued Task resolution");
            let (sender, receiver) = oneshot::channel();
            window_handle
                .update(cx, |_, window, cx| {
                    workspace.update(cx, |workspace, cx| {
                        workspace.schedule_resolved_task_with_completion(
                            TaskSourceKind::UserInput,
                            resolved,
                            false,
                            move |result, _| {
                                sender.send(result).expect("queued completion receiver")
                            },
                            window,
                            cx,
                        );
                    })
                })
                .expect("production queued scheduler");
            receiver
        };
        let blocker_completion = schedule(&blocker, cx);
        for _ in 0..100 {
            if blocker_started.exists() {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        assert!(
            blocker_started.exists(),
            "actual predecessor is held running"
        );
        let rejected_completion = schedule(&template, cx);
        cx.run_until_parked();
        let (_, rejected_task) = project.read_with(cx, |project, cx| {
            project
                .task_store()
                .read(cx)
                .task_inventory()
                .expect("history owner")
                .read(cx)
                .last_scheduled_task(None)
                .expect("queued bound invocation")
        });
        let rejected_output = rejected_task
            .resolved_artifact_group()
            .expect("bound group")
            .files
            .get("report")
            .expect("report role")
            .path
            .clone();
        let loading = store.read_with(cx, |store, _| {
            store.selected_snapshot().expect("queued capture").clone()
        });
        assert_eq!(loading.status, SourceCoverageStatus::Loading);
        assert!(
            !child_started.exists(),
            "capture child must wait for its predecessor"
        );
        assert!(!root.path().join(&rejected_output).exists());
        assert_eq!(
            terminal_panel.update(cx, |panel, cx| panel
                .terminals_for_task(&template.label, cx)
                .len()),
            1
        );
        source_editor.update(cx, |editor, cx| {
            editor.edit(
                [(
                    editor::MultiBufferOffset(0)..editor::MultiBufferOffset(0),
                    "// source changed while terminal is queued\n",
                )],
                cx,
            )
        });
        cx.run_until_parked();
        std::fs::write(&blocker_release, b"release").expect("release actual predecessor");
        let completion = futures::future::select(
            Box::pin(blocker_completion),
            Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
        )
        .await;
        match completion {
            futures::future::Either::Left((result, _)) => assert_eq!(
                result.expect("predecessor result"),
                ScheduledTaskResult::Success
            ),
            futures::future::Either::Right(_) => panic!("actual predecessor did not finish"),
        }
        let completion = futures::future::select(
            Box::pin(rejected_completion),
            Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
        )
        .await;
        match completion {
            futures::future::Either::Left((result, _)) => assert_ne!(
                result.expect("queued rejection"),
                ScheduledTaskResult::Success
            ),
            futures::future::Either::Right(_) => {
                panic!("queued capture was not rejected at eligibility")
            }
        }
        cx.run_until_parked();
        assert!(
            !child_started.exists(),
            "stale source is rejected before any coverage child executes"
        );
        assert!(
            !root.path().join(&rejected_output).exists(),
            "rejected producer never writes a report"
        );
        store.read_with(cx, |store, _| {
            let snapshot = store.selected_snapshot().expect("owned rejected state");
            assert_eq!(snapshot.generation, loading.generation);
            assert_eq!(snapshot.status, SourceCoverageStatus::Stale);
            assert!(snapshot.files.is_empty());
        });
        let timeout = Box::pin(cx.background_executor.timer(Duration::from_secs(10)));
        let recovery_timeout = match futures::future::select(
            Box::pin(async {
                let absolute_source = root.path().join("src/bin/fixture.rs");
                let source = project.read_with(cx, |project, cx| {
                    project
                        .find_project_path(&absolute_source, cx)
                        .expect("mapped recovery source")
                });
                let buffer = project
                    .update(cx, |project, cx| project.open_buffer(source.clone(), cx))
                    .await
                    .expect("actual dirty recovery Buffer");
                buffer.read_with(cx, |buffer, _| {
                    assert!(
                        buffer.is_dirty(),
                        "recovery must follow the actual unsaved mapped edit"
                    )
                });
                project
                    .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
                    .await
                    .expect("actual Project save before fresh recovery");
                buffer.read_with(cx, |buffer, _| {
                    assert!(
                        !buffer.is_dirty(),
                        "fresh admission requires the actually saved mapped Buffer"
                    )
                });
                let (filesystem, worktree) = project.read_with(cx, |project, cx| {
                    (
                        project.fs().clone(),
                        project
                            .worktrees(cx)
                            .find(|worktree| worktree.read(cx).id() == source.worktree_id)
                            .expect("actual recovery Worktree"),
                    )
                });
                let metadata = filesystem
                    .metadata(&absolute_source)
                    .await
                    .expect("saved source metadata")
                    .expect("saved source exists");
                let entry = worktree
                    .update(cx, |worktree, cx| {
                        worktree
                            .as_local()
                            .expect("local mapped-source owner")
                            .refresh_entry(source.path.clone(), None, cx)
                    })
                    .await
                    .expect("actual saved source refresh")
                    .expect("saved source indexed");
                assert_eq!(
                    (entry.inode, entry.mtime, entry.size),
                    (metadata.inode, Some(metadata.mtime), metadata.len),
                    "saved mapped source is current before fresh admission"
                );
                worktree.read_with(cx, |worktree, _| {
                    let indexed = worktree
                        .entry_for_path(&source.path)
                        .expect("current indexed recovery source");
                    assert_eq!(
                        (indexed.inode, indexed.mtime, indexed.size),
                        (metadata.inode, Some(metadata.mtime), metadata.len)
                    );
                });
            }),
            timeout,
        )
        .await
        {
            futures::future::Either::Left(((), timeout)) => timeout,
            futures::future::Either::Right(_) => {
                panic!("saved queued recovery source did not become ready")
            }
        };
        let recovery_completion = schedule(&template, cx);
        let completion =
            futures::future::select(Box::pin(recovery_completion), recovery_timeout).await;
        match completion {
            futures::future::Either::Left((result, _)) => assert_eq!(
                result.expect("fresh capture result"),
                ScheduledTaskResult::Success
            ),
            futures::future::Either::Right(_) => panic!("fresh capture did not complete"),
        }
        cx.run_until_parked();
        assert!(
            child_started.exists(),
            "fresh admission launches the actual producer"
        );
        let (_, recovered_task) = project.read_with(cx, |project, cx| {
            project
                .task_store()
                .read(cx)
                .task_inventory()
                .expect("history owner")
                .read(cx)
                .last_scheduled_task(None)
                .expect("fresh bound invocation")
        });
        assert_eq!(recovered_task.id, rejected_task.id);
        assert_eq!(recovered_task.original_task(), &template);
        let recovered_output = &recovered_task
            .resolved_artifact_group()
            .expect("fresh group")
            .files
            .get("report")
            .expect("fresh report")
            .path;
        assert_ne!(recovered_output, &rejected_output);
        assert_eq!(
            std::fs::read_to_string(root.path().join(recovered_output))
                .expect("actual fresh report"),
            report
        );
        store.read_with(cx, |store, _| {
            let snapshot = store.selected_snapshot().expect("fresh publication");
            assert!(snapshot.generation > loading.generation);
            assert_eq!(snapshot.status, SourceCoverageStatus::Current);
            assert_eq!(snapshot.files.len(), 1);
        });
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn profile_artifact_opens_after_real_terminal_success(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt as _;

        use collections::HashSet;
        use project::{TaskSourceKind, trusted_worktrees::PathTrust};
        use task::{SaveStrategy, TaskArtifact, TaskArtifactKind, TaskContext, TaskTemplate};
        use workspace::tasks::ScheduledTaskResult;

        cx.executor().allow_parking();
        init_test(cx);
        let root = tempfile::tempdir().expect("temporary profile worktree");
        std::fs::write(
            root.path().join("Cargo.toml"),
            "[package]\nname = \"profile-fixture\"\nversion = \"0.1.0\"\n",
        )
        .expect("write project manifest");
        let profiler = root.path().join("write-artifact.sh");
        std::fs::write(
            &profiler,
            "#!/bin/sh\nprintf produced > \"$1\"\nexit \"$2\"\n",
        )
        .expect("write explicit profiler command");
        std::fs::set_permissions(&profiler, std::fs::Permissions::from_mode(0o755))
            .expect("make profiler command executable");
        let project = Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [root.path()],
            cx,
        )
        .await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let (workspace, terminal_panel) = window_handle
            .update(cx, |multi_workspace, window, cx| {
                let workspace = multi_workspace.workspace().clone();
                let terminal_panel = workspace.update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    workspace.set_terminal_provider(TerminalProvider(panel.clone()));
                    panel
                });
                (workspace, terminal_panel)
            })
            .expect("initialize production terminal provider");
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = workspace.read_with(cx, |workspace, cx| {
            let worktree_store = workspace.project().read(cx).worktree_store();
            let worktree_id = worktree_store
                .read(cx)
                .worktrees()
                .next()
                .expect("profile worktree")
                .read(cx)
                .id();
            (worktree_store, worktree_id)
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("trust initialized")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });

        for (artifact_path, exit_code, expected_result) in [
            ("profile-success.txt", 0, ScheduledTaskResult::Success),
            ("profile-failure.txt", 7, ScheduledTaskResult::Failure),
        ] {
            let task = TaskTemplate {
                label: format!("explicit profile artifact {exit_code}"),
                command: profiler.to_string_lossy().into_owned(),
                args: vec![artifact_path.into(), exit_code.to_string()],
                cwd: Some(root.path().to_string_lossy().into_owned()),
                save: SaveStrategy::None,
                artifact: Some(TaskArtifact {
                    path: artifact_path.into(),
                    kind: TaskArtifactKind::File,
                    max_bytes: 1024,
                }),
                ..TaskTemplate::default()
            }
            .resolve_task("profile-real-terminal", &TaskContext::default())
            .expect("explicit profile Task");
            let (sender, receiver) = oneshot::channel();
            window_handle
                .update(cx, |_, window, cx| {
                    workspace.update(cx, |workspace, cx| {
                        workspace.schedule_resolved_task_with_completion(
                            TaskSourceKind::UserInput,
                            task,
                            false,
                            move |result, _| {
                                sender.send(result).expect("profile completion receiver")
                            },
                            window,
                            cx,
                        );
                    });
                })
                .expect("schedule profile Task through terminal panel");
            assert_eq!(
                receiver.await.expect("profile Task result"),
                expected_result
            );
            assert_eq!(
                std::fs::read_to_string(root.path().join(artifact_path))
                    .expect("Task-created artifact"),
                "produced"
            );
            cx.run_until_parked();
            let (opened, active_path) = workspace.read_with(cx, |workspace, cx| {
                (
                    workspace.active_pane().read(cx).items_len(),
                    workspace
                        .active_item(cx)
                        .and_then(|item| item.project_path(cx)),
                )
            });
            assert_eq!(opened, 1, "failed Task must not open its artifact");
            assert_eq!(
                active_path.as_ref().map(|path| path.path.as_unix_str()),
                Some("profile-success.txt"),
                "the Editor must retain only the successful Task artifact"
            );
        }

        let cancellable_profiler = root.path().join("wait-artifact.sh");
        std::fs::write(
            &cancellable_profiler,
            "#!/bin/sh\nprintf produced > \"$1\"\nprintf started > \"$2\"\nsleep 60\n",
        )
        .expect("write cancellable profiler command");
        std::fs::set_permissions(
            &cancellable_profiler,
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("make cancellable profiler command executable");
        let cancelled_artifact = "profile-cancelled.txt";
        let started = root.path().join("profile-cancel-started");
        let task = TaskTemplate {
            label: "explicit cancellable profile artifact".into(),
            command: cancellable_profiler.to_string_lossy().into_owned(),
            args: vec![
                cancelled_artifact.into(),
                started.to_string_lossy().into_owned(),
            ],
            cwd: Some(root.path().to_string_lossy().into_owned()),
            save: SaveStrategy::None,
            artifact: Some(TaskArtifact {
                path: cancelled_artifact.into(),
                kind: TaskArtifactKind::File,
                max_bytes: 1024,
            }),
            ..TaskTemplate::default()
        }
        .resolve_task("profile-real-terminal", &TaskContext::default())
        .expect("cancellable profile Task");
        let (sender, receiver) = oneshot::channel();
        window_handle
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.schedule_resolved_task_with_completion(
                        TaskSourceKind::UserInput,
                        task,
                        false,
                        move |result, _| {
                            sender
                                .send(result)
                                .expect("cancellable completion receiver")
                        },
                        window,
                        cx,
                    );
                });
            })
            .expect("schedule cancellable Task through terminal panel");
        for _ in 0..100 {
            if started.exists() {
                break;
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        assert!(started.exists(), "cancellable profile process must start");
        let terminal = terminal_panel.update(cx, |panel, cx| {
            panel
                .terminals_for_task("explicit cancellable profile artifact", cx)
                .into_iter()
                .next()
                .map(|(_, _, view)| view.read(cx).terminal().clone())
                .expect("running profile Task terminal")
        });
        terminal.update(cx, |terminal, _| terminal.kill_active_task());
        let completion = futures::future::select(
            Box::pin(receiver),
            Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
        )
        .await;
        let result = match completion {
            futures::future::Either::Left((result, _)) => {
                result.expect("cancelled Task completion result")
            }
            futures::future::Either::Right(_) => panic!("cancelled Task did not finish"),
        };
        assert_ne!(result, ScheduledTaskResult::Success);
        assert_eq!(
            std::fs::read_to_string(root.path().join(cancelled_artifact))
                .expect("cancelled Task-created artifact"),
            "produced"
        );
        cx.run_until_parked();
        let (opened, active_path) = workspace.read_with(cx, |workspace, cx| {
            (
                workspace.active_pane().read(cx).items_len(),
                workspace
                    .active_item(cx)
                    .and_then(|item| item.project_path(cx)),
            )
        });
        assert_eq!(opened, 1, "cancelled Task must not open its artifact");
        assert_eq!(
            active_path.as_ref().map(|path| path.path.as_unix_str()),
            Some("profile-success.txt"),
            "cancelled Task must retain the successful Editor"
        );
        assert!(
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len()) > 0
        );
        let editor = workspace.read_with(cx, |workspace, cx| {
            workspace.item_of_type::<editor::Editor>(cx)
        });
        assert!(
            editor.is_some(),
            "successful Task must open its artifact in the Editor"
        );
    }

    #[cfg(all(feature = "jvm-tools", any(unix, windows)))]
    #[gpui::test]
    async fn groovy_build_uses_existing_terminal_panel(cx: &mut TestAppContext) {
        groovy_build_uses_existing_terminal_panel_for(cx, false).await;
    }

    #[cfg(all(feature = "jvm-tools", any(unix, windows)))]
    #[gpui::test]
    async fn groovy_build_uses_existing_terminal_panel_gradle(cx: &mut TestAppContext) {
        groovy_build_uses_existing_terminal_panel_for(cx, true).await;
    }

    #[cfg(all(feature = "jvm-tools", any(unix, windows)))]
    async fn groovy_build_uses_existing_terminal_panel_for(
        cx: &mut TestAppContext,
        use_gradle: bool,
    ) {
        #[cfg(unix)]
        use std::os::unix::fs::PermissionsExt as _;

        use collections::HashSet;
        use project::{TaskSourceKind, trusted_worktrees::PathTrust};
        use serde_json::json;
        use task::{TaskContext, TaskTemplate, TaskVariables, VariableName};
        use workspace::tasks::ScheduledTaskResult;

        cx.executor().allow_parking();
        init_test(cx);
        let root = tempfile::tempdir().expect("temporary host worktree");
        let (manifest_name, label, argument) = if use_gradle {
            ("build.gradle.kts", "Build Gradle project", "build")
        } else {
            ("pom.xml", "Build Maven project", "package")
        };
        let manifest = root.path().join(manifest_name);
        #[cfg(unix)]
        let (wrapper_name, command) = if use_gradle {
            ("gradlew", "./gradlew")
        } else {
            ("mvnw", "./mvnw")
        };
        #[cfg(windows)]
        let (wrapper_name, command) = if use_gradle {
            ("gradlew.bat", ".\\gradlew.bat")
        } else {
            ("mvnw.cmd", ".\\mvnw.cmd")
        };
        #[cfg(unix)]
        let (success_script, failure_script) = (
            "#!/bin/sh\npwd > task-cwd.txt\nprintf '%s\\n' \"$*\" > task-args.txt\nexit 0\n",
            "#!/bin/sh\nexit 7\n",
        );
        #[cfg(windows)]
        let (success_script, failure_script) = (
            "@echo off\r\ncd > task-cwd.txt\r\necho %* > task-args.txt\r\nexit /b 0\r\n",
            "@echo off\r\nexit /b 7\r\n",
        );
        let wrapper = root.path().join(wrapper_name);
        std::fs::write(&manifest, "manifest").expect("write host manifest");
        std::fs::write(&wrapper, success_script).expect("write host wrapper");
        #[cfg(unix)]
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755))
            .expect("make host wrapper executable");

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            root.path(),
            json!({ (manifest_name): "manifest", (wrapper_name): success_script }),
        )
        .await;
        let project = Project::test(fs, [root.path()], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let (workspace, terminal_panel) = window_handle
            .update(cx, |multi_workspace, window, cx| {
                let workspace = multi_workspace.workspace().clone();
                let terminal_panel = workspace.update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    workspace.set_terminal_provider(TerminalProvider(panel.clone()));
                    panel
                });
                (workspace, terminal_panel)
            })
            .expect("initialize production terminal provider");
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = workspace.read_with(cx, |workspace, cx| {
            let worktree_store = workspace.project().read(cx).worktree_store();
            let worktree_id = worktree_store
                .read(cx)
                .worktrees()
                .next()
                .expect("host worktree")
                .read(cx)
                .id();
            (worktree_store, worktree_id)
        });
        let source = TaskSourceKind::groovy_jvm_build(worktree_id);
        let mut variables = TaskVariables::default();
        variables.insert(
            VariableName::WorktreeRoot,
            root.path().to_string_lossy().into_owned(),
        );
        let resolved = TaskTemplate {
            label: label.to_string(),
            command: command.to_string(),
            args: vec![argument.to_string()],
            cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
            ..TaskTemplate::default()
        }
        .resolve_task(
            &source.to_id_base(),
            &TaskContext {
                cwd: Some(root.path().to_path_buf()),
                task_variables: variables,
                ..TaskContext::default()
            },
        )
        .expect("detected JVM build Task");

        let (sender, receiver) = oneshot::channel();
        window_handle
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.schedule_resolved_task_with_completion(
                        source.clone(),
                        resolved.clone(),
                        false,
                        move |result, _| sender.send(result).expect("blocked Task receiver"),
                        window,
                        cx,
                    );
                });
            })
            .expect("dispatch blocked Task");
        assert_eq!(
            receiver.await.expect("blocked Task result"),
            ScheduledTaskResult::SpawnFailed
        );
        assert!(!root.path().join("task-cwd.txt").exists());

        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("trust initialized")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let (sender, receiver) = oneshot::channel();
        window_handle
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.schedule_resolved_task_with_completion(
                        source.clone(),
                        resolved.clone(),
                        false,
                        move |result, _| sender.send(result).expect("host Task receiver"),
                        window,
                        cx,
                    );
                });
            })
            .expect("dispatch host Task");
        assert_eq!(
            receiver.await.expect("host Task result"),
            ScheduledTaskResult::Success
        );
        assert_eq!(
            std::path::Path::new(
                std::fs::read_to_string(root.path().join("task-cwd.txt"))
                    .expect("host working directory")
                    .trim(),
            )
            .canonicalize()
            .expect("canonical Task working directory"),
            root.path().canonicalize().expect("canonical host worktree")
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join("task-args.txt"))
                .expect("host arguments")
                .trim(),
            argument
        );
        assert!(
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len()) > 0
        );

        std::fs::write(&wrapper, failure_script).expect("write failing wrapper");
        let (sender, receiver) = oneshot::channel();
        window_handle
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.schedule_resolved_task_with_completion(
                        source,
                        resolved,
                        false,
                        move |result, _| sender.send(result).expect("failing Task receiver"),
                        window,
                        cx,
                    );
                });
            })
            .expect("dispatch failing Task");
        assert_eq!(
            receiver.await.expect("failing Task result"),
            ScheduledTaskResult::Failure
        );
    }

    #[cfg(all(feature = "jvm-tools", unix))]
    const GROOVY_MAVEN_BUILD_FILES: &[&str] = &[
        "pom.xml",
        "mvnw",
        ".mvn/wrapper/maven-wrapper.properties",
        "library/pom.xml",
        "library/src/main/kotlin/evidence/Greeting.kt",
        "application/pom.xml",
        "application/src/main/java/evidence/Main.java",
        "application/src/test/java/evidence/GreetingTest.java",
    ];

    #[cfg(all(feature = "jvm-tools", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and a populated Maven wrapper and dependency cache"]
    async fn groovy_maven_fixture_builds_through_terminal_panel(cx: &mut TestAppContext) {
        groovy_fixture_builds_through_terminal_panel(
            cx,
            "groovy_jvm_maven",
            GROOVY_MAVEN_BUILD_FILES,
            "mvnw",
            "Build Maven project",
            "./mvnw",
            "package",
            "application/target/surefire-reports/TEST-evidence.GreetingTest.xml",
            false,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-tools", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and a cached Maven wrapper distribution"]
    async fn groovy_maven_missing_offline_cache_fails_through_terminal_panel(
        cx: &mut TestAppContext,
    ) {
        groovy_fixture_builds_through_terminal_panel(
            cx,
            "groovy_jvm_maven",
            GROOVY_MAVEN_BUILD_FILES,
            "mvnw",
            "Build Maven project",
            "./mvnw",
            "package",
            "application/target/surefire-reports/TEST-evidence.GreetingTest.xml",
            true,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-tools", unix))]
    #[gpui::test]
    #[ignore = "requires a JDK and a populated Gradle wrapper and dependency cache"]
    async fn groovy_gradle_fixture_builds_through_terminal_panel(cx: &mut TestAppContext) {
        groovy_fixture_builds_through_terminal_panel(
            cx,
            "groovy_jvm_gradle",
            &[
                "settings.gradle.kts",
                "build.gradle.kts",
                "gradlew",
                "gradle/wrapper/gradle-wrapper.jar",
                "gradle/wrapper/gradle-wrapper.properties",
                "library/build.gradle.kts",
                "library/src/main/kotlin/evidence/Greeting.kt",
                "application/build.gradle.kts",
                "application/src/main/java/evidence/Main.java",
                "application/src/test/java/evidence/GreetingTest.java",
            ],
            "gradlew",
            "Build Gradle project",
            "./gradlew",
            "build",
            "application/build/test-results/test/TEST-evidence.GreetingTest.xml",
            false,
        )
        .await;
    }

    #[cfg(all(feature = "jvm-tools", unix))]
    async fn groovy_fixture_builds_through_terminal_panel(
        cx: &mut TestAppContext,
        fixture_directory: &str,
        files: &[&str],
        wrapper_name: &str,
        label: &str,
        command: &str,
        argument: &str,
        report_path: &str,
        empty_maven_repository: bool,
    ) {
        use std::{os::unix::fs::PermissionsExt as _, path::Path};

        use collections::HashSet;
        use project::{TaskSourceKind, trusted_worktrees::PathTrust};
        use serde_json::json;
        use task::{TaskContext, TaskTemplate, TaskVariables, VariableName};
        use workspace::tasks::ScheduledTaskResult;

        cx.executor().allow_parking();
        init_test(cx);
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../project/test_data")
            .join(fixture_directory);
        let root = tempfile::tempdir().expect("temporary JVM worktree");
        for &relative in files {
            let destination = root.path().join(relative);
            std::fs::create_dir_all(destination.parent().expect("fixture parent"))
                .expect("create fixture parent");
            std::fs::copy(fixture.join(relative), destination).expect("copy fixture file");
        }
        std::fs::set_permissions(
            root.path().join(wrapper_name),
            std::fs::Permissions::from_mode(0o755),
        )
        .expect("make project wrapper executable");
        if empty_maven_repository {
            let repository = root.path().join("empty-maven-repository");
            std::fs::create_dir(&repository).expect("create empty Maven repository");
            std::fs::write(
                root.path().join(".mvn/maven.config"),
                format!("-o\n-Dmaven.repo.local={}\n", repository.display()),
            )
            .expect("require Maven offline from an empty repository");
        }

        let fs = FakeFs::new(cx.executor());
        let snapshot = if wrapper_name == "mvnw" {
            json!({ "pom.xml": "<project/>", "mvnw": "#!/bin/sh\n" })
        } else {
            json!({ "build.gradle.kts": "plugins {}", "gradlew": "#!/bin/sh\n" })
        };
        fs.insert_tree(root.path(), snapshot).await;
        let project = Project::test(fs, [root.path()], cx).await;
        let window_handle =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let (workspace, terminal_panel) = window_handle
            .update(cx, |multi_workspace, window, cx| {
                let workspace = multi_workspace.workspace().clone();
                let terminal_panel = workspace.update(cx, |workspace, cx| {
                    let panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
                    workspace.add_panel(panel.clone(), window, cx);
                    workspace.set_terminal_provider(TerminalProvider(panel.clone()));
                    panel
                });
                (workspace, terminal_panel)
            })
            .expect("initialize production terminal provider");
        cx.update(|cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = workspace.read_with(cx, |workspace, cx| {
            let worktree_store = workspace.project().read(cx).worktree_store();
            let worktree_id = worktree_store
                .read(cx)
                .worktrees()
                .next()
                .expect("JVM worktree")
                .read(cx)
                .id();
            (worktree_store, worktree_id)
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("trust initialized")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });

        let source = TaskSourceKind::groovy_jvm_build(worktree_id);
        let mut variables = TaskVariables::default();
        variables.insert(
            VariableName::WorktreeRoot,
            root.path().to_string_lossy().into_owned(),
        );
        let resolved = TaskTemplate {
            label: label.to_string(),
            command: command.to_string(),
            args: vec![argument.to_string()],
            cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
            ..TaskTemplate::default()
        }
        .resolve_task(
            &source.to_id_base(),
            &TaskContext {
                cwd: Some(root.path().to_path_buf()),
                task_variables: variables,
                ..TaskContext::default()
            },
        )
        .expect("resolve generated JVM build");
        let (sender, receiver) = oneshot::channel();
        window_handle
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.schedule_resolved_task_with_completion(
                        source,
                        resolved,
                        false,
                        move |result, _| sender.send(result).expect("JVM Task receiver"),
                        window,
                        cx,
                    );
                });
            })
            .expect("dispatch JVM build");
        let result = receiver.await.expect("JVM build result");
        let output = terminal_panel
            .read_with(cx, |panel, cx| {
                panel
                    .active_pane
                    .read(cx)
                    .items()
                    .find_map(|item| item.act_as::<TerminalView>(cx))
                    .map(|view| view.read(cx).terminal().read(cx).get_content())
            })
            .expect("JVM Task terminal output");
        if empty_maven_repository {
            assert_eq!(result, ScheduledTaskResult::Failure);
            assert!(!root.path().join(report_path).exists());
            assert!(
                output.contains("offline"),
                "missing offline error: {output}"
            );
            assert!(output.contains("plugin"), "missing plugin error: {output}");
        } else {
            assert_eq!(result, ScheduledTaskResult::Success, "{output}");
            assert!(
                output.contains("BUILD SUCCESS"),
                "missing build result: {output}"
            );
            let report = std::fs::read_to_string(root.path().join(report_path))
                .expect("fresh JVM JUnit report");
            assert!(report.contains("classname="));
            assert!(report.contains("messageIsAvailable"));
        }
    }

    #[gpui::test]
    async fn test_terminal_panel_starts_open_follows_setting(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        let (window_handle, terminal_panel) = init_workspace_with_panel(cx).await;

        window_handle
            .update(cx, |_, window, cx| {
                terminal_panel.update(cx, |terminal_panel, cx| {
                    assert!(
                        !terminal_panel.starts_open(window, cx),
                        "terminal panel should not start open by default"
                    );
                });
            })
            .expect("Failed to read terminal panel starts_open default");

        cx.update_global(|store: &mut SettingsStore, cx| {
            store.update_user_settings(cx, |settings| {
                settings.terminal.get_or_insert_default().starts_open = Some(true);
            });
        });

        window_handle
            .update(cx, |_, window, cx| {
                terminal_panel.update(cx, |terminal_panel, cx| {
                    assert!(
                        terminal_panel.starts_open(window, cx),
                        "terminal panel should start open when configured"
                    );
                });
            })
            .expect("Failed to read configured terminal panel starts_open");
    }

    #[gpui::test]
    async fn test_new_terminal_opens_in_panel_by_default(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        let (window_handle, terminal_panel) = init_workspace_with_panel(cx).await;

        let panel_items_before =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());
        let center_items_before = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::new_terminal(
                        workspace,
                        &workspace::NewTerminal::default(),
                        window,
                        cx,
                    );
                })
            })
            .expect("Failed to dispatch new_terminal");

        cx.run_until_parked();

        let panel_items_after =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());
        let center_items_after = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");

        assert_eq!(
            panel_items_after,
            panel_items_before + 1,
            "Terminal should be added to the panel when no center terminal is focused"
        );
        assert_eq!(
            center_items_after, center_items_before,
            "Center pane should not gain a new terminal"
        );
    }

    #[gpui::test]
    async fn test_new_terminal_opens_in_center_when_center_terminal_focused(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        init_test(cx);

        let (window_handle, terminal_panel) = init_workspace_with_panel(cx).await;

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::add_center_terminal(workspace, window, cx, |project, cx| {
                        project.create_terminal_shell(None, cx)
                    })
                })
            })
            .expect("Failed to update workspace")
            .await
            .expect("Failed to create center terminal");
        cx.run_until_parked();

        let center_items_before = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");
        assert_eq!(center_items_before, 1, "Center pane should have 1 terminal");

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let active_item = workspace
                        .active_pane()
                        .read(cx)
                        .active_item()
                        .expect("Center pane should have an active item");
                    let terminal_view = active_item
                        .downcast::<TerminalView>()
                        .expect("Active center item should be a TerminalView");
                    window.focus(&terminal_view.focus_handle(cx), cx);
                })
            })
            .expect("Failed to focus terminal view");
        cx.run_until_parked();

        let panel_items_before =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::new_terminal(
                        workspace,
                        &workspace::NewTerminal::default(),
                        window,
                        cx,
                    );
                })
            })
            .expect("Failed to dispatch new_terminal");
        cx.run_until_parked();

        let center_items_after = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");
        let panel_items_after =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());

        assert_eq!(
            center_items_after,
            center_items_before + 1,
            "New terminal should be added to the center pane"
        );
        assert_eq!(
            panel_items_after, panel_items_before,
            "Terminal panel should not gain a new terminal"
        );
    }

    #[gpui::test]
    async fn test_new_terminal_opens_in_panel_when_panel_focused(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);

        let (window_handle, terminal_panel) = init_workspace_with_panel(cx).await;

        window_handle
            .update(cx, |_, window, cx| {
                terminal_panel.update(cx, |panel, cx| {
                    panel.add_terminal_shell(None, RevealStrategy::Always, window, cx)
                })
            })
            .expect("Failed to update workspace")
            .await
            .expect("Failed to create panel terminal");
        cx.run_until_parked();

        window_handle
            .update(cx, |_, window, cx| {
                window.focus(&terminal_panel.read(cx).focus_handle(cx), cx);
            })
            .expect("Failed to focus terminal panel");
        cx.run_until_parked();

        let panel_items_before =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());

        let center_items_before = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::new_terminal(
                        workspace,
                        &workspace::NewTerminal::default(),
                        window,
                        cx,
                    );
                })
            })
            .expect("Failed to dispatch new_terminal");
        cx.run_until_parked();

        let panel_items_after =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());
        let center_items_after = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");

        assert_eq!(
            panel_items_after,
            panel_items_before + 1,
            "New terminal should be added to the panel when panel is focused"
        );
        assert_eq!(
            center_items_after, center_items_before,
            "Center pane should not gain a new terminal"
        );
    }

    #[gpui::test]
    async fn test_new_local_terminal_opens_in_center_when_center_terminal_focused(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        init_test(cx);

        let (window_handle, terminal_panel) = init_workspace_with_panel(cx).await;

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::add_center_terminal(workspace, window, cx, |project, cx| {
                        project.create_terminal_shell(None, cx)
                    })
                })
            })
            .expect("Failed to update workspace")
            .await
            .expect("Failed to create center terminal");
        cx.run_until_parked();

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let active_item = workspace
                        .active_pane()
                        .read(cx)
                        .active_item()
                        .expect("Center pane should have an active item");
                    let terminal_view = active_item
                        .downcast::<TerminalView>()
                        .expect("Active center item should be a TerminalView");
                    window.focus(&terminal_view.focus_handle(cx), cx);
                })
            })
            .expect("Failed to focus terminal view");
        cx.run_until_parked();

        let center_items_before = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");
        let panel_items_before =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::new_terminal(
                        workspace,
                        &workspace::NewTerminal { local: true },
                        window,
                        cx,
                    );
                })
            })
            .expect("Failed to dispatch new_terminal with local=true");
        cx.run_until_parked();

        let center_items_after = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");
        let panel_items_after =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());

        assert_eq!(
            center_items_after,
            center_items_before + 1,
            "New local terminal should be added to the center pane"
        );
        assert_eq!(
            panel_items_after, panel_items_before,
            "Terminal panel should not gain a new terminal"
        );
    }

    #[gpui::test]
    async fn test_new_terminal_opens_in_panel_when_panel_focused_and_center_has_terminal(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        init_test(cx);

        let (window_handle, terminal_panel) = init_workspace_with_panel(cx).await;

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::add_center_terminal(workspace, window, cx, |project, cx| {
                        project.create_terminal_shell(None, cx)
                    })
                })
            })
            .expect("Failed to update workspace")
            .await
            .expect("Failed to create center terminal");
        cx.run_until_parked();

        window_handle
            .update(cx, |_, window, cx| {
                terminal_panel.update(cx, |panel, cx| {
                    panel.add_terminal_shell(None, RevealStrategy::Always, window, cx)
                })
            })
            .expect("Failed to update workspace")
            .await
            .expect("Failed to create panel terminal");
        cx.run_until_parked();

        window_handle
            .update(cx, |_, window, cx| {
                window.focus(&terminal_panel.read(cx).focus_handle(cx), cx);
            })
            .expect("Failed to focus terminal panel");
        cx.run_until_parked();

        let panel_items_before =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());
        let center_items_before = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");

        window_handle
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    TerminalPanel::new_terminal(
                        workspace,
                        &workspace::NewTerminal::default(),
                        window,
                        cx,
                    );
                })
            })
            .expect("Failed to dispatch new_terminal");
        cx.run_until_parked();

        let panel_items_after =
            terminal_panel.read_with(cx, |panel, cx| panel.active_pane.read(cx).items_len());
        let center_items_after = window_handle
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .items_len()
            })
            .expect("Failed to read center pane items");

        assert_eq!(
            panel_items_after,
            panel_items_before + 1,
            "New terminal should go to panel when panel is focused, even if center has a terminal"
        );
        assert_eq!(
            center_items_after, center_items_before,
            "Center pane should not gain a new terminal when panel is focused"
        );
    }

    fn set_max_tabs(cx: &mut TestAppContext, value: Option<usize>) {
        cx.update_global(|store: &mut SettingsStore, cx| {
            store.update_user_settings(cx, |settings| {
                settings.workspace.max_tabs = value.map(|v| NonZero::new(v).unwrap())
            });
        });
    }

    pub fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::init(cx);
        });
    }
}
