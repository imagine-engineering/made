use agent_ui::{
    AgentPanel, ThreadId,
    thread_metadata_store::{
        TaskOrganization, TaskProjectAssociation, ThreadMetadata, ThreadMetadataStore,
    },
};
use editor::Editor;
use gpui::{
    App, Context, DismissEvent, Div, Entity, EventEmitter, FocusHandle, Focusable, Render,
    SharedString, Stateful, WeakEntity,
};
use ui::{
    ContextMenu, ContextMenuEntry, Modal, ModalFooter, ModalHeader, Section, prelude::*,
    right_click_menu,
};
use workspace::{
    ModalView, MultiWorkspace, ProjectGroupKey, Workspace,
    collaborative_navigation::CollaborativeNavigationTarget,
};

use crate::{Sidebar, collaborative_navigation::navigation_action};

fn task_menu_entry(label: impl Into<SharedString>, icon: IconName) -> ContextMenuEntry {
    ContextMenuEntry::new(label)
        .icon(icon)
        .icon_size(IconSize::Small)
}

pub(crate) fn task_row_menu(
    row: Stateful<Div>,
    thread_id: ThreadId,
    multi_workspace: WeakEntity<MultiWorkspace>,
    cx: &App,
) -> AnyElement {
    let deleting = ThreadMetadataStore::global(cx)
        .read(cx)
        .task_deletion_pending(thread_id);
    let handle = ui::RightClickMenuHandle::<ContextMenu>::default();
    let keyboard_handle = handle.clone();
    let overflow_handle = handle.clone();
    let row = row
        .key_context("TaskRow")
        .when(deleting, |row| {
            row.child(
                Label::new("Deleting…")
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            )
        })
        .debug_selector(move || format!("COLLABORATIVE-TASK-ROW-{}", thread_id.to_key_string()))
        .child(
            navigation_action(
                SharedString::from(format!("task-menu-{thread_id:?}")),
                IconName::Ellipsis,
                "Task actions",
            )
            .disabled(deleting)
            .on_click(move |_, window, cx| {
                cx.stop_propagation();
                overflow_handle.show_at(window.mouse_position(), window, cx);
            }),
        )
        .on_action(move |_: &agent_ui::OpenTaskContextMenu, window, cx| {
            cx.stop_propagation();
            keyboard_handle.show_at(window.mouse_position(), window, cx);
        });
    right_click_menu(SharedString::from(format!("task-context-{thread_id:?}")))
        .with_handle(handle)
        .attach(gpui::Anchor::BottomRight)
        .anchor(gpui::Anchor::TopLeft)
        .trigger(move |_, _, _| row)
        .menu(move |window, cx| build_task_menu(thread_id, multi_workspace.clone(), window, cx))
        .into_any_element()
}

type ResolvedTask = (
    ThreadMetadata,
    Entity<Sidebar>,
    Entity<Workspace>,
    Option<Entity<Workspace>>,
);

fn resolve_action(
    thread_id: ThreadId,
    multi_workspace: &WeakEntity<MultiWorkspace>,
    cx: &mut App,
) -> Option<ResolvedTask> {
    let resolved = resolve(thread_id, multi_workspace, cx)?;
    if ThreadMetadataStore::global(cx)
        .read(cx)
        .task_deletion_pending(thread_id)
    {
        resolved.2.update(cx, |workspace, cx| {
            workspace.show_error("Task deletion is already in progress", cx);
        });
        return None;
    }
    Some(resolved)
}

fn resolve(
    thread_id: ThreadId,
    multi_workspace: &WeakEntity<MultiWorkspace>,
    cx: &App,
) -> Option<ResolvedTask> {
    let metadata = ThreadMetadataStore::try_global(cx)?
        .read(cx)
        .entry(thread_id)?
        .clone();
    let multi_workspace = multi_workspace.upgrade()?;
    let multi_workspace = multi_workspace.read(cx);
    let sidebar = multi_workspace
        .sidebar()?
        .to_any()
        .downcast::<Sidebar>()
        .ok()?;
    let active_workspace = multi_workspace.workspace().clone();
    let key = ProjectGroupKey::from_worktree_paths(
        &metadata.worktree_paths,
        metadata.remote_connection.clone(),
    );
    let thread_workspace = multi_workspace
        .workspaces()
        .find(|workspace| {
            workspace
                .read(cx)
                .panel::<AgentPanel>(cx)
                .is_some_and(|panel| {
                    panel
                        .read(cx)
                        .conversation_view_for_id(&thread_id, cx)
                        .is_some()
                })
        })
        .cloned()
        .or_else(|| multi_workspace.last_active_workspace_for_group(&key, cx));
    Some((metadata, sidebar, active_workspace, thread_workspace))
}

pub(crate) fn build_task_menu(
    thread_id: ThreadId,
    multi_workspace: WeakEntity<MultiWorkspace>,
    window: &mut Window,
    cx: &mut App,
) -> Entity<ContextMenu> {
    ContextMenu::build(window, cx, move |menu, _, cx| {
        let Some((metadata, sidebar, workspace, thread_workspace)) =
            resolve(thread_id, &multi_workspace, cx)
        else {
            return menu.label("This task is no longer available");
        };
        if ThreadMetadataStore::global(cx)
            .read(cx)
            .task_deletion_pending(thread_id)
        {
            return menu.label("Deleting task…");
        }
        let live = thread_workspace
            .as_ref()
            .and_then(|workspace| workspace.read(cx).panel::<AgentPanel>(cx))
            .and_then(|panel| {
                panel
                    .read(cx)
                    .conversation_view_for_id(&thread_id, cx)
                    .cloned()
            })
            .and_then(|view| view.read(cx).root_thread_view());
        let target = CollaborativeNavigationTarget::thread(thread_id.to_key_string());
        let pinned = workspace
            .read(cx)
            .collaborative_navigation()
            .is_pinned(&target);
        let mut menu = menu.item(
            task_menu_entry(
                if pinned { "Unpin" } else { "Pin" },
                if pinned {
                    IconName::Unpin
                } else {
                    IconName::Pin
                },
            )
            .handler({
                let multi_workspace = multi_workspace.clone();
                move |window, cx| {
                    if let Some((_, _, workspace, _)) =
                        resolve_action(thread_id, &multi_workspace, cx)
                    {
                        workspace.update(cx, |workspace, cx| {
                            if let Err(error) =
                                workspace.toggle_collaborative_pin(target.clone(), window, cx)
                            {
                                workspace.show_error(error.to_string(), cx);
                            }
                        });
                    }
                }
            }),
        );
        let native = metadata.agent_id.as_ref() == agent::ZED_AGENT_ID.as_ref();
        menu = menu.item(
            task_menu_entry("Share", IconName::Share)
                .disabled(!native)
                .handler({
                    let multi_workspace = multi_workspace.clone();
                    move |window, cx| {
                        copy_transcript(
                            thread_id,
                            multi_workspace.clone(),
                            TranscriptExport::Share,
                            window,
                            cx,
                        )
                    }
                }),
        );
        menu = menu.item(task_menu_entry("Rename", IconName::Pencil).handler({
            let multi_workspace = multi_workspace.clone();
            move |window, cx| {
                if let Some((metadata, sidebar, workspace, _)) =
                    resolve_action(thread_id, &multi_workspace, cx)
                {
                    workspace.update(cx, |workspace, cx| {
                        workspace.toggle_modal(window, cx, |window, cx| {
                            TaskRenameModal::new(metadata, sidebar.downgrade(), window, cx)
                        });
                    });
                }
            }
        }));
        let regenerating = sidebar.read(cx).regenerating_titles.contains(&thread_id)
            || live
                .as_ref()
                .and_then(|view| view.read(cx).as_native_thread(cx))
                .is_some_and(|thread| thread.read(cx).is_generating_title());
        menu = menu.item(
            ContextMenuEntry::new(if regenerating {
                "Regenerating title…"
            } else {
                "Regenerate title"
            })
            .icon(IconName::RotateCw)
            .icon_size(IconSize::Small)
            .disabled(!native || metadata.session_id.is_none() || regenerating)
            .handler({
                let multi_workspace = multi_workspace.clone();
                move |_, cx| {
                    if let Some((metadata, sidebar, _, thread_workspace)) =
                        resolve_action(thread_id, &multi_workspace, cx)
                        && let Some(session_id) = metadata.session_id.as_ref()
                    {
                        sidebar.update(cx, |sidebar, cx| {
                            sidebar.regenerate_thread_title(
                                session_id,
                                thread_id,
                                metadata.folder_paths().clone(),
                                thread_workspace,
                                cx,
                            )
                        });
                    }
                }
            }),
        );
        let organization = ThreadMetadataStore::global(cx)
            .read(cx)
            .task_organization(thread_id);
        let pending = ThreadMetadataStore::global(cx)
            .read(cx)
            .task_organization_pending(thread_id);
        menu = menu.item(
            ContextMenuEntry::new(if pending {
                "Updating task…"
            } else if organization.unread {
                "Mark as read"
            } else {
                "Mark as unread"
            })
            .icon(if organization.unread {
                IconName::Check
            } else {
                IconName::Circle
            })
            .icon_size(IconSize::Small)
            .disabled(pending)
            .handler({
                let multi_workspace = multi_workspace.clone();
                move |_, cx| {
                    if let Some((_, _, workspace, _)) =
                        resolve_action(thread_id, &multi_workspace, cx)
                    {
                        update_organization(
                            thread_id,
                            workspace,
                            |state| state.unread = !state.unread,
                            cx,
                        );
                    }
                }
            }),
        );
        menu = menu.item(task_menu_entry("View details", IconName::Info).handler({
            let multi_workspace = multi_workspace.clone();
            move |window, cx| {
                if let Some((metadata, _, workspace, _)) =
                    resolve_action(thread_id, &multi_workspace, cx)
                {
                    workspace.update(cx, |workspace, cx| {
                        workspace.toggle_modal(window, cx, |_, cx| TaskDetailsModal {
                            metadata,
                            focus: cx.focus_handle(),
                        });
                    });
                }
            }
        }));
        let exporting = sidebar.read(cx).exporting_transcripts.contains(&thread_id);
        for structured in [false, true] {
            menu = menu.item(
                ContextMenuEntry::new(if exporting {
                    "Loading transcript…"
                } else if structured {
                    "Copy JSON"
                } else {
                    "Copy content"
                })
                .icon(if structured {
                    IconName::Code
                } else {
                    IconName::Copy
                })
                .icon_size(IconSize::Small)
                .disabled(exporting || (live.is_none() && metadata.session_id.is_none()))
                .handler({
                    let multi_workspace = multi_workspace.clone();
                    move |window, cx| {
                        copy_transcript(
                            thread_id,
                            multi_workspace.clone(),
                            if structured {
                                TranscriptExport::Json
                            } else {
                                TranscriptExport::Content
                            },
                            window,
                            cx,
                        )
                    }
                }),
            );
        }
        let projects = multi_workspace
            .upgrade()
            .map(|multi_workspace| multi_workspace.read(cx).project_groups(cx))
            .unwrap_or_default();
        menu = menu
            .separator()
            .submenu_with_icon("Projects", IconName::Folder, {
                let multi_workspace = multi_workspace.clone();
                move |mut menu, _, cx| {
                    let store = ThreadMetadataStore::global(cx);
                    let associations = store.read(cx).task_projects(thread_id);
                    let pending = store.read(cx).task_organization_pending(thread_id);
                    let mut keys: Vec<_> =
                        projects.iter().map(|project| project.key.clone()).collect();
                    for association in &associations {
                        if !keys.iter().any(|key| association.matches(key)) {
                            keys.push(ProjectGroupKey::new(
                                association.remote_connection.clone(),
                                workspace::PathList::new(&association.paths),
                            ));
                        }
                    }
                    for key in keys {
                        let associated = associations
                            .iter()
                            .any(|association| association.matches(&key));
                        let label = format!(
                            "{} {}",
                            if associated { "Remove from" } else { "Add to" },
                            key.display_name(&Default::default())
                        );
                        let multi_workspace = multi_workspace.clone();
                        menu = menu.item(
                            task_menu_entry(
                                label,
                                if associated {
                                    IconName::Dash
                                } else {
                                    IconName::Plus
                                },
                            )
                            .disabled(pending)
                            .handler(move |_, cx| {
                                if let Some((_, _, workspace, _)) =
                                    resolve_action(thread_id, &multi_workspace, cx)
                                {
                                    let mut associations = ThreadMetadataStore::global(cx)
                                        .read(cx)
                                        .task_projects(thread_id);
                                    if let Some(index) = associations
                                        .iter()
                                        .position(|association| association.matches(&key))
                                    {
                                        associations.remove(index);
                                    } else {
                                        associations
                                            .push(TaskProjectAssociation::from_project_group(&key));
                                    }
                                    update_organization(
                                        thread_id,
                                        workspace,
                                        move |state| state.projects = Some(associations),
                                        cx,
                                    );
                                }
                            }),
                        );
                    }
                    if projects.is_empty() && associations.is_empty() {
                        menu = menu.label("Open a project to associate this task");
                    }
                    menu
                }
            });
        let connection_store = thread_workspace
            .as_ref()
            .unwrap_or(&workspace)
            .read(cx)
            .panel::<AgentPanel>(cx)
            .map(|panel| panel.read(cx).connection_store().clone());
        let running = live.as_ref().is_some_and(|view| {
            view.read(cx).thread.read(cx).status() == acp_thread::ThreadStatus::Generating
        });
        let supports_delete = connection_store.as_ref().and_then(|store| {
            let entry = store
                .read(cx)
                .entry(&agent_ui::Agent::from(metadata.agent_id.clone()))?;
            let connection = match entry.read(cx) {
                agent_ui::agent_connection_store::AgentConnectionEntry::Connected(state) => {
                    state.connection.clone()
                }
                _ => return None,
            };
            Some(
                connection
                    .session_list(cx)
                    .is_some_and(|list| list.supports_delete()),
            )
        });
        let delete_available = connection_store.is_some()
            && !running
            && (metadata.session_id.is_none() || supports_delete != Some(false));
        menu.separator().item(task_menu_entry(if metadata.archived { "Restore task" } else { "Archive task" }, if metadata.archived { IconName::Undo } else { IconName::Archive })
            .disabled(metadata.session_id.is_none())
            .handler({
                let multi_workspace = multi_workspace.clone();
                move |window, cx| {
                    if let Some((metadata, sidebar, _, _)) = resolve_action(thread_id, &multi_workspace, cx) {
                        if metadata.archived {
                            sidebar.update(cx, |sidebar, cx| sidebar.open_thread_from_archive(metadata, window, cx));
                        } else if let Some(session_id) = metadata.session_id.as_ref() {
                            sidebar.update(cx, |sidebar, cx| sidebar.archive_thread(session_id, window, cx));
                        }
                    }
                }
            }))
            .item(task_menu_entry(if running { "Stop task before deleting" } else if metadata.session_id.is_some() && supports_delete == Some(false) { "Provider does not support deletion" } else { "Delete" }, IconName::Trash).disabled(!delete_available)
                .handler(move |window, cx| {
                    let Some(connection_store) = connection_store.clone() else { return; };
                    let expected_session = metadata.session_id.clone();
                    let expected_agent = metadata.agent_id.clone();
                    let confirmation = window.prompt(gpui::PromptLevel::Warning, "Delete this task permanently?", Some("The provider transcript will be deleted. Archiving keeps the task available for restoration."), &["Delete", "Cancel"], cx);
                    let multi_workspace = multi_workspace.clone();
                    window.spawn(cx, async move |cx| {
                        if confirmation.await? != 0 { return anyhow::Ok(()); }
                        let operation = cx.update(|_, cx| {
                            let Some((metadata, _, workspace, thread_workspace)) = resolve_action(thread_id, &multi_workspace, cx) else { return None; };
                            let running = thread_workspace.and_then(|workspace| workspace.read(cx).panel::<AgentPanel>(cx))
                                .and_then(|panel| panel.read(cx).conversation_view_for_id(&thread_id, cx).cloned())
                                .and_then(|view| view.read(cx).root_thread_view())
                                .is_some_and(|view| view.read(cx).thread.read(cx).status() == acp_thread::ThreadStatus::Generating);
                            if running || metadata.session_id != expected_session || metadata.agent_id != expected_agent {
                                workspace.update(cx, |workspace, cx| workspace.show_error("The task changed while confirming deletion. Stop it if running, then reopen its menu and retry.", cx));
                                return None;
                            }
                            let task = agent_ui::threads_archive_view::delete_task_transcript(thread_id, metadata.session_id, metadata.agent_id, connection_store, cx);
                            Some((task, workspace))
                        })?;
                        if let Some((task, workspace)) = operation {
                            match task.await {
                                Err(error) => workspace.update(cx, |workspace, cx| workspace.show_error(error.to_string(), cx)),
                                Ok(()) => cx.update(|window, cx| {
                                    if let Some(multi_workspace) = multi_workspace.upgrade() {
                                        let panels: Vec<_> = multi_workspace.read(cx).workspaces()
                                            .filter_map(|workspace| workspace.read(cx).panel::<AgentPanel>(cx)).collect();
                                        for panel in panels {
                                            panel.update(cx, |panel, cx| panel.remove_thread_without_activating_draft(thread_id, window, cx));
                                        }
                                    }
                                })?,
                            }
                        }
                        Ok(())
                    }).detach_and_log_err(cx);
                }))
    })
}

fn update_organization(
    thread_id: ThreadId,
    workspace: Entity<Workspace>,
    change: impl FnOnce(&mut TaskOrganization),
    cx: &mut App,
) {
    let update = ThreadMetadataStore::global(cx).update(cx, |store, cx| {
        store.update_task_organization(thread_id, change, cx)
    });
    cx.spawn(async move |cx| {
        if let Err(error) = update.await {
            workspace.update(cx, |workspace, cx| {
                workspace.show_error(error.to_string(), cx)
            });
        }
    })
    .detach();
}

pub(crate) fn mark_task_read(thread_id: ThreadId, workspace: Entity<Workspace>, cx: &mut App) {
    if ThreadMetadataStore::global(cx)
        .read(cx)
        .task_organization(thread_id)
        .unread
    {
        update_organization(thread_id, workspace, |state| state.unread = false, cx);
    }
}

#[derive(Clone, Copy)]
enum TranscriptExport {
    Content,
    Json,
    Share,
}

fn copy_transcript(
    thread_id: ThreadId,
    multi_workspace: WeakEntity<MultiWorkspace>,
    export: TranscriptExport,
    window: &mut Window,
    cx: &mut App,
) {
    let Some((metadata, sidebar, workspace, thread_workspace)) =
        resolve_action(thread_id, &multi_workspace, cx)
    else {
        return;
    };
    if sidebar.read(cx).exporting_transcripts.contains(&thread_id) {
        workspace.update(cx, |workspace, cx| {
            workspace.show_error("Transcript export is already in progress", cx)
        });
        return;
    }
    let panel = thread_workspace
        .as_ref()
        .or_else(|| metadata.folder_paths().is_empty().then_some(&workspace))
        .and_then(|workspace| workspace.read(cx).panel::<AgentPanel>(cx));
    let view = panel
        .as_ref()
        .and_then(|panel| {
            panel
                .read(cx)
                .conversation_view_for_id(&thread_id, cx)
                .cloned()
        })
        .and_then(|view| view.read(cx).root_thread_view());
    if matches!(export, TranscriptExport::Content)
        && metadata.agent_id.as_ref() == agent::ZED_AGENT_ID.as_ref()
        && let Some(view) = view.as_ref()
    {
        cx.write_to_clipboard(gpui::ClipboardItem::new_string(
            view.read(cx).thread.read(cx).to_markdown(cx),
        ));
        return;
    }
    let load = if metadata.agent_id.as_ref() != agent::ZED_AGENT_ID.as_ref() {
        let Some(panel) = panel.filter(|_| !matches!(export, TranscriptExport::Share)) else {
            workspace.update(cx, |workspace, cx| {
                workspace.show_error(
                    if matches!(export, TranscriptExport::Share) {
                        "This provider does not support sharing this task"
                    } else {
                        "Open this task's project before exporting its transcript"
                    },
                    cx,
                )
            });
            return;
        };
        let load = panel.update(cx, |panel, cx| {
            panel.load_transcript_for_export(thread_id, window, cx)
        });
        cx.spawn(async move |cx| {
            let view = load.await?;
            cx.update(|cx| {
                let thread = &view.read(cx).thread;
                match export {
                    TranscriptExport::Json => thread.read(cx).to_json(cx),
                    TranscriptExport::Content => Ok(thread.read(cx).to_markdown(cx)),
                    TranscriptExport::Share => anyhow::bail!("Provider sharing is unavailable"),
                }
            })
        })
    } else {
        let native = view.and_then(|view| view.read(cx).as_native_thread(cx));
        let load = if let Some(native) = native {
            let load = native.read(cx).to_db(cx);
            cx.spawn(async move |_| Ok(Some(load.await)))
        } else if let Some(session_id) = metadata.session_id.clone() {
            agent::ThreadStore::global(cx).update(cx, |store, cx| store.load_thread(session_id, cx))
        } else {
            workspace.update(cx, |workspace, cx| {
                workspace.show_error("This task has no saved transcript", cx)
            });
            return;
        };
        cx.spawn(async move |_| {
            let thread = load
                .await?
                .ok_or_else(|| anyhow::anyhow!("The task transcript is no longer available"))?;
            match export {
                TranscriptExport::Json => serde_json::to_string_pretty(&thread).map_err(Into::into),
                TranscriptExport::Content => Ok(thread.to_markdown()),
                TranscriptExport::Share => AgentPanel::shared_thread_clipboard_text(&thread),
            }
        })
    };
    sidebar.update(cx, |sidebar, cx| {
        sidebar.exporting_transcripts.insert(thread_id);
        cx.notify();
    });
    window
        .spawn(cx, async move |cx| {
            let result = load.await;
            cx.update(|_, cx| {
                sidebar.update(cx, |sidebar, cx| {
                    sidebar.exporting_transcripts.remove(&thread_id);
                    cx.notify();
                });
                let store = ThreadMetadataStore::global(cx);
                let store = store.read(cx);
                let result = result.and_then(|content| {
                    anyhow::ensure!(
                        !store.task_deletion_pending(thread_id)
                            && store.entry(thread_id).is_some_and(|current| {
                                current.session_id == metadata.session_id
                                    && current.agent_id == metadata.agent_id
                            }),
                        "The task changed while loading its transcript; reopen its menu and retry"
                    );
                    Ok(content)
                });
                match result {
                Ok(content) => {
                    cx.write_to_clipboard(gpui::ClipboardItem::new_string(content));
                    if matches!(export, TranscriptExport::Share) {
                        workspace.update(cx, |workspace, cx| {
                            struct SharedTask;
                            workspace.show_toast(workspace::Toast::new(workspace::notifications::NotificationId::unique::<SharedTask>(), "Task copied for sharing. Use Import Thread from Clipboard to open it.").autohide(), cx);
                        });
                    }
                },
                Err(error) => workspace.update(cx, |workspace, cx| {
                    workspace.show_error(error.to_string(), cx)
                }),
                }
            })
        })
        .detach_and_log_err(cx);
}

struct TaskRenameModal {
    metadata: ThreadMetadata,
    sidebar: WeakEntity<Sidebar>,
    editor: Entity<Editor>,
    error: Option<SharedString>,
}
impl TaskRenameModal {
    fn new(
        metadata: ThreadMetadata,
        sidebar: WeakEntity<Sidebar>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text(metadata.display_title(), window, cx);
            editor.select_all(&editor::actions::SelectAll, window, cx);
            editor
        });
        Self {
            metadata,
            sidebar,
            editor,
            error: None,
        }
    }
    fn confirm(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let title = self.editor.read(cx).text(cx).trim().to_owned();
        if title.is_empty() {
            self.error = Some("Enter a task title".into());
            cx.notify();
            return;
        }
        if ThreadMetadataStore::global(cx)
            .read(cx)
            .entry(self.metadata.thread_id)
            .is_none()
        {
            self.error = Some("This task is no longer available".into());
            cx.notify();
            return;
        }
        match self.sidebar.update(cx, |sidebar, cx| {
            sidebar.apply_thread_rename(self.metadata.thread_id, title.into(), window, cx)
        }) {
            Ok(()) => cx.emit(DismissEvent),
            Err(error) => {
                self.error = Some(error.to_string().into());
                cx.notify();
            }
        }
    }
}
impl EventEmitter<DismissEvent> for TaskRenameModal {}
impl ModalView for TaskRenameModal {}
impl Focusable for TaskRenameModal {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.editor.focus_handle(cx)
    }
}
impl Render for TaskRenameModal {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .w_96()
            .max_w_full()
            .elevation_3(cx)
            .key_context("Modal")
            .on_action(cx.listener(Self::confirm))
            .on_action(cx.listener(|_, _: &menu::Cancel, _, cx| cx.emit(DismissEvent)))
            .child(
                Modal::new("rename-task", None)
                    .header(
                        ModalHeader::new()
                            .headline("Rename task")
                            .show_dismiss_button(true),
                    )
                    .section(
                        Section::new()
                            .child(self.editor.clone())
                            .when_some(self.error.clone(), |this, error| {
                                this.child(Label::new(error).color(Color::Error))
                            }),
                    )
                    .footer(ModalFooter::new().end_slot(
                        Button::new("save-task-title", "Save").on_click(cx.listener(
                            |this, _, window, cx| this.confirm(&menu::Confirm, window, cx),
                        )),
                    )),
            )
    }
}

struct TaskDetailsModal {
    metadata: ThreadMetadata,
    focus: FocusHandle,
}
impl EventEmitter<DismissEvent> for TaskDetailsModal {}
impl ModalView for TaskDetailsModal {}
impl Focusable for TaskDetailsModal {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}
impl Render for TaskDetailsModal {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let metadata = &self.metadata;
        v_flex()
            .w_96()
            .max_w_full()
            .elevation_3(cx)
            .key_context("Modal")
            .track_focus(&self.focus)
            .on_action(cx.listener(|_, _: &menu::Cancel, _, cx| cx.emit(DismissEvent)))
            .child(
                Modal::new("task-details", None)
                    .header(
                        ModalHeader::new()
                            .headline("Task details")
                            .show_dismiss_button(true),
                    )
                    .section(
                        Section::new().children(
                            [
                                format!("Title: {}", metadata.display_title()),
                                format!("Task: {}", metadata.thread_id.to_key_string()),
                                format!("Provider: {}", metadata.agent_id.as_ref()),
                                format!(
                                    "Session: {}",
                                    metadata
                                        .session_id
                                        .as_ref()
                                        .map(|id| id.to_string())
                                        .unwrap_or_else(|| "Draft".into())
                                ),
                                format!("Updated: {}", metadata.updated_at.to_rfc3339()),
                                format!(
                                    "Created: {}",
                                    metadata
                                        .created_at
                                        .map(|date| date.to_rfc3339())
                                        .unwrap_or_else(|| "Unavailable".into())
                                ),
                                format!("Archived: {}", metadata.archived),
                                format!("Execution paths: {:?}", metadata.worktree_paths),
                                format!(
                                    "Project associations: {:?}",
                                    ThreadMetadataStore::global(cx)
                                        .read(cx)
                                        .task_projects(metadata.thread_id)
                                        .iter()
                                        .flat_map(|association| &association.paths)
                                        .collect::<Vec<_>>()
                                ),
                            ]
                            .into_iter()
                            .map(|text| Label::new(text).size(LabelSize::Default)),
                        ),
                    ),
            )
    }
}
