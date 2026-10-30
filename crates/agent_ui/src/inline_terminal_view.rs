use crate::TerminalId;
use crate::agent_panel::AgentPanel;
use crate::message_editor::{MessageEditor, MessageEditorEvent};
use crate::terminal_thread_metadata_store::TerminalThreadMetadataStore;
use gpui::{Entity, FocusHandle, Focusable, WeakEntity};
use terminal_view::TerminalView;
use ui::{Indicator, Tooltip, prelude::*};
use util::ResultExt;
use workspace::Workspace;

pub(crate) struct InlineTerminalView {
    terminal_id: TerminalId,
    occurrence_id: Option<uuid::Uuid>,
    terminal: Entity<TerminalView>,
    workspace: WeakEntity<Workspace>,
    history: WeakEntity<TerminalThreadMetadataStore>,
    editor: WeakEntity<MessageEditor>,
    collapsed: bool,
    block_id: Option<editor::display_map::CustomBlockId>,
    error: Option<SharedString>,
    _subscriptions: Vec<gpui::Subscription>,
}

impl InlineTerminalView {
    pub(crate) fn new(
        terminal_id: TerminalId,
        occurrence_id: Option<uuid::Uuid>,
        terminal: Entity<TerminalView>,
        workspace: WeakEntity<Workspace>,
        history: WeakEntity<TerminalThreadMetadataStore>,
        editor: WeakEntity<MessageEditor>,
        collapsed: bool,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut subscriptions = Vec::new();
        let is_draft = history.upgrade().is_some_and(|history| {
            history
                .read(cx)
                .entry(terminal_id)
                .and_then(|metadata| metadata.inline_context.as_ref())
                .and_then(|context| context.occurrence(occurrence_id))
                .is_some_and(|context| context.is_draft)
        });
        terminal.update(cx, |terminal, cx| {
            terminal.set_embedded_history_expanded(false, cx);
            terminal
                .set_embedded_read_only(!is_draft || !terminal.terminal().read(cx).is_pty(), cx);
        });
        if let Some(history) = history.upgrade() {
            subscriptions.push(cx.observe(&history, |this, history, cx| {
                if let Some(current) = history
                    .read(cx)
                    .entry(this.terminal_id)
                    .and_then(|metadata| metadata.inline_context.as_ref())
                    && let Some(context) = current.occurrence(this.occurrence_id)
                {
                    this.collapsed = context.collapsed;
                    if current.occurrence_id == this.occurrence_id && !context.is_draft {
                        this.terminal
                            .update(cx, |terminal, cx| terminal.set_embedded_read_only(true, cx));
                    }
                }
                cx.notify();
            }));
        }
        subscriptions.push(cx.observe(&terminal, |_, _, cx| cx.notify()));
        Self {
            terminal_id,
            occurrence_id,
            terminal,
            workspace,
            history,
            editor,
            collapsed,
            block_id: None,
            error: None,
            _subscriptions: subscriptions,
        }
    }

    fn occurrence_key(&self) -> crate::terminal_thread_metadata_store::InlineTerminalOccurrenceKey {
        crate::terminal_thread_metadata_store::InlineTerminalOccurrenceKey {
            terminal_id: self.terminal_id,
            occurrence_id: self.occurrence_id,
        }
    }

    pub(crate) fn set_block_id(&mut self, block_id: editor::display_map::CustomBlockId) {
        self.block_id = Some(block_id);
    }

    pub(crate) fn prompt_content(
        &self,
        position: crate::terminal_thread_metadata_store::InlineTerminalDraftPosition,
        cx: &App,
    ) -> anyhow::Result<(
        agent_client_protocol::schema::v1::ContentBlock,
        Entity<TerminalThreadMetadataStore>,
        crate::terminal_thread_metadata_store::TerminalThreadMetadata,
    )> {
        use crate::terminal_thread_metadata_store::InlineTerminalPromptReference;
        let history = TerminalThreadMetadataStore::try_global(cx)
            .filter(|store| store.entity_id() == self.history.entity_id())
            .ok_or_else(|| anyhow::anyhow!("The terminal history owner has changed"))?;
        let workspace = self
            .workspace
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("The terminal workspace is unavailable"))?;
        let panel = workspace
            .read(cx)
            .panel::<AgentPanel>(cx)
            .ok_or_else(|| anyhow::anyhow!("The terminal execution owner is unavailable"))?;
        let current = history
            .read(cx)
            .entry(self.terminal_id)
            .and_then(|metadata| metadata.inline_context.as_ref())
            .is_some_and(|context| context.occurrence_id == self.occurrence_id);
        let mut metadata = if current
            && panel
                .read(cx)
                .inline_terminal_view(self.terminal_id)
                .is_some()
        {
            let mut metadata = panel
                .read(cx)
                .inline_terminal_snapshot_metadata(self.terminal_id, cx)?
                .1;
            if let Some(context) = metadata.inline_context.as_mut() {
                context.previous_occurrences.clear();
            }
            metadata
        } else {
            history
                .read(cx)
                .occurrence_entry(self.occurrence_key())
                .ok_or_else(|| anyhow::anyhow!("The terminal occurrence history is unavailable"))?
        };
        anyhow::ensure!(
            remote::same_remote_connection_identity(
                metadata.remote_connection.as_ref(),
                workspace
                    .read(cx)
                    .project()
                    .read(cx)
                    .remote_connection_options(cx)
                    .as_ref(),
            ),
            "The inline terminal belongs to a different remote project"
        );
        let context = metadata
            .inline_context
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("The terminal conversation is unavailable"))?;
        let reference = InlineTerminalPromptReference {
            occurrence_id: self.occurrence_id,

            terminal_id: self.terminal_id,
            thread_id: context.thread_id,
            session_id: context
                .session_id
                .clone()
                .ok_or_else(|| anyhow::anyhow!("The terminal session is unavailable"))?,
            position: position.clone(),
        };
        let content = reference.content(context.output.clone())?;
        context.draft_position = Some(position);
        Ok((content, history, metadata))
    }

    fn toggle(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !TerminalThreadMetadataStore::try_global(cx)
            .is_some_and(|store| store.entity_id() == self.history.entity_id())
        {
            self.error = Some("The terminal history owner has changed".into());
            cx.notify();
            return;
        }
        let workspace = match self.workspace.upgrade() {
            Some(workspace) => workspace,
            None => {
                self.error = Some("The terminal workspace is unavailable".into());
                cx.notify();
                return;
            }
        };
        let result = self
            .history
            .update(cx, |history, cx| {
                let mut metadata = history
                    .occurrence_entry(self.occurrence_key())
                    .ok_or_else(|| anyhow::anyhow!("The inline terminal history is unavailable"))?;
                anyhow::ensure!(
                    remote::same_remote_connection_identity(
                        metadata.remote_connection.as_ref(),
                        workspace
                            .read(cx)
                            .project()
                            .read(cx)
                            .remote_connection_options(cx)
                            .as_ref(),
                    ),
                    "The inline terminal belongs to a different remote project"
                );
                let context = metadata.inline_context.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("The terminal has no conversation association")
                })?;
                context.collapsed = !self.collapsed;
                history.save_inline_occurrence_batch(vec![metadata], cx)
            })
            .and_then(|result| result);
        match result {
            Ok(()) => {
                self.collapsed = !self.collapsed;
                self.error = None;
                if self.collapsed && self.terminal.focus_handle(cx).contains_focused(window, cx) {
                    if let Some(block_id) = self.block_id {
                        self.editor
                            .update(cx, |editor, cx| {
                                editor.focus_after_inline_terminal(block_id, window, cx)
                            })
                            .and_then(|result| result)
                            .log_err();
                    }
                }
                self.editor.update(cx, |_, cx| cx.notify()).log_err();
            }
            Err(error) => self.error = Some(error.to_string().into()),
        }
        cx.notify();
    }

    fn return_to_conversation(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        self.block_id
            .ok_or_else(|| anyhow::anyhow!("The terminal placement is unavailable"))
            .and_then(|block_id| {
                self.editor
                    .update(cx, |editor, cx| {
                        editor.focus_after_inline_terminal(block_id, window, cx)
                    })
                    .and_then(|result| result)
            })
    }

    fn release_focus(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.error = self
            .return_to_conversation(window, cx)
            .err()
            .map(|error| error.to_string().into());
        cx.notify();
    }

    fn toggle_history(&mut self, focused: bool, window: &mut Window, cx: &mut Context<Self>) {
        let expanded = self.terminal.update(cx, |terminal, cx| {
            let expanded = !terminal.embedded_history_expanded();
            terminal.set_embedded_history_expanded(expanded, cx);
            if focused && !terminal.is_read_only() {
                terminal.focus_handle(cx).focus(window, cx);
            }
            expanded
        });
        self.editor
            .update(cx, |_, cx| {
                if expanded {
                    cx.emit(MessageEditorEvent::InlineTerminalHistoryExpanded);
                }
                cx.notify();
            })
            .log_err();
        cx.notify();
    }

    fn background(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match self.return_to_conversation(window, cx) {
            Ok(()) => {
                self.terminal
                    .update(cx, |terminal, cx| terminal.set_embedded_read_only(true, cx));
                self.error = None;
            }
            Err(error) => self.error = Some(error.to_string().into()),
        }
        cx.notify();
    }
}

impl Focusable for InlineTerminalView {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.terminal.focus_handle(cx)
    }
}

impl Render for InlineTerminalView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let current = self
            .history
            .read_with(cx, |history, _| {
                history
                    .entry(self.terminal_id)
                    .and_then(|metadata| metadata.inline_context.as_ref())
                    .is_some_and(|context| context.occurrence_id == self.occurrence_id)
            })
            .unwrap_or(false);
        let interactive = current
            && self.terminal.read(cx).terminal().read(cx).is_pty()
            && self
                .workspace
                .read_with(cx, |workspace, cx| {
                    workspace
                        .panel::<AgentPanel>(cx)
                        .is_some_and(|panel| panel.read(cx).has_terminal(self.terminal_id))
                })
                .unwrap_or(false);
        let draft = self
            .history
            .read_with(cx, |history, _| {
                history
                    .entry(self.terminal_id)
                    .and_then(|metadata| metadata.inline_context.as_ref())
                    .and_then(|context| context.occurrence(self.occurrence_id))
                    .is_some_and(|context| context.is_draft)
            })
            .unwrap_or(false);
        let backgrounded = self.terminal.read(cx).is_read_only();
        let focused = !backgrounded && self.terminal.focus_handle(cx).contains_focused(window, cx);
        let title = self
            .history
            .read_with(cx, |history, _| {
                history
                    .entry(self.terminal_id)
                    .map(|metadata| metadata.display_title())
            })
            .ok()
            .flatten()
            .filter(|title| !title.trim().is_empty())
            .unwrap_or_else(|| "Terminal".into());
        let history_error = self
            .history
            .read_with(cx, |history, _| {
                history.write_error(self.terminal_id).cloned()
            })
            .ok()
            .flatten();
        let error = self.error.clone().or(history_error);
        let hidden_lines = self.terminal.read(cx).embedded_hidden_lines(window, cx);
        let history_expanded = self.terminal.read(cx).embedded_history_expanded();
        let scrollable = self
            .terminal
            .read(cx)
            .content_mode(window, cx)
            .is_scrollable();
        v_flex()
            .w_full()
            .min_w_0()
            .border_1()
            .border_color(if focused {
                cx.theme().colors().border_focused
            } else {
                cx.theme().colors().border
            })
            .rounded_sm()
            .child(
                h_flex()
                    .gap_1()
                    .px_2()
                    .py_1()
                    .w_full()
                    .child(
                        Button::new("inline-terminal-disclosure", title)
                            .size(ButtonSize::Compact)
                            .tab_index(0isize)
                            .aria_label(if self.collapsed {
                                "Expand terminal output"
                            } else {
                                "Collapse terminal output"
                            })
                            .style(ButtonStyle::Transparent)
                            .start_icon(
                                Icon::new(if self.collapsed {
                                    IconName::ChevronRight
                                } else {
                                    IconName::ChevronDown
                                })
                                .size(IconSize::XSmall),
                            )
                            .tooltip(Tooltip::text(if self.collapsed {
                                "Expand terminal output"
                            } else {
                                "Collapse terminal output"
                            }))
                            .on_click(cx.listener(|this, _, window, cx| this.toggle(window, cx))),
                    )
                    .child(div().flex_1())
                    .when(interactive && draft && !backgrounded, |header| {
                        header.child(
                            div()
                                .debug_selector(|| "inline-terminal-background".into())
                                .child(
                                    Button::new("inline-terminal-background", "Background")
                                        .size(ButtonSize::Compact)
                                        .tab_index(0isize)
                                        .aria_label(
                                            "Keep this terminal running and return to the conversation",
                                        )
                                        .style(ButtonStyle::Transparent)
                                        .tooltip(Tooltip::text(
                                            "Keep this terminal running and return to the conversation",
                                        ))
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.background(window, cx)
                                        })),
                                ),
                        )
                    })
                    .when(interactive && draft && backgrounded, |header| {
                        header.child(
                            h_flex()
                                .gap_1()
                                .child(Indicator::dot().color(Color::Accent))
                                .child(
                                    Label::new("Background")
                                        .size(LabelSize::Small)
                                        .color(Color::Muted),
                                ),
                        )
                    })
                    .when(interactive && draft && !backgrounded && focused, |header| {
                        header.child(
                            div()
                                .debug_selector(|| "inline-terminal-release-focus".into())
                                .child(
                                    IconButton::new(
                                        "inline-terminal-release-focus",
                                        IconName::Escape,
                                    )
                                    .size(ButtonSize::Compact)
                                    .tab_index(0isize)
                                    .aria_label(
                                        "Return to the conversation without stopping the terminal",
                                    )
                                    .tooltip(Tooltip::text("Return to the conversation (Escape)"))
                                    .on_click(cx.listener(
                                        |this, _, window, cx| this.release_focus(window, cx),
                                    )),
                                ),
                        )
                    })
                    .when(
                        interactive && draft && !backgrounded && !focused && !self.collapsed,
                        |header| {
                            header.child(
                                div()
                                    .debug_selector(|| "inline-terminal-focus".into())
                                    .child(
                                        IconButton::new("inline-terminal-focus", IconName::Return)
                                            .size(ButtonSize::Compact)
                                            .tab_index(0isize)
                                            .aria_label("Focus this terminal")
                                            .tooltip(Tooltip::text("Focus this terminal"))
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.terminal.focus_handle(cx).focus(window, cx);
                                            })),
                                    ),
                            )
                        },
                    ),
            )
            .when(
                !self.collapsed && (hidden_lines > 0 || history_expanded),
                |view| {
                    view.child(
                        h_flex()
                            .w_full()
                            .gap_2()
                            .px_2()
                            .child(
                                div()
                                    .debug_selector(|| "inline-terminal-history-disclosure".into())
                                    .child(
                                        Button::new(
                                            "inline-terminal-history-disclosure",
                                            if history_expanded {
                                                "Show Less".into()
                                            } else {
                                                format!("Show {hidden_lines} More Lines")
                                            },
                                        )
                                        .size(ButtonSize::Compact)
                                        .style(ButtonStyle::Transparent)
                                        .tab_index(0isize)
                                        .aria_label(if history_expanded {
                                            "Show the latest terminal output"
                                        } else {
                                            "Expand earlier terminal output in the conversation"
                                        })
                                        .start_icon(
                                            Icon::new(if history_expanded {
                                                IconName::ChevronDown
                                            } else {
                                                IconName::ChevronRight
                                            })
                                            .size(IconSize::XSmall),
                                        )
                                        .tooltip(Tooltip::text(if history_expanded {
                                            "Show the latest terminal output"
                                        } else {
                                            "Expand earlier terminal output without stopping the process"
                                        }))
                                        .on_click(cx.listener(
                                            move |this, _, window, cx| {
                                                this.toggle_history(focused, window, cx)
                                            },
                                        )),
                                    ),
                            )
                            .child(div().flex_1().h_px().bg(cx.theme().colors().border)),
                    )
                },
            )
            .when(!self.collapsed, |view| {
                view.child(
                    div()
                        .w_full()
                        .min_w_0()
                        .when(scrollable, |body| body.h_72())
                        .child(self.terminal.clone()),
                )
            })
            .when_some(error, |view, error| {
                view.child(
                    div()
                        .px_2()
                        .py_1()
                        .child(Label::new(error).size(LabelSize::Small).color(Color::Error)),
                )
            })
    }
}
