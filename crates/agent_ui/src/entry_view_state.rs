use std::{ops::Range, sync::Arc};

use acp_thread::{AcpThread, AgentThreadEntry, AssistantMessageChunk};
use agent::ThreadStore;
use agent_client_protocol::schema::v1 as acp;
use agent_settings::AgentSettings;
use collections::{HashMap, HashSet};
use editor::{
    Editor, EditorEvent, EditorMode, MinimapVisibility, RestoreOnlyUnstagedDiffHunkDelegate,
    SizingBehavior,
};
use gpui::{
    AnyEntity, App, AppContext as _, Entity, EntityId, EventEmitter, FocusHandle, Focusable,
    ScrollHandle, TextStyleRefinement, WeakEntity, Window,
};
use language::language_settings::SoftWrap;
use project::{AgentId, Project, project_settings::DiagnosticSeverity};
use rope::Point;
use settings::{Settings as _, ThinkingBlockDisplay};
use terminal_view::TerminalView;
use theme_settings::ThemeSettings;
use ui::Context;
use workspace::Workspace;

use crate::message_editor::{
    MessageEditor, MessageEditorEvent, MessageEditorInputRevision, SharedSessionCapabilities,
};

/// Maps an entry index through the removal of `removed` (a contiguous range of
/// entries), returning `None` if the index referred to a removed entry.
fn reindex_after_removal(index: usize, removed: &Range<usize>) -> Option<usize> {
    if index < removed.start {
        Some(index)
    } else if index < removed.end {
        None
    } else {
        Some(index - removed.len())
    }
}

pub struct EntryViewState {
    workspace: WeakEntity<Workspace>,
    project: WeakEntity<Project>,
    thread_store: Option<Entity<ThreadStore>>,
    entries: Vec<Entry>,
    #[cfg(feature = "multiplayer-tools")]
    inline_terminal_scope: Option<(crate::thread_metadata_store::ThreadId, acp::SessionId)>,
    session_capabilities: SharedSessionCapabilities,
    agent_id: AgentId,
    expanded_thinking_blocks: HashSet<(usize, usize)>,
    auto_expanded_thinking_block: Option<(usize, usize)>,
    user_toggled_thinking_blocks: HashSet<(usize, usize)>,
    expanded_compactions: HashSet<usize>,
    expanded_tool_calls: HashSet<acp::ToolCallId>,
}

impl EntryViewState {
    pub fn new(
        workspace: WeakEntity<Workspace>,
        project: WeakEntity<Project>,
        thread_store: Option<Entity<ThreadStore>>,
        session_capabilities: SharedSessionCapabilities,
        agent_id: AgentId,
    ) -> Self {
        Self {
            workspace,
            project,
            thread_store,
            entries: Vec::new(),
            #[cfg(feature = "multiplayer-tools")]
            inline_terminal_scope: None,
            session_capabilities,
            agent_id,
            expanded_thinking_blocks: HashSet::default(),
            auto_expanded_thinking_block: None,
            user_toggled_thinking_blocks: HashSet::default(),
            expanded_compactions: HashSet::default(),
            expanded_tool_calls: HashSet::default(),
        }
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn set_inline_terminal_scope(
        &mut self,
        thread_id: crate::thread_metadata_store::ThreadId,
        session_id: acp::SessionId,
        cx: &mut Context<Self>,
    ) {
        for entry in &self.entries {
            if let Some(editor) = entry.message_editor() {
                editor.update(cx, |editor, _| {
                    editor.set_inline_terminal_scope(thread_id, session_id.clone());
                });
            }
        }
        self.inline_terminal_scope = Some((thread_id, session_id));
    }

    pub(crate) fn is_tool_call_expanded(&self, tool_call_id: &acp::ToolCallId) -> bool {
        self.expanded_tool_calls.contains(tool_call_id)
    }

    pub(crate) fn expand_tool_call(&mut self, tool_call_id: acp::ToolCallId) {
        self.expanded_tool_calls.insert(tool_call_id);
    }

    pub(crate) fn collapse_tool_call(&mut self, tool_call_id: &acp::ToolCallId) {
        self.expanded_tool_calls.remove(tool_call_id);
    }

    pub(crate) fn toggle_tool_call_expansion(&mut self, tool_call_id: &acp::ToolCallId) {
        if !self.expanded_tool_calls.remove(tool_call_id) {
            self.expanded_tool_calls.insert(tool_call_id.clone());
        }
    }

    pub(crate) fn is_compaction_expanded(&self, entry_ix: usize) -> bool {
        self.expanded_compactions.contains(&entry_ix)
    }

    pub(crate) fn collapse_compaction(&mut self, entry_ix: usize) {
        self.expanded_compactions.remove(&entry_ix);
    }

    pub(crate) fn toggle_compaction_expansion(&mut self, entry_ix: usize) {
        if !self.expanded_compactions.remove(&entry_ix) {
            self.expanded_compactions.insert(entry_ix);
        }
    }

    pub(crate) fn clear_auto_expand_tracking(&mut self) {
        self.auto_expanded_thinking_block = None;
    }

    pub(crate) fn is_auto_expanded_thinking_block(&self, key: (usize, usize)) -> bool {
        self.auto_expanded_thinking_block == Some(key)
    }

    pub(crate) fn auto_expand_streaming_thought(&mut self, thread: &AcpThread, cx: &App) -> bool {
        let thinking_display = AgentSettings::get_global(cx).thinking_display;

        if !matches!(
            thinking_display,
            ThinkingBlockDisplay::Auto | ThinkingBlockDisplay::Preview
        ) {
            return false;
        }

        let last_ix = thread.entries().len().saturating_sub(1);
        let key = match thread.entries().get(last_ix) {
            Some(AgentThreadEntry::AssistantMessage(message)) => match message.chunks.last() {
                Some(AssistantMessageChunk::Thought { .. }) => {
                    Some((last_ix, message.chunks.len() - 1))
                }
                _ => None,
            },
            _ => None,
        };

        if let Some(key) = key {
            if self.auto_expanded_thinking_block != Some(key) {
                self.auto_expanded_thinking_block = Some(key);
                self.expanded_thinking_blocks.insert(key);
                return true;
            }
        } else if self.auto_expanded_thinking_block.is_some() {
            if thinking_display == ThinkingBlockDisplay::Auto
                && let Some(key) = self.auto_expanded_thinking_block
                && !self.user_toggled_thinking_blocks.contains(&key)
            {
                self.expanded_thinking_blocks.remove(&key);
            }
            self.auto_expanded_thinking_block = None;
            return true;
        }

        false
    }

    pub(crate) fn toggle_thinking_block_expansion(&mut self, key: (usize, usize), cx: &App) {
        match AgentSettings::get_global(cx).thinking_display {
            ThinkingBlockDisplay::Auto => {
                let is_open = self.expanded_thinking_blocks.contains(&key)
                    || self.user_toggled_thinking_blocks.contains(&key);

                if is_open {
                    self.expanded_thinking_blocks.remove(&key);
                    self.user_toggled_thinking_blocks.remove(&key);
                } else {
                    self.expanded_thinking_blocks.insert(key);
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
            ThinkingBlockDisplay::Preview => {
                let is_user_expanded = self.user_toggled_thinking_blocks.contains(&key);
                let is_in_expanded_set = self.expanded_thinking_blocks.contains(&key);

                if is_user_expanded {
                    self.user_toggled_thinking_blocks.remove(&key);
                    self.expanded_thinking_blocks.remove(&key);
                } else if is_in_expanded_set {
                    self.user_toggled_thinking_blocks.insert(key);
                } else {
                    self.expanded_thinking_blocks.insert(key);
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
            ThinkingBlockDisplay::AlwaysExpanded => {
                if self.user_toggled_thinking_blocks.contains(&key) {
                    self.user_toggled_thinking_blocks.remove(&key);
                } else {
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
            ThinkingBlockDisplay::AlwaysCollapsed => {
                if self.user_toggled_thinking_blocks.contains(&key) {
                    self.user_toggled_thinking_blocks.remove(&key);
                    self.expanded_thinking_blocks.remove(&key);
                } else {
                    self.expanded_thinking_blocks.insert(key);
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
        }
    }

    pub(crate) fn thinking_block_state(&self, key: (usize, usize), cx: &App) -> (bool, bool) {
        let is_user_toggled = self.user_toggled_thinking_blocks.contains(&key);
        let is_in_expanded_set = self.expanded_thinking_blocks.contains(&key);

        match AgentSettings::get_global(cx).thinking_display {
            ThinkingBlockDisplay::Auto => {
                let is_open = is_user_toggled || is_in_expanded_set;
                (is_open, false)
            }
            ThinkingBlockDisplay::Preview => {
                let is_open = is_user_toggled || is_in_expanded_set;
                let is_constrained = is_in_expanded_set && !is_user_toggled;
                (is_open, is_constrained)
            }
            ThinkingBlockDisplay::AlwaysExpanded => (!is_user_toggled, false),
            ThinkingBlockDisplay::AlwaysCollapsed => (is_user_toggled, false),
        }
    }

    pub fn entry(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index)
    }

    pub fn sync_entry(
        &mut self,
        index: usize,
        thread: &Entity<AcpThread>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(thread_entry) = thread.read(cx).entries().get(index) else {
            return;
        };

        match thread_entry {
            AgentThreadEntry::UserMessage(message) => {
                let can_rewind = thread.read(cx).supports_truncate(cx);
                let has_client_id = message.client_id.is_some();
                let is_subagent = thread.read(cx).parent_session_id().is_some();
                let chunks = message.chunks.clone();
                #[cfg(feature = "multiplayer-tools")]
                let chunks = if let Some((thread_id, session_id)) =
                    self.inline_terminal_scope.as_ref()
                    && let Some(client_id) = message.client_id.as_ref()
                    && let Some(project) = self.project.upgrade()
                    && let Some(history) = crate::terminal_thread_metadata_store::
                        TerminalThreadMetadataStore::try_global(cx)
                {
                    let remote = project.read(cx).remote_connection_options(cx);
                    match history.read(cx).recover_submitted_terminal_references(
                        *thread_id,
                        session_id,
                        remote.as_ref(),
                        client_id,
                        &chunks,
                    ) {
                        Ok(recovered) => recovered,
                        Err(error) => {
                            log::error!("Failed to recover submitted terminal context: {error:#}");
                            chunks
                        }
                    }
                } else {
                    chunks
                };
                if let Some(Entry::UserMessage(entry)) = self.entries.get_mut(index) {
                    let editor = &entry.editor;
                    if !editor.focus_handle(cx).is_focused(window)
                        && (entry.last_synced_chunks != chunks
                            || entry.last_synced_revision != editor.read(cx).input_revision(cx))
                    {
                        // Resetting an unchanged projection moves character-boundary anchors and cursor state.
                        // Local edits still reset after focus leaves, including an edit followed by Undo.
                        entry.last_synced_revision = editor.update(cx, |editor, cx| {
                            editor.set_message(chunks.clone(), window, cx);
                            editor.input_revision(cx)
                        });
                        entry.last_synced_chunks = chunks;
                    }
                } else {
                    let message_editor = cx.new(|cx| {
                        let mut editor = MessageEditor::new(
                            self.workspace.clone(),
                            self.project.clone(),
                            self.thread_store.clone(),
                            self.session_capabilities.clone(),
                            self.agent_id.clone(),
                            "Edit message － @ to include context",
                            editor::EditorMode::AutoHeight {
                                min_lines: 1,
                                max_lines: None,
                            },
                            window,
                            cx,
                        );
                        if !can_rewind || !has_client_id || is_subagent {
                            editor.set_read_only(true, cx);
                        }
                        #[cfg(feature = "multiplayer-tools")]
                        if let Some((thread_id, session_id)) = self.inline_terminal_scope.as_ref() {
                            editor.set_inline_terminal_scope(*thread_id, session_id.clone());
                        }
                        editor.set_message(chunks.clone(), window, cx);
                        editor
                    });
                    cx.subscribe(&message_editor, move |_, editor, event, cx| {
                        cx.emit(EntryViewEvent {
                            entry_index: index,
                            view_event: ViewEvent::MessageEditorEvent(editor, event.clone()),
                        })
                    })
                    .detach();
                    let last_synced_revision = message_editor.read(cx).input_revision(cx);
                    self.set_entry(
                        index,
                        Entry::UserMessage(UserMessageEntry {
                            editor: message_editor,
                            last_synced_chunks: chunks,
                            last_synced_revision,
                        }),
                    );
                }
            }
            AgentThreadEntry::ToolCall(tool_call) => {
                let id = tool_call.id.clone();
                let terminals = tool_call.terminals().cloned().collect::<Vec<_>>();
                let diffs = tool_call.diffs().cloned().collect::<Vec<_>>();

                let views = if let Some(Entry::ToolCall(tool_call)) = self.entries.get_mut(index) {
                    &mut tool_call.content
                } else {
                    self.set_entry(
                        index,
                        Entry::ToolCall(ToolCallEntry {
                            content: HashMap::default(),
                            focus_handle: cx.focus_handle(),
                        }),
                    );
                    let Some(Entry::ToolCall(tool_call)) = self.entries.get_mut(index) else {
                        unreachable!()
                    };
                    &mut tool_call.content
                };

                let is_tool_call_completed =
                    matches!(tool_call.status, acp_thread::ToolCallStatus::Completed);

                for terminal in terminals {
                    match views.entry(terminal.entity_id()) {
                        collections::hash_map::Entry::Vacant(entry) => {
                            let element = create_terminal(
                                self.workspace.clone(),
                                self.project.clone(),
                                terminal.clone(),
                                window,
                                cx,
                            )
                            .into_any();
                            cx.emit(EntryViewEvent {
                                entry_index: index,
                                view_event: ViewEvent::NewTerminal(id.clone()),
                            });
                            entry.insert(element);
                        }
                        collections::hash_map::Entry::Occupied(_entry) => {
                            if is_tool_call_completed && terminal.read(cx).output().is_none() {
                                cx.emit(EntryViewEvent {
                                    entry_index: index,
                                    view_event: ViewEvent::TerminalMovedToBackground(id.clone()),
                                });
                            }
                        }
                    }
                }

                for diff in diffs {
                    views.entry(diff.entity_id()).or_insert_with(|| {
                        let editor = create_editor_diff(diff.clone(), window, cx);
                        cx.subscribe(&editor, {
                            let diff = diff.clone();
                            let entry_index = index;
                            move |_this, _editor, event: &EditorEvent, cx| {
                                if let EditorEvent::OpenExcerptsRequested {
                                    selections_by_buffer,
                                    split,
                                } = event
                                {
                                    let multibuffer = diff.read(cx).multibuffer();
                                    if let Some((buffer_id, (ranges, _))) =
                                        selections_by_buffer.iter().next()
                                    {
                                        if let Some(buffer) =
                                            multibuffer.read(cx).buffer(*buffer_id)
                                        {
                                            if let Some(range) = ranges.first() {
                                                let point =
                                                    buffer.read(cx).offset_to_point(range.start.0);
                                                if let Some(path) = diff.read(cx).file_path(cx) {
                                                    cx.emit(EntryViewEvent {
                                                        entry_index,
                                                        view_event: ViewEvent::OpenDiffLocation {
                                                            path,
                                                            position: point,
                                                            split: *split,
                                                        },
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        })
                        .detach();
                        cx.emit(EntryViewEvent {
                            entry_index: index,
                            view_event: ViewEvent::NewDiff(id.clone()),
                        });
                        editor.into_any()
                    });
                }
            }
            AgentThreadEntry::Elicitation(_) => {
                if !matches!(self.entries.get(index), Some(Entry::Elicitation { .. })) {
                    self.set_entry(
                        index,
                        Entry::Elicitation {
                            focus_handle: cx.focus_handle(),
                        },
                    );
                }
            }
            AgentThreadEntry::AssistantMessage(message) => {
                let entry = if let Some(Entry::AssistantMessage(entry)) =
                    self.entries.get_mut(index)
                {
                    entry
                } else {
                    self.set_entry(
                        index,
                        Entry::AssistantMessage(AssistantMessageEntry {
                            scroll_handles_by_chunk_index: HashMap::default(),
                            focus_handle: cx.focus_handle(),
                        }),
                    );
                    let Some(Entry::AssistantMessage(entry)) = self.entries.get_mut(index) else {
                        unreachable!()
                    };
                    entry
                };
                entry.sync(message);
            }
            AgentThreadEntry::CompletedPlan(_) => {
                if !matches!(self.entries.get(index), Some(Entry::CompletedPlan)) {
                    self.set_entry(index, Entry::CompletedPlan);
                }
            }
            AgentThreadEntry::ContextCompaction(_) => {
                if !matches!(self.entries.get(index), Some(Entry::ContextCompaction)) {
                    self.set_entry(index, Entry::ContextCompaction);
                }
            }
        };
    }

    fn set_entry(&mut self, index: usize, entry: Entry) {
        if index == self.entries.len() {
            self.entries.push(entry);
        } else {
            self.entries[index] = entry;
        }
    }

    pub fn remove(&mut self, range: Range<usize>) {
        self.entries.drain(range.clone());

        self.expanded_compactions = self
            .expanded_compactions
            .iter()
            .filter_map(|&entry_ix| reindex_after_removal(entry_ix, &range))
            .collect();
        self.expanded_thinking_blocks = self
            .expanded_thinking_blocks
            .iter()
            .filter_map(|&(entry_ix, chunk_ix)| {
                reindex_after_removal(entry_ix, &range).map(|entry_ix| (entry_ix, chunk_ix))
            })
            .collect();
        self.user_toggled_thinking_blocks = self
            .user_toggled_thinking_blocks
            .iter()
            .filter_map(|&(entry_ix, chunk_ix)| {
                reindex_after_removal(entry_ix, &range).map(|entry_ix| (entry_ix, chunk_ix))
            })
            .collect();
        self.auto_expanded_thinking_block =
            self.auto_expanded_thinking_block
                .and_then(|(entry_ix, chunk_ix)| {
                    reindex_after_removal(entry_ix, &range).map(|entry_ix| (entry_ix, chunk_ix))
                });
    }

    pub fn agent_ui_font_size_changed(&mut self, cx: &mut App) {
        for entry in self.entries.iter() {
            match entry {
                Entry::UserMessage { .. }
                | Entry::AssistantMessage { .. }
                | Entry::Elicitation { .. }
                | Entry::CompletedPlan
                | Entry::ContextCompaction => {}
                Entry::ToolCall(ToolCallEntry { content, .. }) => {
                    for view in content.values() {
                        if let Ok(diff_editor) = view.clone().downcast::<Editor>() {
                            diff_editor.update(cx, |diff_editor, cx| {
                                diff_editor.set_text_style_refinement(
                                    diff_editor_text_style_refinement(cx),
                                );
                                cx.notify();
                            })
                        }
                    }
                }
            }
        }
    }
}

impl EventEmitter<EntryViewEvent> for EntryViewState {}

pub struct EntryViewEvent {
    pub entry_index: usize,
    pub view_event: ViewEvent,
}

pub enum ViewEvent {
    NewDiff(acp::ToolCallId),
    NewTerminal(acp::ToolCallId),
    TerminalMovedToBackground(acp::ToolCallId),
    MessageEditorEvent(Entity<MessageEditor>, MessageEditorEvent),
    OpenDiffLocation {
        path: String,
        position: Point,
        split: bool,
    },
}

#[derive(Debug)]
pub struct AssistantMessageEntry {
    scroll_handles_by_chunk_index: HashMap<usize, ScrollHandle>,
    focus_handle: FocusHandle,
}

impl AssistantMessageEntry {
    pub fn scroll_handle_for_chunk(&self, ix: usize) -> Option<ScrollHandle> {
        self.scroll_handles_by_chunk_index.get(&ix).cloned()
    }

    pub fn sync(&mut self, message: &acp_thread::AssistantMessage) {
        if let Some(acp_thread::AssistantMessageChunk::Thought { .. }) = message.chunks.last() {
            let ix = message.chunks.len() - 1;
            let handle = self.scroll_handles_by_chunk_index.entry(ix).or_default();
            handle.scroll_to_bottom();
        }
    }
}

#[derive(Debug)]
pub struct ToolCallEntry {
    content: HashMap<EntityId, AnyEntity>,
    focus_handle: FocusHandle,
}

#[derive(Debug)]
pub struct UserMessageEntry {
    editor: Entity<MessageEditor>,
    last_synced_chunks: Vec<acp::ContentBlock>,
    last_synced_revision: MessageEditorInputRevision,
}

#[derive(Debug)]
pub enum Entry {
    UserMessage(UserMessageEntry),
    AssistantMessage(AssistantMessageEntry),
    ToolCall(ToolCallEntry),
    Elicitation { focus_handle: FocusHandle },
    CompletedPlan,
    ContextCompaction,
}

impl Entry {
    pub fn focus_handle(&self, cx: &App) -> Option<FocusHandle> {
        match self {
            Self::UserMessage(entry) => Some(entry.editor.read(cx).focus_handle(cx)),
            Self::AssistantMessage(message) => Some(message.focus_handle.clone()),
            Self::ToolCall(tool_call) => Some(tool_call.focus_handle.clone()),
            Self::Elicitation { focus_handle } => Some(focus_handle.clone()),
            Self::CompletedPlan | Self::ContextCompaction => None,
        }
    }

    pub fn message_editor(&self) -> Option<&Entity<MessageEditor>> {
        match self {
            Self::UserMessage(entry) => Some(&entry.editor),
            Self::AssistantMessage(_)
            | Self::ToolCall(_)
            | Self::Elicitation { .. }
            | Self::CompletedPlan
            | Self::ContextCompaction => None,
        }
    }

    pub fn editor_for_diff(&self, diff: &Entity<acp_thread::Diff>) -> Option<Entity<Editor>> {
        self.content_map()?
            .get(&diff.entity_id())
            .cloned()
            .and_then(|entity| entity.downcast::<Editor>().ok())
    }

    pub fn terminal(
        &self,
        terminal: &Entity<acp_thread::Terminal>,
    ) -> Option<Entity<TerminalView>> {
        self.content_map()?
            .get(&terminal.entity_id())
            .cloned()
            .and_then(|entity| entity.downcast::<TerminalView>().ok())
    }

    pub fn scroll_handle_for_assistant_message_chunk(
        &self,
        chunk_ix: usize,
    ) -> Option<ScrollHandle> {
        match self {
            Self::AssistantMessage(message) => message.scroll_handle_for_chunk(chunk_ix),
            Self::UserMessage(_)
            | Self::ToolCall(_)
            | Self::Elicitation { .. }
            | Self::CompletedPlan
            | Self::ContextCompaction => None,
        }
    }

    fn content_map(&self) -> Option<&HashMap<EntityId, AnyEntity>> {
        match self {
            Self::ToolCall(ToolCallEntry { content, .. }) => Some(content),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn has_content(&self) -> bool {
        match self {
            Self::ToolCall(ToolCallEntry { content, .. }) => !content.is_empty(),
            Self::UserMessage(_)
            | Self::AssistantMessage(_)
            | Self::Elicitation { .. }
            | Self::CompletedPlan
            | Self::ContextCompaction => false,
        }
    }
}

impl Focusable for ToolCallEntry {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Focusable for Entry {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        match self {
            Self::UserMessage(entry) => entry.editor.read(cx).focus_handle(cx),
            Self::AssistantMessage(message) => message.focus_handle.clone(),
            Self::ToolCall(tool_call) => tool_call.focus_handle.clone(),
            Self::Elicitation { focus_handle } => focus_handle.clone(),
            Self::CompletedPlan | Self::ContextCompaction => cx.focus_handle(),
        }
    }
}

fn create_terminal(
    workspace: WeakEntity<Workspace>,
    project: WeakEntity<Project>,
    terminal: Entity<acp_thread::Terminal>,
    window: &mut Window,
    cx: &mut App,
) -> Entity<TerminalView> {
    cx.new(|cx| {
        let mut view = TerminalView::new(
            terminal.read(cx).inner().clone(),
            workspace,
            None,
            project,
            window,
            cx,
        );
        view.set_embedded_mode(Some(1000), cx);
        view
    })
}

fn create_editor_diff(
    diff: Entity<acp_thread::Diff>,
    window: &mut Window,
    cx: &mut App,
) -> Entity<Editor> {
    cx.new(|cx| {
        let mut editor = Editor::new(
            EditorMode::Full {
                scale_ui_elements_with_buffer_font_size: false,
                show_active_line_background: false,
                sizing_behavior: SizingBehavior::SizeByContent,
            },
            diff.read(cx).multibuffer().clone(),
            None,
            window,
            cx,
        );
        editor.set_show_gutter(false, cx);
        editor.disable_diagnostics(cx);
        editor.set_max_diagnostics_severity(DiagnosticSeverity::Off, cx);
        editor.disable_expand_excerpt_buttons(cx);
        editor.set_show_vertical_scrollbar(false, cx);
        editor.set_minimap_visibility(MinimapVisibility::Disabled, window, cx);
        editor.set_soft_wrap_mode(SoftWrap::None, cx);
        editor.set_forbid_vertical_scroll(true);
        editor.set_show_indent_guides(false, cx);
        editor.set_read_only(true);
        editor.set_delegate_open_excerpts(true);
        editor.set_show_bookmarks(false, cx);
        editor.set_show_breakpoints(false, cx);
        editor.set_show_code_actions(false, cx);
        editor.set_show_git_diff_gutter(false, cx);
        editor.set_expand_all_diff_hunks(cx);
        editor.set_diff_hunk_delegate(Some(Arc::new(RestoreOnlyUnstagedDiffHunkDelegate)), cx);
        editor.set_text_style_refinement(diff_editor_text_style_refinement(cx));
        editor
    })
}

fn diff_editor_text_style_refinement(cx: &mut App) -> TextStyleRefinement {
    let settings = ThemeSettings::get_global(cx);
    TextStyleRefinement {
        font_family: Some(settings.agent_buffer_font_family().clone()),
        font_size: Some(settings.agent_buffer_font_size(cx).into()),
        font_weight: Some(settings.buffer_font.weight),
        font_features: Some(settings.buffer_font.features.clone()),
        font_fallbacks: settings.buffer_font.fallbacks.clone(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::rc::Rc;
    use std::sync::Arc;

    use acp_thread::{AgentConnection, StubAgentConnection};
    use agent_client_protocol::schema::v1 as acp;
    use buffer_diff::{DiffHunkStatus, DiffHunkStatusKind};
    use editor::RowInfo;
    use fs::FakeFs;
    use gpui::{AppContext as _, TestAppContext, UpdateGlobal as _};
    use parking_lot::RwLock;

    use crate::entry_view_state::{Entry, EntryViewState};
    use crate::message_editor::SessionCapabilities;
    use multi_buffer::MultiBufferRow;
    use pretty_assertions::assert_matches;
    use project::Project;
    use serde_json::json;
    use settings::SettingsStore;
    use util::path;
    use workspace::{MultiWorkspace, PathList};

    #[test]
    fn test_reindex_after_removal() {
        use super::reindex_after_removal;

        // Entries before the removed range keep their index.
        assert_eq!(reindex_after_removal(0, &(2..4)), Some(0));
        assert_eq!(reindex_after_removal(1, &(2..4)), Some(1));
        // Entries inside the removed range are dropped.
        assert_eq!(reindex_after_removal(2, &(2..4)), None);
        assert_eq!(reindex_after_removal(3, &(2..4)), None);
        // Entries after the removed range slide down by its length.
        assert_eq!(reindex_after_removal(4, &(2..4)), Some(2));
        assert_eq!(reindex_after_removal(5, &(2..4)), Some(3));
        // An empty removal range leaves indices untouched.
        assert_eq!(reindex_after_removal(3, &(2..2)), Some(3));
    }

    async fn setup_user_message_entry(
        cx: &mut TestAppContext,
    ) -> (
        gpui::Entity<EntryViewState>,
        gpui::Entity<crate::message_editor::MessageEditor>,
        gpui::Entity<acp_thread::AcpThread>,
        &mut gpui::VisualTestContext,
    ) {
        crate::conversation_view::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file.txt": ""})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        let connection = Rc::new(StubAgentConnection::new());
        let thread = cx
            .update(|_, cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new(path!("/project"))]),
                    cx,
                )
            })
            .await
            .expect("canonical test session");
        thread.update(cx, |thread, cx| {
            thread
                .handle_session_update(
                    acp::SessionUpdate::UserMessageChunk(
                        acp::ContentChunk::new("AB".into()).message_id("message"),
                    ),
                    cx,
                )
                .expect("canonical user entry");
        });
        let view_state = cx.new(|_| {
            EntryViewState::new(
                workspace.downgrade(),
                project.downgrade(),
                None,
                Arc::new(RwLock::new(SessionCapabilities::default())),
                "Test Agent".into(),
            )
        });
        view_state.update_in(cx, |state, window, cx| {
            state.sync_entry(0, &thread, window, cx)
        });
        let message_editor = view_state.read_with(cx, |state, _| {
            state
                .entry(0)
                .and_then(Entry::message_editor)
                .expect("projected user editor")
                .clone()
        });
        let editor = message_editor.read_with(cx, |editor, _| editor.editor().clone());
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(Box::new(editor), true, true, None, window, cx);
            });
        });
        cx.run_until_parked();
        (view_state, message_editor, thread, cx)
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_user_message_sync_recovers_legacy_submitted_terminal_without_losing_source(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::{
            InlineTerminalContext, InlineTerminalDraftPosition, InlineTerminalPromptReference,
            TerminalThreadMetadata, TerminalThreadMetadataStore,
        };
        use crate::thread_metadata_store::{ThreadId, WorktreePaths};
        use settings::Settings as _;

        let (view_state, message_editor, thread, cx) = setup_user_message_entry(cx).await;
        let terminal_view = cx.update(|window, cx| {
            let state = view_state.read(cx);
            let workspace = state.workspace.upgrade().expect("workspace");
            let project = state.project.upgrade().expect("project");
            let settings = terminal::terminal_settings::TerminalSettings::get_global(cx);
            let builder = terminal::TerminalBuilder::new_display_only(
                settings.cursor_shape,
                settings.alternate_scroll,
                settings.max_scroll_history_lines,
                window.window_handle().window_id().as_u64(),
                cx.background_executor(),
                project.read(cx).path_style(cx),
            );
            let terminal = cx.new(|cx| builder.subscribe(cx));
            terminal.update(cx, |terminal, cx| {
                terminal.write_output(b"\x1b[31maccepted history\x1b[0m\r\n", cx)
            });
            cx.new(|cx| {
                terminal_view::TerminalView::new(
                    terminal,
                    workspace.downgrade(),
                    None,
                    project.downgrade(),
                    window,
                    cx,
                )
            })
        });
        cx.run_until_parked();
        let terminal = terminal_view.read_with(cx, |view, _| view.terminal().clone());
        let (output, snapshot) = terminal.read_with(cx, |terminal, _| {
            (
                terminal.get_content(),
                terminal.snapshot().expect("styled output"),
            )
        });
        let terminal_id = crate::TerminalId::new();
        let thread_id = ThreadId::new();
        let session_id = thread.read_with(cx, |thread, _| thread.session_id().clone());
        let client_id = acp_thread::ClientUserMessageId::new();
        let position = InlineTerminalDraftPosition::new("ABλ", 2).expect("source boundary");
        let reference = InlineTerminalPromptReference {
            occurrence_id: None,
            terminal_id,
            thread_id,
            session_id: session_id.clone(),
            position: position.clone(),
        };
        let mut resource = reference
            .content(output.clone())
            .expect("terminal resource");
        let acp::ContentBlock::Resource(embedded) = &mut resource else {
            panic!("terminal resource must be embedded");
        };
        embedded.meta = None;
        let store = cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            TerminalThreadMetadataStore::global(cx)
        });
        store.read_with(cx, |store, _| store.reload_task()).await;
        store
            .update(cx, |store, cx| {
                store.save_inline(
                    TerminalThreadMetadata {
                        terminal_id,
                        title: "Accepted fixture terminal".into(),
                        custom_title: None,
                        created_at: chrono::Utc::now(),
                        worktree_paths: WorktreePaths::from_folder_paths(&PathList::new(&[
                            Path::new(path!("/project")),
                        ])),
                        remote_connection: None,
                        working_directory: None,
                        inline_context: Some(InlineTerminalContext {
                            occurrence_id: None,
                            previous_occurrences: Vec::new(),
                            thread_id,
                            session_id: Some(session_id.clone()),
                            restored_draft_source_session_id: None,
                            preceding_entry_count: 0,
                            message_client_id: Some(client_id.clone()),
                            is_draft: false,
                            presentation_order: Some(0),
                            draft_position: Some(position.clone()),
                            collapsed: false,
                            output,
                            snapshot: Some(snapshot),
                        }),
                    },
                    cx,
                )
            })
            .expect("canonical accepted history");
        store
            .update(cx, |store, cx| store.flush(cx))
            .await
            .expect("durable history");
        store.update(cx, |store, cx| store.reload_for_test(cx));
        store.read_with(cx, |store, _| store.reload_task()).await;
        thread.update(cx, |thread, cx| {
            thread.push_user_content_block(Some(client_id.clone()), resource.clone(), cx);
            thread.push_user_content_block(Some(client_id), "λ".into(), cx);
        });
        view_state.update_in(cx, |state, window, cx| {
            state.set_inline_terminal_scope(thread_id, session_id, cx);
            window.blur();
            state.sync_entry(0, &thread, window, cx);
        });
        assert_eq!(
            message_editor.read_with(cx, |editor, cx| editor.text(cx)),
            "ABλ"
        );
        let block_id = message_editor.update(cx, |editor, cx| {
            editor
                .mount_inline_terminal(terminal_view, &position, cx)
                .expect("inline block")
        });
        let revision = message_editor.read_with(cx, |editor, cx| editor.input_revision(cx));
        view_state.update_in(cx, |state, window, cx| {
            state.sync_entry(0, &thread, window, cx)
        });
        message_editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.text(cx), "ABλ");
            assert_eq!(editor.input_revision(cx), revision);
            assert_eq!(
                editor
                    .inline_terminal_position(block_id, cx)
                    .expect("retained inline block")
                    .byte_offset_for_source("ABλ")
                    .expect("verified source"),
                2
            );
        });
        thread.update(cx, |thread, cx| {
            thread.push_user_content_block(None, "changed".into(), cx)
        });
        thread.read_with(cx, |thread, _| {
            let Some(acp_thread::AgentThreadEntry::UserMessage(message)) = thread.entries().first()
            else {
                panic!("canonical user message");
            };
            assert_eq!(
                message.chunks.last(),
                Some(&acp::ContentBlock::from("changed"))
            );
        });
        view_state.update_in(cx, |state, window, cx| {
            window.blur();
            assert!(!gpui::Focusable::focus_handle(message_editor.read(cx), cx).is_focused(window));
            state.sync_entry(0, &thread, window, cx)
        });
        assert!(
            message_editor
                .read_with(cx, |editor, cx| editor.text(cx))
                .contains("zed:///agent/terminal-selection"),
            "changed source must retain context instead of accepting a fabricated position"
        );
        thread.read_with(cx, |thread, _| {
            let acp_thread::AgentThreadEntry::UserMessage(message) = &thread.entries()[0] else {
                panic!("accepted user message");
            };
            assert_eq!(
                message.chunks[1], resource,
                "projection must not rewrite stored content"
            );
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_user_message_sync_preserves_inline_terminal_on_unchanged_projection(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::InlineTerminalDraftPosition;
        use settings::Settings as _;
        let (view_state, message_editor, thread, cx) = setup_user_message_entry(cx).await;
        let terminal_view = cx.update(|window, cx| {
            let state = view_state.read(cx);
            let workspace = state.workspace.upgrade().expect("workspace");
            let project = state.project.upgrade().expect("project");
            let settings = terminal::terminal_settings::TerminalSettings::get_global(cx);
            let builder = terminal::TerminalBuilder::new_display_only(
                settings.cursor_shape,
                settings.alternate_scroll,
                settings.max_scroll_history_lines,
                window.window_handle().window_id().as_u64(),
                cx.background_executor(),
                project.read(cx).path_style(cx),
            );
            let terminal = cx.new(|cx| builder.subscribe(cx));
            terminal.update(cx, |terminal, cx| {
                terminal.write_output(b"\x1b[31mretained output\x1b[0m\r\n", cx)
            });
            cx.new(|cx| {
                terminal_view::TerminalView::new(
                    terminal,
                    workspace.downgrade(),
                    None,
                    project.downgrade(),
                    window,
                    cx,
                )
            })
        });
        let block_id = message_editor.update(cx, |editor, cx| {
            editor
                .mount_inline_terminal(
                    terminal_view.clone(),
                    &InlineTerminalDraftPosition::new("AB", 1).expect("position"),
                    cx,
                )
                .expect("actual terminal view block")
        });
        let revision = message_editor.read_with(cx, |editor, cx| editor.input_revision(cx));
        for _ in 0..3 {
            view_state.update_in(cx, |state, window, cx| {
                window.blur();
                state.sync_entry(0, &thread, window, cx);
            });
            message_editor.read_with(cx, |editor, cx| {
                let position = editor
                    .inline_terminal_position(block_id, cx)
                    .expect("unchanged projection must retain the mounted terminal boundary");
                assert_eq!(position.byte_offset_for_source("AB").expect("source"), 1);
                assert!(editor.input_revision(cx) == revision);
                assert_eq!(editor.text(cx), "AB");
            });
        }
        terminal_view.read_with(cx, |view, cx| {
            assert!(!view.terminal().read(cx).is_pty());
        });
        thread.update(cx, |thread, cx| {
            thread
                .handle_session_update(
                    acp::SessionUpdate::UserMessageChunk(
                        acp::ContentChunk::new("C".into()).message_id("message"),
                    ),
                    cx,
                )
                .expect("changed canonical entry");
        });
        view_state.update_in(cx, |state, window, cx| {
            state.sync_entry(0, &thread, window, cx)
        });
        message_editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.text(cx), "ABC");
            assert!(editor.inline_terminal_position(block_id, cx).is_err());
        });
    }

    #[gpui::test]
    async fn test_user_message_sync_respects_focused_edit_and_restores_unfocused_edits(
        cx: &mut TestAppContext,
    ) {
        let (view_state, message_editor, thread, cx) = setup_user_message_entry(cx).await;
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("focused local edit", window, cx);
            gpui::Focusable::focus_handle(editor, cx).focus(window, cx);
        });
        thread.update(cx, |thread, cx| {
            thread
                .handle_session_update(
                    acp::SessionUpdate::UserMessageChunk(
                        acp::ContentChunk::new("C".into()).message_id("message"),
                    ),
                    cx,
                )
                .expect("canonical update while editing");
        });
        view_state.update_in(cx, |state, window, cx| {
            assert!(gpui::Focusable::focus_handle(message_editor.read(cx), cx).is_focused(window));
            state.sync_entry(0, &thread, window, cx)
        });
        message_editor.read_with(cx, |editor, cx| {
            assert_eq!(editor.text(cx), "focused local edit")
        });
        view_state.update_in(cx, |state, window, cx| {
            window.blur();
            state.sync_entry(0, &thread, window, cx);
        });
        message_editor.read_with(cx, |editor, cx| assert_eq!(editor.text(cx), "ABC"));
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("unfocused local edit", window, cx);
        });
        view_state.update_in(cx, |state, window, cx| {
            window.blur();
            assert!(!gpui::Focusable::focus_handle(message_editor.read(cx), cx).is_focused(window));
            state.sync_entry(0, &thread, window, cx)
        });
        message_editor.read_with(cx, |editor, cx| assert_eq!(editor.text(cx), "ABC"));
    }

    #[gpui::test]
    fn collaborative_inline_diff_uses_independent_code_font(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.theme.agent_ui_font_size = Some(24.0.into());
                    settings.theme.agent_buffer_font_size = Some(18.0.into());
                    settings.theme.agent_buffer_font_family =
                        Some("Test Code Font".to_owned().into());
                });
            });
            let style = super::diff_editor_text_style_refinement(cx);
            assert_eq!(style.font_size, Some(gpui::px(18.0).into()));
            assert_eq!(style.font_family.as_deref(), Some("Test Code Font"));
        });
    }

    #[gpui::test]
    async fn test_diff_sync(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                "hello.txt": "hi world"
            }),
        )
        .await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let tool_call = acp::ToolCall::new("tool", "Tool call")
            .status(acp::ToolCallStatus::InProgress)
            .content(vec![acp::ToolCallContent::Diff(
                acp::Diff::new("/project/hello.txt", "hello world").old_text("hi world"),
            )]);
        let connection = Rc::new(StubAgentConnection::new());
        let thread = cx
            .update(|_, cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new(path!("/project"))]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = thread.update(cx, |thread, _| thread.session_id().clone());

        cx.update(|_, cx| {
            connection.send_update(session_id, acp::SessionUpdate::ToolCall(tool_call), cx)
        });

        let thread_store = None;

        let view_state = cx.new(|_cx| {
            EntryViewState::new(
                workspace.downgrade(),
                project.downgrade(),
                thread_store,
                Arc::new(RwLock::new(SessionCapabilities::default())),
                "Test Agent".into(),
            )
        });

        view_state.update_in(cx, |view_state, window, cx| {
            view_state.sync_entry(0, &thread, window, cx)
        });

        let diff = thread.read_with(cx, |thread, _| {
            thread
                .entries()
                .get(0)
                .unwrap()
                .diffs()
                .next()
                .unwrap()
                .clone()
        });

        cx.run_until_parked();

        let diff_editor = view_state.read_with(cx, |view_state, _cx| {
            view_state.entry(0).unwrap().editor_for_diff(&diff).unwrap()
        });
        assert_eq!(
            diff_editor.read_with(cx, |editor, cx| editor.text(cx)),
            "hi world\nhello world"
        );
        let row_infos = diff_editor.read_with(cx, |editor, cx| {
            let multibuffer = editor.buffer().read(cx);
            multibuffer
                .snapshot(cx)
                .row_infos(MultiBufferRow(0))
                .collect::<Vec<_>>()
        });
        assert_matches!(
            row_infos.as_slice(),
            [
                RowInfo {
                    multibuffer_row: Some(MultiBufferRow(0)),
                    diff_status: Some(DiffHunkStatus {
                        kind: DiffHunkStatusKind::Deleted,
                        ..
                    }),
                    ..
                },
                RowInfo {
                    multibuffer_row: Some(MultiBufferRow(1)),
                    diff_status: Some(DiffHunkStatus {
                        kind: DiffHunkStatusKind::Added,
                        ..
                    }),
                    ..
                }
            ]
        );
    }

    #[gpui::test]
    async fn test_elicitation_preserves_entry_index(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let connection = Rc::new(StubAgentConnection::new());
        let thread = cx
            .update(|_, cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new(path!("/project"))]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = thread.update(cx, |thread, _| thread.session_id().clone());

        let _response_task = thread.update(cx, |thread, cx| {
            thread
                .request_elicitation(
                    acp::CreateElicitationRequest::new(
                        acp::ElicitationFormMode::new(
                            acp::ElicitationSessionScope::new(session_id.clone()),
                            acp::ElicitationSchema::new().string("name", true),
                        ),
                        "Provide a name",
                    ),
                    cx,
                )
                .unwrap()
        });
        cx.update(|_, cx| {
            connection.send_update(
                session_id,
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    acp::ContentBlock::Text(acp::TextContent::new("hello")),
                )),
                cx,
            );
        });

        let view_state = cx.new(|_cx| {
            EntryViewState::new(
                workspace.downgrade(),
                project.downgrade(),
                None,
                Arc::new(RwLock::new(SessionCapabilities::default())),
                "Test Agent".into(),
            )
        });

        view_state.update_in(cx, |view_state, window, cx| {
            view_state.sync_entry(0, &thread, window, cx);
            view_state.sync_entry(1, &thread, window, cx);
        });

        view_state.read_with(cx, |view_state, _cx| {
            assert!(matches!(
                view_state.entry(0),
                Some(Entry::Elicitation { .. })
            ));
            assert!(matches!(
                view_state.entry(1),
                Some(Entry::AssistantMessage(_))
            ));
        });
    }

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let mut settings_store = SettingsStore::test(cx);
            settings_store.register_setting::<feature_flags::FeatureFlagsSettings>();
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
    }
}
