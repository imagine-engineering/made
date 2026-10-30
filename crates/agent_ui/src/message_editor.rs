use crate::DEFAULT_THREAD_TITLE;
use crate::SendImmediately;
use crate::{
    ChatWithFollow,
    completion_provider::{
        AgentContextSelection, AvailableCommand, AvailableSkill, PromptCompletionProvider,
        PromptCompletionProviderDelegate, PromptContextAction, PromptContextType,
        PromptLocalCommand, SlashCommandCompletion,
    },
    mention_set::{Mention, MentionImage, MentionSet, insert_crease_for_mention},
};
use acp_thread::MentionUri;
use agent::ThreadStore;
use agent_client_protocol::schema::v1 as acp;
use anyhow::{Result, anyhow};
use base64::Engine as _;
use editor::{
    Addon, AnchorRangeExt, ContextMenuOptions, Editor, EditorElement, EditorEvent, EditorMode,
    EditorStyle, Inlay, MultiBuffer, MultiBufferOffset, MultiBufferSnapshot, ToOffset,
    actions::{Copy, Cut, Paste},
    code_context_menus::CodeContextMenu,
    display_map::{CreaseId, CreaseSnapshot},
    scroll::Autoscroll,
};
use futures::{FutureExt as _, future::join_all};
use gpui::{
    AppContext, ClipboardEntry, ClipboardItem, Context, Entity, EventEmitter, FocusHandle,
    Focusable, Image, ImageFormat, KeyContext, SharedString, Subscription, Task, TaskExt,
    TextStyle, WeakEntity,
};
use language::{Buffer, language_settings::InlayHintKind};
use parking_lot::RwLock;
use project::AgentId;
use project::{
    CompletionIntent, InlayHint, InlayHintLabel, InlayId, Project, ProjectPath, Worktree,
};
use rope::Point;
use settings::Settings;
use std::{cmp::min, fmt::Write, ops::Range, rc::Rc, sync::Arc};
use text::LineEnding;
use theme_settings::ThemeSettings;
use ui::{ContextMenu, prelude::*};
use util::paths::PathStyle;
use util::{ResultExt, debug_panic};
use workspace::{CollaboratorId, Workspace};
use zed_actions::agent::{Chat, PasteRaw};

#[derive(Default)]
pub struct SessionCapabilities {
    prompt_capabilities: acp::PromptCapabilities,
    available_commands: Vec<acp::AvailableCommand>,
    available_skills: Vec<AvailableSkill>,
}

impl SessionCapabilities {
    pub fn new(
        prompt_capabilities: acp::PromptCapabilities,
        available_commands: Vec<acp::AvailableCommand>,
        available_skills: Vec<AvailableSkill>,
    ) -> Self {
        Self {
            prompt_capabilities,
            available_commands,
            available_skills,
        }
    }

    pub fn from_acp_commands(
        prompt_capabilities: acp::PromptCapabilities,
        available_commands: Vec<acp::AvailableCommand>,
    ) -> Self {
        Self::new(prompt_capabilities, available_commands, Vec::new())
    }

    pub fn supports_images(&self) -> bool {
        self.prompt_capabilities.image
    }

    pub fn supports_embedded_context(&self) -> bool {
        self.prompt_capabilities.embedded_context
    }

    pub fn available_commands(&self) -> &[acp::AvailableCommand] {
        &self.available_commands
    }

    pub fn available_skills(&self) -> &[AvailableSkill] {
        &self.available_skills
    }

    pub fn has_slash_completions(&self) -> bool {
        !self.available_commands.is_empty() || !self.available_skills.is_empty()
    }

    fn supported_modes(&self, has_thread_store: bool) -> Vec<PromptContextType> {
        let mut supported = vec![PromptContextType::File, PromptContextType::Symbol];
        if self.prompt_capabilities.embedded_context {
            if has_thread_store {
                supported.push(PromptContextType::Thread);
            }
            supported.extend(&[
                PromptContextType::Diagnostics,
                PromptContextType::Fetch,
                PromptContextType::Skill,
                PromptContextType::BranchDiff,
            ]);
        }
        supported
    }

    pub fn completion_commands(&self) -> Vec<AvailableCommand> {
        self.available_commands
            .iter()
            .map(|command| AvailableCommand {
                name: command.name.clone().into(),
                description: command.description.clone().into(),
                requires_argument: command.input.is_some(),
                source: None,
                category: acp_thread::command_category_from_meta(&command.meta),
            })
            .collect()
    }

    pub fn completion_skills(&self) -> Vec<AvailableSkill> {
        self.available_skills.clone()
    }

    pub fn set_prompt_capabilities(&mut self, prompt_capabilities: acp::PromptCapabilities) {
        self.prompt_capabilities = prompt_capabilities;
    }

    pub fn set_available_commands(&mut self, available_commands: Vec<acp::AvailableCommand>) {
        self.available_commands = available_commands;
    }

    pub fn set_available_skills(&mut self, available_skills: Vec<AvailableSkill>) {
        self.available_skills = available_skills;
    }
}

pub type SharedSessionCapabilities = Arc<RwLock<SessionCapabilities>>;

pub type SharedLocalCommands = Arc<RwLock<Vec<PromptLocalCommand>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MessageEditorInputRevision {
    buffer_id: gpui::EntityId,
    edit_count: usize,
}

#[derive(Clone, Copy)]
enum ContentWhitespace {
    Preserve,
    Trim,
}

struct MessageEditorCompletionDelegate {
    session_capabilities: SharedSessionCapabilities,
    has_thread_store: bool,
    message_editor: WeakEntity<MessageEditor>,
    local_commands: SharedLocalCommands,
}

impl PromptCompletionProviderDelegate for MessageEditorCompletionDelegate {
    fn supports_images(&self, _cx: &App) -> bool {
        self.session_capabilities.read().supports_images()
    }

    fn supported_modes(&self, _cx: &App) -> Vec<PromptContextType> {
        self.session_capabilities
            .read()
            .supported_modes(self.has_thread_store)
    }

    fn available_commands(&self, _cx: &App) -> Vec<AvailableCommand> {
        self.session_capabilities.read().completion_commands()
    }

    fn available_skills(&self, _cx: &App) -> Vec<AvailableSkill> {
        self.session_capabilities.read().completion_skills()
    }

    fn available_local_commands(&self, _cx: &App) -> Vec<PromptLocalCommand> {
        self.local_commands.read().clone()
    }

    fn run_local_command(&self, command: PromptLocalCommand, cx: &mut App) {
        self.message_editor
            .update(cx, |_this, cx| {
                cx.emit(MessageEditorEvent::LocalCommandInvoked(command));
            })
            .ok();
    }

    fn slash_autocomplete_invoked(&self, cx: &mut App) {
        // This may be called synchronously from inside a `MessageEditor`
        // update (e.g. when pasting a slash command triggers completions),
        // so we defer the emit to avoid a reentrant update panic.
        let Some(editor) = self.message_editor.upgrade() else {
            return;
        };
        cx.defer(move |cx| {
            editor.update(cx, |_editor, cx| {
                cx.emit(MessageEditorEvent::SlashAutocompleteOpened);
            });
        });
    }

    fn confirm_command(&self, cx: &mut App) {
        let _ = self.message_editor.update(cx, |this, cx| this.send(cx));
    }
}

pub struct MessageEditor {
    mention_set: Entity<MentionSet>,
    editor: Entity<Editor>,
    #[cfg(feature = "multiplayer-tools")]
    inline_terminal_blocks: collections::HashMap<
        editor::display_map::CustomBlockId,
        Option<gpui::WeakEntity<crate::inline_terminal_view::InlineTerminalView>>,
    >,
    #[cfg(feature = "multiplayer-tools")]
    inline_terminal_scope: Option<(crate::thread_metadata_store::ThreadId, acp::SessionId)>,
    workspace: WeakEntity<Workspace>,
    session_capabilities: SharedSessionCapabilities,
    local_commands: SharedLocalCommands,
    agent_id: AgentId,
    thread_store: Option<Entity<ThreadStore>>,
    _subscriptions: Vec<Subscription>,
    _parse_slash_command_task: Task<()>,
}

#[derive(Clone, Debug)]
pub enum InputAttempt {
    Text(Arc<str>),
    Paste(ClipboardItem),
}

#[derive(Clone, Debug)]
pub enum MessageEditorEvent {
    Send,
    SendImmediately,
    Cancel,
    Focus,
    LostFocus,
    Edited,
    #[cfg(feature = "multiplayer-tools")]
    InlineTerminalHistoryExpanded,
    /// Emitted when the user opens slash-command autocomplete in this
    /// editor. Used by `ThreadView` to fire the global-skills scan
    /// trigger; see `NativeAgent::ensure_skills_scan_started`.
    SlashAutocompleteOpened,
    /// Emitted when the user confirms a local slash command (scrolling,
    /// exporting, feedback) in this editor's completion popup. `ThreadView`
    /// handles it by running the corresponding action; see
    /// `handle_message_editor_event`.
    LocalCommandInvoked(PromptLocalCommand),
    InputAttempted {
        attempt: InputAttempt,
        cursor_offset: usize,
    },
}

impl EventEmitter<MessageEditorEvent> for MessageEditor {}

const COMMAND_HINT_INLAY_ID: InlayId = InlayId::Hint(0);

fn insert_mention_for_project_path(
    project_path: &ProjectPath,
    editor: &Entity<Editor>,
    mention_set: &Entity<MentionSet>,
    project: &Entity<Project>,
    workspace: &Entity<Workspace>,
    supports_images: bool,
    window: &mut Window,
    cx: &mut App,
) -> Option<Task<()>> {
    let (file_name, mention_uri) = {
        let project = project.read(cx);
        let path_style = project.path_style(cx);
        let entry = project.entry_for_path(project_path, cx)?;
        let worktree = project.worktree_for_id(project_path.worktree_id, cx)?;
        let abs_path = worktree.read(cx).absolutize(&project_path.path);
        let (file_name, _) = crate::completion_provider::extract_file_name_and_directory(
            &project_path.path,
            worktree.read(cx).root_name(),
            path_style,
        );
        let mention_uri = if entry.is_dir() {
            MentionUri::Directory { abs_path }
        } else {
            MentionUri::File { abs_path }
        };
        (file_name, mention_uri)
    };

    let mention_text = mention_uri.as_link().to_string();
    let content_len = mention_text.len();

    let text_anchor = editor.update(cx, |editor, cx| {
        let buffer = editor.buffer().read(cx);
        let snapshot = buffer.snapshot(cx);
        let buffer_snapshot = snapshot.as_singleton()?;
        let text_anchor = snapshot
            .anchor_to_buffer_anchor(editor.selections.newest_anchor().start)?
            .0
            .bias_left(&buffer_snapshot);

        editor.insert(&mention_text, window, cx);
        editor.insert(" ", window, cx);

        Some(text_anchor)
    })?;

    Some(mention_set.update(cx, |mention_set, cx| {
        mention_set.confirm_mention_completion(
            file_name,
            text_anchor,
            content_len,
            mention_uri,
            supports_images,
            editor.clone(),
            workspace,
            window,
            cx,
        )
    }))
}

enum ResolvedPastedContextItem {
    Image(gpui::Image, gpui::SharedString),
    ProjectPath(ProjectPath),
}

async fn resolve_pasted_context_items(
    project: Entity<Project>,
    project_is_local: bool,
    supports_images: bool,
    entries: Vec<ClipboardEntry>,
    cx: &mut gpui::AsyncWindowContext,
) -> (Vec<ResolvedPastedContextItem>, Vec<Entity<Worktree>>) {
    let mut items = Vec::new();
    let mut added_worktrees = Vec::new();
    let default_image_name: SharedString = "Image".into();

    for entry in entries {
        match entry {
            ClipboardEntry::String(_) => {}
            ClipboardEntry::Image(image) => {
                if supports_images {
                    items.push(ResolvedPastedContextItem::Image(
                        image,
                        default_image_name.clone(),
                    ));
                }
            }
            ClipboardEntry::ExternalPaths(paths) => {
                for path in paths.paths().iter() {
                    if let Some((image, name)) = cx
                        .background_spawn({
                            let path = path.clone();
                            let default_image_name = default_image_name.clone();
                            async move {
                                crate::mention_set::load_external_image_from_path(
                                    &path,
                                    &default_image_name,
                                )
                            }
                        })
                        .await
                    {
                        if supports_images {
                            items.push(ResolvedPastedContextItem::Image(image, name));
                        }
                        continue;
                    }

                    if !project_is_local {
                        continue;
                    }

                    let path = path.clone();
                    let Ok(resolve_task) = cx.update({
                        let project = project.clone();
                        move |_, cx| Workspace::project_path_for_path(project, &path, false, cx)
                    }) else {
                        continue;
                    };

                    if let Some((worktree, project_path)) = resolve_task.await.log_err() {
                        added_worktrees.push(worktree);
                        items.push(ResolvedPastedContextItem::ProjectPath(project_path));
                    }
                }
            }
        }
    }

    (items, added_worktrees)
}

fn insert_project_path_as_context(
    project_path: ProjectPath,
    editor: Entity<Editor>,
    mention_set: Entity<MentionSet>,
    workspace: WeakEntity<Workspace>,
    supports_images: bool,
    cx: &mut gpui::AsyncWindowContext,
) -> Option<Task<()>> {
    let workspace = workspace.upgrade()?;

    cx.update(move |window, cx| {
        let project = workspace.read(cx).project().clone();
        insert_mention_for_project_path(
            &project_path,
            &editor,
            &mention_set,
            &project,
            &workspace,
            supports_images,
            window,
            cx,
        )
    })
    .ok()
    .flatten()
}

async fn insert_resolved_pasted_context_items(
    items: Vec<ResolvedPastedContextItem>,
    added_worktrees: Vec<Entity<Worktree>>,
    editor: Entity<Editor>,
    mention_set: Entity<MentionSet>,
    workspace: WeakEntity<Workspace>,
    supports_images: bool,
    cx: &mut gpui::AsyncWindowContext,
) {
    let mut path_mention_tasks = Vec::new();

    for item in items {
        match item {
            ResolvedPastedContextItem::Image(image, name) => {
                crate::mention_set::insert_images_as_context(
                    vec![(image, name)],
                    editor.clone(),
                    mention_set.clone(),
                    workspace.clone(),
                    cx,
                )
                .await;
            }
            ResolvedPastedContextItem::ProjectPath(project_path) => {
                if let Some(task) = insert_project_path_as_context(
                    project_path,
                    editor.clone(),
                    mention_set.clone(),
                    workspace.clone(),
                    supports_images,
                    cx,
                ) {
                    path_mention_tasks.push(task);
                }
            }
        }
    }

    join_all(path_mention_tasks).await;
    drop(added_worktrees);
}

impl MessageEditor {
    pub fn new(
        workspace: WeakEntity<Workspace>,
        project: WeakEntity<Project>,
        thread_store: Option<Entity<ThreadStore>>,
        session_capabilities: SharedSessionCapabilities,
        agent_id: AgentId,
        placeholder: &str,
        mode: EditorMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let language_registry = project
            .upgrade()
            .map(|project| project.read(cx).languages().clone());

        let editor = cx.new(|cx| {
            let buffer = cx.new(|cx| {
                let buffer = Buffer::local("", cx);
                if let Some(language_registry) = language_registry.as_ref() {
                    buffer.set_language_registry(language_registry.clone());
                }
                buffer
            });
            let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));

            let mut editor = Editor::new(mode, buffer, None, window, cx);
            editor.set_placeholder_text(placeholder, window, cx);
            editor.set_show_indent_guides(false, cx);
            editor.set_show_completions_on_input(Some(true));
            editor.set_soft_wrap();
            editor.disable_mouse_wheel_zoom();
            editor.set_use_modal_editing(true);
            editor.set_context_menu_options(ContextMenuOptions {
                min_entries_visible: 12,
                max_entries_visible: 12,
                placement: None,
            });
            editor.register_addon(MessageEditorAddon::new());

            editor.set_custom_context_menu(|editor, _point, window, cx| {
                let has_selection = editor.has_non_empty_selection(&editor.display_snapshot(cx));

                Some(ContextMenu::build(window, cx, |menu, _, _| {
                    menu.action("Cut", Box::new(editor::actions::Cut))
                        .action_disabled_when(
                            !has_selection,
                            "Copy",
                            Box::new(editor::actions::Copy),
                        )
                        .action("Paste", Box::new(editor::actions::Paste))
                        .action("Paste as Plain Text", Box::new(PasteRaw))
                }))
            });

            editor
        });
        let mention_set = cx.new(|_cx| MentionSet::new(project, thread_store.clone()));
        let local_commands: SharedLocalCommands = Arc::new(RwLock::new(Vec::new()));
        let completion_provider = Rc::new(PromptCompletionProvider::new(
            MessageEditorCompletionDelegate {
                session_capabilities: session_capabilities.clone(),
                has_thread_store: thread_store.is_some(),
                message_editor: cx.weak_entity(),
                local_commands: local_commands.clone(),
            },
            editor.downgrade(),
            mention_set.clone(),
            workspace.clone(),
        ));
        editor.update(cx, |editor, _cx| {
            editor.set_completion_provider(Some(completion_provider.clone()))
        });

        cx.on_focus_in(&editor.focus_handle(cx), window, |_, _, cx| {
            cx.emit(MessageEditorEvent::Focus)
        })
        .detach();
        cx.on_focus_out(&editor.focus_handle(cx), window, |_, _, _, cx| {
            cx.emit(MessageEditorEvent::LostFocus)
        })
        .detach();

        let mut has_hint = false;
        let mut subscriptions = Vec::new();

        subscriptions.push(cx.subscribe_in(&editor, window, {
            move |this, editor, event, window, cx| {
                let input_attempted_text = match event {
                    EditorEvent::InputHandled { text, .. } => Some(text),
                    EditorEvent::InputIgnored { text } => Some(text),
                    _ => None,
                };
                if let Some(text) = input_attempted_text
                    && editor.read(cx).read_only(cx)
                    && !text.is_empty()
                {
                    let editor = editor.read(cx);
                    let cursor_anchor = editor.selections.newest_anchor().head();
                    let cursor_offset = cursor_anchor
                        .to_offset(&editor.buffer().read(cx).snapshot(cx))
                        .0;
                    cx.emit(MessageEditorEvent::InputAttempted {
                        attempt: InputAttempt::Text(text.clone()),
                        cursor_offset,
                    });
                }

                if let EditorEvent::Edited { .. } = event
                    && !editor.read(cx).read_only(cx)
                {
                    cx.emit(MessageEditorEvent::Edited);
                    editor.update(cx, |editor, cx| {
                        let snapshot = editor.snapshot(window, cx);
                        this.mention_set
                            .update(cx, |mention_set, _cx| mention_set.remove_invalid(&snapshot));

                        let new_hints = this
                            .command_hint(snapshot.buffer())
                            .into_iter()
                            .collect::<Vec<_>>();
                        let has_new_hint = !new_hints.is_empty();
                        editor.splice_inlays(
                            if has_hint {
                                &[COMMAND_HINT_INLAY_ID]
                            } else {
                                &[]
                            },
                            new_hints,
                            cx,
                        );
                        has_hint = has_new_hint;
                    });
                    cx.notify();
                }
            }
        }));

        if let Some(language_registry) = language_registry {
            let editor = editor.clone();
            cx.spawn(async move |_, cx| {
                let markdown = language_registry.language_for_name("Markdown").await?;
                editor.update(cx, |editor, cx| {
                    if let Some(buffer) = editor.buffer().read(cx).as_singleton() {
                        buffer.update(cx, |buffer, cx| {
                            buffer.set_language(Some(markdown), cx);
                        });
                    }
                });
                anyhow::Ok(())
            })
            .detach_and_log_err(cx);
        }

        Self {
            editor,
            mention_set,
            #[cfg(feature = "multiplayer-tools")]
            inline_terminal_blocks: collections::HashMap::default(),
            #[cfg(feature = "multiplayer-tools")]
            inline_terminal_scope: None,
            workspace,
            session_capabilities,
            local_commands,
            agent_id,
            thread_store,
            _subscriptions: subscriptions,
            _parse_slash_command_task: Task::ready(()),
        }
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn set_inline_terminal_scope(
        &mut self,
        thread_id: crate::thread_metadata_store::ThreadId,
        session_id: acp::SessionId,
    ) {
        self.inline_terminal_scope = Some((thread_id, session_id));
    }

    #[cfg(feature = "multiplayer-tools")]
    fn admits_inline_terminal_reference(
        &self,
        reference: &crate::terminal_thread_metadata_store::InlineTerminalPromptReference,
        cx: &App,
    ) -> bool {
        let Some((thread_id, session_id)) = self.inline_terminal_scope.as_ref() else {
            return false;
        };
        if thread_id != &reference.thread_id || session_id != &reference.session_id {
            return false;
        }
        let Some(workspace) = self.workspace.upgrade() else {
            return false;
        };
        let remote = workspace
            .read(cx)
            .project()
            .read(cx)
            .remote_connection_options(cx);
        crate::terminal_thread_metadata_store::TerminalThreadMetadataStore::try_global(cx)
            .is_some_and(|store| {
                store
                    .read(cx)
                    .entry(reference.terminal_id)
                    .is_some_and(|metadata| {
                        reference.matches_metadata(metadata)
                            && remote::same_remote_connection_identity(
                                metadata.remote_connection.as_ref(),
                                remote.as_ref(),
                            )
                    })
            })
    }

    pub fn set_local_commands(&self, commands: Vec<PromptLocalCommand>) {
        *self.local_commands.write() = commands;
    }

    pub fn set_session_capabilities(
        &mut self,
        session_capabilities: SharedSessionCapabilities,
        _cx: &mut Context<Self>,
    ) {
        self.session_capabilities = session_capabilities;
    }

    fn command_hint(&self, snapshot: &MultiBufferSnapshot) -> Option<Inlay> {
        let session_capabilities = self.session_capabilities.read();
        let available_commands = session_capabilities.available_commands();
        if available_commands.is_empty() {
            return None;
        }

        let parsed_command = SlashCommandCompletion::try_parse(&snapshot.text(), 0)?;
        if parsed_command.argument.is_some() {
            return None;
        }

        let command_name = parsed_command.command?;
        let available_command = available_commands
            .iter()
            .find(|available_command| available_command.name == command_name)?;

        let acp::AvailableCommandInput::Unstructured(acp::UnstructuredCommandInput {
            mut hint,
            ..
        }) = available_command.input.clone()?
        else {
            return None;
        };

        let mut hint_pos = MultiBufferOffset(parsed_command.source_range.end) + 1usize;
        if hint_pos > snapshot.len() {
            hint_pos = snapshot.len();
            hint.insert(0, ' ');
        }

        let hint_pos = snapshot.anchor_after(hint_pos);

        Some(Inlay::hint(
            COMMAND_HINT_INLAY_ID,
            hint_pos,
            &InlayHint {
                position: snapshot.anchor_to_buffer_anchor(hint_pos)?.0,
                label: InlayHintLabel::String(hint),
                kind: Some(InlayHintKind::Parameter),
                padding_left: false,
                padding_right: false,
                tooltip: None,
                resolve_state: project::ResolveState::Resolved,
            },
        ))
    }

    pub fn insert_thread_summary(
        &mut self,
        session_id: acp::SessionId,
        title: Option<SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.thread_store.is_none() {
            return;
        }
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };
        let thread_title = title
            .filter(|title| !title.is_empty())
            .unwrap_or_else(|| SharedString::new_static(DEFAULT_THREAD_TITLE));
        let uri = MentionUri::Thread {
            id: session_id,
            name: thread_title.to_string(),
        };
        let content = format!("{}\n", uri.as_link());

        let content_len = content.len() - 1;

        let start = self.editor.update(cx, |editor, cx| {
            editor.set_text(content, window, cx);
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            snapshot
                .anchor_to_buffer_anchor(snapshot.anchor_before(Point::zero()))
                .unwrap()
                .0
        });

        let supports_images = self.session_capabilities.read().supports_images();

        self.mention_set
            .update(cx, |mention_set, cx| {
                mention_set.confirm_mention_completion(
                    thread_title,
                    start,
                    content_len,
                    uri,
                    supports_images,
                    self.editor.clone(),
                    &workspace,
                    window,
                    cx,
                )
            })
            .detach();
    }

    pub(crate) fn editor(&self) -> &Entity<Editor> {
        &self.editor
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn inline_terminal_insertion_position(
        &self,
        cx: &App,
    ) -> Result<crate::terminal_thread_metadata_store::InlineTerminalDraftPosition> {
        let editor = self.editor.read(cx);
        let snapshot = editor.buffer().read(cx).snapshot(cx);
        let offset = editor
            .selections
            .newest_anchor()
            .head()
            .to_offset(&snapshot)
            .0;
        crate::terminal_thread_metadata_store::InlineTerminalDraftPosition::new(
            &self.text(cx),
            offset,
        )
    }

    #[cfg(feature = "multiplayer-tools")]
    pub fn mount_inline_terminal(
        &mut self,
        terminal_view: Entity<terminal_view::TerminalView>,
        position: &crate::terminal_thread_metadata_store::InlineTerminalDraftPosition,
        cx: &mut Context<Self>,
    ) -> Result<editor::display_map::CustomBlockId> {
        self.mount_inline_terminal_view(terminal_view, None, position, cx)
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn mount_inline_terminal_view(
        &mut self,
        terminal_view: Entity<terminal_view::TerminalView>,
        inline_view: Option<Entity<crate::inline_terminal_view::InlineTerminalView>>,
        position: &crate::terminal_thread_metadata_store::InlineTerminalDraftPosition,
        cx: &mut Context<Self>,
    ) -> Result<editor::display_map::CustomBlockId> {
        use editor::display_map::{BlockId, BlockPlacement, BlockProperties, BlockStyle};

        let offset = position.byte_offset_for_source(&self.text(cx))?;
        let editor_handle = self.editor.downgrade();
        let boundary = self
            .editor
            .read(cx)
            .buffer()
            .read(cx)
            .snapshot(cx)
            .anchor_after(MultiBufferOffset(offset));
        let inline_view_handle = inline_view.as_ref().map(Entity::downgrade);
        let block_id = self.editor.update(cx, |editor, cx| {
            editor.insert_character_block(
                BlockProperties {
                    placement: BlockPlacement::BelowWrappedLine(boundary),
                    height: Some(1),
                    style: BlockStyle::Flex,
                    priority: 0,
                    render: Arc::new({
                        let terminal_view = terminal_view.clone();
                        move |block| {
                            let terminal_view = terminal_view.clone();
                            let terminal_element = if let Some(inline_view) = inline_view.clone() {
                                inline_view.into_any_element()
                            } else if terminal_view
                                .read(block.app)
                                .content_mode(block.window, block.app)
                                .is_scrollable()
                            {
                                div().h_72().child(terminal_view).into_any_element()
                            } else {
                                terminal_view.into_any_element()
                            };
                            let block_id = match block.block_id {
                                BlockId::Custom(block_id) => block_id,
                                _ => return div().into_any_element(),
                            };
                            let editor_for_cancel = editor_handle.clone();
                            let editor_for_keystroke = editor_handle.clone();
                            div()
                                .w_full()
                                .capture_action(
                                    move |action: &terminal_view::SendKeystroke, window, cx| {
                                        if action.keystroke() != "escape" {
                                            cx.propagate();
                                            return;
                                        }
                                        editor_for_keystroke
                                            .update(cx, |editor, cx| {
                                                editor.focus_after_character_block(
                                                    block_id, window, cx,
                                                )
                                            })
                                            .and_then(|result| result)
                                            .log_err();
                                        cx.stop_propagation();
                                    },
                                )
                                .on_action({
                                    let editor_handle = editor_handle.clone();
                                    move |_: &menu::Cancel, window, cx| {
                                        editor_handle
                                            .update(cx, |editor, cx| {
                                                editor.focus_after_character_block(
                                                    block_id, window, cx,
                                                )
                                            })
                                            .and_then(|result| result)
                                            .log_err();
                                        cx.stop_propagation();
                                    }
                                })
                                .on_action(move |_: &editor::actions::Cancel, window, cx| {
                                    editor_for_cancel
                                        .update(cx, |editor, cx| {
                                            editor.focus_after_character_block(block_id, window, cx)
                                        })
                                        .and_then(|result| result)
                                        .log_err();
                                    cx.stop_propagation();
                                })
                                .child(terminal_element)
                                .into_any_element()
                        }
                    }),
                },
                None,
                cx,
            )
        })?;
        self.inline_terminal_blocks
            .insert(block_id, inline_view_handle);
        terminal_view.update(cx, |view, cx| {
            view.set_embedded_mode(Some(1000), cx);
            view.set_show_workspace_actions(false, cx);
        });
        cx.notify();
        Ok(block_id)
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn remove_inline_terminal(
        &mut self,
        block_id: editor::display_map::CustomBlockId,
        cx: &mut Context<Self>,
    ) {
        self.inline_terminal_blocks.remove(&block_id);
        self.editor.update(cx, |editor, cx| {
            editor.remove_blocks([block_id].into_iter().collect(), None, cx)
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn focus_after_inline_terminal(
        &mut self,
        block_id: editor::display_map::CustomBlockId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.editor.update(cx, |editor, cx| {
            editor.focus_after_character_block(block_id, window, cx)
        })
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn focus_inline_terminal(
        &mut self,
        block_id: editor::display_map::CustomBlockId,
        focus_handle: &FocusHandle,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.inline_terminal_blocks
                .get(&block_id)
                .and_then(Option::as_ref)
                .and_then(WeakEntity::upgrade)
                .is_some(),
            "The inline terminal's mounted occurrence is unavailable"
        );
        self.inline_terminal_position(block_id, cx)?;
        self.editor.update(cx, |editor, cx| {
            editor.focus_character_block(block_id, focus_handle, window, cx)
        })
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn scroll_to_inline_terminal(
        &mut self,
        block_id: editor::display_map::CustomBlockId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.editor.update(cx, |editor, cx| {
            let row = editor
                .row_for_block(block_id, cx)
                .ok_or_else(|| anyhow!("The inline terminal's display position is unavailable"))?;
            let position = editor.scroll_position(cx);
            editor.set_scroll_position(gpui::Point::new(position.x, f64::from(row.0)), window, cx);
            Ok(())
        })
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn inline_terminal_positions_in_display_order(
        &self,
        cx: &mut Context<Self>,
    ) -> Result<
        Vec<(
            editor::display_map::CustomBlockId,
            crate::terminal_thread_metadata_store::InlineTerminalDraftPosition,
        )>,
    > {
        let mut positions = self
            .inline_terminal_blocks
            .keys()
            .map(|block_id| Ok((*block_id, self.inline_terminal_position(*block_id, cx)?)))
            .collect::<Result<Vec<_>>>()?;
        let rows = self.editor.update(cx, |editor, cx| {
            positions
                .iter()
                .map(|(block_id, _)| {
                    editor
                        .row_for_block(*block_id, cx)
                        .map(|row| (*block_id, row.0))
                        .ok_or_else(|| {
                            anyhow!("The inline terminal's display position is unavailable")
                        })
                })
                .collect::<Result<collections::HashMap<_, _>>>()
        })?;
        positions.sort_by_key(|(block_id, _)| rows.get(block_id).copied());
        Ok(positions)
    }

    #[cfg(feature = "multiplayer-tools")]
    pub fn inline_terminal_position(
        &self,
        block_id: editor::display_map::CustomBlockId,
        cx: &App,
    ) -> Result<crate::terminal_thread_metadata_store::InlineTerminalDraftPosition> {
        let editor = self.editor.read(cx);
        let anchor = editor
            .character_block_position(block_id, cx)
            .ok_or_else(|| anyhow!("The inline terminal's character boundary is unavailable"))?;
        let snapshot = editor.buffer().read(cx).snapshot(cx);
        crate::terminal_thread_metadata_store::InlineTerminalDraftPosition::new(
            &editor.text(cx),
            anchor.to_offset(&snapshot).0,
        )
    }

    pub fn is_empty(&self, cx: &App) -> bool {
        #[cfg(feature = "multiplayer-tools")]
        if self.inline_terminal_blocks.values().any(Option::is_some) {
            return false;
        }
        self.editor.read(cx).text(cx).trim().is_empty()
    }

    pub fn is_completions_menu_visible(&self, cx: &App) -> bool {
        self.editor
            .read(cx)
            .context_menu()
            .borrow()
            .as_ref()
            .is_some_and(|menu| matches!(menu, CodeContextMenu::Completions(_)) && menu.visible())
    }

    #[cfg(test)]
    pub fn mention_set(&self) -> &Entity<MentionSet> {
        &self.mention_set
    }

    fn validate_slash_commands(
        text: &str,
        available_commands: &[acp::AvailableCommand],
        available_skills: &[AvailableSkill],
        agent_id: &AgentId,
    ) -> Result<()> {
        if let Some(parsed_command) = SlashCommandCompletion::try_parse(text, 0) {
            if parsed_command.source_range.start != 0 {
                return Ok(());
            }
            if let Some(command_name) = parsed_command.command {
                // Two acceptance paths:
                //
                // 1. Direct name match. Covers bare slash commands
                //    (`/help`), MCP prompts that were prefixed at the
                //    agent because of a server-name collision
                //    (`/github.create_pr`), and skills (whose bare name
                //    is registered for the unqualified `/<name>` form).
                //
                // 2. Trusted native skill scope qualifier `/<scope>:<name>`. The popup
                //    inserts this colon-separated form to disambiguate
                //    same-named skills, so the validator splits on the
                //    LAST `:` to recover scope + bare name. Skill
                //    names are restricted to `[a-z0-9-]+` (no colons),
                //    so the rightmost colon is always the scope/name
                //    boundary — this lets scope labels (e.g. worktree
                //    root names) themselves contain colons. The
                //    scope is allowed to be empty: `/:<name>` is the
                //    qualified form for a global skill (see
                //    `SkillSource::scope_prefix`). The validator then
                //    checks the `available_skills` slice for an entry
                //    whose `skill.name` matches the bare name and
                //    whose `skill.source` equals the typed scope
                //    (including empty for globals). Without this
                //    branch, every autocomplete pick of a same-named
                //    skill would be rejected as "not supported"
                //    before reaching the resolver.
                let direct_match = available_commands
                    .iter()
                    .any(|available_command| available_command.name == command_name)
                    || available_skills
                        .iter()
                        .any(|skill| skill.name.as_ref() == command_name);
                let scope_match = !direct_match
                    && command_name.rsplit_once(':').is_some_and(|(scope, bare)| {
                        !bare.is_empty()
                            && available_skills.iter().any(|skill| {
                                skill.name.as_ref() == bare && skill.source.as_ref() == scope
                            })
                    });

                if !direct_match && !scope_match {
                    return Err(anyhow!(indoc::formatdoc!(
                        "/{command_name} is not a recognized command in {agent_id}. \
                         Messages that start with `/` are interpreted as commands.

                         If you are trying to send a message and not run a command, \
                         try preceding the `/` with a space.

                         Available commands for {agent_id}: {commands}",
                        commands =
                            Self::format_available_commands(available_commands, available_skills),
                    )));
                }
            }
        }
        Ok(())
    }

    /// Render the available-commands list for error messages. Trusted native skills
    /// are shown in their qualified `/<scope>:<name>` form so users
    /// see the exact text the popup would insert — otherwise the
    /// listing would contain confusing duplicates like `/foo, /foo`
    /// when both a global and a project-local skill share a name.
    /// Globals carry an empty scope and so render as `/:<name>`.
    fn format_available_commands(
        commands: &[acp::AvailableCommand],
        skills: &[AvailableSkill],
    ) -> String {
        if commands.is_empty() && skills.is_empty() {
            return "none".to_string();
        }
        skills
            .iter()
            .map(|skill| format!("/{}:{}", skill.source, skill.name))
            .chain(commands.iter().map(|command| format!("/{}", command.name)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn contents(
        &self,
        full_mention_content: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<(Vec<acp::ContentBlock>, Vec<Entity<Buffer>>)>> {
        let text = self.editor.read(cx).text(cx);
        let (available_commands, available_skills) = {
            let session_capabilities = self.session_capabilities.read();
            (
                session_capabilities.available_commands().to_vec(),
                session_capabilities.available_skills().to_vec(),
            )
        };
        let agent_id = self.agent_id.clone();
        let build_task =
            self.build_content_blocks(full_mention_content, ContentWhitespace::Trim, cx);

        cx.spawn(async move |_, _cx| {
            Self::validate_slash_commands(
                &text,
                &available_commands,
                &available_skills,
                &agent_id,
            )?;
            build_task.await
        })
    }

    pub fn draft_contents(&self, cx: &mut Context<Self>) -> Task<Result<Vec<acp::ContentBlock>>> {
        let build_task = self.build_content_blocks(false, ContentWhitespace::Preserve, cx);
        cx.spawn(async move |_, _cx| {
            let (blocks, _tracked_buffers) = build_task.await?;
            Ok(blocks)
        })
    }

    fn build_content_blocks(
        &self,
        full_mention_content: bool,
        whitespace: ContentWhitespace,
        cx: &mut Context<Self>,
    ) -> Task<Result<(Vec<acp::ContentBlock>, Vec<Entity<Buffer>>)>> {
        #[cfg(feature = "multiplayer-tools")]
        let inline_positions = self.inline_terminal_positions_in_display_order(cx);
        let contents = self
            .mention_set
            .update(cx, |store, cx| store.contents(full_mention_content, cx));
        let editor = self.editor.read(cx);
        let crease_snapshot = editor.display_map.read(cx).crease_snapshot();
        let buffer_snapshot = editor.buffer().read(cx).snapshot(cx);
        let text = editor.text(cx);
        let supports_embedded_context =
            self.session_capabilities.read().supports_embedded_context();

        #[cfg(feature = "multiplayer-tools")]
        let inline_contents = (|| -> Result<Vec<(usize, acp::ContentBlock)>> {
            let mut insertions = Vec::new();
            let mut history_owner = None;
            let mut updates = Vec::new();
            for (order, (block_id, position)) in inline_positions?.into_iter().enumerate() {
                let Some(Some(view)) = self.inline_terminal_blocks.get(&block_id) else {
                    continue;
                };
                let offset = position.byte_offset_for_source(&text)?;
                anyhow::ensure!(
                    !crease_snapshot.creases().any(|(_, crease)| {
                        let range = crease.range().to_offset(&buffer_snapshot);
                        range.start.0 < offset && offset < range.end.0
                    }),
                    "An inline terminal cannot be submitted from inside a context mention"
                );
                let (content, history, mut metadata) =
                    view.read_with(cx, |view, cx| view.prompt_content(position, cx))??;
                self.validate_inline_terminal_content(&content, cx)?;
                if let Some(context) = metadata.inline_context.as_mut() {
                    context.presentation_order = Some(order);
                }
                if let Some(owner) = &history_owner {
                    anyhow::ensure!(
                        owner == &history,
                        "The terminal history owner changed during composition"
                    );
                } else {
                    history_owner = Some(history);
                }
                updates.push(metadata);
                insertions.push((offset, content));
            }
            if let Some(history) = history_owner {
                history.update(cx, |history, cx| {
                    history.save_inline_occurrence_batch(updates, cx)
                })?;
            }
            insertions.sort_by_key(|(offset, _)| *offset);
            Ok(insertions)
        })();
        #[cfg(not(feature = "multiplayer-tools"))]
        let inline_contents: Result<Vec<(usize, acp::ContentBlock)>> = Ok(Vec::new());

        cx.spawn(async move |_, _cx| {
            let mut contents = contents.await?;
            let inline_contents = inline_contents?;
            anyhow::ensure!(
                inline_contents.is_empty()
                    || supports_embedded_context
                    || matches!(whitespace, ContentWhitespace::Preserve),
                "The selected agent does not support inline terminal context"
            );
            Ok(build_chunks_with_inline_context(
                &text,
                &crease_snapshot,
                &buffer_snapshot,
                supports_embedded_context,
                whitespace,
                inline_contents,
                |crease_id| {
                    contents
                        .remove(crease_id)
                        .map(|(uri, mention)| (uri, Some(mention)))
                },
            ))
        })
    }

    /// Snapshots the editor's current draft into a list of `ContentBlock`s
    /// without awaiting any pending mention resolution.
    pub fn draft_content_blocks_snapshot(&self, cx: &App) -> Result<Vec<acp::ContentBlock>> {
        let editor = self.editor.read(cx);
        let crease_snapshot = editor.display_map.read(cx).crease_snapshot();
        let buffer_snapshot = editor.buffer().read(cx).snapshot(cx);
        let text = editor.text(cx);
        let mention_set = self.mention_set.read(cx);
        let supports_embedded_context =
            self.session_capabilities.read().supports_embedded_context();
        #[cfg(feature = "multiplayer-tools")]
        let inline_contents = self.inline_content_snapshot(cx)?;
        #[cfg(not(feature = "multiplayer-tools"))]
        let inline_contents = Vec::new();
        let (chunks, _tracked_buffers) = build_chunks_with_inline_context(
            &text,
            &crease_snapshot,
            &buffer_snapshot,
            supports_embedded_context,
            ContentWhitespace::Preserve,
            inline_contents,
            |crease_id| mention_set.resolved_mention_for_crease(crease_id),
        );
        Ok(chunks)
    }

    #[cfg(feature = "multiplayer-tools")]
    fn validate_inline_terminal_content(
        &self,
        content: &acp::ContentBlock,
        cx: &App,
    ) -> Result<()> {
        let reference =
            crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(
                content,
            )?
            .ok_or_else(|| anyhow!("The inline terminal context is missing"))?;
        anyhow::ensure!(
            self.admits_inline_terminal_reference(&reference, cx),
            "The inline terminal belongs to a different conversation"
        );
        Ok(())
    }

    #[cfg(feature = "multiplayer-tools")]
    fn inline_content_snapshot(&self, cx: &App) -> Result<Vec<(usize, acp::ContentBlock)>> {
        let mut insertions = Vec::new();
        let mut blocks = self.inline_terminal_blocks.iter().collect::<Vec<_>>();
        blocks.sort_by_key(|(block_id, _)| block_id.0);
        for (block_id, view) in blocks {
            let Some(view) = view else { continue };
            let position = self.inline_terminal_position(*block_id, cx)?;
            let offset = position.byte_offset_for_source(&self.text(cx))?;
            let (content, _, metadata) =
                view.read_with(cx, |view, cx| view.prompt_content(position, cx))??;
            self.validate_inline_terminal_content(&content, cx)?;
            let order = metadata
                .inline_context
                .as_ref()
                .and_then(|context| context.presentation_order);
            insertions.push((offset, order, content));
        }
        insertions.sort_by_key(|(offset, order, _)| (*offset, *order));
        Ok(insertions
            .into_iter()
            .map(|(offset, _, content)| (offset, content))
            .collect())
    }

    pub(crate) fn input_revision(&self, cx: &App) -> MessageEditorInputRevision {
        let buffer = self.editor.read(cx).buffer();
        MessageEditorInputRevision {
            buffer_id: buffer.entity_id(),
            edit_count: buffer.read(cx).snapshot(cx).edit_count(),
        }
    }

    pub(crate) fn clear_if_unchanged(
        &mut self,
        input_revision: MessageEditorInputRevision,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        // Edit counters distinguish a later draft even after Undo restores identical text.
        if self.input_revision(cx) != input_revision {
            return false;
        }
        self.clear(window, cx);
        true
    }

    pub fn clear(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        #[cfg(feature = "multiplayer-tools")]
        let inline_terminal_blocks = std::mem::take(&mut self.inline_terminal_blocks);
        self.editor.update(cx, |editor, cx| {
            #[cfg(feature = "multiplayer-tools")]
            editor.remove_blocks(inline_terminal_blocks.into_keys().collect(), None, cx);
            editor.clear(window, cx);
            editor.remove_creases(
                self.mention_set.update(cx, |mention_set, _cx| {
                    mention_set
                        .clear()
                        .map(|(crease_id, _)| crease_id)
                        .collect::<Vec<_>>()
                }),
                cx,
            )
        });
    }

    pub fn send(&mut self, cx: &mut Context<Self>) {
        if !self.is_empty(cx) {
            self.editor.update(cx, |editor, cx| {
                editor.clear_inlay_hints(cx);
            });
        }
        cx.emit(MessageEditorEvent::Send)
    }

    pub fn trigger_completion_menu(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.insert_context_prefix("@", window, cx);
    }

    pub fn insert_context_type(
        &mut self,
        context_keyword: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let prefix = format!("@{}", context_keyword);
        self.insert_context_prefix(&prefix, window, cx);
    }

    fn insert_context_prefix(&mut self, prefix: &str, window: &mut Window, cx: &mut Context<Self>) {
        let editor = self.editor.clone();
        let prefix = prefix.to_string();

        cx.spawn_in(window, async move |_, cx| {
            editor
                .update_in(cx, |editor, window, cx| {
                    let menu_is_open =
                        editor.context_menu().borrow().as_ref().is_some_and(|menu| {
                            matches!(menu, CodeContextMenu::Completions(_)) && menu.visible()
                        });

                    let has_prefix = {
                        let snapshot = editor.display_snapshot(cx);
                        let cursor = editor.selections.newest::<text::Point>(&snapshot).head();
                        let offset = cursor.to_offset(&snapshot);
                        let buffer_snapshot = snapshot.buffer_snapshot();
                        let prefix_char_count = prefix.chars().count();
                        buffer_snapshot
                            .reversed_chars_at(offset)
                            .take(prefix_char_count)
                            .eq(prefix.chars().rev())
                    };

                    if menu_is_open && has_prefix {
                        return;
                    }

                    editor.insert(&prefix, window, cx);
                    editor.show_completions(&editor::actions::ShowCompletions, window, cx);
                })
                .log_err();
        })
        .detach();
    }

    fn chat(&mut self, _: &Chat, _: &mut Window, cx: &mut Context<Self>) {
        self.send(cx);
    }

    fn send_immediately(&mut self, _: &SendImmediately, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_empty(cx) {
            return;
        }

        self.editor.update(cx, |editor, cx| {
            editor.clear_inlay_hints(cx);
        });

        cx.emit(MessageEditorEvent::SendImmediately)
    }

    fn chat_with_follow(
        &mut self,
        _: &ChatWithFollow,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.workspace
            .update(cx, |this, cx| {
                this.follow(CollaboratorId::Agent, window, cx)
            })
            .log_err();

        self.send(cx);
    }

    fn cancel(&mut self, _: &editor::actions::Cancel, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(MessageEditorEvent::Cancel)
    }

    pub fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        let Some(clipboard) = cx.read_from_clipboard() else {
            return;
        };

        if self.editor.read(cx).read_only(cx) {
            let editor = self.editor.read(cx);
            let cursor_offset = editor
                .selections
                .newest_anchor()
                .head()
                .to_offset(&editor.buffer().read(cx).snapshot(cx))
                .0;
            cx.emit(MessageEditorEvent::InputAttempted {
                attempt: InputAttempt::Paste(clipboard),
                cursor_offset,
            });
            cx.stop_propagation();
            return;
        }

        cx.stop_propagation();
        self.paste_item(&clipboard, window, cx);
    }

    pub fn paste_item(
        &mut self,
        clipboard: &ClipboardItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };
        let editor_clipboard_selections =
            clipboard.entries().iter().find_map(|entry| match entry {
                ClipboardEntry::String(text) => {
                    text.metadata_json::<Vec<editor::ClipboardSelection>>()
                }
                _ => None,
            });

        // Insert creases for pasted clipboard selections that:
        // 1. Contain exactly one selection
        // 2. Have an associated file path
        // 3. Span multiple lines (not single-line selections)
        // 4. Belong to a file that exists in the current project
        let should_insert_creases = util::maybe!({
            let selections = editor_clipboard_selections.as_ref()?;
            if selections.len() > 1 {
                return Some(false);
            }
            let selection = selections.first()?;
            let file_path = selection.file_path.as_ref()?;
            let line_range = selection.line_range.as_ref()?;

            if line_range.start() == line_range.end() {
                return Some(false);
            }

            Some(
                workspace
                    .read(cx)
                    .project()
                    .read(cx)
                    .project_path_for_absolute_path(file_path, cx)
                    .is_some(),
            )
        })
        .unwrap_or(false);

        if should_insert_creases && let Some(selections) = editor_clipboard_selections {
            let snapshot = self.editor.read(cx).buffer().read(cx).snapshot(cx);
            let (insertion_target, _) = snapshot
                .anchor_to_buffer_anchor(self.editor.read(cx).selections.newest_anchor().start)
                .unwrap();

            let project = workspace.read(cx).project().clone();
            for selection in selections {
                if let (Some(file_path), Some(line_range)) =
                    (selection.file_path, selection.line_range)
                {
                    let crease_text =
                        acp_thread::selection_name(Some(file_path.as_ref()), &line_range);

                    let mention_uri = MentionUri::Selection {
                        abs_path: Some(file_path.clone()),
                        line_range: line_range.clone(),
                        column: None,
                    };

                    let mention_text = mention_uri.as_link().to_string();
                    let (text_anchor, content_len) = self.editor.update(cx, |editor, cx| {
                        let buffer = editor.buffer().read(cx);
                        let snapshot = buffer.snapshot(cx);
                        let buffer_snapshot = snapshot.as_singleton().unwrap();
                        let text_anchor = insertion_target.bias_left(&buffer_snapshot);

                        editor.insert(&mention_text, window, cx);
                        editor.insert(" ", window, cx);

                        (text_anchor, mention_text.len())
                    });

                    let Some((crease_id, tx, crease_entity)) = insert_crease_for_mention(
                        text_anchor,
                        content_len,
                        crease_text.into(),
                        mention_uri.icon_path(cx),
                        mention_uri.tooltip_text(),
                        Some(mention_uri.clone()),
                        Some(self.workspace.clone()),
                        None,
                        self.editor.clone(),
                        window,
                        cx,
                    ) else {
                        continue;
                    };
                    drop(tx);

                    let mention_task = cx
                        .spawn({
                            let project = project.clone();
                            async move |_, cx| {
                                let project_path = project
                                    .update(cx, |project, cx| {
                                        project.project_path_for_absolute_path(&file_path, cx)
                                    })
                                    .ok_or_else(|| {
                                        format!(
                                            "project path not found for pasted selection {file_path:?}"
                                        )
                                    })?;

                                let buffer = project
                                    .update(cx, |project, cx| project.open_buffer(project_path, cx))
                                    .await
                                    .map_err(|e| e.to_string())?;

                                Ok(buffer.update(cx, |buffer, cx| {
                                    let start =
                                        Point::new(*line_range.start(), 0).min(buffer.max_point());
                                    let end = Point::new(*line_range.end() + 1, 0)
                                        .min(buffer.max_point());
                                    let content = buffer.text_for_range(start..end).collect();
                                    Mention::Text {
                                        content,
                                        tracked_buffers: vec![cx.entity()],
                                    }
                                }))
                            }
                        })
                        .shared();

                    self.mention_set.update(cx, |mention_set, cx| {
                        mention_set.insert_mention(
                            crease_id,
                            mention_uri.clone(),
                            mention_task,
                            crease_entity,
                            cx,
                        )
                    });
                }
            }
            return;
        }
        // Handle text paste with potential markdown mention links before
        // clipboard context entries so markdown text still pastes as text.
        let clipboard_text = clipboard.entries().iter().find_map(|entry| match entry {
            ClipboardEntry::String(text) => Some(text.text().to_string()),
            _ => None,
        });
        if let Some(clipboard_text) = clipboard_text.as_deref() {
            if clipboard_text.contains("[@") {
                let selections_before = self.editor.update(cx, |editor, cx| {
                    let snapshot = editor.buffer().read(cx).snapshot(cx);
                    editor
                        .selections
                        .disjoint_anchors()
                        .iter()
                        .map(|selection| {
                            (
                                selection.start.bias_left(&snapshot),
                                selection.end.bias_right(&snapshot),
                            )
                        })
                        .collect::<Vec<_>>()
                });

                self.editor.update(cx, |editor, cx| {
                    editor.insert(clipboard_text, window, cx);
                });

                let snapshot = self.editor.read(cx).buffer().read(cx).snapshot(cx);
                let path_style = workspace.read(cx).project().read(cx).path_style(cx);

                let mut all_mentions = Vec::new();
                for (start_anchor, end_anchor) in selections_before {
                    let start_offset = start_anchor.to_offset(&snapshot);
                    let end_offset = end_anchor.to_offset(&snapshot);

                    // Get the actual inserted text from the buffer (may differ due to auto-indent)
                    let inserted_text: String =
                        snapshot.text_for_range(start_offset..end_offset).collect();

                    let parsed_mentions = parse_mention_links(&inserted_text, path_style);
                    for (range, mention_uri) in parsed_mentions {
                        let mention_start_offset = MultiBufferOffset(start_offset.0 + range.start);
                        let anchor = snapshot.anchor_before(mention_start_offset);
                        let content_len = range.end - range.start;
                        all_mentions.push((anchor, content_len, mention_uri));
                    }
                }

                if !all_mentions.is_empty() {
                    let supports_images = self.session_capabilities.read().supports_images();
                    let http_client = workspace.read(cx).client().http_client();

                    for (anchor, content_len, mention_uri) in all_mentions {
                        let Some((crease_id, tx, crease_entity)) = insert_crease_for_mention(
                            snapshot.anchor_to_buffer_anchor(anchor).unwrap().0,
                            content_len,
                            mention_uri.name().into(),
                            mention_uri.icon_path(cx),
                            mention_uri.tooltip_text(),
                            Some(mention_uri.clone()),
                            Some(self.workspace.clone()),
                            None,
                            self.editor.clone(),
                            window,
                            cx,
                        ) else {
                            continue;
                        };

                        // Create the confirmation task based on the mention URI type.
                        // This properly loads file content, fetches URLs, etc.
                        let task = self.mention_set.update(cx, |mention_set, cx| {
                            mention_set.confirm_mention_for_uri(
                                mention_uri.clone(),
                                supports_images,
                                http_client.clone(),
                                cx,
                            )
                        });
                        let task = cx
                            .spawn(async move |_, _| task.await.map_err(|e| e.to_string()))
                            .shared();

                        self.mention_set.update(cx, |mention_set, cx| {
                            mention_set.insert_mention(
                                crease_id,
                                mention_uri.clone(),
                                task.clone(),
                                crease_entity,
                                cx,
                            )
                        });

                        // Drop the tx after inserting to signal the crease is ready
                        drop(tx);
                    }
                    return;
                }
            }
        }

        if self.handle_pasted_context(clipboard, window, cx) {
            return;
        }

        self.editor.update(cx, |editor, cx| {
            editor.paste_item(clipboard, window, cx);
        });
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        let Some((text, _)) = self.serialize_selection_with_mentions(false, cx) else {
            cx.propagate();
            return;
        };

        cx.stop_propagation();
        cx.write_to_clipboard(ClipboardItem::new_string(text));
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        let Some((text, ranges)) = self.serialize_selection_with_mentions(true, cx) else {
            cx.propagate();
            return;
        };

        cx.stop_propagation();
        self.editor.update(cx, |editor, cx| {
            editor.transact(window, cx, |editor, window, cx| {
                editor.change_selections(Default::default(), window, cx, |selections| {
                    selections.select_ranges(ranges);
                });
                editor.insert("", window, cx);
            });
        });
        cx.write_to_clipboard(ClipboardItem::new_string(text));
    }

    fn paste_raw(&mut self, _: &PasteRaw, window: &mut Window, cx: &mut Context<Self>) {
        let editor = self.editor.clone();
        window.defer(cx, move |window, cx| {
            editor.update(cx, |editor, cx| editor.paste(&Paste, window, cx));
        });
    }

    fn handle_pasted_context(
        &mut self,
        clipboard: &ClipboardItem,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if matches!(
            clipboard.entries().first(),
            Some(ClipboardEntry::String(_)) | None
        ) {
            return false;
        }

        let Some(workspace) = self.workspace.upgrade() else {
            return false;
        };
        let project = workspace.read(cx).project().clone();
        let project_is_local = project.read(cx).is_local();
        let supports_images = self.session_capabilities.read().supports_images();
        if !project_is_local && !supports_images {
            return false;
        }
        let editor = self.editor.clone();
        let mention_set = self.mention_set.clone();
        let workspace = self.workspace.clone();
        let entries = clipboard.clone().into_entries().collect::<Vec<_>>();

        window
            .spawn(cx, async move |mut cx| {
                let (items, added_worktrees) = resolve_pasted_context_items(
                    project,
                    project_is_local,
                    supports_images,
                    entries,
                    &mut cx,
                )
                .await;
                insert_resolved_pasted_context_items(
                    items,
                    added_worktrees,
                    editor,
                    mention_set,
                    workspace,
                    supports_images,
                    &mut cx,
                )
                .await;
                Ok::<(), anyhow::Error>(())
            })
            .detach_and_log_err(cx);

        true
    }

    pub fn insert_dragged_files(
        &mut self,
        paths: Vec<project::ProjectPath>,
        added_worktrees: Vec<Entity<Worktree>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };
        let project = workspace.read(cx).project().clone();
        let supports_images = self.session_capabilities.read().supports_images();
        let mut tasks = Vec::new();
        for path in paths {
            if let Some(task) = insert_mention_for_project_path(
                &path,
                &self.editor,
                &self.mention_set,
                &project,
                &workspace,
                supports_images,
                window,
                cx,
            ) {
                tasks.push(task);
            }
        }
        cx.spawn(async move |_, _| {
            join_all(tasks).await;
            drop(added_worktrees);
        })
        .detach();
    }

    pub fn insert_branch_diff_crease(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };

        let project = workspace.read(cx).project().clone();

        let Some(repo) = project.read(cx).active_repository(cx) else {
            return;
        };

        let default_branch_receiver = repo.update(cx, |repo, _| repo.default_branch(false));
        let editor = self.editor.clone();
        let mention_set = self.mention_set.clone();
        let weak_workspace = self.workspace.clone();

        window
            .spawn(cx, async move |cx| {
                let base_ref: SharedString = default_branch_receiver
                    .await
                    .ok()
                    .and_then(|r| r.ok())
                    .flatten()
                    .ok_or_else(|| anyhow!("Could not determine default branch"))?;

                cx.update(|window, cx| {
                    let mention_uri = MentionUri::GitDiff {
                        base_ref: base_ref.to_string(),
                    };
                    let mention_text = mention_uri.as_link().to_string();

                    let (text_anchor, content_len) = editor.update(cx, |editor, cx| {
                        let buffer = editor.buffer().read(cx);
                        let snapshot = buffer.snapshot(cx);
                        let buffer_snapshot = snapshot.as_singleton().unwrap();
                        let text_anchor = snapshot
                            .anchor_to_buffer_anchor(editor.selections.newest_anchor().start)
                            .unwrap()
                            .0
                            .bias_left(&buffer_snapshot);

                        editor.insert(&mention_text, window, cx);
                        editor.insert(" ", window, cx);

                        (text_anchor, mention_text.len())
                    });

                    let Some((crease_id, tx, crease_entity)) = insert_crease_for_mention(
                        text_anchor,
                        content_len,
                        mention_uri.name().into(),
                        mention_uri.icon_path(cx),
                        mention_uri.tooltip_text(),
                        Some(mention_uri.clone()),
                        Some(weak_workspace),
                        None,
                        editor,
                        window,
                        cx,
                    ) else {
                        return;
                    };
                    drop(tx);

                    let confirm_task = mention_set.update(cx, |mention_set, cx| {
                        mention_set.confirm_mention_for_git_diff(base_ref, cx)
                    });

                    let mention_task = cx
                        .spawn(async move |_cx| confirm_task.await.map_err(|e| e.to_string()))
                        .shared();

                    mention_set.update(cx, |mention_set, cx| {
                        mention_set.insert_mention(
                            crease_id,
                            mention_uri,
                            mention_task,
                            crease_entity,
                            cx,
                        );
                    });
                })
            })
            .detach_and_log_err(cx);
    }

    pub fn insert_skill_crease(
        &mut self,
        skill: &AvailableSkill,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };

        let mention_uri = MentionUri::Skill {
            name: skill.name.to_string(),
            source: skill.source.to_string(),
            skill_file_path: skill.skill_file_path.clone(),
        };

        let link_text = mention_uri.as_link().to_string();
        let content_len = link_text.len();
        let mention_text = format!("{} ", link_text);
        let crease_text: SharedString = mention_uri.name().into();

        let start_anchor = self.editor.update(cx, |editor, cx| {
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let buffer_snapshot = snapshot.as_singleton()?;
            let cursor = editor.selections.newest_anchor().start;
            let text_anchor = snapshot
                .anchor_to_buffer_anchor(cursor)?
                .0
                .bias_left(buffer_snapshot);

            editor.insert(&mention_text, window, cx);
            Some(text_anchor)
        });

        let Some(start_anchor) = start_anchor else {
            return;
        };

        self.mention_set
            .update(cx, |mention_set, cx| {
                mention_set.confirm_mention_completion(
                    crease_text,
                    start_anchor,
                    content_len,
                    mention_uri,
                    false,
                    self.editor.clone(),
                    &workspace,
                    window,
                    cx,
                )
            })
            .detach();
    }

    pub(crate) fn insert_selections(
        &mut self,
        selection: AgentContextSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let editor = self.editor.read(cx);
        let editor_buffer = editor.buffer().read(cx);
        let Some(buffer) = editor_buffer.as_singleton() else {
            return;
        };
        let cursor_anchor = editor.selections.newest_anchor().head();
        let cursor_offset = cursor_anchor.to_offset(&editor_buffer.snapshot(cx));
        let anchor = buffer.update(cx, |buffer, _cx| {
            buffer.anchor_before(cursor_offset.0.min(buffer.len()))
        });
        let Some(completion) =
            PromptCompletionProvider::<MessageEditorCompletionDelegate>::completion_for_action(
                PromptContextAction::AddSelections,
                anchor..anchor,
                self.editor.downgrade(),
                self.mention_set.downgrade(),
                self.workspace.clone(),
                Some(selection),
            )
        else {
            return;
        };

        self.editor.update(cx, |message_editor, cx| {
            message_editor.edit([(cursor_anchor..cursor_anchor, completion.new_text)], cx);
            message_editor.request_autoscroll(Autoscroll::fit(), cx);
        });
        if let Some(confirm) = completion.confirm {
            confirm(CompletionIntent::Complete, window, cx);
        }
    }

    pub fn add_images_from_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.session_capabilities.read().supports_images() {
            return;
        }

        let editor = self.editor.clone();
        let mention_set = self.mention_set.clone();
        let workspace = self.workspace.clone();

        let paths_receiver = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Select Images".into()),
        });

        window
            .spawn(cx, async move |cx| {
                let paths = match paths_receiver.await {
                    Ok(Ok(Some(paths))) => paths,
                    _ => return Ok::<(), anyhow::Error>(()),
                };

                let default_image_name: SharedString = "Image".into();
                let images = cx
                    .background_spawn(async move {
                        paths
                            .into_iter()
                            .filter_map(|path| {
                                crate::mention_set::load_external_image_from_path(
                                    &path,
                                    &default_image_name,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .await;

                crate::mention_set::insert_images_as_context(
                    images,
                    editor,
                    mention_set,
                    workspace,
                    cx,
                )
                .await;
                Ok(())
            })
            .detach_and_log_err(cx);
    }

    pub fn set_read_only(&mut self, read_only: bool, cx: &mut Context<Self>) {
        self.editor.update(cx, |message_editor, cx| {
            message_editor.set_read_only(read_only);
            cx.notify()
        })
    }

    pub fn set_mode(&mut self, mode: EditorMode, cx: &mut Context<Self>) {
        self.editor.update(cx, |editor, cx| {
            if *editor.mode() != mode {
                editor.set_mode(mode);
                cx.notify()
            }
        });
    }

    pub fn set_message(
        &mut self,
        message: Vec<acp::ContentBlock>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.clear(window, cx);
        self.insert_message_blocks(message, false, true, window, cx);
    }

    pub fn append_message(
        &mut self,
        message: Vec<acp::ContentBlock>,
        separator: Option<&str>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if message.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "multiplayer-tools")]
        for chunk in &message {
            if let Some(reference) =
                crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(
                    chunk,
                )?
                && self.admits_inline_terminal_reference(&reference, cx)
            {
                let store =
                    crate::terminal_thread_metadata_store::TerminalThreadMetadataStore::try_global(
                        cx,
                    )
                    .ok_or_else(|| anyhow!("The queued terminal history is unavailable"))?;
                let context = store
                    .read(cx)
                    .entry(reference.terminal_id)
                    .and_then(|metadata| metadata.inline_context.as_ref())
                    .and_then(|context| context.occurrence(reference.occurrence_id))
                    .ok_or_else(|| anyhow!("The queued terminal conversation is unavailable"))?;
                anyhow::ensure!(
                    context.is_draft,
                    "A submitted terminal cannot be moved into a draft"
                );
                store.read(cx).validate_inline_context(context)?;
            }
        }

        // Queued content must preserve a selection in the existing draft;
        // mention ranges are computed relative to that draft's end below.
        self.set_cursor_offset(self.text(cx).len(), window, cx);

        if let Some(separator) = separator
            && !separator.is_empty()
            && !self.is_empty(cx)
        {
            self.editor.update(cx, |editor, cx| {
                editor.insert(separator, window, cx);
            });
        }

        self.insert_message_blocks(message, true, true, window, cx);
        Ok(())
    }

    fn insert_message_blocks(
        &mut self,
        message: Vec<acp::ContentBlock>,
        append_to_existing: bool,
        project_inline_resources: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };

        let path_style = workspace.read(cx).project().read(cx).path_style(cx);
        let mut text = String::new();
        let mut mentions = Vec::new();
        #[cfg(feature = "multiplayer-tools")]
        let mut inline_references = Vec::new();
        let append_normalized = |text: &mut String, mut segment: String| {
            LineEnding::normalize(&mut segment);
            text.push_str(&segment);
        };

        for chunk in message.iter().cloned() {
            #[cfg(feature = "multiplayer-tools")]
            match crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(
                &chunk,
            ) {
                Ok(Some(reference))
                    if project_inline_resources
                        && self.admits_inline_terminal_reference(&reference, cx) =>
                {
                    inline_references.push((text.len(), reference));
                    continue;
                }
                Err(error) => log::error!("Invalid inline terminal context: {error}"),
                _ => {}
            }
            match chunk {
                acp::ContentBlock::Text(text_content) => {
                    append_normalized(&mut text, text_content.text);
                }
                acp::ContentBlock::Resource(acp::EmbeddedResource {
                    resource: acp::EmbeddedResourceResource::TextResourceContents(resource),
                    ..
                }) => {
                    let Some(mention_uri) = MentionUri::parse(&resource.uri, path_style).log_err()
                    else {
                        continue;
                    };
                    let start = text.len();
                    append_normalized(&mut text, mention_uri.as_link().to_string());
                    let end = text.len();
                    mentions.push((
                        start..end,
                        mention_uri,
                        Mention::Text {
                            content: resource.text,
                            tracked_buffers: Vec::new(),
                        },
                    ));
                }
                acp::ContentBlock::ResourceLink(resource) => {
                    if let Some(mention_uri) =
                        MentionUri::parse(&resource.uri, path_style).log_err()
                    {
                        let start = text.len();
                        append_normalized(&mut text, mention_uri.as_link().to_string());
                        let end = text.len();
                        mentions.push((start..end, mention_uri, Mention::Link));
                    }
                }
                acp::ContentBlock::Image(acp::ImageContent {
                    uri,
                    data,
                    mime_type,
                    ..
                }) => {
                    let mention_uri = if let Some(uri) = uri {
                        MentionUri::parse(&uri, path_style)
                    } else {
                        Ok(MentionUri::PastedImage {
                            name: "Image".to_string(),
                        })
                    };
                    let Some(mention_uri) = mention_uri.log_err() else {
                        continue;
                    };
                    let Some(format) = ImageFormat::from_mime_type(&mime_type) else {
                        log::error!("failed to parse MIME type for image: {mime_type:?}");
                        continue;
                    };
                    let start = text.len();
                    append_normalized(&mut text, mention_uri.as_link().to_string());
                    let end = text.len();
                    mentions.push((
                        start..end,
                        mention_uri,
                        Mention::Image(MentionImage {
                            data: data.into(),
                            format,
                        }),
                    ));
                }
                _ => {}
            }
        }

        #[cfg(feature = "multiplayer-tools")]
        for (offset, reference) in &inline_references {
            if !reference
                .position
                .byte_offset_for_source(&text)
                .is_ok_and(|position| position == *offset)
            {
                log::error!(
                    "Inline terminal context has no verified position in this message; retaining it as context"
                );
                self.insert_message_blocks(message, append_to_existing, false, window, cx);
                return;
            }
        }
        #[cfg(not(feature = "multiplayer-tools"))]
        let _ = project_inline_resources;

        if text.is_empty() && mentions.is_empty() {
            #[cfg(feature = "multiplayer-tools")]
            if !inline_references.is_empty() {
                self.rebase_appended_inline_context(
                    &inline_references,
                    append_to_existing,
                    self.text(cx).len(),
                    cx,
                );
            }
            return;
        }

        let insertion_start = if append_to_existing {
            self.editor.read(cx).text(cx).len()
        } else {
            0
        };

        let snapshot = if append_to_existing {
            self.editor.update(cx, |editor, cx| {
                editor.insert(&text, window, cx);
                editor.buffer().read(cx).snapshot(cx)
            })
        } else {
            self.editor.update(cx, |editor, cx| {
                editor.set_text(text, window, cx);
                editor.buffer().read(cx).snapshot(cx)
            })
        };

        #[cfg(feature = "multiplayer-tools")]
        self.rebase_appended_inline_context(
            &inline_references,
            append_to_existing,
            insertion_start,
            cx,
        );

        for (range, mention_uri, mention) in mentions {
            let adjusted_start = insertion_start + range.start;
            let anchor = snapshot.anchor_before(MultiBufferOffset(adjusted_start));
            let image_preview = image_preview_task_for_mention(&mention);
            let Some((crease_id, tx, crease_entity)) = insert_crease_for_mention(
                snapshot.anchor_to_buffer_anchor(anchor).unwrap().0,
                range.end - range.start,
                mention_uri.name().into(),
                mention_uri.icon_path(cx),
                mention_uri.tooltip_text(),
                Some(mention_uri.clone()),
                Some(self.workspace.clone()),
                image_preview,
                self.editor.clone(),
                window,
                cx,
            ) else {
                continue;
            };
            drop(tx);

            self.mention_set.update(cx, |mention_set, cx| {
                mention_set.insert_mention(
                    crease_id,
                    mention_uri.clone(),
                    Task::ready(Ok(mention)).shared(),
                    crease_entity,
                    cx,
                )
            });
        }

        cx.notify();
    }

    #[cfg(feature = "multiplayer-tools")]
    fn rebase_appended_inline_context(
        &self,
        references: &[(
            usize,
            crate::terminal_thread_metadata_store::InlineTerminalPromptReference,
        )],
        append_to_existing: bool,
        insertion_start: usize,
        cx: &mut Context<Self>,
    ) {
        if !append_to_existing || references.is_empty() {
            return;
        }
        let Some(store) =
            crate::terminal_thread_metadata_store::TerminalThreadMetadataStore::try_global(cx)
        else {
            return;
        };
        let source = self.text(cx);
        for (offset, reference) in references {
            let result = (|| -> Result<()> {
                let mut metadata = store
                    .read(cx)
                    .occurrence_entry(
                        crate::terminal_thread_metadata_store::InlineTerminalOccurrenceKey {
                            terminal_id: reference.terminal_id,
                            occurrence_id: reference.occurrence_id,
                        },
                    )
                    .ok_or_else(|| anyhow::anyhow!("The queued terminal history is unavailable"))?;
                let context = metadata.inline_context.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("The queued terminal conversation is unavailable")
                })?;
                anyhow::ensure!(
                    context.is_draft,
                    "A submitted terminal cannot be moved into a draft"
                );
                let position = insertion_start
                    .checked_add(*offset)
                    .ok_or_else(|| anyhow::anyhow!("The queued terminal position is too large"))?;
                context.draft_position = Some(
                    crate::terminal_thread_metadata_store::InlineTerminalDraftPosition::new(
                        &source, position,
                    )?,
                );
                store.update(cx, |store, cx| {
                    store.save_inline_occurrence_batch(vec![metadata], cx)
                })
            })();
            if let Err(error) = result {
                self.workspace
                    .update(cx, |workspace, cx| workspace.show_error(error, cx))
                    .log_err();
            }
        }
    }

    pub fn text(&self, cx: &App) -> String {
        self.editor.read(cx).text(cx)
    }

    pub fn set_cursor_offset(
        &mut self,
        offset: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.editor.update(cx, |editor, cx| {
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let offset = snapshot.clip_offset(MultiBufferOffset(offset), text::Bias::Left);
            editor.change_selections(Default::default(), window, cx, |selections| {
                selections.select_ranges([offset..offset]);
            });
        });
    }

    pub fn insert_text(&mut self, text: &str, window: &mut Window, cx: &mut Context<Self>) {
        if text.is_empty() {
            return;
        }

        self.editor.update(cx, |editor, cx| {
            editor.insert(text, window, cx);
        });
    }

    pub fn set_placeholder_text(
        &mut self,
        placeholder: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.editor.update(cx, |editor, cx| {
            editor.set_placeholder_text(placeholder, window, cx);
        });
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn set_text(&mut self, text: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.editor.update(cx, |editor, cx| {
            editor.set_text(text, window, cx);
        });
    }

    fn serialize_selection_with_mentions(
        &self,
        expand_empty_to_line: bool,
        cx: &mut App,
    ) -> Option<(String, Vec<Range<MultiBufferOffset>>)> {
        if self.mention_set.read(cx).is_empty() {
            return None;
        }

        let display_snapshot = self
            .editor
            .update(cx, |editor, cx| editor.display_snapshot(cx));
        let editor = self.editor.read(cx);
        if !expand_empty_to_line && !editor.has_non_empty_selection(&display_snapshot) {
            return None;
        }

        let snapshot = editor.buffer().read(cx).snapshot(cx);
        let mention_set = self.mention_set.read(cx);
        let mention_ranges = display_snapshot
            .crease_snapshot
            .crease_items_with_offsets(&snapshot)
            .into_iter()
            .filter_map(|(crease_id, range)| {
                mention_set.mention_uri_for_crease(&crease_id).map(|uri| {
                    (
                        range.start.to_offset(&snapshot),
                        range.end.to_offset(&snapshot),
                        uri,
                    )
                })
            })
            .collect::<Vec<_>>();

        let line_mode = editor.selections.line_mode();
        let max_point = snapshot.max_point();
        let point_selections = editor.selections.all::<Point>(&display_snapshot);

        let mut text = String::new();
        let mut ranges = Vec::with_capacity(point_selections.len());
        let mut has_mentions = false;
        let mut is_first = true;
        let mut prev_was_entire_line = false;

        for mut selection in point_selections {
            let is_entire_line = (selection.is_empty() && expand_empty_to_line) || line_mode;
            if is_entire_line {
                selection.start = Point::new(selection.start.row, 0);
                if !selection.is_empty() && selection.end.column == 0 {
                    selection.end = min(max_point, selection.end);
                } else {
                    selection.end = min(max_point, Point::new(selection.end.row + 1, 0));
                }
            }
            let range = selection.start.to_offset(&snapshot)..selection.end.to_offset(&snapshot);

            if is_first {
                is_first = false;
            } else if !prev_was_entire_line {
                text.push('\n');
            }
            prev_was_entire_line = is_entire_line;

            let mut cursor = range.start;
            for (start, end, uri) in mention_ranges
                .iter()
                .filter(|(start, end, _)| *start < range.end && range.start < *end)
            {
                if cursor < *start {
                    text.extend(snapshot.text_for_range(cursor..*start));
                }
                write!(text, "{}", uri.as_link()).unwrap();
                cursor = *end;
                has_mentions = true;
            }
            if cursor < range.end {
                text.extend(snapshot.text_for_range(cursor..range.end));
            }

            ranges.push(range);
        }

        has_mentions.then_some((text, ranges))
    }
}

impl Focusable for MessageEditor {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.editor.focus_handle(cx)
    }
}

impl Render for MessageEditor {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("MessageEditor")
            .on_action(cx.listener(Self::chat))
            .on_action(cx.listener(Self::send_immediately))
            .on_action(cx.listener(Self::chat_with_follow))
            .on_action(cx.listener(Self::cancel))
            .capture_action(cx.listener(Self::copy))
            .capture_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::paste_raw))
            .capture_action(cx.listener(Self::paste))
            .flex_1()
            .child({
                let settings = ThemeSettings::get_global(cx);

                let text_style = TextStyle {
                    color: cx.theme().colors().text,
                    font_family: settings.agent_ui_font_family().clone(),
                    font_fallbacks: settings.ui_font.fallbacks.clone(),
                    font_features: settings.ui_font.features.clone(),
                    font_size: settings.agent_ui_font_size(cx).into(),
                    font_weight: settings.ui_font.weight,
                    line_height: relative(ui::fractal::BODY_LINE_HEIGHT),
                    ..Default::default()
                };

                EditorElement::new(
                    &self.editor,
                    EditorStyle {
                        background: cx.theme().colors().editor_background,
                        local_player: cx.theme().players().local(),
                        text: text_style,
                        syntax: cx.theme().syntax().clone(),
                        inlay_hints_style: editor::make_inlay_hints_style(cx),
                        ..Default::default()
                    },
                )
            })
    }
}

pub struct MessageEditorAddon {}

impl MessageEditorAddon {
    pub fn new() -> Self {
        Self {}
    }
}

impl Addon for MessageEditorAddon {
    fn to_any(&self) -> &dyn std::any::Any {
        self
    }

    fn to_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn extend_key_context(&self, key_context: &mut KeyContext, cx: &App) {
        let settings = agent_settings::AgentSettings::get_global(cx);
        if settings.use_modifier_to_send {
            key_context.add("use_modifier_to_send");
        }
    }
}

/// Walks the editor's creases in order, interleaving plain-text chunks from
/// `text` with mention blocks produced from `resolve`.
fn build_chunks_with_inline_context(
    text: &str,
    crease_snapshot: &CreaseSnapshot,
    buffer_snapshot: &MultiBufferSnapshot,
    supports_embedded_context: bool,
    whitespace: ContentWhitespace,
    inline_contents: Vec<(usize, acp::ContentBlock)>,
    mut resolve: impl FnMut(&CreaseId) -> Option<(MentionUri, Option<Mention>)>,
) -> (Vec<acp::ContentBlock>, Vec<Entity<Buffer>>) {
    // Terminal positions describe the exact editor source, including whitespace.
    let whitespace = if inline_contents.is_empty() {
        whitespace
    } else {
        ContentWhitespace::Preserve
    };
    let mut inline_contents = inline_contents.into_iter().peekable();
    let mut ix = match whitespace {
        ContentWhitespace::Preserve => 0,
        ContentWhitespace::Trim => text
            .char_indices()
            .find(|(_, character)| !character.is_whitespace())
            .map_or(text.len(), |(index, _)| index),
    };
    let mut chunks = Vec::new();
    let mut tracked_buffers = Vec::new();

    for (crease_id, crease) in crease_snapshot.creases() {
        let Some((uri, mention)) = resolve(&crease_id) else {
            continue;
        };
        let crease_range = crease.range().to_offset(buffer_snapshot);
        append_text_with_inline_context(
            text,
            ix,
            crease_range.start.0,
            &mut inline_contents,
            &mut chunks,
        );
        chunks.push(mention_to_content_block(
            &uri,
            mention.as_ref(),
            supports_embedded_context,
            &mut tracked_buffers,
        ));
        ix = crease_range.end.0;
    }

    let end = match whitespace {
        ContentWhitespace::Preserve => text.len(),
        ContentWhitespace::Trim => text.trim_end().len().max(ix),
    };
    append_text_with_inline_context(text, ix, end, &mut inline_contents, &mut chunks);
    (chunks, tracked_buffers)
}

fn append_text_with_inline_context(
    text: &str,
    mut start: usize,
    end: usize,
    insertions: &mut std::iter::Peekable<std::vec::IntoIter<(usize, acp::ContentBlock)>>,
    chunks: &mut Vec<acp::ContentBlock>,
) {
    while insertions.peek().is_some_and(|(offset, _)| *offset <= end) {
        if let Some((offset, content)) = insertions.next() {
            if let Some(segment) = text
                .get(start..offset)
                .filter(|segment| !segment.is_empty())
            {
                chunks.push(segment.to_owned().into());
            }
            chunks.push(content);
            start = offset;
        }
    }
    if let Some(segment) = text.get(start..end).filter(|segment| !segment.is_empty()) {
        chunks.push(segment.to_owned().into());
    }
}

fn image_preview_task_for_mention(
    mention: &Mention,
) -> Option<futures::future::Shared<Task<Result<Arc<Image>, String>>>> {
    let Mention::Image(mention_image) = mention else {
        return None;
    };

    let bytes =
        match base64::engine::general_purpose::STANDARD.decode(mention_image.data.as_bytes()) {
            Ok(bytes) => bytes,
            Err(error) => {
                log::error!("failed to decode image mention: {error}");
                return None;
            }
        };

    Some(
        Task::ready(Ok::<Arc<Image>, String>(Arc::new(Image::from_bytes(
            mention_image.format,
            bytes,
        ))))
        .shared(),
    )
}

fn mention_to_content_block(
    uri: &MentionUri,
    mention: Option<&Mention>,
    supports_embedded_context: bool,
    tracked_buffers: &mut Vec<Entity<Buffer>>,
) -> acp::ContentBlock {
    match mention {
        Some(Mention::Text {
            content,
            tracked_buffers: mention_tracked_buffers,
        }) => {
            tracked_buffers.extend(mention_tracked_buffers.iter().cloned());
            if supports_embedded_context {
                acp::ContentBlock::Resource(acp::EmbeddedResource::new(
                    acp::EmbeddedResourceResource::TextResourceContents(
                        acp::TextResourceContents::new(content.clone(), uri.to_uri().to_string()),
                    ),
                ))
            } else {
                acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                    uri.name(),
                    uri.to_uri().to_string(),
                ))
            }
        }
        Some(Mention::Image(mention_image)) => acp::ContentBlock::Image(
            acp::ImageContent::new(mention_image.data.clone(), mention_image.format.mime_type())
                .uri(match uri {
                    MentionUri::File { .. } | MentionUri::PastedImage { .. } => {
                        Some(uri.to_uri().to_string())
                    }
                    other => {
                        debug_panic!("unexpected mention uri for image: {:?}", other);
                        None
                    }
                }),
        ),
        _ => acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
            uri.name(),
            uri.to_uri().to_string(),
        )),
    }
}

/// Parses markdown mention links in the format `[@name](uri)` from text.
/// Returns a vector of (range, MentionUri) pairs where range is the byte range in the text.
fn parse_mention_links(text: &str, path_style: PathStyle) -> Vec<(Range<usize>, MentionUri)> {
    let mut mentions = Vec::new();
    let mut search_start = 0;

    while let Some(link_start) = text[search_start..].find("[@") {
        let absolute_start = search_start + link_start;

        // Find the matching closing bracket for the name, handling nested brackets.
        // Start at the '[' character so find_matching_bracket can track depth correctly.
        let Some(name_end) = find_matching_bracket(&text[absolute_start..], '[', ']') else {
            search_start = absolute_start + 2;
            continue;
        };
        let name_end = absolute_start + name_end;

        // Check for opening parenthesis immediately after
        if text.get(name_end + 1..name_end + 2) != Some("(") {
            search_start = name_end + 1;
            continue;
        }

        // Find the matching closing parenthesis for the URI, handling nested parens
        let uri_start = name_end + 2;
        let Some(uri_end_relative) = find_matching_bracket(&text[name_end + 1..], '(', ')') else {
            search_start = uri_start;
            continue;
        };
        let uri_end = name_end + 1 + uri_end_relative;
        let link_end = uri_end + 1;

        let uri_str = &text[uri_start..uri_end];

        // Try to parse the URI as a MentionUri
        if let Ok(mention_uri) = MentionUri::parse(uri_str, path_style) {
            mentions.push((absolute_start..link_end, mention_uri));
        }

        search_start = link_end;
    }

    mentions
}

/// Finds the position of the matching closing bracket, handling nested brackets.
/// The input `text` should start with the opening bracket.
/// Returns the index of the matching closing bracket relative to `text`.
fn find_matching_bracket(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0;
    for (index, character) in text.char_indices() {
        if character == open {
            depth += 1;
        } else if character == close {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::{ops::Range, path::Path, path::PathBuf, rc::Rc, sync::Arc};

    use super::PromptLocalCommand;
    use acp_thread::MentionUri;
    use agent::{ThreadStore, outline};
    use agent_client_protocol::schema::v1 as acp;
    use base64::Engine as _;
    use editor::{
        AnchorRangeExt as _, Editor, EditorMode, MultiBufferOffset, SelectionEffects,
        actions::{Cut, Paste},
    };

    use fs::FakeFs;
    use futures::{FutureExt as _, StreamExt as _};
    use gpui::{
        AppContext, ClipboardEntry, ClipboardItem, Entity, EventEmitter, ExternalPaths,
        FocusHandle, Focusable, Task, TestAppContext, VisualTestContext,
    };
    use language_model::LanguageModelRegistry;
    use lsp::{CompletionContext, CompletionTriggerKind};
    use parking_lot::RwLock;
    use project::{AgentId, CompletionIntent, Project, ProjectPath};
    use serde_json::{Value, json};

    use text::Point;
    use ui::{App, Context, IntoElement, Render, SharedString, Window};
    use util::{path, paths::PathStyle, rel_path::rel_path};
    use workspace::{AppState, Item, MultiWorkspace, Workspace};

    use crate::completion_provider::{AgentContextSelection, AvailableSkill, PromptContextType};
    use crate::{
        conversation_view::tests::init_test,
        mention_set::insert_crease_for_mention,
        message_editor::{
            Mention, MessageEditor, MessageEditorEvent, SessionCapabilities, parse_mention_links,
        },
    };

    #[test]
    fn test_inline_context_interleaving_preserves_unicode_source_and_terminal_only_input() {
        let source = "  Aλ🦀B  ";
        let first = acp::ContentBlock::Resource(acp::EmbeddedResource::new(
            acp::EmbeddedResourceResource::TextResourceContents(acp::TextResourceContents::new(
                "FIRST",
                "zed:///agent/terminal-selection?lines=1",
            )),
        ));
        let second = acp::ContentBlock::Resource(acp::EmbeddedResource::new(
            acp::EmbeddedResourceResource::TextResourceContents(acp::TextResourceContents::new(
                "SECOND",
                "zed:///agent/terminal-selection?lines=1",
            )),
        ));
        let mut insertions = vec![(5, first.clone()), (5, second.clone())]
            .into_iter()
            .peekable();
        let mut chunks = Vec::new();
        super::append_text_with_inline_context(
            source,
            0,
            source.len(),
            &mut insertions,
            &mut chunks,
        );
        assert_eq!(
            chunks,
            vec!["  Aλ".into(), first.clone(), second, "🦀B  ".into()]
        );
        assert!(insertions.next().is_none());
        let mut insertions = vec![(0, first.clone())].into_iter().peekable();
        let mut chunks = Vec::new();
        super::append_text_with_inline_context("", 0, 0, &mut insertions, &mut chunks);
        assert_eq!(chunks, vec![first]);
    }

    #[test]
    fn test_session_capabilities_keep_commands_and_skills_separate() {
        let skill_file_path = PathBuf::from("/tmp/SKILL.md");
        let skill = AvailableSkill {
            name: "deploy".into(),
            description: "Deploy the app".into(),
            source: "".into(),
            skill_file_path: skill_file_path.clone(),
            warning: None,
        };
        let session_capabilities = SessionCapabilities::new(
            acp::PromptCapabilities::default(),
            vec![acp::AvailableCommand::new("help", "Get help")],
            vec![skill],
        );

        assert_eq!(session_capabilities.completion_commands().len(), 1);
        let skills = session_capabilities.completion_skills();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name.as_ref(), "deploy");
        assert_eq!(skills[0].skill_file_path, skill_file_path);
    }

    #[test]
    fn test_completion_commands_derive_category_from_meta() {
        let session_capabilities = SessionCapabilities::new(
            acp::PromptCapabilities::default(),
            vec![
                acp::AvailableCommand::new("compact", "Built-in").meta(
                    acp_thread::meta_with_command_category(acp_thread::CommandCategory::Native),
                ),
                acp::AvailableCommand::new("deploy", "MCP").meta(
                    acp_thread::meta_with_command_category(acp_thread::CommandCategory::Mcp),
                ),
                // No category meta: this is how external ACP agents' commands
                // arrive, and they should group on their own.
                acp::AvailableCommand::new("help", "External"),
            ],
            Vec::new(),
        );

        let commands = session_capabilities.completion_commands();
        let category = |name: &str| {
            commands
                .iter()
                .find(|command| command.name.as_ref() == name)
                .unwrap()
                .category
        };
        assert_eq!(
            category("compact"),
            Some(acp_thread::CommandCategory::Native)
        );
        assert_eq!(category("deploy"), Some(acp_thread::CommandCategory::Mcp));
        assert_eq!(category("help"), None);
    }

    #[test]
    fn recipe_command_validation_uses_the_advertised_catalog() {
        let name = "recipe/0/.agents/recipes/review.yaml";
        let command = acp::AvailableCommand::new(name, "Review changes")
            .meta(acp_thread::meta_with_command_category(
                acp_thread::CommandCategory::Native,
            ))
            .input(acp::AvailableCommandInput::Unstructured(
                acp::UnstructuredCommandInput::new("<focus=value>"),
            ));
        let agent_id = AgentId::from("Zed");
        MessageEditor::validate_slash_commands(
            &format!("/{name} focus='a/b'"),
            &[command],
            &[],
            &agent_id,
        )
        .expect("advertised qualified recipe command");
        assert!(
            MessageEditor::validate_slash_commands(&format!("/{name}"), &[], &[], &agent_id)
                .is_err()
        );
    }

    #[test]
    fn test_validate_slash_commands_accepts_scope_qualified_skill() {
        let agent_id = AgentId::from("Zed");
        let make_skill = |name: &str, source: &str| AvailableSkill {
            name: name.into(),
            description: "desc".into(),
            source: source.into(),
            skill_file_path: PathBuf::from(format!("/tmp/{source}-{name}/SKILL.md")),
            warning: None,
        };

        // Global skills carry an empty scope (so the popup inserts
        // `/:<name>`); project-local skills carry their worktree root
        // name. The empty-scope encoding means a worktree literally
        // named `global` no longer collides with the global source.
        let commands = vec![acp::AvailableCommand::new("help", "Get help")];
        let skills = vec![make_skill("deploy", ""), make_skill("deploy", "zed")];
        let no_skills = Vec::new();

        // Bare name still works (current behavior — the resolver
        // applies project-overrides-global for unqualified commands).
        MessageEditor::validate_slash_commands("/deploy", &commands, &skills, &agent_id)
            .expect("bare /deploy should validate when a skill named `deploy` exists");
        MessageEditor::validate_slash_commands("/zed:deploy", &commands, &no_skills, &agent_id)
            .expect_err("scope-qualified skills should require a first-class available skill");

        // Scope-qualified forms both validate, each pointing at the
        // matching source. `/:<name>` is the qualified form for a
        // global skill; `/<worktree>:<name>` is the qualified form
        // for a project-local skill.
        MessageEditor::validate_slash_commands("/:deploy", &commands, &skills, &agent_id)
            .expect("/:deploy should validate when a global skill named `deploy` exists");
        MessageEditor::validate_slash_commands("/zed:deploy", &commands, &skills, &agent_id).expect(
            "/zed:deploy should validate when a project skill named `deploy` exists in the `zed` worktree",
        );

        // Hand-typed `/global:<name>` is NOT an alias for `/:<name>`.
        // It looks for a project-local skill from a worktree named
        // `global`, and fails when no such worktree skill exists.
        MessageEditor::validate_slash_commands("/global:deploy", &commands, &skills, &agent_id)
            .expect_err(
                "/global:deploy should fail when no worktree named `global` has a `deploy` skill",
            );

        // The `:` separator is what distinguishes a skill scope from
        // an MCP server prefix — the dotted form `/zed.deploy` is an
        // MCP-style lookup, which doesn't match here.
        MessageEditor::validate_slash_commands("/zed.deploy", &commands, &skills, &agent_id)
            .expect_err("/zed.deploy (dotted) should be treated as an MCP-style prefix and fail");

        // Wrong scope is rejected so the resolver doesn't silently
        // fall through when the user meant a skill. `zed:help` looks
        // like a skill scope qualifier but no skill named `help`
        // exists in the `zed` worktree (it's an MCP command).
        let err =
            MessageEditor::validate_slash_commands("/zed:help", &commands, &skills, &agent_id)
                .expect_err(
                    "/zed:help should fail — `help` is an MCP command, not a worktree skill",
                );
        let err_message = err.to_string();
        assert!(
            err_message.contains("/zed:help"),
            "error should mention the typed command: {err_message}"
        );
        // Error listing shows qualified forms for skills so users see
        // the exact text the popup would have inserted. Globals
        // render with an empty scope as `/:<name>`.
        assert!(
            err_message.contains("/:deploy"),
            "error listing should show qualified global form: {err_message}"
        );
        assert!(
            err_message.contains("/zed:deploy"),
            "error listing should show qualified worktree form: {err_message}"
        );
        assert!(
            err_message.contains("/help"),
            "error listing should still show bare MCP commands: {err_message}"
        );

        // Slashes that appear mid-text (paths, URLs, pasted logs)
        // should NOT be validated as commands.
        MessageEditor::validate_slash_commands(
            "check /docs for info",
            &commands,
            &skills,
            &agent_id,
        )
        .expect("mid-text /docs should not be treated as a slash command");

        MessageEditor::validate_slash_commands(
            "see /usr/local/bin/foo",
            &commands,
            &skills,
            &agent_id,
        )
        .expect("file paths containing slashes should not trigger validation");
    }

    #[test]
    fn test_parse_mention_links() {
        // Single file mention
        let text = "[@bundle-mac](file:///Users/test/zed/script/bundle-mac)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].0, 0..text.len());
        assert!(matches!(mentions[0].1, MentionUri::File { .. }));

        // Multiple mentions
        let text = "Check [@file1](file:///path/to/file1) and [@file2](file:///path/to/file2)!";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 2);

        // Text without mentions
        let text = "Just some regular text without mentions";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 0);

        // Malformed mentions (should be skipped)
        let text = "[@incomplete](invalid://uri) and [@missing](";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 0);

        // Mixed content with valid mention
        let text = "Before [@valid](file:///path/to/file) after";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].0.start, 7);

        // HTTP URL mention (Fetch)
        let text = "Check out [@docs](https://example.com/docs) for more info";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 1);
        assert!(matches!(mentions[0].1, MentionUri::Fetch { .. }));

        // Directory mention (trailing slash)
        let text = "[@src](file:///path/to/src/)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 1);
        assert!(matches!(mentions[0].1, MentionUri::Directory { .. }));

        // Multiple different mention types
        let text = "File [@f](file:///a) and URL [@u](https://b.com) and dir [@d](file:///c/)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 3);
        assert!(matches!(mentions[0].1, MentionUri::File { .. }));
        assert!(matches!(mentions[1].1, MentionUri::Fetch { .. }));
        assert!(matches!(mentions[2].1, MentionUri::Directory { .. }));

        // Adjacent mentions without separator
        let text = "[@a](file:///a)[@b](file:///b)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 2);

        // Regular markdown link (not a mention) should be ignored
        let text = "[regular link](https://example.com)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 0);

        // Incomplete mention link patterns
        let text = "[@name] without url and [@name( malformed";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 0);

        // Nested brackets in name portion
        let text = "[@name [with brackets]](file:///path/to/file)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].0, 0..text.len());

        // Deeply nested brackets
        let text = "[@outer [inner [deep]]](file:///path)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 1);

        // Unbalanced brackets should fail gracefully
        let text = "[@unbalanced [bracket](file:///path)";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 0);

        // Nested parentheses in URI (common in URLs with query params)
        let text = "[@wiki](https://en.wikipedia.org/wiki/Rust_(programming_language))";
        let mentions = parse_mention_links(text, PathStyle::local());
        assert_eq!(mentions.len(), 1);
        if let MentionUri::Fetch { url } = &mentions[0].1 {
            assert!(url.as_str().contains("Rust_(programming_language)"));
        } else {
            panic!("Expected Fetch URI");
        }
    }

    #[gpui::test]
    async fn test_at_mention_removal(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file": ""})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = None;

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            })
        });
        let editor = message_editor.update(cx, |message_editor, _| message_editor.editor.clone());

        cx.run_until_parked();

        let completions = editor.update_in(cx, |editor, window, cx| {
            editor.set_text("Hello @file ", window, cx);
            let buffer = editor.buffer().read(cx).as_singleton().unwrap();
            let completion_provider = editor.completion_provider().unwrap();
            completion_provider.completions(
                &buffer,
                text::Anchor::max_for_buffer(buffer.read(cx).remote_id()),
                CompletionContext {
                    trigger_kind: CompletionTriggerKind::TRIGGER_CHARACTER,
                    trigger_character: Some("@".into()),
                },
                window,
                cx,
            )
        });
        let [_, completion]: [_; 2] = completions
            .await
            .unwrap()
            .into_iter()
            .flat_map(|response| response.completions)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        editor.update_in(cx, |editor, window, cx| {
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let range = snapshot
                .buffer_anchor_range_to_anchor_range(completion.replace_range)
                .unwrap();
            editor.edit([(range, completion.new_text)], cx);
            (completion.confirm.unwrap())(CompletionIntent::Complete, window, cx);
        });

        cx.run_until_parked();

        // Backspace over the inserted crease (and the following space).
        editor.update_in(cx, |editor, window, cx| {
            editor.backspace(&Default::default(), window, cx);
            editor.backspace(&Default::default(), window, cx);
        });

        let (content, _) = message_editor
            .update(cx, |message_editor, cx| message_editor.contents(false, cx))
            .await
            .unwrap();

        // We don't send a resource link for the deleted crease.
        pretty_assertions::assert_matches!(content.as_slice(), [acp::ContentBlock::Text { .. }]);
    }

    #[gpui::test]
    async fn test_slash_command_validation(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/test",
            json!({
                ".zed": {
                    "tasks.json": r#"[{"label": "test", "command": "echo"}]"#
                },
                "src": {
                    "main.rs": "fn main() {}",
                },
            }),
        )
        .await;

        let project = Project::test(fs.clone(), ["/test".as_ref()], cx).await;
        let thread_store = None;
        let session_capabilities = Arc::new(RwLock::new(SessionCapabilities::from_acp_commands(
            acp::PromptCapabilities::default(),
            vec![],
        )));

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());
        let workspace_handle = workspace.downgrade();
        let message_editor = workspace.update_in(cx, |_, window, cx| {
            cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle.clone(),
                    project.downgrade(),
                    thread_store.clone(),
                    session_capabilities.clone(),
                    "Claude Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            })
        });
        let editor = message_editor.update(cx, |message_editor, _| message_editor.editor.clone());

        // Test that slash commands fail when no available_commands are set (empty list means no commands supported)
        editor.update_in(cx, |editor, window, cx| {
            editor.set_text("/file test.txt", window, cx);
        });

        let contents_result = message_editor
            .update(cx, |message_editor, cx| message_editor.contents(false, cx))
            .await;

        // Should fail because available_commands is empty (no commands supported)
        assert!(contents_result.is_err());
        let error_message = contents_result.unwrap_err().to_string();
        assert!(error_message.contains("is not a recognized command in Claude Agent"));
        assert!(error_message.contains("Available commands for Claude Agent: none"));

        // Now simulate Claude providing its list of available commands (which doesn't include file)
        session_capabilities
            .write()
            .set_available_commands(vec![acp::AvailableCommand::new("help", "Get help")]);

        // Test that unsupported slash commands trigger an error when we have a list of available commands
        editor.update_in(cx, |editor, window, cx| {
            editor.set_text("/file test.txt", window, cx);
        });

        let contents_result = message_editor
            .update(cx, |message_editor, cx| message_editor.contents(false, cx))
            .await;

        assert!(contents_result.is_err());
        let error_message = contents_result.unwrap_err().to_string();
        assert!(error_message.contains("is not a recognized command in Claude Agent"));
        assert!(error_message.contains("/file"));
        assert!(error_message.contains("Available commands for Claude Agent: /help"));

        // Test that supported commands work fine
        editor.update_in(cx, |editor, window, cx| {
            editor.set_text("/help", window, cx);
        });

        let contents_result = message_editor
            .update(cx, |message_editor, cx| message_editor.contents(false, cx))
            .await;

        // Should succeed because /help is in available_commands
        assert!(contents_result.is_ok());

        // Test that regular text works fine
        editor.update_in(cx, |editor, window, cx| {
            editor.set_text("Hello Claude!", window, cx);
        });

        let (content, _) = message_editor
            .update(cx, |message_editor, cx| message_editor.contents(false, cx))
            .await
            .unwrap();

        assert_eq!(content.len(), 1);
        if let acp::ContentBlock::Text(text) = &content[0] {
            assert_eq!(text.text, "Hello Claude!");
        } else {
            panic!("Expected ContentBlock::Text");
        }

        // Test that @ mentions still work
        editor.update_in(cx, |editor, window, cx| {
            editor.set_text("Check this @", window, cx);
        });

        // The @ mention functionality should not be affected
        let (content, _) = message_editor
            .update(cx, |message_editor, cx| message_editor.contents(false, cx))
            .await
            .unwrap();

        assert_eq!(content.len(), 1);
        if let acp::ContentBlock::Text(text) = &content[0] {
            assert_eq!(text.text, "Check this @");
        } else {
            panic!("Expected ContentBlock::Text");
        }
    }

    struct MessageEditorItem(Entity<MessageEditor>);

    impl Item for MessageEditorItem {
        type Event = ();

        fn include_in_nav_history() -> bool {
            false
        }

        fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
            "Test".into()
        }
    }

    impl EventEmitter<()> for MessageEditorItem {}

    impl Focusable for MessageEditorItem {
        fn focus_handle(&self, cx: &App) -> FocusHandle {
            self.0.read(cx).focus_handle(cx)
        }
    }

    impl Render for MessageEditorItem {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.0.clone().into_any_element()
        }
    }

    #[gpui::test]
    async fn test_completion_provider_commands(cx: &mut TestAppContext) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        let project = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let thread_store = None;
        let session_capabilities = Arc::new(RwLock::new(SessionCapabilities::from_acp_commands(
            acp::PromptCapabilities::default(),
            vec![
                acp::AvailableCommand::new("quick-math", "2 + 2 = 4 - 1 = 3"),
                acp::AvailableCommand::new("say-hello", "Say hello to whoever you want").input(
                    acp::AvailableCommandInput::Unstructured(acp::UnstructuredCommandInput::new(
                        "<name>",
                    )),
                ),
            ],
        )));

        let editor = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    thread_store.clone(),
                    session_capabilities.clone(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            message_editor.read(cx).editor().clone()
        });

        cx.simulate_input("/");

        editor.update_in(&mut cx, |editor, window, cx| {
            assert_eq!(editor.text(cx), "/");
            assert!(editor.has_visible_completions_menu());

            assert_eq!(
                current_completion_labels_with_documentation(editor),
                &[
                    ("quick-math".into(), "2 + 2 = 4 - 1 = 3".into()),
                    ("say-hello".into(), "Say hello to whoever you want".into())
                ]
            );
            editor.set_text("", window, cx);
        });

        cx.simulate_input("/qui");

        editor.update_in(&mut cx, |editor, window, cx| {
            assert_eq!(editor.text(cx), "/qui");
            assert!(editor.has_visible_completions_menu());

            assert_eq!(
                current_completion_labels_with_documentation(editor),
                &[("quick-math".into(), "2 + 2 = 4 - 1 = 3".into())]
            );
            editor.set_text("", window, cx);
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            assert!(editor.has_visible_completions_menu());
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        cx.run_until_parked();

        editor.update_in(&mut cx, |editor, window, cx| {
            assert_eq!(editor.display_text(cx), "/quick-math ");
            assert!(!editor.has_visible_completions_menu());
            editor.set_text("", window, cx);
        });

        cx.simulate_input("/say");

        editor.update_in(&mut cx, |editor, _window, cx| {
            assert_eq!(editor.display_text(cx), "/say");
            assert!(editor.has_visible_completions_menu());

            assert_eq!(
                current_completion_labels_with_documentation(editor),
                &[("say-hello".into(), "Say hello to whoever you want".into())]
            );
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            assert!(editor.has_visible_completions_menu());
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        cx.run_until_parked();

        editor.update_in(&mut cx, |editor, _window, cx| {
            assert_eq!(editor.text(cx), "/say-hello ");
            assert_eq!(editor.display_text(cx), "/say-hello <name>");
            assert!(!editor.has_visible_completions_menu());
        });

        cx.simulate_input("GPT5");

        cx.run_until_parked();

        editor.update_in(&mut cx, |editor, window, cx| {
            assert_eq!(editor.text(cx), "/say-hello GPT5");
            assert_eq!(editor.display_text(cx), "/say-hello GPT5");
            assert!(!editor.has_visible_completions_menu());

            // Delete argument
            for _ in 0..5 {
                editor.backspace(&editor::actions::Backspace, window, cx);
            }
        });

        cx.run_until_parked();

        editor.update_in(&mut cx, |editor, window, cx| {
            assert_eq!(editor.text(cx), "/say-hello");
            // Hint is visible because argument was deleted
            assert_eq!(editor.display_text(cx), "/say-hello <name>");

            // Delete last command letter
            editor.backspace(&editor::actions::Backspace, window, cx);
        });

        cx.run_until_parked();

        editor.update_in(&mut cx, |editor, _window, cx| {
            // Hint goes away once command no longer matches an available one
            assert_eq!(editor.text(cx), "/say-hell");
            assert_eq!(editor.display_text(cx), "/say-hell");
            assert!(!editor.has_visible_completions_menu());
        });
    }

    /// Local commands set via `set_local_commands` must surface in the
    /// slash-command popup, and confirming one must emit
    /// `MessageEditorEvent::LocalCommandInvoked` (so `ThreadView` can run the
    /// corresponding action) without leaving the `/keyword` in the editor.
    #[gpui::test]
    async fn test_local_commands_complete_and_emit_event(cx: &mut TestAppContext) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        let project = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let session_capabilities = Arc::new(RwLock::new(SessionCapabilities::from_acp_commands(
            acp::PromptCapabilities::default(),
            Vec::new(),
        )));

        let (message_editor, editor) = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    None,
                    session_capabilities.clone(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            let editor = message_editor.read(cx).editor().clone();
            (message_editor, editor)
        });

        message_editor.read_with(&cx, |message_editor, _| {
            message_editor.set_local_commands(vec![
                PromptLocalCommand::ThumbsUp,
                PromptLocalCommand::ThumbsDown,
            ]);
        });

        let invoked = Rc::new(std::cell::RefCell::new(Vec::new()));
        let _subscription = cx.update(|_, cx| {
            cx.subscribe(&message_editor, {
                let invoked = invoked.clone();
                move |_editor, event, _cx| {
                    if let MessageEditorEvent::LocalCommandInvoked(command) = event {
                        invoked.borrow_mut().push(*command);
                    }
                }
            })
        });

        // `/helpful` would fuzzy-match both commands ("helpful" is a
        // subsequence of "not-helpful"), so drive the unambiguous keyword.
        cx.simulate_input("/not-helpful");
        cx.run_until_parked();

        editor.read_with(&cx, |editor, _| {
            assert!(editor.has_visible_completions_menu());
            assert_eq!(
                current_completion_labels(editor),
                &[PromptLocalCommand::ThumbsDown.label().to_string()],
            );
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        cx.run_until_parked();

        editor.read_with(&cx, |editor, cx| {
            // The `/keyword` text is removed when a local command is confirmed.
            assert_eq!(editor.text(cx), "");
            assert!(!editor.has_visible_completions_menu());
        });

        assert_eq!(
            invoked.borrow().as_slice(),
            &[PromptLocalCommand::ThumbsDown],
        );
    }

    /// Opening slash-command autocomplete must emit
    /// [`MessageEditorEvent::SlashAutocompleteOpened`]. `ThreadView`
    /// subscribes to that event to fire the global-skills scan trigger
    /// (see `NativeAgent::ensure_skills_scan_started`); without the
    /// event the trigger never runs and lazily-discovered skills never
    /// appear in autocomplete.
    #[gpui::test]
    async fn test_slash_autocomplete_emits_opened_event(cx: &mut TestAppContext) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        let project = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let session_capabilities = Arc::new(RwLock::new(SessionCapabilities::from_acp_commands(
            acp::PromptCapabilities::default(),
            vec![acp::AvailableCommand::new("hello", "Say hello")],
        )));

        // Track every event emitted by the message editor across the
        // lifetime of the test. We expect to see Focus (from the focus
        // call below) and SlashAutocompleteOpened (from typing "/").
        let received_events: Arc<parking_lot::Mutex<Vec<MessageEditorEvent>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));

        let editor = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    None,
                    session_capabilities.clone(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });

            let received_events = received_events.clone();
            cx.subscribe(
                &message_editor,
                move |_editor: &mut Workspace, _, event: &MessageEditorEvent, _cx| {
                    received_events.lock().push(event.clone());
                },
            )
            .detach();

            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            message_editor.read(cx).editor().clone()
        });

        cx.simulate_input("/");

        editor.update_in(&mut cx, |editor, _window, cx| {
            assert_eq!(editor.text(cx), "/");
            assert!(editor.has_visible_completions_menu());
        });

        let events = received_events.lock();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, MessageEditorEvent::SlashAutocompleteOpened)),
            "expected SlashAutocompleteOpened to have been emitted; saw events: {events:?}",
        );
    }

    #[gpui::test]
    async fn test_context_completion_provider_mentions(cx: &mut TestAppContext) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/dir"),
                json!({
                    "editor": "",
                    "a": {
                        "one.txt": "1",
                        "two.txt": "2",
                        "three.txt": "3",
                        "four.txt": "4"
                    },
                    "b": {
                        "five.txt": "5",
                        "six.txt": "6",
                        "seven.txt": "7",
                        "eight.txt": "8",
                    },
                    "x.png": "",
                }),
            )
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let worktree = project.update(cx, |project, cx| {
            let mut worktrees = project.worktrees(cx).collect::<Vec<_>>();
            assert_eq!(worktrees.len(), 1);
            worktrees.pop().unwrap()
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let paths = vec![
            rel_path("a/one.txt"),
            rel_path("a/two.txt"),
            rel_path("a/three.txt"),
            rel_path("a/four.txt"),
            rel_path("b/five.txt"),
            rel_path("b/six.txt"),
            rel_path("b/seven.txt"),
            rel_path("b/eight.txt"),
        ];

        let slash = PathStyle::local().primary_separator();

        let mut opened_editors = Vec::new();
        for path in paths {
            let buffer = workspace
                .update_in(&mut cx, |workspace, window, cx| {
                    workspace.open_path(
                        ProjectPath {
                            worktree_id,
                            path: path.into(),
                        },
                        None,
                        false,
                        window,
                        cx,
                    )
                })
                .await
                .unwrap();
            opened_editors.push(buffer);
        }

        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let session_capabilities = Arc::new(RwLock::new(SessionCapabilities::from_acp_commands(
            acp::PromptCapabilities::default(),
            vec![],
        )));

        let (message_editor, editor) = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    Some(thread_store),
                    session_capabilities.clone(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            let editor = message_editor.read(cx).editor().clone();
            (message_editor, editor)
        });

        cx.simulate_input("Lorem @");

        editor.update_in(&mut cx, |editor, window, cx| {
            assert_eq!(editor.text(cx), "Lorem @");
            assert!(editor.has_visible_completions_menu());

            assert_eq!(
                current_completion_labels(editor),
                &[
                    format!("eight.txt b{slash}"),
                    format!("seven.txt b{slash}"),
                    format!("six.txt b{slash}"),
                    format!("five.txt b{slash}"),
                    "Files & Directories".into(),
                    "Symbols".into()
                ]
            );
            editor.set_text("", window, cx);
        });

        message_editor.update(&mut cx, |editor, _cx| {
            editor.session_capabilities.write().set_prompt_capabilities(
                acp::PromptCapabilities::new()
                    .image(true)
                    .audio(true)
                    .embedded_context(true),
            );
        });

        cx.simulate_input("Lorem ");

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), "Lorem ");
            assert!(!editor.has_visible_completions_menu());
        });

        cx.simulate_input("@");

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), "Lorem @");
            assert!(editor.has_visible_completions_menu());
            assert_eq!(
                current_completion_labels(editor),
                &[
                    format!("eight.txt b{slash}"),
                    format!("seven.txt b{slash}"),
                    format!("six.txt b{slash}"),
                    format!("five.txt b{slash}"),
                    "Files & Directories".into(),
                    "Symbols".into(),
                    "Threads".into(),
                    "Fetch".into()
                ]
            );
        });

        // Select and confirm "File"
        editor.update_in(&mut cx, |editor, window, cx| {
            assert!(editor.has_visible_completions_menu());
            editor.context_menu_next(&editor::actions::ContextMenuNext, window, cx);
            editor.context_menu_next(&editor::actions::ContextMenuNext, window, cx);
            editor.context_menu_next(&editor::actions::ContextMenuNext, window, cx);
            editor.context_menu_next(&editor::actions::ContextMenuNext, window, cx);
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        cx.run_until_parked();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), "Lorem @file ");
            assert!(editor.has_visible_completions_menu());
        });

        cx.simulate_input("one");

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), "Lorem @file one");
            assert!(editor.has_visible_completions_menu());
            assert_eq!(
                current_completion_labels(editor),
                vec![format!("one.txt a{slash}")]
            );
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            assert!(editor.has_visible_completions_menu());
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        let url_one = MentionUri::File {
            abs_path: path!("/dir/a/one.txt").into(),
        }
        .to_uri()
        .to_string();
        editor.update(&mut cx, |editor, cx| {
            let text = editor.text(cx);
            assert_eq!(text, format!("Lorem [@one.txt]({url_one}) "));
            assert!(!editor.has_visible_completions_menu());
            assert_eq!(fold_ranges(editor, cx).len(), 1);
        });

        let contents = message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor
                    .mention_set()
                    .update(cx, |mention_set, cx| mention_set.contents(false, cx))
            })
            .await
            .unwrap()
            .into_values()
            .collect::<Vec<_>>();

        {
            let [(uri, Mention::Text { content, .. })] = contents.as_slice() else {
                panic!("Unexpected mentions");
            };
            pretty_assertions::assert_eq!(content, "1");
            pretty_assertions::assert_eq!(
                uri,
                &MentionUri::parse(&url_one, PathStyle::local()).unwrap()
            );
        }

        cx.simulate_input(" ");

        editor.update(&mut cx, |editor, cx| {
            let text = editor.text(cx);
            assert_eq!(text, format!("Lorem [@one.txt]({url_one})  "));
            assert!(!editor.has_visible_completions_menu());
            assert_eq!(fold_ranges(editor, cx).len(), 1);
        });

        cx.simulate_input("Ipsum ");

        editor.update(&mut cx, |editor, cx| {
            let text = editor.text(cx);
            assert_eq!(text, format!("Lorem [@one.txt]({url_one})  Ipsum "),);
            assert!(!editor.has_visible_completions_menu());
            assert_eq!(fold_ranges(editor, cx).len(), 1);
        });

        cx.simulate_input("@file ");

        editor.update(&mut cx, |editor, cx| {
            let text = editor.text(cx);
            assert_eq!(text, format!("Lorem [@one.txt]({url_one})  Ipsum @file "),);
            assert!(editor.has_visible_completions_menu());
            assert_eq!(fold_ranges(editor, cx).len(), 1);
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        cx.run_until_parked();

        let contents = message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor
                    .mention_set()
                    .update(cx, |mention_set, cx| mention_set.contents(false, cx))
            })
            .await
            .unwrap()
            .into_values()
            .collect::<Vec<_>>();

        let url_eight = MentionUri::File {
            abs_path: path!("/dir/b/eight.txt").into(),
        }
        .to_uri()
        .to_string();

        {
            let [_, (uri, Mention::Text { content, .. })] = contents.as_slice() else {
                panic!("Unexpected mentions");
            };
            pretty_assertions::assert_eq!(content, "8");
            pretty_assertions::assert_eq!(
                uri,
                &MentionUri::parse(&url_eight, PathStyle::local()).unwrap()
            );
        }

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!("Lorem [@one.txt]({url_one})  Ipsum [@eight.txt]({url_eight}) ")
            );
            assert!(!editor.has_visible_completions_menu());
            assert_eq!(fold_ranges(editor, cx).len(), 2);
        });

        let plain_text_language = Arc::new(language::Language::new(
            language::LanguageConfig {
                name: "Plain Text".into(),
                matcher: (language::LanguageMatcher {
                    path_suffixes: vec!["txt".to_string()],
                    ..Default::default()
                })
                .into(),
                ..Default::default()
            },
            None,
        ));

        // Register the language and fake LSP
        let language_registry = project.read_with(&cx, |project, _| project.languages().clone());
        language_registry.add(plain_text_language);

        let mut fake_language_servers = language_registry.register_fake_lsp(
            "Plain Text",
            language::FakeLspAdapter {
                capabilities: lsp::ServerCapabilities {
                    workspace_symbol_provider: Some(lsp::OneOf::Left(true)),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        // Open the buffer to trigger LSP initialization
        let buffer = project
            .update(&mut cx, |project, cx| {
                project.open_local_buffer(path!("/dir/a/one.txt"), cx)
            })
            .await
            .unwrap();

        // Register the buffer with language servers
        let _handle = project.update(&mut cx, |project, cx| {
            project.register_buffer_with_language_servers(&buffer, cx)
        });

        cx.run_until_parked();

        let fake_language_server = fake_language_servers.next().await.unwrap();
        fake_language_server.set_request_handler::<lsp::WorkspaceSymbolRequest, _, _>(
            move |_, _| async move {
                Ok(Some(lsp::WorkspaceSymbolResponse::Flat(vec![
                    #[allow(deprecated)]
                    lsp::SymbolInformation {
                        name: "MySymbol".into(),
                        location: lsp::Location {
                            uri: lsp::Uri::from_file_path(path!("/dir/a/one.txt")).unwrap(),
                            range: lsp::Range::new(
                                lsp::Position::new(0, 0),
                                lsp::Position::new(0, 1),
                            ),
                        },
                        kind: lsp::SymbolKind::CONSTANT,
                        tags: None,
                        container_name: None,
                        deprecated: None,
                    },
                ])))
            },
        );

        cx.simulate_input("@symbol ");

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!("Lorem [@one.txt]({url_one})  Ipsum [@eight.txt]({url_eight}) @symbol ")
            );
            assert!(editor.has_visible_completions_menu());
            assert_eq!(current_completion_labels(editor), &["MySymbol one.txt L1"]);
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        let symbol = MentionUri::Symbol {
            abs_path: path!("/dir/a/one.txt").into(),
            name: "MySymbol".into(),
            line_range: 0..=0,
        };

        let contents = message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor
                    .mention_set()
                    .update(cx, |mention_set, cx| mention_set.contents(false, cx))
            })
            .await
            .unwrap()
            .into_values()
            .collect::<Vec<_>>();

        {
            let [_, _, (uri, Mention::Text { content, .. })] = contents.as_slice() else {
                panic!("Unexpected mentions");
            };
            pretty_assertions::assert_eq!(content, "1");
            pretty_assertions::assert_eq!(uri, &symbol);
        }

        cx.run_until_parked();

        editor.read_with(&cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!(
                    "Lorem [@one.txt]({url_one})  Ipsum [@eight.txt]({url_eight}) [@MySymbol]({}) ",
                    symbol.to_uri(),
                )
            );
        });

        // Try to mention an "image" file that will fail to load
        cx.simulate_input("@file x.png");

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!("Lorem [@one.txt]({url_one})  Ipsum [@eight.txt]({url_eight}) [@MySymbol]({}) @file x.png", symbol.to_uri())
            );
            assert!(editor.has_visible_completions_menu());
            assert_eq!(current_completion_labels(editor), &["x.png "]);
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        // Getting the message contents fails
        message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor
                    .mention_set()
                    .update(cx, |mention_set, cx| mention_set.contents(false, cx))
            })
            .await
            .expect_err("Should fail to load x.png");

        cx.run_until_parked();

        // Mention was removed
        editor.read_with(&cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!(
                    "Lorem [@one.txt]({url_one})  Ipsum [@eight.txt]({url_eight}) [@MySymbol]({}) ",
                    symbol.to_uri()
                )
            );
        });

        // Once more
        cx.simulate_input("@file x.png");

        editor.update(&mut cx, |editor, cx| {
                    assert_eq!(
                        editor.text(cx),
                        format!("Lorem [@one.txt]({url_one})  Ipsum [@eight.txt]({url_eight}) [@MySymbol]({}) @file x.png", symbol.to_uri())
                    );
                    assert!(editor.has_visible_completions_menu());
                    assert_eq!(current_completion_labels(editor), &["x.png "]);
                });

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        // This time don't immediately get the contents, just let the confirmed completion settle
        cx.run_until_parked();

        // Mention was removed
        editor.read_with(&cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!(
                    "Lorem [@one.txt]({url_one})  Ipsum [@eight.txt]({url_eight}) [@MySymbol]({}) ",
                    symbol.to_uri()
                )
            );
        });

        // Now getting the contents succeeds, because the invalid mention was removed
        let contents = message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor
                    .mention_set()
                    .update(cx, |mention_set, cx| mention_set.contents(false, cx))
            })
            .await
            .unwrap();
        assert_eq!(contents.len(), 3);
    }

    fn fold_ranges(editor: &Editor, cx: &mut App) -> Vec<Range<Point>> {
        let snapshot = editor.buffer().read(cx).snapshot(cx);
        editor.display_map.update(cx, |display_map, cx| {
            display_map
                .snapshot(cx)
                .folds_in_range(MultiBufferOffset(0)..snapshot.len())
                .map(|fold| fold.range.to_point(&snapshot))
                .collect()
        })
    }

    fn current_completion_labels(editor: &Editor) -> Vec<String> {
        let completions = editor.current_completions().expect("Missing completions");
        completions
            .into_iter()
            .map(|completion| completion.label.text)
            .collect::<Vec<_>>()
    }

    fn current_completion_labels_with_documentation(editor: &Editor) -> Vec<(String, String)> {
        let completions = editor.current_completions().expect("Missing completions");
        completions
            .into_iter()
            .map(|completion| {
                (
                    completion.label.text,
                    completion
                        .documentation
                        .map(|d| d.text().to_string())
                        .unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>()
    }

    #[gpui::test]
    async fn test_large_file_mention_fallback(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());

        // Create a large file that exceeds AUTO_OUTLINE_SIZE
        // Using plain text without a configured language, so no outline is available
        const LINE: &str = "This is a line of text in the file\n";
        let large_content = LINE.repeat(2 * (outline::AUTO_OUTLINE_SIZE / LINE.len()));
        assert!(large_content.len() > outline::AUTO_OUTLINE_SIZE);

        // Create a small file that doesn't exceed AUTO_OUTLINE_SIZE
        let small_content = "fn small_function() { /* small */ }\n";
        assert!(small_content.len() < outline::AUTO_OUTLINE_SIZE);

        fs.insert_tree(
            "/project",
            json!({
                "large_file.txt": large_content.clone(),
                "small_file.txt": small_content,
            }),
        )
        .await;

        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = Some(cx.new(|cx| ThreadStore::new(cx)));

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                let editor = MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                );
                // Enable embedded context so files are actually included
                editor
                    .session_capabilities
                    .write()
                    .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true));
                editor
            })
        });

        // Test large file mention
        // Get the absolute path using the project's worktree
        let large_file_abs_path = project.read_with(cx, |project, cx| {
            let worktree = project.worktrees(cx).next().unwrap();
            let worktree_root = worktree.read(cx).abs_path();
            worktree_root.join("large_file.txt")
        });
        let large_file_task = message_editor.update(cx, |editor, cx| {
            editor.mention_set().update(cx, |set, cx| {
                set.confirm_mention_for_file(large_file_abs_path, true, cx)
            })
        });

        let large_file_mention = large_file_task.await.unwrap();
        match large_file_mention {
            Mention::Text { content, .. } => {
                // Should contain some of the content but not all of it
                assert!(
                    content.contains(LINE),
                    "Should contain some of the file content"
                );
                assert!(
                    !content.contains(&LINE.repeat(100)),
                    "Should not contain the full file"
                );
                // Should be much smaller than original
                assert!(
                    content.len() < large_content.len() / 10,
                    "Should be significantly truncated"
                );
            }
            _ => panic!("Expected Text mention for large file"),
        }

        // Test small file mention
        // Get the absolute path using the project's worktree
        let small_file_abs_path = project.read_with(cx, |project, cx| {
            let worktree = project.worktrees(cx).next().unwrap();
            let worktree_root = worktree.read(cx).abs_path();
            worktree_root.join("small_file.txt")
        });
        let small_file_task = message_editor.update(cx, |editor, cx| {
            editor.mention_set().update(cx, |set, cx| {
                set.confirm_mention_for_file(small_file_abs_path, true, cx)
            })
        });

        let small_file_mention = small_file_task.await.unwrap();
        match small_file_mention {
            Mention::Text { content, .. } => {
                // Should contain the full actual content
                assert_eq!(content, small_content);
            }
            _ => panic!("Expected Text mention for small file"),
        }
    }

    #[gpui::test]
    async fn test_insert_thread_summary(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(LanguageModelRegistry::test);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file": ""})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = Some(cx.new(|cx| ThreadStore::new(cx)));

        let session_id = acp::SessionId::new("thread-123");
        let title = Some("Previous Conversation".into());

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                let mut editor = MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                );
                editor.insert_thread_summary(session_id.clone(), title.clone(), window, cx);
                editor
            })
        });

        // Construct expected values for verification
        let expected_uri = MentionUri::Thread {
            id: session_id.clone(),
            name: title.as_ref().unwrap().to_string(),
        };
        let expected_title = title.as_ref().unwrap();
        let expected_link = format!("[@{}]({})", expected_title, expected_uri.to_uri());

        message_editor.read_with(cx, |editor, cx| {
            let text = editor.text(cx);

            assert!(
                text.contains(&expected_link),
                "Expected editor text to contain thread mention link.\nExpected substring: {}\nActual text: {}",
                expected_link,
                text
            );

            let mentions = editor.mention_set().read(cx).mentions();
            assert_eq!(
                mentions.len(),
                1,
                "Expected exactly one mention after inserting thread summary"
            );

            assert!(
                mentions.contains(&expected_uri),
                "Expected mentions to contain the thread URI"
            );
        });
    }

    #[gpui::test]
    async fn test_insert_thread_summary_skipped_for_external_agents(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(LanguageModelRegistry::test);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file": ""})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = None;

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                let mut editor = MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                );
                editor.insert_thread_summary(
                    acp::SessionId::new("thread-123"),
                    Some("Previous Conversation".into()),
                    window,
                    cx,
                );
                editor
            })
        });

        message_editor.read_with(cx, |editor, cx| {
            assert!(
                editor.text(cx).is_empty(),
                "Expected thread summary to be skipped for external agents"
            );
            assert!(
                editor.mention_set().read(cx).mentions().is_empty(),
                "Expected no mentions when thread summary is skipped"
            );
        });
    }

    #[gpui::test]
    async fn test_thread_mode_hidden_when_disabled(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file": ""})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = None;

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            })
        });

        message_editor.update(cx, |editor, _cx| {
            editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true));
        });

        let supported_modes = {
            let app = cx.app.borrow();
            let _ = &app;
            message_editor
                .read(&app)
                .session_capabilities
                .read()
                .supported_modes(false)
        };

        assert!(
            !supported_modes.contains(&PromptContextType::Thread),
            "Expected thread mode to be hidden when thread mentions are disabled"
        );
    }

    #[gpui::test]
    async fn test_thread_mode_visible_when_enabled(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file": ""})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = Some(cx.new(|cx| ThreadStore::new(cx)));

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            })
        });

        message_editor.update(cx, |editor, _cx| {
            editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true));
        });

        let supported_modes = {
            let app = cx.app.borrow();
            let _ = &app;
            message_editor
                .read(&app)
                .session_capabilities
                .read()
                .supported_modes(true)
        };

        assert!(
            supported_modes.contains(&PromptContextType::Thread),
            "Expected thread mode to be visible when enabled"
        );
    }

    #[gpui::test]
    async fn test_whitespace_trimming(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file.rs": "fn main() {}"}))
            .await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = Some(cx.new(|cx| ThreadStore::new(cx)));

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            })
        });
        let editor = message_editor.update(cx, |message_editor, _| message_editor.editor.clone());

        cx.run_until_parked();

        editor.update_in(cx, |editor, window, cx| {
            editor.set_text("  \u{A0}してhello world  ", window, cx);
        });

        let (content, _) = message_editor
            .update(cx, |message_editor, cx| message_editor.contents(false, cx))
            .await
            .unwrap();

        assert_eq!(content, vec!["してhello world".into()]);
    }

    #[gpui::test]
    async fn test_editor_respects_embedded_context_capability(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());

        let file_content = "fn main() { println!(\"Hello, world!\"); }\n";

        fs.insert_tree(
            "/project",
            json!({
                "src": {
                    "main.rs": file_content,
                }
            }),
        )
        .await;

        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let thread_store = Some(cx.new(|cx| ThreadStore::new(cx)));

        let (message_editor, editor) = workspace.update_in(cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            let editor = message_editor.read(cx).editor().clone();
            (message_editor, editor)
        });

        cx.simulate_input("What is in @file main");

        editor.update_in(cx, |editor, window, cx| {
            assert!(editor.has_visible_completions_menu());
            assert_eq!(editor.text(cx), "What is in @file main");
            editor.confirm_completion(&editor::actions::ConfirmCompletion::default(), window, cx);
        });

        let content = message_editor
            .update(cx, |editor, cx| editor.contents(false, cx))
            .await
            .unwrap()
            .0;

        let main_rs_uri = if cfg!(windows) {
            "file:///C:/project/src/main.rs"
        } else {
            "file:///project/src/main.rs"
        };

        // When embedded context is `false` we should get a resource link
        pretty_assertions::assert_eq!(
            content,
            vec![
                "What is in ".into(),
                acp::ContentBlock::ResourceLink(acp::ResourceLink::new("main.rs", main_rs_uri))
            ]
        );

        message_editor.update(cx, |editor, _cx| {
            editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true))
        });

        let content = message_editor
            .update(cx, |editor, cx| editor.contents(false, cx))
            .await
            .unwrap()
            .0;

        // When embedded context is `true` we should get a resource
        pretty_assertions::assert_eq!(
            content,
            vec![
                "What is in ".into(),
                acp::ContentBlock::Resource(acp::EmbeddedResource::new(
                    acp::EmbeddedResourceResource::TextResourceContents(
                        acp::TextResourceContents::new(file_content, main_rs_uri)
                    )
                ))
            ]
        );
    }

    #[gpui::test]
    async fn test_autoscroll_after_insert_selections(cx: &mut TestAppContext) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/dir"),
                json!({
                    "test.txt": "line1\nline2\nline3\nline4\nline5\n",
                }),
            )
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let worktree = project.update(cx, |project, cx| {
            let mut worktrees = project.worktrees(cx).collect::<Vec<_>>();
            assert_eq!(worktrees.len(), 1);
            worktrees.pop().unwrap()
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        // Open a regular editor with the created file, and select a portion of
        // the text that will be used for the selections that are meant to be
        // inserted in the agent panel.
        let editor = workspace
            .update_in(&mut cx, |workspace, window, cx| {
                workspace.open_path(
                    ProjectPath {
                        worktree_id,
                        path: rel_path("test.txt").into(),
                    },
                    None,
                    false,
                    window,
                    cx,
                )
            })
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.change_selections(Default::default(), window, cx, |selections| {
                selections.select_ranges([Point::new(0, 0)..Point::new(0, 5)]);
            });
        });

        let thread_store = Some(cx.new(|cx| ThreadStore::new(cx)));

        // Create a new `MessageEditor`. The `EditorMode::full()` has to be used
        // to ensure we have a fixed viewport, so we can eventually actually
        // place the cursor outside of the visible area.
        let message_editor = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    thread_store.clone(),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::full(),
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });

            message_editor
        });

        message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.editor.update(cx, |editor, cx| {
                // Update the Agent Panel's Message Editor text to have 100
                // lines, ensuring that the cursor is set at line 90 and that we
                // then scroll all the way to the top, so the cursor's position
                // remains off screen.
                let mut lines = String::new();
                for _ in 1..=100 {
                    lines.push_str(&"Another line in the agent panel's message editor\n");
                }
                editor.set_text(lines.as_str(), window, cx);
                editor.change_selections(Default::default(), window, cx, |selections| {
                    selections.select_ranges([Point::new(90, 0)..Point::new(90, 0)]);
                });
                editor.set_scroll_position(gpui::Point::new(0., 0.), window, cx);
            });
        });

        cx.run_until_parked();

        // Before proceeding, let's assert that the cursor is indeed off screen,
        // otherwise the rest of the test doesn't make sense.
        message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.editor.update(cx, |editor, cx| {
                let snapshot = editor.snapshot(window, cx);
                let cursor_row = editor.selections.newest::<Point>(&snapshot).head().row;
                let scroll_top = snapshot.scroll_position().y as u32;
                let visible_lines = editor.visible_line_count().unwrap() as u32;
                let visible_range = scroll_top..(scroll_top + visible_lines);

                assert!(!visible_range.contains(&cursor_row));
            })
        });

        let text_editor_selection = editor.update(&mut cx, |editor, cx| {
            let multibuffer = editor.buffer().read(cx);
            let buffer = multibuffer.as_singleton().unwrap();
            let buffer_snapshot = buffer.read(cx).snapshot();
            let start = buffer_snapshot.anchor_before(0);
            let end = buffer_snapshot.anchor_after(5);
            AgentContextSelection::Editor(vec![(buffer, start..end)])
        });

        message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.insert_selections(text_editor_selection, window, cx);
        });

        cx.run_until_parked();

        message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.editor.update(cx, |editor, cx| {
                let snapshot = editor.snapshot(window, cx);
                let cursor_row = editor.selections.newest::<Point>(&snapshot).head().row;
                let scroll_top = snapshot.scroll_position().y as u32;
                let visible_lines = editor.visible_line_count().unwrap() as u32;
                let visible_range = scroll_top..(scroll_top + visible_lines);

                assert!(visible_range.contains(&cursor_row));
            })
        });
    }

    #[gpui::test]
    async fn test_insert_context_with_multibyte_characters(cx: &mut TestAppContext) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/dir"), json!({}))
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let thread_store = cx.new(|cx| ThreadStore::new(cx));

        let (message_editor, editor) = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    Some(thread_store.clone()),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            let editor = message_editor.read(cx).editor().clone();
            (message_editor, editor)
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.set_text("😄😄", window, cx);
        });

        cx.run_until_parked();

        message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.insert_context_type("file", window, cx);
        });

        cx.run_until_parked();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), "😄😄@file");
        });
    }

    #[gpui::test]
    async fn test_paste_mention_link_with_multiple_selections(cx: &mut TestAppContext) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/project"), json!({"file.txt": "content"}))
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/project").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let thread_store = cx.new(|cx| ThreadStore::new(cx));

        let (message_editor, editor) = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    Some(thread_store),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            let editor = message_editor.read(cx).editor().clone();
            (message_editor, editor)
        });

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.set_text(
                "AAAAAAAAAAAAAAAAAAAAAAAAA     AAAAAAAAAAAAAAAAAAAAAAAAA",
                window,
                cx,
            );
        });

        cx.run_until_parked();

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                s.select_ranges([
                    MultiBufferOffset(0)..MultiBufferOffset(25), // First selection (large)
                    MultiBufferOffset(30)..MultiBufferOffset(55), // Second selection (newest)
                ]);
            });
        });

        let mention_link = "[@f](file:///test.txt)";
        cx.write_to_clipboard(ClipboardItem::new_string(mention_link.into()));

        message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.paste(&Paste, window, cx);
        });

        let text = editor.update(&mut cx, |editor, cx| editor.text(cx));
        assert!(
            text.contains("[@f](file:///test.txt)"),
            "Expected mention link to be pasted, got: {}",
            text
        );
    }

    #[gpui::test]
    async fn test_copy_with_selection_mentions_serializes_links(cx: &mut TestAppContext) {
        init_test(cx);

        let (source_message_editor, _source_editor, mut cx) = setup_paste_test_message_editor(
            json!({"file.rs": "line 1\nline 2\nline 3\nline 4\n"}),
            cx,
        )
        .await;

        let workspace = source_message_editor.read_with(&cx, |message_editor, _| {
            message_editor.workspace.upgrade().expect("workspace")
        });
        let project = workspace.read_with(&cx, |workspace, _| workspace.project().clone());

        let source_text = "selection needs work\nselection looks fine";
        let first_range = 0..9;
        let second_start = "selection needs work\n".len();
        let second_range = second_start..(second_start + "selection".len());
        let first_uri = MentionUri::Selection {
            abs_path: Some(path!("/project/file.rs").into()),
            line_range: 0..=1,
            column: None,
        };
        let second_uri = MentionUri::Selection {
            abs_path: Some(path!("/project/file.rs").into()),
            line_range: 2..=3,
            column: None,
        };

        source_message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.set_text(source_text, window, cx);

            let snapshot = message_editor
                .editor
                .read(cx)
                .buffer()
                .read(cx)
                .snapshot(cx);
            for (range, uri, content) in [
                (
                    first_range.clone(),
                    first_uri.clone(),
                    "line 1\nline 2\n".to_string(),
                ),
                (
                    second_range.clone(),
                    second_uri.clone(),
                    "line 3\nline 4\n".to_string(),
                ),
            ] {
                let Some((crease_id, tx, _crease_entity)) = insert_crease_for_mention(
                    snapshot
                        .anchor_to_buffer_anchor(
                            snapshot.anchor_before(MultiBufferOffset(range.start)),
                        )
                        .expect("selection mention anchor should map to a buffer")
                        .0,
                    range.len(),
                    uri.name().into(),
                    uri.icon_path(cx),
                    uri.tooltip_text(),
                    Some(uri.clone()),
                    Some(message_editor.workspace.clone()),
                    None,
                    message_editor.editor.clone(),
                    window,
                    cx,
                ) else {
                    panic!("expected mention crease insertion");
                };
                drop(tx);

                message_editor.mention_set.update(cx, |mention_set, cx| {
                    mention_set.insert_mention(
                        crease_id,
                        uri,
                        Task::ready(Ok(Mention::Text {
                            content,
                            tracked_buffers: Vec::new(),
                        }))
                        .shared(),
                        None,
                        cx,
                    );
                });
            }

            let buffer_len = snapshot.len();
            message_editor.editor.update(cx, |editor, cx| {
                editor.change_selections(Default::default(), window, cx, |selections| {
                    selections.select_ranges([MultiBufferOffset(0)..buffer_len]);
                });
            });
        });

        let copied_text = source_message_editor.update(&mut cx, |message_editor, cx| {
            message_editor
                .serialize_selection_with_mentions(false, cx)
                .map(|(text, _)| text)
                .expect("selection mentions should serialize")
        });
        let expected_text = format!(
            "{} needs work\n{} looks fine",
            first_uri.as_link(),
            second_uri.as_link()
        );
        assert_eq!(copied_text, expected_text);

        let target_message_editor = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let thread_store = cx.new(|cx| ThreadStore::new(cx));
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    Some(thread_store),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            message_editor
        });

        cx.write_to_clipboard(ClipboardItem::new_string(copied_text));
        target_message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.paste(&Paste, window, cx);
        });
        cx.run_until_parked();

        let target_text = target_message_editor.read_with(&cx, |message_editor, cx| {
            message_editor.editor.read(cx).text(cx)
        });
        assert_eq!(target_text, expected_text);

        let contents = mention_contents(&target_message_editor, &mut cx).await;
        assert_eq!(contents.len(), 2);
        assert!(contents.iter().any(|(uri, _)| uri == &first_uri));
        assert!(contents.iter().any(|(uri, _)| uri == &second_uri));
    }

    struct SelectionMentionFixture {
        message_editor: Entity<MessageEditor>,
        first_uri: MentionUri,
        first_range: Range<usize>,
        second_uri: MentionUri,
        second_range: Range<usize>,
        buffer_len: MultiBufferOffset,
    }

    async fn setup_selection_mention_fixture(
        cx: &mut TestAppContext,
    ) -> (SelectionMentionFixture, VisualTestContext) {
        let (message_editor, _source_editor, mut cx) = setup_paste_test_message_editor(
            json!({"file.rs": "line 1\nline 2\nline 3\nline 4\n"}),
            cx,
        )
        .await;

        let source_text = "selection needs work\nselection looks fine";
        let first_range = 0..9;
        let second_start = "selection needs work\n".len();
        let second_range = second_start..(second_start + "selection".len());
        let first_uri = MentionUri::Selection {
            abs_path: Some(path!("/project/file.rs").into()),
            line_range: 0..=1,
            column: None,
        };
        let second_uri = MentionUri::Selection {
            abs_path: Some(path!("/project/file.rs").into()),
            line_range: 2..=3,
            column: None,
        };

        let buffer_len = message_editor.update_in(&mut cx, |message_editor, window, cx| {
            message_editor.set_text(source_text, window, cx);

            let snapshot = message_editor
                .editor
                .read(cx)
                .buffer()
                .read(cx)
                .snapshot(cx);
            for (range, uri, content) in [
                (
                    first_range.clone(),
                    first_uri.clone(),
                    "line 1\nline 2\n".to_string(),
                ),
                (
                    second_range.clone(),
                    second_uri.clone(),
                    "line 3\nline 4\n".to_string(),
                ),
            ] {
                let Some((crease_id, tx, _crease_entity)) = insert_crease_for_mention(
                    snapshot
                        .anchor_to_buffer_anchor(
                            snapshot.anchor_before(MultiBufferOffset(range.start)),
                        )
                        .expect("selection mention anchor should map to a buffer")
                        .0,
                    range.len(),
                    uri.name().into(),
                    uri.icon_path(cx),
                    uri.tooltip_text(),
                    Some(uri.clone()),
                    Some(message_editor.workspace.clone()),
                    None,
                    message_editor.editor.clone(),
                    window,
                    cx,
                ) else {
                    panic!("expected mention crease insertion");
                };
                drop(tx);

                message_editor.mention_set.update(cx, |mention_set, cx| {
                    mention_set.insert_mention(
                        crease_id,
                        uri,
                        Task::ready(Ok(Mention::Text {
                            content,
                            tracked_buffers: Vec::new(),
                        }))
                        .shared(),
                        None,
                        cx,
                    );
                });
            }

            snapshot.len()
        });

        (
            SelectionMentionFixture {
                message_editor,
                first_uri,
                first_range,
                second_uri,
                second_range,
                buffer_len,
            },
            cx,
        )
    }

    #[gpui::test]
    async fn test_serialized_copy_text_selection_covers_only_mention(cx: &mut TestAppContext) {
        init_test(cx);

        let (fixture, mut cx) = setup_selection_mention_fixture(cx).await;

        fixture
            .message_editor
            .update_in(&mut cx, |message_editor, window, cx| {
                let range = fixture.first_range.clone();
                message_editor.editor.update(cx, |editor, cx| {
                    editor.change_selections(Default::default(), window, cx, |selections| {
                        selections.select_ranges([
                            MultiBufferOffset(range.start)..MultiBufferOffset(range.end)
                        ]);
                    });
                });
            });

        let copied = fixture
            .message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor
                    .serialize_selection_with_mentions(false, cx)
                    .map(|(text, _)| text)
            });

        assert_eq!(copied, Some(fixture.first_uri.as_link().to_string()));
    }

    #[gpui::test]
    async fn test_serialized_copy_text_returns_none_when_mentions_outside_selection(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let (fixture, mut cx) = setup_selection_mention_fixture(cx).await;

        let between_start = fixture.first_range.end;
        let between_end = fixture.second_range.start - 1;

        fixture
            .message_editor
            .update_in(&mut cx, |message_editor, window, cx| {
                message_editor.editor.update(cx, |editor, cx| {
                    editor.change_selections(Default::default(), window, cx, |selections| {
                        selections.select_ranges([
                            MultiBufferOffset(between_start)..MultiBufferOffset(between_end)
                        ]);
                    });
                });
            });

        let copied = fixture
            .message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor
                    .serialize_selection_with_mentions(false, cx)
                    .map(|(text, _)| text)
            });

        assert_eq!(copied, None);
    }

    #[gpui::test]
    async fn test_draft_content_blocks_snapshot_preserves_selection_mentions(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let (fixture, mut cx) = setup_selection_mention_fixture(cx).await;

        let blocks = fixture.message_editor.update(&mut cx, |editor, cx| {
            editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true));
            editor
                .draft_content_blocks_snapshot(cx)
                .expect("valid draft snapshot")
        });

        // Each selection mention must round-trip as a `Resource` block carrying
        // its URI and content, not as a `Text` block containing the fold
        // placeholder string.
        let resource_uris: Vec<&str> =
            blocks
                .iter()
                .filter_map(|block| match block {
                    acp::ContentBlock::Resource(acp::EmbeddedResource {
                        resource:
                            acp::EmbeddedResourceResource::TextResourceContents(
                                acp::TextResourceContents { uri, .. },
                            ),
                        ..
                    }) => Some(uri.as_str()),
                    _ => None,
                })
                .collect();
        assert_eq!(
            resource_uris.len(),
            2,
            "snapshot should emit one Resource block per selection mention; got {blocks:#?}"
        );
        assert!(resource_uris.contains(&fixture.first_uri.to_uri().to_string().as_str()));
        for block in &blocks {
            if let acp::ContentBlock::Text(text) = block {
                assert!(
                    !text.text.split_whitespace().any(|word| word == "selection"),
                    "text block must not contain bare fold placeholder: {:?}",
                    text.text
                );
            }
        }
    }

    #[gpui::test]
    async fn test_cut_with_selection_mentions_serializes_and_removes(cx: &mut TestAppContext) {
        init_test(cx);

        let (fixture, mut cx) = setup_selection_mention_fixture(cx).await;

        let buffer_len = fixture.buffer_len;
        fixture
            .message_editor
            .update_in(&mut cx, |message_editor, window, cx| {
                message_editor.editor.update(cx, |editor, cx| {
                    editor.change_selections(Default::default(), window, cx, |selections| {
                        selections.select_ranges([MultiBufferOffset(0)..buffer_len]);
                    });
                });
                message_editor.cut(&Cut, window, cx);
            });

        let expected_text = format!(
            "{} needs work\n{} looks fine",
            fixture.first_uri.as_link(),
            fixture.second_uri.as_link()
        );

        let clipboard_text = cx
            .read_from_clipboard()
            .and_then(|item| match item.entries().first().cloned() {
                Some(ClipboardEntry::String(entry)) => Some(entry.text().to_string()),
                _ => None,
            })
            .expect("cut should write serialized text to clipboard");
        assert_eq!(clipboard_text, expected_text);

        let remaining_text = fixture.message_editor.read_with(&cx, |message_editor, cx| {
            message_editor.editor.read(cx).text(cx)
        });
        assert_eq!(remaining_text, "");
    }

    #[gpui::test]
    async fn test_cut_with_empty_cursor_on_mention_line_removes_whole_line(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let (fixture, mut cx) = setup_selection_mention_fixture(cx).await;

        let cursor_offset = MultiBufferOffset(fixture.first_range.end + 4);
        fixture
            .message_editor
            .update_in(&mut cx, |message_editor, window, cx| {
                message_editor.editor.update(cx, |editor, cx| {
                    editor.change_selections(Default::default(), window, cx, |selections| {
                        selections.select_ranges([cursor_offset..cursor_offset]);
                    });
                });
                message_editor.cut(&Cut, window, cx);
            });

        let clipboard_text = cx
            .read_from_clipboard()
            .and_then(|item| match item.entries().first().cloned() {
                Some(ClipboardEntry::String(entry)) => Some(entry.text().to_string()),
                _ => None,
            })
            .expect("cut should write serialized text to clipboard");
        assert_eq!(
            clipboard_text,
            format!("{} needs work\n", fixture.first_uri.as_link())
        );

        let remaining_text = fixture.message_editor.read_with(&cx, |message_editor, cx| {
            message_editor.editor.read(cx).text(cx)
        });
        assert_eq!(remaining_text, "selection looks fine");
    }

    #[gpui::test]
    async fn test_serialized_cut_text_returns_none_when_mentions_outside_selection(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let (fixture, mut cx) = setup_selection_mention_fixture(cx).await;

        let between_start = fixture.first_range.end;
        let between_end = fixture.second_range.start - 1;
        fixture
            .message_editor
            .update_in(&mut cx, |message_editor, window, cx| {
                message_editor.editor.update(cx, |editor, cx| {
                    editor.change_selections(Default::default(), window, cx, |selections| {
                        selections.select_ranges([
                            MultiBufferOffset(between_start)..MultiBufferOffset(between_end)
                        ]);
                    });
                });
            });

        let result = fixture
            .message_editor
            .update(&mut cx, |message_editor, cx| {
                message_editor.serialize_selection_with_mentions(true, cx)
            });

        assert!(
            result.is_none(),
            "serialize_selection_with_mentions should return None so the default editor cut runs"
        );
    }

    #[gpui::test]
    async fn test_paste_mention_link_with_completion_trigger_does_not_panic(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/project"), json!({"file.txt": "content"}))
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/project").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let thread_store = cx.new(|cx| ThreadStore::new(cx));

        let (_message_editor, editor) = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    Some(thread_store),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            let editor = message_editor.read(cx).editor().clone();
            (message_editor, editor)
        });

        cx.simulate_input("@");

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), "@");
            assert!(editor.has_visible_completions_menu());
        });

        cx.write_to_clipboard(ClipboardItem::new_string("[@f](file:///test.txt) @".into()));
        cx.dispatch_action(Paste);

        editor.update(&mut cx, |editor, cx| {
            assert!(editor.text(cx).contains("[@f](file:///test.txt)"));
        });
    }

    #[gpui::test]
    async fn test_paste_external_file_path_inserts_file_mention(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, editor, mut cx) =
            setup_paste_test_message_editor(json!({"file.txt": "content"}), cx).await;
        paste_external_paths(
            &message_editor,
            vec![PathBuf::from(path!("/project/file.txt"))],
            &mut cx,
        );

        let expected_uri = MentionUri::File {
            abs_path: path!("/project/file.txt").into(),
        }
        .to_uri()
        .to_string();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), format!("[@file.txt]({expected_uri}) "));
        });

        let contents = mention_contents(&message_editor, &mut cx).await;

        let [(uri, Mention::Text { content, .. })] = contents.as_slice() else {
            panic!("Unexpected mentions");
        };
        assert_eq!(content, "content");
        assert_eq!(
            uri,
            &MentionUri::File {
                abs_path: path!("/project/file.txt").into(),
            }
        );
    }

    #[gpui::test]
    async fn test_paste_external_directory_path_inserts_directory_mention(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, editor, mut cx) = setup_paste_test_message_editor(
            json!({
                "src": {
                    "main.rs": "fn main() {}\n",
                }
            }),
            cx,
        )
        .await;
        paste_external_paths(
            &message_editor,
            vec![PathBuf::from(path!("/project/src"))],
            &mut cx,
        );

        let expected_uri = MentionUri::Directory {
            abs_path: path!("/project/src").into(),
        }
        .to_uri()
        .to_string();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(editor.text(cx), format!("[@src]({expected_uri}) "));
        });

        let contents = mention_contents(&message_editor, &mut cx).await;

        let [(uri, Mention::Link)] = contents.as_slice() else {
            panic!("Unexpected mentions");
        };
        assert_eq!(
            uri,
            &MentionUri::Directory {
                abs_path: path!("/project/src").into(),
            }
        );
    }

    #[gpui::test]
    async fn test_paste_external_file_path_inserts_at_cursor(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, editor, mut cx) =
            setup_paste_test_message_editor(json!({"file.txt": "content"}), cx).await;

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.set_text("Hello world", window, cx);
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(6)..MultiBufferOffset(6)]);
            });
        });

        paste_external_paths(
            &message_editor,
            vec![PathBuf::from(path!("/project/file.txt"))],
            &mut cx,
        );

        let expected_uri = MentionUri::File {
            abs_path: path!("/project/file.txt").into(),
        }
        .to_uri()
        .to_string();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!("Hello [@file.txt]({expected_uri}) world")
            );
        });
    }

    #[gpui::test]
    async fn test_dragged_file_path_inserts_at_cursor(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, editor, mut cx) =
            setup_paste_test_message_editor(json!({"file.txt": "content"}), cx).await;

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.set_text("Hello world", window, cx);
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(6)..MultiBufferOffset(6)]);
            });
        });

        insert_dragged_project_paths(&message_editor, vec!["file.txt"], &mut cx);

        let expected_uri = MentionUri::File {
            abs_path: path!("/project/file.txt").into(),
        }
        .to_uri()
        .to_string();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!("Hello [@file.txt]({expected_uri}) world")
            );
        });

        let contents = mention_contents(&message_editor, &mut cx).await;

        let [(uri, Mention::Text { content, .. })] = contents.as_slice() else {
            panic!("Unexpected mentions");
        };
        assert_eq!(content, "content");
        assert_eq!(
            uri,
            &MentionUri::File {
                abs_path: path!("/project/file.txt").into(),
            }
        );
    }

    #[gpui::test]
    async fn test_dragged_file_paths_insert_in_order_at_cursor(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, editor, mut cx) = setup_paste_test_message_editor(
            json!({
                "one.txt": "one",
                "two.txt": "two",
            }),
            cx,
        )
        .await;

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.set_text("Hello world", window, cx);
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |selections| {
                selections.select_ranges([MultiBufferOffset(6)..MultiBufferOffset(6)]);
            });
        });

        insert_dragged_project_paths(&message_editor, vec!["one.txt", "two.txt"], &mut cx);

        let first_uri = MentionUri::File {
            abs_path: path!("/project/one.txt").into(),
        }
        .to_uri()
        .to_string();
        let second_uri = MentionUri::File {
            abs_path: path!("/project/two.txt").into(),
        }
        .to_uri()
        .to_string();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!("Hello [@one.txt]({first_uri}) [@two.txt]({second_uri}) world")
            );
        });
    }

    #[gpui::test]
    async fn test_paste_mixed_external_image_without_extension_and_file_path(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (message_editor, editor, mut cx) =
            setup_paste_test_message_editor(json!({"file.txt": "content"}), cx).await;

        message_editor.update(&mut cx, |message_editor, _cx| {
            message_editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().image(true));
        });

        let temporary_image_path = write_test_png_file(None);
        paste_external_paths(
            &message_editor,
            vec![
                temporary_image_path.clone(),
                PathBuf::from(path!("/project/file.txt")),
            ],
            &mut cx,
        );

        let image_name = temporary_image_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Image")
            .to_string();
        std::fs::remove_file(&temporary_image_path).expect("remove temp png");

        let expected_file_uri = MentionUri::File {
            abs_path: path!("/project/file.txt").into(),
        }
        .to_uri()
        .to_string();
        let expected_image_uri = MentionUri::PastedImage {
            name: image_name.clone(),
        }
        .to_uri()
        .to_string();

        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                format!("[@{image_name}]({expected_image_uri}) [@file.txt]({expected_file_uri}) ")
            );
        });

        let contents = mention_contents(&message_editor, &mut cx).await;

        assert_eq!(contents.len(), 2);
        assert!(contents.iter().any(|(uri, mention)| {
            matches!(uri, MentionUri::PastedImage { .. }) && matches!(mention, Mention::Image(_))
        }));
        assert!(contents.iter().any(|(uri, mention)| {
            *uri == MentionUri::File {
                abs_path: path!("/project/file.txt").into(),
            } && matches!(
                mention,
                Mention::Text {
                    content,
                    tracked_buffers: _,
                } if content == "content"
            )
        }));
    }

    async fn setup_paste_test_message_editor(
        project_tree: Value,
        cx: &mut TestAppContext,
    ) -> (Entity<MessageEditor>, Entity<Editor>, VisualTestContext) {
        let app_state = cx.update(AppState::test);

        cx.update(|cx| {
            editor::init(cx);
            workspace::init(app_state.clone(), cx);
        });

        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/project"), project_tree)
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/project").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(window.into(), cx);

        let thread_store = cx.new(|cx| ThreadStore::new(cx));

        let (message_editor, editor) = workspace.update_in(&mut cx, |workspace, window, cx| {
            let workspace_handle = cx.weak_entity();
            let message_editor = cx.new(|cx| {
                MessageEditor::new(
                    workspace_handle,
                    project.downgrade(),
                    Some(thread_store),
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        max_lines: None,
                        min_lines: 1,
                    },
                    window,
                    cx,
                )
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
            message_editor.read(cx).focus_handle(cx).focus(window, cx);
            let editor = message_editor.read(cx).editor().clone();
            (message_editor, editor)
        });

        (message_editor, editor, cx)
    }

    fn paste_external_paths(
        message_editor: &Entity<MessageEditor>,
        paths: Vec<PathBuf>,
        cx: &mut VisualTestContext,
    ) {
        cx.write_to_clipboard(ClipboardItem {
            entries: vec![ClipboardEntry::ExternalPaths(ExternalPaths(paths.into()))],
        });

        message_editor.update_in(cx, |message_editor, window, cx| {
            message_editor.paste(&Paste, window, cx);
        });
        cx.run_until_parked();
    }

    fn insert_dragged_project_paths(
        message_editor: &Entity<MessageEditor>,
        paths: Vec<&str>,
        cx: &mut VisualTestContext,
    ) {
        message_editor.update_in(cx, |message_editor, window, cx| {
            let workspace = message_editor
                .workspace
                .upgrade()
                .expect("message editor should keep workspace alive");
            let project = workspace.read(cx).project().clone();
            let worktree_id = project.update(cx, |project, cx| {
                let mut worktrees = project.worktrees(cx).collect::<Vec<_>>();
                assert_eq!(worktrees.len(), 1, "expected a single worktree");
                worktrees.pop().unwrap().read(cx).id()
            });

            let paths = paths
                .into_iter()
                .map(|path| ProjectPath {
                    worktree_id,
                    path: rel_path(path).into(),
                })
                .collect();

            message_editor.insert_dragged_files(paths, vec![], window, cx);
        });
        cx.run_until_parked();
    }

    async fn mention_contents(
        message_editor: &Entity<MessageEditor>,
        cx: &mut VisualTestContext,
    ) -> Vec<(MentionUri, Mention)> {
        message_editor
            .update(cx, |message_editor, cx| {
                message_editor
                    .mention_set()
                    .update(cx, |mention_set, cx| mention_set.contents(false, cx))
            })
            .await
            .unwrap()
            .into_values()
            .collect::<Vec<_>>()
    }

    fn write_test_png_file(extension: Option<&str>) -> PathBuf {
        let bytes = base64::prelude::BASE64_STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
            .expect("decode png");
        let file_name = match extension {
            Some(extension) => format!("zed-agent-ui-test-{}.{}", uuid::Uuid::new_v4(), extension),
            None => format!("zed-agent-ui-test-{}", uuid::Uuid::new_v4()),
        };
        let path = std::env::temp_dir().join(file_name);
        std::fs::write(&path, bytes).expect("write temp png");
        path
    }

    // Helper that creates a minimal MessageEditor inside a window, returning both
    // the entity and the underlying VisualTestContext so callers can drive updates.
    async fn setup_message_editor(
        cx: &mut TestAppContext,
    ) -> (Entity<MessageEditor>, &mut VisualTestContext) {
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file.txt": ""})).await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    None,
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            })
        });

        cx.run_until_parked();
        (message_editor, cx)
    }

    #[cfg(feature = "multiplayer-tools")]
    fn inline_terminal_history_view(
        message_editor: &Entity<MessageEditor>,
        cx: &mut VisualTestContext,
    ) -> Entity<terminal_view::TerminalView> {
        use settings::Settings as _;
        cx.update(|window, cx| {
            let workspace = message_editor
                .read(cx)
                .workspace
                .upgrade()
                .expect("workspace");
            let project = workspace.read(cx).project().clone();
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
        })
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_inline_context_projection_is_conversation_bound_and_rebases_queued_append(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::{
            InlineTerminalContext, InlineTerminalDraftPosition, InlineTerminalPromptReference,
            TerminalThreadMetadata, TerminalThreadMetadataStore,
        };
        use crate::thread_metadata_store::{ThreadId, WorktreePaths};
        init_test(cx);
        cx.update(|cx| TerminalThreadMetadataStore::init_global(cx));
        let (message_editor, cx) = setup_message_editor(cx).await;
        let terminal_id = crate::TerminalId::new();
        let thread_id = ThreadId::new();
        let session_id = acp::SessionId::new("inline-context-session");
        let reference = InlineTerminalPromptReference {
            occurrence_id: None,

            terminal_id,
            thread_id,
            session_id: session_id.clone(),
            position: InlineTerminalDraftPosition::new("AλB", 3).expect("Unicode source boundary"),
        };
        let resource = reference
            .content("actual retained fixture output".into())
            .expect("resource");
        let content = vec!["Aλ".into(), resource.clone(), "B".into()];
        message_editor.update_in(cx, |editor, window, cx| {
            let store = TerminalThreadMetadataStore::global(cx);
            store
                .update(cx, |store, cx| {
                    store.save_inline(
                        TerminalThreadMetadata {
                            terminal_id,
                            title: "Fixture terminal".into(),
                            custom_title: None,
                            created_at: chrono::Utc::now(),
                            worktree_paths: WorktreePaths::from_folder_paths(
                                &workspace::PathList::new(&[PathBuf::from("/project")]),
                            ),
                            remote_connection: None,
                            working_directory: None,
                            inline_context: Some(InlineTerminalContext {
                                occurrence_id: None,
                                previous_occurrences: Vec::new(),

                                thread_id,
                                session_id: Some(session_id.clone()),
                                restored_draft_source_session_id: None,
                                preceding_entry_count: 0,
                                message_client_id: None,
                                is_draft: true,
                                presentation_order: None,
                                draft_position: Some(reference.position.clone()),
                                collapsed: false,
                                output: "actual retained fixture output".into(),
                                snapshot: None,
                            }),
                        },
                        cx,
                    )
                })
                .expect("canonical history admission");
            editor.set_inline_terminal_scope(ThreadId::new(), session_id.clone());
            assert!(
                editor
                    .validate_inline_terminal_content(&resource, cx)
                    .is_err()
            );
            editor.set_message(content.clone(), window, cx);
            assert!(
                editor.text(cx).contains("zed:///agent/terminal-selection"),
                "foreign conversation must retain a visible context mention"
            );
            editor.set_inline_terminal_scope(thread_id, session_id.clone());
            assert!(
                editor
                    .validate_inline_terminal_content(&resource, cx)
                    .is_ok()
            );
            editor.set_message(content.clone(), window, cx);
            assert_eq!(editor.text(cx), "AλB");
            editor.set_text("X", window, cx);
            editor
                .append_message(content.clone(), Some("\n\n"), window, cx)
                .expect("append must preserve pending content");
            assert_eq!(editor.text(cx), "X\n\nAλB");
            let metadata = store
                .read(cx)
                .entry(terminal_id)
                .expect("retained metadata");
            assert_eq!(
                metadata
                    .inline_context
                    .as_ref()
                    .expect("inline context")
                    .draft_position
                    .as_ref()
                    .expect("rebased position")
                    .byte_offset_for_source("X\n\nAλB")
                    .expect("new source identity"),
                6
            );
            editor.set_message(vec!["changed".into(), resource], window, cx);
            assert!(
                editor.text(cx).contains("zed:///agent/terminal-selection"),
                "invalid source identity must preserve context instead of hiding it"
            );
            editor.set_text("PRESERVED_DRAFT", window, cx);
            store.read(cx).close_write_queue_for_test();
            assert!(
                editor
                    .append_message(content, Some("\n\n"), window, cx)
                    .is_err()
            );
            assert_eq!(
                editor.text(cx),
                "PRESERVED_DRAFT",
                "failed history preflight must precede any draft edit"
            );
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_restored_inline_terminal_output_accepts_pointer_and_keyboard_selection(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::InlineTerminalDraftPosition;
        use gpui::BorrowAppContext as _;

        init_test(cx);
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.workspace.workspace_presentation =
                        Some(workspace::WorkspacePresentation::Editor);
                });
            });
        });
        cx.update(|cx| {
            let mut bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-macos.json",
                cx,
            )
            .expect("canonical macOS bindings");
            for binding in &mut bindings {
                binding.set_meta(settings::KeybindSource::Default.meta());
            }
            cx.bind_keys(bindings);
        });
        let (message_editor, cx) = setup_message_editor(cx).await;
        let source_view = inline_terminal_history_view(&message_editor, cx);
        let source_terminal = source_view.read_with(cx, |view, _| view.terminal().clone());
        source_terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"\x1b[2J\x1b[H\x1b[31m78876\x1b[0m", cx);
        });
        cx.run_until_parked();
        let snapshot = source_terminal.read_with(cx, |terminal, _| {
            assert!(!terminal.is_pty());
            terminal.snapshot().expect("canonical frozen grid capture")
        });
        let encoded = serde_json::to_vec(&snapshot).expect("persist canonical snapshot");
        let restored_snapshot: terminal::TerminalSnapshot =
            serde_json::from_slice(&encoded).expect("reload canonical snapshot");
        restored_snapshot.validate().expect("valid retained grid");
        assert_eq!(snapshot, restored_snapshot);

        let view = inline_terminal_history_view(&message_editor, cx);
        let terminal = view.read_with(cx, |view, _| view.terminal().clone());
        terminal.update(cx, |terminal, cx| {
            terminal
                .restore_snapshot(&restored_snapshot, cx)
                .expect("restore into fresh display-only terminal");
            assert!(!terminal.is_pty());
            assert!(terminal.take_input_log().is_empty());
            assert!(terminal.take_pty_write_log().is_empty());
        });
        let workspace = message_editor.read_with(cx, |editor, _| {
            editor.workspace.upgrade().expect("workspace")
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
        });
        workspace.read_with(cx, |workspace, cx| {
            assert_eq!(
                workspace.workspace_presentation(),
                workspace::WorkspacePresentation::Editor,
                "component fixture must render its actual center pane"
            );
            let active = workspace.active_item(cx).expect("active component item");
            let item = active
                .act_as::<MessageEditorItem>(cx)
                .expect("actual active MessageEditorItem");
            assert_eq!(item.read(cx).0.entity_id(), message_editor.entity_id());
        });
        let draft = "PREFIXλSUFFIX";
        message_editor.update_in(cx, |editor, window, cx| {
            editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true));
            assert!(
                editor
                    .session_capabilities
                    .read()
                    .supports_embedded_context()
            );
            editor.set_text(draft, window, cx);
            let position = InlineTerminalDraftPosition::new(draft, "PREFIXλ".len())
                .expect("source-bound Unicode insertion position");
            editor
                .mount_inline_terminal(view.clone(), &position, cx)
                .expect("mount actual terminal in canonical character block");
        });
        view.update(cx, |view, cx| view.set_embedded_read_only(true, cx));
        cx.simulate_resize(gpui::size(gpui::px(900.), gpui::px(600.)));
        cx.run_until_parked();
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.run_until_parked();
        cx.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
        message_editor.update(cx, |editor, _cx| {
            let block_ids = editor
                .inline_terminal_blocks
                .keys()
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(block_ids.len(), 1, "one canonical inline block");
        });
        let editor_bounds = message_editor.read_with(cx, |editor, cx| {
            editor
                .editor
                .read(cx)
                .last_bounds()
                .cloned()
                .expect("mounted editor viewport")
        });
        let (pointer, before_output, before_snapshot) =
            terminal.update_in(cx, |terminal, window, cx| {
                terminal.sync(window, cx);
                assert!(!terminal.is_pty());
                let content = terminal.last_content();
                let bounds = content.terminal_bounds;
                assert!(bounds.cell_width > gpui::px(0.));
                assert!(bounds.line_height > gpui::px(0.));
                assert!(bounds.bounds.size.width > gpui::px(0.));
                assert!(bounds.bounds.size.height > gpui::px(0.));
                let mut first_line = content
                    .cells
                    .iter()
                    .filter(|cell| cell.point.line == 0 && cell.point.column < 5)
                    .collect::<Vec<_>>();
                first_line.sort_by_key(|cell| cell.point.column);
                let rendered_text = first_line
                    .iter()
                    .map(|cell| cell.cell.character())
                    .collect::<String>();
                assert_eq!(rendered_text, "78876", "measured retained glyphs");
                assert_eq!(content.cursor.point.line, 0);
                assert_eq!(content.cursor.point.column, 5);
                assert!(content.selection_text.is_none());
                let pointer = bounds.bounds.origin
                    + gpui::point(bounds.cell_width * 5., bounds.line_height * 0.5);
                assert!(bounds.bounds.contains(&pointer));
                assert!(editor_bounds.contains(&pointer));
                (
                    pointer,
                    terminal.get_content(),
                    terminal.snapshot().expect("mounted frozen grid baseline"),
                )
            });
        view.read_with(cx, |view, _| assert!(view.is_read_only()));
        let view_bounds = cx
            .debug_bounds("terminal-view")
            .expect("actual rendered terminal subtree");
        assert!(view_bounds.contains(&pointer));

        cx.write_to_clipboard(ClipboardItem::new_string("BEFORE_OUTPUT_SELECTION".into()));
        cx.simulate_click(pointer, gpui::Modifiers::default());
        cx.run_until_parked();
        let pointer_focus = cx.update(|window, cx| {
            window.draw(cx).clear(cx);
            window.focused(cx)
        });
        cx.simulate_keystrokes("shift-left shift-left shift-left");
        cx.run_until_parked();
        cx.update(|window, cx| window.draw(cx).clear(cx));
        let terminal_selection = terminal.read_with(cx, |terminal, _| {
            terminal.last_content().selection_text.clone()
        });
        cx.simulate_keystrokes("cmd-c");
        cx.run_until_parked();
        let clipboard_text =
            cx.read_from_clipboard()
                .and_then(|item| match item.entries().first().cloned() {
                    Some(ClipboardEntry::String(entry)) => Some(entry.text().to_string()),
                    _ => None,
                });
        assert_eq!(
            clipboard_text.as_deref(),
            Some("876"),
            "actual output ShiftLeft selection and Copy; pointer focus {pointer_focus:?}, terminal selection {terminal_selection:?}"
        );
        terminal.update_in(cx, |terminal, window, cx| {
            terminal.sync(window, cx);
            assert_eq!(terminal.get_content(), before_output);
            assert_eq!(
                terminal.snapshot().expect("selected retained grid"),
                before_snapshot,
                "selection must not mutate retained terminal cells"
            );
            assert!(!terminal.is_pty());
            assert!(terminal.take_input_log().is_empty());
            assert!(terminal.take_pty_write_log().is_empty());
        });
        view.read_with(cx, |view, _| assert!(view.is_read_only()));
        message_editor.read_with(cx, |editor, cx| {
            assert_eq!(
                editor.text(cx),
                draft,
                "output selection preserves the draft"
            );
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_inline_terminal_mount_rejects_changed_draft_before_mutation(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::InlineTerminalDraftPosition;
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        let view = inline_terminal_history_view(&message_editor, cx);
        let position = InlineTerminalDraftPosition::new("λ🦀", 2).expect("position");
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("later draft", window, cx);
            let snapshot = editor
                .editor
                .update(cx, |editor, cx| editor.display_snapshot(cx));
            let before = snapshot
                .blocks_in_range(
                    editor::display_map::DisplayRow(0)..editor::display_map::DisplayRow(u32::MAX),
                )
                .count();
            assert!(
                editor
                    .mount_inline_terminal(view.clone(), &position, cx)
                    .is_err()
            );
            assert_eq!(editor.text(cx), "later draft");
            let snapshot = editor
                .editor
                .update(cx, |editor, cx| editor.display_snapshot(cx));
            assert_eq!(
                snapshot
                    .blocks_in_range(
                        editor::display_map::DisplayRow(0)
                            ..editor::display_map::DisplayRow(u32::MAX)
                    )
                    .count(),
                before
            );
        });
        view.read_with(cx, |view, cx| assert!(!view.terminal().read(cx).is_pty()));
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_inline_terminal_escape_returns_suffix_without_forwarding_to_terminal(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::InlineTerminalDraftPosition;
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        let view = inline_terminal_history_view(&message_editor, cx);
        let terminal = view.read_with(cx, |view, _| view.terminal().clone());
        let workspace = message_editor.read_with(cx, |editor, _| {
            editor.workspace.upgrade().expect("workspace")
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(
                    Box::new(cx.new(|_| MessageEditorItem(message_editor.clone()))),
                    true,
                    true,
                    None,
                    window,
                    cx,
                );
            });
        });
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("AB", window, cx);
            let position = InlineTerminalDraftPosition::new("AB", 1).expect("position");
            editor
                .mount_inline_terminal(view.clone(), &position, cx)
                .expect("real view mount");
        });
        cx.run_until_parked();
        cx.update(|window, cx| {
            view.focus_handle(cx).focus(window, cx);
            window.draw(cx).clear(cx);
            assert!(view.focus_handle(cx).is_focused(window));
        });
        let arrow: terminal_view::SendKeystroke =
            serde_json::from_value(json!("up")).expect("canonical terminal action");
        cx.dispatch_action(arrow);
        let input = terminal.update(cx, |terminal, _| terminal.take_input_log());
        assert!(
            !input.is_empty(),
            "ordinary terminal keys must reach the existing terminal owner"
        );
        let escape: terminal_view::SendKeystroke =
            serde_json::from_value(json!("escape")).expect("canonical Escape action");
        cx.dispatch_action(escape);
        assert!(
            terminal
                .update(cx, |terminal, _| terminal.take_input_log())
                .is_empty(),
            "inline Escape must return to the draft instead of writing to the terminal"
        );
        message_editor.update_in(cx, |editor, window, cx| {
            assert!(
                editor.focus_handle(cx).is_focused(window),
                "Escape focuses the actual editor"
            );
            editor
                .editor
                .update(cx, |editor, cx| editor.handle_input("X", window, cx));
            assert_eq!(editor.text(cx), "AXB");
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_inline_terminal_scroll_reentry_after_retained_character_block(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::InlineTerminalDraftPosition;
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        let retained = inline_terminal_history_view(&message_editor, cx);
        let next = inline_terminal_history_view(&message_editor, cx);
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("AB", window, cx);
            let first = editor
                .mount_inline_terminal(
                    retained,
                    &InlineTerminalDraftPosition::new("AB", 1).expect("first boundary"),
                    cx,
                )
                .expect("retained terminal");
            let second = editor
                .mount_inline_terminal(
                    next,
                    &InlineTerminalDraftPosition::new("AB", 2).expect("following boundary"),
                    cx,
                )
                .expect("new terminal after retained history");
            editor.editor.update(cx, |editor, cx| {
                let first_row = editor.row_for_block(first, cx).expect("first display row");
                let second_row = editor
                    .row_for_block(second, cx)
                    .expect("second display row");
                assert!(first_row < second_row, "terminal rows follow source order");
            });
            for block_id in [first, second] {
                editor
                    .scroll_to_inline_terminal(block_id, window, cx)
                    .expect("every inline block has a reachable display row");
            }
            editor
                .focus_after_inline_terminal(second, window, cx)
                .expect("following draft focus");
            editor.editor.update(cx, |editor, cx| {
                assert!(editor.focus_handle(cx).is_focused(window));
                editor.handle_input("C", window, cx);
            });
            assert_eq!(editor.text(cx), "ABC");
            assert_eq!(
                editor
                    .inline_terminal_position(second, cx)
                    .expect("second position")
                    .byte_offset_for_source("ABC")
                    .expect("moving source"),
                2
            );
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_inline_terminal_real_view_keeps_moving_source_and_suffix_focus(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::InlineTerminalDraftPosition;
        use collections::HashSet;
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        let view = inline_terminal_history_view(&message_editor, cx);
        let terminal_id = view.read_with(cx, |view, _| view.terminal().entity_id());
        for (source, offset, suffix_source) in [("", 0, "X"), ("AB", 1, "AXB"), ("λ🦀", 2, "λX🦀")]
        {
            message_editor.update_in(cx, |editor, window, cx| {
                editor.set_text(source, window, cx);
                let position = InlineTerminalDraftPosition::new(source, offset).expect("position");
                let block_id = editor
                    .mount_inline_terminal(view.clone(), &position, cx)
                    .expect("real view mount");
                assert_eq!(
                    editor.text(cx),
                    source,
                    "mounting a terminal cannot insert transcript or sentinel text into the draft"
                );
                editor.editor.update(cx, |editor, cx| {
                    editor
                        .focus_after_character_block(block_id, window, cx)
                        .expect("suffix focus");
                    editor.handle_input("X", window, cx);
                });
                assert_eq!(editor.text(cx), suffix_source);
                assert_eq!(
                    editor
                        .inline_terminal_position(block_id, cx)
                        .expect("current moving position")
                        .byte_offset_for_source(suffix_source)
                        .expect("exact source identity"),
                    offset
                );
                editor.editor.update(cx, |editor, cx| {
                    editor.undo(&editor::actions::Undo, window, cx);
                    editor.change_selections(Default::default(), window, cx, |selections| {
                        let prefix =
                            editor::DisplayPoint::new(editor::display_map::DisplayRow(0), 0);
                        selections.select_display_ranges([prefix..prefix]);
                    });
                    editor.handle_input("P", window, cx);
                });
                let prefixed = format!("P{source}");
                assert_eq!(
                    editor
                        .inline_terminal_position(block_id, cx)
                        .expect("moved position")
                        .byte_offset_for_source(&prefixed)
                        .expect("updated source identity"),
                    offset + 1
                );
                editor.editor.update(cx, |editor, cx| {
                    editor.remove_blocks(HashSet::from_iter([block_id]), None, cx)
                });
                assert!(editor.inline_terminal_position(block_id, cx).is_err());
                assert_eq!(
                    editor.text(cx),
                    prefixed,
                    "unmounting only removes presentation and its virtual boundary"
                );
            });
        }
        view.read_with(cx, |view, cx| {
            assert_eq!(
                view.terminal().entity_id(),
                terminal_id,
                "the same canonical terminal is retained through placement and edits"
            );
            assert!(
                !view.terminal().read(cx).is_pty(),
                "history projection does not create a process"
            );
        });
    }

    #[gpui::test]
    async fn test_pending_source_submission_keeps_requested_text_and_context(
        cx: &mut TestAppContext,
    ) {
        assert_pending_source_capture(false, cx).await;
    }

    #[gpui::test]
    async fn test_pending_source_draft_keeps_requested_text_and_context(cx: &mut TestAppContext) {
        assert_pending_source_capture(true, cx).await;
    }

    async fn assert_pending_source_capture(draft: bool, cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        let prefix = "  original λ ";
        let suffix = " 🦀 \n";
        let (sender, receiver) = futures::channel::oneshot::channel::<Result<Mention, String>>();
        let pending = cx.update(|_, cx| {
            cx.spawn(async move |_| receiver.await.expect("controlled pending source context"))
                .shared()
        });
        message_editor.update_in(cx, |editor, window, cx| {
            editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true));
            editor.set_message(
                vec![
                    acp::ContentBlock::from(prefix),
                    acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                        "original.rs",
                        "file:///project/original.rs",
                    )),
                    acp::ContentBlock::from(suffix),
                ],
                window,
                cx,
            );
            editor.mention_set.update(cx, |mentions, cx| {
                let crease = mentions
                    .creases()
                    .into_iter()
                    .next()
                    .expect("actual source crease");
                let uri = mentions
                    .mention_uri_for_crease(&crease)
                    .expect("actual source URI");
                mentions.insert_mention(crease, uri, pending, None, cx);
            });
        });
        let contents = message_editor.update(cx, |editor, cx| {
            if draft {
                editor.draft_contents(cx)
            } else {
                let contents = editor.contents(false, cx);
                cx.spawn(async move |_, _cx| {
                    let (blocks, tracked) = contents.await?;
                    assert!(tracked.is_empty());
                    Ok(blocks)
                })
            }
        });
        cx.run_until_parked();
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("newer unsent β", window, cx);
        });
        sender
            .send(Ok(Mention::Text {
                content: "original authorized context".to_string(),
                tracked_buffers: Vec::new(),
            }))
            .expect("requested context resolution remains alive");
        let blocks = contents.await.expect("resolve the actual captured request");
        let expected = vec![
            acp::ContentBlock::from(if draft { prefix } else { prefix.trim_start() }),
            acp::ContentBlock::Resource(acp::EmbeddedResource::new(
                acp::EmbeddedResourceResource::TextResourceContents(
                    acp::TextResourceContents::new(
                        "original authorized context",
                        "file:///project/original.rs",
                    ),
                ),
            )),
            acp::ContentBlock::from(if draft { suffix } else { suffix.trim_end() }),
        ];
        assert_eq!(
            blocks, expected,
            "resolved content must use the requested source and crease snapshot"
        );
        assert_eq!(
            message_editor.read_with(cx, |editor, cx| editor.text(cx)),
            "newer unsent β"
        );
    }

    #[gpui::test]
    async fn test_pending_source_consumption_preserves_edit_undo_revision(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        message_editor.update_in(cx, |editor, window, cx| editor.set_text("λ", window, cx));
        let revision = message_editor.read_with(cx, |editor, cx| editor.input_revision(cx));
        message_editor.update_in(cx, |editor, window, cx| {
            editor.editor.update(cx, |editor, cx| {
                editor.handle_input("🦀", window, cx);
                editor.undo(&editor::actions::Undo, window, cx);
            });
            assert_eq!(editor.text(cx), "λ");
            assert!(
                !editor.clear_if_unchanged(revision, window, cx),
                "Undo returning identical text must not erase a later input revision"
            );
            assert_eq!(editor.text(cx), "λ");
            let current = editor.input_revision(cx);
            assert!(editor.clear_if_unchanged(current, window, cx));
            assert!(
                editor.text(cx).is_empty(),
                "unchanged accepted input keeps normal consumption"
            );
        });
    }

    #[gpui::test]
    async fn test_draft_boundaries_preserve_typed_text_and_submission_normalization(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        for source in [
            "",
            "  Aλ\n🦀B \t\n",
            "\nA\n\n",
            "\u{2003}λ\u{2003}",
            "\n \t\u{2003}\n",
        ] {
            message_editor.update_in(cx, |editor, window, cx| {
                editor.set_text(source, window, cx);
            });
            let snapshot = message_editor.read_with(cx, |editor, cx| {
                editor
                    .draft_content_blocks_snapshot(cx)
                    .expect("valid draft snapshot")
            });
            let expected = if source.is_empty() {
                Vec::new()
            } else {
                vec![acp::ContentBlock::Text(acp::TextContent::new(source))]
            };
            assert_eq!(snapshot, expected, "snapshot lost typed draft boundaries");
            let draft = message_editor
                .update(cx, |editor, cx| editor.draft_contents(cx))
                .await
                .expect("capture canonical draft");
            assert_eq!(
                draft, expected,
                "async persistence lost typed draft boundaries"
            );
            let (submission, _) = message_editor
                .update(cx, |editor, cx| editor.contents(false, cx))
                .await
                .expect("resolve submission");
            let expected_submission = if source.trim().is_empty() {
                Vec::new()
            } else {
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    source.trim(),
                ))]
            };
            assert_eq!(
                submission, expected_submission,
                "submission normalization changed"
            );
            message_editor.update_in(cx, |editor, window, cx| {
                editor.set_message(snapshot, window, cx);
            });
            assert_eq!(
                message_editor.read_with(cx, |editor, cx| editor.text(cx)),
                source
            );
        }
    }

    #[gpui::test]
    async fn test_draft_boundaries_preserve_rich_mention_order_and_restore(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;
        let uri = MentionUri::Selection {
            abs_path: Some(path!("/project/file.txt").into()),
            line_range: 0..=0,
            column: None,
        };
        let mention = acp::ContentBlock::Resource(acp::EmbeddedResource::new(
            acp::EmbeddedResourceResource::TextResourceContents(acp::TextResourceContents::new(
                "retained authorized selection",
                uri.to_uri().to_string(),
            )),
        ));
        let prefix = " \nλ ";
        let suffix = " 🦀\n\t";
        let expected = vec![
            acp::ContentBlock::Text(acp::TextContent::new(prefix)),
            mention.clone(),
            acp::ContentBlock::Text(acp::TextContent::new(suffix)),
        ];
        message_editor.update_in(cx, |editor, window, cx| {
            editor
                .session_capabilities
                .write()
                .set_prompt_capabilities(acp::PromptCapabilities::new().embedded_context(true));
            editor.set_message(expected.clone(), window, cx);
        });
        let source = format!("{prefix}{}{suffix}", uri.as_link());
        assert_eq!(
            message_editor.read_with(cx, |editor, cx| editor.text(cx)),
            source
        );
        let snapshot = message_editor.read_with(cx, |editor, cx| {
            editor
                .draft_content_blocks_snapshot(cx)
                .expect("valid draft snapshot")
        });
        assert_eq!(snapshot, expected);
        let draft = message_editor
            .update(cx, |editor, cx| editor.draft_contents(cx))
            .await
            .expect("resolve actual retained selection draft");
        assert_eq!(draft, expected);
        let (submission, _) = message_editor
            .update(cx, |editor, cx| editor.contents(false, cx))
            .await
            .expect("resolve actual selection submission");
        assert_eq!(
            submission,
            vec![
                acp::ContentBlock::Text(acp::TextContent::new(prefix.trim_start())),
                mention,
                acp::ContentBlock::Text(acp::TextContent::new(suffix.trim_end())),
            ]
        );
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(snapshot, window, cx);
        });
        assert_eq!(
            message_editor.read_with(cx, |editor, cx| editor.text(cx)),
            source
        );
        assert_eq!(
            message_editor.read_with(cx, |editor, cx| {
                editor
                    .draft_content_blocks_snapshot(cx)
                    .expect("valid draft snapshot")
            }),
            expected
        );
    }

    #[gpui::test]
    async fn test_set_message_plain_text(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;

        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "hello world".to_string(),
                ))],
                window,
                cx,
            );
        });

        let text = message_editor.update(cx, |editor, cx| editor.text(cx));
        assert_eq!(text, "hello world");
        assert!(!message_editor.update(cx, |editor, cx| editor.is_empty(cx)));
    }

    #[gpui::test]
    async fn test_set_message_normalizes_crlf_before_mention(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;

        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![
                    acp::ContentBlock::Text(acp::TextContent::new("before\r\n".to_string())),
                    acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                        "file.txt",
                        "file:///project/file.txt",
                    )),
                ],
                window,
                cx,
            );
        });

        let text = message_editor.update(cx, |editor, cx| editor.text(cx));
        assert_eq!(text, "before\n[@file.txt](file:///project/file.txt)");

        let mention_uris =
            message_editor.update(cx, |editor, cx| editor.mention_set.read(cx).mentions());
        assert_eq!(mention_uris.len(), 1);
    }

    #[gpui::test]
    async fn test_set_message_replaces_existing_content(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;

        // Set initial content.
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "old content".to_string(),
                ))],
                window,
                cx,
            );
        });

        // Replace with new content.
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "new content".to_string(),
                ))],
                window,
                cx,
            );
        });

        let text = message_editor.update(cx, |editor, cx| editor.text(cx));
        assert_eq!(
            text, "new content",
            "set_message should replace old content"
        );
    }

    #[gpui::test]
    async fn test_append_message_to_empty_editor(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;

        message_editor.update_in(cx, |editor, window, cx| {
            editor
                .append_message(
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "appended".to_string(),
                    ))],
                    Some("\n\n"),
                    window,
                    cx,
                )
                .expect("append must preserve pending content");
        });

        let text = message_editor.update(cx, |editor, cx| editor.text(cx));
        assert_eq!(
            text, "appended",
            "No separator should be inserted when the editor is empty"
        );
    }

    #[gpui::test]
    async fn test_append_message_to_non_empty_editor(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;

        // Seed initial content.
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "initial".to_string(),
                ))],
                window,
                cx,
            );
        });

        // Append with separator.
        message_editor.update_in(cx, |editor, window, cx| {
            editor
                .append_message(
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "appended".to_string(),
                    ))],
                    Some("\n\n"),
                    window,
                    cx,
                )
                .expect("append must preserve pending content");
        });

        let text = message_editor.update(cx, |editor, cx| editor.text(cx));
        assert_eq!(
            text, "initial\n\nappended",
            "Separator should appear between existing and appended content"
        );
    }

    #[gpui::test]
    async fn test_append_message_preserves_existing_selection(cx: &mut TestAppContext) {
        init_test(cx);
        let (message_editor, cx) = setup_message_editor(cx).await;

        for (source, selection) in [
            ("initial", 0..0),
            ("αβγ", 2..4),
            ("before\nmiddle\nafter", 7..13),
        ] {
            message_editor.update_in(cx, |message_editor, window, cx| {
                message_editor.set_message(vec![source.into()], window, cx);
                message_editor.editor.update(cx, |editor, cx| {
                    editor.change_selections(Default::default(), window, cx, |selections| {
                        selections
                            .select_ranges([MultiBufferOffset(selection.start)
                                ..MultiBufferOffset(selection.end)]);
                    });
                });
                message_editor
                    .append_message(vec!["queued".into()], Some("\n\n"), window, cx)
                    .expect("append must preserve pending content");
                assert_eq!(message_editor.text(cx), format!("{source}\n\nqueued"));
            });
        }
    }

    #[gpui::test]
    async fn test_append_message_preserves_mention_offset(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"file.txt": "content"}))
            .await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let message_editor = cx.update(|window, cx| {
            cx.new(|cx| {
                MessageEditor::new(
                    workspace.downgrade(),
                    project.downgrade(),
                    None,
                    Default::default(),
                    "Test Agent".into(),
                    "Test",
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: None,
                    },
                    window,
                    cx,
                )
            })
        });

        cx.run_until_parked();

        // Seed plain-text prefix so the editor is non-empty before appending.
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "prefix text".to_string(),
                ))],
                window,
                cx,
            );
        });

        // Append a message that contains a ResourceLink mention.
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_cursor_offset(0, window, cx);
            editor
                .append_message(
                    vec![acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                        "file.txt",
                        "file:///project/file.txt",
                    ))],
                    Some("\n\n"),
                    window,
                    cx,
                )
                .expect("append must preserve pending content");
        });

        cx.run_until_parked();

        // The mention should be registered in the mention_set so that contents()
        // will emit it as a structured block rather than plain text.
        let mention_uris =
            message_editor.update(cx, |editor, cx| editor.mention_set.read(cx).mentions());
        assert_eq!(
            mention_uris.len(),
            1,
            "Expected exactly one mention in the mention_set after append, got: {mention_uris:?}"
        );

        // The editor text should start with the prefix, then the separator, then
        // the mention placeholder — confirming the offset was computed correctly.
        let text = message_editor.update(cx, |editor, cx| editor.text(cx));
        assert!(
            text.starts_with("prefix text\n\n"),
            "Expected text to start with 'prefix text\\n\\n', got: {text:?}"
        );
        let (chunks, _) = message_editor
            .update(cx, |editor, cx| editor.contents(false, cx))
            .await
            .expect("appended structured content");
        assert!(matches!(
            chunks.first(),
            Some(acp::ContentBlock::Text(text)) if text.text == "prefix text\n\n"
        ));
        assert!(chunks.iter().any(|chunk| matches!(
            chunk,
            acp::ContentBlock::ResourceLink(resource) if resource.uri == "file:///project/file.txt"
        )));
    }
}
