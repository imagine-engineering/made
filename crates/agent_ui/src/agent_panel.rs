use std::{
    cell::Cell,
    fmt,
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use acp_thread::{AcpThread, AcpThreadEvent, MentionUri, ThreadStatus, line_range_suffix};
use agent::{ContextServerRegistry, SharedThread, ThreadStore};
use agent_client_protocol::schema::v1 as acp;
use agent_servers::AgentServer;
use agent_settings::UserAgentsMd;
use collections::HashSet;
use db::kvp::{Dismissable, KeyValueStore};
use itertools::Itertools;
use project::{AgentId, ProjectItem};
use serde::{Deserialize, Serialize};

use zed_actions::{
    DecreaseBufferFontSize, IncreaseBufferFontSize, ResetBufferFontSize,
    agent::{
        AddSelectionToThread, ConflictContent, LogoutAgent, OpenSettings, ReauthenticateAgent,
        ResetAgentZoom, ResetOnboarding, ResolveConflictedFilesWithAgent,
        ResolveConflictsWithAgent, ReviewBranchDiff, SelectAgent,
    },
    assistant::{
        FocusAgent, ManageSkills, OpenGlobalAgentsMdRules, OpenProjectAgentsMdRules, Toggle,
        ToggleFocus,
    },
};

use crate::ExpandMessageEditor;
use crate::ManageProfiles;
use crate::agent_connection_store::AgentConnectionStore;
use crate::completion_provider::{AgentContextSelection, AgentContextSource};
#[cfg(feature = "multiplayer-tools")]
use crate::terminal_thread_metadata_store::InlineTerminalContext;
use crate::terminal_thread_metadata_store::{
    TerminalThreadMetadata, TerminalThreadMetadataStore, compose_terminal_thread_title,
    terminal_title_without_prefix,
};
use crate::thread_metadata_store::{ThreadId, ThreadMetadataStore, ThreadMetadataStoreEvent};
use crate::{
    Agent, AgentInitialContent, AgentThreadSource, ExternalSourcePrompt, NewExternalAgentThread,
    NewNativeAgentThreadFromSummary,
};
use crate::{
    AgentDiffPane, ConversationView, CopyThreadToClipboard, Follow, LoadThreadFromClipboard,
    NewTerminalThread, NewThread, OpenActiveThreadAsMarkdown, OpenAgentDiff, ResetFastModeWarnings,
    ResetTrialEndUpsell, ResetTrialUpsell, ShowAllSidebarThreadMetadata, ShowThreadMetadata,
    ToggleNewThreadMenu, ToggleOptionsMenu,
    conversation_view::{
        AcpThreadViewEvent, RootThreadUpdated, ThreadView, reset_fast_mode_warnings,
    },
    ui::{AgentNotification, AgentNotificationEvent, EndTrialUpsell},
};
use agent_settings::AgentSettings;
use ai_onboarding::AgentPanelOnboarding;
use anyhow::{Context as _, Result, anyhow};
#[cfg(feature = "audio")]
use audio::{Audio, Sound};
use chrono::{DateTime, Utc};
use client::UserStore;
use cloud_api_types::Plan;
use collections::HashMap;
use editor::{Editor, MultiBuffer};
use extension_host::ExtensionStore;
use feature_flags::{CreateThreadToolFeatureFlag, FeatureFlagAppExt as _};

use fs::Fs;
use futures::FutureExt as _;
use gpui::{
    Action, Anchor, Animation, AnimationExt, AnyElement, App, AsyncWindowContext, ClipboardItem,
    Entity, EventEmitter, ExternalPaths, FocusHandle, Focusable, KeyContext, Pixels,
    PlatformDisplay, Subscription, Task, TaskExt, WeakEntity, WindowHandle, prelude::*,
    pulsating_between,
};
use language::LanguageRegistry;
use language_model::LanguageModelRegistry;
use notifications::status_toast::StatusToast;
use project::{Project, ProjectPath, Worktree};
use settings::TerminalDockPosition;
use settings::{NotifyWhenAgentWaiting, Settings, update_settings_file};

use search::{BufferSearchBar, buffer_search::Deploy as DeployBufferSearch};
use terminal::{Event as TerminalEvent, terminal_settings::TerminalSettings};
use terminal_view::{TerminalView, terminal_panel::TerminalPanel};
use text::OffsetRangeExt;
use theme_settings::ThemeSettings;
use ui::{
    ContextMenu, ContextMenuEntry, GradientFade, IconButton, KeyBinding, PopoverMenu,
    PopoverMenuHandle, ProjectEmptyState, Tab, Tooltip, prelude::*, utils::WithRemSize,
};
use util::ResultExt as _;
use workspace::{
    CollaboratorId, DraggedSelection, DraggedTab, MultiWorkspace, PathList, SerializedPathList,
    ToggleWorkspaceSidebar, ToggleZoom, ToolbarItemView, Workspace, WorkspaceId,
    dock::{DockPosition, Panel, PanelEvent},
    item::{ItemEvent, ItemHandle},
};

const AGENT_PANEL_KEY: &str = "agent_panel";
const MIN_PANEL_WIDTH: Pixels = px(300.);
const LAST_USED_AGENT_KEY: &str = "agent_panel__last_used_external_agent";
const LAST_CREATED_ENTRY_KIND_KEY: &str = "agent_panel__last_created_entry_kind";
const TERMINAL_AGENT_TELEMETRY_ID: &str = "terminal";
const TERMINAL_INIT_COMMAND_STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const KNOWN_TERMINAL_AGENT_COMMANDS: &[&str] = &[
    "agent", // Unfortunately, both Cursor cli + grok
    "agy",
    "aider",
    "amp",
    "claude",
    "codex",
    "copilot",
    "crush",
    "devin",
    "droid",
    "gemini",
    "goose",
    "grok",
    "openhands",
    "opencode",
    "pi",
    "qwen",
];

fn is_known_terminal_agent_command(command: &str) -> bool {
    KNOWN_TERMINAL_AGENT_COMMANDS.contains(&command)
}

fn terminal_program_to_report(
    last_observed_program: &mut Option<String>,
    current_program: Option<String>,
) -> Option<String> {
    let current_program =
        current_program.filter(|program| is_known_terminal_agent_command(program));
    let program_to_report =
        if current_program.is_some() && current_program != *last_observed_program {
            current_program.clone()
        } else {
            None
        };
    *last_observed_program = current_program;
    program_to_report
}

/// Maximum number of idle threads kept in the agent panel's retained list.
/// Set as a GPUI global to override; otherwise defaults to 5.
pub struct MaxIdleRetainedThreads(pub usize);
impl gpui::Global for MaxIdleRetainedThreads {}

impl MaxIdleRetainedThreads {
    pub fn global(cx: &App) -> usize {
        cx.try_global::<Self>().map_or(5, |g| g.0)
    }
}

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct TerminalId(uuid::Uuid);

impl TerminalId {
    pub(crate) fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }

    pub(crate) fn to_key_string(self) -> String {
        self.0.hyphenated().to_string()
    }

    pub(crate) fn from_key_string(key: &str) -> anyhow::Result<Self> {
        Ok(Self(uuid::Uuid::parse_str(key)?))
    }
}

impl fmt::Display for TerminalId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug)]
pub struct AgentPanelTerminalInfo {
    pub id: TerminalId,
    pub title: SharedString,
    pub created_at: DateTime<Utc>,
    pub has_notification: bool,
    pub custom_title: Option<SharedString>,
    pub working_directory: Option<PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct LastUsedAgent {
    agent: Agent,
}

#[derive(Serialize, Deserialize)]
struct LastCreatedEntryKind {
    entry_kind: AgentPanelEntryKind,
}

struct SourcePanelInitialization {
    agent: Agent,
    initial_content: Option<AgentInitialContent>,
}

/// Reads the most recently used agent across all workspaces. Used as a fallback
/// when opening a workspace that has no per-workspace agent preference yet.
fn read_global_last_used_agent(kvp: &KeyValueStore) -> Option<Agent> {
    kvp.read_kvp(LAST_USED_AGENT_KEY)
        .log_err()
        .flatten()
        .and_then(|json| serde_json::from_str::<LastUsedAgent>(&json).log_err())
        .map(|entry| entry.agent)
}

async fn write_global_last_used_agent(kvp: KeyValueStore, agent: Agent) {
    if let Some(json) = serde_json::to_string(&LastUsedAgent { agent }).log_err() {
        kvp.write_kvp(LAST_USED_AGENT_KEY.to_string(), json)
            .await
            .log_err();
    }
}

fn read_global_last_created_entry_kind(kvp: &KeyValueStore) -> Option<AgentPanelEntryKind> {
    kvp.read_kvp(LAST_CREATED_ENTRY_KIND_KEY)
        .log_err()
        .flatten()
        .and_then(|json| serde_json::from_str::<LastCreatedEntryKind>(&json).log_err())
        .map(|entry| entry.entry_kind)
}

fn project_agents_md_path(
    project: &Entity<Project>,
    require_existing_file: bool,
    cx: &App,
) -> Option<PathBuf> {
    let rel_path = util::rel_path::RelPath::from_unix_str("AGENTS.md").ok()?;
    project
        .read(cx)
        .visible_worktrees(cx)
        .next()
        .and_then(|worktree| {
            let worktree = worktree.read(cx);

            if require_existing_file {
                let entry = worktree.entry_for_path(rel_path)?;
                if !entry.is_file() {
                    return None;
                }
            }

            Some(worktree.absolutize(rel_path))
        })
}

fn open_global_rules(workspace: &mut Workspace, window: &mut Window, cx: &mut Context<Workspace>) {
    workspace
        .open_abs_path(
            paths::agents_file().clone(),
            workspace::OpenOptions {
                focus: Some(true),
                ..Default::default()
            },
            window,
            cx,
        )
        .detach_and_log_err(cx);
}

fn open_project_rules(workspace: &mut Workspace, window: &mut Window, cx: &mut Context<Workspace>) {
    if let Some(path) = project_agents_md_path(workspace.project(), false, cx) {
        workspace
            .open_abs_path(
                path,
                workspace::OpenOptions {
                    focus: Some(true),
                    ..Default::default()
                },
                window,
                cx,
            )
            .detach_and_log_err(cx);
    }
}

async fn write_global_last_created_entry_kind(kvp: KeyValueStore, entry_kind: AgentPanelEntryKind) {
    if let Some(json) = serde_json::to_string(&LastCreatedEntryKind { entry_kind }).log_err() {
        kvp.write_kvp(LAST_CREATED_ENTRY_KIND_KEY.to_string(), json)
            .await
            .log_err();
    }
}

fn read_serialized_panel(
    workspace_id: workspace::WorkspaceId,
    kvp: &KeyValueStore,
) -> Option<SerializedAgentPanel> {
    let scope = kvp.scoped(AGENT_PANEL_KEY);
    let key = i64::from(workspace_id).to_string();
    scope
        .read(&key)
        .log_err()
        .flatten()
        .and_then(|json| serde_json::from_str::<SerializedAgentPanel>(&json).log_err())
}

async fn save_serialized_panel(
    workspace_id: workspace::WorkspaceId,
    panel: SerializedAgentPanel,
    kvp: KeyValueStore,
) -> Result<()> {
    let scope = kvp.scoped(AGENT_PANEL_KEY);
    let key = i64::from(workspace_id).to_string();
    scope.write(key, serde_json::to_string(&panel)?).await?;
    Ok(())
}

/// Migration: reads the original single-panel format stored under the
/// `"agent_panel"` KVP key before per-workspace keying was introduced.
fn read_legacy_serialized_panel(kvp: &KeyValueStore) -> Option<SerializedAgentPanel> {
    kvp.read_kvp(AGENT_PANEL_KEY)
        .log_err()
        .flatten()
        .and_then(|json| serde_json::from_str::<SerializedAgentPanel>(&json).log_err())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadTitleRegenerationResult {
    NotOpen,
    Started,
    NoModel,
    AlreadyGenerating,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
enum AgentPanelEntryKind {
    #[default]
    Thread,
    Terminal,
}

#[derive(Serialize, Deserialize, Debug)]
struct SerializedAgentPanel {
    selected_agent: Option<Agent>,
    #[serde(default)]
    last_created_entry_kind: AgentPanelEntryKind,
    #[serde(default)]
    last_active_thread: Option<SerializedActiveThread>,
    #[serde(default)]
    last_active_terminal_id: Option<String>,
    #[serde(default)]
    new_draft_thread_id: Option<ThreadId>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SerializedActiveThread {
    /// For drafts this is `None`; use `thread_id` to address them instead.
    session_id: Option<String>,
    /// Optional for back-compat with older serialized payloads that only carried `session_id`.
    #[serde(default)]
    thread_id: Option<ThreadId>,
    agent_type: Agent,
    title: Option<String>,
    work_dirs: Option<SerializedPathList>,
}

fn new_thread_in_workspace(
    workspace: &mut Workspace,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
        panel.update(cx, |panel, cx| {
            panel.new_thread_with_workspace(Some(workspace), window, cx)
        });
        AgentPanel::reveal_conversation(workspace, true, window, cx);
    }
}

fn doctor_history_session_available(project: &Entity<Project>, session: &str, cx: &App) -> bool {
    if session.is_empty() || session.len() > 4096 || session.chars().any(char::is_control) {
        return false;
    }
    let Some(store) = ThreadMetadataStore::try_global(cx) else {
        return false;
    };
    let store = store.read(cx);
    let Some(metadata) = store.entry_by_session(&acp::SessionId::new(session.to_owned())) else {
        return false;
    };
    let owner = project.read(cx);
    let connection = owner
        .remote_client()
        .map(|client| client.read(cx).connection_options());
    metadata.agent_id == *agent::ZED_AGENT_ID
        && metadata.matches_remote_connection(connection.as_ref())
        && owner
            .visible_worktrees(cx)
            .any(|root| metadata.references_folder_path(root.read(cx).abs_path().as_ref()))
}

pub fn init(cx: &mut App) {
    cx.observe_new(|view: &mut diagnostics::doctor_view::DoctorView, _, cx| {
        let subscription = cx.subscribe(
            &cx.entity(),
            |_view, _, request: &Rc<diagnostics::doctor::DoctorNativeRepairRequest>, cx| {
                if !request.claim() {
                    return;
                }
                let result = request
                    .window
                    .update(cx, |_, window, cx| {
                        request
                            .workspace
                            .update(cx, |workspace, cx| {
                                request.revalidate(cx)?;
                                anyhow::ensure!(
                                    workspace.project() == &request.project,
                                    "Doctor repair workspace project changed"
                                );
                                let panel = workspace.panel::<AgentPanel>(cx).ok_or_else(|| {
                                    anyhow!(
                                        "Native agent panel is unavailable; no repair was launched"
                                    )
                                })?;
                                Ok::<_, anyhow::Error>(panel.update(cx, |panel, cx| {
                                    panel.doctor_native_repair(request.clone(), window, cx)
                                }))
                            })
                            .and_then(|result| result)
                    })
                    .and_then(|result| result);
                match result {
                    Err(error) => request.complete(Err(error), cx),
                    Ok(operation) => {
                        let request = request.clone();
                        cx.spawn(async move |_, cx| {
                            let result = operation.await;
                            cx.update(|cx| request.complete(result, cx));
                        })
                        .detach();
                    }
                }
            },
        );
        view.retain_native_repair_subscription(subscription);
    })
    .detach();
    cx.observe_new(
        |workspace: &mut Workspace, _window, _cx: &mut Context<Workspace>| {
            workspace
                .register_action(|workspace, _: &NewThread, window, cx| {
                    new_thread_in_workspace(workspace, window, cx);
                })
                .register_action(|workspace, _: &workspace::NewThread, window, cx| {
                    new_thread_in_workspace(workspace, window, cx);
                })
                .register_action(|workspace, _: &NewTerminalThread, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, cx| {
                            panel.new_terminal(
                                Some(workspace),
                                AgentThreadSource::AgentPanel,
                                window,
                                cx,
                            )
                        });
                        #[cfg(feature = "multiplayer-tools")]
                        if workspace.workspace_presentation()
                            == workspace::WorkspacePresentation::Collaborative
                        {
                            // Attachment focuses the PTY; a deferred composer focus would steal it.
                            AgentPanel::reveal_conversation(workspace, false, window, cx);
                            return;
                        }
                        workspace.focus_panel::<AgentPanel>(window, cx);
                    }
                })
                .register_action(
                    |workspace, action: &NewNativeAgentThreadFromSummary, window, cx| {
                        if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                            panel.update(cx, |panel, cx| {
                                panel.new_native_agent_thread_from_summary(action, window, cx)
                            });
                            AgentPanel::reveal_conversation(workspace, true, window, cx);
                        }
                    },
                )
                .register_action(|workspace, _: &ExpandMessageEditor, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        AgentPanel::reveal_conversation(workspace, true, window, cx);
                        panel.update(cx, |panel, cx| panel.expand_message_editor(window, cx));
                    }
                })
                .register_action(AgentPanel::open_settings)
                .register_action(|workspace, action: &NewExternalAgentThread, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        AgentPanel::reveal_conversation(workspace, true, window, cx);
                        panel.update(cx, |panel, cx| {
                            panel.new_external_agent_thread(action, window, cx);
                        });
                    }
                })
                .register_action(|workspace, action: &SelectAgent, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, cx| {
                            let agent = AgentId::new(action.agent.clone()).into();
                            panel.select_agent(agent, window, cx);
                        });
                    }
                })
                .register_action(|workspace, action: &ManageSkills, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        workspace.focus_panel::<AgentPanel>(window, cx);
                        panel.update(cx, |panel, cx| panel.manage_skills(action, window, cx));
                    }
                })
                .register_action(|workspace, _: &OpenGlobalAgentsMdRules, window, cx| {
                    open_global_rules(workspace, window, cx);
                })
                .register_action(|workspace, _: &OpenProjectAgentsMdRules, window, cx| {
                    open_project_rules(workspace, window, cx);
                })
                .register_action(|workspace, _: &Follow, window, cx| {
                    workspace.follow(CollaboratorId::Agent, window, cx);
                })
                .register_action(|workspace, _: &OpenAgentDiff, window, cx| {
                    let thread = workspace
                        .panel::<AgentPanel>(cx)
                        .and_then(|panel| panel.read(cx).active_conversation_view().cloned())
                        .and_then(|conversation| {
                            conversation
                                .read(cx)
                                .root_thread_view()
                                .map(|r| r.read(cx).thread.clone())
                        });

                    if let Some(thread) = thread {
                        AgentDiffPane::deploy_in_workspace(thread, workspace, window, cx);
                    }
                })
                .register_action(|workspace, _: &ToggleOptionsMenu, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        workspace.focus_panel::<AgentPanel>(window, cx);
                        panel.update(cx, |panel, cx| {
                            panel.toggle_options_menu(&ToggleOptionsMenu, window, cx);
                        });
                    }
                })
                .register_action(|workspace, _: &ToggleNewThreadMenu, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        workspace.focus_panel::<AgentPanel>(window, cx);
                        panel.update(cx, |panel, cx| {
                            panel.toggle_new_thread_menu(&ToggleNewThreadMenu, window, cx);
                        });
                    }
                })
                .register_action(|_workspace, _: &ResetOnboarding, window, cx| {
                    window.dispatch_action(workspace::RestoreBanner.boxed_clone(), cx);
                    window.refresh();
                })
                .register_action(|workspace, _: &ResetTrialUpsell, _window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, _| {
                            panel
                                .new_user_onboarding_upsell_dismissed
                                .store(false, Ordering::Release);
                        });
                    }
                    OnboardingUpsell::set_dismissed(false, cx);
                })
                .register_action(|_workspace, _: &ResetTrialEndUpsell, _window, cx| {
                    TrialEndUpsell::set_dismissed(false, cx);
                })
                .register_action(|_workspace, _: &ResetFastModeWarnings, _window, cx| {
                    reset_fast_mode_warnings(cx);
                })
                .register_action(|workspace, _: &ResetAgentZoom, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, cx| {
                            panel.reset_agent_zoom(window, cx);
                        });
                    }
                })
                .register_action(|workspace, _: &CopyThreadToClipboard, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, cx| {
                            panel.copy_thread_to_clipboard(window, cx);
                        });
                    }
                })
                .register_action(|workspace, action: &crate::OpenDoctorSession, window, cx| {
                    let session_id = acp::SessionId::new(action.session_id.clone());
                    let available = doctor_history_session_available(workspace.project(), &action.session_id, cx);
                    let panel = workspace.panel::<AgentPanel>(cx);
                    if !available || panel.is_none() {
                        workspace.show_toast(workspace::Toast::new(
                            workspace::notifications::NotificationId::unique::<crate::OpenDoctorSession>(),
                            "Doctor's existing native session is unavailable; no session was created or imported",
                        ).autohide(), cx);
                        return;
                    }
                    if let Some(panel) = panel {
                        AgentPanel::reveal_conversation(workspace, true, window, cx);
                        panel.update(cx, |panel, cx| panel.open_thread(session_id, None, None, window, cx));
                    }
                })
                .register_action(|workspace, _: &LoadThreadFromClipboard, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        AgentPanel::reveal_conversation(workspace, true, window, cx);
                        panel.update(cx, |panel, cx| {
                            panel.load_thread_from_clipboard(window, cx);
                        });
                    }
                })
                .register_action(|workspace, _: &ShowThreadMetadata, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, cx| {
                            panel.show_thread_metadata(&ShowThreadMetadata, window, cx);
                        });
                    }
                })
                .register_action(|workspace, _: &ShowAllSidebarThreadMetadata, window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        panel.update(cx, |panel, cx| {
                            panel.show_all_sidebar_thread_metadata(
                                &ShowAllSidebarThreadMetadata,
                                window,
                                cx,
                            );
                        });
                    }
                })
                .register_action(|workspace, action: &ReviewBranchDiff, window, cx| {
                    let Some(panel) = workspace.panel::<AgentPanel>(cx) else {
                        return;
                    };

                    let mention_uri = MentionUri::GitDiff {
                        base_ref: action.base_ref.to_string(),
                    };
                    let diff_uri = mention_uri.to_uri().to_string();

                    let content_blocks = vec![
                        acp::ContentBlock::Text(acp::TextContent::new(
                            "Please review this branch diff carefully. Point out any issues, \
                             potential bugs, or improvement opportunities you find.\n\n"
                                .to_string(),
                        )),
                        acp::ContentBlock::Resource(acp::EmbeddedResource::new(
                            acp::EmbeddedResourceResource::TextResourceContents(
                                acp::TextResourceContents::new(
                                    action.diff_text.to_string(),
                                    diff_uri,
                                ),
                            ),
                        )),
                    ];

                    AgentPanel::reveal_conversation(workspace, true, window, cx);

                    panel.update(cx, |panel, cx| {
                        panel.external_thread(
                            None,
                            None,
                            None,
                            None,
                            Some(AgentInitialContent::ContentBlock {
                                blocks: content_blocks,
                                auto_submit: true,
                            }),
                            true,
                            AgentThreadSource::GitPanel,
                            window,
                            cx,
                        );
                    });
                })
                .register_action(
                    |workspace, action: &ResolveConflictsWithAgent, window, cx| {
                        let Some(panel) = workspace.panel::<AgentPanel>(cx) else {
                            return;
                        };

                        let content_blocks = build_conflict_resolution_prompt(&action.conflicts);

                        AgentPanel::reveal_conversation(workspace, true, window, cx);

                        panel.update(cx, |panel, cx| {
                            panel.external_thread(
                                None,
                                None,
                                None,
                                None,
                                Some(AgentInitialContent::ContentBlock {
                                    blocks: content_blocks,
                                    auto_submit: true,
                                }),
                                true,
                                AgentThreadSource::GitPanel,
                                window,
                                cx,
                            );
                        });
                    },
                )
                .register_action(
                    |workspace, action: &ResolveConflictedFilesWithAgent, window, cx| {
                        let Some(panel) = workspace.panel::<AgentPanel>(cx) else {
                            return;
                        };

                        let content_blocks =
                            build_conflicted_files_resolution_prompt(&action.conflicted_file_paths);

                        AgentPanel::reveal_conversation(workspace, true, window, cx);

                        panel.update(cx, |panel, cx| {
                            panel.external_thread(
                                None,
                                None,
                                None,
                                None,
                                Some(AgentInitialContent::ContentBlock {
                                    blocks: content_blocks,
                                    auto_submit: true,
                                }),
                                true,
                                AgentThreadSource::GitPanel,
                                window,
                                cx,
                            );
                        });
                    },
                )
                .register_action(
                    |workspace: &mut Workspace, _: &AddSelectionToThread, window, cx| {
                        let active_editor = workspace
                            .active_item(cx)
                            .and_then(|item| item.act_as::<Editor>(cx));
                        let has_editor_selection = active_editor.is_some_and(|editor| {
                            editor.update(cx, |editor, cx| {
                                editor.has_non_empty_selection(&editor.display_snapshot(cx))
                            })
                        });

                        let has_terminal_selection = workspace
                            .active_item(cx)
                            .and_then(|item| item.act_as::<TerminalView>(cx))
                            .is_some_and(|terminal_view| {
                                terminal_view
                                    .read(cx)
                                    .terminal()
                                    .read(cx)
                                    .last_content
                                    .selection_text
                                    .as_ref()
                                    .is_some_and(|text| !text.is_empty())
                            });

                        let has_terminal_panel_selection =
                            workspace.panel::<TerminalPanel>(cx).is_some_and(|panel| {
                                let position = match TerminalSettings::get_global(cx).dock {
                                    TerminalDockPosition::Left => DockPosition::Left,
                                    TerminalDockPosition::Bottom => DockPosition::Bottom,
                                    TerminalDockPosition::Right => DockPosition::Right,
                                };
                                let dock_is_open =
                                    workspace.dock_at_position(position).read(cx).is_open();
                                dock_is_open && !panel.read(cx).terminal_selections(cx).is_empty()
                            });

                        if !has_editor_selection
                            && !has_terminal_selection
                            && !has_terminal_panel_selection
                        {
                            return;
                        }

                        let Some(agent_panel) = workspace.panel::<AgentPanel>(cx) else {
                            return;
                        };

                        let source = AgentContextSource::from_focused(workspace, window, cx);
                        let source = source.or_else(|| {
                            let cached = agent_panel.read(cx).last_context_source.clone()?;
                            cached.exists(workspace, cx).then_some(cached)
                        });
                        let source =
                            source.or_else(|| AgentContextSource::from_active(workspace, cx));

                        let Some(source) = source else {
                            return;
                        };

                        let Some(selection) = source.read_selection(workspace, true, cx) else {
                            return;
                        };

                        if !agent_panel.focus_handle(cx).contains_focused(window, cx) {
                            AgentPanel::reveal_conversation(workspace, true, window, cx);
                        }

                        agent_panel.update(cx, |panel, cx| {
                            panel.last_context_source = Some(source);
                            cx.defer_in(window, move |panel, window, cx| {
                                if let Some(conversation_view) = panel.active_conversation_view() {
                                    conversation_view.update(cx, |conversation_view, cx| {
                                        conversation_view.insert_selection(selection, window, cx);
                                    });
                                } else if let Some(terminal_id) = panel.active_terminal_id()
                                    && let Some(agent_terminal) = panel.terminals.get(&terminal_id)
                                {
                                    // Resolve mentions against the cwd: live cwd, else spawn dir.
                                    let working_directory = agent_terminal
                                        .view
                                        .read(cx)
                                        .terminal()
                                        .read(cx)
                                        .working_directory()
                                        .or_else(|| agent_terminal.working_directory.clone());
                                    let text = format_selection_for_terminal(
                                        &selection,
                                        &panel.project,
                                        working_directory.as_deref(),
                                        cx,
                                    );
                                    if !text.is_empty() {
                                        let view = agent_terminal.view.clone();
                                        view.update(cx, |view, cx| {
                                            view.terminal().update(cx, |terminal, _| {
                                                terminal.paste(&text);
                                            });
                                            window.focus(&view.focus_handle(cx), cx);
                                        });
                                    }
                                }
                            });
                        });
                    },
                )
                .register_action(|workspace, _: &menu::Cancel, _window, cx| {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        let dismissed =
                            panel.update(cx, |panel, cx| panel.dismiss_all_notifications(cx));
                        if dismissed {
                            return;
                        }
                    }
                    cx.propagate();
                });
        },
    )
    .detach();
}

fn format_selection_for_terminal(
    selection: &AgentContextSelection,
    project: &Entity<Project>,
    working_directory: Option<&std::path::Path>,
    cx: &App,
) -> String {
    match selection {
        AgentContextSelection::Editor(ranges) => {
            let path_style = project.read(cx).path_style(cx);
            let mut parts: Vec<String> = Vec::new();
            for (buffer, range) in ranges {
                let buffer = buffer.read(cx);
                let Some(project_path) = buffer.project_path(cx) else {
                    continue;
                };
                let snapshot = buffer.snapshot();
                let point_range = range.to_point(&snapshot);
                let line_range = point_range.start.row..=point_range.end.row;
                let path = mention_path_for_terminal(
                    project,
                    &project_path,
                    working_directory,
                    path_style,
                    cx,
                );
                parts.push(format!("{path}{}", line_range_suffix(&line_range)));
            }
            if parts.is_empty() {
                String::new()
            } else {
                // Trailing space so the mention doesn't fuse with the next input.
                format!("{} ", parts.join(" "))
            }
        }
        AgentContextSelection::Terminal(texts) => texts.join("\n"),
    }
}

/// Path for a terminal mention: relative to the terminal cwd if possible, else absolute.
fn mention_path_for_terminal(
    project: &Entity<Project>,
    project_path: &ProjectPath,
    working_directory: Option<&std::path::Path>,
    path_style: util::paths::PathStyle,
    cx: &App,
) -> String {
    let abs_path = project.read(cx).absolute_path(project_path, cx);
    match (abs_path, working_directory) {
        (Some(abs_path), Some(working_directory)) => path_style
            .strip_prefix(&abs_path, working_directory)
            .map(|relative| relative.display(path_style).into_owned())
            .unwrap_or_else(|| abs_path.to_string_lossy().into_owned()),
        (Some(abs_path), None) => abs_path.to_string_lossy().into_owned(),
        (None, _) => project_path.path.display(path_style).into_owned(),
    }
}

fn conflict_resource_block(conflict: &ConflictContent) -> acp::ContentBlock {
    let mention_uri = MentionUri::MergeConflict {
        file_path: conflict.file_path.clone(),
    };
    acp::ContentBlock::Resource(acp::EmbeddedResource::new(
        acp::EmbeddedResourceResource::TextResourceContents(acp::TextResourceContents::new(
            conflict.conflict_text.clone(),
            mention_uri.to_uri().to_string(),
        )),
    ))
}

fn build_conflict_resolution_prompt(conflicts: &[ConflictContent]) -> Vec<acp::ContentBlock> {
    if conflicts.is_empty() {
        return Vec::new();
    }

    let mut blocks = Vec::new();

    if conflicts.len() == 1 {
        let conflict = &conflicts[0];

        blocks.push(acp::ContentBlock::Text(acp::TextContent::new(
            "Please resolve the following merge conflict in ",
        )));
        let mention = MentionUri::File {
            abs_path: PathBuf::from(conflict.file_path.clone()),
        };
        blocks.push(acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
            mention.name(),
            mention.to_uri(),
        )));

        blocks.push(acp::ContentBlock::Text(acp::TextContent::new(
            indoc::formatdoc!(
                "\nThe conflict is between branch `{ours}` (ours) and `{theirs}` (theirs).

                Analyze both versions carefully and resolve the conflict by editing \
                the file directly. Choose the resolution that best preserves the intent \
                of both changes, or combine them if appropriate.

                ",
                ours = conflict.ours_branch_name,
                theirs = conflict.theirs_branch_name,
            ),
        )));
    } else {
        let n = conflicts.len();
        let unique_files: HashSet<&str> = conflicts.iter().map(|c| c.file_path.as_str()).collect();
        let ours = &conflicts[0].ours_branch_name;
        let theirs = &conflicts[0].theirs_branch_name;
        blocks.push(acp::ContentBlock::Text(acp::TextContent::new(
            indoc::formatdoc!(
                "Please resolve all {n} merge conflicts below.

                The conflicts are between branch `{ours}` (ours) and `{theirs}` (theirs).

                For each conflict, analyze both versions carefully and resolve them \
                by editing the file{suffix} directly. Choose resolutions that best preserve \
                the intent of both changes, or combine them if appropriate.

                ",
                suffix = if unique_files.len() > 1 { "s" } else { "" },
            ),
        )));
    }

    for conflict in conflicts {
        blocks.push(conflict_resource_block(conflict));
    }

    blocks
}

fn build_conflicted_files_resolution_prompt(
    conflicted_file_paths: &[String],
) -> Vec<acp::ContentBlock> {
    if conflicted_file_paths.is_empty() {
        return Vec::new();
    }

    let instruction = indoc::indoc!(
        "The following files have unresolved merge conflicts. Please open each \
         file, find the conflict markers (`<<<<<<<` / `=======` / `>>>>>>>`), \
         and resolve every conflict by editing the files directly.

         Choose resolutions that best preserve the intent of both changes, \
         or combine them if appropriate.

         Files with conflicts:
         ",
    );

    let mut content = vec![acp::ContentBlock::Text(acp::TextContent::new(instruction))];
    for path in conflicted_file_paths {
        let mention = MentionUri::File {
            abs_path: PathBuf::from(path),
        };
        content.push(acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
            mention.name(),
            mention.to_uri(),
        )));
        content.push(acp::ContentBlock::Text(acp::TextContent::new("\n")));
    }
    content
}

fn format_timestamp_human(dt: &DateTime<Utc>) -> String {
    let now = Utc::now();
    let duration = now.signed_duration_since(*dt);

    let relative = if duration.num_seconds() < 0 {
        "in the future".to_string()
    } else if duration.num_seconds() < 60 {
        let seconds = duration.num_seconds();
        format!("{seconds} seconds ago")
    } else if duration.num_minutes() < 60 {
        let minutes = duration.num_minutes();
        format!("{minutes} minutes ago")
    } else if duration.num_hours() < 24 {
        let hours = duration.num_hours();
        format!("{hours} hours ago")
    } else {
        let days = duration.num_days();
        format!("{days} days ago")
    };

    format!("{} ({})", dt.to_rfc3339(), relative)
}

/// Used for `dev: show thread metadata` action
fn thread_metadata_to_debug_json(
    metadata: &crate::thread_metadata_store::ThreadMetadata,
) -> serde_json::Value {
    serde_json::json!({
        "thread_id": metadata.thread_id,
        "session_id": metadata.session_id.as_ref().map(|s| s.0.to_string()),
        "agent_id": metadata.agent_id.0.to_string(),
        "title": metadata.title.as_ref().map(|t| t.to_string()),
        "title_override": metadata.title_override.as_ref().map(|t| t.to_string()),
        "updated_at": format_timestamp_human(&metadata.updated_at),
        "created_at": metadata.created_at.as_ref().map(format_timestamp_human),
        "interacted_at": metadata.interacted_at.as_ref().map(format_timestamp_human),
        "worktree_paths": format!("{:?}", metadata.worktree_paths),
        "archived": metadata.archived,
    })
}

/// Optional parameters for `AgentPanel::create_thread_with_options`. All
/// fields default to the panel's current selection so the agent tool only
/// needs to override what it actually cares about.
#[derive(Default)]
pub struct CreateThreadOptions {
    /// Title to assign to the new thread up front.
    pub title: Option<SharedString>,
    /// Initial content to populate in the thread (optionally auto-submitted).
    pub initial_content: Option<AgentInitialContent>,
    /// Agent to use. Defaults to the panel's selected agent.
    pub agent: Option<Agent>,
    /// Model override, as `provider/model-id`. Only applied when the thread
    /// uses the native Zed agent.
    pub model: Option<String>,
    /// Working directories to attach to the new thread (e.g., the path of a
    /// freshly-created sibling worktree). When `None`, the thread inherits
    /// the project's default path list.
    pub work_dirs: Option<PathList>,
}

pub(crate) struct AgentThread {
    conversation_view: Entity<ConversationView>,
}

enum TerminalSpawnAssociation {
    Standalone,
    #[cfg(feature = "multiplayer-tools")]
    Inline(Box<InlineTerminalSpawnContext>),
}

#[cfg(feature = "multiplayer-tools")]
struct InlineTerminalSpawnContext {
    parent: Entity<ThreadView>,
    context: InlineTerminalContext,
    history_owner: Entity<TerminalThreadMetadataStore>,
    project_id: gpui::EntityId,
    remote_connection: Option<remote::RemoteConnectionOptions>,
    init_command: Option<String>,
}

struct AgentTerminal {
    view: Entity<TerminalView>,
    #[cfg(feature = "multiplayer-tools")]
    inline_history_owner: Option<WeakEntity<TerminalThreadMetadataStore>>,
    title_editor: Option<Entity<Editor>>,
    title_editor_initial_title: Option<String>,
    title_editor_subscription: Option<Subscription>,
    last_known_title: String,
    last_known_terminal_title: String,
    last_observed_program: Option<String>,
    working_directory: Option<PathBuf>,
    created_at: DateTime<Utc>,
    has_notification: bool,
    search_bar: Option<Entity<BufferSearchBar>>,
    notification_windows: Vec<WindowHandle<AgentNotification>>,
    notification_subscriptions: Vec<Subscription>,
    _subscriptions: Vec<Subscription>,
}

impl AgentTerminal {
    fn terminal_title_for_view(view: &TerminalView, cx: &App) -> SharedString {
        let terminal = view.terminal().read(cx);
        if terminal.breadcrumb_text.is_empty() {
            let title = terminal.title(true);
            if title == "Terminal" {
                SharedString::from("")
            } else {
                title.into()
            }
        } else {
            terminal.breadcrumb_text.clone().into()
        }
    }

    fn current_terminal_title(&self, cx: &App) -> SharedString {
        let view = self.view.read(cx);
        Self::terminal_title_for_view(view, cx)
    }

    fn terminal_title(&self, cx: &App) -> SharedString {
        let title = self.current_terminal_title(cx);
        if title.is_empty() && !self.last_known_terminal_title.is_empty() {
            SharedString::from(self.last_known_terminal_title.clone())
        } else {
            title
        }
    }

    fn title(&self, cx: &App) -> SharedString {
        let terminal_title = self.terminal_title(cx);
        let custom_title = self.custom_title(cx);
        compose_terminal_thread_title(
            terminal_title.as_ref(),
            custom_title.as_ref().map(|title| title.as_ref()),
        )
    }

    fn editable_title(&self, cx: &App) -> SharedString {
        if let Some(custom_title) = self.custom_title(cx) {
            custom_title
        } else {
            let terminal_title = self.terminal_title(cx);
            SharedString::from(terminal_title_without_prefix(terminal_title.as_ref()).to_string())
        }
    }

    fn refresh_title(&mut self, cx: &mut App) -> bool {
        let terminal_title = self.current_terminal_title(cx);
        if !terminal_title.is_empty() {
            self.last_known_terminal_title = terminal_title.to_string();
        }

        let title = self.title(cx);
        let changed = self.last_known_title != title.as_ref();
        if changed {
            self.last_known_title = title.to_string();
        }
        changed
    }

    fn refresh_metadata(&mut self, cx: &mut App) -> bool {
        let title_changed = self.refresh_title(cx);
        let current_working_directory = self.view.read(cx).terminal().read(cx).working_directory();
        let working_directory_changed = current_working_directory
            .as_ref()
            .is_some_and(|current| self.working_directory.as_ref() != Some(current));
        if working_directory_changed {
            self.working_directory = current_working_directory;
        }
        title_changed || working_directory_changed
    }

    fn custom_title(&self, cx: &App) -> Option<SharedString> {
        self.view.read(cx).custom_title().map(SharedString::from)
    }

    fn report_started_terminal_program(
        &mut self,
        terminal_id: TerminalId,
        source: AgentThreadSource,
        cx: &App,
    ) {
        let current_program = self
            .view
            .read(cx)
            .terminal()
            .read(cx)
            .foreground_process_command_name();

        if let Some(program) =
            terminal_program_to_report(&mut self.last_observed_program, current_program)
        {
            telemetry::event!(
                "Agent Terminal Program Started",
                agent = TERMINAL_AGENT_TELEMETRY_ID,
                terminal_id = terminal_id.to_key_string(),
                program = program,
                source = source.as_str(),
                side = crate::agent_sidebar_side(cx),
                thread_location = "current_worktree",
            );
        }
    }
}

enum BaseView {
    Uninitialized,
    AgentThread {
        conversation_view: Entity<ConversationView>,
    },
    Terminal {
        terminal_id: TerminalId,
    },
}

impl From<AgentThread> for BaseView {
    fn from(thread: AgentThread) -> Self {
        BaseView::AgentThread {
            conversation_view: thread.conversation_view,
        }
    }
}

enum VisibleSurface<'a> {
    Uninitialized,
    AgentThread(&'a Entity<ConversationView>),
    Terminal(&'a Entity<TerminalView>),
}

enum WhichFontSize {
    AgentFont,
    None,
}

impl BaseView {
    pub fn which_font_size_used(&self) -> WhichFontSize {
        match self {
            BaseView::AgentThread { .. } => WhichFontSize::AgentFont,
            BaseView::Terminal { .. } | BaseView::Uninitialized => WhichFontSize::None,
        }
    }
}

pub struct AgentPanel {
    workspace: WeakEntity<Workspace>,
    /// Workspace id is used as a database key
    workspace_id: Option<WorkspaceId>,
    user_store: Entity<UserStore>,
    project: Entity<Project>,
    fs: Arc<dyn Fs>,
    language_registry: Arc<LanguageRegistry>,
    thread_store: Entity<ThreadStore>,
    connection_store: Entity<AgentConnectionStore>,
    context_server_registry: Entity<ContextServerRegistry>,
    focus_handle: FocusHandle,
    base_view: BaseView,
    last_created_entry_kind: AgentPanelEntryKind,
    draft_thread: Option<Entity<ConversationView>>,
    retained_threads: HashMap<ThreadId, Entity<ConversationView>>,
    terminals: HashMap<TerminalId, AgentTerminal>,
    pending_terminal_spawn: Option<TerminalId>,
    #[cfg(feature = "multiplayer-tools")]
    pending_inline_terminal_open: Option<Subscription>,
    new_thread_menu_handle: PopoverMenuHandle<ContextMenu>,
    agent_panel_menu_handle: PopoverMenuHandle<ContextMenu>,
    _extension_subscription: Option<Subscription>,
    _doctor_subscription: Option<(gpui::EntityId, Subscription)>,
    _doctor_context_subscription: Option<(gpui::EntityId, acp::SessionId, Subscription)>,
    _project_subscription: Subscription,
    zoomed: bool,
    pending_serialization: Option<Task<Result<()>>>,
    pending_last_used_agent_persistence: Option<Task<()>>,
    new_user_onboarding: Entity<AgentPanelOnboarding>,
    new_user_onboarding_upsell_dismissed: AtomicBool,
    selected_agent: Agent,
    _thread_view_subscription: Option<Subscription>,
    _active_thread_focus_subscription: Option<Subscription>,
    _base_view_observation: Option<Subscription>,
    _draft_editor_observation: Option<Subscription>,
    _active_draft_reclaim_observation: Option<Subscription>,
    _thread_metadata_store_subscription: Subscription,
    last_context_source: Option<AgentContextSource>,

    is_active: bool,
}

impl AgentPanel {
    pub fn reveal_conversation(
        workspace: &mut Workspace,
        focus: bool,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        #[cfg(feature = "multiplayer-tools")]
        if workspace.workspace_presentation() == workspace::WorkspacePresentation::Collaborative {
            for dock in workspace.all_docks() {
                dock.update(cx, |dock, cx| {
                    if dock
                        .visible_panel()
                        .is_some_and(|panel| panel.is_agent_panel(cx))
                    {
                        dock.set_open(false, window, cx);
                    }
                });
            }
            if focus {
                let workspace = cx.weak_entity();
                // Resolve the input after active-thread adapters and their dispatch tree mount.
                window.on_next_frame(move |window, _| {
                    window.on_next_frame(move |window, cx| {
                        if let Some(workspace) = workspace.upgrade()
                            && let Some(focus_handle) =
                                workspace.read(cx).collaborative_composer().focus_handle()
                        {
                            focus_handle.focus(window, cx);
                        }
                    });
                });
            }
            return;
        }
        if focus {
            workspace.focus_panel::<Self>(window, cx);
        } else {
            workspace.reveal_panel::<Self>(window, cx);
        }
    }

    fn serialize(&mut self, cx: &mut App) {
        let Some(workspace_id) = self.workspace_id else {
            return;
        };

        let selected_agent = self.selected_agent.clone();
        let last_created_entry_kind = self.last_created_entry_kind;
        let last_active_terminal_id = self
            .active_terminal_id()
            .map(|terminal_id| terminal_id.to_key_string());

        let last_active_thread = if last_active_terminal_id.is_some() {
            None
        } else {
            let is_draft_active = self.active_thread_is_draft(cx);
            let active_thread_id = self.active_thread_id(cx);
            let active_thread_agent = self
                .active_conversation_view()
                .map(|cv| cv.read(cx).agent_key().clone())
                .unwrap_or_else(|| self.selected_agent.clone());
            self.active_agent_thread(cx)
                .map(|thread| {
                    let thread = thread.read(cx);

                    let title = thread.title();
                    let work_dirs = thread.work_dirs().cloned();
                    SerializedActiveThread {
                        session_id: (!is_draft_active).then(|| thread.session_id().0.to_string()),
                        thread_id: active_thread_id,
                        agent_type: active_thread_agent.clone(),
                        title: title.map(|t| t.to_string()),
                        work_dirs: work_dirs.map(|dirs| dirs.serialize()),
                    }
                })
                .or_else(|| {
                    if let Some(thread_id) = active_thread_id
                        && let Some(metadata) = ThreadMetadataStore::try_global(cx)
                            .and_then(|store| store.read(cx).entry(thread_id).cloned())
                            .filter(|metadata| metadata.is_draft())
                    {
                        return Some(SerializedActiveThread {
                            session_id: None,
                            thread_id: Some(thread_id),
                            agent_type: active_thread_agent.clone(),
                            title: metadata.title.as_ref().map(|title| title.to_string()),
                            work_dirs: Some(metadata.folder_paths().serialize()),
                        });
                    }
                    // The active view may be in `Loading` or `LoadError` — for
                    // example, while a restored thread is waiting for a custom
                    // agent to finish registering. Without this fallback, a
                    // stray `serialize()` triggered during that window would
                    // write `session_id=None` and wipe the restored session
                    if is_draft_active {
                        return None;
                    }
                    let conversation_view = self.active_conversation_view()?;
                    let session_id = conversation_view.read(cx).root_session_id.clone()?;
                    let metadata = ThreadMetadataStore::try_global(cx)
                        .and_then(|store| store.read(cx).entry_by_session(&session_id).cloned());
                    Some(SerializedActiveThread {
                        session_id: Some(session_id.0.to_string()),
                        thread_id: active_thread_id,
                        agent_type: active_thread_agent.clone(),
                        title: metadata
                            .as_ref()
                            .and_then(|m| m.title.as_ref())
                            .map(|t| t.to_string()),
                        work_dirs: metadata.map(|m| m.folder_paths().serialize()),
                    })
                })
        };

        let new_draft_thread_id = self
            .draft_thread
            .as_ref()
            .map(|draft| draft.read(cx).thread_id);

        let kvp = KeyValueStore::global(cx);
        self.pending_serialization = Some(cx.background_spawn(async move {
            save_serialized_panel(
                workspace_id,
                SerializedAgentPanel {
                    selected_agent: Some(selected_agent),
                    last_created_entry_kind,
                    last_active_thread,
                    last_active_terminal_id,
                    new_draft_thread_id,
                },
                kvp,
            )
            .await?;
            anyhow::Ok(())
        }));
    }

    pub fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> Task<Result<Entity<Self>>> {
        let kvp = cx.update(|_window, cx| KeyValueStore::global(cx)).ok();
        cx.spawn(async move |cx| {
            let workspace_id = workspace
                .read_with(cx, |workspace, _| workspace.database_id())
                .ok()
                .flatten();

            let (serialized_panel, global_last_used_agent, global_last_created_entry_kind) = cx
                .background_spawn(async move {
                    match kvp {
                        Some(kvp) => {
                            let panel = workspace_id
                                .and_then(|id| read_serialized_panel(id, &kvp))
                                .or_else(|| read_legacy_serialized_panel(&kvp));
                            let global_agent = read_global_last_used_agent(&kvp);
                            let global_entry_kind = read_global_last_created_entry_kind(&kvp);
                            (panel, global_agent, global_entry_kind)
                        }
                        None => (None, None, None),
                    }
                })
                .await;

            let has_open_project = workspace
                .read_with(cx, |workspace, cx| !workspace.root_paths(cx).is_empty())
                .unwrap_or(false);
            let terminal_id_to_restore = if has_open_project {
                serialized_panel
                    .as_ref()
                    .and_then(|panel| panel.last_active_terminal_id.as_deref())
                    .and_then(|terminal_id| {
                        match TerminalId::from_key_string(terminal_id) {
                            Ok(terminal_id) => Some(terminal_id),
                            Err(error) => {
                                log::warn!("failed to parse last active terminal id: {error}");
                                None
                            }
                        }
                    })
            } else {
                None
            };
            let terminal_to_restore = if let Some(terminal_id) = terminal_id_to_restore {
                match cx.update(|_window, cx| {
                    TerminalThreadMetadataStore::try_global(cx).map(|store| {
                        let reload_task = store.read(cx).reload_task();
                        (store, reload_task)
                    })
                }) {
                    Ok(Some((store, reload_task))) => {
                        reload_task.await;
                        match store
                            .read_with(cx, |store, _cx| store.entry(terminal_id).cloned())
                        {
                            Some(metadata) => Some(metadata),
                            None => {
                                log::info!(
                                    "last active terminal is missing, skipping restoration"
                                );
                                None
                            }
                        }
                    }
                    Ok(None) => {
                        log::warn!("failed to restore active terminal: metadata store missing");
                        None
                    }
                    Err(err) => {
                        log::warn!("failed to access terminal metadata store: {err}");
                        None
                    }
                }
            } else {
                None
            };

            let thread_to_restore = if has_open_project && terminal_to_restore.is_none() {
                if let Some(info) = serialized_panel
                    .as_ref()
                    .and_then(|panel| panel.last_active_thread.as_ref())
                {
                    match cx.update(|_window, cx| {
                        ThreadMetadataStore::try_global(cx).map(|store| {
                            let reload_task = store.read(cx).reload_task();
                            (store, reload_task)
                        })
                    }) {
                        Ok(Some((store, reload_task))) => {
                            reload_task.await;
                            let thread_id = store.read_with(cx, |store, _cx| {
                                let primary = info.thread_id.and_then(|tid| store.entry(tid));
                                let fallback = info.session_id.as_ref().and_then(|sid| {
                                    store.entry_by_session(&acp::SessionId::new(sid.clone()))
                                });
                                primary
                                    .or(fallback)
                                    .filter(|entry| !entry.archived)
                                    .map(|entry| entry.thread_id)
                            });
                            match thread_id {
                                Some(thread_id) => Some((info, thread_id)),
                                None => {
                                    log::info!(
                                        "last active thread is archived or missing, skipping restoration"
                                    );
                                    None
                                }
                            }
                        }
                        Ok(None) => {
                            log::warn!("failed to restore active thread: metadata store missing");
                            None
                        }
                        Err(err) => {
                            log::warn!("failed to access thread metadata store: {err}");
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let panel = workspace.update_in(cx, |workspace, window, cx| {
                let panel = cx.new(|cx| Self::new(workspace, window, cx));

                panel.update(cx, |panel, cx| {
                    let is_via_collab = panel.project.read(cx).is_via_collab();
                    // Collab workspaces only support NativeAgent; clamp any
                    // non-native choice so `set_active` can't bypass the
                    // collab guard in `external_thread`.
                    let clamp = |agent: Agent| {
                        if is_via_collab && !agent.is_native() {
                            Agent::NativeAgent
                        } else {
                            agent
                        }
                    };
                    let global_fallback =
                        global_last_used_agent.filter(|agent| !is_via_collab || agent.is_native());

                    if let Some(serialized_panel) = &serialized_panel {
                        panel.last_created_entry_kind = serialized_panel.last_created_entry_kind;
                    } else if let Some(entry_kind) = global_last_created_entry_kind {
                        panel.last_created_entry_kind = entry_kind;
                    }

                    // The thread being restored may have been bound to an
                    // agent different from the panel's last selected one
                    // (e.g. a draft created while a different agent was
                    // active). When restoring a thread, prefer its agent
                    // so the draft survives reload bound to the right
                    // backend; otherwise fall back to the serialized
                    // selection, then the global last-used agent.
                    let initial_agent = match &thread_to_restore {
                        Some((info, _)) => Some(clamp(info.agent_type.clone())),
                        None => serialized_panel
                            .as_ref()
                            .and_then(|p| p.selected_agent.clone())
                            .map(clamp)
                            .or(global_fallback),
                    };
                    if let Some(agent) = initial_agent {
                        panel.selected_agent = agent;
                    }

                    if let Some(metadata) = terminal_to_restore {
                        panel.restore_terminal_for_panel_load(
                            metadata,
                            false,
                            AgentThreadSource::AgentPanel,
                            Some(workspace),
                            window,
                            cx,
                        );
                    } else if let Some((info, thread_id)) = thread_to_restore {
                        let agent = panel.selected_agent.clone();
                        panel.load_agent_thread(
                            agent,
                            thread_id,
                            info.work_dirs.as_ref().map(PathList::deserialize),
                            info.title.clone().map(Into::into),
                            false,
                            AgentThreadSource::AgentPanel,
                            window,
                            cx,
                        );
                    }
                    if let Some(new_draft_thread_id) = serialized_panel
                        .as_ref()
                        .and_then(|p| p.new_draft_thread_id)
                    {
                        panel.restore_new_draft(new_draft_thread_id, window, cx);
                    }
                    cx.notify();
                });

                panel
            })?;

            Ok(panel)
        })
    }

    pub(crate) fn new(workspace: &Workspace, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let fs = workspace.app_state().fs.clone();
        let user_store = workspace.app_state().user_store.clone();
        let project = workspace.project();
        let language_registry = project.read(cx).languages().clone();
        let client = workspace.client().clone();
        let workspace_id = workspace.database_id();
        let workspace = workspace.weak_handle();

        let context_server_registry =
            cx.new(|cx| ContextServerRegistry::new(project.read(cx).context_server_store(), cx));

        let thread_store = ThreadStore::global(cx);

        let base_view = BaseView::Uninitialized;

        let weak_panel = cx.entity().downgrade();
        let onboarding = cx.new(|cx| {
            AgentPanelOnboarding::new(
                user_store.clone(),
                client,
                move |_window, cx| {
                    weak_panel
                        .update(cx, |panel, cx| {
                            panel.dismiss_ai_onboarding(cx);
                        })
                        .ok();
                },
                cx,
            )
        });

        // Subscribe to extension events to sync agent servers when extensions change
        let extension_subscription = ExtensionStore::try_global(cx).map(|store| {
            cx.subscribe(&store, |this, _source, event, cx| match event {
                extension_host::Event::ExtensionUninstalled(id) => {
                    this.migrate_agent_server_from_extensions(id.clone(), cx);
                }
                _ => {}
            })
        });

        let connection_store = cx.new(|cx| AgentConnectionStore::new(project.clone(), cx));
        if let Some(store) = agent::recipes::RecipeStore::try_global(cx) {
            cx.subscribe_in(&store, window, |this, _, event, window, cx| match event {
                agent::recipes::RecipeStoreEvent::Review { project, proposal }
                    if project == &this.project =>
                {
                    this.open_recipe_management(None, Some(proposal.clone()), None, window, cx);
                }
                agent::recipes::RecipeStoreEvent::Manage { project, schedule }
                    if project == &this.project =>
                {
                    this.manage_recipes(*schedule, window, cx);
                }
                agent::recipes::RecipeStoreEvent::OpenSession {
                    project,
                    session_id,
                } if project == &this.project => {
                    this.open_thread(session_id.clone(), None, None, window, cx);
                }
                agent::recipes::RecipeStoreEvent::RunFinished { schedule, status }
                    if this.project.read(cx).worktree_paths(cx).folder_path_list()
                        == &PathList::deserialize(&schedule.project_paths)
                        && AgentSettings::get_global(cx).notify_when_agent_waiting
                            != NotifyWhenAgentWaiting::Never =>
                {
                    let id = schedule.id;
                    let title = format!("{}: {}", schedule.title, status.label());
                    let panel = cx.entity().downgrade();
                    this.workspace
                        .update(cx, |workspace, cx| {
                            let toast = StatusToast::new(title, cx, |toast, _| {
                                toast.dismiss_button(true).action(
                                    "View run history",
                                    move |window, cx| {
                                        panel
                                            .update(cx, |panel, cx| {
                                                panel.manage_recipes(Some(id), window, cx)
                                            })
                                            .log_err();
                                    },
                                )
                            });
                            workspace.toggle_status_toast(toast, cx);
                        })
                        .log_err();
                    window.request_attention();
                }
                _ => {}
            })
            .detach();
        }
        let _project_subscription =
            cx.subscribe(&project, |this, _project, event, cx| match event {
                project::Event::WorktreeAdded(_)
                | project::Event::WorktreeRemoved(_)
                | project::Event::WorktreeOrderChanged
                | project::Event::WorktreePathsChanged { .. } => {
                    this.ensure_native_agent_connection(cx);
                    this.update_thread_work_dirs(cx);
                    this.persist_all_terminal_metadata(cx);
                    cx.notify();
                }
                _ => {}
            });

        let _thread_metadata_store_subscription = cx.subscribe(
            &ThreadMetadataStore::global(cx),
            |this, _store, event, cx| {
                let ThreadMetadataStoreEvent::ThreadArchived(thread_id) = event;
                if this.retained_threads.remove(thread_id).is_some() {
                    cx.notify();
                }
            },
        );

        #[cfg(feature = "multiplayer-tools")]
        cx.on_app_quit(|this, cx| {
            let flush = this.flush_inline_terminal_history(cx);
            async move {
                if let Err(error) = flush.await {
                    log::error!("Failed to flush inline terminal history before quit: {error:#}");
                }
            }
        })
        .detach();

        cx.on_release(|this, cx| {
            this.dismiss_all_terminal_notifications(cx);
            if let Some(task) = this.pending_last_used_agent_persistence.take() {
                task.detach();
            }
        })
        .detach();

        let panel = Self {
            workspace_id,
            base_view,
            last_created_entry_kind: AgentPanelEntryKind::Thread,
            workspace,
            user_store,
            project: project.clone(),
            fs: fs.clone(),
            language_registry,
            connection_store,
            focus_handle: cx.focus_handle(),
            context_server_registry,
            draft_thread: None,
            retained_threads: HashMap::default(),
            terminals: HashMap::default(),
            pending_terminal_spawn: None,
            #[cfg(feature = "multiplayer-tools")]
            pending_inline_terminal_open: None,
            new_thread_menu_handle: PopoverMenuHandle::default(),
            agent_panel_menu_handle: PopoverMenuHandle::default(),

            _extension_subscription: extension_subscription,
            _doctor_subscription: None,
            _doctor_context_subscription: None,
            _project_subscription,
            zoomed: false,
            pending_serialization: None,
            pending_last_used_agent_persistence: None,
            new_user_onboarding: onboarding,
            thread_store,
            selected_agent: Agent::default(),
            _thread_view_subscription: None,
            _active_thread_focus_subscription: None,
            new_user_onboarding_upsell_dismissed: AtomicBool::new(OnboardingUpsell::dismissed(cx)),
            _base_view_observation: None,
            _draft_editor_observation: None,
            _active_draft_reclaim_observation: None,
            _thread_metadata_store_subscription,
            last_context_source: None,
            is_active: false,
        };

        panel.ensure_native_agent_connection(cx);
        panel
    }

    pub fn open_settings(
        workspace: &mut Workspace,
        _: &OpenSettings,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        if let Some(panel) = workspace.panel::<Self>(cx) {
            Self::reveal_conversation(workspace, true, window, cx);
            panel.update(cx, |panel, cx| panel.open_configuration(window, cx));
        }
    }

    pub fn toggle_focus(
        workspace: &mut Workspace,
        _: &ToggleFocus,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        if workspace
            .panel::<Self>(cx)
            .is_some_and(|panel| panel.read(cx).enabled(cx))
        {
            #[cfg(feature = "multiplayer-tools")]
            if workspace.workspace_presentation() == workspace::WorkspacePresentation::Collaborative
            {
                Self::reveal_conversation(workspace, true, window, cx);
                return;
            }
            workspace.toggle_panel_focus::<Self>(window, cx);
        }
    }

    pub fn focus(
        workspace: &mut Workspace,
        _: &FocusAgent,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        if workspace
            .panel::<Self>(cx)
            .is_some_and(|panel| panel.read(cx).enabled(cx))
        {
            Self::reveal_conversation(workspace, true, window, cx);
        }
    }

    pub fn toggle(
        workspace: &mut Workspace,
        _: &Toggle,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        if workspace
            .panel::<Self>(cx)
            .is_some_and(|panel| panel.read(cx).enabled(cx))
        {
            #[cfg(feature = "multiplayer-tools")]
            if workspace.workspace_presentation() == workspace::WorkspacePresentation::Collaborative
            {
                Self::reveal_conversation(workspace, true, window, cx);
                return;
            }
            if !workspace.toggle_panel_focus::<Self>(window, cx) {
                workspace.close_panel::<Self>(window, cx);
            }
        }
    }

    pub fn thread_store(&self) -> &Entity<ThreadStore> {
        &self.thread_store
    }

    pub fn connection_store(&self) -> &Entity<AgentConnectionStore> {
        &self.connection_store
    }

    pub fn selected_agent(&self, cx: &App) -> Agent {
        if self.project.read(cx).is_via_collab() {
            Agent::NativeAgent
        } else {
            self.selected_agent.clone()
        }
    }

    pub fn open_thread(
        &mut self,
        session_id: acp::SessionId,
        work_dirs: Option<PathList>,
        title: Option<SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Share links / clipboard imports enter with only a session id. If
        // this machine already has a metadata row for the session, route
        // through the normal thread-id path.
        let existing_thread_id = ThreadMetadataStore::try_global(cx).and_then(|store| {
            store
                .read(cx)
                .entry_by_session(&session_id)
                .map(|m| m.thread_id)
        });
        if let Some(thread_id) = existing_thread_id {
            self.load_agent_thread(
                crate::Agent::NativeAgent,
                thread_id,
                work_dirs,
                title,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        } else {
            self.external_thread_by_session(
                crate::Agent::NativeAgent,
                session_id,
                work_dirs,
                title,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        }
    }

    fn external_thread_by_session(
        &mut self,
        agent: Agent,
        session_id: acp::SessionId,
        work_dirs: Option<PathList>,
        title: Option<SharedString>,
        focus: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let thread = self.create_agent_thread_with_server_for_external_session(
            agent, None, session_id, work_dirs, title, None, source, window, cx,
        );
        self.set_base_view(thread.into(), focus, window, cx);
    }

    pub(crate) fn context_server_registry(&self) -> &Entity<ContextServerRegistry> {
        &self.context_server_registry
    }

    pub fn is_visible(workspace: &Entity<Workspace>, cx: &App) -> bool {
        let workspace_read = workspace.read(cx);

        workspace_read
            .panel::<AgentPanel>(cx)
            .map(|panel| {
                let panel_id = Entity::entity_id(&panel);

                workspace_read.all_docks().iter().any(|dock| {
                    dock.read(cx)
                        .visible_panel()
                        .is_some_and(|visible_panel| visible_panel.panel_id() == panel_id)
                })
            })
            .unwrap_or(false)
    }

    /// Clear the active view, retaining any running thread in the background.
    pub fn clear_base_view(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let old_view = std::mem::replace(&mut self.base_view, BaseView::Uninitialized);
        self.retain_running_thread(old_view, cx);
        self.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
        self.serialize(cx);
        cx.emit(AgentPanelEvent::ActiveViewChanged);
        cx.notify();
    }

    pub fn new_thread(&mut self, _action: &NewThread, window: &mut Window, cx: &mut Context<Self>) {
        if !self.has_open_project(cx) {
            return;
        }

        self.new_thread_with_workspace(None, window, cx);
    }

    fn new_thread_with_workspace(
        &mut self,
        workspace: Option<&Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.should_create_terminal_for_new_entry(cx) {
            self.new_terminal(workspace, AgentThreadSource::AgentPanel, window, cx);
        } else {
            self.activate_new_thread(true, AgentThreadSource::AgentPanel, window, cx);
        }
    }

    pub fn activate_new_thread(
        &mut self,
        focus: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.has_open_project(cx) {
            return;
        }

        self.set_last_created_entry_kind_from_user_action(AgentPanelEntryKind::Thread, cx);

        // If the user is viewing a *parked* draft and the ephemeral
        // new-draft slot is occupied, pressing `+` should just focus the
        // ephemeral draft — not park it and create yet another empty one.
        // This matches the mental model of `+` as "go to my new-thread
        // slot". The parked draft will be put back into `retained_threads`
        // by `set_base_view`'s `retain_running_thread` call.
        if let Some(draft) = self.draft_thread.clone()
            && self.active_thread_is_draft(cx)
            && !self.active_view_is_new_draft(cx)
            && *draft.read(cx).agent_key() == self.selected_agent
        {
            self.set_base_view(
                BaseView::AgentThread {
                    conversation_view: draft,
                },
                focus,
                window,
                cx,
            );
            return;
        }

        if let Some(draft) = self.draft_thread.clone() {
            if self.draft_has_content(&draft, cx) {
                let draft_id = draft.read(cx).thread_id;
                self.draft_thread = None;
                self._draft_editor_observation = None;
                self.retained_threads.insert(draft_id, draft);
            } else if *draft.read(cx).agent_key() != self.selected_agent {
                let old_draft_id = draft.read(cx).thread_id;
                self.discard_matching_draft_prompt(old_draft_id, cx);
                ThreadMetadataStore::global(cx).update(cx, |store, cx| {
                    store.delete(old_draft_id, cx);
                });
                self.draft_thread = None;
                self._draft_editor_observation = None;
            }
        }
        self.activate_draft(focus, source, window, cx);
    }

    fn draft_has_content(&self, draft: &Entity<ConversationView>, cx: &App) -> bool {
        let cv = draft.read(cx);
        if cv.has_pending_restored_draft() {
            return true;
        }
        if let Some(thread_view) = cv.active_thread() {
            let text = thread_view.read(cx).message_editor.read(cx).text(cx);
            if !text.trim().is_empty() {
                return true;
            }
        }
        if let Some(acp_thread) = cv.root_thread(cx) {
            let thread = acp_thread.read(cx);
            if !thread.is_draft_thread() {
                return true;
            }
            if thread
                .draft_prompt()
                .is_some_and(|blocks| !blocks.is_empty())
            {
                return true;
            }
        }
        false
    }

    /// Reattaches the panel's new-draft slot to the persisted `thread_id`,
    /// seeding the editor with any prompt text from the draft-prompt kvp
    /// store.
    ///
    /// If the active view already holds this thread — because the user's
    /// last-active thread was the new-draft itself — we reuse that
    /// ConversationView instead of building a second one.
    fn restore_new_draft(
        &mut self,
        thread_id: ThreadId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.has_open_project(cx) {
            return;
        }

        let active_matching = match &self.base_view {
            BaseView::AgentThread { conversation_view }
                if conversation_view.read(cx).thread_id == thread_id =>
            {
                Some(conversation_view.clone())
            }
            _ => None,
        };
        if let Some(conversation_view) = active_matching {
            self.observe_draft_editor(&conversation_view, cx);
            self.draft_thread = Some(conversation_view);
            return;
        }

        let Some(metadata) = ThreadMetadataStore::try_global(cx)
            .and_then(|store| store.read(cx).entry(thread_id).cloned())
            .filter(|m| m.is_draft())
        else {
            return;
        };

        let agent = if self.project.read(cx).is_via_collab() {
            Agent::NativeAgent
        } else {
            Agent::from(metadata.agent_id.clone())
        };
        let restored = match crate::draft_prompt_store::try_read_for_restore(thread_id, cx) {
            Ok(restored) => restored,
            Err(error) => {
                self.report_draft_restore_error(error, cx);
                return;
            }
        };
        let initial_content = restored.map(|blocks| AgentInitialContent::RestoredDraft { blocks });
        let thread = self.create_agent_thread_with_server(
            agent,
            None,
            Some(thread_id),
            Some(metadata.folder_paths().clone()),
            metadata.title.clone(),
            initial_content,
            None,
            AgentThreadSource::AgentPanel,
            window,
            cx,
        );
        self.observe_draft_editor(&thread.conversation_view, cx);
        self.draft_thread = Some(thread.conversation_view);
    }

    pub fn new_external_agent_thread(
        &mut self,
        action: &NewExternalAgentThread,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.has_open_project(cx) {
            return;
        }

        self.selected_agent = action.agent.clone().into();
        self.activate_new_thread(true, AgentThreadSource::AgentPanel, window, cx);
    }

    fn set_selected_agent_and_persist(&mut self, agent: Agent, cx: &mut Context<Self>) {
        if self.selected_agent != agent {
            self.selected_agent = agent.clone();
            self.serialize(cx);
        }

        let previous_persistence = self.pending_last_used_agent_persistence.take();
        self.pending_last_used_agent_persistence = Some(cx.background_spawn({
            let kvp = KeyValueStore::global(cx);
            async move {
                if let Some(previous_persistence) = previous_persistence {
                    previous_persistence.await;
                }
                write_global_last_used_agent(kvp, agent).await;
            }
        }));
    }

    /// Sets the panel's selected agent without opening the panel or focusing
    /// it, so the agent is launched the next time the panel is opened (or
    /// right away, if the panel is already showing the empty new-thread
    /// draft).
    pub fn select_agent(&mut self, agent: Agent, window: &mut Window, cx: &mut Context<Self>) {
        if self.project.read(cx).is_via_collab() && !agent.is_native() {
            return;
        }

        let showing_new_draft = matches!(
            (&self.base_view, &self.draft_thread),
            (BaseView::AgentThread { conversation_view }, Some(draft))
                if conversation_view.entity_id() == draft.entity_id()
        );

        if matches!(self.base_view, BaseView::AgentThread { .. }) && showing_new_draft {
            self.set_selected_agent_and_persist(agent, cx);
            self.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
            cx.notify();
        }
    }

    pub fn new_terminal(
        &mut self,
        workspace: Option<&Workspace>,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.supports_terminal(cx) {
            return;
        }
        #[cfg(feature = "multiplayer-tools")]
        if workspace
            .map(Workspace::workspace_presentation)
            .or_else(|| {
                // Workspace actions already hold its entity's mutable context.
                self.workspace
                    .upgrade()
                    .map(|workspace| workspace.read(cx).workspace_presentation())
            })
            == Some(settings::WorkspacePresentation::Collaborative)
        {
            self.open_conversation_terminal(workspace, window, cx);
            return;
        }
        self.set_last_created_entry_kind_from_user_action(AgentPanelEntryKind::Terminal, cx);
        let working_directory = self.terminal_working_directory(workspace, cx);
        self.spawn_terminal(
            TerminalId::new(),
            working_directory,
            None,
            None,
            None,
            TerminalSpawnAssociation::Standalone,
            true,
            true,
            true,
            source,
            window,
            cx,
        )
        .detach_and_log_err(cx);
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn inline_terminal_view(
        &self,
        terminal_id: TerminalId,
    ) -> Option<Entity<TerminalView>> {
        self.terminals
            .get(&terminal_id)
            .filter(|terminal| terminal.inline_history_owner.is_some())
            .map(|terminal| terminal.view.clone())
    }

    #[cfg(feature = "multiplayer-tools")]
    fn open_conversation_terminal(
        &mut self,
        workspace: Option<&Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.pending_inline_terminal_open = None;
        let Some(parent) = self.active_thread_view(cx) else {
            self.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
            if self.active_thread_view(cx).is_some() {
                self.open_conversation_terminal(workspace, window, cx);
                return;
            }
            if let Some(conversation) = self.active_conversation_view().cloned() {
                self.pending_inline_terminal_open = Some(cx.subscribe_in(
                    &conversation,
                    window,
                    move |this, conversation, _: &RootThreadUpdated, window, cx| {
                        if conversation.read(cx).root_thread_view().is_none() {
                            return;
                        }
                        this.pending_inline_terminal_open = None;
                        if this
                            .active_conversation_view()
                            .is_some_and(|active| active == conversation)
                        {
                            this.open_conversation_terminal(None, window, cx);
                        }
                    },
                ));
            }
            return;
        };
        let context = match parent.read(cx).inline_terminal_context(cx) {
            Ok(context) => context,
            Err(error) => {
                self.report_inline_terminal_history_error(&error, cx);
                return;
            }
        };
        if let Err(error) = parent.update(cx, |parent, cx| {
            parent.persist_inline_terminal_positions(cx)
        }) {
            self.report_inline_terminal_history_error(&error, cx);
            return;
        }
        let (spawn, reused) = match self.prepare_inline_terminal_reuse(context.clone(), cx) {
            Ok(Some((terminal_id, terminal, history, retained, previous, created))) => (
                Task::ready(Ok((terminal_id, terminal))),
                Some((history, retained, previous, created)),
            ),
            Ok(None) => (
                self.spawn_inline_terminal(context, workspace, window, cx),
                None,
            ),
            Err(error) => {
                self.report_inline_terminal_history_error(&error, cx);
                return;
            }
        };

        cx.spawn_in(window, async move |this, cx| {
            let result = async {
                let (terminal_id, terminal) = spawn.await?;
                let attached = parent
                    .update_in(cx, |parent, window, cx| {
                        if let Some((history, retained, _, created)) = &reused {
                            anyhow::ensure!(
                                TerminalThreadMetadataStore::try_global(cx)
                                    .is_some_and(|owner| owner.entity_id() == history.entity_id()),
                                "The live inline terminal's history owner has changed"
                            );
                            anyhow::ensure!(
                                history.read(cx).entry(terminal_id) == Some(retained),
                                "The live inline terminal's history changed before attachment"
                            );
                            history.update(cx, |history, cx| {
                                history.save_inline(created.clone(), cx)
                            })?;
                        }
                        let attached =
                            parent.attach_inline_terminal(terminal_id, terminal, window, cx);
                        if let Err(error) = attached {
                            if let Some((history, _, previous, created)) = &reused {
                                anyhow::ensure!(
                                    TerminalThreadMetadataStore::try_global(cx).is_some_and(|owner| {
                                        owner.entity_id() == history.entity_id()
                                    }) && history.read(cx).entry(terminal_id) == Some(created),
                                    "Inline terminal attachment failed ({error:#}); its history changed before rollback"
                                );
                                history.update(cx, |history, cx| {
                                    history.save_inline(previous.clone(), cx)
                                })?;
                                if let Err(reconcile_error) =
                                    parent.sync_inline_terminal_history(window, cx)
                                {
                                    return Err(anyhow!(
                                        "Inline terminal attachment failed ({error:#}); rolled-back history reconciliation failed: {reconcile_error:#}"
                                    ));
                                }
                            }
                            return Err(error);
                        }
                        anyhow::Ok(())
                    })
                    .and_then(|result| result);
                if let Err(error) = attached {
                    if reused.is_none() {
                        this.update(cx, |panel, cx| {
                            if let Err(stop_error) =
                                panel.request_stop_inline_terminal(terminal_id, cx)
                            {
                                panel.report_inline_terminal_history_error(&stop_error, cx);
                            }
                        })?;
                    }
                    return Err(error);
                }
                this.update(cx, |_, cx| cx.notify())?;
                anyhow::Ok(())
            }
            .await;
            if let Err(error) = result {
                this.update(cx, |panel, cx| {
                    panel.report_inline_terminal_history_error(&error, cx)
                })?;
                return Err(error);
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    #[cfg(feature = "multiplayer-tools")]
    fn prepare_inline_terminal_reuse(
        &mut self,
        mut context: InlineTerminalContext,
        cx: &mut Context<Self>,
    ) -> Result<
        Option<(
            TerminalId,
            Entity<TerminalView>,
            Entity<TerminalThreadMetadataStore>,
            TerminalThreadMetadata,
            TerminalThreadMetadata,
            TerminalThreadMetadata,
        )>,
    > {
        let history = TerminalThreadMetadataStore::try_global(cx)
            .ok_or_else(|| anyhow!("Terminal history storage is unavailable"))?;
        history.read(cx).validate_inline_context(&context)?;
        let parent = self
            .active_thread_view(cx)
            .ok_or_else(|| anyhow!("The inline terminal parent session is unavailable"))?;
        let parent = parent.read(cx);
        anyhow::ensure!(
            self.active_thread_id(cx) == Some(context.thread_id)
                && parent.root_thread_id == context.thread_id
                && context.session_id.as_ref() == Some(&parent.session_id)
                && parent
                    .project
                    .upgrade()
                    .is_some_and(|project| project.entity_id() == self.project.entity_id()),
            "The inline terminal parent conversation or project no longer matches"
        );
        anyhow::ensure!(
            self.project.read(cx).supports_terminal(cx),
            "This project does not support an interactive terminal"
        );
        if let Some(position) = context.draft_position.as_ref() {
            position.byte_offset_for_source(&parent.message_editor.read(cx).text(cx))?;
        }
        let project_remote = self.project.read(cx).remote_connection_options(cx);
        let mut eligible = Vec::new();
        for (terminal_id, terminal) in &self.terminals {
            if terminal.inline_history_owner.is_none()
                || !terminal
                    .view
                    .read(cx)
                    .terminal()
                    .read(cx)
                    .can_reuse_interactive_process()
            {
                continue;
            }
            let owner = terminal
                .inline_history_owner
                .as_ref()
                .and_then(WeakEntity::upgrade)
                .ok_or_else(|| anyhow!("A live inline terminal's history owner is unavailable"))?;
            let metadata = owner
                .read(cx)
                .entry(*terminal_id)
                .ok_or_else(|| anyhow!("A live inline terminal's history is unavailable"))?;
            let current = metadata
                .inline_context
                .as_ref()
                .ok_or_else(|| anyhow!("A live inline terminal's conversation is unavailable"))?;
            if current.thread_id != context.thread_id || current.session_id != context.session_id {
                continue;
            }
            anyhow::ensure!(
                owner.entity_id() == history.entity_id(),
                "The live inline terminal's history owner has changed"
            );
            anyhow::ensure!(
                remote::same_remote_connection_identity(
                    metadata.remote_connection.as_ref(),
                    project_remote.as_ref()
                ),
                "The live inline terminal belongs to a different remote project"
            );
            eligible.push((*terminal_id, terminal.view.clone()));
        }
        let [(terminal_id, view)] = eligible.as_slice() else {
            // Multiple eligible executions retain the existing explicit-spawn policy.
            return Ok(None);
        };
        let (owner, mut previous) = self.inline_terminal_snapshot_metadata(*terminal_id, cx)?;
        let retained = owner
            .read(cx)
            .entry(*terminal_id)
            .cloned()
            .ok_or_else(|| anyhow!("The live inline terminal's history is unavailable"))?;
        let current = previous
            .inline_context
            .as_mut()
            .ok_or_else(|| anyhow!("The terminal conversation association is unavailable"))?;
        current.validate_occurrences()?;
        let mut frozen = current.clone();
        frozen.previous_occurrences.clear();
        frozen
            .snapshot
            .as_ref()
            .ok_or_else(|| anyhow!("The terminal occurrence has no styled snapshot"))?
            .validate()?;
        context.occurrence_id = Some(uuid::Uuid::new_v4());
        context.previous_occurrences = current.previous_occurrences.clone();
        context.previous_occurrences.push(frozen);
        context.output = current.output.clone();
        context.snapshot = current.snapshot.clone();
        let mut created = previous.clone();
        created.inline_context = Some(context);
        // Commit and attachment must share one update so queued position observers cannot
        // change the created aggregate before an exact-preimage failure rollback.
        Ok(Some((
            *terminal_id,
            view.clone(),
            owner,
            retained,
            previous,
            created,
        )))
    }

    #[cfg(feature = "multiplayer-tools")]
    pub fn spawn_inline_terminal(
        &mut self,
        context: InlineTerminalContext,
        workspace: Option<&Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<(TerminalId, Entity<TerminalView>)>> {
        if self.active_thread_id(cx) != Some(context.thread_id) {
            return Task::ready(Err(anyhow!(
                "The inline terminal's parent conversation is no longer active"
            )));
        }
        let Some(parent) = self.active_thread_view(cx) else {
            return Task::ready(Err(anyhow!(
                "The inline terminal's parent session is unavailable"
            )));
        };
        if context.session_id.as_ref() != Some(&parent.read(cx).session_id)
            || parent.read(cx).root_thread_id != context.thread_id
        {
            return Task::ready(Err(anyhow!(
                "The inline terminal's parent session no longer matches"
            )));
        }
        let Some(store) = TerminalThreadMetadataStore::try_global(cx) else {
            return Task::ready(Err(anyhow!("Terminal history storage is unavailable")));
        };
        if let Err(error) = store.read(cx).validate_inline_context(&context) {
            return Task::ready(Err(error));
        }
        if !self.project.read(cx).supports_terminal(cx) {
            return Task::ready(Err(anyhow!(
                "This project does not support an interactive terminal"
            )));
        }
        let terminal_id = TerminalId::new();
        let working_directory = self.terminal_working_directory(workspace, cx);
        let association = TerminalSpawnAssociation::Inline(Box::new(InlineTerminalSpawnContext {
            parent,
            context,
            history_owner: store,
            project_id: self.project.entity_id(),
            remote_connection: self.project.read(cx).remote_connection_options(cx),
            init_command: Self::terminal_init_command(true, cx),
        }));
        let spawn = self.spawn_terminal(
            terminal_id,
            working_directory,
            None,
            None,
            None,
            association,
            false,
            false,
            false,
            AgentThreadSource::AgentPanel,
            window,
            cx,
        );
        cx.spawn_in(window, async move |_this, _cx| {
            anyhow::Ok((terminal_id, spawn.await?))
        })
    }

    #[cfg(feature = "multiplayer-tools")]
    fn validate_inline_terminal_parent(
        &self,
        association: &InlineTerminalSpawnContext,
        cx: &App,
    ) -> Result<()> {
        let parent = association.parent.read(cx);
        let parent_is_retained = self
            .conversation_for_review_thread(&parent.thread, cx)
            .is_some_and(|conversation| {
                conversation.read(cx).thread_id == association.context.thread_id
            });
        let project_matches = self.project.entity_id() == association.project_id
            && parent
                .project
                .upgrade()
                .is_some_and(|project| project.entity_id() == association.project_id)
            && remote::same_remote_connection_identity(
                self.project.read(cx).remote_connection_options(cx).as_ref(),
                association.remote_connection.as_ref(),
            );
        let history_owner_matches = TerminalThreadMetadataStore::try_global(cx)
            .is_some_and(|current| current.entity_id() == association.history_owner.entity_id());
        if !parent_is_retained
            || !project_matches
            || !history_owner_matches
            || association.context.session_id.as_ref() != Some(&parent.session_id)
        {
            return Err(anyhow!(
                "The inline terminal's original conversation or history owner is no longer available"
            ));
        }
        association
            .history_owner
            .read(cx)
            .validate_inline_context(&association.context)?;
        if association.context.is_draft {
            anyhow::ensure!(
                association.context.preceding_entry_count == parent.thread.read(cx).entries().len(),
                "The conversation changed while the inline terminal was opening; try opening it again"
            );
            if let Some(position) = association.context.draft_position.as_ref() {
                position.byte_offset_for_source(&parent.message_editor.read(cx).text(cx))?;
            }
        }
        Ok(())
    }

    fn terminal_working_directory(
        &self,
        workspace: Option<&Workspace>,
        cx: &App,
    ) -> Option<PathBuf> {
        workspace
            .map(|workspace| terminal_view::default_working_directory(workspace, cx))
            .unwrap_or_else(|| self.default_terminal_working_directory(cx))
    }

    pub fn supports_terminal(&self, cx: &App) -> bool {
        self.has_open_project(cx) && self.project.read(cx).supports_terminal(cx)
    }

    pub fn should_create_terminal_for_new_entry(&self, cx: &App) -> bool {
        self.last_created_entry_kind == AgentPanelEntryKind::Terminal
            && self.project.read(cx).supports_terminal(cx)
    }

    fn set_last_created_entry_kind_from_user_action(
        &mut self,
        entry_kind: AgentPanelEntryKind,
        cx: &mut Context<Self>,
    ) {
        if self.last_created_entry_kind != entry_kind {
            self.last_created_entry_kind = entry_kind;
            self.serialize(cx);
        }

        cx.background_spawn({
            let kvp = KeyValueStore::global(cx);
            async move {
                write_global_last_created_entry_kind(kvp, entry_kind).await;
            }
        })
        .detach();
    }

    fn spawn_terminal(
        &mut self,
        terminal_id: TerminalId,
        working_directory: Option<PathBuf>,
        custom_title: Option<SharedString>,
        initial_title: Option<SharedString>,
        created_at: Option<DateTime<Utc>>,
        association: TerminalSpawnAssociation,
        select: bool,
        focus: bool,
        run_init_command: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<TerminalView>>> {
        let terminal_working_directory = working_directory.clone();
        let init_command = Self::terminal_init_command(run_init_command, cx);
        let terminal_task = self.project.update(cx, |project, cx| {
            project.create_terminal_shell(working_directory, cx)
        });
        let workspace = self.workspace.clone();
        let workspace_id = self.workspace_id;
        let project = self.project.downgrade();

        cx.spawn_in(window, async move |this, cx| {
            let terminal = match terminal_task.await {
                Ok(terminal) => terminal,
                Err(error) => {
                    let error_message = format!("failed to spawn agent panel terminal: {error:#}");
                    log::error!("{error_message}");
                    workspace
                        .update(cx, |workspace, cx| workspace.show_error(error, cx))
                        .log_err();
                    this.update(cx, |this, cx| {
                        if this.pending_terminal_spawn == Some(terminal_id) {
                            this.pending_terminal_spawn = None;
                            cx.notify();
                        }
                    })
                    .log_err();
                    return Err(anyhow!(error_message));
                }
            };
            let view = this.update_in(cx, |this, window, cx| {
                #[cfg(feature = "multiplayer-tools")]
                if let TerminalSpawnAssociation::Inline(association) = &association {
                    this.validate_inline_terminal_parent(association, cx)?;
                }
                let terminal_for_init_command = terminal.clone();
                let terminal_view = cx.new(|cx| {
                    let mut view =
                        TerminalView::new(terminal, workspace, workspace_id, project, window, cx);
                    view.set_show_workspace_actions(false, cx);
                    view
                });
                this.insert_terminal(
                    terminal_id,
                    terminal_view.clone(),
                    terminal_working_directory,
                    custom_title,
                    initial_title,
                    created_at,
                    association,
                    select,
                    focus,
                    source,
                    window,
                    cx,
                )?;
                Self::write_terminal_init_command(&terminal_for_init_command, init_command, cx);
                anyhow::Ok(terminal_view)
            })??;
            Ok(view)
        })
    }

    fn terminal_init_command(run_init_command: bool, cx: &App) -> Option<String> {
        run_init_command
            .then(|| AgentSettings::get_global(cx).terminal_init_command.clone())
            .flatten()
            .filter(|command| !command.trim().is_empty())
    }

    fn write_terminal_init_command(
        terminal: &Entity<terminal::Terminal>,
        init_command: Option<String>,
        cx: &mut Context<Self>,
    ) {
        let Some(command) = init_command else {
            return;
        };

        if !terminal.read(cx).is_pty() {
            terminal.update(cx, |terminal, _| {
                terminal.write_init_command(Self::terminal_init_command_input(command))
            });
            return;
        }

        let startup = terminal.update(cx, |terminal, _| {
            terminal.start_init_command_startup_handshake()
        });

        let terminal = terminal.downgrade();
        cx.spawn(async move |_this, cx| {
            // Fall back to the timeout so the init command is still delivered if
            // the shell never echoes the marker.
            let timeout = cx
                .background_executor()
                .timer(TERMINAL_INIT_COMMAND_STARTUP_TIMEOUT);
            futures::select_biased! {
                _ = startup.fuse() => {}
                _ = timeout.fuse() => {}
            }

            let input = Self::terminal_init_command_input(command);
            if let Err(error) = terminal.update(cx, move |terminal, cx| {
                if !terminal.write_init_command_after_startup(input, cx) {
                    log::debug!(
                        "skipping terminal init command because the terminal is no longer eligible"
                    );
                }
            }) {
                log::debug!("skipping terminal init command because the terminal closed: {error}");
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn terminal_init_command_input(command: String) -> Vec<u8> {
        let mut input = command.into_bytes();
        // CR, not "\r\n": "\r\n" puts PowerShell into continuation
        // mode (same convention as the activation-script writes in
        // `TerminalBuilder::new`).
        input.push(b'\x0d');
        input
    }

    fn insert_terminal(
        &mut self,
        terminal_id: TerminalId,
        terminal_view: Entity<TerminalView>,
        working_directory: Option<PathBuf>,
        custom_title: Option<SharedString>,
        initial_title: Option<SharedString>,
        created_at: Option<DateTime<Utc>>,
        association: TerminalSpawnAssociation,
        select: bool,
        focus: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if let Some(custom_title) = custom_title {
            terminal_view.update(cx, |terminal_view, cx| {
                terminal_view.set_custom_title(Some(custom_title.to_string()), cx);
            });
        }
        let terminal_entity = terminal_view.read(cx).terminal().clone();
        let view_subscription = cx.subscribe(
            &terminal_view,
            move |this, _terminal_view, event: &ItemEvent, cx| match event {
                ItemEvent::UpdateTab | ItemEvent::UpdateBreadcrumbs => {
                    this.refresh_terminal_metadata(terminal_id, cx);
                }
                ItemEvent::CloseItem | ItemEvent::Edit => {}
            },
        );
        // Listen on the underlying `Terminal` entity for shell-driven metadata
        // changes and bell.
        let terminal_subscription = cx.subscribe_in(
            &terminal_entity,
            window,
            move |this, _terminal, event: &TerminalEvent, window, cx| match event {
                TerminalEvent::TitleChanged
                | TerminalEvent::Wakeup
                | TerminalEvent::BreadcrumbsChanged => {
                    this.refresh_terminal_metadata(terminal_id, cx);
                    this.report_terminal_program(terminal_id, source, cx);
                }
                TerminalEvent::Bell => this.mark_terminal_notification(terminal_id, window, cx),
                TerminalEvent::CloseTerminal => {
                    this.request_close_terminal_from_terminal_event(terminal_id, window, cx);
                }
                TerminalEvent::BlinkChanged(_)
                | TerminalEvent::SelectionsChanged
                | TerminalEvent::NewNavigationTarget(_)
                | TerminalEvent::Open(_) => {}
            },
        );

        let last_known_terminal_title = initial_title
            .map(|title| title.to_string())
            .unwrap_or_default();
        let mut terminal = AgentTerminal {
            view: terminal_view,
            #[cfg(feature = "multiplayer-tools")]
            inline_history_owner: match &association {
                TerminalSpawnAssociation::Standalone => None,
                TerminalSpawnAssociation::Inline(association) => {
                    Some(association.history_owner.downgrade())
                }
            },
            title_editor: None,
            title_editor_initial_title: None,
            title_editor_subscription: None,
            last_known_title: last_known_terminal_title.clone(),
            last_known_terminal_title,
            last_observed_program: None,
            working_directory,
            created_at: created_at.unwrap_or_else(Utc::now),
            has_notification: false,
            search_bar: None,
            notification_windows: Vec::new(),
            notification_subscriptions: Vec::new(),
            _subscriptions: vec![view_subscription, terminal_subscription],
        };
        if self.pending_terminal_spawn == Some(terminal_id) {
            self.pending_terminal_spawn = None;
        }
        terminal.refresh_metadata(cx);
        #[cfg(feature = "multiplayer-tools")]
        let inline_init_command = match association {
            TerminalSpawnAssociation::Standalone => None,
            TerminalSpawnAssociation::Inline(association) => {
                let mut metadata = self.metadata_for_terminal(terminal_id, &terminal, cx);
                metadata.inline_context = Some(association.context);
                association
                    .history_owner
                    .update(cx, |store, cx| store.save_inline(metadata, cx))?;
                association.init_command
            }
        };
        #[cfg(not(feature = "multiplayer-tools"))]
        let _association = association;
        terminal.report_started_terminal_program(terminal_id, source, cx);
        self.terminals.insert(terminal_id, terminal);
        self.persist_terminal_metadata(terminal_id, cx);
        #[cfg(feature = "multiplayer-tools")]
        Self::write_terminal_init_command(&terminal_entity, inline_init_command, cx);
        self.emit_terminal_thread_started(terminal_id, source, cx);
        if select {
            self.set_base_view(BaseView::Terminal { terminal_id }, focus, window, cx);
        }
        cx.emit(AgentPanelEvent::EntryChanged);
        cx.notify();
        Ok(())
    }

    pub fn activate_terminal(
        &mut self,
        terminal_id: TerminalId,
        focus: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(terminal) = self.terminals.get_mut(&terminal_id) else {
            return;
        };
        let had_notification = terminal.has_notification;
        terminal.has_notification = false;
        if had_notification {
            self.dismiss_terminal_notifications(terminal_id, cx);
        }
        self.set_base_view(BaseView::Terminal { terminal_id }, focus, window, cx);
        if had_notification {
            cx.emit(AgentPanelEvent::EntryChanged);
            cx.notify();
        }
    }

    pub fn close_terminal(
        &mut self,
        terminal_id: TerminalId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.close_terminal_internal(terminal_id, true, window, cx);
    }

    pub fn close_terminal_without_activating_draft(
        &mut self,
        terminal_id: TerminalId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.close_terminal_internal(terminal_id, false, window, cx);
    }

    fn close_terminal_internal(
        &mut self,
        terminal_id: TerminalId,
        activate_draft_after_close: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let was_active = self.active_terminal_id() == Some(terminal_id);
        #[cfg(feature = "multiplayer-tools")]
        let history_owner = self
            .terminals
            .get(&terminal_id)
            .and_then(|terminal| terminal.inline_history_owner.clone());

        if self.pending_terminal_spawn == Some(terminal_id) {
            self.pending_terminal_spawn = None;
        }
        self.dismiss_terminal_notifications(terminal_id, cx);
        if self.terminals.remove(&terminal_id).is_none() {
            return;
        }
        if let Some(store) = TerminalThreadMetadataStore::try_global(cx) {
            #[cfg(feature = "multiplayer-tools")]
            let owner_matches = history_owner.as_ref().is_none_or(|owner| {
                owner
                    .upgrade()
                    .is_some_and(|owner| owner.entity_id() == store.entity_id())
            });
            #[cfg(not(feature = "multiplayer-tools"))]
            let owner_matches = true;
            if owner_matches {
                store.update(cx, |store, cx| {
                    store.delete(terminal_id, cx);
                });
            } else {
                log::error!(
                    "Cannot delete inline terminal history after history owner replacement"
                );
            }
        }
        if was_active {
            self.base_view = BaseView::Uninitialized;
            self.refresh_base_view_subscriptions(window, cx);
            if activate_draft_after_close {
                self.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
            }
        }

        cx.emit(AgentPanelEvent::EntryChanged);
        cx.notify();
    }

    fn request_close_terminal_from_terminal_event(
        &mut self,
        terminal_id: TerminalId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        #[cfg(feature = "multiplayer-tools")]
        let metadata = if self
            .terminals
            .get(&terminal_id)
            .is_some_and(|terminal| terminal.inline_history_owner.is_some())
        {
            match self.finish_inline_terminal(terminal_id, window, cx) {
                Ok(metadata) => Some(metadata),
                Err(error) => {
                    log::error!("Failed to retain inline terminal history: {error:#}");
                    self.workspace
                        .update(cx, |workspace, cx| workspace.show_error(error, cx))
                        .log_err();
                    None
                }
            }
        } else {
            self.terminal_metadata(terminal_id, cx)
        };
        #[cfg(not(feature = "multiplayer-tools"))]
        let metadata = {
            let _window = window;
            self.terminal_metadata(terminal_id, cx)
        };
        if let Some(metadata) = metadata {
            cx.emit(AgentPanelEvent::TerminalCloseRequested { metadata });
        }
    }

    #[cfg(feature = "multiplayer-tools")]
    pub fn request_stop_inline_terminal(
        &self,
        terminal_id: TerminalId,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let terminal = self
            .terminals
            .get(&terminal_id)
            .ok_or_else(|| anyhow!("The inline terminal is no longer running"))?;
        let owner = terminal
            .inline_history_owner
            .as_ref()
            .and_then(WeakEntity::upgrade)
            .ok_or_else(|| anyhow!("The inline terminal's history owner is unavailable"))?;
        if !TerminalThreadMetadataStore::try_global(cx)
            .is_some_and(|store| store.entity_id() == owner.entity_id())
        {
            return Err(anyhow!("The inline terminal's history owner has changed"));
        }
        if owner
            .read(cx)
            .entry(terminal_id)
            .and_then(|metadata| metadata.inline_context.as_ref())
            .is_none()
        {
            return Err(anyhow!(
                "The terminal has no inline conversation association"
            ));
        }
        let terminal = terminal.view.read(cx).terminal().clone();
        terminal.update(cx, |terminal, _| {
            terminal.request_stop_interactive_process()
        })
    }

    #[cfg(feature = "multiplayer-tools")]
    fn finish_inline_terminal(
        &mut self,
        terminal_id: TerminalId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<TerminalThreadMetadata> {
        let (owner, metadata) = self.inline_terminal_snapshot_metadata(terminal_id, cx)?;
        let terminal = self
            .terminals
            .get(&terminal_id)
            .ok_or_else(|| anyhow!("The inline terminal is no longer available"))?;
        let was_focused = terminal.view.focus_handle(cx).contains_focused(window, cx);
        let context = metadata
            .inline_context
            .as_ref()
            .ok_or_else(|| anyhow!("The inline terminal's conversation is unavailable"))?;
        let parent_id = context.thread_id;
        let parent_session = context.session_id.clone();
        owner.update(cx, |store, cx| store.save(metadata.clone(), cx));
        self.dismiss_terminal_notifications(terminal_id, cx);
        self.terminals.remove(&terminal_id);
        if was_focused
            && let Some(parent) = self.active_thread_view(cx)
            && parent.read(cx).root_thread_id == parent_id
            && parent_session.as_ref() == Some(&parent.read(cx).session_id)
        {
            parent
                .update(cx, |parent, cx| {
                    parent.focus_after_inline_terminal(terminal_id, window, cx)
                })
                .log_err();
        }
        cx.emit(AgentPanelEvent::EntryChanged);
        cx.notify();
        Ok(metadata)
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(crate) fn inline_terminal_snapshot_metadata(
        &self,
        terminal_id: TerminalId,
        cx: &App,
    ) -> Result<(Entity<TerminalThreadMetadataStore>, TerminalThreadMetadata)> {
        let terminal = self
            .terminals
            .get(&terminal_id)
            .ok_or_else(|| anyhow!("The inline terminal is no longer available"))?;
        let owner = terminal
            .inline_history_owner
            .as_ref()
            .and_then(WeakEntity::upgrade)
            .ok_or_else(|| anyhow!("The inline terminal's history owner is unavailable"))?;
        // A replaced global must not turn a conversation terminal into a
        // standalone close request or receive writes from the previous owner.
        if !TerminalThreadMetadataStore::try_global(cx)
            .is_some_and(|store| store.entity_id() == owner.entity_id())
        {
            return Err(anyhow!("The inline terminal's history owner has changed"));
        }
        if owner.read(cx).entry(terminal_id).is_none() {
            return Err(anyhow!("The inline terminal's history is unavailable"));
        }
        let mut metadata = self.metadata_for_terminal(terminal_id, terminal, cx);
        let context = metadata
            .inline_context
            .as_mut()
            .ok_or_else(|| anyhow!("The inline terminal's conversation is unavailable"))?;
        let terminal = terminal.view.read(cx).terminal().read(cx);
        context.snapshot = Some(terminal.snapshot()?);
        context.output = terminal.get_content();
        Ok((owner, metadata))
    }

    #[cfg(feature = "multiplayer-tools")]
    fn flush_inline_terminal_history(
        &self,
        cx: &mut Context<Self>,
    ) -> futures::future::LocalBoxFuture<'static, Result<()>> {
        let mut owners = HashMap::default();
        let mut first_error = None;
        for (terminal_id, terminal) in &self.terminals {
            if terminal.inline_history_owner.is_none() {
                continue;
            }
            match self.inline_terminal_snapshot_metadata(*terminal_id, cx) {
                Ok((owner, metadata)) => {
                    owner.update(cx, |store, cx| store.save(metadata, cx));
                    owners.insert(owner.entity_id(), owner);
                }
                Err(error) => {
                    self.report_inline_terminal_history_error(&error, cx);
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        let flushes = owners
            .into_values()
            .map(|owner| {
                let flush = owner.read(cx).flush_before_quit();
                async move {
                    let _owner = owner;
                    flush.await
                }
            })
            .collect::<Vec<_>>();
        async move {
            for flush in flushes {
                if let Err(error) = flush.await {
                    log::error!("Inline terminal history write failed: {error:#}");
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
            match first_error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }
        .boxed_local()
    }

    #[cfg(feature = "multiplayer-tools")]
    fn report_inline_terminal_history_error(&self, error: &anyhow::Error, cx: &mut Context<Self>) {
        log::error!("Failed to retain inline terminal history: {error:#}");
        let workspace = self.workspace.clone();
        let error = anyhow!("Failed to retain inline terminal history: {error:#}");
        // File actions can hold the workspace update while invoking this panel.
        cx.defer(move |cx| {
            workspace
                .update(cx, |workspace, cx| workspace.show_error(error, cx))
                .log_err();
        });
    }

    fn emit_terminal_thread_started(
        &self,
        terminal_id: TerminalId,
        source: AgentThreadSource,
        cx: &App,
    ) {
        telemetry::event!(
            "Agent Thread Started",
            agent = TERMINAL_AGENT_TELEMETRY_ID,
            terminal_id = terminal_id.to_key_string(),
            source = source.as_str(),
            side = crate::agent_sidebar_side(cx),
            thread_location = "current_worktree",
        );
    }

    fn refresh_terminal_metadata(&mut self, terminal_id: TerminalId, cx: &mut Context<Self>) {
        let Some(terminal) = self.terminals.get_mut(&terminal_id) else {
            return;
        };
        let metadata_changed = terminal.refresh_metadata(cx);
        #[cfg(feature = "multiplayer-tools")]
        let snapshot_changed = terminal.inline_history_owner.is_some();
        #[cfg(not(feature = "multiplayer-tools"))]
        let snapshot_changed = false;
        if !metadata_changed && !snapshot_changed {
            return;
        }
        self.persist_terminal_metadata(terminal_id, cx);
        if metadata_changed {
            cx.emit(AgentPanelEvent::EntryChanged);
        }
        cx.notify();
    }

    fn report_terminal_program(
        &mut self,
        terminal_id: TerminalId,
        source: AgentThreadSource,
        cx: &mut Context<Self>,
    ) {
        if let Some(terminal) = self.terminals.get_mut(&terminal_id) {
            terminal.report_started_terminal_program(terminal_id, source, cx);
        }
    }

    fn persist_all_terminal_metadata(&self, cx: &mut Context<Self>) {
        let terminal_ids = self.terminals.keys().copied().collect::<Vec<_>>();
        for terminal_id in terminal_ids {
            self.persist_terminal_metadata(terminal_id, cx);
        }
    }

    fn persist_terminal_metadata(&self, terminal_id: TerminalId, cx: &mut Context<Self>) {
        #[cfg(feature = "multiplayer-tools")]
        if self
            .terminals
            .get(&terminal_id)
            .is_some_and(|terminal| terminal.inline_history_owner.is_some())
        {
            match self.inline_terminal_snapshot_metadata(terminal_id, cx) {
                Ok((owner, metadata)) => {
                    owner.update(cx, |store, cx| store.save(metadata, cx));
                }
                Err(error) => self.report_inline_terminal_history_error(&error, cx),
            }
            return;
        }
        let Some(store) = TerminalThreadMetadataStore::try_global(cx) else {
            return;
        };
        let Some(metadata) = self.terminal_metadata(terminal_id, cx) else {
            return;
        };
        store.update(cx, |store, cx| {
            store.save(metadata, cx);
        });
    }

    fn terminal_metadata(
        &self,
        terminal_id: TerminalId,
        cx: &App,
    ) -> Option<TerminalThreadMetadata> {
        let terminal = self.terminals.get(&terminal_id)?;
        Some(self.metadata_for_terminal(terminal_id, terminal, cx))
    }

    fn metadata_for_terminal(
        &self,
        terminal_id: TerminalId,
        terminal: &AgentTerminal,
        cx: &App,
    ) -> TerminalThreadMetadata {
        #[cfg(feature = "multiplayer-tools")]
        let history_owner = match &terminal.inline_history_owner {
            Some(owner) => owner.upgrade(),
            None => TerminalThreadMetadataStore::try_global(cx),
        };
        #[cfg(not(feature = "multiplayer-tools"))]
        let history_owner = TerminalThreadMetadataStore::try_global(cx);
        let project = self.project.read(cx);
        TerminalThreadMetadata {
            terminal_id,
            title: terminal.terminal_title(cx),
            custom_title: terminal.custom_title(cx),
            created_at: terminal.created_at,
            worktree_paths: project.worktree_paths(cx),
            remote_connection: project.remote_connection_options(cx),
            working_directory: terminal.working_directory.clone(),
            inline_context: history_owner.and_then(|store| {
                store
                    .read(cx)
                    .entry(terminal_id)
                    .and_then(|metadata| metadata.inline_context.clone())
            }),
        }
    }

    pub fn restore_terminal(
        &mut self,
        metadata: TerminalThreadMetadata,
        focus: bool,
        source: AgentThreadSource,
        workspace: Option<&Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        #[cfg(feature = "multiplayer-tools")]
        if metadata.inline_context.is_some() {
            self.workspace
                .update(cx, |workspace, cx| {
                    workspace.show_error(
                        anyhow!(
                            "Cannot restore conversation terminal history as a standalone shell"
                        ),
                        cx,
                    );
                })
                .log_err();
            return;
        }
        if self.has_terminal(metadata.terminal_id) {
            self.activate_terminal(metadata.terminal_id, focus, window, cx);
            return;
        }

        if !self.supports_terminal(cx) {
            return;
        }

        self.pending_terminal_spawn = Some(metadata.terminal_id);
        let working_directory = self.terminal_restore_working_directory(&metadata, workspace, cx);
        let initial_title = Self::terminal_restore_initial_title(&metadata);
        self.spawn_terminal(
            metadata.terminal_id,
            working_directory,
            metadata.custom_title.clone(),
            initial_title,
            Some(metadata.created_at),
            TerminalSpawnAssociation::Standalone,
            true,
            focus,
            true,
            source,
            window,
            cx,
        )
        .detach_and_log_err(cx);
    }

    fn restore_terminal_for_panel_load(
        &mut self,
        metadata: TerminalThreadMetadata,
        focus: bool,
        source: AgentThreadSource,
        workspace: Option<&Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        #[cfg(test)]
        self.restore_test_terminal(metadata, focus, source, workspace, window, cx)
            .log_err();

        #[cfg(not(test))]
        self.restore_terminal(metadata, focus, source, workspace, window, cx);
    }

    fn terminal_restore_working_directory(
        &self,
        metadata: &TerminalThreadMetadata,
        workspace: Option<&Workspace>,
        cx: &App,
    ) -> Option<PathBuf> {
        if let Some(working_directory) = metadata.working_directory.clone() {
            return Some(working_directory);
        }

        if let Some(workspace) = workspace {
            return terminal_view::default_working_directory(workspace, cx);
        }

        self.default_terminal_working_directory(cx)
    }

    fn terminal_restore_initial_title(metadata: &TerminalThreadMetadata) -> Option<SharedString> {
        (!metadata.title.is_empty()).then(|| metadata.title.clone())
    }

    fn edit_terminal_title(
        &mut self,
        terminal_id: TerminalId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(terminal) = self.terminals.get_mut(&terminal_id) else {
            return;
        };

        if let Some(title_editor) = terminal.title_editor.as_ref() {
            title_editor.focus_handle(cx).focus(window, cx);
            return;
        }

        let title = terminal.editable_title(cx).to_string();
        let title_editor_initial_title = title.clone();
        let title_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text(title, window, cx);
            editor
        });
        let title_editor_subscription = cx.subscribe_in(
            &title_editor,
            window,
            move |this, title_editor, event: &editor::EditorEvent, window, cx| {
                this.handle_terminal_title_editor_event(
                    terminal_id,
                    title_editor,
                    event,
                    window,
                    cx,
                );
            },
        );
        title_editor.update(cx, |editor, cx| {
            editor.select_all(&editor::actions::SelectAll, window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        terminal.title_editor = Some(title_editor);
        terminal.title_editor_initial_title = Some(title_editor_initial_title);
        terminal.title_editor_subscription = Some(title_editor_subscription);
        cx.notify();
    }

    fn stop_editing_terminal_title(
        &mut self,
        terminal_id: TerminalId,
        focus_terminal: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(terminal) = self.terminals.get_mut(&terminal_id) else {
            return;
        };
        let terminal_view = terminal.view.clone();
        terminal.title_editor = None;
        terminal.title_editor_initial_title = None;
        terminal.title_editor_subscription = None;
        let title_changed = terminal.refresh_title(cx);

        if focus_terminal {
            terminal_view.focus_handle(cx).focus(window, cx);
        }
        if title_changed {
            cx.emit(AgentPanelEvent::EntryChanged);
        }
        cx.notify();
    }

    fn handle_terminal_title_editor_event(
        &mut self,
        terminal_id: TerminalId,
        title_editor: &Entity<Editor>,
        event: &editor::EditorEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            editor::EditorEvent::BufferEdited => {
                if !title_editor.read(cx).is_focused(window) {
                    return;
                }
                let Some((terminal_view, initial_title, terminal_title)) =
                    self.terminals.get(&terminal_id).and_then(|terminal| {
                        terminal
                            .title_editor
                            .as_ref()
                            .is_some_and(|current_editor| current_editor == title_editor)
                            .then(|| {
                                (
                                    terminal.view.clone(),
                                    terminal.title_editor_initial_title.clone(),
                                    terminal.terminal_title(cx),
                                )
                            })
                    })
                else {
                    return;
                };
                let new_title = title_editor.read(cx).text(cx);
                if initial_title.as_deref() == Some(new_title.as_str()) {
                    return;
                }
                let label = if new_title.trim().is_empty()
                    || new_title == terminal_title_without_prefix(terminal_title.as_ref())
                {
                    None
                } else {
                    Some(new_title)
                };

                cx.defer(move |cx| {
                    terminal_view.update(cx, |terminal_view, cx| {
                        terminal_view.set_custom_title(label, cx);
                    });
                });
            }
            editor::EditorEvent::Blurred => {
                if self
                    .terminals
                    .get(&terminal_id)
                    .and_then(|terminal| terminal.title_editor.as_ref())
                    .is_some_and(|current_editor| current_editor == title_editor)
                {
                    self.stop_editing_terminal_title(terminal_id, false, window, cx);
                }
            }
            _ => {}
        }
    }

    fn mark_terminal_notification(
        &mut self,
        terminal_id: TerminalId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.active_terminal_visible(terminal_id, window, cx) {
            return;
        }
        let newly_notified = {
            let Some(terminal) = self.terminals.get_mut(&terminal_id) else {
                return;
            };
            if terminal.has_notification {
                false
            } else {
                terminal.has_notification = true;
                true
            }
        };
        if newly_notified {
            cx.emit(AgentPanelEvent::EntryChanged);
            cx.notify();
            #[cfg(feature = "audio")]
            self.play_terminal_notification_sound(
                self.terminal_status_visible(terminal_id, window, cx),
                cx,
            );
            self.show_terminal_notification(terminal_id, window, cx);
        }
    }

    fn show_terminal_notification(
        &mut self,
        terminal_id: TerminalId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(terminal) = self.terminals.get(&terminal_id) else {
            return;
        };
        if !terminal.notification_windows.is_empty() {
            return;
        }
        let title = terminal.title(cx);
        if self.terminal_status_visible(terminal_id, window, cx) {
            return;
        }
        let settings = AgentSettings::get_global(cx);
        match settings.notify_when_agent_waiting {
            NotifyWhenAgentWaiting::PrimaryScreen => {
                window.request_attention();
                if let Some(primary) = cx.primary_display() {
                    self.pop_up_terminal_notification(terminal_id, &title, primary, window, cx);
                }
            }
            NotifyWhenAgentWaiting::AllScreens => {
                window.request_attention();
                for screen in cx.displays() {
                    self.pop_up_terminal_notification(terminal_id, &title, screen, window, cx);
                }
            }
            NotifyWhenAgentWaiting::Never => {}
        }
    }

    fn pop_up_terminal_notification(
        &mut self,
        terminal_id: TerminalId,
        title: &SharedString,
        screen: Rc<dyn PlatformDisplay>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let options = AgentNotification::window_options(screen, cx);
        let project_name = self.workspace.upgrade().and_then(|workspace| {
            workspace
                .read(cx)
                .project()
                .read(cx)
                .visible_worktrees(cx)
                .next()
                .map(|worktree| worktree.read(cx).root_name_str().to_string())
        });
        let title = title.clone();
        let Ok(screen_window) = cx.open_window(options, |_window, cx| {
            cx.new(|_cx| AgentNotification::new(title, None, IconName::Terminal, project_name))
        }) else {
            return;
        };
        let Ok(pop_up) = screen_window.entity(cx) else {
            return;
        };

        let event_subscription = cx.subscribe_in(&pop_up, window, {
            move |this, _, event: &AgentNotificationEvent, window, cx| match event {
                AgentNotificationEvent::Accepted => {
                    let Some(handle) = window.window_handle().downcast::<MultiWorkspace>() else {
                        log::error!("root view should be a MultiWorkspace");
                        return;
                    };
                    cx.activate(true);

                    let workspace = this.workspace.clone();
                    cx.defer(move |cx| {
                        handle
                            .update(cx, |multi_workspace, window, cx| {
                                window.activate_window();

                                let Some(workspace) = workspace.upgrade() else {
                                    return;
                                };
                                multi_workspace.activate(workspace.clone(), None, window, cx);

                                workspace.update(cx, |workspace, cx| {
                                    workspace.reveal_panel::<AgentPanel>(window, cx);
                                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                                        panel.update(cx, |panel, cx| {
                                            panel.activate_terminal(terminal_id, true, window, cx);
                                        });
                                    }
                                    workspace.focus_panel::<AgentPanel>(window, cx);
                                });
                            })
                            .log_err();
                    });

                    this.dismiss_terminal_notifications(terminal_id, cx);
                }
                AgentNotificationEvent::Dismissed => {
                    this.dismiss_terminal_notifications(terminal_id, cx);
                }
            }
        });

        let pop_up_weak = pop_up.downgrade();
        let window_activation_subscription = cx.observe_window_activation(window, {
            let pop_up_weak = pop_up_weak.clone();
            move |this, window, cx| {
                this.dismiss_terminal_pop_up_if_visible(terminal_id, &pop_up_weak, window, cx);
            }
        });

        let multi_workspace_subscription = {
            let pop_up_weak = pop_up_weak.clone();
            window.root::<MultiWorkspace>().flatten().map(|mw| {
                cx.observe_in(&mw, window, move |this, _, window, cx| {
                    this.dismiss_terminal_pop_up_if_visible(terminal_id, &pop_up_weak, window, cx);
                })
            })
        };

        let this_panel = cx.entity();
        let agent_panel_subscription = cx.subscribe_in(&this_panel, window, {
            move |this, _, event: &AgentPanelEvent, window, cx| match event {
                AgentPanelEvent::ActiveViewChanged | AgentPanelEvent::ActiveViewFocused => {
                    this.dismiss_terminal_pop_up_if_visible(terminal_id, &pop_up_weak, window, cx);
                }
                AgentPanelEvent::EntryChanged
                | AgentPanelEvent::TerminalCloseRequested { .. }
                | AgentPanelEvent::ThreadInteracted { .. } => {}
            }
        });

        let Some(terminal) = self.terminals.get_mut(&terminal_id) else {
            screen_window
                .update(cx, |_, window, _| window.remove_window())
                .ok();
            return;
        };
        terminal.notification_windows.push(screen_window);
        terminal.notification_subscriptions.push(event_subscription);
        terminal
            .notification_subscriptions
            .push(window_activation_subscription);
        terminal
            .notification_subscriptions
            .push(agent_panel_subscription);
        if let Some(subscription) = multi_workspace_subscription {
            terminal.notification_subscriptions.push(subscription);
        }
    }

    fn dismiss_terminal_notifications(&mut self, terminal_id: TerminalId, cx: &mut App) {
        let Some(terminal) = self.terminals.get_mut(&terminal_id) else {
            return;
        };
        let windows = std::mem::take(&mut terminal.notification_windows);
        terminal.notification_subscriptions.clear();
        for window in windows {
            window
                .update(cx, |_, window, _| {
                    window.remove_window();
                })
                .ok();
        }
    }

    fn dismiss_all_terminal_notifications(&mut self, cx: &mut App) {
        let terminal_ids = self.terminals.keys().copied().collect::<Vec<_>>();
        for terminal_id in terminal_ids {
            self.dismiss_terminal_notifications(terminal_id, cx);
        }
    }

    pub fn dismiss_all_notifications(&mut self, cx: &mut Context<Self>) -> bool {
        let mut dismissed = false;
        for conversation_view in self.conversation_views() {
            dismissed |= conversation_view.update(cx, |view, cx| view.dismiss_notifications(cx));
        }
        let had_terminal_notifications = self
            .terminals
            .values()
            .any(|t| !t.notification_windows.is_empty());
        if had_terminal_notifications {
            self.dismiss_all_terminal_notifications(cx);
            dismissed = true;
        }
        dismissed
    }

    fn active_terminal_visible(&self, terminal_id: TerminalId, window: &Window, cx: &App) -> bool {
        if !window.is_window_active() {
            return false;
        }
        if !self.terminal_surface_visible(terminal_id) {
            return false;
        }
        let Some(workspace) = self.workspace.upgrade() else {
            return false;
        };
        if let Some(multi_workspace) = window.root::<MultiWorkspace>().flatten() {
            let multi_workspace = multi_workspace.read(cx);
            if multi_workspace.workspace() != &workspace {
                return false;
            }
        }
        AgentPanel::is_visible(&workspace, cx)
    }

    fn terminal_surface_visible(&self, terminal_id: TerminalId) -> bool {
        self.active_terminal_id() == Some(terminal_id)
            && matches!(self.visible_surface(), VisibleSurface::Terminal(_))
    }

    fn terminal_status_visible(&self, terminal_id: TerminalId, window: &Window, cx: &App) -> bool {
        if !window.is_window_active() {
            return false;
        }

        if let Some(multi_workspace) = window.root::<MultiWorkspace>().flatten() {
            let multi_workspace = multi_workspace.read(cx);
            if multi_workspace.sidebar_open() && multi_workspace.is_threads_list_view_active(cx) {
                return true;
            }

            let Some(workspace) = self.workspace.upgrade() else {
                return false;
            };

            return multi_workspace.workspace() == &workspace
                && self.terminal_surface_visible(terminal_id)
                && AgentPanel::is_visible(&workspace, cx);
        }

        self.workspace.upgrade().is_some_and(|workspace| {
            self.terminal_surface_visible(terminal_id) && AgentPanel::is_visible(&workspace, cx)
        })
    }

    fn dismiss_terminal_pop_up_if_visible(
        &mut self,
        terminal_id: TerminalId,
        pop_up: &WeakEntity<AgentNotification>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.terminal_status_visible(terminal_id, window, cx) {
            return;
        }
        if self.active_terminal_visible(terminal_id, window, cx)
            && let Some(terminal) = self.terminals.get_mut(&terminal_id)
            && terminal.has_notification
        {
            terminal.has_notification = false;
            cx.emit(AgentPanelEvent::EntryChanged);
            cx.notify();
        }
        if let Some(pop_up) = pop_up.upgrade() {
            pop_up.update(cx, |notification, cx| {
                notification.dismiss(cx);
            });
        }
    }

    #[cfg(feature = "audio")]
    fn play_terminal_notification_sound(&self, visible: bool, cx: &mut App) {
        let settings = AgentSettings::get_global(cx);
        if settings.play_sound_when_agent_done.should_play(visible) {
            Audio::play_sound(Sound::AgentDone, cx);
        }
    }

    fn default_terminal_working_directory(&self, cx: &App) -> Option<PathBuf> {
        // Reuse the workspace-based helper so behavior matches the regular
        // terminal panel (e.g. `WorkingDirectory::FirstProjectDirectory` falling
        // back to a file's parent directory when the worktree root is a file).
        self.workspace
            .upgrade()
            .and_then(|workspace| terminal_view::default_working_directory(workspace.read(cx), cx))
    }

    fn has_open_project(&self, cx: &App) -> bool {
        self.project.read(cx).visible_worktrees(cx).next().is_some()
    }

    fn ensure_native_agent_connection(&self, cx: &mut Context<Self>) {
        if !self.has_open_project(cx) {
            return;
        }

        let fs = self.fs.clone();
        let thread_store = self.thread_store.clone();
        self.connection_store.update(cx, |store, cx| {
            store.request_connection(
                Agent::NativeAgent,
                Agent::NativeAgent.server(fs, thread_store),
                cx,
            );
        });
    }

    pub fn activate_draft(
        &mut self,
        focus: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.has_open_project(cx) {
            return;
        }

        let draft = self.ensure_draft(source, window, cx);
        if let BaseView::AgentThread { conversation_view } = &self.base_view {
            if conversation_view.entity_id() == draft.entity_id() {
                if focus {
                    self.focus_base_view(window, cx);
                }
                return;
            }
        }
        self.set_base_view(
            BaseView::AgentThread {
                conversation_view: draft,
            },
            focus,
            window,
            cx,
        );
    }

    fn ensure_draft(
        &mut self,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Entity<ConversationView> {
        let desired_agent = self.selected_agent(cx);
        if let Some(draft) = &self.draft_thread {
            let draft_entity = draft.entity_id();
            let agent_matches = *draft.read(cx).agent_key() == desired_agent;
            let has_editor_content = draft.read(cx).root_thread_view().is_some_and(|tv| {
                !tv.read(cx)
                    .message_editor
                    .read(cx)
                    .text(cx)
                    .trim()
                    .is_empty()
            });
            // Only retarget the empty draft when the user is actively
            // viewing it — that's the case where switching agents in the
            // toolbar should replace the draft with one bound to the
            // newly-selected agent. When the draft is parked in its slot
            // while the user is viewing a real thread, `selected_agent`
            // reflects that real thread's agent and must not be allowed
            // to silently rebuild the draft.
            let draft_is_active = matches!(
                &self.base_view,
                BaseView::AgentThread { conversation_view }
                    if conversation_view.entity_id() == draft_entity
            );

            if agent_matches || has_editor_content || !draft_is_active {
                return draft.clone();
            }

            // Clean up the old empty draft's metadata so it doesn't
            // linger as a ghost entry in the sidebar.
            let old_draft_id = draft.read(cx).thread_id;
            self.discard_matching_draft_prompt(old_draft_id, cx);
            ThreadMetadataStore::global(cx).update(cx, |store, cx| {
                store.delete(old_draft_id, cx);
            });

            self.draft_thread = None;
            self._draft_editor_observation = None;
        }

        let thread = self.create_agent_thread_with_server(
            desired_agent,
            None,
            None,
            None,
            None,
            None,
            None,
            source,
            window,
            cx,
        );

        self.draft_thread = Some(thread.conversation_view.clone());
        self.observe_draft_editor(&thread.conversation_view, cx);
        thread.conversation_view
    }

    fn observe_draft_editor(
        &mut self,
        conversation_view: &Entity<ConversationView>,
        cx: &mut Context<Self>,
    ) {
        if let Some(acp_thread) = conversation_view.read(cx).root_thread(cx) {
            self._draft_editor_observation = Some(cx.subscribe(
                &acp_thread,
                |this, acp_thread, event: &AcpThreadEvent, cx| {
                    if !acp_thread.read(cx).is_draft_thread()
                        && this.draft_thread.as_ref().is_some_and(|draft| {
                            draft
                                .read(cx)
                                .root_thread(cx)
                                .is_some_and(|thread| thread.entity_id() == acp_thread.entity_id())
                        })
                    {
                        this.draft_thread = None;
                        this._draft_editor_observation = None;
                        this.serialize(cx);
                        return;
                    }

                    if let AcpThreadEvent::PromptUpdated = event {
                        this.serialize(cx);
                    }
                },
            ));
        } else {
            let cv = conversation_view.clone();
            self._draft_editor_observation = Some(cx.observe(&cv, |this, cv, cx| {
                if cv.read(cx).root_thread(cx).is_some() {
                    this.observe_draft_editor(&cv, cx);
                }
            }));
        }
    }

    /// Sets up an editor observation on the active view that reclaims
    /// it as ephemeral when the editor becomes empty. Only activates
    /// for non-ephemeral draft threads.
    fn observe_active_draft_for_empty_editor(
        &mut self,
        conversation_view: &Entity<ConversationView>,
        cx: &mut Context<Self>,
    ) {
        let thread_id = conversation_view.read(cx).thread_id;
        let is_ephemeral = self
            .draft_thread
            .as_ref()
            .is_some_and(|d| d.read(cx).thread_id == thread_id);
        if is_ephemeral {
            self._active_draft_reclaim_observation = None;
            return;
        }
        let is_draft = conversation_view
            .read(cx)
            .root_thread(cx)
            .is_some_and(|t| t.read(cx).is_draft_thread());
        if !is_draft {
            self._active_draft_reclaim_observation = None;
            return;
        }
        let Some(editor) = conversation_view
            .read(cx)
            .active_thread()
            .map(|tv| tv.read(cx).message_editor.clone())
        else {
            self._active_draft_reclaim_observation = None;
            return;
        };
        let cv = conversation_view.clone();
        self._active_draft_reclaim_observation =
            Some(cx.observe(&editor, move |this, _editor, cx| {
                let editor_has_text = cv.read(cx).active_thread().is_some_and(|tv| {
                    !tv.read(cx)
                        .message_editor
                        .read(cx)
                        .text(cx)
                        .trim()
                        .is_empty()
                });
                if editor_has_text {
                    return;
                }
                if this.ephemeral_draft_thread_id(cx) == Some(thread_id) {
                    return;
                }
                if this.active_thread_id(cx) != Some(thread_id) {
                    return;
                }
                if this.try_make_empty_draft_ephemeral(cv.clone(), cx) {
                    this._active_draft_reclaim_observation = None;
                    cx.emit(AgentPanelEvent::EntryChanged);
                    cx.notify();
                }
            }));
    }

    fn try_make_empty_draft_ephemeral(
        &mut self,
        conversation_view: Entity<ConversationView>,
        cx: &mut Context<Self>,
    ) -> bool {
        let (thread_id, is_draft, is_empty) = {
            let conversation = conversation_view.read(cx);
            let thread_id = conversation.thread_id;
            let is_draft = conversation
                .root_thread(cx)
                .is_some_and(|thread| thread.read(cx).is_draft_thread());
            let is_empty = if let Some(thread_view) = conversation.active_thread() {
                thread_view
                    .read(cx)
                    .message_editor
                    .read(cx)
                    .text(cx)
                    .trim()
                    .is_empty()
            } else {
                !self.draft_has_content(&conversation_view, cx)
            };

            (thread_id, is_draft, is_empty)
        };

        if !is_draft || !is_empty {
            return false;
        }

        self.retained_threads.remove(&thread_id);
        self.set_ephemeral_draft(conversation_view, cx);
        true
    }

    /// Moves a conversation view into the ephemeral `draft_thread` slot,
    /// cleaning up any previous ephemeral draft and deleting the thread's
    /// metadata so it no longer appears in the sidebar.
    fn set_ephemeral_draft(
        &mut self,
        conversation_view: Entity<ConversationView>,
        cx: &mut Context<Self>,
    ) {
        if let Some(old_draft) = self.draft_thread.take() {
            let old_id = old_draft.read(cx).thread_id;
            let new_id = conversation_view.read(cx).thread_id;
            if old_id != new_id {
                old_draft.update(cx, |draft, _| draft.discard_draft_prompt());
                self.discard_matching_draft_prompt(old_id, cx);
                ThreadMetadataStore::global(cx).update(cx, |store, cx| {
                    store.delete(old_id, cx);
                });
            }
            self._draft_editor_observation = None;
        }
        self.draft_thread = Some(conversation_view.clone());
        self.observe_draft_editor(&conversation_view, cx);
        self.serialize(cx);
    }

    /// Creates a new retained thread and inserts it into the sidebar without
    /// switching the active view to it. Used by the `create_thread` agent tool,
    /// which passes an initial prompt, and optionally an agent and model
    /// override.
    pub fn create_thread_with_options(
        &mut self,
        options: CreateThreadOptions,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> ThreadId {
        let (agent, override_used) = if self.project.read(cx).is_via_collab() {
            (Agent::NativeAgent, false)
        } else if let Some(override_agent) = options.agent {
            (override_agent, true)
        } else {
            (self.selected_agent.clone(), false)
        };
        // If the caller explicitly overrode the agent (e.g., the `create_thread`
        // tool wants to spawn a sibling thread using a specific agent), we
        // shouldn't let that change the panel's selected_agent or the
        // last-used-agent preference. Snapshot and restore both.
        let saved_selected_agent = override_used.then(|| self.selected_agent.clone());
        let thread = self.create_agent_thread_with_server(
            agent,
            None,
            None,
            options.work_dirs,
            options.title.clone(),
            options.initial_content,
            options.model,
            source,
            window,
            cx,
        );
        if let Some(original) = saved_selected_agent {
            self.set_selected_agent_and_persist(original, cx);
        }
        let thread_id = thread.conversation_view.read(cx).thread_id;
        self.retained_threads
            .insert(thread_id, thread.conversation_view);
        thread_id
    }

    pub fn activate_retained_thread(
        &mut self,
        id: ThreadId,
        focus: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let conversation_view = if let Some(view) = self.retained_threads.remove(&id) {
            self.try_make_empty_draft_ephemeral(view.clone(), cx);
            view
        } else if let Some(draft) = &self.draft_thread {
            if draft.read(cx).thread_id == id {
                draft.clone()
            } else {
                return;
            }
        } else {
            return;
        };
        self.set_base_view(
            BaseView::AgentThread { conversation_view },
            focus,
            window,
            cx,
        );
    }

    pub fn active_thread_id(&self, cx: &App) -> Option<ThreadId> {
        match &self.base_view {
            BaseView::AgentThread { conversation_view } => {
                Some(conversation_view.read(cx).thread_id)
            }
            _ => None,
        }
    }

    /// Drops a thread — retained or the active ephemeral draft — from
    /// the panel and deletes its metadata row. Used by the sidebar when
    /// the user dismisses a parked draft.
    pub fn remove_thread(&mut self, id: ThreadId, window: &mut Window, cx: &mut Context<Self>) {
        self.remove_thread_internal(id, true, window, cx);
    }

    pub fn remove_thread_without_activating_draft(
        &mut self,
        id: ThreadId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.remove_thread_internal(id, false, window, cx);
    }

    fn discard_matching_draft_prompt(&self, id: ThreadId, cx: &mut App) {
        for conversation in self
            .retained_threads
            .get(&id)
            .into_iter()
            .chain(self.draft_thread.as_ref())
            .chain(self.active_conversation_view())
        {
            if conversation.read(cx).thread_id == id {
                conversation.update(cx, |conversation, _| conversation.discard_draft_prompt());
            }
        }
    }

    fn remove_thread_internal(
        &mut self,
        id: ThreadId,
        activate_draft_after_remove: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.discard_matching_draft_prompt(id, cx);
        self.retained_threads.remove(&id);
        ThreadMetadataStore::global(cx).update(cx, |store, cx| {
            store.delete(id, cx);
        });

        if self
            .draft_thread
            .as_ref()
            .is_some_and(|d| d.read(cx).thread_id == id)
        {
            self.draft_thread = None;
            self._draft_editor_observation = None;
        }

        if self.active_thread_id(cx) == Some(id) {
            if activate_draft_after_remove {
                self.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
            } else {
                self.base_view = BaseView::Uninitialized;
                self.refresh_base_view_subscriptions(window, cx);
            }
            self.serialize(cx);
            cx.emit(AgentPanelEvent::ActiveViewChanged);
            cx.notify();
        }
    }

    pub fn ephemeral_draft_thread_id(&self, cx: &App) -> Option<ThreadId> {
        let draft = self.draft_thread.as_ref()?;
        let draft = draft.read(cx);
        draft
            .root_thread(cx)
            .is_some_and(|thread| thread.read(cx).is_draft_thread())
            .then_some(draft.thread_id)
    }

    pub fn active_terminal_id(&self) -> Option<TerminalId> {
        match &self.base_view {
            BaseView::Terminal { terminal_id } => Some(*terminal_id),
            _ => None,
        }
    }

    pub fn has_terminal(&self, terminal_id: TerminalId) -> bool {
        self.terminals.contains_key(&terminal_id)
    }

    pub fn terminals(&self, cx: &App) -> Vec<AgentPanelTerminalInfo> {
        self.terminals
            .iter()
            .map(|(id, terminal)| AgentPanelTerminalInfo {
                id: *id,
                title: terminal.title(cx),
                created_at: terminal.created_at,
                has_notification: terminal.has_notification,
                custom_title: terminal.custom_title(cx),
                working_directory: terminal.working_directory.clone(),
            })
            .collect()
    }

    pub fn editor_text(&self, id: ThreadId, cx: &App) -> Option<String> {
        self.editor_text_if_in_memory(id, cx).flatten()
    }

    pub fn editor_text_if_in_memory(&self, id: ThreadId, cx: &App) -> Option<Option<String>> {
        let cv = self
            .retained_threads
            .get(&id)
            .or_else(|| {
                self.draft_thread
                    .as_ref()
                    .filter(|draft| draft.read(cx).thread_id == id)
            })
            .or_else(|| match &self.base_view {
                BaseView::AgentThread { conversation_view }
                    if conversation_view.read(cx).thread_id == id =>
                {
                    Some(conversation_view)
                }
                _ => None,
            })?;
        let tv = cv.read(cx).root_thread_view()?;
        let text = tv.read(cx).message_editor.read(cx).text(cx);
        if text.trim().is_empty() {
            Some(None)
        } else {
            Some(Some(text))
        }
    }

    pub fn draft_prompt_blocks_if_in_memory(
        &self,
        id: ThreadId,
        cx: &App,
    ) -> Result<Option<Vec<acp::ContentBlock>>> {
        let cv = self
            .retained_threads
            .get(&id)
            .or_else(|| {
                self.draft_thread
                    .as_ref()
                    .filter(|draft| draft.read(cx).thread_id == id)
            })
            .or_else(|| match &self.base_view {
                BaseView::AgentThread { conversation_view }
                    if conversation_view.read(cx).thread_id == id =>
                {
                    Some(conversation_view)
                }
                _ => None,
            });
        let Some(cv) = cv else { return Ok(None) };
        let Some(thread_view) = cv.read(cx).root_thread_view() else {
            return Ok(None);
        };
        let thread_view = thread_view.read(cx);
        Ok(Some(
            thread_view
                .message_editor
                .read(cx)
                .draft_content_blocks_snapshot(cx)?,
        ))
    }

    fn new_native_agent_thread_from_summary(
        &mut self,
        action: &NewNativeAgentThreadFromSummary,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let session_id = action.from_session_id.clone();

        let Some(content) = Self::initial_content_for_thread_summary(session_id.clone(), cx) else {
            log::error!("No session found for summarization with id {}", session_id);
            return;
        };

        cx.spawn_in(window, async move |this, cx| {
            this.update_in(cx, |this, window, cx| {
                this.external_thread(
                    Some(Agent::NativeAgent),
                    None,
                    None,
                    None,
                    Some(content),
                    true,
                    AgentThreadSource::AgentPanel,
                    window,
                    cx,
                );
                anyhow::Ok(())
            })
        })
        .detach_and_log_err(cx);
    }

    fn initial_content_for_thread_summary(
        session_id: acp::SessionId,
        cx: &App,
    ) -> Option<AgentInitialContent> {
        let thread = ThreadStore::global(cx)
            .read(cx)
            .entries()
            .find(|t| t.id == session_id)?;

        Some(AgentInitialContent::ThreadSummary {
            session_id: thread.id,
            title: Some(thread.title),
        })
    }

    fn external_thread(
        &mut self,
        agent_choice: Option<crate::Agent>,
        resume_thread_id: Option<ThreadId>,
        work_dirs: Option<PathList>,
        title: Option<SharedString>,
        initial_content: Option<AgentInitialContent>,
        focus: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if resume_thread_id.is_none() && !self.has_open_project(cx) {
            return;
        }

        let agent = agent_choice.unwrap_or_else(|| self.selected_agent(cx));
        let thread = self.create_agent_thread_with_server(
            agent,
            None,
            resume_thread_id,
            work_dirs,
            title,
            initial_content,
            None,
            source,
            window,
            cx,
        );
        self.set_base_view(thread.into(), focus, window, cx);
    }

    fn manage_skills(
        &mut self,
        _action: &ManageSkills,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.dispatch_action(
            Box::new(zed_actions::OpenSettingsAt {
                path: zed_actions::AGENT_SKILLS_SETTINGS_PATH.to_string(),
                target: None,
            }),
            cx,
        );
    }

    fn manage_recipes(
        &mut self,
        schedule: Option<uuid::Uuid>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.open_recipe_management(schedule, None, None, window, cx);
    }

    pub fn review_recipe_link(
        &mut self,
        link: agent::recipe_import::RecipeLink,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.open_recipe_management(None, None, Some(link), window, cx);
    }

    fn open_recipe_management(
        &mut self,
        schedule: Option<uuid::Uuid>,
        proposal: Option<Rc<agent::recipes::RecipeProposal>>,
        link: Option<agent::recipe_import::RecipeLink>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.ensure_native_agent_connection(cx);
        let Some(connection) = self.connection_store.update(cx, |store, cx| {
            store
                .entry(&Agent::NativeAgent)
                .map(|entry| entry.read(cx).wait_for_connection())
        }) else {
            self.workspace
                .update(cx, |workspace, cx| {
                    workspace.show_error("The native agent is unavailable for recipe review", cx)
                })
                .log_err();
            return;
        };
        let project = self.project.clone();
        let workspace = self.workspace.clone();
        let panel = cx.entity().downgrade();
        cx.spawn_in(window, async move |_, cx| {
            let result: Result<()> = async {
                if let Some(link) = &link { link.encode()?; }
                let connected = connection.await?;
                let connection = connected.connection.downcast::<agent::NativeAgentConnection>()
                    .context("Native agent is unavailable")?;
                workspace.update_in(cx, |workspace, window, cx| {
                    if let Some(proposal) = &proposal {
                        if proposal.status.get() != agent::recipes::RecipeReviewStatus::Pending { return; }
                        if workspace.has_active_modal(window, cx) {
                            proposal.status.set(agent::recipes::RecipeReviewStatus::Unavailable);
                            return;
                        }
                    }
                    if link.is_some() && workspace.has_active_modal(window, cx) {
                        workspace.show_error("Close the current dialog and open the recipe link again; your existing draft was preserved", cx);
                        return;
                    }
                    if let Some(store) = agent::recipes::RecipeStore::try_global(cx) {
                        workspace.toggle_modal(window, cx, |window, cx| {
                            let mut view = crate::recipes_view::RecipesView::new(
                                panel, project, connection, store, schedule, window, cx,
                            );
                            if let Some(proposal) = proposal { view.review_proposal(proposal, window, cx); }
                            if let Some(link) = link { view.review_link(link, window, cx); }
                            view
                        });
                    } else {
                        workspace.show_error("Recipe storage is unavailable", cx);
                    }
                })?;
                Ok(())
            }.await;
            if let Err(error) = result {
                workspace.update_in(cx, |workspace, _, cx| workspace.show_error(error, cx))?;
            }
            anyhow::Ok(())
        }).detach_and_log_err(cx);
    }

    /// Refresh the native agent's view of available skills
    pub fn refresh_skills(&mut self, cx: &mut Context<Self>) {
        if !self.has_open_project(cx) {
            return;
        }

        self.ensure_native_agent_connection(cx);
        let Some(connect_task) = self.connection_store.update(cx, |store, cx| {
            store
                .entry(&Agent::NativeAgent)
                .map(|entry| entry.read(cx).wait_for_connection())
        }) else {
            return;
        };
        let project = self.project.clone();
        cx.spawn(async move |_this, cx| -> Result<()> {
            let connected = connect_task.await?;
            if let Some(native_connection) = connected
                .connection
                .downcast::<agent::NativeAgentConnection>()
            {
                cx.update(|cx| native_connection.refresh_skills_for_project(project, cx));
            }
            Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn expand_message_editor(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(conversation_view) = self.active_conversation_view() else {
            return;
        };

        let Some(active_thread) = conversation_view.read(cx).root_thread_view() else {
            return;
        };

        active_thread.update(cx, |active_thread, cx| {
            active_thread.expand_message_editor(&ExpandMessageEditor, window, cx);
            active_thread.activation_focus_handle(cx).focus(window, cx);
        })
    }

    pub fn toggle_options_menu(
        &mut self,
        _: &ToggleOptionsMenu,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle, cx);
        self.agent_panel_menu_handle.toggle(window, cx);
    }

    pub fn toggle_new_thread_menu(
        &mut self,
        _: &ToggleNewThreadMenu,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.has_open_project(cx) {
            return;
        }

        self.new_thread_menu_handle.toggle(window, cx);
    }

    pub fn increase_font_size(
        &mut self,
        action: &IncreaseBufferFontSize,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.handle_font_size_action(action.persist, px(1.0), cx);
    }

    pub fn decrease_font_size(
        &mut self,
        action: &DecreaseBufferFontSize,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.handle_font_size_action(action.persist, px(-1.0), cx);
    }

    fn handle_font_size_action(&mut self, persist: bool, delta: Pixels, cx: &mut Context<Self>) {
        match self.visible_font_size() {
            WhichFontSize::AgentFont => {
                if persist {
                    update_settings_file(self.fs.clone(), cx, move |settings, cx| {
                        let agent_ui_font_size =
                            ThemeSettings::get_global(cx).agent_ui_font_size(cx) + delta;
                        let agent_buffer_font_size =
                            ThemeSettings::get_global(cx).agent_buffer_font_size(cx) + delta;

                        let _ = settings.theme.agent_ui_font_size.insert(
                            f32::from(theme_settings::clamp_font_size(agent_ui_font_size)).into(),
                        );
                        let _ = settings.theme.agent_buffer_font_size.insert(
                            f32::from(theme_settings::clamp_font_size(agent_buffer_font_size))
                                .into(),
                        );
                    });
                } else {
                    theme_settings::adjust_agent_ui_font_size(cx, |size| size + delta);
                    theme_settings::adjust_agent_buffer_font_size(cx, |size| size + delta);
                }
            }
            WhichFontSize::None => {
                // The agent panel does not own this font size (e.g. when a
                // terminal is the visible surface). Let the action bubble up
                // to the workspace handler so the global buffer font size is
                // adjusted instead.
                cx.propagate();
            }
        }
    }

    pub fn reset_font_size(
        &mut self,
        action: &ResetBufferFontSize,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match self.visible_font_size() {
            WhichFontSize::AgentFont => {
                if action.persist {
                    update_settings_file(self.fs.clone(), cx, move |settings, _| {
                        settings.theme.agent_ui_font_size = None;
                        settings.theme.agent_buffer_font_size = None;
                    });
                } else {
                    theme_settings::reset_agent_ui_font_size(cx);
                    theme_settings::reset_agent_buffer_font_size(cx);
                }
            }
            WhichFontSize::None => {
                // Let the workspace handler reset the global buffer font size
                // that the terminal uses.
                cx.propagate();
            }
        }
    }

    pub fn reset_agent_zoom(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        theme_settings::reset_agent_ui_font_size(cx);
        theme_settings::reset_agent_buffer_font_size(cx);
    }

    pub fn toggle_zoom(&mut self, _: &ToggleZoom, window: &mut Window, cx: &mut Context<Self>) {
        if self.zoomed {
            cx.emit(PanelEvent::ZoomOut);
        } else {
            if !self.focus_handle(cx).contains_focused(window, cx) {
                self.activation_focus_handle(cx).focus(window, cx);
            }
            cx.emit(PanelEvent::ZoomIn);
        }
    }

    pub(crate) fn open_configuration(&mut self, window: &mut Window, _cx: &mut Context<Self>) {
        // Revealing the panel can focus an input absent from the current dispatch tree.
        window.on_next_frame(|window, cx| {
            window.dispatch_action(
                Box::new(zed_actions::OpenSettingsAt {
                    path: "llm_providers".into(),
                    target: None,
                }),
                cx,
            );
        });
    }

    pub(crate) fn open_active_thread_as_markdown(
        &mut self,
        _: &OpenActiveThreadAsMarkdown,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(workspace) = self.workspace.upgrade()
            && let Some(conversation_view) = self.active_conversation_view()
            && let Some(active_thread) = conversation_view.read(cx).active_thread().cloned()
        {
            active_thread.update(cx, |thread, cx| {
                thread
                    .open_thread_as_markdown(workspace, window, cx)
                    .detach_and_log_err(cx);
            });
        }
    }

    pub fn open_thread_as_markdown(
        &mut self,
        thread_id: ThreadId,
        workspace: Entity<Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(conversation_view) = self.conversation_view_for_id(&thread_id, cx).cloned() else {
            return false;
        };
        let Some(thread_view) = conversation_view.read(cx).root_thread_view() else {
            return false;
        };
        thread_view.update(cx, |thread, cx| {
            thread
                .open_thread_as_markdown(workspace, window, cx)
                .detach_and_log_err(cx);
        });
        true
    }

    pub fn shared_thread_clipboard_text(thread: &agent::DbThread) -> anyhow::Result<String> {
        let thread_data = SharedThread::from_db_thread(thread).to_bytes()?;
        Ok(base64::Engine::encode(
            &base64::prelude::BASE64_STANDARD,
            &thread_data,
        ))
    }

    fn copy_thread_to_clipboard(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(thread) = self.active_native_agent_thread(cx) else {
            Self::show_deferred_toast(&self.workspace, "No active native thread to copy", cx);
            return;
        };

        let workspace = self.workspace.clone();
        let load_task = thread.read(cx).to_db(cx);

        cx.spawn_in(window, async move |_this, cx| {
            let db_thread = load_task.await;
            let encoded = Self::shared_thread_clipboard_text(&db_thread)?;

            cx.update(|_window, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(encoded));
                if let Some(workspace) = workspace.upgrade() {
                    workspace.update(cx, |workspace, cx| {
                        struct ThreadCopiedToast;
                        workspace.show_toast(
                            workspace::Toast::new(
                                workspace::notifications::NotificationId::unique::<ThreadCopiedToast>(),
                                "Thread copied to clipboard (base64 encoded)",
                            )
                            .autohide(),
                            cx,
                        );
                    });
                }
            })?;

            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn show_deferred_toast(
        workspace: &WeakEntity<workspace::Workspace>,
        message: &'static str,
        cx: &mut App,
    ) {
        let workspace = workspace.clone();
        cx.defer(move |cx| {
            if let Some(workspace) = workspace.upgrade() {
                workspace.update(cx, |workspace, cx| {
                    struct ClipboardToast;
                    workspace.show_toast(
                        workspace::Toast::new(
                            workspace::notifications::NotificationId::unique::<ClipboardToast>(),
                            message,
                        )
                        .autohide(),
                        cx,
                    );
                });
            }
        });
    }

    fn load_thread_from_clipboard(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.has_open_project(cx) {
            Self::show_deferred_toast(&self.workspace, "Open a project to load a thread", cx);
            return;
        }

        let Some(clipboard) = cx.read_from_clipboard() else {
            Self::show_deferred_toast(&self.workspace, "No clipboard content available", cx);
            return;
        };

        let Some(encoded) = clipboard.text() else {
            Self::show_deferred_toast(&self.workspace, "Clipboard does not contain text", cx);
            return;
        };

        let thread_data = match base64::Engine::decode(&base64::prelude::BASE64_STANDARD, &encoded)
        {
            Ok(data) => data,
            Err(_) => {
                Self::show_deferred_toast(
                    &self.workspace,
                    "Failed to decode clipboard content (expected base64)",
                    cx,
                );
                return;
            }
        };

        let shared_thread = match SharedThread::from_bytes(&thread_data) {
            Ok(thread) => thread,
            Err(_) => {
                Self::show_deferred_toast(
                    &self.workspace,
                    "Failed to parse thread data from clipboard",
                    cx,
                );
                return;
            }
        };

        let db_thread = shared_thread.to_db_thread();
        let session_id = acp::SessionId::new(uuid::Uuid::new_v4().to_string());
        let thread_store = self.thread_store.clone();
        let title = db_thread.title.clone();
        let workspace = self.workspace.clone();

        cx.spawn_in(window, async move |this, cx| {
            thread_store
                .update(&mut cx.clone(), |store, cx| {
                    store.save_thread(session_id.clone(), db_thread, Default::default(), cx)
                })
                .await?;

            this.update_in(cx, |this, window, cx| {
                this.open_thread(session_id, None, Some(title), window, cx);
            })?;

            this.update_in(cx, |_, _window, cx| {
                if let Some(workspace) = workspace.upgrade() {
                    workspace.update(cx, |workspace, cx| {
                        struct ThreadLoadedToast;
                        workspace.show_toast(
                            workspace::Toast::new(
                                workspace::notifications::NotificationId::unique::<ThreadLoadedToast>(),
                                "Thread loaded from clipboard",
                            )
                            .autohide(),
                            cx,
                        );
                    });
                }
            })?;

            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn show_thread_metadata(
        &mut self,
        _: &ShowThreadMetadata,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(thread_id) = self.active_thread_id(cx) else {
            Self::show_deferred_toast(&self.workspace, "No active thread", cx);
            return;
        };

        let Some(store) = ThreadMetadataStore::try_global(cx) else {
            Self::show_deferred_toast(&self.workspace, "Thread metadata store not available", cx);
            return;
        };

        let Some(metadata) = store.read(cx).entry(thread_id).cloned() else {
            Self::show_deferred_toast(&self.workspace, "No metadata found for active thread", cx);
            return;
        };

        let json = thread_metadata_to_debug_json(&metadata);
        let text = serde_json::to_string_pretty(&json).unwrap_or_default();
        let title = format!("Thread Metadata: {}", metadata.display_title());

        self.open_json_buffer(title, text, window, cx);
    }

    fn show_all_sidebar_thread_metadata(
        &mut self,
        _: &ShowAllSidebarThreadMetadata,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(store) = ThreadMetadataStore::try_global(cx) else {
            Self::show_deferred_toast(&self.workspace, "Thread metadata store not available", cx);
            return;
        };

        let entries: Vec<serde_json::Value> = store
            .read(cx)
            .entries()
            .filter(|t| !t.archived)
            .map(thread_metadata_to_debug_json)
            .collect();

        let json = serde_json::Value::Array(entries);
        let text = serde_json::to_string_pretty(&json).unwrap_or_default();

        self.open_json_buffer("All Sidebar Thread Metadata".to_string(), text, window, cx);
    }

    fn open_json_buffer(
        &self,
        title: String,
        text: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let json_language = self.language_registry.language_for_name("JSON");
        let project = self.project.clone();
        let workspace = self.workspace.clone();

        window
            .spawn(cx, async move |cx| {
                let json_language = json_language.await.ok();

                let buffer = project
                    .update(cx, |project, cx| {
                        project.create_buffer(json_language, false, cx)
                    })
                    .await?;

                buffer.update(cx, |buffer, cx| {
                    buffer.set_text(text, cx);
                    buffer.set_capability(language::Capability::ReadWrite, cx);
                });

                workspace.update_in(cx, |workspace, window, cx| {
                    let buffer =
                        cx.new(|cx| MultiBuffer::singleton(buffer, cx).with_title(title.clone()));

                    workspace.add_item_to_active_pane(
                        Box::new(cx.new(|cx| {
                            let mut editor =
                                Editor::for_multibuffer(buffer, Some(project.clone()), window, cx);
                            editor.set_breadcrumb_header(title);
                            editor.disable_mouse_wheel_zoom();
                            editor
                        })),
                        None,
                        true,
                        window,
                        cx,
                    );
                })?;

                anyhow::Ok(())
            })
            .detach_and_log_err(cx);
    }

    pub fn workspace_id(&self) -> Option<WorkspaceId> {
        self.workspace_id
    }

    pub fn retained_threads(&self) -> &HashMap<ThreadId, Entity<ConversationView>> {
        &self.retained_threads
    }

    pub fn active_conversation_view(&self) -> Option<&Entity<ConversationView>> {
        match &self.base_view {
            BaseView::AgentThread { conversation_view } => Some(conversation_view),
            _ => None,
        }
    }

    pub(crate) fn visible_conversation_view(&self) -> Option<&Entity<ConversationView>> {
        match self.visible_surface() {
            VisibleSurface::AgentThread(conversation_view) => Some(conversation_view),
            _ => None,
        }
    }

    pub fn visible_terminal_view(&self) -> Option<&Entity<TerminalView>> {
        match self.visible_surface() {
            VisibleSurface::Terminal(terminal_view) => Some(terminal_view),
            _ => None,
        }
    }

    fn toggle_terminal_thread_search(
        &mut self,
        _: &crate::ToggleSearch,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(terminal) = self
            .active_terminal_id()
            .and_then(|terminal_id| self.terminals.get_mut(&terminal_id))
        else {
            cx.propagate();
            return;
        };

        let terminal_view = terminal.view.clone();
        let search_bar = terminal
            .search_bar
            .get_or_insert_with(|| cx.new(|cx| BufferSearchBar::new(None, window, cx)))
            .clone();
        let deployed = search_bar.update(cx, |search_bar, cx| {
            let terminal_item: &dyn ItemHandle = &terminal_view;
            search_bar.set_active_pane_item(Some(terminal_item), window, cx);
            search_bar.deploy(&DeployBufferSearch::find(), None, window, cx)
        });
        if deployed {
            cx.stop_propagation();
        } else {
            cx.propagate();
        }
    }

    pub fn conversation_view_for_id(
        &self,
        thread_id: &ThreadId,
        cx: &App,
    ) -> Option<&Entity<ConversationView>> {
        self.retained_threads.get(thread_id).or_else(|| {
            if let Some(view) = self.active_conversation_view()
                && view.read(cx).thread_id == *thread_id
            {
                Some(view)
            } else {
                None
            }
        })
    }

    pub fn regenerate_thread_title(
        &mut self,
        thread_id: ThreadId,
        cx: &mut Context<Self>,
    ) -> ThreadTitleRegenerationResult {
        let Some(conversation_view) = self.conversation_view_for_id(&thread_id, cx).cloned() else {
            return ThreadTitleRegenerationResult::NotOpen;
        };
        Self::regenerate_conversation_thread_title(conversation_view, cx)
    }

    fn regenerate_conversation_thread_title(
        conversation_view: Entity<ConversationView>,
        cx: &mut App,
    ) -> ThreadTitleRegenerationResult {
        let Some(thread) = conversation_view.read(cx).as_native_thread(cx) else {
            return ThreadTitleRegenerationResult::NotOpen;
        };
        let thread_id = conversation_view.read(cx).parent_id();
        thread.update(cx, |thread, cx| {
            if thread.is_generating_title() {
                ThreadTitleRegenerationResult::AlreadyGenerating
            } else if thread.summarization_model().is_none() {
                ThreadTitleRegenerationResult::NoModel
            } else if thread.regenerate_title_with_callback(cx, move |title, cx| {
                ThreadMetadataStore::global(cx).update(cx, |store, cx| {
                    store.set_generated_title(thread_id, title, cx);
                });
            }) {
                ThreadTitleRegenerationResult::Started
            } else {
                ThreadTitleRegenerationResult::AlreadyGenerating
            }
        })
    }

    pub fn conversation_views(&self) -> Vec<Entity<ConversationView>> {
        self.active_conversation_view()
            .into_iter()
            .cloned()
            .chain(self.retained_threads.values().cloned())
            .collect()
    }

    pub fn active_thread_view(&self, cx: &App) -> Option<Entity<ThreadView>> {
        let server_view = self.active_conversation_view()?;
        server_view.read(cx).root_thread_view()
    }

    pub(crate) fn conversation_for_review_thread(
        &self,
        thread: &Entity<AcpThread>,
        cx: &App,
    ) -> Option<Entity<ConversationView>> {
        let session = thread.read(cx).session_id();
        self.active_conversation_view()
            .into_iter()
            .chain(self.retained_threads.values())
            .chain(self.draft_thread.iter())
            .find(|conversation| {
                conversation
                    .read(cx)
                    .thread_view(session)
                    .is_some_and(|view| view.read(cx).thread == *thread)
            })
            .cloned()
    }

    pub(crate) fn focus_review_composer(
        &mut self,
        thread: &Entity<AcpThread>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        let conversation = self
            .conversation_for_review_thread(thread, cx)
            .ok_or_else(|| {
                anyhow!("This task's composer is unavailable. Reopen the task and try again.")
            })?;
        let session = thread.read(cx).session_id().clone();
        self.retained_threads
            .remove(&conversation.read(cx).thread_id);
        self.set_base_view(
            BaseView::AgentThread {
                conversation_view: conversation.clone(),
            },
            true,
            window,
            cx,
        );
        conversation.update(cx, |conversation, cx| {
            conversation.navigate_to_thread(session.clone(), window, cx)
        });
        let view = conversation
            .read(cx)
            .thread_view(&session)
            .ok_or_else(|| anyhow!("The task closed while opening its composer"))?;
        view.read(cx)
            .message_editor
            .focus_handle(cx)
            .focus(window, cx);
        Ok(())
    }

    pub fn active_agent_thread(&self, cx: &App) -> Option<Entity<AcpThread>> {
        match &self.base_view {
            BaseView::AgentThread { conversation_view } => {
                conversation_view.read(cx).root_thread(cx)
            }
            _ => None,
        }
    }

    pub fn is_retained_thread(&self, id: &ThreadId) -> bool {
        self.retained_threads.contains_key(id)
    }

    pub fn cancel_thread(&self, thread_id: &ThreadId, cx: &mut Context<Self>) -> bool {
        let conversation_views = self
            .active_conversation_view()
            .into_iter()
            .chain(self.retained_threads.values());

        for conversation_view in conversation_views {
            if *thread_id == conversation_view.read(cx).thread_id {
                if let Some(thread_view) = conversation_view.read(cx).root_thread_view() {
                    thread_view.update(cx, |view, cx| view.cancel_generation(cx));
                    return true;
                }
            }
        }
        false
    }

    fn update_thread_work_dirs(&self, cx: &mut Context<Self>) {
        let new_work_dirs = self.project.read(cx).default_path_list(cx);
        let new_worktree_paths = self.project.read(cx).worktree_paths(cx);

        if let Some(conversation_view) = self.active_conversation_view() {
            conversation_view.update(cx, |conversation_view, cx| {
                conversation_view.set_work_dirs(new_work_dirs.clone(), cx);
            });
        }

        for conversation_view in self.retained_threads.values() {
            conversation_view.update(cx, |conversation_view, cx| {
                conversation_view.set_work_dirs(new_work_dirs.clone(), cx);
            });
        }

        if self.project.read(cx).is_via_collab() {
            return;
        }

        // Update metadata store so threads' path lists stay in sync with
        // the project's current worktrees. Without this, threads saved
        // before a worktree was added would have stale paths and not
        // appear under the correct sidebar group.
        let mut thread_ids: Vec<ThreadId> = self.retained_threads.keys().copied().collect();
        if let Some(active_id) = self.active_thread_id(cx) {
            thread_ids.push(active_id);
        }
        if !thread_ids.is_empty() {
            ThreadMetadataStore::global(cx).update(cx, |store, cx| {
                store.update_worktree_paths(&thread_ids, new_worktree_paths, cx);
            });
        }
    }

    fn retain_running_thread(&mut self, old_view: BaseView, cx: &mut Context<Self>) {
        let BaseView::AgentThread { conversation_view } = old_view else {
            return;
        };

        if self
            .draft_thread
            .as_ref()
            .is_some_and(|d| d.entity_id() == conversation_view.entity_id())
        {
            if self.draft_has_content(&conversation_view, cx) {
                let thread_id = conversation_view.read(cx).thread_id;
                self.draft_thread = None;
                self._draft_editor_observation = None;
                self.retained_threads.insert(thread_id, conversation_view);
                self.cleanup_retained_threads(cx);
            }
            return;
        }

        let thread_id = conversation_view.read(cx).thread_id;

        if self.retained_threads.contains_key(&thread_id) {
            return;
        }

        self.retained_threads.insert(thread_id, conversation_view);
        self.cleanup_retained_threads(cx);
    }

    fn cleanup_retained_threads(&mut self, cx: &App) {
        let mut potential_removals = self
            .retained_threads
            .iter()
            .filter(|(_id, view)| {
                let Some(thread_view) = view.read(cx).root_thread_view() else {
                    return !view.read(cx).is_loading();
                };
                let thread = thread_view.read(cx).thread.read(cx);
                thread.connection().supports_load_session() && thread.status() == ThreadStatus::Idle
            })
            .collect::<Vec<_>>();

        let max_idle = MaxIdleRetainedThreads::global(cx);

        potential_removals.sort_unstable_by_key(|(_, view)| view.read(cx).updated_at(cx));
        let n = potential_removals.len().saturating_sub(max_idle);
        let to_remove = potential_removals
            .into_iter()
            .map(|(id, _)| *id)
            .take(n)
            .collect::<Vec<_>>();
        for id in to_remove {
            self.retained_threads.remove(&id);
        }
    }

    pub(crate) fn active_native_agent_thread(&self, cx: &App) -> Option<Entity<agent::Thread>> {
        match &self.base_view {
            BaseView::AgentThread { conversation_view } => {
                conversation_view.read(cx).as_native_thread(cx)
            }
            _ => None,
        }
    }

    fn set_base_view(
        &mut self,
        new_view: BaseView,
        focus: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        #[cfg(feature = "multiplayer-tools")]
        {
            self.pending_inline_terminal_open = None;
        }
        let old_view = std::mem::replace(&mut self.base_view, new_view);
        self.retain_running_thread(old_view, cx);

        if let BaseView::AgentThread { conversation_view } = &self.base_view {
            let conversation_view = conversation_view.read(cx);
            let thread_agent = conversation_view.agent_key().clone();
            if self.selected_agent != thread_agent {
                self.selected_agent = thread_agent;
                self.serialize(cx);
            }
        }

        self.refresh_base_view_subscriptions(window, cx);

        if focus {
            self.focus_base_view(window, cx);
        }
        cx.emit(AgentPanelEvent::ActiveViewChanged);
    }

    fn focus_base_view(&self, window: &mut Window, cx: &mut Context<Self>) {
        self.activation_focus_handle(cx).focus(window, cx);
        #[cfg(feature = "multiplayer-tools")]
        if let Some(conversation) = self.active_conversation_view() {
            let conversation_id = conversation.entity_id();
            let panel = cx.weak_entity();
            let workspace = self.workspace.clone();
            // Navigation serializes the workspace, so wait until this panel update finishes.
            window.defer(cx, move |window, cx| {
                let Some(panel) = panel.upgrade() else {
                    return;
                };
                if panel
                    .read(cx)
                    .active_conversation_view()
                    .map(Entity::entity_id)
                    != Some(conversation_id)
                {
                    return;
                }
                let Some(workspace) = workspace.upgrade() else {
                    return;
                };
                workspace.update(cx, |workspace, cx| {
                    Self::synchronize_collaborative_thread_navigation(
                        workspace, &panel, false, window, cx,
                    )
                    .log_err();
                });
            });
        }
    }

    #[cfg(feature = "multiplayer-tools")]
    pub fn synchronize_collaborative_thread_navigation(
        workspace: &mut Workspace,
        panel: &Entity<Self>,
        preserve_non_thread_target: bool,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> anyhow::Result<()> {
        use workspace::collaborative_navigation::CollaborativeNavigationTarget;

        if workspace.workspace_presentation() != workspace::WorkspacePresentation::Collaborative
            || workspace.panel::<Self>(cx).as_ref() != Some(panel)
            || panel.read(cx).project != *workspace.project()
        {
            return Ok(());
        }
        if preserve_non_thread_target
            && workspace
                .collaborative_navigation()
                .current()
                .is_some_and(|target| {
                    !matches!(target, CollaborativeNavigationTarget::Thread { .. })
                })
        {
            return Ok(());
        }
        let Some(conversation) = panel.read(cx).active_conversation_view() else {
            return Ok(());
        };
        let target = CollaborativeNavigationTarget::thread(
            conversation.read(cx).parent_id().to_key_string(),
        );
        workspace.navigate_collaborative_to(
            target.clone(),
            |candidate| candidate == &target,
            window,
            cx,
        )?;
        Ok(())
    }

    fn refresh_base_view_subscriptions(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self._base_view_observation = match &self.base_view {
            BaseView::AgentThread { conversation_view } => {
                self._thread_view_subscription =
                    Self::subscribe_to_active_thread_view(conversation_view, window, cx);
                let focus_handle = conversation_view.focus_handle(cx);
                self._active_thread_focus_subscription =
                    Some(cx.on_focus_in(&focus_handle, window, |_this, _window, cx| {
                        cx.emit(AgentPanelEvent::ActiveViewFocused);
                        cx.notify();
                    }));
                let cv = conversation_view.clone();
                self.observe_active_draft_for_empty_editor(&cv, cx);
                Some(cx.observe_in(&cv, window, |this, server_view, window, cx| {
                    this._thread_view_subscription =
                        Self::subscribe_to_active_thread_view(&server_view, window, cx);
                    this.observe_active_draft_for_empty_editor(&server_view, cx);
                    cx.emit(AgentPanelEvent::ActiveViewChanged);
                    this.serialize(cx);
                    cx.notify();
                }))
            }
            BaseView::Terminal { terminal_id } => {
                self._thread_view_subscription = None;
                if let Some(terminal) = self.terminals.get(terminal_id) {
                    let terminal_id = *terminal_id;
                    let focus_handle = terminal.view.focus_handle(cx);
                    self._active_thread_focus_subscription =
                        Some(
                            cx.on_focus_in(&focus_handle, window, move |this, _window, cx| {
                                if let Some(terminal) = this.terminals.get_mut(&terminal_id) {
                                    terminal.has_notification = false;
                                }
                                cx.emit(AgentPanelEvent::ActiveViewFocused);
                                cx.notify();
                            }),
                        );
                } else {
                    self._active_thread_focus_subscription = None;
                }
                None
            }
            BaseView::Uninitialized => {
                self._thread_view_subscription = None;
                self._active_thread_focus_subscription = None;
                None
            }
        };
        self.serialize(cx);
    }

    fn visible_surface(&self) -> VisibleSurface<'_> {
        match &self.base_view {
            BaseView::Uninitialized => VisibleSurface::Uninitialized,
            BaseView::AgentThread { conversation_view } => {
                VisibleSurface::AgentThread(conversation_view)
            }
            BaseView::Terminal { terminal_id } => self
                .terminals
                .get(terminal_id)
                .map(|terminal| VisibleSurface::Terminal(&terminal.view))
                .unwrap_or(VisibleSurface::Uninitialized),
        }
    }

    fn visible_font_size(&self) -> WhichFontSize {
        self.base_view.which_font_size_used()
    }

    fn subscribe_to_active_thread_view(
        server_view: &Entity<ConversationView>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Subscription> {
        server_view.read(cx).root_thread_view().map(|tv| {
            cx.subscribe_in(
                &tv,
                window,
                |this, _view, event: &AcpThreadViewEvent, _window, cx| match event {
                    AcpThreadViewEvent::Interacted => {
                        let Some(thread_id) = this.active_thread_id(cx) else {
                            return;
                        };
                        // If the draft was the active thread, it has now been
                        // promoted to a real thread. Clear the ephemeral
                        // pointer; the ConversationView itself stays put as
                        // the active base view.
                        if this
                            .draft_thread
                            .as_ref()
                            .is_some_and(|draft| draft.read(cx).thread_id == thread_id)
                        {
                            this.draft_thread = None;
                            this._draft_editor_observation = None;
                        }
                        this.retained_threads.remove(&thread_id);
                        cx.emit(AgentPanelEvent::ThreadInteracted { thread_id });
                    }
                },
            )
        })
    }

    fn migrate_agent_server_from_extensions(&mut self, id: Arc<str>, cx: &mut Context<Self>) {
        self.project.update(cx, |project, cx| {
            project.agent_server_store().update(cx, |store, cx| {
                store.migrate_agent_server_from_extensions(id, project.fs().clone(), cx);
            });
        });
    }

    pub fn new_agent_thread_with_external_source_prompt(
        &mut self,
        external_source_prompt: Option<ExternalSourcePrompt>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.external_thread(
            None,
            None,
            None,
            None,
            external_source_prompt.map(AgentInitialContent::from),
            true,
            AgentThreadSource::AgentPanel,
            window,
            cx,
        );
    }

    fn report_draft_restore_error(&self, error: anyhow::Error, cx: &mut Context<Self>) {
        let workspace = self.workspace.clone();
        let error =
            error.context("This unsent draft could not be restored. Its saved data was retained");
        log::error!("{error:#}");
        cx.defer(move |cx| {
            workspace
                .update(cx, |workspace, cx| workspace.show_error(error, cx))
                .log_err();
        });
    }

    pub fn load_agent_thread(
        &mut self,
        agent: Agent,
        thread_id: ThreadId,
        work_dirs: Option<PathList>,
        title: Option<SharedString>,
        focus: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(store) = ThreadMetadataStore::try_global(cx) {
            store.update(cx, |store, cx| {
                store.unarchive(thread_id, cx);
            });
        }

        // Check if the active view already holds this thread.
        if let BaseView::AgentThread { conversation_view } = &self.base_view {
            if conversation_view.read(cx).thread_id == thread_id {
                cx.emit(AgentPanelEvent::ActiveViewChanged);
                return;
            }
        }

        // Check if the thread is already in memory — either as the
        // ephemeral draft pointer or in retained_threads. Either way we
        // can just reactivate without touching storage.
        if let Some(draft) = self.draft_thread.clone()
            && draft.read(cx).thread_id == thread_id
        {
            self.set_base_view(
                BaseView::AgentThread {
                    conversation_view: draft,
                },
                focus,
                window,
                cx,
            );
            return;
        }
        if let Some(conversation_view) = self.retained_threads.remove(&thread_id) {
            self.try_make_empty_draft_ephemeral(conversation_view.clone(), cx);
            self.set_base_view(
                BaseView::AgentThread { conversation_view },
                focus,
                window,
                cx,
            );
            return;
        }

        // Not in memory. Build a fresh ConversationView. For drafts we
        // also seed the message editor with any prompt text the user had
        // typed before closing the window (persisted in the scoped kvp
        // draft-prompt store).
        let is_draft = ThreadMetadataStore::try_global(cx)
            .and_then(|store| store.read(cx).entry(thread_id).map(|m| m.is_draft()))
            .unwrap_or(false);
        let restored = if is_draft {
            match crate::draft_prompt_store::try_read_for_restore(thread_id, cx) {
                Ok(restored) => restored,
                Err(error) => {
                    self.report_draft_restore_error(error, cx);
                    return;
                }
            }
        } else {
            None
        };
        let initial_content = restored.map(|blocks| AgentInitialContent::RestoredDraft { blocks });

        self.external_thread(
            Some(agent),
            Some(thread_id),
            work_dirs,
            title,
            initial_content,
            focus,
            source,
            window,
            cx,
        );
    }

    pub fn load_transcript_for_export(
        &mut self,
        thread_id: ThreadId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<anyhow::Result<Entity<ThreadView>>> {
        let metadata_store = ThreadMetadataStore::global(cx);
        let store = metadata_store.read(cx);
        let Some(metadata) = store.entry(thread_id).cloned() else {
            return Task::ready(Err(anyhow!("This task is no longer available")));
        };
        if store.task_deletion_pending(thread_id) {
            return Task::ready(Err(anyhow!("Task deletion is in progress")));
        }
        let conversation = if let Some(view) = self.conversation_view_for_id(&thread_id, cx) {
            view.clone()
        } else {
            let Some(session_id) = metadata.session_id.clone() else {
                return Task::ready(Err(anyhow!("This task has no saved transcript")));
            };
            let thread = self.create_agent_thread_inner(
                Agent::from(metadata.agent_id.clone()),
                None,
                Some(thread_id),
                Some(session_id),
                Some(metadata.folder_paths().clone()),
                metadata.title(),
                None,
                None,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
            let view = thread.conversation_view;
            self.retained_threads.insert(thread_id, view.clone());
            view
        };
        conversation.update(cx, |conversation, cx| {
            conversation.wait_for_transcript(window, cx)
        })
    }

    pub(crate) fn create_agent_thread_with_server(
        &mut self,
        agent: Agent,
        server_override: Option<Rc<dyn AgentServer>>,
        resume_thread_id: Option<ThreadId>,
        work_dirs: Option<PathList>,
        title: Option<SharedString>,
        initial_content: Option<AgentInitialContent>,
        model_override: Option<String>,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AgentThread {
        self.set_selected_agent_and_persist(agent.clone(), cx);
        let resume_session_id = resume_thread_id.and_then(|tid| {
            ThreadMetadataStore::try_global(cx)
                .and_then(|store| store.read(cx).entry(tid).and_then(|m| m.session_id.clone()))
        });
        self.create_agent_thread_inner(
            agent,
            server_override,
            resume_thread_id,
            resume_session_id,
            work_dirs,
            title,
            initial_content,
            model_override,
            source,
            window,
            cx,
        )
    }

    /// Legacy entry that resumes a thread by raw ACP session id when no
    /// local [`ThreadMetadata`] row exists yet (share-link imports and
    /// clipboard imports).
    ///
    /// TODO(legacy-session-id): migrate remaining callers (share-link
    /// handler, clipboard import) to mint a [`ThreadId`] + seed metadata
    /// so they can route through [`create_agent_thread_with_server`] and
    /// this entry can be deleted.
    fn create_agent_thread_with_server_for_external_session(
        &mut self,
        agent: Agent,
        server_override: Option<Rc<dyn AgentServer>>,
        resume_session_id: acp::SessionId,
        work_dirs: Option<PathList>,
        title: Option<SharedString>,
        initial_content: Option<AgentInitialContent>,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AgentThread {
        self.set_selected_agent_and_persist(agent.clone(), cx);
        self.create_agent_thread_inner(
            agent,
            server_override,
            None,
            Some(resume_session_id),
            work_dirs,
            title,
            initial_content,
            None,
            source,
            window,
            cx,
        )
    }

    fn create_agent_thread_inner(
        &mut self,
        agent: Agent,
        server_override: Option<Rc<dyn AgentServer>>,
        resume_thread_id: Option<ThreadId>,
        resume_session_id: Option<acp::SessionId>,
        work_dirs: Option<PathList>,
        title: Option<SharedString>,
        initial_content: Option<AgentInitialContent>,
        model_override: Option<String>,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AgentThread {
        let thread_id = resume_thread_id.unwrap_or_else(ThreadId::new);
        let workspace = self.workspace.clone();
        let project = self.project.clone();

        let server = server_override
            .unwrap_or_else(|| agent.server(self.fs.clone(), self.thread_store.clone()));
        let thread_store = server
            .clone()
            .downcast::<agent::NativeAgentServer>()
            .is_some()
            .then(|| self.thread_store.clone());

        let connection_store = self.connection_store.clone();

        let conversation_view = cx.new(|cx| {
            crate::ConversationView::new(
                server,
                connection_store,
                agent,
                resume_session_id,
                Some(thread_id),
                work_dirs,
                title,
                initial_content,
                workspace.clone(),
                project,
                thread_store,
                source,
                window,
                cx,
            )
        });

        cx.observe_in(
            &conversation_view,
            window,
            |this, server_view, window, cx| {
                let is_active = this
                    .active_conversation_view()
                    .is_some_and(|active| active.entity_id() == server_view.entity_id());
                if is_active {
                    cx.emit(AgentPanelEvent::ActiveViewChanged);
                    this.serialize(cx);
                } else {
                    cx.emit(AgentPanelEvent::EntryChanged);
                }
                this.ensure_doctor_events_installed(&server_view, window, cx);
                this.ensure_sibling_host_installed(&server_view, window, cx);
                cx.notify();
            },
        )
        .detach();

        // Try installing the host eagerly as well, in case the connection is
        // already established by the time the observe fires.
        self.ensure_doctor_events_installed(&conversation_view, window, cx);
        self.ensure_sibling_host_installed(&conversation_view, window, cx);

        if let Some(model) = model_override {
            // The native thread is constructed asynchronously after the
            // connection establishes. Wait for the first `RootThreadUpdated`
            // event that yields a native thread, then apply the override once.
            let applied = Cell::new(false);
            cx.subscribe(
                &conversation_view,
                move |_this, view, _event: &RootThreadUpdated, cx| {
                    if applied.get() {
                        return;
                    }
                    let Some(native_thread) = view.read(cx).as_native_thread(cx) else {
                        return;
                    };
                    apply_native_model_override(&native_thread, &model, cx);
                    applied.set(true);
                },
            )
            .detach();
        }

        AgentThread { conversation_view }
    }

    fn doctor_native_repair(
        &mut self,
        request: Rc<diagnostics::doctor::DoctorNativeRepairRequest>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<diagnostics::doctor::DoctorNativeRepairResult>> {
        use diagnostics::doctor::{
            DiagnosticText, DoctorNativeRepairResult, DoctorRepairOperationOutcome,
        };
        let prepare = (|| {
            request.revalidate(cx)?;
            anyhow::ensure!(
                request.project == self.project,
                "Native Doctor repair project changed"
            );
            let path = self
                .project
                .read(cx)
                .absolute_path(&request.directory, cx)
                .ok_or_else(|| anyhow!("Selected native repair worktree is unavailable"))?;
            let thread = self.create_agent_thread_with_server(
                Agent::NativeAgent,
                None,
                None,
                Some(PathList::new(&[path.as_path()])),
                Some("Doctor repair proposal".into()),
                None,
                None,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
            Ok::<_, anyhow::Error>(thread.conversation_view)
        })();
        let conversation = match prepare {
            Ok(conversation) => conversation,
            Err(error) => return Task::ready(Err(error)),
        };
        let executor = cx.background_executor().clone();
        cx.spawn(async move |this, cx| {
            let started = executor.now();
            let (thread, native) = loop {
                cx.update(|cx| request.revalidate(cx))?;
                let ready = conversation.read_with(cx, |view, cx| {
                    if view.as_connected().is_some_and(|connected| connected.has_thread_error(cx)) { anyhow::bail!("Native session initialization failed; inspect its existing error surface"); }
                    Ok(view.root_thread(cx).zip(view.as_native_thread(cx)))
                })?;
                if let Some(ready) = ready { break ready; }
                anyhow::ensure!(executor.now().duration_since(started) < Duration::from_secs(30), "Native repair session initialization timed out; no prompt was submitted");
                executor.timer(Duration::from_millis(100)).await;
            };
            let session = thread.read_with(cx, |thread, _| thread.session_id().to_string());
            let restriction = cx.update(|cx| {
                request.revalidate(cx)?;
                let connection = conversation.read(cx).as_native_connection(cx)
                    .ok_or_else(|| anyhow!("Native repair connection is unavailable"))?;
                let session_id = thread.read(cx).session_id().clone();
                anyhow::ensure!(connection.thread(&session_id, cx).as_ref() == Some(&native), "Native repair session owner changed");
                Ok::<_, anyhow::Error>(connection.restrict_doctor_repair_proposal(&session_id, cx))
            });
            use futures::FutureExt;
            let restricted = match restriction {
                Ok(restriction) => {
                    let restriction = restriction.fuse();
                    let authority = async {
                        loop {
                            if let Err(error) = cx.update(|cx| request.revalidate(cx)) { return error; }
                            if executor.now().duration_since(started) >= Duration::from_secs(30) {
                                return anyhow!("Native proposal restriction persistence timed out; no prompt was submitted");
                            }
                            executor.timer(Duration::from_millis(100)).await;
                        }
                    }.fuse();
                    futures::pin_mut!(restriction, authority);
                    futures::select_biased! {
                        error = authority => Err(error),
                        result = restriction => result,
                    }
                }
                Err(error) => Err(error),
            };
            if let Err(error) = restricted {
                let outcome = if request.cancelled.is_closed() || !request.cancelled.is_empty() { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Failed };
                return Ok(DoctorNativeRepairResult { session_id: None, outcome, detail: DiagnosticText::new(format!("Native proposal initialization stopped: {error}. No prompt was submitted.")) });
            }
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
            request.record_session(&session, now, cx).await?;
            request.window.update(cx, |_, window, cx| {
                request.revalidate(cx)?;
                this.update(cx, |panel, cx| {
                    panel.set_base_view(AgentThread { conversation_view: conversation.clone() }.into(), false, window, cx);
                })?;
                Ok::<_, anyhow::Error>(())
            })??;
            let prepare = cx.update(|cx| {
                request.revalidate(cx)?;
                let configured = language_model::LanguageModelRegistry::read_global(cx).default_model()
                    .ok_or_else(|| anyhow!("Reviewed native repair model is unavailable"))?;
                anyhow::ensure!(configured.provider.id().0.as_ref() == request.provider
                    && configured.model.id().0.as_ref() == request.model, "Reviewed native repair model changed");
                native.update(cx, |thread, cx| thread.set_model(configured.model, cx));
                request.mark_prompt_submitted(cx)?;
                Ok::<_, anyhow::Error>(thread.update(cx, |thread, cx| thread.send(vec![request.prompt.clone().into()], cx)))
            });
            let send = match prepare {
                Ok(send) => send,
                Err(error) => return Ok(DoctorNativeRepairResult { session_id: Some(session), outcome: DoctorRepairOperationOutcome::Failed, detail: DiagnosticText::new(error.to_string()) }),
            };
            let outcome = {
                let send = send.fuse();
                let authority = async {
                    loop {
                        if let Err(error) = cx.update(|cx| request.revalidate(cx)) { return error; }
                        executor.timer(Duration::from_millis(100)).await;
                    }
                }.fuse();
                futures::pin_mut!(send, authority);
                futures::select_biased! {
                    error = authority => Err(error),
                    response = send => Ok(response),
                }
            };
            let (operation, detail) = match outcome {
                Err(error) => {
                    let cancellation = thread.update(cx, |thread, cx| thread.cancel(cx));
                    cancellation.await;
                    (if request.cancelled.is_closed() || !request.cancelled.is_empty() { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Failed },
                        DiagnosticText::new(format!("Native session cancelled through AcpThread: {error}. Reached source changes remain in normal review; no fix is verified.")))
                }
                Ok(Ok(Some(response))) if response.stop_reason == acp::StopReason::EndTurn => (
                    DoctorRepairOperationOutcome::AwaitingUserAction,
                    DiagnosticText::new("Native proposal turn ended. Inspect the existing session and normal change review; separately review and rerun the relevant validation target. Agent text is not verification.")),
                Ok(_) => (DoctorRepairOperationOutcome::Failed, DiagnosticText::new("Native proposal did not reach a complete normal turn. Inspect the existing session error surface; no fix is verified.")),
            };
            Ok(DoctorNativeRepairResult { session_id: Some(session), outcome: operation, detail })
        })
    }

    fn ensure_doctor_events_installed(
        &mut self,
        conversation_view: &Entity<ConversationView>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(connection) = conversation_view.read(cx).as_native_connection(cx) else {
            return;
        };
        let id = connection.0.entity_id();
        if self
            ._doctor_subscription
            .as_ref()
            .is_some_and(|(current, _)| *current == id)
        {
            return;
        }
        let context_connection = connection.clone();
        let subscription = cx.subscribe_in(
            &connection.0,
            window,
            move |this, _, request: &agent::DoctorOpenRequest, window, cx| {
                if request.project != this.project {
                    return;
                }
                let workspace = this.workspace.clone();
                let request = request.clone();
                let panel = cx.entity().downgrade();
                let context_connection = context_connection.clone();
                // Revealing the editor inspects dock panels, including this subscriber.
                window.defer(cx, move |window, cx| {
                    let result = workspace.update(cx, |workspace, cx| {
                        anyhow::ensure!(cx.has_flag::<feature_flags::CopperDoctor>(), "Copper Doctor is disabled");
                        anyhow::ensure!(workspace.project() == &request.project, "Doctor project scope changed");
                        anyhow::ensure!(
                            !request.project.read(cx).is_isolated_recipe_test(),
                            "Isolated recipe execution cannot open other desktop surfaces"
                        );
                        diagnostics::doctor_view::DoctorView::deploy(workspace, &diagnostics::Doctor, window, cx);
                        workspace.item_of_type::<diagnostics::doctor_view::DoctorView>(cx)
                            .ok_or_else(|| anyhow::anyhow!("Doctor desktop view is unavailable"))
                    }).and_then(|result| result);
                    match result {
                        Ok(view) => {
                            if let Err(error) = panel.update(cx, |panel, cx| {
                                let session_id = request.session_id.clone();
                                if panel._doctor_context_subscription.as_ref().is_some_and(|(entity, session, _)| *entity == view.entity_id() && *session == session_id) {
                                    return;
                                }
                                let project = request.project.clone();
                                let subscription = cx.subscribe(&view, {
                                    let context_connection = context_connection.clone();
                                    let session_id = session_id.clone();
                                    let project = project.clone();
                                    move |panel, view, _: &diagnostics::doctor_view::DoctorEvidenceUpdated, cx| {
                                        if panel.project != project { return; }
                                        if let Some(summary) = view.read(cx).conversation_context(cx)
                                            && let Err(error) = context_connection.record_doctor_context(&session_id, &project, &summary, cx)
                                        {
                                            log::warn!("Doctor conversation context unavailable: {error}");
                                            view.update(cx, |view, cx| view.conversation_context_failed(cx));
                                        }
                                    }
                                });
                                panel._doctor_context_subscription = Some((view.entity_id(), session_id.clone(), subscription));
                                if let Some(summary) = view.read(cx).conversation_context(cx)
                                    && let Err(error) = context_connection.record_doctor_context(&session_id, &project, &summary, cx)
                                {
                                    log::warn!("Doctor conversation context unavailable: {error}");
                                    view.update(cx, |view, cx| view.conversation_context_failed(cx));
                                }
                            }) {
                                request.complete(Err(error));
                                return;
                            }
                            window.on_next_frame(move |window, cx| {
                            let focused = view.focus_handle(cx).contains_focused(window, cx);
                            request.complete(if focused { Ok(()) } else {
                                Err(anyhow::anyhow!("Doctor could not receive focus in the project editor; invoke Doctor again"))
                            });
                            });
                        },
                        Err(error) => request.complete(Err(error)),
                    }
                });
            },
        );
        self._doctor_subscription = Some((id, subscription));
    }

    fn ensure_sibling_host_installed(
        &self,
        conversation_view: &Entity<ConversationView>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !cx.has_flag::<CreateThreadToolFeatureFlag>() {
            return;
        }
        let Some(native_connection) = conversation_view.read(cx).as_native_connection(cx) else {
            return;
        };
        let host = Rc::new(AgentPanelSiblingHost::new(
            cx.weak_entity(),
            window.window_handle(),
        )) as Rc<dyn agent::SiblingThreadHost>;
        native_connection.0.update(cx, |native_agent, _cx| {
            native_agent.set_sibling_thread_host(host);
        });
    }

    fn active_thread_has_messages(&self, cx: &App) -> bool {
        self.active_agent_thread(cx)
            .is_some_and(|thread| !thread.read(cx).entries().is_empty())
    }

    /// Whether the active view is in the **ephemeral** new-draft slot
    pub fn active_view_is_new_draft(&self, cx: &App) -> bool {
        self.draft_thread.as_ref().is_some_and(|draft| {
            draft
                .read(cx)
                .root_thread(cx)
                .is_some_and(|thread| thread.read(cx).is_draft_thread())
                && self
                    .active_conversation_view()
                    .is_some_and(|active| active.entity_id() == draft.entity_id())
        })
    }
    /// Whether the active thread is any kind of draft
    pub fn active_thread_is_draft(&self, cx: &App) -> bool {
        self.active_agent_thread(cx)
            .is_some_and(|thread| thread.read(cx).is_draft_thread())
    }
}

/// Apply a `provider/model-id` model override to a freshly-created native thread.
/// Best-effort: logs an error and leaves the default model in place if the
/// string can't be parsed or the model isn't registered.
pub(crate) fn apply_native_model_override(
    thread: &Entity<agent::Thread>,
    model_id: &str,
    cx: &mut App,
) {
    let Some(selected) = parse_provider_slash_model(model_id) else {
        log::warn!(
            "create_thread: could not parse model override {model_id:?}; expected `provider/model-id`"
        );
        return;
    };
    let configured = LanguageModelRegistry::global(cx)
        .update(cx, |registry, cx| registry.select_model(&selected, cx));
    let Some(configured) = configured else {
        log::warn!(
            "create_thread: no model registered for {model_id:?}; using thread's default model"
        );
        return;
    };
    thread.update(cx, |thread, cx| {
        thread.set_model(configured.model, cx);
    });
}

fn parse_provider_slash_model(input: &str) -> Option<language_model::SelectedModel> {
    let (provider, model) = input.split_once('/')?;
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    Some(language_model::SelectedModel {
        provider: language_model::LanguageModelProviderId::from(provider.to_string()),
        model: language_model::LanguageModelId::from(model.to_string()),
    })
}

/// Bridges agent-side `SiblingThreadHost` calls to `AgentPanel`. Constructed
/// and installed on a `NativeAgent` by the agent panel when a native-agent
/// thread is created.
pub(crate) struct AgentPanelSiblingHost {
    panel: WeakEntity<AgentPanel>,
    window: gpui::AnyWindowHandle,
}

impl AgentPanelSiblingHost {
    pub(crate) fn new(panel: WeakEntity<AgentPanel>, window: gpui::AnyWindowHandle) -> Self {
        Self { panel, window }
    }
}

impl agent::SiblingThreadHost for AgentPanelSiblingHost {
    fn create_sibling_thread(
        &self,
        request: agent::SiblingThreadRequest,
        cx: &mut gpui::AsyncApp,
    ) -> Task<Result<agent::SiblingThreadInfo>> {
        let panel = self.panel.clone();
        let window = self.window;
        cx.spawn(async move |cx| {
            let agent_choice = match request.agent_id.as_deref() {
                None => None,
                Some(id) if id == agent::ZED_AGENT_ID.as_ref() => Some(Agent::NativeAgent),
                Some(id) => {
                    // Reject unknown agent ids up front so the model gets a
                    // structured error pointing at `list_agents_and_models`,
                    // rather than a thread that silently fails to launch in
                    // the user's sidebar.
                    let known = panel
                        .read_with(cx, |panel, cx| {
                            let store = panel.project.read(cx).agent_server_store().clone();
                            store
                                .read(cx)
                                .external_agents()
                                .any(|known_id| known_id.0.as_ref() == id)
                        })
                        .unwrap_or(false);
                    if !known {
                        return Err(anyhow!(
                            "Unknown agent id {id:?}. Call `list_agents_and_models` \
                             to see the agents available for `create_thread`."
                        ));
                    }
                    Some(Agent::Custom {
                        id: project::AgentId(id.to_string().into()),
                    })
                }
            };

            let initial_content = AgentInitialContent::ContentBlock {
                blocks: vec![acp::ContentBlock::Text(acp::TextContent::new(
                    request.prompt.clone(),
                ))],
                auto_submit: true,
            };

            let title: SharedString = request.title.clone();
            let options = CreateThreadOptions {
                title: Some(title.clone()),
                initial_content: Some(initial_content),
                agent: agent_choice.clone(),
                model: request.model.clone(),
                work_dirs: None,
            };

            // If the caller asked for a fresh worktree, open a new workspace
            // backed by a linked git worktree of each git repo in the parent
            // project — the same flow the user gets when they pick "Create
            // worktree" from the worktree picker. The sibling thread is then
            // created inside the new workspace's agent panel, so it lives
            // alongside any threads the user would create there manually.
            let mut worktree_warning: Option<String> = None;
            let target_panel = if request.use_new_worktree {
                let workspace = panel.read_with(cx, |panel, _cx| panel.workspace.clone())?;
                let workspace = workspace
                    .upgrade()
                    .ok_or_else(|| anyhow!("Source workspace is no longer available"))?;
                // The branch target follows the existing UI semantics: when
                // `base_ref` is set, treat it as the ref to base off of
                // (resolved like `git switch --detach <ref>`); otherwise base
                // off the current HEAD. Either way the new worktrees are in
                // detached HEAD state — the agent can attach to a branch via
                // git afterwards.
                let branch_target = match request.base_ref.as_ref() {
                    Some(ref_name) => zed_actions::NewWorktreeBranchTarget::ExistingBranch {
                        name: ref_name.clone(),
                    },
                    None => zed_actions::NewWorktreeBranchTarget::CurrentBranch,
                };
                let action = zed_actions::CreateWorktree {
                    worktree_name: request.worktree_name.clone(),
                    branch_target,
                };
                let creation = window.update(cx, |_root, window, cx| {
                    workspace.update(cx, |workspace, cx| {
                        git_ui_core::worktree_service::create_worktree_workspace(
                            workspace, &action, window, None, cx,
                        )
                    })
                })?;
                let created = creation
                    .await
                    .context("failed to create worktree workspace")?;
                // The creation flow tells us when the project had multiple
                // worktrees of the same underlying repo, which it consolidates
                // into one new worktree — flag it so the calling agent knows
                // the result may not reflect every source worktree's state.
                if created.consolidated_worktrees {
                    worktree_warning = Some(
                        "The project contained multiple worktrees backed by the same git \
                         repository, so they were consolidated into a single new worktree. \
                         The new thread's worktree is based on one of them and may not \
                         reflect the exact state of the others."
                            .to_string(),
                    );
                }
                // Locate the agent panel on the new workspace. We rely on
                // the panel having registered by the time
                // `create_worktree_workspace` returns — `open_worktree_workspace`
                // explicitly awaits `take_panels_task` and the initial scan.
                created
                    .workspace
                    .read_with(cx, |workspace, cx| workspace.panel::<AgentPanel>(cx))
                    .ok_or_else(|| anyhow!("new workspace did not register an agent panel"))?
                    .downgrade()
            } else {
                panel.clone()
            };
            // Both the source panel and any newly-opened worktree workspace
            // live in the same OS window (the new workspace is a tab on the
            // existing MultiWorkspace), so the original window handle is
            // still the right context for the `create_thread_with_options`
            // call regardless of which panel ends up the target.
            let target_window = window;

            // We deliberately don't wait for the new thread's session to
            // become available here: there are currently no agent tools that
            // operate on sibling threads by session ID, so requiring one would
            // just introduce a race for no benefit.
            let resolved_agent_id = target_window.update(cx, |_root, window, cx| {
                target_panel.update(cx, |panel, cx| {
                    panel.create_thread_with_options(
                        options,
                        AgentThreadSource::AgentPanel,
                        window,
                        cx,
                    );
                    let resolved_agent = agent_choice
                        .clone()
                        .unwrap_or_else(|| panel.selected_agent.clone());
                    resolved_agent.id()
                })
            })??;

            Ok(agent::SiblingThreadInfo {
                title,
                agent_id: resolved_agent_id.0.to_string(),
                model: request.model,
                warning: worktree_warning,
            })
        })
    }

    fn list_available_agents(&self, cx: &mut App) -> Result<agent::AvailableAgents> {
        let panel = self
            .panel
            .upgrade()
            .ok_or_else(|| anyhow!("Agent panel is no longer available"))?;

        let mut agents = Vec::new();

        // Native Zed agent — always available, and we can enumerate models
        // directly from the language model registry.
        let native_models = {
            let registry = LanguageModelRegistry::read_global(cx);
            let default = registry.default_model();
            let mut models = Vec::new();
            for provider in registry.providers() {
                if !provider.is_authenticated(cx) {
                    continue;
                }
                let provider_id = provider.id();
                for model in provider.provided_models(cx) {
                    let id = format!("{}/{}", provider_id.0, model.id().0);
                    let is_default = default
                        .as_ref()
                        .map(|cm| cm.provider.id() == provider_id && cm.model.id() == model.id())
                        .unwrap_or(false);
                    models.push(agent::AvailableModel {
                        id,
                        name: model.name().0,
                        is_default,
                    });
                }
            }
            models
        };
        agents.push(agent::AvailableAgent {
            id: agent::ZED_AGENT_ID.to_string(),
            name: Agent::NativeAgent.label(),
            is_native: true,
            models: native_models,
        });

        let project = panel.read(cx).project.clone();
        let agent_server_store = project.read(cx).agent_server_store().clone();
        let store = agent_server_store.read(cx);
        for agent_id in store.external_agents() {
            let display = store
                .agent_display_name(agent_id)
                .unwrap_or_else(|| agent_id.0.clone());
            agents.push(agent::AvailableAgent {
                id: agent_id.0.to_string(),
                name: display,
                is_native: false,
                // External agents pick their own models dynamically; we don't
                // try to enumerate them ahead of time.
                models: Vec::new(),
            });
        }

        Ok(agent::AvailableAgents { agents })
    }
}

impl Focusable for AgentPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

fn agent_panel_dock_position(cx: &App) -> DockPosition {
    AgentSettings::get_global(cx).dock.into()
}

pub enum AgentPanelEvent {
    ActiveViewChanged,
    ActiveViewFocused,
    EntryChanged,
    TerminalCloseRequested { metadata: TerminalThreadMetadata },
    ThreadInteracted { thread_id: ThreadId },
}

impl EventEmitter<PanelEvent> for AgentPanel {}
impl EventEmitter<AgentPanelEvent> for AgentPanel {}

impl Panel for AgentPanel {
    fn restart_blocker(&self, cx: &App) -> Option<&'static str> {
        let busy = |view: &Entity<ConversationView>| view.read(cx).blocks_application_restart(cx);
        let visible = match &self.base_view {
            BaseView::AgentThread { conversation_view } => busy(conversation_view),
            _ => false,
        };
        (visible
            || self.retained_threads.values().any(busy)
            || self.terminals.values().any(|terminal| {
                terminal
                    .view
                    .read(cx)
                    .terminal()
                    .read(cx)
                    .blocks_application_restart()
            }))
        .then_some(
            "Resolve active/loading agent sessions and open agent terminals before restarting",
        )
    }

    fn persistent_name() -> &'static str {
        "AgentPanel"
    }

    fn panel_key() -> &'static str {
        AGENT_PANEL_KEY
    }

    fn activation_focus_handle(&self, cx: &App) -> FocusHandle {
        match self.visible_surface() {
            VisibleSurface::Uninitialized => self.focus_handle.clone(),
            VisibleSurface::AgentThread(conversation_view) => {
                conversation_view.read(cx).activation_focus_handle(cx)
            }
            VisibleSurface::Terminal(terminal_view) => terminal_view.focus_handle(cx),
        }
    }

    fn position(&self, _window: &Window, cx: &App) -> DockPosition {
        agent_panel_dock_position(cx)
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        position != DockPosition::Bottom
    }

    fn set_position(&mut self, position: DockPosition, _: &mut Window, cx: &mut Context<Self>) {
        let side = match position {
            DockPosition::Left => "left",
            DockPosition::Right | DockPosition::Bottom => "right",
        };
        telemetry::event!("Agent Panel Side Changed", side = side);
        settings::update_settings_file(self.fs.clone(), cx, move |settings, _| {
            settings
                .agent
                .get_or_insert_default()
                .set_dock(position.into());
        });
    }

    fn default_size(&self, window: &Window, cx: &App) -> Pixels {
        let settings = AgentSettings::get_global(cx);
        match self.position(window, cx) {
            DockPosition::Left | DockPosition::Right => settings.default_width,
            DockPosition::Bottom => settings.default_height,
        }
    }

    fn min_size(&self, window: &Window, cx: &App) -> Option<Pixels> {
        match self.position(window, cx) {
            DockPosition::Left | DockPosition::Right => Some(MIN_PANEL_WIDTH),
            DockPosition::Bottom => None,
        }
    }

    fn supports_flexible_size(&self) -> bool {
        true
    }

    fn has_flexible_size(&self, _window: &Window, cx: &App) -> bool {
        AgentSettings::get_global(cx).flexible
    }

    fn set_flexible_size(&mut self, flexible: bool, _window: &mut Window, cx: &mut Context<Self>) {
        settings::update_settings_file(self.fs.clone(), cx, move |settings, _| {
            settings
                .agent
                .get_or_insert_default()
                .set_flexible_size(flexible);
        });
    }

    fn set_active(&mut self, active: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.is_active = active;
        if active {
            self.ensure_thread_initialized(window, cx);
        }
    }

    fn remote_id() -> Option<proto::PanelId> {
        Some(proto::PanelId::AssistantPanel)
    }

    fn icon(&self, _window: &Window, cx: &App) -> Option<IconName> {
        (self.enabled(cx) && AgentSettings::get_global(cx).button).then_some(IconName::ZedAssistant)
    }

    fn icon_tooltip(&self, _window: &Window, _cx: &App) -> Option<&'static str> {
        Some("Agent Panel")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleFocus)
    }

    fn activation_priority(&self) -> u32 {
        0
    }

    fn enabled(&self, cx: &App) -> bool {
        AgentSettings::get_global(cx).enabled(cx)
    }

    fn is_agent_panel(&self) -> bool {
        true
    }

    fn hide_button_setting(&self, _: &App) -> Option<workspace::HideStatusItem> {
        Some(workspace::HideStatusItem::new(|settings| {
            settings.agent.get_or_insert_default().button = Some(false);
        }))
    }

    fn is_zoomed(&self, _window: &Window, _cx: &App) -> bool {
        self.zoomed
    }

    fn set_zoomed(&mut self, zoomed: bool, _window: &mut Window, cx: &mut Context<Self>) {
        self.zoomed = zoomed;
        cx.notify();
    }
}

impl AgentPanel {
    fn ensure_thread_initialized(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(self.base_view, BaseView::Uninitialized) {
            if self.pending_terminal_spawn.is_some() {
                return;
            }
            if self.should_create_terminal_for_new_entry(cx) {
                let terminal_id = TerminalId::new();
                self.pending_terminal_spawn = Some(terminal_id);
                cx.defer_in(window, move |this, window, cx| {
                    if matches!(this.base_view, BaseView::Uninitialized)
                        && this.pending_terminal_spawn == Some(terminal_id)
                        && this.should_create_terminal_for_new_entry(cx)
                    {
                        this.create_initial_terminal(
                            terminal_id,
                            AgentThreadSource::AgentPanel,
                            window,
                            cx,
                        );
                    } else if this.pending_terminal_spawn == Some(terminal_id) {
                        this.pending_terminal_spawn = None;
                    }
                });
            } else {
                self.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
            }
        }
    }

    fn create_initial_terminal(
        &mut self,
        terminal_id: TerminalId,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.supports_terminal(cx) {
            if self.pending_terminal_spawn == Some(terminal_id) {
                self.pending_terminal_spawn = None;
            }
            return;
        }
        let working_directory = self.terminal_working_directory(None, cx);
        self.spawn_initial_terminal(terminal_id, working_directory, source, window, cx);
    }

    #[cfg(not(test))]
    fn spawn_initial_terminal(
        &mut self,
        terminal_id: TerminalId,
        working_directory: Option<PathBuf>,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.spawn_terminal(
            terminal_id,
            working_directory,
            None,
            None,
            None,
            TerminalSpawnAssociation::Standalone,
            true,
            false,
            true,
            source,
            window,
            cx,
        )
        .detach_and_log_err(cx);
    }

    #[cfg(test)]
    fn spawn_initial_terminal(
        &mut self,
        terminal_id: TerminalId,
        working_directory: Option<PathBuf>,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Err(error) = self.insert_display_only_terminal(
            terminal_id,
            working_directory,
            None,
            None,
            None,
            true,
            false,
            true,
            source,
            window,
            cx,
        ) {
            log::error!("failed to spawn test agent panel terminal: {error:#}");
            if self.pending_terminal_spawn == Some(terminal_id) {
                self.pending_terminal_spawn = None;
                cx.notify();
            }
        }
    }

    fn destination_has_meaningful_state(&self, cx: &App) -> bool {
        if !self.retained_threads.is_empty() || !self.terminals.is_empty() {
            return true;
        }

        match &self.base_view {
            BaseView::Uninitialized => false,
            BaseView::Terminal { .. } => true,
            BaseView::AgentThread { conversation_view } => {
                let has_entries = conversation_view
                    .read(cx)
                    .root_thread_view()
                    .is_some_and(|tv| !tv.read(cx).thread.read(cx).entries().is_empty());
                if has_entries {
                    return true;
                }

                conversation_view
                    .read(cx)
                    .root_thread_view()
                    .is_some_and(|thread_view| {
                        let thread_view = thread_view.read(cx);
                        thread_view
                            .thread
                            .read(cx)
                            .draft_prompt()
                            .is_some_and(|draft| !draft.is_empty())
                            || !thread_view.message_editor.read(cx).is_empty(cx)
                    })
            }
        }
    }

    fn active_initial_content(&self, cx: &App) -> Result<Option<AgentInitialContent>> {
        let Some(thread_view) = self.active_thread_view(cx) else {
            return Ok(None);
        };
        let thread_view = thread_view.read(cx);
        let blocks = thread_view
            .message_editor
            .read(cx)
            .draft_content_blocks_snapshot(cx)?;
        if blocks.is_empty() {
            return Ok(None);
        }
        Ok(Some(AgentInitialContent::ContentBlock {
            blocks,
            auto_submit: false,
        }))
    }

    fn source_panel_initialization(
        source_workspace: &WeakEntity<Workspace>,
        cx: &App,
    ) -> Result<Option<SourcePanelInitialization>> {
        let Some(source_workspace) = source_workspace.upgrade() else {
            return Ok(None);
        };
        let Some(source_panel) = source_workspace.read(cx).panel::<AgentPanel>(cx) else {
            return Ok(None);
        };
        let source_panel = source_panel.read(cx);
        Ok(Some(SourcePanelInitialization {
            agent: source_panel.selected_agent(cx),
            initial_content: source_panel.active_initial_content(cx)?,
        }))
    }

    pub fn initialize_from_source_workspace_if_needed(
        &mut self,
        source_workspace: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if !self.has_open_project(cx) {
            return false;
        }

        if self.destination_has_meaningful_state(cx) {
            return false;
        }

        let initialization = match Self::source_panel_initialization(&source_workspace, cx) {
            Ok(Some(initialization)) => initialization,
            Ok(None) => return false,
            Err(error) => {
                log::error!("Cannot transfer the current unsent draft: {error:#}");
                if let Some(workspace) = self.workspace.upgrade() {
                    workspace.update(cx, |workspace, cx| workspace.show_error(error, cx));
                }
                return false;
            }
        };

        let mut initialized = false;
        if self.selected_agent != initialization.agent {
            self.selected_agent = initialization.agent.clone();
            self.serialize(cx);
            initialized = true;
        }

        if let Some(initial_content) = initialization.initial_content {
            let thread = self.create_agent_thread_with_server(
                initialization.agent,
                None,
                None,
                None,
                None,
                Some(initial_content),
                None,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
            self.draft_thread = Some(thread.conversation_view.clone());
            self.observe_draft_editor(&thread.conversation_view, cx);
            self.set_base_view(thread.into(), false, window, cx);
            true
        } else {
            if initialized
                && matches!(
                    &self.base_view,
                    BaseView::AgentThread { conversation_view }
                        if self.draft_thread.as_ref().is_some_and(|draft| {
                            draft.entity_id() == conversation_view.entity_id()
                        })
                )
            {
                self.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
            } else if initialized {
                cx.notify();
            }
            initialized
        }
    }

    fn is_title_editor_focused(&self, window: &Window, cx: &Context<Self>) -> bool {
        match self.visible_surface() {
            VisibleSurface::AgentThread(conversation_view) => conversation_view
                .read(cx)
                .root_thread_view()
                .is_some_and(|view| view.read(cx).title_editor.read(cx).is_focused(window)),
            VisibleSurface::Terminal(_) => self
                .active_terminal_id()
                .and_then(|id| self.terminals.get(&id))
                .and_then(|terminal| terminal.title_editor.as_ref())
                .is_some_and(|editor| editor.read(cx).is_focused(window)),
            _ => false,
        }
    }

    fn should_show_title_edit(&self, window: &Window, cx: &Context<Self>) -> bool {
        matches!(
            self.visible_surface(),
            VisibleSurface::AgentThread(_) | VisibleSurface::Terminal(_)
        ) && self.has_open_project(cx)
            && !self.is_title_editor_focused(window, cx)
    }

    fn render_title_view(&self, window: &mut Window, cx: &Context<Self>) -> AnyElement {
        let content = match self.visible_surface() {
            VisibleSurface::AgentThread(conversation_view) => {
                let server_view_ref = conversation_view.read(cx);
                let native_thread = server_view_ref.as_native_thread(cx);
                let is_generating_title = native_thread
                    .as_ref()
                    .is_some_and(|thread| thread.read(cx).is_generating_title());
                let title_generation_error = native_thread
                    .as_ref()
                    .and_then(|thread| thread.read(cx).title_generation_error());

                if let Some(title_editor) = server_view_ref
                    .root_thread_view()
                    .map(|r| r.read(cx).title_editor.clone())
                {
                    if is_generating_title {
                        Label::new(server_view_ref.title(cx))
                            .color(Color::Muted)
                            .truncate()
                            .with_animation(
                                "generating_title",
                                Animation::new(Duration::from_secs(2))
                                    .repeat()
                                    .with_easing(pulsating_between(0.4, 0.8)),
                                |label, delta| label.alpha(delta),
                            )
                            .into_any_element()
                    } else {
                        let editable_title = div()
                            .flex_1()
                            .on_action({
                                let conversation_view = conversation_view.downgrade();
                                move |_: &menu::Confirm, window, cx| {
                                    if let Some(conversation_view) = conversation_view.upgrade() {
                                        conversation_view
                                            .read(cx)
                                            .activation_focus_handle(cx)
                                            .focus(window, cx);
                                    }
                                }
                            })
                            .on_action({
                                let conversation_view = conversation_view.downgrade();
                                move |_: &editor::actions::Cancel, window, cx| {
                                    if let Some(conversation_view) = conversation_view.upgrade() {
                                        conversation_view
                                            .read(cx)
                                            .activation_focus_handle(cx)
                                            .focus(window, cx);
                                    }
                                }
                            })
                            .child(title_editor);

                        if let Some(title_generation_error) = title_generation_error {
                            h_flex()
                                .w_full()
                                .gap_1()
                                .child(editable_title)
                                .child(
                                    IconButton::new("retry-thread-title", IconName::XCircle)
                                        .icon_color(Color::Error)
                                        .icon_size(IconSize::Small)
                                        .tooltip(move |_window, cx| {
                                            Tooltip::with_meta(
                                                "Title generation failed. Click to retry.",
                                                None,
                                                title_generation_error.clone(),
                                                cx,
                                            )
                                        })
                                        .on_click({
                                            let conversation_view = conversation_view.clone();
                                            let workspace = self.workspace.clone();
                                            move |_event, _window, cx| {
                                                Self::handle_regenerate_thread_title(
                                                    conversation_view.clone(),
                                                    workspace.clone(),
                                                    cx,
                                                );
                                            }
                                        }),
                                )
                                .into_any_element()
                        } else {
                            editable_title.w_full().into_any_element()
                        }
                    }
                } else {
                    Label::new(conversation_view.read(cx).title(cx))
                        .color(Color::Muted)
                        .truncate()
                        .into_any_element()
                }
            }
            VisibleSurface::Terminal(_) => {
                if let Some((terminal_id, title_editor, title)) =
                    self.active_terminal_id().and_then(|terminal_id| {
                        self.terminals.get(&terminal_id).map(|terminal| {
                            (
                                terminal_id,
                                terminal.title_editor.clone(),
                                terminal.title(cx),
                            )
                        })
                    })
                {
                    if let Some(title_editor) = title_editor {
                        div()
                            .flex_1()
                            .on_action(cx.listener(move |this, _: &menu::Confirm, window, cx| {
                                this.stop_editing_terminal_title(terminal_id, true, window, cx);
                            }))
                            .on_action(cx.listener(
                                move |this, _: &editor::actions::Cancel, window, cx| {
                                    this.stop_editing_terminal_title(terminal_id, true, window, cx);
                                },
                            ))
                            .child(title_editor)
                            .into_any_element()
                    } else {
                        div()
                            .id("terminal-title")
                            .flex_1()
                            .cursor_text()
                            .overflow_x_scroll()
                            .child(Label::new(title).color(Color::Muted).single_line())
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.edit_terminal_title(terminal_id, window, cx);
                            }))
                            .into_any_element()
                    }
                } else {
                    Label::new("Terminal").into_any_element()
                }
            }

            VisibleSurface::Uninitialized => Label::new("Agent").truncate().into_any_element(),
        };

        let toolbar_bg = cx.theme().colors().tab_bar_background;
        let gradient_overlay = GradientFade::new(toolbar_bg, toolbar_bg, toolbar_bg)
            .width(px(64.0))
            .right(px(0.0))
            .gradient_stop(0.75);
        // The fade gradient renders as a visible patch on transparent windows
        // (the title already truncates).
        let opaque_window =
            cx.theme().window_background_appearance() == gpui::WindowBackgroundAppearance::Opaque;

        h_flex()
            .key_context("TitleEditor")
            .group("title_editor")
            .flex_grow_1()
            .w_full()
            .min_w_0()
            .max_w_full()
            .overflow_x_hidden()
            .child(content)
            .when(self.should_show_title_edit(window, cx), |this| {
                this.when(opaque_window, |this| this.child(gradient_overlay))
                    .child(
                        h_flex()
                            .visible_on_hover("title_editor")
                            .absolute()
                            .right_0()
                            .h_full()
                            .bg(cx.theme().colors().tab_bar_background)
                            .child(
                                IconButton::new("edit_tile", IconName::Pencil)
                                    .icon_size(IconSize::Small)
                                    .tooltip(Tooltip::text("Edit Thread Title")),
                            ),
                    )
            })
            .into_any()
    }

    fn show_no_thread_summary_model_toast(workspace: Entity<Workspace>, cx: &mut App) {
        workspace.update(cx, |workspace, cx| {
            let toast = StatusToast::new(
                "No model is configured for summarizing thread titles.",
                cx,
                |this, _cx| {
                    this.icon(
                        Icon::new(IconName::Warning)
                            .size(IconSize::Small)
                            .color(Color::Warning),
                    )
                    .dismiss_button(true)
                },
            );
            workspace.toggle_status_toast(toast, cx);
        });
    }

    fn handle_regenerate_thread_title(
        conversation_view: Entity<ConversationView>,
        workspace: WeakEntity<Workspace>,
        cx: &mut App,
    ) {
        match Self::regenerate_conversation_thread_title(conversation_view, cx) {
            ThreadTitleRegenerationResult::NoModel => {
                if let Some(workspace) = workspace.upgrade() {
                    Self::show_no_thread_summary_model_toast(workspace, cx);
                }
            }
            ThreadTitleRegenerationResult::NotOpen
            | ThreadTitleRegenerationResult::Started
            | ThreadTitleRegenerationResult::AlreadyGenerating => {}
        }
    }

    fn render_panel_options_menu(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let focus_handle = self.focus_handle(cx);
        // Resolve menu shortcuts at the thread root; the active editor can
        // shadow panel-level commands such as ManageSkills.
        let menu_action_context = match &self.base_view {
            BaseView::AgentThread { conversation_view } => conversation_view
                .read(cx)
                .active_thread()
                .map(|thread| thread.read(cx).focus_handle.clone())
                .unwrap_or_else(|| focus_handle.clone()),
            _ => focus_handle.clone(),
        };
        let showing_terminal = matches!(self.visible_surface(), VisibleSurface::Terminal(_));

        let conversation_view = match &self.base_view {
            BaseView::AgentThread { conversation_view } => Some(conversation_view.clone()),
            _ => None,
        };

        let can_regenerate_thread_title =
            conversation_view.as_ref().is_some_and(|conversation_view| {
                let conversation_view = conversation_view.read(cx);
                conversation_view.has_user_submitted_prompt(cx)
                    && conversation_view
                        .as_native_thread(cx)
                        .is_some_and(|thread| !thread.read(cx).is_generating_title())
            });

        let has_thread_messages = conversation_view.as_ref().is_some_and(|conversation_view| {
            conversation_view.read(cx).has_user_submitted_prompt(cx)
        });

        let has_auth_methods = match &self.base_view {
            BaseView::AgentThread { conversation_view } => {
                conversation_view.read(cx).has_auth_methods()
            }
            _ => false,
        };
        let supports_logout = self
            .active_conversation_view()
            .is_some_and(|conversation_view| conversation_view.read(cx).supports_logout());

        let project_agents_md_path = project_agents_md_path(&self.project, true, cx);

        let global_agents_md_loaded = UserAgentsMd::global(cx)
            .and_then(|md| md.content())
            .is_some();

        let workspace = self.workspace.clone();

        PopoverMenu::new("agent-options-menu")
            .trigger_with_tooltip(
                IconButton::new("agent-options-menu", IconName::Ellipsis)
                    .icon_size(IconSize::Small),
                move |_window, cx| {
                    Tooltip::for_action_in(
                        "Toggle Agent Menu",
                        &ToggleOptionsMenu,
                        &focus_handle,
                        cx,
                    )
                },
            )
            .anchor(Anchor::TopRight)
            .with_handle(self.agent_panel_menu_handle.clone())
            .menu({
                move |window, cx| {
                    Some(ContextMenu::build(window, cx, |mut menu, _window, cx| {
                        menu = menu.context(menu_action_context.clone());

                        if has_thread_messages {
                            menu = menu.header("Current Thread");

                            if let Some(conversation_view) = conversation_view.as_ref() {
                                if can_regenerate_thread_title {
                                    menu = menu.entry("Regenerate Thread Title", None, {
                                        let conversation_view = conversation_view.clone();
                                        let workspace = workspace.clone();
                                        move |_, cx| {
                                            Self::handle_regenerate_thread_title(
                                                conversation_view.clone(),
                                                workspace.clone(),
                                                cx,
                                            );
                                        }
                                    });
                                }

                                let root_thread_view =
                                    conversation_view.read(cx).root_thread_view();
                                if let Some(thread_view) = root_thread_view {
                                    let workspace = workspace.clone();
                                    menu = menu.entry("Open Thread as Markdown", None, {
                                        move |window, cx| {
                                            if let Some(workspace) = workspace.upgrade() {
                                                thread_view.update(cx, |thread_view, cx| {
                                                    thread_view
                                                        .open_thread_as_markdown(
                                                            workspace, window, cx,
                                                        )
                                                        .detach_and_log_err(cx);
                                                });
                                            }
                                        }
                                    });
                                }

                                menu = menu.separator();
                            }
                        }

                        if !showing_terminal {
                            menu = menu
                                .header("MCP Servers")
                                .action(
                                    "Add Server…",
                                    Box::new(zed_actions::OpenSettingsAt {
                                        path: "context_servers".to_string(),
                                        target: None,
                                    }),
                                )
                                .action(
                                    "Install New Servers…",
                                    Box::new(zed_actions::Extensions {
                                        category_filter: Some(
                                            zed_actions::ExtensionCategoryFilter::ContextServers,
                                        ),
                                        id: None,
                                    }),
                                )
                                .separator()
                                .header("Context")
                                .action("Skills", Box::new(ManageSkills))
                                .entry("Recipes and Schedules", None, {
                                    let workspace = workspace.clone();
                                    move |window, cx| {
                                        workspace
                                            .update(cx, |workspace, cx| {
                                                if let Some(panel) =
                                                    workspace.panel::<AgentPanel>(cx)
                                                {
                                                    panel.update(cx, |panel, cx| {
                                                        panel.manage_recipes(None, window, cx)
                                                    });
                                                }
                                            })
                                            .log_err();
                                    }
                                });

                            if project_agents_md_path.is_some() || global_agents_md_loaded {
                                if global_agents_md_loaded {
                                    let workspace = workspace.clone();

                                    menu = menu.custom_entry(
                                        |_window, _cx| {
                                            h_flex()
                                                .w_full()
                                                .gap_1()
                                                .child(Label::new("Open Global Rules"))
                                                .child(
                                                    Label::new("(AGENTS.md)")
                                                        .color(Color::Muted)
                                                        .size(LabelSize::Small),
                                                )
                                                .into_any_element()
                                        },
                                        move |window, cx| {
                                            workspace
                                                .update(cx, |workspace, cx| {
                                                    open_global_rules(workspace, window, cx);
                                                })
                                                .log_err();
                                        },
                                    );
                                }

                                if project_agents_md_path.is_some() {
                                    let workspace = workspace.clone();
                                    menu = menu.custom_entry(
                                        |_window, _cx| {
                                            h_flex()
                                                .w_full()
                                                .gap_1()
                                                .child(Label::new("Open Project Rules"))
                                                .child(
                                                    Label::new("(AGENTS.md)")
                                                        .color(Color::Muted)
                                                        .size(LabelSize::Small),
                                                )
                                                .into_any_element()
                                        },
                                        move |window, cx| {
                                            workspace
                                                .update(cx, |workspace, cx| {
                                                    open_project_rules(workspace, window, cx);
                                                })
                                                .log_err();
                                        },
                                    );
                                }
                            }

                            menu = menu
                                .separator()
                                .header("MCP Servers")
                                .action(
                                    "Add Server…",
                                    Box::new(zed_actions::OpenSettingsAt {
                                        path: "context_servers".to_string(),
                                        target: None,
                                    }),
                                )
                                .action(
                                    "Install New Servers…",
                                    Box::new(zed_actions::Extensions {
                                        category_filter: Some(
                                            zed_actions::ExtensionCategoryFilter::ContextServers,
                                        ),
                                        id: None,
                                    }),
                                )
                                .separator()
                                .action("Profiles", Box::new(ManageProfiles::default()));
                        }

                        menu = menu
                            .action("Settings", Box::new(OpenSettings))
                            .separator()
                            .action("Toggle Threads Sidebar", Box::new(ToggleWorkspaceSidebar));

                        if has_auth_methods || supports_logout {
                            menu = menu.separator()
                        }
                        if has_auth_methods {
                            menu = menu.action("Reauthenticate", Box::new(ReauthenticateAgent))
                        }
                        if supports_logout {
                            menu = menu.action("Log Out", Box::new(LogoutAgent))
                        }

                        menu
                    }))
                }
            })
    }

    fn render_no_project_state(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let focus_handle = self.focus_handle(cx);

        ProjectEmptyState::new(
            "Agent Panel",
            focus_handle.clone(),
            KeyBinding::for_action_in(&workspace::Open::default(), &focus_handle, cx),
        )
        .on_open_project(|_, window, cx| {
            telemetry::event!("Agent Panel Add Project Clicked");
            window.dispatch_action(workspace::Open::default().boxed_clone(), cx);
        })
        .on_clone_repo(|_, window, cx| {
            telemetry::event!("Agent Panel Clone Repo Clicked");
            window.dispatch_action(git::Clone.boxed_clone(), cx);
        })
    }

    fn render_toolbar(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let agent_server_store = self.project.read(cx).agent_server_store().clone();

        let focus_handle = self.focus_handle(cx);

        let can_create_entries = self.has_open_project(cx);
        let supports_terminal = self.supports_terminal(cx);
        let showing_terminal = matches!(self.visible_surface(), VisibleSurface::Terminal(_));

        let (selected_agent_custom_icon, selected_agent_label) = if showing_terminal {
            (None, SharedString::from("Terminal"))
        } else if let Agent::Custom { id, .. } = &self.selected_agent {
            let store = agent_server_store.read(cx);
            let icon = store.agent_icon(&id);

            let label = store
                .agent_display_name(&id)
                .unwrap_or_else(|| self.selected_agent.label());
            (icon, label)
        } else {
            (None, self.selected_agent.label())
        };

        let new_thread_menu_builder: Rc<
            dyn Fn(&mut Window, &mut App) -> Option<Entity<ContextMenu>>,
        > = {
            let selected_agent = self.selected_agent.clone();
            let is_agent_selected = move |agent: Agent| selected_agent == agent;

            let workspace = self.workspace.clone();
            let is_via_collab = workspace
                .update(cx, |workspace, cx| {
                    workspace.project().read(cx).is_via_collab()
                })
                .unwrap_or_default();

            let focus_handle = focus_handle.clone();
            let agent_server_store = agent_server_store;

            Rc::new(move |window, cx| {
                Some(ContextMenu::build(window, cx, |menu, _window, cx| {
                    menu.context(focus_handle.clone())
                        .item(
                            ContextMenuEntry::new(agent::native_agent_name())
                                .when(
                                    !showing_terminal && is_agent_selected(Agent::NativeAgent),
                                    |this| this.action(Box::new(NewThread)),
                                )
                                .icon(IconName::ZedAgent)
                                .icon_color(Color::Muted)
                                .handler({
                                    let workspace = workspace.clone();
                                    move |window, cx| {
                                        if let Some(workspace) = workspace.upgrade() {
                                            workspace.update(cx, |workspace, cx| {
                                                if let Some(panel) =
                                                    workspace.panel::<AgentPanel>(cx)
                                                {
                                                    panel.update(cx, |panel, cx| {
                                                        panel.selected_agent = Agent::NativeAgent;
                                                        panel.activate_new_thread(
                                                            true,
                                                            AgentThreadSource::AgentPanel,
                                                            window,
                                                            cx,
                                                        );
                                                    });
                                                }
                                            });
                                        }
                                    }
                                }),
                        )
                        .when(supports_terminal, |menu| {
                            menu.item(
                                ContextMenuEntry::new("Terminal")
                                    .when(showing_terminal, |this| this.action(Box::new(NewThread)))
                                    .when(!showing_terminal, |this| {
                                        this.action(Box::new(NewTerminalThread))
                                    })
                                    .icon(IconName::Terminal)
                                    .icon_color(Color::Muted)
                                    .handler({
                                        let workspace = workspace.clone();
                                        move |window, cx| {
                                            if let Some(workspace) = workspace.upgrade() {
                                                workspace.update(cx, |workspace, cx| {
                                                    if let Some(panel) =
                                                        workspace.panel::<AgentPanel>(cx)
                                                    {
                                                        panel.update(cx, |panel, cx| {
                                                            panel.new_terminal(
                                                                Some(workspace),
                                                                AgentThreadSource::AgentPanel,
                                                                window,
                                                                cx,
                                                            );
                                                        });
                                                    }
                                                });
                                            }
                                        }
                                    }),
                            )
                        })
                        .map(|mut menu| {
                            let agent_server_store = agent_server_store.read(cx);
                            let registry_store = project::AgentRegistryStore::try_global(cx);
                            let registry_store_ref = registry_store.as_ref().map(|s| s.read(cx));

                            struct AgentMenuItem {
                                id: AgentId,
                                display_name: SharedString,
                            }

                            let agent_items = agent_server_store
                                .external_agents()
                                .map(|agent_id| {
                                    let display_name = agent_server_store
                                        .agent_display_name(agent_id)
                                        .or_else(|| {
                                            registry_store_ref
                                                .as_ref()
                                                .and_then(|store| store.agent(agent_id))
                                                .map(|a| a.name().clone())
                                        })
                                        .unwrap_or_else(|| agent_id.0.clone());
                                    AgentMenuItem {
                                        id: agent_id.clone(),
                                        display_name,
                                    }
                                })
                                .sorted_unstable_by_key(|e| e.display_name.to_lowercase())
                                .collect::<Vec<_>>();

                            if !agent_items.is_empty() {
                                menu = menu.separator().header("External Agents");
                            }
                            for item in &agent_items {
                                let mut entry = ContextMenuEntry::new(item.display_name.clone());

                                let icon_path =
                                    agent_server_store.agent_icon(&item.id).or_else(|| {
                                        registry_store_ref
                                            .as_ref()
                                            .and_then(|store| store.agent(&item.id))
                                            .and_then(|a| a.icon_path().cloned())
                                    });

                                if let Some(icon_path) = icon_path {
                                    entry = entry.custom_icon_svg(icon_path);
                                } else {
                                    entry = entry.icon(IconName::Sparkle);
                                }

                                entry = entry
                                    .when(
                                        !showing_terminal
                                            && is_agent_selected(Agent::Custom {
                                                id: item.id.clone(),
                                            }),
                                        |this| this.action(Box::new(NewThread)),
                                    )
                                    .icon_color(Color::Muted)
                                    .disabled(is_via_collab)
                                    .handler({
                                        let workspace = workspace.clone();
                                        let agent_id = item.id.clone();
                                        move |window, cx| {
                                            if let Some(workspace) = workspace.upgrade() {
                                                workspace.update(cx, |workspace, cx| {
                                                    if let Some(panel) =
                                                        workspace.panel::<AgentPanel>(cx)
                                                    {
                                                        panel.update(cx, |panel, cx| {
                                                            panel.new_external_agent_thread(
                                                                &NewExternalAgentThread {
                                                                    agent: agent_id.clone(),
                                                                },
                                                                window,
                                                                cx,
                                                            );
                                                        });
                                                    }
                                                });
                                            }
                                        }
                                    });

                                menu = menu.item(entry);
                            }

                            menu
                        })
                        .separator()
                        .item(
                            ContextMenuEntry::new("Add More Agents")
                                .icon(IconName::Plus)
                                .icon_color(Color::Muted)
                                .handler({
                                    move |window, cx| {
                                        window
                                            .dispatch_action(Box::new(zed_actions::AcpRegistry), cx)
                                    }
                                }),
                        )
                }))
            })
        };

        let is_thread_loading = self
            .active_conversation_view()
            .map(|thread| thread.read(cx).is_loading())
            .unwrap_or(false);

        let has_custom_icon = selected_agent_custom_icon.is_some();
        let selected_agent_builtin_icon = if showing_terminal {
            Some(IconName::Terminal)
        } else {
            self.selected_agent.icon()
        };
        let selected_agent_label_for_tooltip = selected_agent_label.clone();

        let selected_agent = div()
            .id("selected_agent_icon")
            .px_0p5()
            .when_some(selected_agent_custom_icon, |this, icon_path| {
                this.child(
                    Icon::from_external_svg(icon_path)
                        .color(Color::Muted)
                        .size(IconSize::Small),
                )
            })
            .when(!has_custom_icon, |this| {
                this.when_some(selected_agent_builtin_icon, |this, icon| {
                    this.child(Icon::new(icon).color(Color::Muted))
                })
            })
            .tooltip(move |_, cx| {
                Tooltip::with_meta(
                    selected_agent_label_for_tooltip.clone(),
                    None,
                    "Selected Agent",
                    cx,
                )
            });

        let selected_agent = if is_thread_loading {
            selected_agent
                .with_animation(
                    "pulsating-icon",
                    Animation::new(Duration::from_secs(1))
                        .repeat()
                        .with_easing(pulsating_between(0.2, 0.6)),
                    |icon, delta| icon.opacity(delta),
                )
                .into_any_element()
        } else {
            selected_agent.into_any_element()
        };

        enum ToolbarMode {
            Terminal,
            EmptyThread,
            ActiveThread,
        }

        let mode = if matches!(self.base_view, BaseView::Terminal { .. }) {
            ToolbarMode::Terminal
        } else if self.active_thread_has_messages(cx) {
            ToolbarMode::ActiveThread
        } else {
            ToolbarMode::EmptyThread
        };

        let is_full_screen = self.is_zoomed(window, cx);
        let (icon_name, tooltip_text) = if is_full_screen {
            (IconName::Minimize, "Disable Full Screen")
        } else {
            (IconName::Maximize, "Enable Full Screen")
        };
        let full_screen_button = IconButton::new("toggle-full-screen", icon_name)
            .icon_size(IconSize::Small)
            .toggle_state(is_full_screen)
            .tooltip(move |_, cx| Tooltip::for_action(tooltip_text, &ToggleZoom, cx))
            .on_click(cx.listener(move |this, _, window, cx| {
                this.toggle_zoom(&ToggleZoom, window, cx);
            }));

        let max_content_width = AgentSettings::get_global(cx).max_content_width;

        let base_container = h_flex()
            .size_full()
            .when(
                matches!(mode, ToolbarMode::EmptyThread | ToolbarMode::ActiveThread),
                |this| this.when_some(max_content_width, |this, max_w| this.max_w(max_w).mx_auto()),
            )
            .flex_none()
            .justify_between();

        let empty_thread_title = matches!(mode, ToolbarMode::EmptyThread).then(|| {
            Label::new(format!("New {} Thread", selected_agent_label))
                .color(Color::Muted)
                .truncate()
                .into_any_element()
        });

        let toolbar_content = {
            let new_thread_menu = PopoverMenu::new("new_thread_menu")
                .trigger_with_tooltip(
                    IconButton::new("new_thread_menu_btn", IconName::Plus)
                        .icon_size(IconSize::Small),
                    {
                        move |_window, cx| {
                            Tooltip::for_action_in(
                                "New Thread\u{2026}",
                                &ToggleNewThreadMenu,
                                &focus_handle,
                                cx,
                            )
                        }
                    },
                )
                .anchor(Anchor::TopRight)
                .with_handle(self.new_thread_menu_handle.clone())
                .menu(move |window, cx| new_thread_menu_builder(window, cx));

            let sandbox_status = self
                .active_conversation_view()
                .and_then(|conversation_view| conversation_view.read(cx).root_thread_view())
                .and_then(|thread_view| {
                    thread_view.update(cx, |thread_view, cx| thread_view.render_sandbox_status(cx))
                });

            base_container
                .child(
                    h_flex()
                        .relative()
                        .h_full()
                        .flex_1()
                        .min_w_0()
                        .overflow_hidden()
                        .gap(DynamicSpacing::Base04.rems(cx))
                        .pl(DynamicSpacing::Base04.rems(cx))
                        .child(selected_agent.into_any_element())
                        .child(match empty_thread_title {
                            Some(title) => title,
                            None => self.render_title_view(window, cx),
                        }),
                )
                .child(
                    h_flex()
                        .px_1()
                        .h_full()
                        .flex_none()
                        .gap_1()
                        .children(sandbox_status)
                        .when(can_create_entries, |this| this.child(new_thread_menu))
                        .child(full_screen_button)
                        .child(self.render_panel_options_menu(window, cx)),
                )
                .into_any_element()
        };

        h_flex()
            .id("agent-panel-toolbar")
            .h(Tab::container_height(cx))
            .flex_shrink_0()
            .max_w_full()
            .bg(cx.theme().colors().tab_bar_background)
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(toolbar_content)
    }

    fn should_render_trial_end_upsell(&self, cx: &mut Context<Self>) -> bool {
        if TrialEndUpsell::dismissed(cx) {
            return false;
        }

        match &self.base_view {
            BaseView::AgentThread { .. } => {
                if LanguageModelRegistry::global(cx)
                    .read(cx)
                    .default_model()
                    .is_some_and(|model| {
                        model.provider.id() != language_model::ZED_CLOUD_PROVIDER_ID
                    })
                {
                    return false;
                }
            }
            BaseView::Terminal { .. } | BaseView::Uninitialized => {
                return false;
            }
        }

        let plan = self.user_store.read(cx).plan();
        let has_previous_trial = self.user_store.read(cx).trial_started_at().is_some();

        plan.is_some_and(|plan| plan == Plan::ZedFree) && has_previous_trial
    }

    fn dismiss_ai_onboarding(&mut self, cx: &mut Context<Self>) {
        self.new_user_onboarding_upsell_dismissed
            .store(true, Ordering::Release);
        OnboardingUpsell::set_dismissed(true, cx);
        cx.notify();
    }

    fn should_render_new_user_onboarding(&mut self, cx: &mut Context<Self>) -> bool {
        if self
            .new_user_onboarding_upsell_dismissed
            .load(Ordering::Acquire)
        {
            return false;
        }

        let user_store = self.user_store.read(cx);

        if user_store.plan().is_some_and(|plan| plan == Plan::ZedPro)
            && user_store
                .subscription_period()
                .and_then(|period| period.0.checked_add_days(chrono::Days::new(1)))
                .is_some_and(|date| date < chrono::Utc::now())
        {
            if !self
                .new_user_onboarding_upsell_dismissed
                .load(Ordering::Acquire)
            {
                self.dismiss_ai_onboarding(cx);
            }
            return false;
        }

        let has_configured_non_zed_providers = LanguageModelRegistry::read_global(cx)
            .visible_providers()
            .iter()
            .any(|provider| {
                provider.is_authenticated(cx)
                    && provider.id() != language_model::ZED_CLOUD_PROVIDER_ID
            });

        match &self.base_view {
            BaseView::Uninitialized | BaseView::Terminal { .. } => false,
            BaseView::AgentThread { conversation_view } => {
                if conversation_view.read(cx).as_native_thread(cx).is_some() {
                    let history_is_empty = ThreadStore::global(cx).read(cx).is_empty();
                    history_is_empty || !has_configured_non_zed_providers
                } else {
                    false
                }
            }
        }
    }

    fn render_new_user_onboarding(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        if !self.should_render_new_user_onboarding(cx) {
            return None;
        }

        Some(
            div()
                .id("agent-onboarding-scroll")
                .max_h(relative(0.6))
                .flex_shrink_0()
                .overflow_y_scroll()
                .bg(cx.theme().colors().editor_background)
                .child(self.new_user_onboarding.clone()),
        )
    }

    fn render_trial_end_upsell(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        if !self.should_render_trial_end_upsell(cx) {
            return None;
        }

        Some(
            v_flex()
                .absolute()
                .inset_0()
                .size_full()
                .bg(cx.theme().colors().panel_background)
                .opacity(0.85)
                .block_mouse_except_scroll()
                .child(EndTrialUpsell::new(Arc::new({
                    let this = cx.entity();
                    move |_, cx| {
                        this.update(cx, |_this, cx| {
                            TrialEndUpsell::set_dismissed(true, cx);
                            cx.notify();
                        });
                    }
                }))),
        )
    }

    fn render_drag_target(&self, cx: &Context<Self>) -> Div {
        let is_local = self.project.read(cx).is_local();
        div()
            .invisible()
            .absolute()
            .top_0()
            .right_0()
            .bottom_0()
            .left_0()
            .bg(cx.theme().colors().drop_target_background)
            .drag_over::<DraggedTab>(|this, _, _, _| this.visible())
            .drag_over::<DraggedSelection>(|this, _, _, _| this.visible())
            .when(is_local, |this| {
                this.drag_over::<ExternalPaths>(|this, _, _, _| this.visible())
            })
            .on_drop(cx.listener(move |this, tab: &DraggedTab, window, cx| {
                let item = tab.pane.read(cx).item_for_index(tab.ix);
                let project_paths = item
                    .and_then(|item| item.project_path(cx))
                    .into_iter()
                    .collect::<Vec<_>>();
                this.handle_drop(project_paths, vec![], window, cx);
            }))
            .on_drop(
                cx.listener(move |this, selection: &DraggedSelection, window, cx| {
                    let project_paths = selection
                        .items()
                        .filter_map(|item| this.project.read(cx).path_for_entry(item.entry_id, cx))
                        .collect::<Vec<_>>();
                    this.handle_drop(project_paths, vec![], window, cx);
                }),
            )
            .on_drop(cx.listener(move |this, paths: &ExternalPaths, window, cx| {
                this.handle_external_paths_drop(paths, window, cx);
            }))
    }

    fn handle_external_paths_drop(
        &mut self,
        paths: &ExternalPaths,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if matches!(&self.base_view, BaseView::Terminal { .. }) {
            // Terminal drops should match normal terminal views by pasting raw OS paths.
            // The agent-thread path below converts paths to project paths, which can add
            // worktrees and is only needed when attaching files to a conversation.
            self.paste_external_paths_into_active_terminal(paths, window, cx);
            return;
        }

        let BaseView::AgentThread { conversation_view } = &self.base_view else {
            return;
        };
        let conversation_view = conversation_view.clone();
        let tasks = paths
            .paths()
            .iter()
            .map(|path| Workspace::project_path_for_path(self.project.clone(), path, false, cx))
            .collect::<Vec<_>>();
        cx.spawn_in(window, async move |_this, cx| {
            let mut paths = vec![];
            let mut added_worktrees = vec![];
            let opened_paths = futures::future::join_all(tasks).await;
            for entry in opened_paths {
                if let Some((worktree, project_path)) = entry.log_err() {
                    added_worktrees.push(worktree);
                    paths.push(project_path);
                }
            }
            conversation_view
                .update_in(cx, |conversation_view, window, cx| {
                    conversation_view.insert_dragged_files(paths, added_worktrees, window, cx);
                })
                .log_err();
        })
        .detach();
    }

    fn paste_external_paths_into_active_terminal(
        &mut self,
        paths: &ExternalPaths,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let BaseView::Terminal { terminal_id } = &self.base_view else {
            return;
        };

        if !self.project.read(cx).is_local() {
            return;
        }

        let Some(terminal_view) = self
            .terminals
            .get(terminal_id)
            .map(|terminal| terminal.view.clone())
        else {
            return;
        };

        terminal_view.update(cx, |terminal_view, cx| {
            terminal_view.add_paths_to_terminal(paths.paths(), window, cx);
        });
    }

    fn handle_drop(
        &mut self,
        paths: Vec<ProjectPath>,
        added_worktrees: Vec<Entity<Worktree>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match &self.base_view {
            BaseView::AgentThread { conversation_view } => {
                conversation_view.update(cx, |conversation_view, cx| {
                    conversation_view.insert_dragged_files(paths, added_worktrees, window, cx);
                });
            }
            BaseView::Terminal { terminal_id } => {
                let paths = {
                    let project = self.project.read(cx);
                    paths
                        .iter()
                        .filter_map(|project_path| project.absolute_path(project_path, cx))
                        .collect::<Vec<_>>()
                };

                if paths.is_empty() {
                    return;
                }

                if let Some(terminal_view) = self
                    .terminals
                    .get(terminal_id)
                    .map(|terminal| terminal.view.clone())
                {
                    terminal_view.update(cx, |terminal_view, cx| {
                        terminal_view.add_paths_to_terminal(&paths, window, cx);
                    });
                }
            }
            BaseView::Uninitialized => {}
        }
    }

    fn key_context(&self) -> KeyContext {
        let mut key_context = KeyContext::new_with_defaults();
        key_context.add("AgentPanel");
        key_context
    }
}

impl Render for AgentPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // WARNING: Changes to this element hierarchy can have
        // non-obvious implications to the layout of children.
        //
        // If you need to change it, please confirm:
        // - The message editor expands (cmd-option-esc) correctly
        // - When expanded, the buttons at the bottom of the panel are displayed correctly
        // - Font size works as expected and can be changed with cmd-+/cmd-
        // - Scrolling in all views works as expected
        // - Files can be dropped into the panel
        let content = v_flex()
            .key_context(self.key_context())
            .relative()
            .size_full()
            .justify_between()
            .track_focus(&self.focus_handle)
            .bg(cx.theme().colors().panel_background)
            .on_action(cx.listener(|this, action: &NewThread, window, cx| {
                this.new_thread(action, window, cx);
            }))
            .on_action(cx.listener(|this, _: &NewTerminalThread, window, cx| {
                cx.stop_propagation();
                this.new_terminal(None, AgentThreadSource::AgentPanel, window, cx);
            }))
            .on_action(cx.listener(|this, _: &OpenSettings, window, cx| {
                this.open_configuration(window, cx);
            }))
            .on_action(cx.listener(Self::open_active_thread_as_markdown))
            .on_action(cx.listener(Self::manage_skills))
            .on_action(cx.listener(Self::toggle_options_menu))
            .on_action(cx.listener(Self::increase_font_size))
            .on_action(cx.listener(Self::decrease_font_size))
            .on_action(cx.listener(Self::reset_font_size))
            .on_action(cx.listener(Self::toggle_zoom))
            .on_action(cx.listener(Self::toggle_terminal_thread_search))
            .on_action(cx.listener(|this, _: &ReauthenticateAgent, window, cx| {
                if let Some(conversation_view) = this.active_conversation_view() {
                    conversation_view.update(cx, |conversation_view, cx| {
                        conversation_view.reauthenticate(window, cx)
                    })
                }
            }))
            .on_action(cx.listener(|this, _: &LogoutAgent, window, cx| {
                if let Some(conversation_view) = this.active_conversation_view() {
                    conversation_view.update(cx, |conversation_view, cx| {
                        conversation_view.logout(window, cx)
                    })
                }
            }))
            .child(self.render_toolbar(window, cx))
            .children(self.render_new_user_onboarding(window, cx))
            .map(|parent| match self.visible_surface() {
                VisibleSurface::Uninitialized if !self.has_open_project(cx) => {
                    parent.child(self.render_no_project_state(cx))
                }
                VisibleSurface::Uninitialized => parent,
                VisibleSurface::AgentThread(conversation_view) => parent
                    .child(conversation_view.clone())
                    .child(self.render_drag_target(cx)),
                VisibleSurface::Terminal(terminal_view) => {
                    let search_bar = self
                        .active_terminal_id()
                        .and_then(|terminal_id| self.terminals.get(&terminal_id))
                        .and_then(|terminal| terminal.search_bar.clone());
                    let terminal_content = v_flex()
                        .size_full()
                        .when_some(search_bar, |this, search_bar| {
                            this.when(!search_bar.read(cx).is_dismissed(), |this| {
                                this.child(
                                    v_flex()
                                        .group("toolbar")
                                        .relative()
                                        .py(DynamicSpacing::Base06.rems(cx))
                                        .px(DynamicSpacing::Base08.rems(cx))
                                        .border_b_1()
                                        .border_color(cx.theme().colors().border_variant)
                                        .bg(cx.theme().colors().toolbar_background)
                                        .child(search_bar),
                                )
                            })
                        })
                        .child(terminal_view.clone());

                    parent
                        .child(terminal_content)
                        .child(self.render_drag_target(cx))
                }
            })
            .children(self.render_trial_end_upsell(window, cx));

        match self.visible_font_size() {
            WhichFontSize::AgentFont => {
                let theme_settings = ThemeSettings::get_global(cx);
                WithRemSize::new(theme_settings.agent_ui_font_size(cx))
                    .size_full()
                    .font_family(theme_settings.agent_ui_font_family().clone())
                    .child(content)
                    .into_any()
            }
            _ => content.into_any(),
        }
    }
}

struct OnboardingUpsell;

impl Dismissable for OnboardingUpsell {
    const KEY: &'static str = "dismissed-trial-upsell";
}

struct TrialEndUpsell;

impl Dismissable for TrialEndUpsell {
    const KEY: &'static str = "dismissed-trial-end-upsell";
}

/// Test-only helper methods
#[cfg(any(test, feature = "test-support"))]
impl AgentPanel {
    pub fn test_new(workspace: &Workspace, window: &mut Window, cx: &mut Context<Self>) -> Self {
        Self::new(workspace, window, cx)
    }

    /// Drops a thread's `ConversationView` from `retained_threads` without
    /// deleting its metadata or kvp state. Simulates the post-restart
    pub fn test_unload_retained_thread(&mut self, id: ThreadId) -> bool {
        self.retained_threads.remove(&id).is_some()
    }

    /// Opens an external thread using an arbitrary AgentServer.
    ///
    /// This is a test-only helper that allows visual tests and integration tests
    /// to inject a stub server without modifying production code paths.
    /// Not compiled into production builds.
    pub fn open_external_thread_with_server(
        &mut self,
        server: Rc<dyn AgentServer>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let ext_agent = Agent::Custom {
            id: server.agent_id(),
        };

        let thread = self.create_agent_thread_with_server(
            ext_agent,
            Some(server),
            None,
            None,
            None,
            None,
            None,
            AgentThreadSource::AgentPanel,
            window,
            cx,
        );
        self.set_base_view(thread.into(), true, window, cx);
    }

    /// Opens a restored external thread with an arbitrary AgentServer and
    /// a specific `resume_session_id` — as if we just restored from the KVP.
    ///
    /// Test-only helper. Not compiled into production builds.
    pub fn open_restored_thread_with_server(
        &mut self,
        server: Rc<dyn AgentServer>,
        resume_session_id: acp::SessionId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let ext_agent = Agent::Custom {
            id: server.agent_id(),
        };

        // The panel addresses threads by `ThreadId` after the draft work;
        // map the test-provided `session_id` back through the metadata
        // store so this helper still resumes the right thread.
        let resume_thread_id = ThreadMetadataStore::try_global(cx).and_then(|store| {
            store
                .read(cx)
                .entry_by_session(&resume_session_id)
                .map(|m| m.thread_id)
        });

        let thread = self.create_agent_thread_with_server(
            ext_agent,
            Some(server),
            resume_thread_id,
            None,
            None,
            None,
            None,
            AgentThreadSource::AgentPanel,
            window,
            cx,
        );
        self.set_base_view(thread.into(), true, window, cx);
    }

    /// Returns the currently active thread view, if any.
    ///
    /// This is a test-only accessor that exposes the private `active_thread_view()`
    /// method for test assertions. Not compiled into production builds.
    pub fn active_thread_view_for_tests(&self) -> Option<&Entity<ConversationView>> {
        self.active_conversation_view()
    }

    /// Creates a draft thread using a stub server and sets it as the active view.
    #[cfg(any(test, feature = "test-support"))]
    pub fn open_draft_with_server(
        &mut self,
        server: Rc<dyn AgentServer>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let ext_agent = Agent::Custom {
            id: server.agent_id(),
        };
        let thread = self.create_agent_thread_with_server(
            ext_agent,
            Some(server),
            None,
            None,
            None,
            None,
            None,
            AgentThreadSource::AgentPanel,
            window,
            cx,
        );
        self.draft_thread = Some(thread.conversation_view.clone());
        self.set_base_view(thread.into(), true, window, cx);
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn insert_test_terminal(
        &mut self,
        title: impl Into<String>,
        focus: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<TerminalId> {
        let terminal_id = TerminalId::new();
        self.set_last_created_entry_kind_from_user_action(AgentPanelEntryKind::Terminal, cx);
        self.insert_display_only_terminal(
            terminal_id,
            None,
            Some(SharedString::from(title.into())),
            None,
            None,
            focus,
            focus,
            true,
            AgentThreadSource::AgentPanel,
            window,
            cx,
        )?;
        Ok(terminal_id)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn restore_test_terminal(
        &mut self,
        metadata: TerminalThreadMetadata,
        focus: bool,
        source: AgentThreadSource,
        workspace: Option<&Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if self.has_terminal(metadata.terminal_id) {
            self.activate_terminal(metadata.terminal_id, focus, window, cx);
            return Ok(());
        }

        if !self.supports_terminal(cx) {
            return Ok(());
        }

        let working_directory = self.terminal_restore_working_directory(&metadata, workspace, cx);
        let initial_title = Self::terminal_restore_initial_title(&metadata);
        self.insert_display_only_terminal(
            metadata.terminal_id,
            working_directory,
            metadata.custom_title.clone(),
            initial_title,
            Some(metadata.created_at),
            true,
            focus,
            true,
            source,
            window,
            cx,
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    fn insert_display_only_terminal(
        &mut self,
        terminal_id: TerminalId,
        working_directory: Option<PathBuf>,
        custom_title: Option<SharedString>,
        initial_title: Option<SharedString>,
        created_at: Option<DateTime<Utc>>,
        select: bool,
        focus: bool,
        run_init_command: bool,
        source: AgentThreadSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let init_command = Self::terminal_init_command(run_init_command, cx);
        let settings = TerminalSettings::get_global(cx).clone();
        let path_style = self.project.read(cx).path_style(cx);
        let builder = terminal::TerminalBuilder::new_display_only(
            settings.cursor_shape,
            settings.alternate_scroll,
            settings.max_scroll_history_lines,
            cx.entity_id().as_u64(),
            cx.background_executor(),
            path_style,
        );
        let terminal = cx.new(|cx| builder.subscribe(cx));
        let terminal_for_init_command = terminal.clone();
        let terminal_view = cx.new(|cx| {
            let mut view = TerminalView::new(
                terminal,
                self.workspace.clone(),
                self.workspace_id,
                self.project.downgrade(),
                window,
                cx,
            );
            view.set_show_workspace_actions(false, cx);
            view
        });
        self.insert_terminal(
            terminal_id,
            terminal_view,
            working_directory,
            custom_title,
            initial_title,
            created_at,
            TerminalSpawnAssociation::Standalone,
            select,
            focus,
            source,
            window,
            cx,
        )?;
        Self::write_terminal_init_command(&terminal_for_init_command, init_command, cx);
        Ok(())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn emit_test_terminal_bell(&mut self, terminal_id: TerminalId, cx: &mut Context<Self>) {
        let Some(terminal_entity) = self
            .terminals
            .get(&terminal_id)
            .map(|terminal| terminal.view.read(cx).terminal().clone())
        else {
            return;
        };
        terminal_entity.update(cx, |_terminal, cx| {
            cx.emit(TerminalEvent::Bell);
        });
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn emit_test_terminal_close(&mut self, terminal_id: TerminalId, cx: &mut Context<Self>) {
        let Some(terminal_entity) = self
            .terminals
            .get(&terminal_id)
            .map(|terminal| terminal.view.read(cx).terminal().clone())
        else {
            return;
        };
        terminal_entity.update(cx, |_terminal, cx| {
            cx.emit(TerminalEvent::CloseTerminal);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NewWorktreeBranchTarget;
    use crate::conversation_view::tests::{StubAgentServer, init_test};
    use crate::test_support::{
        active_session_id, active_thread_id, open_thread_with_connection,
        open_thread_with_custom_connection, register_test_sidebar, send_message,
    };
    use acp_thread::{AgentConnection, StubAgentConnection, ThreadStatus};
    use action_log::ActionLog;
    use anyhow::{Result, anyhow};
    use feature_flags::FeatureFlagAppExt;
    use fs::FakeFs;
    use gpui::{App, Modifiers, TestAppContext, UpdateGlobal, VisualTestContext, px, size};
    use parking_lot::Mutex;
    use project::{Project, WorktreePaths};
    use settings::{SettingsStore, WorkingDirectory};
    use std::any::Any;

    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    #[test]
    fn test_is_known_terminal_agent_command() {
        assert!(is_known_terminal_agent_command("claude"));
        assert!(is_known_terminal_agent_command("codex"));
        assert!(!is_known_terminal_agent_command("cargo"));
        assert!(!is_known_terminal_agent_command("internal-agent"));
    }

    #[test]
    fn test_terminal_program_reports_known_agent_transitions() {
        let mut last_observed_program = None;

        assert_eq!(
            terminal_program_to_report(&mut last_observed_program, Some("codex".to_string())),
            Some("codex".to_string())
        );
        assert_eq!(
            terminal_program_to_report(&mut last_observed_program, Some("codex".to_string())),
            None
        );
        assert_eq!(
            terminal_program_to_report(&mut last_observed_program, Some("zsh".to_string())),
            None
        );
        assert_eq!(
            terminal_program_to_report(
                &mut last_observed_program,
                Some("customer-data-export".to_string())
            ),
            None
        );
        assert_eq!(
            terminal_program_to_report(&mut last_observed_program, Some("codex".to_string())),
            Some("codex".to_string())
        );
        assert_eq!(
            terminal_program_to_report(&mut last_observed_program, None),
            None
        );
        assert_eq!(
            terminal_program_to_report(&mut last_observed_program, Some("codex".to_string())),
            Some("codex".to_string())
        );
    }

    #[derive(Clone, Default)]
    struct SessionTrackingConnection {
        next_session_number: Arc<Mutex<usize>>,
        sessions: Arc<Mutex<HashSet<acp::SessionId>>>,
    }

    impl SessionTrackingConnection {
        fn new() -> Self {
            Self::default()
        }

        fn create_session(
            self: Rc<Self>,
            session_id: acp::SessionId,
            project: Entity<Project>,
            work_dirs: PathList,
            title: Option<SharedString>,
            cx: &mut App,
        ) -> Entity<AcpThread> {
            self.sessions.lock().insert(session_id.clone());

            let action_log = cx.new(|_| ActionLog::new(project.clone()));
            cx.new(|cx| {
                AcpThread::new(
                    None,
                    title,
                    Some(work_dirs),
                    self,
                    project,
                    action_log,
                    session_id,
                    watch::Receiver::constant(
                        acp::PromptCapabilities::new()
                            .image(true)
                            .audio(true)
                            .embedded_context(true),
                    ),
                    cx,
                )
            })
        }
    }

    impl AgentConnection for SessionTrackingConnection {
        fn agent_id(&self) -> AgentId {
            agent::ZED_AGENT_ID.clone()
        }

        fn telemetry_id(&self) -> SharedString {
            "session-tracking-test".into()
        }

        fn new_session(
            self: Rc<Self>,
            project: Entity<Project>,
            work_dirs: PathList,
            cx: &mut App,
        ) -> Task<Result<Entity<AcpThread>>> {
            let session_id = {
                let mut next_session_number = self.next_session_number.lock();
                let session_id = acp::SessionId::new(format!(
                    "session-tracking-session-{}",
                    *next_session_number
                ));
                *next_session_number += 1;
                session_id
            };
            let thread = self.create_session(session_id, project, work_dirs, None, cx);
            Task::ready(Ok(thread))
        }

        fn supports_load_session(&self) -> bool {
            true
        }

        fn load_session(
            self: Rc<Self>,
            session_id: acp::SessionId,
            project: Entity<Project>,
            work_dirs: PathList,
            title: Option<SharedString>,
            cx: &mut App,
        ) -> Task<Result<Entity<AcpThread>>> {
            let thread = self.create_session(session_id, project, work_dirs, title, cx);
            thread.update(cx, |thread, cx| {
                thread
                    .handle_session_update(
                        acp::SessionUpdate::UserMessageChunk(acp::ContentChunk::new(
                            "Restored user message".into(),
                        )),
                        cx,
                    )
                    .expect("restored user message should be applied");
                thread
                    .handle_session_update(
                        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                            "Restored assistant message".into(),
                        )),
                        cx,
                    )
                    .expect("restored assistant message should be applied");
            });
            Task::ready(Ok(thread))
        }

        fn supports_close_session(&self) -> bool {
            true
        }

        fn close_session(
            self: Rc<Self>,
            session_id: &acp::SessionId,
            _cx: &mut App,
        ) -> Task<Result<()>> {
            self.sessions.lock().remove(session_id);
            Task::ready(Ok(()))
        }

        fn auth_methods(&self) -> &[acp::AuthMethod] {
            &[]
        }

        fn authenticate(&self, _method_id: acp::AuthMethodId, _cx: &mut App) -> Task<Result<()>> {
            Task::ready(Ok(()))
        }

        fn prompt(
            &self,
            params: acp::PromptRequest,
            _cx: &mut App,
        ) -> Task<Result<acp::PromptResponse>> {
            if !self.sessions.lock().contains(&params.session_id) {
                return Task::ready(Err(anyhow!("Session not found")));
            }

            Task::ready(Ok(acp::PromptResponse::new(acp::StopReason::EndTurn)))
        }

        fn cancel(&self, _session_id: &acp::SessionId, _cx: &mut App) {}

        fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
            self
        }
    }

    #[gpui::test]
    async fn test_clicking_tool_call_output_keeps_agent_panel_focused_and_zoomed(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        cx.simulate_resize(size(px(900.), px(700.)));

        let connection = StubAgentConnection::new();
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            workspace.focus_panel::<AgentPanel>(window, cx);
            panel
        });
        open_thread_with_connection(&panel, connection.clone(), cx);

        let session_id = active_session_id(&panel, cx);
        let tool_call_id = acp::ToolCallId::new("tool-call-output-focus-regression");
        cx.update(|_window, cx| {
            connection.send_update(
                session_id.clone(),
                acp::SessionUpdate::ToolCall(
                    acp::ToolCall::new(tool_call_id.clone(), "Read file")
                        .kind(acp::ToolKind::Fetch)
                        .status(acp::ToolCallStatus::InProgress),
                ),
                cx,
            );
            connection.send_update(
                session_id,
                acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(
                    tool_call_id.clone(),
                    acp::ToolCallUpdateFields::new()
                        .status(acp::ToolCallStatus::Completed)
                        .content(vec![acp::ToolCallContent::Content(acp::Content::new(
                            acp::ContentBlock::Text(acp::TextContent::new(
                                "tool output text".to_string(),
                            )),
                        ))]),
                )),
                cx,
            );
        });
        cx.run_until_parked();

        let thread_view = panel.read_with(cx, |panel, cx| panel.active_thread_view(cx).unwrap());
        thread_view.update(cx, |thread_view, cx| {
            thread_view.entry_view_state.update(cx, |state, _cx| {
                state.expand_tool_call(tool_call_id);
            });
            cx.notify();
        });

        panel.update_in(cx, |panel, window, cx| {
            panel.toggle_zoom(&ToggleZoom, window, cx);
        });

        // The thread receives only tool-call updates, so entry index 0 should remain stable.
        let output_bounds = cx
            .debug_bounds("tool-call-output-0-0")
            .expect("tool call output should be rendered");
        cx.simulate_click(output_bounds.center(), Modifiers::default());
        cx.run_until_parked();

        panel.update_in(cx, |panel, window, cx| {
            assert!(
                panel.focus_handle(cx).contains_focused(window, cx),
                "clicking tool call output should keep focus within the agent panel"
            );
            assert!(
                panel.is_zoomed(window, cx),
                "clicking tool call output should not close Zen Mode"
            );
        });

        let title_editor_focus_handle = panel.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("active thread view should be present")
                .read(cx)
                .title_editor
                .focus_handle(cx)
        });
        cx.update(|window, cx| {
            title_editor_focus_handle.focus(window, cx);
        });
        cx.run_until_parked();

        panel.update_in(cx, |panel, window, cx| {
            assert!(
                panel.focus_handle(cx).contains_focused(window, cx),
                "focusing the thread title editor should keep focus within the agent panel"
            );
            assert!(
                panel.is_zoomed(window, cx),
                "focusing the thread title editor should not close Zen Mode"
            );
        });
    }

    #[gpui::test]
    async fn test_active_thread_serialize_and_load_round_trip(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        // Create a MultiWorkspace window with two workspaces.
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        let project_a = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;
        let project_b = Project::test(fs, [], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));

        let workspace_a = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        workspace_a.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });
        workspace_b.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        // Set up workspace A: with an active thread.
        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        panel_a.update_in(cx, |panel, window, cx| {
            panel.open_external_thread_with_server(
                Rc::new(StubAgentServer::default_response()),
                window,
                cx,
            );
        });

        cx.run_until_parked();

        panel_a.read_with(cx, |panel, cx| {
            assert!(
                panel.active_agent_thread(cx).is_some(),
                "workspace A should have an active thread after connection"
            );
        });

        send_message(&panel_a, cx);

        let agent_type_a = panel_a.read_with(cx, |panel, _cx| panel.selected_agent.clone());

        // Set up workspace B: ClaudeCode, no active thread.
        let panel_b = workspace_b.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        panel_b.update(cx, |panel, _cx| {
            panel.selected_agent = Agent::Custom {
                id: "claude-acp".into(),
            };
        });

        // Serialize both panels.
        panel_a.update(cx, |panel, cx| panel.serialize(cx));
        panel_b.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        let workspace_a_id = workspace_a
            .read_with(cx, |workspace, _cx| workspace.database_id())
            .expect("workspace A should have a database id");
        let kvp = cx.update(|_window, cx| KeyValueStore::global(cx));
        let serialized_a: SerializedAgentPanel = cx
            .background_spawn(async move { read_serialized_panel(workspace_a_id, &kvp) })
            .await
            .expect("workspace A should serialize panel state");
        assert!(
            serialized_a.last_active_thread.is_some(),
            "active thread should be the thread restore target"
        );
        assert!(
            serialized_a.last_active_terminal_id.is_none(),
            "active thread serialization should not also include a terminal restore target"
        );

        cx.update(|_window, cx| {
            ThreadMetadataStore::init_global(cx);
        });

        // Load fresh panels for each workspace and verify independent state.
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded_a = AgentPanel::load(workspace_a.downgrade(), async_cx)
            .await
            .expect("panel A load should succeed");
        cx.run_until_parked();

        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded_b = AgentPanel::load(workspace_b.downgrade(), async_cx)
            .await
            .expect("panel B load should succeed");
        cx.run_until_parked();

        // Workspace A should restore its thread and agent type
        loaded_a.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent, agent_type_a,
                "workspace A agent type should be restored"
            );
            assert!(
                panel.active_conversation_view().is_some(),
                "workspace A should have its active thread restored"
            );
        });

        // Workspace B should restore its own agent type but have no active thread.
        loaded_b.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent,
                Agent::Custom {
                    id: "claude-acp".into()
                },
                "workspace B agent type should be restored"
            );
            assert!(
                panel.active_conversation_view().is_none(),
                "workspace B should have no active thread when it had no prior conversation"
            );
        });
    }

    #[gpui::test]
    async fn test_active_terminal_serialize_and_load_round_trip(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            TerminalThreadMetadataStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        panel.update_in(cx, |panel, window, cx| {
            panel.activate_new_thread(false, AgentThreadSource::AgentPanel, window, cx);
        });
        let terminal_id = panel
            .update_in(cx, |panel, window, cx| {
                panel.insert_test_terminal("Dev Server", true, window, cx)
            })
            .expect("test terminal should be inserted");
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        let workspace_id = workspace
            .read_with(cx, |workspace, _cx| workspace.database_id())
            .expect("workspace should have a database id");
        let kvp = cx.update(|_window, cx| KeyValueStore::global(cx));
        let serialized: SerializedAgentPanel = cx
            .background_spawn(async move { read_serialized_panel(workspace_id, &kvp) })
            .await
            .expect("workspace should serialize panel state");
        assert_eq!(
            serialized.last_active_terminal_id,
            Some(terminal_id.to_key_string())
        );
        assert!(
            serialized.last_active_thread.is_none(),
            "active terminal serialization should not also include a thread restore target"
        );

        cx.update(|_window, cx| {
            TerminalThreadMetadataStore::init_global(cx);
        });
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded = AgentPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel load should succeed");
        for _ in 0..8 {
            cx.run_until_parked();
        }

        loaded.read_with(cx, |panel, cx| {
            assert_eq!(panel.active_terminal_id(), Some(terminal_id));
            assert!(
                panel.active_conversation_view().is_none(),
                "the restored terminal should remain active instead of falling back to a draft"
            );
            assert!(
                panel
                    .terminals(cx)
                    .into_iter()
                    .any(|terminal| terminal.id == terminal_id),
                "active terminal metadata should be restored into the loaded panel"
            );
        });
    }

    #[gpui::test]
    async fn test_terminal_restore_working_directory_does_not_read_leased_workspace(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);

            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .terminal
                        .get_or_insert_default()
                        .project
                        .working_directory = Some(WorkingDirectory::AlwaysHome);
                });
            });
        });

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        project.update(cx, |project, _cx| {
            project.mark_as_collab_for_testing();
        });
        project.read_with(cx, |project, _cx| {
            assert!(project.is_remote());
        });

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .expect("multi workspace should have an active workspace");
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        assert_eq!(
            workspace.read_with(cx, |workspace, cx| {
                terminal_view::default_working_directory(workspace, cx)
            }),
            None
        );

        let metadata = TerminalThreadMetadata {
            terminal_id: TerminalId::new(),
            title: "Dev Server".into(),
            custom_title: None,
            created_at: Utc::now(),
            worktree_paths: project.read_with(cx, |project, cx| project.worktree_paths(cx)),
            remote_connection: None,
            working_directory: None,
            inline_context: None,
        };
        assert_eq!(metadata.working_directory, None);

        let working_directory = workspace.update_in(cx, |workspace, _window, cx| {
            panel
                .read(cx)
                .terminal_restore_working_directory(&metadata, Some(workspace), cx)
        });

        assert_eq!(working_directory, None);
    }

    #[gpui::test]
    async fn test_pending_terminal_restore_prevents_initial_terminal_creation(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.last_created_entry_kind = AgentPanelEntryKind::Terminal;
            panel.pending_terminal_spawn = Some(TerminalId::new());
            panel.set_active(true, window, cx);
        });
        for _ in 0..4 {
            cx.run_until_parked();
        }

        panel.read_with(&cx, |panel, cx| {
            assert!(
                panel.terminals(cx).is_empty(),
                "activation while a terminal restore is pending should not create a second terminal"
            );
            assert!(
                panel.active_conversation_view().is_none(),
                "activation while a terminal restore is pending should not fall back to a draft"
            );
        });
    }

    #[gpui::test]
    async fn test_repeated_activation_only_creates_one_initial_terminal(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.last_created_entry_kind = AgentPanelEntryKind::Terminal;
            panel.set_active(true, window, cx);
            panel.set_active(true, window, cx);
        });
        for _ in 0..8 {
            cx.run_until_parked();
        }

        panel.read_with(&cx, |panel, cx| {
            assert_eq!(
                panel.terminals(cx).len(),
                1,
                "repeated activation should only enqueue one initial terminal"
            );
            assert!(
                panel.active_terminal_id().is_some(),
                "the single initial terminal should become active"
            );
        });
    }

    #[gpui::test]
    async fn test_restored_terminal_runs_init_command_once(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.update(|_, cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.terminal_init_command = Some(" claude --resume ".to_string());
            AgentSettings::override_global(settings, cx);
        });

        let metadata = TerminalThreadMetadata {
            terminal_id: TerminalId::new(),
            title: "Restored Terminal".into(),
            custom_title: None,
            created_at: Utc::now(),
            worktree_paths: WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from(
                "/project",
            )])),
            remote_connection: None,
            working_directory: None,
            inline_context: None,
        };
        let terminal_id = metadata.terminal_id;
        panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.restore_test_terminal(
                    metadata.clone(),
                    true,
                    AgentThreadSource::AgentPanel,
                    None,
                    window,
                    cx,
                )
            })
            .expect("test terminal should be restored");
        cx.run_until_parked();

        let terminal = panel.read_with(&cx, |panel, cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should exist")
                .view
                .read(cx)
                .terminal()
                .clone()
        });
        let input_log = terminal.update(&mut cx, |terminal, _| terminal.take_input_log());
        assert_eq!(input_log, vec![b" claude --resume \r".to_vec()]);
        assert!(
            !terminal.read_with(&cx, |terminal, _| terminal.keyboard_input_sent()),
            "writing the init command must not mark the terminal as having received \
             user keyboard input, otherwise a shell that fails to spawn would be \
             auto-closed before the user can see the error"
        );

        panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.restore_test_terminal(
                    metadata,
                    true,
                    AgentThreadSource::AgentPanel,
                    None,
                    window,
                    cx,
                )
            })
            .expect("restoring an existing test terminal should succeed");
        cx.run_until_parked();

        let input_log = terminal.update(&mut cx, |terminal, _| terminal.take_input_log());
        assert!(
            input_log.is_empty(),
            "activating an already-restored terminal should not re-run the init command, got {input_log:?}"
        );
    }

    /// Exercises the real `spawn_terminal` path with a genuine shell PTY (not the
    /// display-only test terminal, where `write_to_pty` is a no-op) to verify the
    /// init command is actually delivered to the shell and executed.
    #[cfg(unix)]
    #[gpui::test]
    async fn test_spawn_terminal_runs_init_command_in_real_shell(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.executor().allow_parking();
        cx.update(|_, cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            // `init_ran_42` is the command's output, not its echoed text, so finding
            // it proves the shell executed the command rather than just echoing it.
            settings.terminal_init_command = Some("printf 'init_ran_%s\\n' 42".to_string());
            AgentSettings::override_global(settings, cx);

            // Force a known POSIX shell so the test doesn't depend on the developer's login shell.
            let mut terminal_settings = TerminalSettings::get_global(cx).clone();
            terminal_settings.shell = task::Shell::Program("/bin/sh".to_string());
            TerminalSettings::override_global(terminal_settings, cx);
        });

        let terminal_id = TerminalId::new();
        panel.update_in(&mut cx, |panel, window, cx| {
            panel
                .spawn_terminal(
                    terminal_id,
                    // No working directory: the FakeFs project path doesn't exist on
                    // the real filesystem the shell process runs against.
                    None,
                    None,
                    None,
                    None,
                    TerminalSpawnAssociation::Standalone,
                    true,
                    true,
                    true,
                    AgentThreadSource::AgentPanel,
                    window,
                    cx,
                )
                .detach_and_log_err(cx);
        });

        // The shell spawns on a background thread and produces output
        // asynchronously, so poll (with a deadline) rather than using a fixed
        // sleep, matching the real-PTY test in `acp_thread`.
        let deadline = Instant::now() + Duration::from_secs(10);
        let terminal = loop {
            cx.run_until_parked();
            let terminal = panel.read_with(&cx, |panel, cx| {
                panel
                    .terminals
                    .get(&terminal_id)
                    .map(|terminal| terminal.view.read(cx).terminal().clone())
            });
            if let Some(terminal) = &terminal
                && terminal
                    .read_with(&cx, |terminal, _| terminal.get_content())
                    .contains("init_ran_42")
            {
                break terminal.clone();
            }
            if Instant::now() >= deadline {
                let terminal_created = terminal.is_some();
                let (content, input_log) = if let Some(terminal) = terminal {
                    let content = terminal.read_with(&cx, |terminal, _| terminal.get_content());
                    let input_log =
                        terminal.update(&mut cx, |terminal, _| terminal.take_input_log());
                    (content, input_log)
                } else {
                    (String::new(), Vec::new())
                };
                panic!(
                    "init command output never appeared in the terminal; terminal_created={terminal_created}, content={content:?}, input_log={input_log:?}"
                );
            }
            cx.executor().timer(Duration::from_millis(50)).await;
        };

        let input_log = terminal.update(&mut cx, |terminal, _| terminal.take_input_log());
        assert_eq!(
            input_log,
            vec![b"printf 'init_ran_%s\\n' 42\r".to_vec()],
            "init command should be written only after terminal startup has settled"
        );
        assert!(
            !terminal.read_with(&cx, |terminal, _| terminal.keyboard_input_sent()),
            "writing the init command must not mark the terminal as having received \
             user keyboard input"
        );
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_real_pty_preserves_parent_draft_and_entry_preference(
        cx: &mut TestAppContext,
    ) {
        assert_inline_real_pty_preserves_parent_draft(
            false,
            false,
            false,
            InlineTerminalCompletion::NaturalExit,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_unassociated_inline_real_pty_preserves_parent_draft(cx: &mut TestAppContext) {
        assert_inline_real_pty_preserves_parent_draft(
            true,
            false,
            false,
            InlineTerminalCompletion::NaturalExit,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_real_pty_exit_preserves_another_selected_draft(cx: &mut TestAppContext) {
        assert_inline_real_pty_preserves_parent_draft(
            false,
            true,
            false,
            InlineTerminalCompletion::NaturalExit,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_real_pty_exit_rejects_replaced_history_owner(cx: &mut TestAppContext) {
        assert_inline_real_pty_preserves_parent_draft(
            false,
            false,
            true,
            InlineTerminalCompletion::NaturalExit,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_running_terminal_menu_scopes_scroll_and_selected_stop(
        cx: &mut TestAppContext,
    ) {
        use crate::collaborative_composer::CollaborativeComposerAdapter;
        use crate::conversation_view::ThreadView;

        type Registrations = (
            workspace::collaborative_composer::CollaborativeComposerRegistration,
            workspace::collaborative_timeline::CollaborativeTimelineRegistration,
        );

        fn mount_footer(
            panel: &Entity<AgentPanel>,
            registrations: &mut Option<Registrations>,
            cx: &mut VisualTestContext,
        ) {
            let workspace = panel.read_with(cx, |panel, _| {
                panel.workspace.upgrade().expect("canonical workspace")
            });
            let adapter = cx
                .update(|_, cx| {
                    CollaborativeComposerAdapter::from_agent_panel(panel, &workspace, cx)
                })
                .expect("canonical active-conversation adapter");
            let timeline = cx
                .update(|_, cx| {
                    crate::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                        panel, &workspace, cx,
                    )
                })
                .expect("canonical active-conversation timeline adapter");
            workspace.update(cx, |workspace, cx| {
                if let Some((composer, timeline)) = registrations.take() {
                    assert!(workspace.unregister_collaborative_composer_provider(composer, cx));
                    assert!(workspace.unregister_collaborative_timeline_provider(timeline, cx));
                }
                let composer = adapter
                    .register_in_workspace(workspace, cx)
                    .expect("canonical footer registration");
                let timeline = timeline
                    .register_in_workspace(workspace, cx)
                    .expect("canonical timeline registration");
                *registrations = Some((composer, timeline));
            });
            if workspace.read_with(cx, |workspace, _| workspace.workspace_presentation())
                != workspace::WorkspacePresentation::Collaborative
            {
                cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
                cx.run_until_parked();
            }
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.close_all_docks(window, cx);
                AgentPanel::reveal_conversation(workspace, true, window, cx);
            });
            cx.run_until_parked();
            cx.refresh().expect("mounted conversation footer");
            for _ in 0..2 {
                cx.update(|window, cx| window.simulate_next_frame(cx));
                cx.refresh().expect("settled conversation mounting frame");
                cx.run_until_parked();
            }
            assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER-FOOTER").is_some());
        }

        fn open_menu(cx: &mut VisualTestContext) {
            cx.refresh().expect("running-terminal footer");
            let footer = cx
                .debug_bounds("COLLABORATIVE-COMPOSER-FOOTER")
                .expect("mounted footer");
            let trigger = cx
                .debug_bounds("ICON-Terminal")
                .expect("actual terminal trigger");
            assert!(
                footer.contains(&trigger.center()),
                "trigger must be inside footer"
            );
            cx.simulate_click(trigger.center(), Modifiers::default());
            cx.run_until_parked();
            cx.refresh().expect("rendered running-terminal menu");
        }

        fn select_item(submenu: &'static str, item: &'static str, cx: &mut VisualTestContext) {
            let bounds = cx
                .debug_bounds(submenu)
                .expect("actual scoped terminal submenu");
            cx.simulate_click(bounds.center(), Modifiers::default());
            cx.run_until_parked();
            cx.refresh().expect("rendered per-terminal submenu");
            let bounds = cx.debug_bounds(item).expect("actual per-terminal action");
            if item == "MENU_ITEM-Stop Process" {
                cx.dispatch_action(menu::SelectLast);
                cx.dispatch_action(menu::Confirm);
            } else {
                cx.simulate_click(bounds.center(), Modifiers::default());
            }
            cx.run_until_parked();
            cx.refresh().expect("selected terminal action");
        }

        async fn wait_for_output(
            terminal: &Entity<terminal::Terminal>,
            marker: &str,
            cx: &mut VisualTestContext,
        ) {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                cx.run_until_parked();
                if terminal.read_with(cx, |terminal, _| {
                    terminal
                        .get_content()
                        .lines()
                        .any(|line| line.trim() == marker)
                }) {
                    break;
                }
                if Instant::now() >= deadline {
                    let (process_id, is_pty, keyboard_input_sent, content) =
                        terminal.read_with(cx, |terminal, _| {
                            (
                                terminal.pid(),
                                terminal.is_pty(),
                                terminal.keyboard_input_sent(),
                                terminal.last_n_non_empty_lines(20),
                            )
                        });
                    let input_log = terminal.update(cx, |terminal, _| terminal.take_input_log());
                    panic!(
                        "real PTY output missing: {marker}; process_id={process_id:?}, is_pty={is_pty}, keyboard_input_sent={keyboard_input_sent}, content={content:?}, input_log={input_log:?}"
                    );
                }
                cx.executor().timer(Duration::from_millis(50)).await;
            }
        }

        async fn spawn_inline(
            panel: &Entity<AgentPanel>,
            parent: &Entity<ThreadView>,
            title: &str,
            independent_execution: bool,
            command: &str,
            marker: &str,
            cx: &mut VisualTestContext,
        ) -> (
            TerminalId,
            Entity<TerminalView>,
            Entity<terminal::Terminal>,
            u32,
        ) {
            cx.update(|_, cx| {
                assert_eq!(
                    TerminalSettings::get_global(cx).shell,
                    task::Shell::Program("/bin/sh".into()),
                    "File-action fixture must retain its authored shell settings"
                );
                let mut settings = AgentSettings::get_global(cx).clone();
                settings.terminal_init_command = Some(command.into());
                AgentSettings::override_global(settings, cx);
            });
            let context = parent
                .read_with(cx, |parent, cx| parent.inline_terminal_context(cx))
                .expect("canonical source-bound context");
            let previous_ids = panel.read_with(cx, |panel, _| {
                panel.terminals.keys().copied().collect::<Vec<_>>()
            });
            if independent_execution {
                // Independent siblings exercise selected Stop; repeated File Open reuses one live shell.
                let (terminal_id, terminal) = panel
                    .update_in(cx, |panel, window, cx| {
                        panel.spawn_inline_terminal(context.clone(), None, window, cx)
                    })
                    .await
                    .expect("independent canonical sibling execution");
                parent
                    .update_in(cx, |parent, window, cx| {
                        parent.attach_inline_terminal(terminal_id, terminal, window, cx)
                    })
                    .expect("independent sibling inline placement");
            } else {
                cx.dispatch_action(NewTerminalThread);
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            let (terminal_id, terminal_view) = loop {
                cx.run_until_parked();
                let created = panel.read_with(cx, |panel, _| {
                    panel
                        .terminals
                        .iter()
                        .filter(|(terminal_id, _)| !previous_ids.contains(terminal_id))
                        .map(|(terminal_id, terminal)| (*terminal_id, terminal.view.clone()))
                        .collect::<Vec<_>>()
                });
                assert!(
                    created.len() <= 1,
                    "File action must create only one terminal"
                );
                if let Some(created) = created.into_iter().next() {
                    break created;
                }
                assert!(
                    Instant::now() < deadline,
                    "the production File action must allocate its canonical inline PTY"
                );
                cx.executor().timer(Duration::from_millis(50)).await;
            };
            terminal_view.update(cx, |view, cx| view.set_custom_title(Some(title.into()), cx));
            panel.update(cx, |panel, cx| {
                panel.persist_terminal_metadata(terminal_id, cx)
            });
            let terminal = terminal_view.read_with(cx, |view, _| view.terminal().clone());
            // The foreground process group changes during commands; ownership belongs to the PTY child.
            let process_id = terminal
                .read_with(cx, |terminal, _| {
                    terminal.pid_getter().map(|getter| getter.fallback_pid())
                })
                .expect("owned real PTY process");
            println!(
                "INLINE_REAL_PTY_OWNER process_id={} role={title}",
                process_id.as_u32()
            );
            wait_for_output(&terminal, marker, cx).await;
            cx.refresh()
                .expect("production File action inline placement");
            let workspace = panel.read_with(cx, |panel, _| {
                panel.workspace.upgrade().expect("canonical workspace")
            });
            workspace.read_with(cx, |workspace, cx| {
                assert_eq!(
                    workspace.workspace_presentation(),
                    workspace::WorkspacePresentation::Collaborative
                );
                assert!(
                    workspace.all_docks().into_iter().all(|dock| {
                        !dock
                            .read(cx)
                            .visible_panel()
                            .is_some_and(|panel| panel.is_agent_panel(cx))
                    }),
                    "File Open Terminal must stay inside the integrated conversation"
                );
            });
            let history = cx.update(|_, cx| {
                TerminalThreadMetadataStore::try_global(cx).expect("canonical terminal history")
            });
            history.read_with(cx, |history, _| {
                let recorded = history
                    .entry(terminal_id)
                    .and_then(|metadata| metadata.inline_context.as_ref())
                    .expect("File action must retain its inline parent association");
                assert_eq!(recorded.thread_id, context.thread_id);
                assert_eq!(recorded.session_id, context.session_id);
            });
            terminal_view.update_in(cx, |view, window, cx| {
                assert!(
                    view.focus_handle(cx).contains_focused(window, cx),
                    "File action must focus the inline PTY rather than its composer or dock: current={:?}, terminal={:?}, directly_focused={}",
                    window.focused(cx),
                    view.focus_handle(cx),
                    view.focus_handle(cx).is_focused(window)
                );
            });
            (terminal_id, terminal_view, terminal, process_id.as_u32())
        }

        let (panel, mut cx) = setup_visible_panel(cx).await;
        let mut registrations = None;
        cx.simulate_resize(size(px(1200.), px(700.)));
        cx.executor().allow_parking();
        let user_settings =
            r#"{"terminal":{"shell":{"program":"/bin/sh"},"working_directory":"always_home"}}"#;
        let settings_fs = panel.read_with(&cx, |panel, cx| {
            panel
                .workspace
                .upgrade()
                .expect("canonical workspace")
                .read(cx)
                .project()
                .read(cx)
                .fs()
                .clone()
        });
        // Switching presentation rereads settings from the project's filesystem.
        settings_fs
            .atomic_write(paths::settings_file().clone(), user_settings.into())
            .await
            .expect("workspace presentation must update the authored fixture settings file");
        cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            SettingsStore::update_global(cx, |store, cx| {
                store
                    .set_user_settings(user_settings, cx)
                    .result()
                    .expect("File-action fixture shell settings must parse");
            });
        });
        open_thread_with_connection(&panel, StubAgentConnection::new(), &mut cx);
        crate::test_support::type_draft_prompt(&panel, "Conversation A\n", &mut cx);
        mount_footer(&panel, &mut registrations, &mut cx);
        let parent_a = panel.read_with(&cx, |panel, cx| panel.active_thread_view(cx).expect("A"));
        let thread_a = panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx).expect("A id"));
        let (sibling_id, sibling_view, sibling, sibling_process_id) = spawn_inline(
            &panel,
            &parent_a,
            "FOOTER_A_SIBLING",
            false,
            "printf 'FOOTER_SIBLING_READY\\n'",
            "FOOTER_SIBLING_READY",
            &mut cx,
        )
        .await;
        cx.refresh().expect("rendered sibling Background control");
        let sibling_background = cx
            .debug_bounds("inline-terminal-background")
            .expect("only sibling terminal is eligible for Background");
        cx.simulate_click(sibling_background.center(), Modifiers::default());
        cx.run_until_parked();
        assert!(sibling_view.read_with(&cx, |view, _| view.is_read_only()));
        parent_a.update_in(&mut cx, |parent, window, cx| {
            parent.message_editor.update(cx, |editor, cx| {
                editor.editor().update(cx, |editor, cx| {
                    editor.insert(&"Prefix before selected terminal\n".repeat(80), window, cx);
                });
            });
        });
        let (selected_id, selected_view, selected, selected_process_id) = spawn_inline(
            &panel, &parent_a, "FOOTER_A_SELECTED", true,
            "index=1; while [ \"$index\" -le 40 ]; do printf 'FOOTER_ROW_%02d\\n' \"$index\"; index=$((index+1)); done; printf 'FOOTER_SELECTED_READY\\n'",
            "FOOTER_SELECTED_READY", &mut cx,
        ).await;
        selected_view.update_in(&mut cx, |view, window, cx| {
            assert!(view.terminal().read(cx).is_pty());
            assert!(!view.is_read_only());
            assert!(view.focus_handle(cx).contains_focused(window, cx));
        });
        for _ in 0..3 {
            cx.update(|window, cx| window.simulate_next_frame(cx));
            cx.refresh().expect("settled production File entry layout");
            cx.run_until_parked();
        }
        let terminal_body = selected.read_with(&cx, |terminal, _| {
            terminal.last_content().terminal_bounds.bounds
        });
        let initial_timeline = cx
            .debug_bounds("COLLABORATIVE-TIMELINE-REGION")
            .expect("production File entry conversation viewport");
        let initial_header = cx
            .debug_bounds("inline-terminal-background")
            .expect("production File entry header control");
        assert!(terminal_body.size.width > px(0.) && terminal_body.size.height > px(0.));
        let fully_contains = |viewport: &gpui::Bounds<Pixels>, body: &gpui::Bounds<Pixels>| {
            body.origin.x >= viewport.origin.x
                && body.origin.y >= viewport.origin.y
                && body.bottom_right().x <= viewport.bottom_right().x
                && body.bottom_right().y <= viewport.bottom_right().y
        };
        assert!(
            fully_contains(&initial_timeline, &terminal_body)
                && initial_timeline.contains(&initial_header.center()),
            "File entry must reveal the measured terminal body before menu Scroll: body={terminal_body:?}, timeline={initial_timeline:?}, header={initial_header:?}"
        );
        parent_a.update_in(&mut cx, |parent, _, cx| {
            let editor = parent.message_editor.read(cx).editor();
            let viewport = editor
                .read(cx)
                .last_bounds()
                .expect("production File entry inner viewport");
            assert!(
                fully_contains(viewport, &terminal_body)
                    && viewport.contains(&initial_header.center()),
                "File entry must reveal the terminal through the inner editor clipping viewport: body={terminal_body:?}, editor={viewport:?}, header={initial_header:?}"
            );
        });
        cx.refresh().expect("mounted selected terminal");
        let background = cx
            .debug_bounds("inline-terminal-background")
            .expect("actual selected terminal Background control");
        selected_view.update_in(&mut cx, |view, window, cx| {
            assert!(view.focus_handle(cx).contains_focused(window, cx));
        });
        cx.simulate_click(background.center(), Modifiers::default());
        cx.run_until_parked();
        assert!(selected_view.read_with(&cx, |view, _| view.is_read_only()));
        parent_a.update_in(&mut cx, |parent, window, cx| {
            assert!(
                parent
                    .message_editor
                    .focus_handle(cx)
                    .contains_focused(window, cx)
            );
            parent.message_editor.update(cx, |editor, cx| {
                editor.editor().update(cx, |editor, cx| {
                    editor.insert(&"Suffix after selected terminal\n".repeat(80), window, cx);
                });
            });
        });
        selected_view.update_in(&mut cx, |view, window, cx| {
            assert!(
                view.embedded_hidden_lines(window, cx) > 0,
                "selected terminal must be the only long-output history control"
            );
        });
        let text_before_stop =
            parent_a.read_with(&cx, |parent, cx| parent.message_editor.read(cx).text(cx));
        let history = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        history.update(&mut cx, |history, cx| {
            let mut metadata = history
                .entry(selected_id)
                .cloned()
                .expect("selected history");
            metadata
                .inline_context
                .as_mut()
                .expect("inline context")
                .collapsed = true;
            history
                .save_inline(metadata, cx)
                .expect("retain collapsed state");
        });
        cx.run_until_parked();
        let (focus_before_scroll, selections_before_scroll) =
            parent_a.update_in(&mut cx, |parent, window, cx| {
                let editor = parent.message_editor.read(cx).editor();
                (
                    window.focused(cx).expect("draft focus before Scroll"),
                    editor.read(cx).selections.disjoint_anchors().to_vec(),
                )
            });
        open_menu(&mut cx);
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SELECTED").is_some());
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SIBLING").is_some());
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_B_OTHER").is_none());
        select_item(
            "MENU_SUBMENU-FOOTER_A_SELECTED",
            "MENU_ITEM-Scroll to Terminal",
            &mut cx,
        );
        assert!(history.read_with(&cx, |history, _| {
            !history
                .entry(selected_id)
                .expect("selected history")
                .inline_context
                .as_ref()
                .expect("inline context")
                .collapsed
        }));
        let visible_terminal = cx
            .debug_bounds("inline-terminal-history-disclosure")
            .expect("Scroll must render the selected terminal within its long draft");
        let timeline = cx
            .debug_bounds("COLLABORATIVE-TIMELINE-REGION")
            .expect("canonical conversation viewport");
        assert!(
            timeline.contains(&visible_terminal.center()),
            "Scroll must bring the actual terminal into the viewport"
        );
        parent_a.update_in(&mut cx, |parent, window, cx| {
            let editor = parent.message_editor.read(cx).editor();
            let editor = editor.read(cx);
            assert!(
                editor
                    .last_bounds()
                    .expect("rendered inline conversation viewport")
                    .size
                    .height
                    >= timeline.size.height * 0.75,
                "long inline drafts must use the conversation height rather than the dock composer cap"
            );
            assert!(
                editor
                    .last_bounds()
                    .expect("rendered inner editor viewport")
                    .contains(&visible_terminal.center()),
                "Scroll must reveal the terminal through the inner editor clipping viewport"
            );
            assert_eq!(
                editor.selections.disjoint_anchors(),
                selections_before_scroll.as_slice(),
                "Scroll must preserve the draft selection"
            );
            assert_eq!(
                window.focused(cx).as_ref(),
                Some(&focus_before_scroll),
                "Scroll must restore the existing focus after menu dismissal"
            );
        });
        selected_view.update_in(&mut cx, |view, window, cx| {
            assert!(view.is_read_only(), "Scroll must preserve Background state");
            assert!(
                !view.focus_handle(cx).contains_focused(window, cx),
                "Scroll must not implicitly focus the PTY"
            );
            assert_eq!(view.terminal().entity_id(), selected.entity_id());
        });
        for height in [900., 700.] {
            cx.simulate_resize(size(px(1200.), px(height)));
            cx.refresh().expect("resized conversation layout");
            cx.refresh().expect("settled conversation viewport");
            let timeline = cx
                .debug_bounds("COLLABORATIVE-TIMELINE-REGION")
                .expect("resized conversation viewport");
            parent_a.read_with(&cx, |parent, cx| {
                assert!(
                    parent
                        .message_editor
                        .read(cx)
                        .editor()
                        .read(cx)
                        .last_bounds()
                        .expect("resized inline editor")
                        .size
                        .height
                        >= timeline.size.height * 0.75,
                    "inline viewport must follow the resized conversation"
                );
            });
        }
        open_thread_with_connection(&panel, StubAgentConnection::new(), &mut cx);
        crate::test_support::type_draft_prompt(&panel, "Conversation B", &mut cx);
        mount_footer(&panel, &mut registrations, &mut cx);
        let footer = cx
            .debug_bounds("COLLABORATIVE-COMPOSER-FOOTER")
            .expect("B footer");
        assert!(
            cx.debug_bounds("ICON-Terminal")
                .is_none_or(|bounds| !footer.contains(&bounds.center())),
            "A terminals must not appear in empty B footer"
        );
        let parent_b = panel.read_with(&cx, |panel, cx| panel.active_thread_view(cx).expect("B"));
        let thread_b = panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx).expect("B id"));
        assert_ne!(thread_a, thread_b);
        let (other_id, other_view, other, other_process_id) = spawn_inline(
            &panel,
            &parent_b,
            "FOOTER_B_OTHER",
            false,
            "printf 'FOOTER_OTHER_READY\\n'",
            "FOOTER_OTHER_READY",
            &mut cx,
        )
        .await;
        open_menu(&mut cx);
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_B_OTHER").is_some());
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SELECTED").is_none());
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SIBLING").is_none());
        cx.dispatch_action(menu::Cancel);
        cx.run_until_parked();
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.activate_retained_thread(thread_a, true, window, cx)
        });
        mount_footer(&panel, &mut registrations, &mut cx);
        open_menu(&mut cx);
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SELECTED").is_some());
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SIBLING").is_some());
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_B_OTHER").is_none());
        assert_eq!(
            panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx)),
            Some(thread_a)
        );
        let sibling_pid = sibling
            .read_with(&cx, |terminal, _| {
                terminal.pid_getter().map(|getter| getter.fallback_pid())
            })
            .expect("sibling PID");
        let other_pid = other
            .read_with(&cx, |terminal, _| {
                terminal.pid_getter().map(|getter| getter.fallback_pid())
            })
            .expect("other PID");
        select_item(
            "MENU_SUBMENU-FOOTER_A_SELECTED",
            "MENU_ITEM-Stop Process",
            &mut cx,
        );
        cx.condition(&panel, |panel, _| !panel.has_terminal(selected_id))
            .await;
        cx.refresh().expect("rendered selected-stop history");
        assert!(history.read_with(&cx, |history, _| {
            history
                .entry(selected_id)
                .and_then(|metadata| metadata.inline_context.as_ref())
                .is_some_and(|context| {
                    context.output.contains("FOOTER_SELECTED_READY")
                        && context.output.contains("FOOTER_ROW_01")
                })
        }));
        assert_eq!(
            parent_a.read_with(&cx, |parent, cx| parent.message_editor.read(cx).text(cx)),
            text_before_stop
        );
        let retained_terminal = cx
            .debug_bounds("inline-terminal-history-disclosure")
            .expect("selected-stop must retain rendered historical output");
        let timeline = cx
            .debug_bounds("COLLABORATIVE-TIMELINE-REGION")
            .expect("retained conversation viewport");
        assert!(timeline.contains(&retained_terminal.center()));
        assert_ne!(sibling_pid, other_pid);
        assert_eq!(sibling_pid.as_u32(), sibling_process_id);
        assert_eq!(other_pid.as_u32(), other_process_id);
        assert_ne!(selected_process_id, sibling_process_id);
        assert_ne!(selected_process_id, other_process_id);
        for (terminal, process_id, marker, command) in [
            (
                &sibling,
                sibling_pid,
                "FOOTER_SIBLING_SURVIVES",
                "printf 'FOOTER_SIBLING_SURVIVES\\n'\r",
            ),
            (
                &other,
                other_pid,
                "FOOTER_OTHER_SURVIVES",
                "printf 'FOOTER_OTHER_SURVIVES\\n'\r",
            ),
        ] {
            assert_eq!(
                terminal.read_with(&cx, |terminal, _| {
                    terminal.pid_getter().map(|getter| getter.fallback_pid())
                }),
                Some(process_id)
            );
            terminal.update(&mut cx, |terminal, _| {
                terminal.input(command.as_bytes().to_vec())
            });
            wait_for_output(terminal, marker, &mut cx).await;
        }
        assert!(
            panel.read_with(&cx, |panel, _| panel.has_terminal(sibling_id)
                && panel.has_terminal(other_id))
        );
        assert_eq!(
            sibling_view.read_with(&cx, |view, _| view.terminal().entity_id()),
            sibling.entity_id()
        );
        assert_eq!(
            other_view.read_with(&cx, |view, _| view.terminal().entity_id()),
            other.entity_id()
        );
        open_menu(&mut cx);
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SELECTED").is_none());
        assert!(cx.debug_bounds("MENU_SUBMENU-FOOTER_A_SIBLING").is_some());
        select_item(
            "MENU_SUBMENU-FOOTER_A_SIBLING",
            "MENU_ITEM-Stop Process",
            &mut cx,
        );
        cx.condition(&panel, |panel, _| !panel.has_terminal(sibling_id))
            .await;
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.activate_retained_thread(thread_b, true, window, cx)
        });
        mount_footer(&panel, &mut registrations, &mut cx);
        open_menu(&mut cx);
        select_item(
            "MENU_SUBMENU-FOOTER_B_OTHER",
            "MENU_ITEM-Stop Process",
            &mut cx,
        );
        cx.condition(&panel, |panel, _| !panel.has_terminal(other_id))
            .await;
        assert!(panel.read_with(&cx, |panel, _| {
            [selected_id, sibling_id, other_id]
                .iter()
                .all(|id| !panel.has_terminal(*id))
        }));
        let background_executor = cx.update(|_, cx| cx.background_executor().clone());
        let deadline = Instant::now() + Duration::from_secs(10);
        for process_id in [selected_process_id, sibling_process_id, other_process_id] {
            loop {
                cx.run_until_parked();
                let output = background_executor
                    .spawn(async move {
                        util::command::new_command("/bin/ps")
                            .args(["-p", &process_id.to_string(), "-o", "pid="])
                            .output()
                            .await
                    })
                    .await
                    .expect("read-only owned PID absence check must execute");
                assert!(
                    output.stderr.is_empty(),
                    "unexpected ps error for owned PID {process_id}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                match output.status.code() {
                    Some(1) => {
                        assert!(
                            output.stdout.is_empty(),
                            "absent owned PID {process_id} must have empty ps stdout"
                        );
                        println!("INLINE_REAL_PTY_OWNER_ABSENT process_id={process_id}");
                        break;
                    }
                    Some(0) => {
                        let pid = std::str::from_utf8(&output.stdout)
                            .expect("ps PID output must be valid UTF-8")
                            .trim();
                        assert_eq!(
                            pid,
                            process_id.to_string(),
                            "ps must report only the requested owned PID"
                        );
                        assert!(
                            Instant::now() < deadline,
                            "rendered Stop left owned PID {process_id} alive"
                        );
                        cx.executor().timer(Duration::from_millis(50)).await;
                    }
                    status => {
                        panic!("unexpected ps exit status for owned PID {process_id}: {status:?}")
                    }
                }
            }
        }
        history
            .update(&mut cx, |history, cx| history.flush(cx))
            .await
            .expect("durable selected-stop output history");
        cx.refresh()
            .expect("rendered empty running-terminal footer");
        let footer = cx
            .debug_bounds("COLLABORATIVE-COMPOSER-FOOTER")
            .expect("empty B footer");
        assert!(
            cx.debug_bounds("ICON-Terminal")
                .is_none_or(|bounds| !footer.contains(&bounds.center()))
        );
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_file_open_reuses_one_live_execution_and_freezes_prior_occurrence(
        cx: &mut TestAppContext,
    ) {
        assert_file_open_reuses_one_live_execution(false, cx).await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_file_open_reuse_partial_attachment_rolls_back_without_stopping_execution(
        cx: &mut TestAppContext,
    ) {
        assert_file_open_reuses_one_live_execution(true, cx).await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[derive(Clone, Copy)]
    enum FileOpenEligibilityCase {
        StopPending,
        ReplacedHistoryOwner,
        ClosedHistoryQueue,
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_file_open_during_stop_pending_allocates_distinct_execution(
        cx: &mut TestAppContext,
    ) {
        assert_file_open_canonical_eligibility_guard(FileOpenEligibilityCase::StopPending, cx)
            .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_file_open_replaced_history_owner_preserves_live_execution_and_saved_composition(
        cx: &mut TestAppContext,
    ) {
        assert_file_open_canonical_eligibility_guard(
            FileOpenEligibilityCase::ReplacedHistoryOwner,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_file_open_closed_history_queue_preserves_live_execution_and_composition(
        cx: &mut TestAppContext,
    ) {
        assert_file_open_canonical_eligibility_guard(
            FileOpenEligibilityCase::ClosedHistoryQueue,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    async fn assert_file_open_canonical_eligibility_guard(
        case: FileOpenEligibilityCase,
        cx: &mut TestAppContext,
    ) {
        use crate::collaborative_composer::CollaborativeComposerAdapter;

        type Registrations = (
            workspace::collaborative_composer::CollaborativeComposerRegistration,
            workspace::collaborative_timeline::CollaborativeTimelineRegistration,
        );

        fn mount_footer(
            panel: &Entity<AgentPanel>,
            registrations: &mut Option<Registrations>,
            cx: &mut VisualTestContext,
        ) {
            let workspace = panel.read_with(cx, |panel, _| {
                panel.workspace.upgrade().expect("canonical workspace")
            });
            let adapter = cx
                .update(|_, cx| {
                    CollaborativeComposerAdapter::from_agent_panel(panel, &workspace, cx)
                })
                .expect("canonical active-conversation adapter");
            let timeline = cx
                .update(|_, cx| {
                    crate::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                        panel, &workspace, cx,
                    )
                })
                .expect("canonical active-conversation timeline adapter");
            workspace.update(cx, |workspace, cx| {
                if let Some((composer, timeline)) = registrations.take() {
                    assert!(workspace.unregister_collaborative_composer_provider(composer, cx));
                    assert!(workspace.unregister_collaborative_timeline_provider(timeline, cx));
                }
                let composer = adapter
                    .register_in_workspace(workspace, cx)
                    .expect("canonical footer registration");
                let timeline = timeline
                    .register_in_workspace(workspace, cx)
                    .expect("canonical timeline registration");
                *registrations = Some((composer, timeline));
            });
            if workspace.read_with(cx, |workspace, _| workspace.workspace_presentation())
                != workspace::WorkspacePresentation::Collaborative
            {
                cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
                cx.run_until_parked();
            }
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.close_all_docks(window, cx);
                AgentPanel::reveal_conversation(workspace, true, window, cx);
            });
            cx.run_until_parked();
            cx.refresh().expect("mounted conversation footer");
            for _ in 0..2 {
                cx.update(|window, cx| window.simulate_next_frame(cx));
                cx.refresh().expect("settled conversation mounting frame");
                cx.run_until_parked();
            }
            assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER-FOOTER").is_some());
        }

        async fn wait_for_output(
            terminal: &Entity<terminal::Terminal>,
            marker: &str,
            cx: &mut VisualTestContext,
        ) {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                cx.run_until_parked();
                if terminal.read_with(cx, |terminal, _| {
                    terminal
                        .get_content()
                        .lines()
                        .any(|line| line.trim() == marker)
                }) {
                    break;
                }
                if Instant::now() >= deadline {
                    let (process_id, is_pty, keyboard_input_sent, content) =
                        terminal.read_with(cx, |terminal, _| {
                            (
                                terminal.pid(),
                                terminal.is_pty(),
                                terminal.keyboard_input_sent(),
                                terminal.last_n_non_empty_lines(20),
                            )
                        });
                    let input_log = terminal.update(cx, |terminal, _| terminal.take_input_log());
                    panic!(
                        "real PTY output missing: {marker}; process_id={process_id:?}, is_pty={is_pty}, keyboard_input_sent={keyboard_input_sent}, content={content:?}, input_log={input_log:?}"
                    );
                }
                cx.executor().timer(Duration::from_millis(50)).await;
            }
        }

        let (panel, mut cx) = setup_visible_panel(cx).await;
        cx.executor().allow_parking();
        cx.simulate_resize(size(px(1100.), px(750.)));
        let init_command = "printf '\\033[31mREUSE_FIRST\\033[0m\\n'";
        let user_settings = json!({
            "terminal": { "shell": { "program": "/bin/sh" }, "working_directory": "always_home" },
            "agent": { "terminal_init_command": init_command }
        })
        .to_string();
        let fs = panel.read_with(&cx, |panel, cx| panel.project.read(cx).fs().clone());
        fs.atomic_write(paths::settings_file().clone(), user_settings.clone())
            .await
            .expect("durable authored fixture settings");
        cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            SettingsStore::update_global(cx, |store, cx| {
                store
                    .set_user_settings(&user_settings, cx)
                    .result()
                    .expect("canonical shell settings")
            });
        });
        open_thread_with_connection(&panel, StubAgentConnection::new(), &mut cx);
        crate::test_support::type_draft_prompt(&panel, "Aλ🦀B", &mut cx);
        let mut registrations = None;
        mount_footer(&panel, &mut registrations, &mut cx);
        let parent = panel.read_with(&cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("canonical conversation")
        });
        parent.update_in(&mut cx, |parent, window, cx| {
            parent.message_editor.update(cx, |editor, cx| {
                editor.set_cursor_offset("Aλ".len(), window, cx)
            })
        });
        cx.update(|_, cx| {
            assert_eq!(
                AgentSettings::get_global(cx)
                    .terminal_init_command
                    .as_deref(),
                Some(init_command),
                "actual File fixture must retain its authored init command after settings reload"
            );
            assert_eq!(
                TerminalSettings::get_global(cx).shell,
                task::Shell::Program("/bin/sh".into())
            );
        });
        cx.dispatch_action(NewTerminalThread);
        let deadline = Instant::now() + Duration::from_secs(10);
        let (terminal_id, view) = loop {
            cx.run_until_parked();
            let live = panel.read_with(&cx, |panel, _| {
                panel
                    .terminals
                    .iter()
                    .map(|(id, terminal)| (*id, terminal.view.clone()))
                    .collect::<Vec<_>>()
            });
            assert!(
                live.len() <= 1,
                "initial File entry allocates exactly one execution"
            );
            if let Some(live) = live.into_iter().next() {
                break live;
            }
            assert!(
                Instant::now() < deadline,
                "real File entry must allocate a PTY"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        };
        let terminal = view.read_with(&cx, |view, _| view.terminal().clone());
        wait_for_output(&terminal, "REUSE_FIRST", &mut cx).await;
        let original_pid = terminal.read_with(&cx, |terminal, _| {
            terminal.pid_getter().expect("real PTY").fallback_pid()
        });
        println!(
            "INLINE_REAL_PTY_OWNER process_id={} role=REUSE_ORIGINAL",
            original_pid.as_u32()
        );
        view.update(&mut cx, |view, cx| {
            view.set_custom_title(Some("REUSE_EXECUTION".into()), cx)
        });
        panel.update(&mut cx, |panel, cx| {
            panel.persist_terminal_metadata(terminal_id, cx)
        });
        let history = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        let first_context = history.read_with(&cx, |history, _| {
            history
                .entry(terminal_id)
                .expect("first canonical row")
                .inline_context
                .clone()
                .expect("first context")
        });
        assert_eq!(first_context.occurrence_id, None);
        let workspace = panel.read_with(&cx, |panel, _| {
            panel.workspace.upgrade().expect("canonical workspace")
        });
        let before_blocks = parent
            .update(&mut cx, |parent, cx| {
                parent
                    .message_editor
                    .update(cx, |editor, cx| editor.draft_contents(cx))
            })
            .await
            .expect("real producer before refusal");
        let thread_id = parent.read_with(&cx, |parent, _| parent.root_thread_id);
        cx.update(|_, cx| {
            crate::draft_prompt_store::write_with_live_draft(
                thread_id,
                Some(&before_blocks),
                &before_blocks,
                cx,
            )
        })
        .await
        .expect("actual pre-refusal KVP");
        let (owner, before_metadata) = panel.read_with(&cx, |panel, cx| {
            panel
                .inline_terminal_snapshot_metadata(terminal_id, cx)
                .expect("genuine snapshot preimage")
        });
        assert_eq!(owner.entity_id(), history.entity_id());
        owner
            .update(&mut cx, |owner, cx| {
                owner.save_inline(before_metadata.clone(), cx)
            })
            .expect("canonical latest-grid preimage");
        owner
            .update(&mut cx, |owner, cx| owner.flush(cx))
            .await
            .expect("original owner actual SQLite preimage");
        let close_seen = Arc::new(Mutex::new(false));
        let _close_subscription = cx.update(|_, cx| {
            let close_seen = close_seen.clone();
            cx.subscribe(&terminal, move |_, event: &TerminalEvent, _| {
                if matches!(event, TerminalEvent::CloseTerminal) {
                    *close_seen.lock() = true;
                }
            })
        });
        match case {
            FileOpenEligibilityCase::StopPending => {
                let pending_action_checked = Arc::new(Mutex::new(false));
                cx.update(|window, cx| {
                assert!(terminal.read(cx).can_reuse_interactive_process(), "real shell eligible before the canonical Stop request");
                panel.update(cx, |panel, cx| panel.request_stop_inline_terminal(terminal_id, cx)).expect("genuine selected execution Stop request");
                assert!(!*close_seen.lock() && terminal.read(cx).is_pty(), "fixture must dispatch while Stop is requested and the close event has not been processed");
                assert_eq!(AgentSettings::get_global(cx).terminal_init_command.as_deref(), Some(init_command), "pending File entry must retain its authored init command");
                assert_eq!(TerminalSettings::get_global(cx).shell, task::Shell::Program("/bin/sh".into()));
                window.dispatch_action(Box::new(NewTerminalThread), cx);
                let pending_action_checked = pending_action_checked.clone();
                let close_seen = close_seen.clone(); let terminal = terminal.clone(); let history = history.clone();
                // Action dispatch is deferred; the next FIFO effect measures the genuine entry before any close callback.
                cx.defer(move |cx| {
                    assert!(!*close_seen.lock() && terminal.read(cx).is_pty(), "STOP_PENDING_PRECONDITION: actual File entry must run before the original close event");
                    let context = history.read(cx).entry(terminal_id).expect("original execution retained").inline_context.as_ref().expect("original association");
                    assert_eq!(context.occurrence_id, None, "STOP_PENDING_FILE_NEW_EXECUTION: actual File Open cannot append a new occurrence to the stopped execution");
                    assert!(context.previous_occurrences.is_empty(), "STOP_PENDING_FILE_NEW_EXECUTION: stopped execution is never reused");
                    *pending_action_checked.lock() = true;
                });
            });
                assert!(
                    *pending_action_checked.lock(),
                    "the real deferred File dispatch and pending-state oracle must both execute"
                );
                let deadline = Instant::now() + Duration::from_secs(10);
                let (new_id, new_view) = loop {
                    cx.run_until_parked();
                    if let Some(created) = panel.read_with(&cx, |panel, _| {
                        panel
                            .terminals
                            .iter()
                            .find(|(id, _)| **id != terminal_id)
                            .map(|(id, terminal)| (*id, terminal.view.clone()))
                    }) {
                        break created;
                    }
                    assert!(
                        Instant::now() < deadline,
                        "pending Stop File Open must create a distinct canonical shell"
                    );
                    cx.executor().timer(Duration::from_millis(50)).await;
                };
                let new_terminal = new_view.read_with(&cx, |view, _| view.terminal().clone());
                let new_pid = new_terminal.read_with(&cx, |terminal, _| {
                    terminal
                        .pid_getter()
                        .expect("new real execution")
                        .fallback_pid()
                });
                println!(
                    "INLINE_REAL_PTY_OWNER process_id={} role=STOP_PENDING_NEW",
                    new_pid.as_u32()
                );
                assert_ne!(new_id, terminal_id);
                assert_ne!(new_pid, original_pid);
                assert_ne!(new_view.entity_id(), view.entity_id());
                panel
                    .update(&mut cx, |panel, cx| {
                        panel.request_stop_inline_terminal(new_id, cx)
                    })
                    .expect("new owned process cleanup");
                let deadline = Instant::now() + Duration::from_secs(10);
                while !*close_seen.lock()
                    || panel.read_with(&cx, |panel, _| panel.has_terminal(new_id))
                {
                    cx.run_until_parked();
                    assert!(
                        Instant::now() < deadline,
                        "both owned Stop-pending and newly created shells must exit"
                    );
                    cx.executor().timer(Duration::from_millis(50)).await;
                }
                history
                    .update(&mut cx, |history, cx| history.flush(cx))
                    .await
                    .expect("both canonical execution histories durable");
                assert!(history.read_with(&cx, |history, _| {
                    history
                        .entry(terminal_id)
                        .expect("original history")
                        .inline_context
                        .as_ref()
                        .expect("original")
                        .previous_occurrences
                        .is_empty()
                }));
            }
            FileOpenEligibilityCase::ReplacedHistoryOwner
            | FileOpenEligibilityCase::ClosedHistoryQueue => {
                let replaced = matches!(case, FileOpenEligibilityCase::ReplacedHistoryOwner);
                let replacement = if replaced {
                    cx.update(|_, cx| TerminalThreadMetadataStore::init_global(cx));
                    let replacement = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
                    replacement
                        .read_with(&cx, |store, _| store.reload_task())
                        .await;
                    assert_ne!(replacement.entity_id(), owner.entity_id());
                    Some(replacement)
                } else {
                    owner.read_with(&cx, |owner, _| owner.close_write_queue_for_test());
                    None
                };
                let replacement_before = replacement.as_ref().map(|replacement| {
                    replacement.read_with(&cx, |owner, _| owner.entry(terminal_id).cloned())
                });
                // show_error replaces the existing error entity at its type's stable ID.
                cx.run_until_parked();
                let error_notification_id =
                    workspace::notifications::NotificationId::unique::<anyhow::Error>();
                let previous_error_notification = workspace.read_with(&cx, |workspace, _| {
                    workspace
                        .message_notification_for_test(&error_notification_id)
                        .map(|notification| notification.entity_id())
                });
                cx.update(|_, cx| {
                    assert_eq!(AgentSettings::get_global(cx).terminal_init_command.as_deref(), Some(init_command), "actual File fixture must retain its authored init command after settings reload");
                    assert_eq!(TerminalSettings::get_global(cx).shell, task::Shell::Program("/bin/sh".into()));
                });
                cx.dispatch_action(NewTerminalThread);
                let deadline = Instant::now() + Duration::from_secs(10);
                loop {
                    cx.run_until_parked();
                    assert_eq!(
                        owner.read_with(&cx, |owner, _| owner.entry(terminal_id).cloned()),
                        Some(before_metadata.clone()),
                        "FILE_HISTORY_REFUSAL_NO_MUTATION: rejected actual File entry must preserve the exact original execution context"
                    );
                    if workspace.read_with(&cx, |workspace, _| {
                        workspace
                            .message_notification_for_test(&error_notification_id)
                            .is_some_and(|notification| {
                                previous_error_notification != Some(notification.entity_id())
                            })
                    }) {
                        break;
                    }
                    assert!(
                        Instant::now() < deadline,
                        "actual File refusal must surface through the canonical workspace error"
                    );
                    cx.executor().timer(Duration::from_millis(50)).await;
                }
                assert_eq!(
                    panel.read_with(&cx, |panel, _| panel.terminals.len()),
                    1,
                    "FILE_REFUSAL_PRESERVES_EXECUTION: history refusal cannot silently allocate an alternative shell"
                );
                assert_eq!(
                    panel.read_with(&cx, |panel, _| panel
                        .inline_terminal_view(terminal_id)
                        .expect("original retained view")
                        .entity_id()),
                    view.entity_id()
                );
                assert_eq!(
                    terminal.read_with(&cx, |terminal, _| terminal
                        .pid_getter()
                        .expect("original live PTY")
                        .fallback_pid()),
                    original_pid
                );
                assert!(
                    terminal.read_with(&cx, |terminal, _| terminal.can_reuse_interactive_process())
                );
                assert!(!*close_seen.lock());
                assert_eq!(
                    owner.read_with(&cx, |owner, _| owner.entry(terminal_id).cloned()),
                    Some(before_metadata.clone()),
                    "original canonical metadata is unchanged by File refusal"
                );
                assert_eq!(
                    parent.read_with(&cx, |parent, cx| parent.message_editor.read(cx).text(cx)),
                    "Aλ🦀B"
                );
                assert_eq!(
                    parent.update(&mut cx, |parent, cx| parent
                        .message_editor
                        .update(cx, |editor, cx| editor
                            .inline_terminal_positions_in_display_order(cx))
                        .expect("retained exact occurrence placement")
                        .len()),
                    1
                );
                assert_eq!(
                    cx.update(|_, cx| crate::draft_prompt_store::read_for_restore(thread_id, cx)),
                    Some(before_blocks.clone()),
                    "unavailable history cannot overwrite the durable original composition"
                );
                if replaced {
                    assert!(
                        parent
                            .read_with(&cx, |parent, cx| parent
                                .message_editor
                                .read(cx)
                                .draft_content_blocks_snapshot(cx))
                            .is_err(),
                        "replaced authority must fail closed while preserving the original resource widget and saved composition"
                    );
                    assert_eq!(
                        replacement.as_ref().map(|replacement| replacement
                            .read_with(&cx, |owner, _| owner.entry(terminal_id).cloned())),
                        replacement_before,
                        "replacement owner receives no command/history mutation"
                    );
                    assert!(
                        panel
                            .update(&mut cx, |panel, cx| panel
                                .request_stop_inline_terminal(terminal_id, cx))
                            .is_err(),
                        "stale panel history owner cannot authorize Stop"
                    );
                    terminal
                        .update(&mut cx, |terminal, _| {
                            terminal.request_stop_interactive_process()
                        })
                        .expect(
                            "explicit fixture cleanup through actual canonical execution owner",
                        );
                } else {
                    assert_eq!(
                        parent.read_with(&cx, |parent, cx| parent
                            .message_editor
                            .read(cx)
                            .draft_content_blocks_snapshot(cx)
                            .expect(
                                "read-only snapshot remains usable despite closed write queue"
                            )),
                        before_blocks
                    );
                    assert!(
                        owner
                            .update(&mut cx, |owner, cx| owner.flush(cx))
                            .await
                            .is_err(),
                        "closed queue cannot claim durable success"
                    );
                    panel
                        .update(&mut cx, |panel, cx| {
                            panel.request_stop_inline_terminal(terminal_id, cx)
                        })
                        .expect("owned closed-queue fixture Stop");
                }
                let deadline = Instant::now() + Duration::from_secs(10);
                while !*close_seen.lock() {
                    cx.run_until_parked();
                    assert!(
                        Instant::now() < deadline,
                        "owned refusal-fixture process must exit"
                    );
                    cx.executor().timer(Duration::from_millis(50)).await;
                }
                if replaced {
                    panel.update_in(&mut cx, |panel, window, cx| {
                        panel.close_terminal_without_activating_draft(terminal_id, window, cx)
                    });
                    assert_eq!(
                        replacement.as_ref().map(|replacement| replacement
                            .read_with(&cx, |owner, _| owner.entry(terminal_id).cloned())),
                        replacement_before,
                        "runtime-only cleanup preserves replacement history"
                    );
                }
                assert_eq!(
                    owner.read_with(&cx, |owner, _| owner
                        .persisted_entry_for_test(terminal_id)
                        .expect("original SQLite read")),
                    Some(before_metadata),
                    "original durable history is retained despite unavailable current admission"
                );
            }
        }
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    async fn assert_file_open_reuses_one_live_execution(
        fail_attachment: bool,
        cx: &mut TestAppContext,
    ) {
        use crate::collaborative_composer::CollaborativeComposerAdapter;

        type Registrations = (
            workspace::collaborative_composer::CollaborativeComposerRegistration,
            workspace::collaborative_timeline::CollaborativeTimelineRegistration,
        );

        fn mount_footer(
            panel: &Entity<AgentPanel>,
            registrations: &mut Option<Registrations>,
            cx: &mut VisualTestContext,
        ) {
            let workspace = panel.read_with(cx, |panel, _| {
                panel.workspace.upgrade().expect("canonical workspace")
            });
            let adapter = cx
                .update(|_, cx| {
                    CollaborativeComposerAdapter::from_agent_panel(panel, &workspace, cx)
                })
                .expect("canonical active-conversation adapter");
            let timeline = cx
                .update(|_, cx| {
                    crate::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                        panel, &workspace, cx,
                    )
                })
                .expect("canonical active-conversation timeline adapter");
            workspace.update(cx, |workspace, cx| {
                if let Some((composer, timeline)) = registrations.take() {
                    assert!(workspace.unregister_collaborative_composer_provider(composer, cx));
                    assert!(workspace.unregister_collaborative_timeline_provider(timeline, cx));
                }
                let composer = adapter
                    .register_in_workspace(workspace, cx)
                    .expect("canonical footer registration");
                let timeline = timeline
                    .register_in_workspace(workspace, cx)
                    .expect("canonical timeline registration");
                *registrations = Some((composer, timeline));
            });
            if workspace.read_with(cx, |workspace, _| workspace.workspace_presentation())
                != workspace::WorkspacePresentation::Collaborative
            {
                cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
                cx.run_until_parked();
            }
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.close_all_docks(window, cx);
                AgentPanel::reveal_conversation(workspace, true, window, cx);
            });
            cx.run_until_parked();
            cx.refresh().expect("mounted conversation footer");
            for _ in 0..2 {
                cx.update(|window, cx| window.simulate_next_frame(cx));
                cx.refresh().expect("settled conversation mounting frame");
                cx.run_until_parked();
            }
            assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER-FOOTER").is_some());
        }

        fn open_menu(cx: &mut VisualTestContext) {
            cx.refresh().expect("running-terminal footer");
            let footer = cx
                .debug_bounds("COLLABORATIVE-COMPOSER-FOOTER")
                .expect("mounted footer");
            let trigger = cx
                .debug_bounds("ICON-Terminal")
                .expect("actual terminal trigger");
            assert!(
                footer.contains(&trigger.center()),
                "trigger must be inside footer"
            );
            cx.simulate_click(trigger.center(), Modifiers::default());
            cx.run_until_parked();
            cx.refresh().expect("rendered running-terminal menu");
        }

        fn select_item(submenu: &'static str, item: &'static str, cx: &mut VisualTestContext) {
            let bounds = cx
                .debug_bounds(submenu)
                .expect("actual scoped terminal submenu");
            cx.simulate_click(bounds.center(), Modifiers::default());
            cx.run_until_parked();
            cx.refresh().expect("rendered per-terminal submenu");
            let bounds = cx.debug_bounds(item).expect("actual per-terminal action");
            if item == "MENU_ITEM-Stop Process" {
                cx.dispatch_action(menu::SelectLast);
                cx.dispatch_action(menu::Confirm);
            } else {
                cx.simulate_click(bounds.center(), Modifiers::default());
            }
            cx.run_until_parked();
            cx.refresh().expect("selected terminal action");
        }

        async fn wait_for_output(
            terminal: &Entity<terminal::Terminal>,
            marker: &str,
            cx: &mut VisualTestContext,
        ) {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                cx.run_until_parked();
                if terminal.read_with(cx, |terminal, _| {
                    terminal
                        .get_content()
                        .lines()
                        .any(|line| line.trim() == marker)
                }) {
                    break;
                }
                if Instant::now() >= deadline {
                    let (process_id, is_pty, keyboard_input_sent, content) =
                        terminal.read_with(cx, |terminal, _| {
                            (
                                terminal.pid(),
                                terminal.is_pty(),
                                terminal.keyboard_input_sent(),
                                terminal.last_n_non_empty_lines(20),
                            )
                        });
                    let input_log = terminal.update(cx, |terminal, _| terminal.take_input_log());
                    panic!(
                        "real PTY output missing: {marker}; process_id={process_id:?}, is_pty={is_pty}, keyboard_input_sent={keyboard_input_sent}, content={content:?}, input_log={input_log:?}"
                    );
                }
                cx.executor().timer(Duration::from_millis(50)).await;
            }
        }

        let (panel, mut cx) = setup_visible_panel(cx).await;
        cx.executor().allow_parking();
        cx.simulate_resize(size(px(1100.), px(750.)));
        let init_command = "printf '\\033[31mREUSE_FIRST\\033[0m\\n'";
        let user_settings = json!({
            "terminal": { "shell": { "program": "/bin/sh" }, "working_directory": "always_home" },
            "agent": { "terminal_init_command": init_command }
        })
        .to_string();
        let fs = panel.read_with(&cx, |panel, cx| panel.project.read(cx).fs().clone());
        fs.atomic_write(paths::settings_file().clone(), user_settings.clone())
            .await
            .expect("durable authored fixture settings");
        cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            SettingsStore::update_global(cx, |store, cx| {
                store
                    .set_user_settings(&user_settings, cx)
                    .result()
                    .expect("canonical shell settings")
            });
        });
        open_thread_with_connection(&panel, StubAgentConnection::new(), &mut cx);
        crate::test_support::type_draft_prompt(&panel, "Aλ🦀B", &mut cx);
        let mut registrations = None;
        mount_footer(&panel, &mut registrations, &mut cx);
        let parent = panel.read_with(&cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("canonical conversation")
        });
        parent.update_in(&mut cx, |parent, window, cx| {
            parent.message_editor.update(cx, |editor, cx| {
                editor.set_cursor_offset("Aλ".len(), window, cx)
            })
        });
        cx.update(|_, cx| {
            assert_eq!(
                AgentSettings::get_global(cx)
                    .terminal_init_command
                    .as_deref(),
                Some(init_command),
                "actual File fixture must retain its authored init command after settings reload"
            );
            assert_eq!(
                TerminalSettings::get_global(cx).shell,
                task::Shell::Program("/bin/sh".into())
            );
        });
        cx.dispatch_action(NewTerminalThread);
        let deadline = Instant::now() + Duration::from_secs(10);
        let (terminal_id, view) = loop {
            cx.run_until_parked();
            let live = panel.read_with(&cx, |panel, _| {
                panel
                    .terminals
                    .iter()
                    .map(|(id, terminal)| (*id, terminal.view.clone()))
                    .collect::<Vec<_>>()
            });
            assert!(
                live.len() <= 1,
                "initial File entry allocates exactly one execution"
            );
            if let Some(live) = live.into_iter().next() {
                break live;
            }
            assert!(
                Instant::now() < deadline,
                "real File entry must allocate a PTY"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        };
        let terminal = view.read_with(&cx, |view, _| view.terminal().clone());
        wait_for_output(&terminal, "REUSE_FIRST", &mut cx).await;
        let original_pid = terminal.read_with(&cx, |terminal, _| {
            terminal.pid_getter().expect("real PTY").fallback_pid()
        });
        println!(
            "INLINE_REAL_PTY_OWNER process_id={} role=REUSE_ORIGINAL",
            original_pid.as_u32()
        );
        view.update(&mut cx, |view, cx| {
            view.set_custom_title(Some("REUSE_EXECUTION".into()), cx)
        });
        panel.update(&mut cx, |panel, cx| {
            panel.persist_terminal_metadata(terminal_id, cx)
        });
        let history = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        let first_context = history.read_with(&cx, |history, _| {
            history
                .entry(terminal_id)
                .expect("first canonical row")
                .inline_context
                .clone()
                .expect("first context")
        });
        assert_eq!(first_context.occurrence_id, None);
        parent.update_in(&mut cx, |parent, window, cx| {
            parent.message_editor.update(cx, |editor, cx| {
                editor.set_cursor_offset("Aλ🦀".len(), window, cx);
                editor.focus_handle(cx).focus(window, cx);
            })
        });
        if fail_attachment {
            let workspace = panel.read_with(&cx, |panel, _| {
                panel.workspace.upgrade().expect("canonical workspace")
            });
            let prior_notifications =
                workspace.read_with(&cx, |workspace, _| workspace.notification_ids());
            let mut invalid_source = panel.read_with(&cx, |panel, cx| {
                panel
                    .inline_terminal_snapshot_metadata(terminal_id, cx)
                    .expect("genuine accepted execution snapshot")
                    .1
            });
            invalid_source.terminal_id = TerminalId::new();
            invalid_source.title = "unrelated retained source".into();
            let invalid_context = invalid_source
                .inline_context
                .as_mut()
                .expect("genuine snapshot context");
            invalid_context.presentation_order = Some(usize::MAX);
            invalid_context.draft_position = Some(
                crate::terminal_thread_metadata_store::InlineTerminalDraftPosition::new(
                    "wrong retained source",
                    0,
                )
                .expect("another valid source"),
            );
            let invalid_id = invalid_source.terminal_id;
            history
                .update(&mut cx, |history, cx| {
                    history.save_inline(invalid_source, cx)
                })
                .expect("admit retained fixture before view/source reconciliation");
            cx.update(|_, cx| {
                assert_eq!(AgentSettings::get_global(cx).terminal_init_command.as_deref(), Some(init_command), "actual File fixture must retain its authored init command after settings reload");
                assert_eq!(TerminalSettings::get_global(cx).shell, task::Shell::Program("/bin/sh".into()));
            });
            cx.dispatch_action(NewTerminalThread);
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                cx.run_until_parked();
                if workspace.read_with(&cx, |workspace, _| {
                    workspace
                        .notification_ids()
                        .iter()
                        .any(|id| !prior_notifications.contains(id))
                }) {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "actual File failure must surface through the canonical workspace notification"
                );
                cx.executor().timer(Duration::from_millis(50)).await;
            }
            assert_eq!(
                panel.read_with(&cx, |panel, _| panel.terminals.len()),
                1,
                "refused reuse cannot allocate or stop a different execution"
            );
            assert_eq!(
                terminal.read_with(&cx, |terminal, _| terminal
                    .pid_getter()
                    .expect("preserved PTY")
                    .fallback_pid()),
                original_pid
            );
            assert!(
                terminal.read_with(&cx, |terminal, _| terminal.can_reuse_interactive_process()),
                "partial presentation failure must never stop the reused execution"
            );
            let retained = history.read_with(&cx, |history, _| {
                history
                    .entry(terminal_id)
                    .expect("retained original")
                    .inline_context
                    .clone()
                    .expect("original context")
            });
            assert_eq!(retained.occurrence_id, None);
            assert!(
                retained.previous_occurrences.is_empty(),
                "exact own aggregate rollback removes the attempted new occurrence"
            );
            assert_eq!(retained.draft_position, first_context.draft_position);
            let blocks = parent.update(&mut cx, |parent, cx| {
        assert_eq!(parent.message_editor.update(cx, |editor, cx| editor.inline_terminal_positions_in_display_order(cx)).expect("valid original widget").len(), 1, "FILE_REUSE_ROLLBACK_NO_ORPHAN: failed File attachment leaves no unresolvable new occurrence widget");
        parent.message_editor.read(cx).draft_content_blocks_snapshot(cx).expect("original composition remains usable after refusal")
    });
            let references = blocks.iter().filter_map(|block| crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(block).expect("strict references")).collect::<Vec<_>>();
            assert_eq!(references.len(), 1);
            assert_eq!(references[0].occurrence_id, None);
            history.update(&mut cx, |history, cx| history.delete(invalid_id, cx));
            parent
                .update_in(&mut cx, |parent, window, cx| {
                    parent.sync_inline_terminal_history(window, cx)
                })
                .expect("removing only disposable invalid fixture restores normal reconciliation");
            panel
                .update(&mut cx, |panel, cx| {
                    panel.request_stop_inline_terminal(terminal_id, cx)
                })
                .expect("owned refusal-fixture PTY cleanup");
            let deadline = Instant::now() + Duration::from_secs(10);
            while panel.read_with(&cx, |panel, _| panel.has_terminal(terminal_id)) {
                cx.run_until_parked();
                assert!(
                    Instant::now() < deadline,
                    "owned refusal-fixture shell must exit"
                );
                cx.executor().timer(Duration::from_millis(50)).await;
            }
            history
                .update(&mut cx, |history, cx| history.flush(cx))
                .await
                .expect("durable refusal preserves canonical original history");
            return;
        }
        cx.update(|_, cx| {
            assert_eq!(
                AgentSettings::get_global(cx)
                    .terminal_init_command
                    .as_deref(),
                Some(init_command),
                "actual File fixture must retain its authored init command after settings reload"
            );
            assert_eq!(
                TerminalSettings::get_global(cx).shell,
                task::Shell::Program("/bin/sh".into())
            );
        });
        cx.dispatch_action(NewTerminalThread);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            cx.run_until_parked();
            assert_eq!(
                panel.read_with(&cx, |panel, _| panel.terminals.len()),
                1,
                "LIVE_REUSE_SAME_EXECUTION: repeated File Open must not allocate a second shell"
            );
            if history.read_with(&cx, |history, _| {
                history
                    .entry(terminal_id)
                    .and_then(|metadata| metadata.inline_context.as_ref())
                    .is_some_and(|context| context.previous_occurrences.len() == 1)
            }) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "second actual File entry must append an occurrence"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        assert_eq!(
            panel.read_with(&cx, |panel, _| panel.terminals.len()),
            1,
            "LIVE_REUSE_SAME_EXECUTION: repeated File Open must not allocate a second shell"
        );
        let reused = panel.read_with(&cx, |panel, _| {
            panel
                .inline_terminal_view(terminal_id)
                .expect("same canonical view")
        });
        assert_eq!(reused.entity_id(), view.entity_id());
        assert_eq!(
            terminal.read_with(&cx, |terminal, _| terminal
                .pid_getter()
                .expect("same PTY")
                .fallback_pid()),
            original_pid
        );
        let admitted_current = history.read_with(&cx, |history, _| {
            history
                .entry(terminal_id)
                .expect("admitted reused execution")
                .inline_context
                .clone()
                .expect("admitted current occurrence")
        });
        assert!(
            admitted_current.occurrence_id.is_some(),
            "the repeat File mount must name its distinct current occurrence"
        );
        let current_position = admitted_current
            .draft_position
            .as_ref()
            .expect("current occurrence character receipt");
        let message_editor = parent.read_with(&cx, |parent, _| parent.message_editor.clone());
        let positions = message_editor
            .update(&mut cx, |editor, cx| {
                editor.inline_terminal_positions_in_display_order(cx)
            })
            .expect("actual mounted occurrence positions");
        let matching_blocks = positions
            .iter()
            .filter(|(_, position)| position == current_position)
            .map(|(block_id, _)| *block_id)
            .collect::<Vec<_>>();
        assert_eq!(
            matching_blocks.len(),
            1,
            "the actual new live occurrence has one exact character block"
        );
        let live_block_id = *matching_blocks
            .first()
            .expect("exact live occurrence block");
        let editor = message_editor.read_with(&cx, |editor, _| editor.editor().clone());
        assert!(
            !view.read_with(&cx, |view, _| view.is_read_only()),
            "the reused current occurrence mounts the canonical interactive view"
        );
        // The real attachment's two-frame measurement callbacks are not dispatcher tasks.
        cx.refresh().expect("render the reused live occurrence");
        for _ in 0..2 {
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
            });
            cx.refresh()
                .expect("render corrected occurrence measurement");
            cx.run_until_parked();
            editor
                .update(&mut cx, |editor, cx| {
                    editor.row_for_block(live_block_id, cx)
                })
                .expect("the actual reused live block has a display row after prepaint");
        }
        let live_control_bounds = cx
            .debug_bounds("inline-terminal-release-focus")
            .or_else(|| cx.debug_bounds("inline-terminal-focus"))
            .expect("the only interactive occurrence must have a rendered focus control");
        assert!(
            live_control_bounds.size.width > px(0.) && live_control_bounds.size.height > px(0.),
            "the reused live occurrence focus control has nonzero mounted bounds"
        );
        view.update_in(&mut cx, |view, window, cx| {
            assert!(
                view.focus_handle(cx).contains_focused(window, cx),
                "repeat File entry focuses the same execution"
            )
        });
        terminal.update(&mut cx, |terminal, _| terminal.take_input_log());
        cx.simulate_keystrokes("x");
        cx.run_until_parked();
        let keyboard_input = terminal
            .update(&mut cx, |terminal, _| terminal.take_input_log())
            .concat();
        assert_eq!(
            keyboard_input, b"x",
            "repeat File focus must route a real keystroke to the original canonical PTY without refocusing"
        );
        assert_eq!(
            terminal.read_with(&cx, |terminal, _| terminal
                .pid_getter()
                .expect("same real keyboard recipient")
                .fallback_pid()),
            original_pid,
            "the mounted keyboard recipient remains the original execution"
        );
        assert_eq!(
            message_editor.read_with(&cx, |editor, cx| editor.text(cx)),
            "Aλ🦀B",
            "real terminal keyboard routing must not modify the integrated draft"
        );
        cx.simulate_keystrokes("backspace");
        cx.run_until_parked();
        let current = history.read_with(&cx, |history, _| {
            history
                .entry(terminal_id)
                .expect("aggregate")
                .inline_context
                .clone()
                .expect("aggregate")
        });
        let frozen = current
            .occurrence(None)
            .expect("legacy receipt still names first occurrence")
            .clone();
        assert!(current.occurrence_id.is_some());
        assert_eq!(frozen.occurrence_id, None);
        assert!(frozen.previous_occurrences.is_empty());
        assert_eq!(
            frozen
                .draft_position
                .as_ref()
                .expect("first position")
                .byte_offset_for_source("Aλ🦀B")
                .expect("first Unicode source"),
            "Aλ".len()
        );
        assert_eq!(
            current
                .draft_position
                .as_ref()
                .expect("second position")
                .byte_offset_for_source("Aλ🦀B")
                .expect("second Unicode source"),
            "Aλ🦀".len()
        );
        assert!(frozen.snapshot.is_some());
        assert!(frozen.output.contains("REUSE_FIRST"));
        cx.simulate_input("printf 'REUSE_SECOND\\n'\n");
        wait_for_output(&terminal, "REUSE_SECOND", &mut cx).await;
        let blocks = parent
            .update(&mut cx, |parent, cx| {
                parent
                    .message_editor
                    .update(cx, |editor, cx| editor.draft_content_blocks_snapshot(cx))
            })
            .expect("real composer source and resources");
        let references = blocks
            .iter()
            .filter_map(|content| {
                crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(
                    content,
                )
                .expect("genuine resources")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            references.len(),
            2,
            "both occurrences remain in integrated ACP composition"
        );
        assert!(
            references
                .iter()
                .all(|reference| reference.terminal_id == terminal_id)
        );
        assert_eq!(references[0].occurrence_id, None);
        assert_eq!(references[1].occurrence_id, current.occurrence_id);
        assert_eq!(
            history.read_with(&cx, |history, _| history
                .entry(terminal_id)
                .expect("aggregate")
                .inline_context
                .as_ref()
                .expect("aggregate")
                .occurrence(None)
                .expect("first")
                .clone()),
            frozen,
            "later real output cannot rewrite frozen first grid or source"
        );
        parent.update(&mut cx, |parent, cx| {
            assert_eq!(
                parent
                    .message_editor
                    .update(cx, |editor, cx| editor
                        .inline_terminal_positions_in_display_order(cx))
                    .expect("mounted canonical blocks")
                    .len(),
                2
            )
        });
        open_menu(&mut cx);
        select_item(
            "MENU_SUBMENU-REUSE_EXECUTION",
            "MENU_ITEM-Stop Process",
            &mut cx,
        );
        assert!(
            !terminal.read_with(&cx, |terminal, _| terminal.can_reuse_interactive_process()),
            "selected Stop makes exact execution ineligible"
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while panel.read_with(&cx, |panel, _| panel.has_terminal(terminal_id)) {
            cx.run_until_parked();
            assert!(
                Instant::now() < deadline,
                "selected reused execution must close"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        assert_eq!(
            history.read_with(&cx, |history, _| history
                .entry(terminal_id)
                .expect("retained execution")
                .inline_context
                .as_ref()
                .expect("aggregate")
                .occurrence(None)
                .expect("frozen")
                .snapshot
                .clone()),
            frozen.snapshot
        );
        parent.update_in(&mut cx, |parent, window, cx| {
            parent.message_editor.focus_handle(cx).focus(window, cx)
        });
        cx.update(|_, cx| {
            assert_eq!(
                AgentSettings::get_global(cx)
                    .terminal_init_command
                    .as_deref(),
                Some(init_command),
                "actual File fixture must retain its authored init command after settings reload"
            );
            assert_eq!(
                TerminalSettings::get_global(cx).shell,
                task::Shell::Program("/bin/sh".into())
            );
        });
        cx.dispatch_action(NewTerminalThread);
        let deadline = Instant::now() + Duration::from_secs(10);
        let (new_id, new_view) = loop {
            cx.run_until_parked();
            if let Some(created) = panel.read_with(&cx, |panel, _| {
                panel
                    .terminals
                    .iter()
                    .find(|(id, _)| **id != terminal_id)
                    .map(|(id, terminal)| (*id, terminal.view.clone()))
            }) {
                break created;
            }
            assert!(
                Instant::now() < deadline,
                "File Open after Stop creates a fresh owned shell"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        };
        let new_terminal = new_view.read_with(&cx, |view, _| view.terminal().clone());
        let new_pid = new_terminal.read_with(&cx, |terminal, _| {
            terminal.pid_getter().expect("new real PTY").fallback_pid()
        });
        println!(
            "INLINE_REAL_PTY_OWNER process_id={} role=REUSE_AFTER_STOP",
            new_pid.as_u32()
        );
        assert_ne!(new_id, terminal_id);
        assert_ne!(new_pid, original_pid);
        panel
            .update(&mut cx, |panel, cx| {
                panel.request_stop_inline_terminal(new_id, cx)
            })
            .expect("owned fresh-shell cleanup");
        let deadline = Instant::now() + Duration::from_secs(10);
        while panel.read_with(&cx, |panel, _| panel.has_terminal(new_id)) {
            cx.run_until_parked();
            assert!(
                Instant::now() < deadline,
                "owned cleanup closes fresh shell"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        history
            .update(&mut cx, |history, cx| history.flush(cx))
            .await
            .expect("durable aggregate and fresh execution history");
        assert_eq!(
            history.read_with(&cx, |history, _| history
                .persisted_entry_for_test(terminal_id)
                .expect("actual SQLite row")
                .expect("retained row")
                .inline_context
                .as_ref()
                .expect("aggregate")
                .occurrence(None)
                .expect("first")
                .snapshot
                .clone()),
            frozen.snapshot
        );
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_real_pty_submission_keeps_context_and_historical_placement(
        cx: &mut TestAppContext,
    ) {
        inline_real_pty_submission_keeps_context_and_historical_placement(false, false, cx).await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_submitted_terminal_focus_does_not_edit_its_message(
        cx: &mut TestAppContext,
    ) {
        inline_real_pty_submission_keeps_context_and_historical_placement(true, false, cx).await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_accepted_history_disclosure_reflows_after_sqlite_reload(
        cx: &mut TestAppContext,
    ) {
        inline_real_pty_submission_keeps_context_and_historical_placement(true, true, cx).await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    async fn inline_real_pty_submission_keeps_context_and_historical_placement(
        supports_client_user_message_ids: bool,
        verify_history_reflow: bool,
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        cx.simulate_resize(size(px(900.), px(600.)));
        cx.executor().allow_parking();
        let connection = StubAgentConnection::new()
            .with_supports_client_user_message_ids(supports_client_user_message_ids);
        crate::test_support::open_draft_with_connection(&panel, connection.clone(), &mut cx);
        crate::test_support::type_draft_prompt(&panel, "  Aλ", &mut cx);
        cx.run_until_parked();
        cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.terminal_init_command = Some(
                "printf 'INLINE_CONTEXT_ORACLE\\n'; index=1; while [ \"$index\" -le 40 ]; do printf 'INLINE_ROW_%02d\\n' \"$index\"; index=$((index+1)); done".into(),
            );
            AgentSettings::override_global(settings, cx);
            let mut settings = TerminalSettings::get_global(cx).clone();
            settings.shell = task::Shell::Program("/bin/sh".into());
            settings.working_directory = settings::WorkingDirectory::AlwaysHome;
            TerminalSettings::override_global(settings, cx);
        });
        let parent = panel.read_with(&cx, |panel, cx| {
            panel.active_thread_view(cx).expect("parent")
        });
        let context = parent
            .read_with(&cx, |parent, cx| parent.inline_terminal_context(cx))
            .expect("canonical source-bound conversation context");
        let session_id = context.session_id.clone().expect("parent session");
        let (terminal_id, terminal_view) = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.spawn_inline_terminal(context, None, window, cx)
            })
            .await
            .expect("actual canonical PTY");
        parent
            .update_in(&mut cx, |parent, window, cx| {
                parent.attach_inline_terminal(terminal_id, terminal_view.clone(), window, cx)
            })
            .expect("mount real PTY inside draft");
        let terminal = terminal_view.read_with(&cx, |view, _| view.terminal().clone());
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            cx.run_until_parked();
            if terminal
                .read_with(&cx, |terminal, _| terminal.get_content())
                .contains("INLINE_ROW_40")
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "real shell must produce context before submission"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        let process_id = terminal
            .read_with(&cx, |terminal, _| {
                terminal.pid_getter().map(|getter| getter.fallback_pid())
            })
            .expect("owned real PTY process identifier");
        println!("INLINE_REAL_PTY_OWNER process_id={}", process_id.as_u32());
        terminal_view.update_in(&mut cx, |view, window, cx| {
            view.focus_handle(cx).focus(window, cx);
            assert_eq!(view.terminal().entity_id(), terminal.entity_id());
            assert!(!view.embedded_history_expanded());
            assert!(
                view.embedded_hidden_lines(window, cx) > 0,
                "history-disclosure precondition: viewport={:?}, used_lines={}",
                window.viewport_size(),
                terminal.read(cx).used_lines(),
            );
        });
        cx.refresh()
            .expect("mounted inline terminal focus preflight");
        let history_bounds = cx
            .debug_bounds("inline-terminal-history-disclosure")
            .expect("earlier real output has a mounted inline history control");
        cx.simulate_click(history_bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        terminal_view.update_in(&mut cx, |view, window, cx| {
            assert!(view.embedded_history_expanded());
            assert!(view.focus_handle(cx).contains_focused(window, cx));
            assert_eq!(view.terminal().entity_id(), terminal.entity_id());
        });
        cx.refresh()
            .expect("expanded history remains mounted inline");
        let history_bounds = cx
            .debug_bounds("inline-terminal-history-disclosure")
            .expect("Show Less control");
        cx.simulate_click(history_bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        terminal_view.update_in(&mut cx, |view, window, cx| {
            assert!(!view.embedded_history_expanded());
            assert!(view.focus_handle(cx).contains_focused(window, cx));
            assert_eq!(view.terminal().entity_id(), terminal.entity_id());
        });
        cx.simulate_input("printf 'INLINE_REENTRY_ORACLE\\n'\n");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            cx.run_until_parked();
            if terminal.read_with(&cx, |terminal, _| {
                terminal
                    .get_content()
                    .lines()
                    .any(|line| line.trim() == "INLINE_REENTRY_ORACLE")
            }) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Show Less must preserve real PTY input focus"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        cx.refresh()
            .expect("contracted history preserves focus controls");
        let release_bounds = cx
            .debug_bounds("inline-terminal-release-focus")
            .expect("rendered inline Escape control");
        cx.simulate_click(release_bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        terminal_view.update_in(&mut cx, |view, window, cx| {
            assert_eq!(view.terminal().entity_id(), terminal.entity_id());
            assert!(
                !view.is_read_only(),
                "Escape must remain distinct from Background"
            );
            assert!(!view.focus_handle(cx).contains_focused(window, cx));
        });
        cx.refresh().expect("mounted inline reentry control");
        let focus_bounds = cx
            .debug_bounds("inline-terminal-focus")
            .expect("rendered inline return-to-terminal control");
        cx.simulate_click(focus_bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        terminal_view.update_in(&mut cx, |view, window, cx| {
            assert!(view.focus_handle(cx).contains_focused(window, cx));
            assert_eq!(view.terminal().entity_id(), terminal.entity_id());
        });
        cx.refresh().expect("mounted inline Background control");
        let background_bounds = cx
            .debug_bounds("inline-terminal-background")
            .expect("rendered inline Background control");
        cx.simulate_click(background_bounds.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        terminal_view.read_with(&cx, |view, _| {
            assert!(view.is_read_only());
            assert_eq!(view.terminal().entity_id(), terminal.entity_id());
        });
        parent.update_in(&mut cx, |parent, window, cx| {
            parent.message_editor.update(cx, |editor, cx| {
                assert!(editor.focus_handle(cx).contains_focused(window, cx));
                editor
                    .editor()
                    .update(cx, |editor, cx| editor.insert("🦀B  ", window, cx));
            });
        });
        let editor = parent.read_with(&cx, |parent, _| parent.message_editor.clone());
        let prompt = editor
            .update(&mut cx, |editor, cx| editor.contents(false, cx))
            .await
            .expect("resolve actual terminal context")
            .0;
        assert!(
            matches!(prompt.first(), Some(acp::ContentBlock::Text(text)) if text.text == "  Aλ")
        );
        assert!(
            matches!(prompt.last(), Some(acp::ContentBlock::Text(text)) if text.text == "🦀B  ")
        );
        let resource = prompt
            .iter()
            .find_map(|block| {
                crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(
                    block,
                )
                .expect("valid typed context")
                .map(|reference| (reference, block))
            })
            .expect("actual PTY context must reach ACP");
        assert_eq!(resource.0.terminal_id, terminal_id);
        assert_eq!(
            resource
                .0
                .position
                .byte_offset_for_source("  Aλ🦀B  ")
                .expect("exact Unicode source"),
            5
        );
        assert!(matches!(resource.1, acp::ContentBlock::Resource(resource)
            if matches!(&resource.resource, acp::EmbeddedResourceResource::TextResourceContents(text)
                if text.text.contains("INLINE_CONTEXT_ORACLE"))));
        let thread_id = parent.read_with(&cx, |parent, _| parent.root_thread_id);
        editor.update(&mut cx, |editor, cx| {
            assert_eq!(
                editor
                    .draft_content_blocks_snapshot(cx)
                    .expect("actual PTY snapshot"),
                prompt
            );
            editor.set_inline_terminal_scope(ThreadId::new(), session_id.clone());
            assert!(
                editor.draft_content_blocks_snapshot(cx).is_err(),
                "foreign scope must fail instead of dropping actual terminal content"
            );
            assert_eq!(editor.text(cx), "  Aλ🦀B  ");
            editor.set_inline_terminal_scope(thread_id, session_id.clone());
        });
        parent.update_in(&mut cx, |parent, window, cx| {
            parent.send_impl(editor.clone(), window, cx)
        });
        cx.run_until_parked();
        assert!(connection.has_pending_turn(&session_id));
        let received = connection.received_prompts();
        assert_eq!(received.len(), 1);
        assert_eq!(received.first().expect("actual ACP prompt").prompt, prompt);
        let history = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        let submitted = history
            .read_with(&cx, |history, _| history.entry(terminal_id).cloned())
            .expect("retained canonical history");
        let context = submitted
            .inline_context
            .as_ref()
            .expect("submitted association");
        assert!(!context.is_draft);
        assert!(
            terminal_view.read_with(&cx, |view, _| view.is_read_only()),
            "submitted history cannot accept input even while its canonical process remains alive"
        );
        assert_eq!(
            context.message_client_id.is_some(),
            supports_client_user_message_ids,
            "retained history must match the connection's real client-ID capability"
        );
        let historical_editor = parent.read_with(&cx, |parent, cx| {
            let state = parent.entry_view_state.read(cx);
            state
                .entry(context.preceding_entry_count)
                .and_then(|entry| entry.message_editor())
                .cloned()
                .expect("actual canonical submitted user editor")
        });
        assert_ne!(historical_editor.entity_id(), editor.entity_id());
        assert_eq!(
            historical_editor.read_with(&cx, |editor, cx| editor.text(cx)),
            "  Aλ🦀B  "
        );
        if supports_client_user_message_ids {
            editor.update_in(&mut cx, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx);
            });
            cx.run_until_parked();
            terminal_view.update_in(&mut cx, |view, window, cx| {
                view.focus_handle(cx).focus(window, cx);
            });
            cx.run_until_parked();
            cx.refresh().expect("mounted submitted terminal focus");
            terminal_view.update_in(&mut cx, |view, window, cx| {
                assert!(view.is_read_only());
                assert!(view.focus_handle(cx).contains_focused(window, cx));
            });
            assert!(parent.read_with(&cx, |parent, _| parent.editing_message.is_none()));
            editor.update_in(&mut cx, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx);
            });
            cx.run_until_parked();
            parent.update_in(&mut cx, |parent, window, cx| {
                parent
                    .focus_after_inline_terminal(terminal_id, window, cx)
                    .expect("canonical submitted terminal return to conversation");
            });
            cx.run_until_parked();
            cx.refresh()
                .expect("mounted direct historical editor focus");
            historical_editor.update_in(&mut cx, |editor, window, cx| {
                assert!(
                    editor.focus_handle(cx).is_focused(window),
                    "actual historical editor must own direct focus: current={:?}, expected={:?}",
                    window.focused(cx),
                    editor.focus_handle(cx),
                );
            });
            parent.read_with(&cx, |parent, cx| {
                assert!(parent.thread.read(cx).supports_truncate(cx));
                assert!(parent.parent_session_id.is_none());
                assert!(
                    parent
                        .thread
                        .read(cx)
                        .entries()
                        .get(context.preceding_entry_count)
                        .and_then(|entry| entry.user_message())
                        .expect("actual submitted user message")
                        .client_id
                        .is_some()
                );
            });
            assert_eq!(
                parent.read_with(&cx, |parent, _| parent.editing_message),
                Some(context.preceding_entry_count),
                "direct message focus must still enter historical editing"
            );
            editor.update_in(&mut cx, |editor, window, cx| {
                editor.focus_handle(cx).focus(window, cx);
            });
            cx.run_until_parked();
            assert!(parent.read_with(&cx, |parent, _| parent.editing_message.is_none()));
            parent.update_in(&mut cx, |parent, window, cx| {
                parent
                    .focus_after_inline_terminal(terminal_id, window, cx)
                    .expect("unchanged history cancellation retains inline placement");
                assert!(historical_editor.focus_handle(cx).is_focused(window));
            });
            cx.run_until_parked();
            historical_editor.update_in(&mut cx, |editor, window, cx| {
                editor.insert_text("UNSENT_HISTORY_EDIT", window, cx);
            });
            assert_ne!(
                historical_editor.read_with(&cx, |editor, cx| editor.text(cx)),
                "  Aλ🦀B  "
            );
            parent.update_in(&mut cx, |parent, window, cx| {
                parent.cancel_editing(&Default::default(), window, cx);
            });
            cx.run_until_parked();
            assert!(parent.read_with(&cx, |parent, _| parent.editing_message.is_none()));
            assert_eq!(
                historical_editor.read_with(&cx, |editor, cx| editor.text(cx)),
                "  Aλ🦀B  "
            );
            parent.update_in(&mut cx, |parent, window, cx| {
                parent
                    .focus_after_inline_terminal(terminal_id, window, cx)
                    .expect("edited history cancellation restores inline placement");
                assert!(historical_editor.focus_handle(cx).is_focused(window));
            });
            assert_eq!(
                terminal_view.read_with(&cx, |view, _| view.terminal().entity_id()),
                terminal.entity_id(),
                "history cancellation must reuse the canonical PTY"
            );
        }
        parent.update_in(&mut cx, |parent, window, cx| {
            parent
                .focus_after_inline_terminal(terminal_id, window, cx)
                .expect("submitted terminal reentry");
            assert!(historical_editor.focus_handle(cx).is_focused(window));
        });
        if verify_history_reflow {
            cx.update(|_, cx| {
                connection.send_update(
                    session_id.clone(),
                    acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                        "ACCEPTED_HISTORY_REFLOW_REPLY".into(),
                    )),
                    cx,
                );
            });
        }
        connection.end_turn(session_id, acp::StopReason::EndTurn);

        editor.update_in(&mut cx, |editor, window, cx| {
            editor.editor().update(cx, |editor, cx| {
                editor.insert("NEXT_DRAFT", window, cx);
            })
        });
        if verify_history_reflow {
            assert_accepted_history_reflow(
                &panel,
                &parent,
                &historical_editor,
                terminal_id,
                &mut cx,
            );
        }
        panel
            .update(&mut cx, |panel, cx| {
                panel.request_stop_inline_terminal(terminal_id, cx)
            })
            .expect("targeted canonical stop");
        cx.condition(&panel, |panel, _| !panel.has_terminal(terminal_id))
            .await;
        history
            .update(&mut cx, |history, cx| history.flush(cx))
            .await
            .expect("durable SQLite output");
        history.update(&mut cx, |history, cx| history.reload_for_test(cx));
        history
            .read_with(&cx, |history, _| history.reload_task())
            .await;
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(&cx, |editor, cx| editor.text(cx)),
            "NEXT_DRAFT"
        );
        parent.update_in(&mut cx, |parent, window, cx| {
            parent
                .sync_inline_terminal_history(window, cx)
                .expect("restore canonical history without respawn");
            parent
                .focus_after_inline_terminal(terminal_id, window, cx)
                .expect("retained historical terminal");
            assert!(historical_editor.focus_handle(cx).is_focused(window));
        });
        if verify_history_reflow {
            assert!(panel.read_with(&cx, |panel, cx| panel.terminals(cx).is_empty()));
            assert_accepted_history_reflow(
                &panel,
                &parent,
                &historical_editor,
                terminal_id,
                &mut cx,
            );
        }
        assert!(history.read_with(&cx, |history, _| {
            history
                .entry(terminal_id)
                .and_then(|metadata| metadata.inline_context.as_ref())
                .is_some_and(|context| {
                    !context.is_draft && context.output.contains("INLINE_CONTEXT_ORACLE")
                })
        }));
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    fn assert_accepted_history_reflow(
        panel: &Entity<AgentPanel>,
        parent: &Entity<ThreadView>,
        historical_editor: &Entity<crate::message_editor::MessageEditor>,
        terminal_id: TerminalId,
        cx: &mut VisualTestContext,
    ) {
        use crate::collaborative_composer::CollaborativeComposerAdapter;
        let workspace =
            panel.read_with(cx, |panel, _| panel.workspace.upgrade().expect("workspace"));
        let adapter = cx.update(|_, cx| {
            CollaborativeComposerAdapter::from_agent_panel(panel, &workspace, cx)
                .expect("canonical accepted conversation adapter")
        });
        let timeline = cx.update(|_, cx| {
            crate::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                panel, &workspace, cx,
            )
            .expect("canonical accepted conversation timeline")
        });
        let registrations = workspace.update(cx, |workspace, cx| {
            (
                adapter
                    .register_in_workspace(workspace, cx)
                    .expect("canonical composer registration"),
                timeline
                    .register_in_workspace(workspace, cx)
                    .expect("canonical timeline registration"),
            )
        });
        if workspace.read_with(cx, |workspace, _| workspace.workspace_presentation())
            != workspace::WorkspacePresentation::Collaborative
        {
            cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
            cx.run_until_parked();
        }
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.close_all_docks(window, cx);
            AgentPanel::reveal_conversation(workspace, true, window, cx);
        });
        let history = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        let before = history.read_with(cx, |history, _| {
            history.entry(terminal_id).cloned().expect("accepted row")
        });
        let source = historical_editor.read_with(cx, |editor, cx| editor.text(cx));
        let entry_index = before
            .inline_context
            .as_ref()
            .expect("accepted context")
            .preceding_entry_count;
        parent.update(cx, |parent, cx| {
            parent.scroll_to_entry_for_tests(entry_index, cx)
        });
        cx.run_until_parked();
        cx.refresh()
            .expect("initial canonical accepted-list mounting");
        for _ in 0..3 {
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.draw(cx).clear(cx);
            });
            cx.run_until_parked();
        }
        let (collapsed, following) = parent.read_with(cx, |parent, _| {
            (
                parent
                    .list_state
                    .bounds_for_item(entry_index)
                    .expect("measured accepted row"),
                parent
                    .list_state
                    .bounds_for_item(entry_index + 1)
                    .expect("measured actual reply"),
            )
        });
        let scroll_before =
            parent.read_with(cx, |parent, _| parent.list_state.logical_scroll_top());
        let following_before =
            parent.read_with(cx, |parent, _| parent.list_state.is_following_tail());
        assert!(
            following_before,
            "explicit history expansion must exercise active tail following"
        );
        let focus_before = cx.update(|window, cx| window.focused(cx));
        let control = cx
            .debug_bounds("inline-terminal-history-disclosure")
            .expect("accepted Show More control");
        cx.simulate_click(control.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        for _ in 0..3 {
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.draw(cx).clear(cx);
            });
            cx.run_until_parked();
        }
        let (expanded, moved_following) = parent.read_with(cx, |parent, _| {
            (
                parent
                    .list_state
                    .bounds_for_item(entry_index)
                    .expect("expanded accepted row"),
                parent
                    .list_state
                    .bounds_for_item(entry_index + 1)
                    .expect("displaced reply"),
            )
        });
        let following_after =
            parent.read_with(cx, |parent, _| parent.list_state.is_following_tail());
        println!("INLINE_ACCEPTED_REFLOW_TAIL before={following_before} after={following_after}");
        assert!(
            !following_after,
            "explicit history expansion must pause tail following"
        );
        println!(
            "INLINE_ACCEPTED_REFLOW entry={entry_index} collapsed={collapsed:?} expanded={expanded:?} following={following:?} moved_following={moved_following:?} scroll_before={scroll_before:?} scroll_after={:?} focus_before={focus_before:?} focus_after={:?} viewport={:?}",
            parent.read_with(cx, |parent, _| parent.list_state.logical_scroll_top()),
            cx.update(|window, cx| window.focused(cx)),
            parent.read_with(cx, |parent, _| parent.list_state.viewport_bounds()),
        );
        assert!(
            expanded.size.height > collapsed.size.height + px(100.),
            "Show More must expand accepted conversation geometry, not an inner scroll viewport"
        );
        assert!(
            moved_following.top() > following.top() + px(100.),
            "Show More must push the actual following reply down"
        );
        assert_eq!(
            cx.update(|window, cx| window.focused(cx)),
            focus_before,
            "accepted disclosure must not steal focus"
        );
        let control = cx
            .debug_bounds("inline-terminal-history-disclosure")
            .expect("accepted Show Less control");
        cx.simulate_click(control.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        for _ in 0..3 {
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.draw(cx).clear(cx);
            });
            cx.run_until_parked();
        }
        parent.read_with(cx, |parent, _| {
            assert_eq!(
                parent
                    .list_state
                    .bounds_for_item(entry_index)
                    .expect("bounded accepted row")
                    .size
                    .height,
                collapsed.size.height
            );
            assert_eq!(
                parent
                    .list_state
                    .bounds_for_item(entry_index + 1)
                    .expect("restored reply placement")
                    .top(),
                following.top()
            );
        });
        assert_eq!(
            historical_editor.read_with(cx, |editor, cx| editor.text(cx)),
            source
        );
        assert_eq!(
            history.read_with(cx, |history, _| history
                .entry(terminal_id)
                .cloned()
                .expect("retained accepted row")),
            before
        );
        workspace.update(cx, |workspace, cx| {
            assert!(workspace.unregister_collaborative_composer_provider(registrations.0, cx));
            assert!(workspace.unregister_collaborative_timeline_provider(registrations.1, cx));
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_inline_queue_promotion_failure_preserves_paused_order_and_draft(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::{
            InlineTerminalDraftPosition, InlineTerminalPromptReference,
        };
        let (panel, mut cx) = setup_panel(cx).await;
        crate::test_support::open_draft_with_connection(
            &panel,
            StubAgentConnection::new(),
            &mut cx,
        );
        cx.run_until_parked();
        cx.update(|_, cx| TerminalThreadMetadataStore::init_global(cx));
        let parent = panel.read_with(&cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("canonical conversation")
        });
        let terminal_id = TerminalId::new();
        parent.update_in(&mut cx, |parent, window, cx| {
            let mut context = parent
                .inline_terminal_context(cx)
                .expect("conversation scope");
            let reference = InlineTerminalPromptReference {
                occurrence_id: None,

                terminal_id,
                thread_id: context.thread_id,
                session_id: context.session_id.clone().expect("canonical session"),
                position: InlineTerminalDraftPosition::new("queued", 6)
                    .expect("queued source boundary"),
            };
            context.draft_position = Some(reference.position.clone());
            let history = TerminalThreadMetadataStore::global(cx);
            history
                .update(cx, |history, cx| {
                    history.save_inline(
                        TerminalThreadMetadata {
                            terminal_id,
                            title: "Retained queue context".into(),
                            custom_title: None,
                            created_at: Utc::now(),
                            worktree_paths: WorktreePaths::from_folder_paths(
                                &workspace::PathList::new(&[PathBuf::from("/project")]),
                            ),
                            remote_connection: None,
                            working_directory: None,
                            inline_context: Some(context),
                        },
                        cx,
                    )
                })
                .expect("canonical retained context");
            parent.add_to_queue(
                vec![
                    "queued".into(),
                    reference
                        .content("retained output".into())
                        .expect("context resource"),
                ],
                Vec::new(),
                window,
                cx,
            );
            parent.add_to_queue(vec!["second".into()], Vec::new(), window, cx);
            parent.add_to_queue(vec!["third".into()], Vec::new(), window, cx);
            parent.message_queue.pause();
            let identities = parent
                .message_queue
                .iter()
                .map(|entry| (entry.id, entry.editor.entity_id(), entry.content.clone()))
                .collect::<Vec<_>>();
            let first_id = parent.message_queue.first_id().expect("front entry");
            let fast_track = parent.message_queue.can_fast_track();
            history.read(cx).close_write_queue_for_test();
            for draft in ["PRESERVED_DRAFT", ""] {
                parent.message_editor.update(cx, |editor, cx| {
                    editor.set_text(draft, window, cx);
                });
                assert!(
                    !parent.move_queued_message_to_main_editor(first_id, None, None, window, cx,)
                );
                assert_eq!(parent.message_editor.read(cx).text(cx), draft);
                assert_eq!(
                    parent
                        .message_queue
                        .iter()
                        .map(|entry| (entry.id, entry.editor.entity_id(), entry.content.clone()))
                        .collect::<Vec<_>>(),
                    identities,
                    "failed promotion must retain ordering, content and subscriptions"
                );
                assert_eq!(parent.message_queue.can_fast_track(), fast_track);
                assert!(
                    parent.message_queue.on_generation_stopped(false).is_none(),
                    "failed promotion must not resume a paused queue"
                );
            }
        });
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_rejects_missing_and_wrong_parent_sessions(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.executor().allow_parking();
        cx.update(|_, cx| {
            let mut agent_settings = AgentSettings::get_global(cx).clone();
            agent_settings.terminal_init_command = None;
            AgentSettings::override_global(agent_settings, cx);
            let mut terminal_settings = TerminalSettings::get_global(cx).clone();
            terminal_settings.shell = task::Shell::Program("/bin/sh".into());
            terminal_settings.working_directory = settings::WorkingDirectory::AlwaysHome;
            TerminalSettings::override_global(terminal_settings, cx);
        });
        crate::test_support::open_draft_with_connection(
            &panel,
            StubAgentConnection::new(),
            &mut cx,
        );
        crate::test_support::type_draft_prompt(&panel, "Retained parent draft", &mut cx);
        cx.run_until_parked();
        cx.update(|_, cx| TerminalThreadMetadataStore::init_global(cx));
        let parent_id = active_thread_id(&panel, &cx);
        let editor = panel.read_with(&cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("parent")
                .read(cx)
                .message_editor
                .clone()
        });
        let preference = panel.read_with(&cx, |panel, _| panel.last_created_entry_kind);
        let initial_terminal_count = panel.read_with(&cx, |panel, _| panel.terminals.len());
        let entry_changes = Arc::new(Mutex::new(0));
        let _subscription = cx.update(|_, cx| {
            let entry_changes = entry_changes.clone();
            cx.subscribe(&panel, move |_, event: &AgentPanelEvent, _| {
                if matches!(event, AgentPanelEvent::EntryChanged) {
                    *entry_changes.lock() += 1;
                }
            })
        });
        for session_id in [None, Some(acp::SessionId::new("different-parent-session"))] {
            let context = InlineTerminalContext {
                occurrence_id: None,
                previous_occurrences: Vec::new(),

                thread_id: parent_id,
                session_id,
                restored_draft_source_session_id: None,
                preceding_entry_count: 0,
                message_client_id: None,
                is_draft: true,
                presentation_order: None,
                draft_position: None,
                collapsed: false,
                output: String::new(),
                snapshot: None,
            };
            let result = panel
                .update_in(&mut cx, |panel, window, cx| {
                    panel.spawn_inline_terminal(context, None, window, cx)
                })
                .await;
            assert!(
                result.is_err(),
                "a mismatched session must reject before PTY creation"
            );
            cx.run_until_parked();
            assert_eq!(
                *entry_changes.lock(),
                0,
                "rejection must not create then clean up a terminal entry"
            );
            panel.read_with(&cx, |panel, cx| {
                assert_eq!(panel.terminals.len(), initial_terminal_count);
                assert_eq!(panel.active_thread_id(cx), Some(parent_id));
                assert_eq!(panel.last_created_entry_kind, preference);
                assert_eq!(
                    panel
                        .active_thread_view(cx)
                        .expect("same parent")
                        .read(cx)
                        .message_editor
                        .entity_id(),
                    editor.entity_id()
                );
                assert_eq!(editor.read(cx).text(cx), "Retained parent draft");
                assert_eq!(
                    TerminalThreadMetadataStore::global(cx)
                        .read(cx)
                        .entries()
                        .count(),
                    0
                );
            });
        }
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_keeps_original_parent_during_async_navigation(
        cx: &mut TestAppContext,
    ) {
        assert_inline_terminal_parent_lifecycle(InlineTerminalAdmissionFailure::None, cx).await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_rejects_changed_draft_during_allocation(cx: &mut TestAppContext) {
        assert_inline_terminal_parent_lifecycle(
            InlineTerminalAdmissionFailure::ChangedDraftDuringAllocation,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_closes_created_orphan_after_parent_removal(
        cx: &mut TestAppContext,
    ) {
        assert_inline_terminal_parent_lifecycle(InlineTerminalAdmissionFailure::RemovedParent, cx)
            .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_rejects_closed_history_before_allocation(
        cx: &mut TestAppContext,
    ) {
        assert_inline_terminal_parent_lifecycle(
            InlineTerminalAdmissionFailure::ClosedHistoryBeforeAllocation,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_rejects_history_closed_during_allocation(
        cx: &mut TestAppContext,
    ) {
        assert_inline_terminal_parent_lifecycle(
            InlineTerminalAdmissionFailure::ClosedHistoryDuringAllocation,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_rejects_invalid_snapshot_before_allocation(
        cx: &mut TestAppContext,
    ) {
        assert_inline_terminal_parent_lifecycle(
            InlineTerminalAdmissionFailure::InvalidSnapshot,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[derive(Clone, Copy)]
    enum InlineTerminalAdmissionFailure {
        None,
        RemovedParent,
        ReplacedHistoryOwner,
        ClosedHistoryBeforeAllocation,
        ClosedHistoryDuringAllocation,
        InvalidSnapshot,
        ChangedDraftDuringAllocation,
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    async fn assert_inline_terminal_parent_lifecycle(
        admission_failure: InlineTerminalAdmissionFailure,
        cx: &mut TestAppContext,
    ) {
        let remove_parent = matches!(
            admission_failure,
            InlineTerminalAdmissionFailure::RemovedParent
        );
        let replace_history_owner = matches!(
            admission_failure,
            InlineTerminalAdmissionFailure::ReplacedHistoryOwner
        );
        let reject_history = matches!(
            admission_failure,
            InlineTerminalAdmissionFailure::ClosedHistoryBeforeAllocation
                | InlineTerminalAdmissionFailure::ClosedHistoryDuringAllocation
                | InlineTerminalAdmissionFailure::InvalidSnapshot
                | InlineTerminalAdmissionFailure::ChangedDraftDuringAllocation
        );
        let (panel, mut cx) = setup_panel(cx).await;
        cx.executor().allow_parking();
        crate::test_support::open_draft_with_connection(
            &panel,
            StubAgentConnection::new(),
            &mut cx,
        );
        crate::test_support::type_draft_prompt(&panel, "Original pending parent", &mut cx);
        cx.run_until_parked();
        let startup_directory = tempfile::TempDir::new().expect("startup oracle directory");
        let startup_marker = startup_directory.path().join("accepted-parent");
        let quoted_path = shlex::try_quote(startup_marker.to_str().expect("UTF-8 test path"))
            .expect("quote startup oracle");
        let startup_command = format!("printf accepted > {quoted_path}");
        cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            let mut agent_settings = AgentSettings::get_global(cx).clone();
            agent_settings.terminal_init_command = Some(startup_command);
            AgentSettings::override_global(agent_settings, cx);
            let mut terminal_settings = TerminalSettings::get_global(cx).clone();
            terminal_settings.shell = task::Shell::Program("/bin/sh".into());
            terminal_settings.working_directory = settings::WorkingDirectory::AlwaysHome;
            TerminalSettings::override_global(terminal_settings, cx);
        });
        let (parent_id, mut context, original_editor, preference, project) =
            panel.read_with(&cx, |panel, cx| {
                let parent_view = panel.active_thread_view(cx).expect("original parent");
                let parent = parent_view.read(cx);
                let parent_id = panel
                    .active_thread_id(cx)
                    .expect("original thread identity");
                (
                    parent_id,
                    InlineTerminalContext {
                        occurrence_id: None,
                        previous_occurrences: Vec::new(),

                        thread_id: parent_id,
                        session_id: Some(parent.session_id.clone()),
                        restored_draft_source_session_id: None,
                        preceding_entry_count: parent.thread.read(cx).entries().len(),
                        message_client_id: None,
                        is_draft: true,
                        presentation_order: None,
                        draft_position: None,
                        collapsed: false,
                        output: String::new(),
                        snapshot: None,
                    },
                    parent.message_editor.clone(),
                    panel.last_created_entry_kind,
                    panel.project.clone(),
                )
            });
        let original_store = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        let changed_draft = matches!(
            admission_failure,
            InlineTerminalAdmissionFailure::ChangedDraftDuringAllocation
        );
        if changed_draft {
            context.draft_position = Some(original_editor.read_with(&cx, |editor, cx| {
                editor
                    .inline_terminal_insertion_position(cx)
                    .expect("original source position")
            }));
        }
        if matches!(
            admission_failure,
            InlineTerminalAdmissionFailure::InvalidSnapshot
        ) {
            let terminal = cx.new(|cx| {
                terminal::TerminalBuilder::new_display_only(
                    terminal::terminal_settings::CursorShape::Block,
                    settings::AlternateScroll::Off,
                    Some(128),
                    cx.entity_id().as_u64(),
                    cx.background_executor(),
                    util::paths::PathStyle::local(),
                )
                .subscribe(cx)
            });
            terminal.update(&mut cx, |terminal, cx| {
                terminal.write_output(b"actual initial history\r\n", cx)
            });
            cx.run_until_parked();
            let snapshot = terminal.read_with(&cx, |terminal, _| {
                terminal.snapshot().expect("actual canonical styled grid")
            });
            let mut encoded = serde_json::to_value(snapshot).expect("encode actual snapshot");
            let columns = encoded.get("columns").cloned().expect("actual columns");
            encoded
                .as_object_mut()
                .expect("snapshot object")
                .insert("cursor_column".into(), columns);
            context.snapshot =
                Some(serde_json::from_value(encoded).expect("typed invalid snapshot"));
        }
        if matches!(
            admission_failure,
            InlineTerminalAdmissionFailure::ClosedHistoryBeforeAllocation
        ) {
            original_store.update(&mut cx, |store, _| store.close_write_queue_for_test());
        }
        let spawn = panel.update_in(&mut cx, |panel, window, cx| {
            panel.spawn_inline_terminal(context.clone(), None, window, cx)
        });
        if changed_draft {
            crate::test_support::type_draft_prompt(&panel, "Changed pending parent", &mut cx);
        }
        if matches!(
            admission_failure,
            InlineTerminalAdmissionFailure::ClosedHistoryDuringAllocation
        ) {
            original_store.update(&mut cx, |store, _| store.close_write_queue_for_test());
        }
        if replace_history_owner {
            cx.update(|_, cx| TerminalThreadMetadataStore::init_global(cx));
        }
        panel.update_in(&mut cx, |panel, window, cx| {
            if remove_parent {
                panel.remove_thread_without_activating_draft(parent_id, window, cx);
            } else {
                panel.open_draft_with_server(
                    Rc::new(StubAgentServer::new(StubAgentConnection::new())),
                    window,
                    cx,
                );
            }
        });
        let selected_thread = panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx));
        assert_ne!(
            selected_thread,
            Some(parent_id),
            "navigate/remove before awaiting PTY completion"
        );
        let result = spawn.await;
        cx.run_until_parked();
        assert_eq!(
            panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx)),
            selected_thread
        );
        assert_eq!(
            original_editor.read_with(&cx, |editor, cx| editor.text(cx)),
            if changed_draft {
                "Changed pending parent"
            } else {
                "Original pending parent"
            }
        );
        assert_eq!(
            panel.read_with(&cx, |panel, _| panel.last_created_entry_kind),
            preference
        );
        if remove_parent || replace_history_owner || reject_history {
            assert!(
                result.is_err(),
                "an invalid original parent or unavailable history owner cannot acquire a terminal"
            );
            if changed_draft {
                assert!(
                    result
                        .as_ref()
                        .err()
                        .expect("changed draft must be rejected")
                        .to_string()
                        .contains("The inline terminal position belongs to a different draft"),
                    "rejection must identify the changed draft rather than an unrelated PTY error"
                );
            }
            assert!(panel.read_with(&cx, |panel, _| panel.terminals.is_empty()));
            assert_eq!(
                original_store.read_with(&cx, |store, _| store.entries().count()),
                0
            );
            for _ in 0..10 {
                cx.run_until_parked();
                cx.executor().timer(Duration::from_millis(50)).await;
            }
            assert!(
                !startup_marker.exists(),
                "invalid ownership must never run the startup command"
            );
            assert!(project.read_with(&cx, |project, _| {
                project.local_terminal_handles().is_empty()
            }));
            assert_eq!(
                cx.update(|_, cx| TerminalThreadMetadataStore::global(cx)
                    .read(cx)
                    .entries()
                    .count()),
                0
            );
        } else {
            let (terminal_id, view) =
                result.expect("retained original parent should accept its PTY");
            let deadline = Instant::now() + Duration::from_secs(10);
            while !startup_marker.exists() {
                cx.run_until_parked();
                assert!(
                    Instant::now() < deadline,
                    "accepted original parent must execute startup"
                );
                cx.executor().timer(Duration::from_millis(50)).await;
            }
            assert!(view.read_with(&cx, |view, cx| view.terminal().read(cx).pid().is_some()));
            let terminal = view.read_with(&cx, |view, _| view.terminal().clone());
            cx.condition(&original_store, |store, cx| {
                let terminal = terminal.read(cx);
                store.entry(terminal_id).is_some_and(|metadata| {
                    metadata.inline_context.as_ref().is_some_and(|context| {
                        context.output == terminal.get_content()
                            && context.snapshot.as_ref()
                                == Some(&terminal.snapshot().expect("actual live PTY grid"))
                    })
                })
            })
            .await;
            let mut context = context;
            context.output = terminal.read_with(&cx, |terminal, _| terminal.get_content());
            context.snapshot = Some(terminal.read_with(&cx, |terminal, _| {
                terminal.snapshot().expect("actual live PTY grid")
            }));
            cx.update(|_, cx| {
                assert_eq!(
                    TerminalThreadMetadataStore::global(cx)
                        .read(cx)
                        .entry(terminal_id)
                        .expect("original canonical association")
                        .inline_context
                        .as_ref(),
                    Some(&context)
                );
            });
            panel.update_in(&mut cx, |panel, window, cx| {
                panel.close_terminal_without_activating_draft(terminal_id, window, cx);
            });
            assert_eq!(
                panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx)),
                selected_thread
            );
        }
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_real_pty_requested_stop_retains_history(cx: &mut TestAppContext) {
        assert_inline_real_pty_preserves_parent_draft(
            false,
            false,
            false,
            InlineTerminalCompletion::RequestedStop,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_real_pty_normal_quit_flushes_live_grid(cx: &mut TestAppContext) {
        assert_inline_real_pty_preserves_parent_draft(
            false,
            false,
            false,
            InlineTerminalCompletion::NormalQuit,
            cx,
        )
        .await;
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    enum InlineTerminalCompletion {
        NaturalExit,
        RequestedStop,
        NormalQuit,
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    async fn assert_inline_real_pty_preserves_parent_draft(
        unassociated: bool,
        navigate_before_exit: bool,
        replace_history_owner_before_exit: bool,
        completion: InlineTerminalCompletion,
        test_app_context: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(test_app_context).await;
        cx.executor().allow_parking();
        if unassociated {
            let project = panel.read_with(&cx, |panel, _| panel.project.clone());
            let ids = project.read_with(&cx, |project, cx| {
                project
                    .worktrees(cx)
                    .map(|worktree| worktree.read(cx).id())
                    .collect::<Vec<_>>()
            });
            project.update(&mut cx, |project, cx| {
                for id in ids {
                    project.remove_worktree(id, cx);
                }
            });
            assert!(!panel.read_with(&cx, |panel, cx| panel.has_open_project(cx)));
        }
        crate::test_support::open_draft_with_connection(
            &panel,
            StubAgentConnection::new(),
            &mut cx,
        );
        crate::test_support::type_draft_prompt(&panel, "Preserved inline terminal draft", &mut cx);
        cx.run_until_parked();
        cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            let mut agent_settings = AgentSettings::get_global(cx).clone();
            agent_settings.terminal_init_command = Some("printf 'inline_real_%s\\n' shell".into());
            AgentSettings::override_global(agent_settings, cx);
            let mut terminal_settings = TerminalSettings::get_global(cx).clone();
            terminal_settings.shell = task::Shell::Program("/bin/sh".into());
            terminal_settings.working_directory = settings::WorkingDirectory::AlwaysHome;
            TerminalSettings::override_global(terminal_settings, cx);
        });
        let parent_id = active_thread_id(&panel, &cx);
        let editor = panel.read_with(&cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("parent thread")
                .read(cx)
                .message_editor
                .clone()
        });
        let preference = panel.read_with(&cx, |panel, _| panel.last_created_entry_kind);
        let persisted_preference =
            cx.update(|_, cx| read_global_last_created_entry_kind(&KeyValueStore::global(cx)));
        let session_id = panel.read_with(&cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("canonical parent session")
                .read(cx)
                .session_id
                .clone()
        });
        let context = InlineTerminalContext {
            occurrence_id: None,
            previous_occurrences: Vec::new(),

            thread_id: parent_id,
            session_id: Some(session_id),
            restored_draft_source_session_id: None,
            preceding_entry_count: 0,
            message_client_id: None,
            is_draft: true,
            presentation_order: None,
            draft_position: None,
            collapsed: false,
            output: String::new(),
            snapshot: None,
        };
        cx.update(|_, cx| {
            assert_eq!(
                AgentPanel::terminal_init_command(true, cx).as_deref(),
                Some("printf 'inline_real_%s\\n' shell"),
                "the actual init command must remain configured after removing worktrees",
            );
            assert_eq!(
                TerminalSettings::get_global(cx).shell,
                task::Shell::Program("/bin/sh".into()),
                "the actual PTY must use the selected POSIX shell",
            );
        });
        let spawn = panel.update_in(&mut cx, |panel, window, cx| {
            panel.spawn_inline_terminal(context.clone(), None, window, cx)
        });
        let (terminal_id, view) = spawn.await.expect("spawn canonical inline PTY");
        let terminal = view.read_with(&cx, |view, _| view.terminal().clone());
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            cx.run_until_parked();
            if terminal
                .read_with(&cx, |terminal, _| terminal.get_content())
                .contains("inline_real_shell")
            {
                break;
            }
            if Instant::now() >= deadline {
                let (is_pty, has_pid, keyboard_input_sent, output_length, input_log) = terminal
                    .update(&mut cx, |terminal, _| {
                        (
                            terminal.is_pty(),
                            terminal.pid().is_some(),
                            terminal.keyboard_input_sent(),
                            terminal.get_content().len(),
                            terminal.take_input_log(),
                        )
                    });
                panic!(
                    "actual PTY init command must execute; unassociated={unassociated}, is_pty={is_pty}, has_pid={has_pid}, keyboard_input_sent={keyboard_input_sent}, output_length={output_length}, input_log={input_log:?}"
                );
            }
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        assert!(
            terminal
                .read_with(&cx, |terminal, _| terminal.pid())
                .is_some(),
            "real PTY identity required"
        );
        panel.read_with(&cx, |panel, cx| {
            assert_eq!(panel.active_thread_id(cx), Some(parent_id));
            assert!(panel.active_terminal_id().is_none());
            assert_eq!(
                panel
                    .active_thread_view(cx)
                    .expect("same parent")
                    .read(cx)
                    .message_editor
                    .entity_id(),
                editor.entity_id()
            );
            assert_eq!(
                panel.editor_text(parent_id, cx).as_deref(),
                Some("Preserved inline terminal draft")
            );
            assert_eq!(panel.last_created_entry_kind, preference);
            assert_eq!(
                panel
                    .terminals
                    .get(&terminal_id)
                    .expect("canonical pool entry")
                    .view
                    .entity_id(),
                view.entity_id()
            );
        });
        let history_owner = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        cx.condition(&history_owner, |store, cx| {
            let terminal = terminal.read(cx);
            store.entry(terminal_id).is_some_and(|metadata| {
                metadata.inline_context.as_ref().is_some_and(|context| {
                    context.output == terminal.get_content()
                        && context.snapshot.as_ref()
                            == Some(&terminal.snapshot().expect("actual live PTY grid"))
                })
            })
        })
        .await;
        let mut context = context;
        context.output = terminal.read_with(&cx, |terminal, _| terminal.get_content());
        context.snapshot = Some(terminal.read_with(&cx, |terminal, _| {
            terminal.snapshot().expect("actual live PTY grid")
        }));
        cx.update(|_, cx| {
            assert_eq!(
                read_global_last_created_entry_kind(&KeyValueStore::global(cx)),
                persisted_preference
            );
            assert_eq!(
                TerminalThreadMetadataStore::global(cx)
                    .read(cx)
                    .entry(terminal_id)
                    .expect("canonical metadata")
                    .inline_context,
                Some(context.clone())
            );
        });
        assert_eq!(
            panel.read_with(&cx, |panel, cx| {
                panel
                    .terminal_metadata(terminal_id, cx)
                    .expect("canonical terminal metadata")
                    .inline_context
            }),
            Some(context.clone()),
            "metadata events must carry the canonical conversation association"
        );
        let close_metadata = Arc::new(Mutex::new(None));
        let _close_subscription = cx.update(|_, cx| {
            let close_metadata = close_metadata.clone();
            cx.subscribe(&panel, move |_, event: &AgentPanelEvent, _| {
                if let AgentPanelEvent::TerminalCloseRequested { metadata } = event
                    && metadata.terminal_id == terminal_id
                {
                    *close_metadata.lock() = Some(metadata.clone());
                }
            })
        });
        if navigate_before_exit {
            crate::test_support::open_draft_with_connection(
                &panel,
                StubAgentConnection::new(),
                &mut cx,
            );
            crate::test_support::type_draft_prompt(&panel, "Other selected draft", &mut cx);
        }
        let selected_parent = panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx));
        if navigate_before_exit {
            assert_ne!(selected_parent, Some(parent_id));
        }
        if replace_history_owner_before_exit {
            let original_owner = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
            let deadline = Instant::now() + Duration::from_secs(10);
            while original_owner.read_with(&cx, |store, _| store.write_pending(terminal_id)) {
                cx.run_until_parked();
                assert!(
                    Instant::now() < deadline,
                    "accepted metadata must reach SQLite"
                );
                cx.executor().timer(Duration::from_millis(50)).await;
            }
            assert!(
                original_owner.read_with(&cx, |store, _| store.write_error(terminal_id).is_none())
            );
            cx.update(|_, cx| TerminalThreadMetadataStore::init_global(cx));
            let replacement_owner = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
            replacement_owner
                .read_with(&cx, |store, _| store.reload_task())
                .await;
            assert_ne!(replacement_owner.entity_id(), original_owner.entity_id());
            let original_metadata =
                original_owner.read_with(&cx, |store, _| store.entry(terminal_id).cloned());
            let replacement_metadata =
                replacement_owner.read_with(&cx, |store, _| store.entry(terminal_id).cloned());
            assert_eq!(replacement_metadata, original_metadata);
            assert_eq!(
                replacement_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.inline_context.as_ref()),
                Some(&context)
            );
            panel.update(&mut cx, |panel, cx| {
                assert!(panel.request_stop_inline_terminal(terminal_id, cx).is_err());
            });
            terminal.update(&mut cx, |terminal, _| {
                terminal.input(b"printf 'OWNER_REPLACEMENT_%s\\n' 20261005\n".as_slice());
            });
            let deadline = Instant::now() + Duration::from_secs(10);
            while !terminal.read_with(&cx, |terminal, _| {
                terminal
                    .get_content()
                    .contains("OWNER_REPLACEMENT_20261005")
            }) {
                cx.run_until_parked();
                assert!(
                    Instant::now() < deadline,
                    "rejected stop must leave the original PTY responsive"
                );
                cx.executor().timer(Duration::from_millis(50)).await;
            }
            let shell_closed = Arc::new(Mutex::new(false));
            let _shell_subscription = cx.update(|_, cx| {
                let shell_closed = shell_closed.clone();
                cx.subscribe(&terminal, move |_, event: &TerminalEvent, _| {
                    if matches!(event, TerminalEvent::CloseTerminal) {
                        *shell_closed.lock() = true;
                    }
                })
            });
            terminal.update(&mut cx, |terminal, _| terminal.input(b"exit\n".as_slice()));
            let deadline = Instant::now() + Duration::from_secs(10);
            while !*shell_closed.lock() {
                cx.run_until_parked();
                assert!(
                    Instant::now() < deadline,
                    "actual PTY must exit after owner replacement"
                );
                cx.executor().timer(Duration::from_millis(50)).await;
            }
            cx.run_until_parked();
            assert!(
                close_metadata.lock().is_none(),
                "stale inline owner must not issue a standalone close request"
            );
            assert!(
                panel.read_with(&cx, |panel, _| panel.has_terminal(terminal_id)),
                "failed history retention must remain available for recovery"
            );
            assert_eq!(
                original_owner.read_with(&cx, |store, _| store.entry(terminal_id).cloned()),
                original_metadata
            );
            assert_eq!(
                replacement_owner.read_with(&cx, |store, _| store.entry(terminal_id).cloned()),
                replacement_metadata
            );
            panel.update_in(&mut cx, |panel, window, cx| {
                panel.close_terminal_without_activating_draft(terminal_id, window, cx);
            });
            assert_eq!(
                replacement_owner.read_with(&cx, |store, _| store.entry(terminal_id).cloned()),
                replacement_metadata,
                "explicit runtime cleanup must not delete replacement-owner history"
            );
            assert_eq!(
                panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx)),
                selected_parent
            );
            assert_eq!(
                editor.read_with(&cx, |editor, cx| editor.text(cx)),
                "Preserved inline terminal draft"
            );
            return;
        }
        match completion {
            InlineTerminalCompletion::NaturalExit => {
                terminal.update(&mut cx, |terminal, _| {
                    terminal.input(b"printf 'INLINE_FINAL_%s\\n' DRAINED; exit\r".as_slice());
                });
            }
            InlineTerminalCompletion::RequestedStop => {
                terminal.update(&mut cx, |terminal, _| {
                    terminal.input(b"printf 'INLINE_STOP_%s\\n' 20261005; sleep 120\n".as_slice());
                });
                let deadline = Instant::now() + Duration::from_secs(10);
                while !terminal.read_with(&cx, |terminal, _| {
                    terminal.get_content().contains("INLINE_STOP_20261005")
                }) {
                    cx.run_until_parked();
                    assert!(
                        Instant::now() < deadline,
                        "real command must run before requested stop"
                    );
                    cx.executor().timer(Duration::from_millis(50)).await;
                }
                panel.update(&mut cx, |panel, cx| {
                    panel
                        .request_stop_inline_terminal(terminal_id, cx)
                        .expect("request canonical inline stop");
                });
            }
            InlineTerminalCompletion::NormalQuit => {
                let database_name = cx.update(|_, cx| {
                    crate::terminal_thread_metadata_store::TestTerminalMetadataDbName::global(cx)
                });
                let expected_snapshot = terminal.read_with(&cx, |terminal, _| {
                    terminal.snapshot().expect("actual live grid before quit")
                });
                let expected_output = terminal.read_with(&cx, |terminal, _| terminal.get_content());
                assert!(expected_output.contains("inline_real_shell"));
                assert!(terminal.read_with(&cx, |terminal, _| terminal.pid().is_some()));
                history_owner
                    .update(&mut cx, |store, cx| {
                        let mut metadata = store.entry(terminal_id).cloned().expect("live history");
                        let context = metadata.inline_context.as_mut().expect("canonical parent");
                        context.output.clear();
                        context.snapshot = None;
                        store.save(metadata, cx);
                        store.flush(cx)
                    })
                    .await
                    .expect("persist precondition with missing latest live output");
                assert!(history_owner.read_with(&cx, |store, _| {
                    store
                        .entry(terminal_id)
                        .and_then(|metadata| metadata.inline_context.as_ref())
                        .is_some_and(|context| {
                            context.output.is_empty() && context.snapshot.is_none()
                        })
                }));
                // This case exercises a completed quit, not the scheduler's randomized
                // early timeout. The real shutdown deadline remains in force.
                test_app_context
                    .background_executor
                    .set_block_on_ticks(usize::MAX..=usize::MAX);
                test_app_context.update(|cx| cx.shutdown());
                let durable_before_restart = history_owner
                    .read_with(test_app_context, |store, _| {
                        store.persisted_entry_for_test(terminal_id)
                    })
                    .expect("read actual SQLite without pumping restart")
                    .expect("normal quit must persist terminal before restart");
                let durable_context = durable_before_restart
                    .inline_context
                    .expect("durable canonical parent");
                assert_eq!(
                    durable_context.output, expected_output,
                    "actual output must be durable at quit"
                );
                assert_eq!(
                    durable_context.snapshot.as_ref(),
                    Some(&expected_snapshot),
                    "actual styled grid must be durable at quit"
                );
                test_app_context.update(|cx| {
                    cx.set_global(
                        crate::terminal_thread_metadata_store::TestTerminalMetadataDbName(
                            database_name,
                        ),
                    );
                    TerminalThreadMetadataStore::init_global(cx);
                });
                let restored_owner =
                    test_app_context.update(|cx| TerminalThreadMetadataStore::global(cx));
                restored_owner
                    .read_with(test_app_context, |store, _| store.reload_task())
                    .await;
                let restored = restored_owner
                    .read_with(test_app_context, |store, _| {
                        store.entry(terminal_id).cloned()
                    })
                    .expect("real normal quit must flush live history before windows disappear");
                let restored_context = restored
                    .inline_context
                    .expect("same canonical conversation");
                context.output = expected_output;
                context.snapshot = Some(expected_snapshot);
                assert_eq!(restored_context, context);
                assert!(
                    close_metadata.lock().is_none(),
                    "quit flush must not synthesize a process exit"
                );
                return;
            }
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while close_metadata.lock().is_none() {
            cx.run_until_parked();
            assert!(
                Instant::now() < deadline,
                "actual shell exit must reach its owner"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        let closed_metadata = close_metadata
            .lock()
            .clone()
            .expect("actual shell-driven close event");
        let closed_context = closed_metadata
            .inline_context
            .as_ref()
            .expect("shell exit must retain the canonical parent in the close event");
        let mut expected_context = context;
        expected_context.output = terminal.read_with(&cx, |terminal, _| terminal.get_content());
        expected_context.snapshot = Some(terminal.read_with(&cx, |terminal, _| {
            terminal.snapshot().expect("actual exited PTY grid")
        }));
        assert!(closed_context.output.contains("inline_real_shell"));
        if matches!(completion, InlineTerminalCompletion::NaturalExit) {
            assert!(closed_context.output.contains("INLINE_FINAL_DRAINED"));
        }
        assert_eq!(closed_context, &expected_context);
        assert!(!panel.read_with(&cx, |panel, _| panel.has_terminal(terminal_id)));
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.restore_terminal(
                closed_metadata.clone(),
                true,
                AgentThreadSource::AgentPanel,
                None,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert!(
            !panel.read_with(&cx, |panel, _| panel.has_terminal(terminal_id)),
            "standalone restoration must not start a new shell for inline history"
        );
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.close_terminal_without_activating_draft(terminal_id, window, cx)
        });
        assert_eq!(
            panel.read_with(&cx, |panel, cx| panel.active_thread_id(cx)),
            selected_parent,
            "finishing an inline PTY must not select or synthesize a draft"
        );
        assert_eq!(
            editor.read_with(&cx, |editor, cx| editor.text(cx)),
            "Preserved inline terminal draft"
        );
        if navigate_before_exit {
            panel.read_with(&cx, |panel, cx| {
                assert_eq!(
                    panel
                        .editor_text(selected_parent.expect("other selected draft"), cx)
                        .as_deref(),
                    Some("Other selected draft")
                );
            });
        }
        let history_owner = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        let deadline = Instant::now() + Duration::from_secs(10);
        while history_owner.read_with(&cx, |store, _| store.write_pending(terminal_id)) {
            cx.run_until_parked();
            assert!(
                Instant::now() < deadline,
                "finished history must reach SQLite"
            );
            cx.executor().timer(Duration::from_millis(50)).await;
        }
        assert!(history_owner.read_with(&cx, |store, _| store.write_error(terminal_id).is_none()));
        cx.update(|_, cx| TerminalThreadMetadataStore::init_global(cx));
        let history_owner = cx.update(|_, cx| TerminalThreadMetadataStore::global(cx));
        history_owner
            .read_with(&cx, |store, _| store.reload_task())
            .await;
        assert_eq!(
            history_owner.read_with(&cx, |store, _| store.entry(terminal_id).cloned()),
            Some(closed_metadata),
            "shell-driven completion must retain the entire canonical grid in SQLite"
        );
    }

    #[cfg(all(unix, feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_inline_terminal_rejects_history_owner_replacement(cx: &mut TestAppContext) {
        assert_inline_terminal_parent_lifecycle(
            InlineTerminalAdmissionFailure::ReplacedHistoryOwner,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn test_restored_terminal_does_not_update_global_entry_kind(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.update(|_, cx| {
            TerminalThreadMetadataStore::init_global(cx);
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.activate_new_thread(false, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        cx.update(|_, cx| {
            assert_eq!(
                read_global_last_created_entry_kind(&KeyValueStore::global(cx)),
                Some(AgentPanelEntryKind::Thread)
            );
        });

        let metadata = TerminalThreadMetadata {
            terminal_id: TerminalId::new(),
            title: "Restored Terminal".into(),
            custom_title: None,
            created_at: Utc::now(),
            worktree_paths: WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from(
                "/project",
            )])),
            remote_connection: None,
            working_directory: None,
            inline_context: None,
        };
        panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.restore_test_terminal(
                    metadata,
                    true,
                    AgentThreadSource::AgentPanel,
                    None,
                    window,
                    cx,
                )
            })
            .expect("test terminal should be restored");
        cx.run_until_parked();

        cx.update(|_, cx| {
            assert_eq!(
                read_global_last_created_entry_kind(&KeyValueStore::global(cx)),
                Some(AgentPanelEntryKind::Thread),
                "restoring a terminal should not change the global new-entry default"
            );
        });
    }

    #[gpui::test]
    async fn test_new_workspace_load_uses_global_terminal_entry_kind(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            TerminalThreadMetadataStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project-a", json!({ "file.txt": "" }))
            .await;
        fs.insert_tree("/project-b", json!({ "file.txt": "" }))
            .await;
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));

        let project_a = Project::test(fs.clone(), [Path::new("/project-a")], cx).await;
        let project_b = Project::test(fs.clone(), [Path::new("/project-b")], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));
        let multi_workspace_entity = multi_workspace.root(cx).unwrap();
        let workspace_a = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        workspace_a.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });
        panel_a
            .update_in(cx, |panel, window, cx| {
                panel.insert_test_terminal("Dev Server", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        cx.update(|_window, cx| {
            assert_eq!(
                read_global_last_created_entry_kind(&KeyValueStore::global(cx)),
                Some(AgentPanelEntryKind::Terminal)
            );
        });

        let workspace_b = multi_workspace_entity.update_in(cx, |multi_workspace, window, cx| {
            multi_workspace.test_add_workspace(project_b.clone(), window, cx)
        });
        workspace_b.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded = AgentPanel::load(workspace_b.downgrade(), async_cx)
            .await
            .expect("panel load should succeed");
        workspace_b.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(loaded.clone(), window, cx);
        });
        loaded.update_in(cx, |panel, window, cx| {
            panel.set_active(true, window, cx);
        });
        for _ in 0..8 {
            cx.run_until_parked();
        }

        loaded.read_with(cx, |panel, cx| {
            assert!(
                panel.active_terminal_id().is_some(),
                "new workspace should initialize to a terminal when terminal was the globally last used entry kind"
            );
            assert!(
                panel.active_conversation_view().is_none(),
                "new workspace should not initialize to a draft when terminal is the global entry kind"
            );
            assert!(panel.should_create_terminal_for_new_entry(cx));
        });
    }

    #[gpui::test]
    async fn test_non_native_thread_without_metadata_is_not_restored(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        panel.update_in(cx, |panel, window, cx| {
            panel.open_external_thread_with_server(
                Rc::new(StubAgentServer::default_response()),
                window,
                cx,
            );
        });

        cx.run_until_parked();

        panel.read_with(cx, |panel, cx| {
            assert!(
                panel.active_agent_thread(cx).is_some(),
                "should have an active thread after connection"
            );
        });

        // Serialize without ever sending a message, so no thread metadata exists.
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded = AgentPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel load should succeed");
        cx.run_until_parked();

        loaded.read_with(cx, |panel, _cx| {
            assert!(
                panel.active_conversation_view().is_none(),
                "thread without metadata should not be restored; the panel should have no active thread"
            );
        });
    }

    #[gpui::test]
    async fn test_serialize_preserves_session_id_in_load_error(cx: &mut TestAppContext) {
        use crate::conversation_view::tests::FlakyAgentServer;
        use crate::thread_metadata_store::{ThreadId, ThreadMetadata};
        use chrono::Utc;
        use project::{AgentId as ProjectAgentId, WorktreePaths};

        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();
        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });
        let workspace_id = workspace
            .read_with(cx, |workspace, _cx| workspace.database_id())
            .expect("workspace should have a database id");

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        // Simulate a previous run that persisted metadata for this session.
        let resume_session_id = acp::SessionId::new("persistent-session");
        cx.update(|_window, cx| {
            ThreadMetadataStore::global(cx).update(cx, |store, cx| {
                store.save(
                    ThreadMetadata {
                        execution_outcome: acp_thread::ThreadExecutionOutcome::Unknown,
                        thread_id: ThreadId::new(),
                        session_id: Some(resume_session_id.clone()),
                        agent_id: ProjectAgentId::new("Flaky"),
                        title: Some("Persistent chat".into()),
                        title_override: None,
                        updated_at: Utc::now(),
                        created_at: Some(Utc::now()),
                        interacted_at: None,
                        worktree_paths: WorktreePaths::from_folder_paths(&PathList::default()),
                        remote_connection: None,
                        archived: false,
                    },
                    cx,
                );
            });
        });

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        // Open a restored thread using a flaky server so the initial connect
        // fails and the view lands in LoadError — mirroring the cold-start
        // race against a custom agent over SSH.
        let (server, _fail) =
            FlakyAgentServer::new(StubAgentConnection::new().with_supports_load_session(true));
        panel.update_in(cx, |panel, window, cx| {
            panel.open_restored_thread_with_server(
                Rc::new(server),
                resume_session_id.clone(),
                window,
                cx,
            );
        });
        cx.run_until_parked();

        // Sanity: the view couldn't connect, so no live AcpThread exists.
        panel.read_with(cx, |panel, cx| {
            assert!(
                panel.active_agent_thread(cx).is_none(),
                "active_agent_thread should be None while the flaky server is failing"
            );
            let conversation_view = panel
                .active_conversation_view()
                .expect("panel should still have an active ConversationView");
            assert_eq!(
                conversation_view.read(cx).root_session_id.as_ref(),
                Some(&resume_session_id),
                "ConversationView should still hold the restored session id"
            );
        });

        // Serialize while in LoadError. Before the fix this wrote
        // `session_id=None` to the KVP and permanently lost the session.
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        let kvp = cx.update(|_window, cx| KeyValueStore::global(cx));
        let serialized: Option<SerializedAgentPanel> = cx
            .background_spawn(async move { read_serialized_panel(workspace_id, &kvp) })
            .await;
        let serialized_session_id = serialized
            .as_ref()
            .and_then(|p| p.last_active_thread.as_ref())
            .and_then(|t| t.session_id.clone());
        assert_eq!(
            serialized_session_id,
            Some(resume_session_id.0.to_string()),
            "serialize() must preserve the restored session id even while the \
             ConversationView is in LoadError; otherwise the bug survives a \
             restart because the KVP has been wiped"
        );
    }

    /// Extracts the text from a Text content block, panicking if it's not Text.
    fn expect_text_block(block: &acp::ContentBlock) -> &str {
        match block {
            acp::ContentBlock::Text(t) => t.text.as_str(),
            other => panic!("expected Text block, got {:?}", other),
        }
    }

    /// Extracts the (text_content, uri) from a Resource content block, panicking
    /// if it's not a TextResourceContents resource.
    fn expect_resource_block(block: &acp::ContentBlock) -> (&str, &str) {
        match block {
            acp::ContentBlock::Resource(r) => match &r.resource {
                acp::EmbeddedResourceResource::TextResourceContents(t) => {
                    (t.text.as_str(), t.uri.as_str())
                }
                other => panic!("expected TextResourceContents, got {:?}", other),
            },
            other => panic!("expected Resource block, got {:?}", other),
        }
    }

    #[gpui::test]
    async fn test_draft_prompt_blocks_use_current_editor_snapshot(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        let _stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        let thread_id = active_thread_id(&panel, cx);
        let thread = panel.read_with(cx, |panel, cx| {
            panel
                .active_agent_thread(cx)
                .expect("draft thread should be active")
        });
        let message_editor = panel.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("draft thread view should be active")
                .read(cx)
                .message_editor
                .clone()
        });

        thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(
                Some(vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "stale prompt",
                ))]),
                cx,
            );
        });
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("fresh prompt", window, cx);
        });
        let blocks = panel.read_with(cx, |panel, cx| {
            panel
                .draft_prompt_blocks_if_in_memory(thread_id, cx)
                .expect("valid live draft snapshot")
                .expect("draft should be in memory")
        });
        assert_eq!(blocks.len(), 1);
        assert_eq!(expect_text_block(&blocks[0]), "fresh prompt");
        let initial = panel.read_with(cx, |panel, cx| {
            panel
                .active_initial_content(cx)
                .expect("valid initial draft snapshot")
        });
        let Some(AgentInitialContent::ContentBlock {
            blocks,
            auto_submit,
        }) = initial
        else {
            panic!("current live source draft must be transferred as content blocks");
        };
        assert!(
            !auto_submit,
            "handoff must not submit the draft automatically"
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            expect_text_block(&blocks[0]),
            "fresh prompt",
            "source-panel handoff must prefer current live draft over stale accepted prompt"
        );

        let (resolution_sender, resolution_receiver) = futures::channel::oneshot::channel::<
            std::result::Result<crate::mention_set::Mention, String>,
        >();
        let unresolved = cx.update(|_, cx| {
            cx.spawn(async move |_| resolution_receiver.await.expect("controlled resolver"))
                .shared()
        });
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![
                    acp::ContentBlock::from("latest draft "),
                    acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                        "pending.rs",
                        "file:///project/pending.rs",
                    )),
                ],
                window,
                cx,
            );
            editor.mention_set().update(cx, |mentions, cx| {
                let crease_id = mentions
                    .creases()
                    .into_iter()
                    .next()
                    .expect("mention crease");
                let uri = mentions
                    .mention_uri_for_crease(&crease_id)
                    .expect("mention URI");
                mentions.insert_mention(crease_id, uri, unresolved.clone(), None, cx);
            });
            cx.notify();
        });
        cx.run_until_parked();
        let live_text = message_editor.read_with(cx, |editor, cx| editor.text(cx));
        let pending_payload = panel.read_with(cx, |panel, cx| {
            panel
                .active_initial_content(cx)
                .expect("valid initial draft snapshot")
        });
        let Some(AgentInitialContent::ContentBlock {
            blocks: pending_blocks,
            auto_submit,
        }) = pending_payload
        else {
            panic!("pending mention must retain its live draft payload");
        };
        assert!(!auto_submit);
        assert!(pending_blocks.iter().any(|block| matches!(block, acp::ContentBlock::Text(text) if text.text.contains("latest draft"))));
        assert!(pending_blocks.iter().any(|block| matches!(block, acp::ContentBlock::ResourceLink(link) if link.uri == "file:///project/pending.rs")));
        resolution_sender
            .send(Err("unavailable context".to_string()))
            .expect("retained resolver");
        assert!(unresolved.await.is_err());
        cx.run_until_parked();
        let failed_payload = panel.read_with(cx, |panel, cx| {
            panel
                .active_initial_content(cx)
                .expect("valid initial draft snapshot")
        });
        let Some(AgentInitialContent::ContentBlock {
            blocks: failed_blocks,
            auto_submit,
        }) = failed_payload
        else {
            panic!("failed context must retain its live draft payload");
        };
        assert!(!auto_submit);
        assert_eq!(
            failed_blocks, pending_blocks,
            "resolution failure must not change user-authored text or mention URI in handoff"
        );
        assert_eq!(
            message_editor.read_with(cx, |editor, cx| editor.text(cx)),
            live_text,
            "handoff must not mutate the source editor"
        );

        thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(
                Some(vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "stale prompt after clear",
                ))]),
                cx,
            );
        });
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("", window, cx);
        });
        let blocks = panel.read_with(cx, |panel, cx| {
            panel
                .draft_prompt_blocks_if_in_memory(thread_id, cx)
                .expect("valid live draft snapshot")
                .expect("draft should be in memory")
        });
        assert!(
            blocks.is_empty(),
            "cleared editor snapshot should override stale saved draft prompt"
        );
        assert!(
            panel
                .read_with(cx, |panel, cx| panel
                    .active_initial_content(cx)
                    .expect("valid initial draft snapshot"))
                .is_none(),
            "source-panel handoff must not resurrect stale draft after explicit clearing"
        );
    }

    #[gpui::test]
    async fn test_draft_has_user_content_checks_all_live_copies(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        fs.insert_tree("/project_b", json!({ "file.txt": "" }))
            .await;
        let project_a = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;
        let project_b = Project::test(fs.clone(), [Path::new("/project_b")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));
        let workspace_a = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        let panel_b = workspace_b.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        let _stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        panel_a.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        let thread_id = active_thread_id(&panel_a, cx);

        panel_b.update_in(cx, |panel, window, cx| {
            panel.load_agent_thread(
                Agent::Stub,
                thread_id,
                Some(PathList::new(&[PathBuf::from("/project_b")])),
                None,
                false,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        crate::test_support::type_draft_prompt(&panel_b, "content in second panel", cx);
        let panel_a_blocks = panel_a.read_with(cx, |panel, cx| {
            panel
                .draft_prompt_blocks_if_in_memory(thread_id, cx)
                .expect("valid live draft snapshot")
                .expect("draft should be live in first panel")
        });
        assert!(
            panel_a_blocks.is_empty(),
            "first live draft copy should be empty"
        );

        let has_user_content = cx.update(|_, cx| {
            crate::draft_prompt_store::draft_has_user_content(
                thread_id,
                [&workspace_a, &workspace_b],
                cx,
            )
        });
        assert!(
            has_user_content,
            "a later live draft copy with content should keep the draft"
        );
    }

    #[test]
    fn test_build_conflict_resolution_prompt_single_conflict() {
        let conflicts = vec![ConflictContent {
            file_path: "src/main.rs".to_string(),
            conflict_text: "<<<<<<< HEAD\nlet x = 1;\n=======\nlet x = 2;\n>>>>>>> feature"
                .to_string(),
            ours_branch_name: "HEAD".to_string(),
            theirs_branch_name: "feature".to_string(),
        }];

        let blocks = build_conflict_resolution_prompt(&conflicts);
        // 2 Text blocks + 1 ResourceLink + 1 Resource for the conflict
        assert_eq!(
            blocks.len(),
            4,
            "expected 2 text + 1 resource link + 1 resource block"
        );

        let intro_text = expect_text_block(&blocks[0]);
        assert!(
            intro_text.contains("Please resolve the following merge conflict in"),
            "prompt should include single-conflict intro text"
        );

        match &blocks[1] {
            acp::ContentBlock::ResourceLink(link) => {
                assert!(
                    link.uri.contains("file://"),
                    "resource link URI should use file scheme"
                );
                assert!(
                    link.uri.contains("main.rs"),
                    "resource link URI should reference file path"
                );
            }
            other => panic!("expected ResourceLink block, got {:?}", other),
        }

        let body_text = expect_text_block(&blocks[2]);
        assert!(
            body_text.contains("`HEAD` (ours)"),
            "prompt should mention ours branch"
        );
        assert!(
            body_text.contains("`feature` (theirs)"),
            "prompt should mention theirs branch"
        );
        assert!(
            body_text.contains("editing the file directly"),
            "prompt should instruct the agent to edit the file"
        );

        let (resource_text, resource_uri) = expect_resource_block(&blocks[3]);
        assert!(
            resource_text.contains("<<<<<<< HEAD"),
            "resource should contain the conflict text"
        );
        assert!(
            resource_uri.contains("merge-conflict"),
            "resource URI should use the merge-conflict scheme"
        );
        assert!(
            resource_uri.contains("main.rs"),
            "resource URI should reference the file path"
        );
    }

    #[test]
    fn test_build_conflict_resolution_prompt_multiple_conflicts_same_file() {
        let conflicts = vec![
            ConflictContent {
                file_path: "src/lib.rs".to_string(),
                conflict_text: "<<<<<<< main\nfn a() {}\n=======\nfn a_v2() {}\n>>>>>>> dev"
                    .to_string(),
                ours_branch_name: "main".to_string(),
                theirs_branch_name: "dev".to_string(),
            },
            ConflictContent {
                file_path: "src/lib.rs".to_string(),
                conflict_text: "<<<<<<< main\nfn b() {}\n=======\nfn b_v2() {}\n>>>>>>> dev"
                    .to_string(),
                ours_branch_name: "main".to_string(),
                theirs_branch_name: "dev".to_string(),
            },
        ];

        let blocks = build_conflict_resolution_prompt(&conflicts);
        // 1 Text instruction + 2 Resource blocks
        assert_eq!(blocks.len(), 3, "expected 1 text + 2 resource blocks");

        let text = expect_text_block(&blocks[0]);
        assert!(
            text.contains("all 2 merge conflicts"),
            "prompt should mention the total count"
        );
        assert!(
            text.contains("`main` (ours)"),
            "prompt should mention ours branch"
        );
        assert!(
            text.contains("`dev` (theirs)"),
            "prompt should mention theirs branch"
        );
        // Single file, so "file" not "files"
        assert!(
            text.contains("file directly"),
            "single file should use singular 'file'"
        );

        let (resource_a, _) = expect_resource_block(&blocks[1]);
        let (resource_b, _) = expect_resource_block(&blocks[2]);
        assert!(
            resource_a.contains("fn a()"),
            "first resource should contain first conflict"
        );
        assert!(
            resource_b.contains("fn b()"),
            "second resource should contain second conflict"
        );
    }

    #[test]
    fn test_build_conflict_resolution_prompt_multiple_conflicts_different_files() {
        let conflicts = vec![
            ConflictContent {
                file_path: "src/a.rs".to_string(),
                conflict_text: "<<<<<<< main\nA\n=======\nB\n>>>>>>> dev".to_string(),
                ours_branch_name: "main".to_string(),
                theirs_branch_name: "dev".to_string(),
            },
            ConflictContent {
                file_path: "src/b.rs".to_string(),
                conflict_text: "<<<<<<< main\nC\n=======\nD\n>>>>>>> dev".to_string(),
                ours_branch_name: "main".to_string(),
                theirs_branch_name: "dev".to_string(),
            },
        ];

        let blocks = build_conflict_resolution_prompt(&conflicts);
        // 1 Text instruction + 2 Resource blocks
        assert_eq!(blocks.len(), 3, "expected 1 text + 2 resource blocks");

        let text = expect_text_block(&blocks[0]);
        assert!(
            text.contains("files directly"),
            "multiple files should use plural 'files'"
        );

        let (_, uri_a) = expect_resource_block(&blocks[1]);
        let (_, uri_b) = expect_resource_block(&blocks[2]);
        assert!(
            uri_a.contains("a.rs"),
            "first resource URI should reference a.rs"
        );
        assert!(
            uri_b.contains("b.rs"),
            "second resource URI should reference b.rs"
        );
    }

    #[test]
    fn test_build_conflicted_files_resolution_prompt_file_paths_only() {
        let file_paths = vec![
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "tests/integration.rs".to_string(),
        ];

        let blocks = build_conflicted_files_resolution_prompt(&file_paths);
        // 1 instruction Text block + (ResourceLink + newline Text) per file
        assert_eq!(
            blocks.len(),
            1 + (file_paths.len() * 2),
            "expected instruction text plus resource links and separators"
        );

        let text = expect_text_block(&blocks[0]);
        assert!(
            text.contains("unresolved merge conflicts"),
            "prompt should describe the task"
        );
        assert!(
            text.contains("conflict markers"),
            "prompt should mention conflict markers"
        );

        for (index, path) in file_paths.iter().enumerate() {
            let link_index = 1 + (index * 2);
            let newline_index = link_index + 1;

            match &blocks[link_index] {
                acp::ContentBlock::ResourceLink(link) => {
                    assert!(
                        link.uri.contains("file://"),
                        "resource link URI should use file scheme"
                    );
                    assert!(
                        link.uri.contains(path),
                        "resource link URI should reference file path: {path}"
                    );
                }
                other => panic!(
                    "expected ResourceLink block at index {}, got {:?}",
                    link_index, other
                ),
            }

            let separator = expect_text_block(&blocks[newline_index]);
            assert_eq!(
                separator, "\n",
                "expected newline separator after each file"
            );
        }
    }

    #[test]
    fn test_build_conflict_resolution_prompt_empty_conflicts() {
        let blocks = build_conflict_resolution_prompt(&[]);
        assert!(
            blocks.is_empty(),
            "empty conflicts should produce no blocks, got {} blocks",
            blocks.len()
        );
    }

    #[test]
    fn test_build_conflicted_files_resolution_prompt_empty_paths() {
        let blocks = build_conflicted_files_resolution_prompt(&[]);
        assert!(
            blocks.is_empty(),
            "empty paths should produce no blocks, got {} blocks",
            blocks.len()
        );
    }

    #[test]
    fn test_conflict_resource_block_structure() {
        let conflict = ConflictContent {
            file_path: "src/utils.rs".to_string(),
            conflict_text: "<<<<<<< HEAD\nold code\n=======\nnew code\n>>>>>>> branch".to_string(),
            ours_branch_name: "HEAD".to_string(),
            theirs_branch_name: "branch".to_string(),
        };

        let block = conflict_resource_block(&conflict);
        let (text, uri) = expect_resource_block(&block);

        assert_eq!(
            text, conflict.conflict_text,
            "resource text should be the raw conflict"
        );
        assert!(
            uri.starts_with("zed:///agent/merge-conflict"),
            "URI should use the zed merge-conflict scheme, got: {uri}"
        );
        assert!(uri.contains("utils.rs"), "URI should encode the file path");
    }

    fn open_generating_thread_with_loadable_connection(
        panel: &Entity<AgentPanel>,
        connection: &StubAgentConnection,
        cx: &mut VisualTestContext,
    ) -> (acp::SessionId, ThreadId) {
        open_thread_with_custom_connection(panel, connection.clone(), cx);
        let session_id = active_session_id(panel, cx);
        let thread_id = active_thread_id(panel, cx);
        send_message(panel, cx);
        cx.update(|_, cx| {
            connection.send_update(
                session_id.clone(),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new("done".into())),
                cx,
            );
        });
        cx.run_until_parked();
        (session_id, thread_id)
    }

    fn open_idle_thread_with_non_loadable_connection(
        panel: &Entity<AgentPanel>,
        connection: &StubAgentConnection,
        cx: &mut VisualTestContext,
    ) -> (acp::SessionId, ThreadId) {
        open_thread_with_custom_connection(panel, connection.clone(), cx);
        let session_id = active_session_id(panel, cx);
        let thread_id = active_thread_id(panel, cx);

        connection.set_next_prompt_updates(vec![acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new("done".into()),
        )]);
        send_message(panel, cx);

        (session_id, thread_id)
    }

    #[gpui::test]
    async fn test_unreadable_draft_restore_preserves_saved_data_and_selection(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |workspace, _| workspace.workspace().clone())
            .expect("test workspace");
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        let _stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        let unreadable_id = active_thread_id(&panel, cx);
        crate::test_support::type_draft_prompt(&panel, "saved input", cx);
        panel.update_in(cx, |panel, window, cx| {
            panel.new_thread(&NewThread, window, cx)
        });
        cx.run_until_parked();
        let selected_id = active_thread_id(&panel, cx);
        assert_ne!(unreadable_id, selected_id);
        assert!(panel.update(cx, |panel, _| {
            panel.test_unload_retained_thread(unreadable_id)
        }));
        let database = cx.read(|cx| db::kvp::KeyValueStore::global(cx));
        let corrupted = "{unreadable validation draft";
        database
            .scoped("agent_draft_prompts")
            .write(unreadable_id.to_key_string(), corrupted.to_string())
            .await
            .expect("isolated corrupt-record control");
        assert!(
            cx.read(|cx| crate::draft_prompt_store::draft_has_user_content(
                unreadable_id,
                std::iter::empty::<&Entity<Workspace>>(),
                cx,
            )),
            "unreadable saved input must block empty-draft archival"
        );
        let replacement = vec![acp::ContentBlock::from("replacement input")];
        let write = cx.read(|cx| {
            crate::draft_prompt_store::write_with_live_draft(
                unreadable_id,
                Some(&replacement),
                &replacement,
                cx,
            )
        });
        assert!(
            write.await.is_err(),
            "automatic persistence must preserve unreadable saved bytes"
        );
        panel.update_in(cx, |panel, window, cx| {
            panel.load_agent_thread(
                Agent::Stub,
                unreadable_id,
                None,
                None,
                false,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        cx.executor()
            .advance_clock(crate::conversation_view::DRAFT_PROMPT_PERSIST_DEBOUNCE * 2);
        cx.run_until_parked();
        assert_eq!(
            active_thread_id(&panel, cx),
            selected_id,
            "unreadable persisted draft must not activate an empty replacement"
        );
        assert_eq!(
            database
                .scoped("agent_draft_prompts")
                .read(&unreadable_id.to_key_string())
                .expect("read retained control"),
            Some(corrupted.to_string()),
            "restore failure must retain original saved bytes"
        );
        assert!(
            workspace.read_with(cx, |workspace, _| workspace
                .message_notification_for_test(&workspace::notifications::NotificationId::unique::<
                    anyhow::Error,
                >())
                .is_some()),
            "restore error must reach the existing workspace notification owner"
        );
        panel.update_in(cx, |panel, window, cx| {
            panel.restore_new_draft(unreadable_id, window, cx);
        });
        cx.run_until_parked();
        assert_eq!(active_thread_id(&panel, cx), selected_id);
        assert_eq!(
            database
                .scoped("agent_draft_prompts")
                .read(&unreadable_id.to_key_string())
                .expect("read after alternate restore"),
            Some(corrupted.to_string())
        );
    }

    #[gpui::test]
    async fn test_failed_context_draft_reload_keeps_latest_unsent_input(cx: &mut TestAppContext) {
        assert_unresolved_draft_reload_keeps_latest_unsent_input(false, cx).await;
    }

    #[gpui::test]
    async fn test_pending_context_draft_reload_keeps_latest_unsent_input(cx: &mut TestAppContext) {
        assert_unresolved_draft_reload_keeps_latest_unsent_input(true, cx).await;
    }

    async fn assert_unresolved_draft_reload_keeps_latest_unsent_input(
        pending: bool,
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |workspace, _| workspace.workspace().clone())
            .expect("test workspace");
        workspace.update(cx, |workspace, _| workspace.set_random_database_id());
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        let _stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        let thread_id = active_thread_id(&panel, cx);
        let original_session = active_session_id(&panel, cx);
        crate::test_support::type_draft_prompt(&panel, "accepted draft", cx);
        let accepted = cx
            .read(|cx| crate::draft_prompt_store::read(thread_id, cx))
            .expect("accepted prompt persisted");
        let editor = panel.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("thread view")
                .read(cx)
                .message_editor
                .clone()
        });
        let (resolution_sender, resolution_receiver) = futures::channel::oneshot::channel::<
            std::result::Result<crate::mention_set::Mention, String>,
        >();
        let resolution = cx.update(|_, cx| {
            cx.spawn(async move |_| resolution_receiver.await.expect("controlled resolver"))
                .shared()
        });
        editor.update_in(cx, |editor, window, cx| {
            editor.set_message(
                vec![
                    acp::ContentBlock::from("latest unsent draft "),
                    acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                        "unavailable.rs",
                        "file:///project/unavailable.rs",
                    )),
                ],
                window,
                cx,
            );
            editor.mention_set().update(cx, |mentions, cx| {
                let crease = mentions
                    .creases()
                    .into_iter()
                    .next()
                    .expect("real mention crease");
                let uri = mentions
                    .mention_uri_for_crease(&crease)
                    .expect("crease URI");
                mentions.insert_mention(
                    crease,
                    uri,
                    if pending {
                        resolution.clone()
                    } else {
                        Task::ready(Err("context unavailable".to_string())).shared()
                    },
                    None,
                    cx,
                );
            });
            cx.notify();
        });
        if !pending {
            assert!(
                editor
                    .update(cx, |editor, cx| editor.draft_contents(cx))
                    .await
                    .is_err()
            );
        }
        cx.run_until_parked();
        cx.executor()
            .advance_clock(crate::conversation_view::DRAFT_PROMPT_PERSIST_DEBOUNCE * 2);
        cx.run_until_parked();
        assert_eq!(
            cx.read(|cx| crate::draft_prompt_store::read(thread_id, cx)),
            Some(accepted.clone()),
            "last trustworthy accepted prompt must remain intact"
        );
        let expected = editor.read_with(cx, |editor, cx| {
            editor
                .draft_content_blocks_snapshot(cx)
                .expect("valid draft snapshot")
        });
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();
        let asynchronous = cx.update(|window, cx| window.to_async(cx));
        let restored = AgentPanel::load(workspace.downgrade(), asynchronous)
            .await
            .expect("canonical panel reload");
        cx.run_until_parked();
        assert_eq!(active_thread_id(&restored, cx), thread_id);
        assert_ne!(active_session_id(&restored, cx), original_session);
        let actual = restored
            .read_with(cx, |panel, cx| {
                panel
                    .draft_prompt_blocks_if_in_memory(thread_id, cx)
                    .expect("valid live draft snapshot")
            })
            .expect("restored live draft");
        assert_eq!(
            actual, expected,
            "canonical reload must restore latest unsent input and mention identity after context failure"
        );
        assert!(
            restored.read_with(cx, |panel, cx| panel
                .active_agent_thread(cx)
                .expect("restored thread")
                .read(cx)
                .entries()
                .is_empty()),
            "restoring a draft must not submit it"
        );
        assert_eq!(
            cx.read(|cx| crate::draft_prompt_store::read(thread_id, cx)),
            Some(accepted),
            "reload must not replace accepted context with unresolved contents"
        );
        let label = cx
            .read(|cx| crate::draft_prompt_store::display_label_for_draft(None, thread_id, cx))
            .expect("persisted draft label");
        assert!(
            label.starts_with("latest unsent draft"),
            "unloaded label must describe latest user input"
        );
        if pending {
            resolution_sender
                .send(Err("context unavailable".to_string()))
                .expect("retained resolver");
            assert!(resolution.await.is_err());
            cx.run_until_parked();
            cx.executor()
                .advance_clock(crate::conversation_view::DRAFT_PROMPT_PERSIST_DEBOUNCE * 2);
            cx.run_until_parked();
        }
        let restored_editor = restored.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("restored view")
                .read(cx)
                .message_editor
                .clone()
        });
        restored_editor.update_in(cx, |editor, window, cx| editor.clear(window, cx));
        cx.run_until_parked();
        cx.executor()
            .advance_clock(crate::conversation_view::DRAFT_PROMPT_PERSIST_DEBOUNCE * 2);
        cx.run_until_parked();
        assert!(
            cx.read(|cx| crate::draft_prompt_store::read(thread_id, cx))
                .is_none()
        );
        assert!(
            cx.read(|cx| crate::draft_prompt_store::read_for_restore(thread_id, cx))
                .is_none(),
            "explicit clearing deletes accepted and live draft state together"
        );
        crate::test_support::type_draft_prompt(&restored, "successful retry", cx);
        let retry = vec![acp::ContentBlock::from("successful retry")];
        assert_eq!(
            cx.read(|cx| crate::draft_prompt_store::read(thread_id, cx)),
            Some(retry.clone())
        );
        assert_eq!(
            cx.read(|cx| crate::draft_prompt_store::read_for_restore(thread_id, cx)),
            Some(retry)
        );
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_draft_reload_rebinds_styled_inline_history_to_fresh_session(
        cx: &mut TestAppContext,
    ) {
        use crate::terminal_thread_metadata_store::InlineTerminalDraftPosition;
        let (initial_panel, mut cx) = setup_visible_panel(cx).await;
        let workspace = initial_panel.read_with(&cx, |panel, _| {
            panel.workspace.upgrade().expect("workspace")
        });
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.remove_panel(&initial_panel, window, cx);
            workspace.set_random_database_id();
        });
        drop(initial_panel);
        cx.run_until_parked();
        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        let _connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        crate::test_support::type_draft_prompt(&panel, "Aλ🦀B", &mut cx);
        cx.update(|_, cx| TerminalThreadMetadataStore::init_global(cx));
        let parent = panel.read_with(&cx, |panel, cx| {
            panel.active_thread_view(cx).expect("draft")
        });
        let thread_id = active_thread_id(&panel, &mut cx);
        let old_session = active_session_id(&panel, &mut cx);
        let terminal_id = TerminalId::new();
        let snapshot = parent.update_in(&mut cx, |parent, window, cx| {
            let settings = TerminalSettings::get_global(cx);
            let project = workspace.read(cx).project().clone();
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
                terminal.write_output(b"\x1b[31mRESTART_STYLE\x1b[0m\r\n", cx)
            });
            let snapshot = terminal.read(cx).snapshot().expect("styled snapshot");
            let mut context = parent.inline_terminal_context(cx).expect("draft scope");
            context.draft_position = Some(
                InlineTerminalDraftPosition::new("Aλ🦀B", "Aλ".len()).expect("Unicode boundary"),
            );
            context.snapshot = Some(snapshot.clone());
            context.output = terminal.read(cx).get_content();
            context.collapsed = true;
            TerminalThreadMetadataStore::global(cx)
                .update(cx, |store, cx| {
                    store.save_inline(
                        TerminalThreadMetadata {
                            terminal_id,
                            title: "Restart history".into(),
                            custom_title: None,
                            created_at: Utc::now(),
                            worktree_paths: project.read(cx).worktree_paths(cx),
                            remote_connection: project.read(cx).remote_connection_options(cx),
                            working_directory: None,
                            inline_context: Some(context),
                        },
                        cx,
                    )
                })
                .expect("retain canonical history");
            parent
                .sync_inline_terminal_history(window, cx)
                .expect("initial mounted history");
            snapshot
        });
        cx.run_until_parked();
        cx.executor()
            .advance_clock(crate::conversation_view::DRAFT_PROMPT_PERSIST_DEBOUNCE * 2);
        cx.run_until_parked();
        panel.update(&mut cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();
        let history = cx.read(|cx| TerminalThreadMetadataStore::global(cx));
        history
            .update(&mut cx, |history, cx| history.flush(cx))
            .await
            .expect("persist history");
        let saved = cx
            .read(|cx| crate::draft_prompt_store::read_for_restore(thread_id, cx))
            .expect("persisted draft");
        assert!(saved.iter().any(|block| {
            crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(
                block,
            )
            .expect("valid saved reference")
            .is_some()
        }));
        drop(parent);
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.remove_panel(&panel, window, cx)
        });
        drop(panel);
        cx.run_until_parked();
        history.update(&mut cx, |history, cx| history.reload_for_test(cx));
        history
            .read_with(&cx, |history, _| history.reload_task())
            .await;
        cx.run_until_parked();
        let retained_before = history.read_with(&cx, |history, _| {
            history
                .entry(terminal_id)
                .cloned()
                .expect("saved association")
        });
        let mut moved = vec![acp::ContentBlock::Text(acp::TextContent::new("Aλ🦀B"))];
        moved.extend(saved.iter().filter(|block| {
            crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(block)
                .expect("valid original reference").is_some()
        }).cloned());
        cx.read(|cx| crate::draft_prompt_store::write(thread_id, &moved, cx))
            .await
            .expect("write moved-reference control");
        let asynchronous = cx.update(|window, cx| window.to_async(cx));
        let rejected = AgentPanel::load(workspace.downgrade(), asynchronous)
            .await
            .expect("load malformed placement without deleting saved data");
        cx.run_until_parked();
        rejected.read_with(&cx, |panel, cx| {
            assert!(
                panel.active_agent_thread(cx).is_none(),
                "moved resource must not mount as a chip"
            );
            let error = panel
                .active_conversation_view()
                .and_then(|view| view.read(cx).load_error_for_test())
                .expect("placement failure reaches the conversation error UI");
            assert!(error.contains("saved draft boundary"), "{error}");
            assert!(
                panel.draft_has_content(
                    panel
                        .draft_thread
                        .as_ref()
                        .expect("failed restore remains a draft"),
                    cx
                ),
                "pending saved content must not be treated as an empty draft"
            );
        });
        assert_eq!(
            history.read_with(&cx, |history, _| history.entry(terminal_id).cloned()),
            Some(retained_before)
        );
        assert_eq!(
            cx.read(|cx| crate::draft_prompt_store::read_for_restore(thread_id, cx)),
            Some(moved)
        );
        drop(rejected);
        cx.read(|cx| crate::draft_prompt_store::write(thread_id, &saved, cx))
            .await
            .expect("restore valid control input");
        let asynchronous = cx.update(|window, cx| window.to_async(cx));
        let restored = AgentPanel::load(workspace.downgrade(), asynchronous)
            .await
            .expect("canonical panel reload");
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.add_panel(restored.clone(), window, cx)
        });
        cx.run_until_parked();
        let new_session = restored.read_with(&cx, |panel, cx| {
            assert!(
                panel.active_agent_thread(cx).is_some(),
                "restore error: {:?}",
                panel
                    .active_conversation_view()
                    .and_then(|view| view.read(cx).load_error_for_test())
            );
            panel
                .active_agent_thread(cx)
                .expect("the persisted draft must reconnect before inline restoration")
                .read(cx)
                .session_id()
                .clone()
        });
        assert_ne!(new_session, old_session);
        assert_eq!(active_thread_id(&restored, &mut cx), thread_id);
        restored.read_with(&cx, |panel, cx| {
            assert!(
                panel.terminals.is_empty(),
                "restoring snapshots must not spawn a PTY"
            );
            assert_eq!(panel.editor_text(thread_id, cx).as_deref(), Some("Aλ🦀B"));
        });
        let restored_parent = restored.read_with(&cx, |panel, cx| {
            panel.active_thread_view(cx).expect("restored draft")
        });
        restored_parent.update_in(&mut cx, |parent, window, cx| {
            parent
                .focus_after_inline_terminal(terminal_id, window, cx)
                .expect("restoration mounted the inline block, not a mention chip");
        });
        history.read_with(&cx, |history, _| {
            let context = history
                .entry(terminal_id)
                .and_then(|metadata| metadata.inline_context.as_ref())
                .expect("retained association");
            assert_eq!(context.thread_id, thread_id);
            assert_eq!(context.session_id.as_ref(), Some(&new_session));
            assert!(
                !context.collapsed,
                "Delta restores terminal output expanded"
            );
            assert_eq!(context.snapshot.as_ref(), Some(&snapshot));
            assert!(context.output.contains("RESTART_STYLE"));
            assert!(context.is_draft && context.message_client_id.is_none());
        });
        cx.executor()
            .advance_clock(crate::conversation_view::DRAFT_PROMPT_PERSIST_DEBOUNCE * 2);
        cx.run_until_parked();
        history
            .update(&mut cx, |history, cx| history.flush(cx))
            .await
            .expect("persist rebound history");
        let rebound = cx
            .read(|cx| crate::draft_prompt_store::read_for_restore(thread_id, cx))
            .expect("rewritten draft");
        assert!(rebound.iter().any(|block| {
            crate::terminal_thread_metadata_store::InlineTerminalPromptReference::from_content(
                block,
            )
            .expect("rewritten reference")
            .is_some_and(|reference| reference.session_id == new_session)
        }));
        // Reproduce an interrupted draft write after metadata became durable.
        cx.read(|cx| crate::draft_prompt_store::write(thread_id, &saved, cx))
            .await
            .expect("retain exact pre-write draft for recovery control");
        drop(restored_parent);
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.remove_panel(&restored, window, cx)
        });
        drop(restored);
        cx.run_until_parked();
        history.update(&mut cx, |history, cx| history.reload_for_test(cx));
        history
            .read_with(&cx, |history, _| history.reload_task())
            .await;
        cx.run_until_parked();
        let asynchronous = cx.update(|window, cx| window.to_async(cx));
        let recovered = AgentPanel::load(workspace.downgrade(), asynchronous)
            .await
            .expect("reload stale draft against durable rebound metadata");
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.add_panel(recovered.clone(), window, cx)
        });
        cx.run_until_parked();
        let recovered_session = active_session_id(&recovered, &cx);
        assert_ne!(recovered_session, new_session);
        assert_eq!(active_thread_id(&recovered, &cx), thread_id);
        recovered.read_with(&cx, |panel, cx| {
            assert!(panel.terminals.is_empty());
            assert_eq!(panel.editor_text(thread_id, cx).as_deref(), Some("Aλ🦀B"));
        });
        recovered
            .read_with(&cx, |panel, cx| {
                panel.active_thread_view(cx).expect("recovered draft")
            })
            .update_in(&mut cx, |parent, window, cx| {
                parent
                    .focus_after_inline_terminal(terminal_id, window, cx)
                    .expect("recovery retained the mounted inline block");
            });
        history.read_with(&cx, |history, _| {
            let context = history
                .entry(terminal_id)
                .and_then(|metadata| metadata.inline_context.as_ref())
                .expect("recovered association");
            assert_eq!(context.session_id.as_ref(), Some(&recovered_session));
            assert_eq!(
                context.restored_draft_source_session_id.as_ref(),
                Some(&old_session)
            );
            assert_eq!(context.snapshot.as_ref(), Some(&snapshot));
            assert!(!context.collapsed);
        });
    }

    #[gpui::test]
    async fn test_draft_promotion_creates_metadata_and_new_session_on_reload(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Register a shared stub connection and use Agent::Stub so the draft
        // (and any reloaded draft) uses it.
        let stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        stub_connection.set_next_prompt_updates(vec![acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new("Response".into()),
        )]);
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        // Verify the thread is considered a draft.
        panel.read_with(cx, |panel, cx| {
            assert!(
                panel.active_thread_is_draft(cx),
                "thread should be a draft before any message is sent"
            );
            assert!(
                panel.draft_thread.is_some(),
                "draft_thread field should be set"
            );
        });
        let draft_session_id = active_session_id(&panel, cx);
        let thread_id = active_thread_id(&panel, cx);

        // A draft thread is persisted with session_id: None.
        cx.update(|_window, cx| {
            let store = ThreadMetadataStore::global(cx).read(cx);
            let entry = store
                .entry(thread_id)
                .expect("draft thread should have a metadata row");
            assert!(
                entry.is_draft(),
                "draft thread metadata should have session_id=None, got {:?}",
                entry.session_id,
            );
        });

        // Type into the message editor; the editor observer pushes the text
        // into `AcpThread.draft_prompt`, which emits `PromptUpdated` and
        // persists the prompt to the kvp store.
        crate::test_support::type_draft_prompt(&panel, "Hello from draft", cx);
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let reloaded_panel = AgentPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel load with draft should succeed");
        cx.run_until_parked();

        reloaded_panel.read_with(cx, |panel, cx| {
            assert!(
                panel.active_thread_is_draft(cx),
                "reloaded panel should still show the draft as active"
            );
            assert!(
                panel.active_view_is_new_draft(cx),
                "reloaded draft should still occupy the new-draft slot: \
                 what's in the new-draft slot stays there across restarts, \
                 regardless of whether it's also the active view"
            );
            let active_entity = panel.active_conversation_view().map(|v| v.entity_id());
            let draft_entity = panel.draft_thread.as_ref().map(|v| v.entity_id());
            assert!(
                active_entity.is_some() && active_entity == draft_entity,
                "active view and draft slot should share a single ConversationView entity \
                 (active={active_entity:?}, draft={draft_entity:?})"
            );
        });

        // Thread identity is stable across reload — the metadata row we wrote
        // pre-reload maps back to the same ConversationView.
        let reloaded_thread_id = active_thread_id(&reloaded_panel, cx);
        assert_eq!(
            reloaded_thread_id, thread_id,
            "reloaded draft should preserve its ThreadId"
        );

        // ACP session_id is NOT preserved: drafts don't persist a session id,
        // so the reloaded ConversationView opens a fresh ACP session.
        let reloaded_session_id = active_session_id(&reloaded_panel, cx);
        assert_ne!(
            reloaded_session_id, draft_session_id,
            "reloaded draft should have a fresh ACP session ID"
        );

        let restored_text =
            reloaded_panel.read_with(cx, |panel, cx| panel.editor_text(reloaded_thread_id, cx));
        assert_eq!(
            restored_text.as_deref(),
            Some("Hello from draft"),
            "draft prompt text should be restored from the draft-prompt kvp store"
        );

        // Send a message on the reloaded panel — this promotes the draft to a
        // real thread. `ThreadId` stays the same; `session_id` is populated.
        let panel = reloaded_panel;
        let promoted_session_id = reloaded_session_id;
        send_message(&panel, cx);

        panel.read_with(cx, |panel, cx| {
            assert!(
                !panel.active_thread_is_draft(cx),
                "thread should no longer be a draft after sending a message"
            );
            assert!(
                panel.draft_thread.is_none(),
                "draft_thread should be None after promotion"
            );
            assert_eq!(
                panel.active_thread_id(cx),
                Some(thread_id),
                "same ThreadId should remain active after promotion"
            );
        });

        cx.update(|_window, cx| {
            let store = ThreadMetadataStore::global(cx).read(cx);
            let metadata = store
                .entry(thread_id)
                .expect("promoted thread should have metadata");
            assert!(
                !metadata.is_draft(),
                "promoted thread metadata should no longer be a draft"
            );
            assert_eq!(
                metadata.session_id.as_ref(),
                Some(&promoted_session_id),
                "metadata session_id should match the thread's ACP session"
            );
        });

        // Serialize the panel, then reload it again.
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded_panel = AgentPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel load should succeed");
        cx.run_until_parked();

        // The second load should restore the promoted real thread, keyed by
        // its session_id.
        loaded_panel.read_with(cx, |panel, cx| {
            let active_id = panel.active_thread_id(cx);
            assert_eq!(
                active_id,
                Some(thread_id),
                "loaded panel should restore the promoted thread"
            );
            assert!(
                !panel.active_thread_is_draft(cx),
                "restored thread should not be a draft"
            );
        });
    }

    #[gpui::test]
    async fn test_new_draft_survives_reload_when_real_thread_is_active(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();
        workspace.update(cx, |workspace, _cx| workspace.set_random_database_id());

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Register a shared stub connection under `Agent::Stub` so every
        // ConversationView the panel creates in this test (including any
        // post-reload rehydrations) reaches Connected synchronously.
        let stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        stub_connection.set_next_prompt_updates(vec![acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new("ok".into()),
        )]);

        // 1. Create a real thread by sending a message.
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        crate::test_support::send_message(&panel, cx);
        let real_thread_id = crate::test_support::active_thread_id(&panel, cx);
        let real_session_id = crate::test_support::active_session_id(&panel, cx);
        cx.run_until_parked();

        // 2. Open a draft, type into it, then press Cmd-N again to
        //    park it into retained_threads as a *retained* draft.
        panel.update_in(cx, |panel, window, cx| {
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        let retained_draft_id = crate::test_support::active_thread_id(&panel, cx);
        crate::test_support::type_draft_prompt(&panel, "retained draft text", cx);

        panel.update_in(cx, |panel, window, cx| {
            panel.new_thread(&NewThread, window, cx);
        });
        cx.run_until_parked();

        // The pre-existing draft is now in retained_threads (parked),
        // and a fresh empty ephemeral new-draft is active.
        panel.read_with(cx, |panel, cx| {
            assert!(
                panel.retained_threads.contains_key(&retained_draft_id),
                "first draft with content should be parked into retained_threads"
            );
            assert_ne!(
                panel.active_thread_id(cx),
                Some(retained_draft_id),
                "active view should be a fresh ephemeral draft, not the retained one"
            );
        });

        // 3. Type into the new ephemeral draft.
        let draft_thread_id = crate::test_support::active_thread_id(&panel, cx);
        crate::test_support::type_draft_prompt(&panel, "in-flight draft text", cx);

        // Sanity-check: both drafts' text has been persisted to the kvp
        // store via the editor observer / PromptUpdated chain.
        let (ephemeral_kvp, retained_kvp) = cx.update(|_, cx| {
            (
                crate::draft_prompt_store::read(draft_thread_id, cx),
                crate::draft_prompt_store::read(retained_draft_id, cx),
            )
        });
        assert!(
            ephemeral_kvp.is_some(),
            "ephemeral draft's prompt should be in the kvp store"
        );
        assert!(
            retained_kvp.is_some(),
            "retained draft's prompt should be in the kvp store"
        );

        assert_ne!(real_thread_id, draft_thread_id);
        assert_ne!(retained_draft_id, draft_thread_id);
        panel.read_with(cx, |panel, cx| {
            assert!(
                panel.active_view_is_new_draft(cx),
                "draft should currently occupy the new-draft slot"
            );
        });

        // 4. Switch the active view back to the real thread. The ephemeral
        //    draft has content, so it gets parked into `retained_threads`
        //    immediately (the `draft_thread` slot is cleared).
        panel.update_in(cx, |panel, window, cx| {
            panel.load_agent_thread(
                Agent::Stub,
                real_thread_id,
                None,
                None,
                false,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        panel.read_with(cx, |panel, cx| {
            assert_eq!(panel.active_thread_id(cx), Some(real_thread_id));
            assert!(!panel.active_view_is_new_draft(cx));
        });

        // 5. Serialize + reload.
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded_panel = AgentPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel load should succeed");
        cx.run_until_parked();

        // 6. The real thread is the active view on reload. The draft
        //    was parked when the user navigated away, so the draft_thread
        //    slot is empty.
        loaded_panel.read_with(cx, |panel, cx| {
            assert_eq!(
                panel.active_thread_id(cx),
                Some(real_thread_id),
                "real thread should be the active view after reload"
            );
            assert!(
                !panel.active_thread_is_draft(cx),
                "real thread is not a draft"
            );
            assert!(
                panel.draft_thread.is_none(),
                "draft_thread slot should be empty since the draft was parked on navigate-away"
            );
        });

        // 7. All three threads' metadata rows survive the reload.
        cx.update(|_window, cx| {
            let store = ThreadMetadataStore::global(cx).read(cx);
            let ephemeral_row = store
                .entry(draft_thread_id)
                .expect("ephemeral draft metadata row should survive reload");
            assert!(
                ephemeral_row.is_draft(),
                "ephemeral draft row should still be a draft"
            );
            let retained_row = store
                .entry(retained_draft_id)
                .expect("retained draft metadata row should survive reload");
            assert!(
                retained_row.is_draft(),
                "retained draft row should still be a draft"
            );
            let real_row = store
                .entry(real_thread_id)
                .expect("real thread metadata row should survive reload");
            assert_eq!(real_row.session_id.as_ref(), Some(&real_session_id));
        });

        // 8. Opening the parked draft via load_agent_thread activates
        //    a fresh ConversationView and exposes its kvp-seeded prompt
        //    text in the editor.
        loaded_panel.update_in(cx, |panel, window, cx| {
            panel.load_agent_thread(
                Agent::Stub,
                draft_thread_id,
                None,
                None,
                false,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let restored_ephemeral_text =
            loaded_panel.read_with(cx, |panel, cx| panel.editor_text(draft_thread_id, cx));
        assert_eq!(
            restored_ephemeral_text.as_deref(),
            Some("in-flight draft text"),
            "ephemeral draft prompt text should be restored from the kvp store"
        );

        // 9. Opening the retained draft via load_agent_thread builds a
        //    fresh ConversationView (since retained_threads was not
        //    carried across the reload) and seeds its editor from the
        //    kvp store.
        loaded_panel.update_in(cx, |panel, window, cx| {
            panel.load_agent_thread(
                Agent::Stub,
                retained_draft_id,
                None,
                None,
                false,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let restored_retained_text =
            loaded_panel.read_with(cx, |panel, cx| panel.editor_text(retained_draft_id, cx));
        assert_eq!(
            restored_retained_text.as_deref(),
            Some("retained draft text"),
            "retained draft prompt text should be restored from the kvp store"
        );
    }

    #[gpui::test]
    async fn test_reloaded_ephemeral_draft_preserves_original_agent(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();
        workspace.update(cx, |workspace, _cx| workspace.set_random_database_id());

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        let _stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        let draft_thread_id = crate::test_support::active_thread_id(&panel, cx);
        crate::test_support::type_draft_prompt(&panel, "pinned to stub", cx);

        // Diverge `selected_agent` from the draft's bound agent before
        // serialize.
        let other_agent = Agent::Custom {
            id: "other-agent".into(),
        };
        panel.update(cx, |panel, _cx| {
            panel.selected_agent = other_agent.clone();
        });
        panel.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        // Sanity-check: the draft's metadata row has agent_id="stub",
        // not "other-agent".
        cx.update(|_, cx| {
            let store = ThreadMetadataStore::global(cx).read(cx);
            let row = store
                .entry(draft_thread_id)
                .expect("draft metadata row should exist");
            assert_eq!(
                row.agent_id.as_ref(),
                "stub",
                "draft metadata should retain its original agent binding"
            );
        });

        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let reloaded_panel = AgentPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel load should succeed");
        cx.run_until_parked();

        reloaded_panel.read_with(cx, |panel, cx| {
            let draft_view = panel
                .draft_thread
                .as_ref()
                .expect("draft slot should be repopulated");
            assert_eq!(
                draft_view.read(cx).thread_id,
                draft_thread_id,
                "restored draft should have the same ThreadId"
            );
            assert_eq!(
                draft_view.read(cx).agent_key(),
                &Agent::Stub,
                "restored draft should still be bound to its original Agent::Stub, \
                 not the panel's current `selected_agent`"
            );
        });
    }

    #[gpui::test]
    async fn test_empty_workspace_does_not_create_agent_entries(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        panel.read_with(cx, |panel, cx| {
            assert_eq!(
                panel
                    .connection_store()
                    .read(cx)
                    .connection_status(&Agent::NativeAgent, cx),
                crate::agent_connection_store::AgentConnectionStatus::Disconnected,
                "empty workspaces should not start the native agent connection"
            );
        });

        panel.update_in(cx, |panel, window, cx| {
            panel.new_thread(&NewThread, window, cx);
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
            panel.new_external_agent_thread(
                &NewExternalAgentThread {
                    agent: AgentId::new("external-agent"),
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();

        panel.read_with(cx, |panel, cx| {
            assert!(
                panel.active_conversation_view().is_none(),
                "empty workspaces should not create agent threads"
            );
            assert!(
                panel.draft_thread.is_none(),
                "empty workspaces should not create draft threads"
            );
            assert!(
                panel.terminals(cx).is_empty(),
                "empty workspaces should not create agent panel terminals"
            );
        });

        cx.update(|_, cx| {
            cx.update_flags(true, vec!["agent-panel-terminal".to_string()]);
        });
        panel.update_in(cx, |panel, window, cx| {
            panel.new_terminal(None, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(cx, |panel, cx| {
            assert!(
                panel.terminals(cx).is_empty(),
                "empty workspaces should not create terminals after the terminal feature is enabled"
            );
            assert_eq!(
                panel
                    .connection_store()
                    .read(cx)
                    .connection_status(&Agent::NativeAgent, cx),
                crate::agent_connection_store::AgentConnectionStatus::Disconnected,
                "empty workspace actions should not start the native agent connection"
            );
        });
    }

    #[gpui::test]
    async fn test_add_selection_to_terminal_thread_pastes_mention(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({ "file.rs": "line one\nline two\nline three\n" }),
        )
        .await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();
        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Make a terminal thread the active conversation. A display-only terminal
        // avoids spawning a real shell; its working directory is supplied directly
        // so the mention resolves relative to it. No agent is started inside it.
        let terminal_id = TerminalId::new();
        panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_display_only_terminal(
                    terminal_id,
                    Some(PathBuf::from("/project")),
                    Some("Terminal".into()),
                    None,
                    None,
                    true,
                    true,
                    false,
                    AgentThreadSource::AgentPanel,
                    window,
                    cx,
                )
            })
            .expect("display-only terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.active_terminal_id(), Some(terminal_id));
            assert!(panel.active_conversation_view().is_none());
        });

        // Open the file in the center pane so the selection comes from a
        // worktree-backed editor (with a project path).
        workspace
            .update_in(&mut cx, |workspace, window, cx| {
                workspace.open_paths(
                    vec![PathBuf::from("/project/file.rs")],
                    workspace::OpenOptions::default(),
                    None,
                    window,
                    cx,
                )
            })
            .await;
        cx.run_until_parked();

        let editor = workspace.update(&mut cx, |workspace, cx| {
            workspace
                .active_item(cx)
                .and_then(|item| item.act_as::<Editor>(cx))
                .expect("opened file should be an editor")
        });

        cx.focus(&editor);
        cx.run_until_parked();

        let terminal = panel.read_with(&cx, |panel, cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should exist")
                .view
                .read(cx)
                .terminal()
                .clone()
        });
        // Drop any input the terminal may have received during setup.
        terminal.update(&mut cx, |terminal, _| {
            terminal.take_input_log();
        });

        // With only a cursor and nothing highlighted, the action is a no-op and
        // must not paste anything into the terminal.
        workspace.update_in(&mut cx, |_, window, cx| {
            window.dispatch_action(AddSelectionToThread.boxed_clone(), cx);
        });
        cx.run_until_parked();
        let pasted_without_selection =
            terminal.update(&mut cx, |terminal, _| terminal.take_input_log());
        assert!(
            pasted_without_selection.is_empty(),
            "no selection should paste nothing, got {pasted_without_selection:?}"
        );

        // Now highlight a portion of the file: from the start of line 2 into line 3.
        editor.update_in(&mut cx, |editor, window, cx| {
            editor.change_selections(Default::default(), window, cx, |selections| {
                selections.select_ranges([text::Point::new(1, 0)..text::Point::new(2, 4)]);
            });
        });
        cx.run_until_parked();

        workspace.update_in(&mut cx, |_, window, cx| {
            window.dispatch_action(AddSelectionToThread.boxed_clone(), cx);
        });
        cx.run_until_parked();

        let pasted: String = terminal
            .update(&mut cx, |terminal, _| terminal.take_input_log())
            .into_iter()
            .map(|bytes| String::from_utf8(bytes).expect("pasted bytes should be valid UTF-8"))
            .collect();

        // Lines are 1-based and inclusive; the path is presented as
        // `<rel-path>:<start>-<end>`, with a trailing space.
        assert_eq!(pasted, "file.rs:2-3 ");
    }

    async fn setup_panel(cx: &mut TestAppContext) -> (Entity<AgentPanel>, VisualTestContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        (panel, cx)
    }

    async fn setup_visible_panel(
        cx: &mut TestAppContext,
    ) -> (Entity<AgentPanel>, VisualTestContext) {
        setup_visible_panel_with_sidebar(cx, true).await
    }

    async fn setup_visible_panel_with_sidebar(
        cx: &mut TestAppContext,
        threads_list_active: bool,
    ) -> (Entity<AgentPanel>, VisualTestContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            AgentSettings::override_global(
                AgentSettings {
                    notify_when_agent_waiting: NotifyWhenAgentWaiting::PrimaryScreen,
                    ..AgentSettings::get_global(cx).clone()
                },
                cx,
            );
        });

        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);
        register_test_sidebar(threads_list_active, &mut cx);

        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            workspace.focus_panel::<AgentPanel>(window, cx);
            panel
        });

        (panel, cx)
    }

    fn expected_terminal_drop_text(paths: &[PathBuf]) -> String {
        let mut text = String::new();
        for path in paths {
            text.push(' ');
            text.push_str(&shlex::try_quote(path.to_str().unwrap()).unwrap());
        }
        text.push(' ');
        text
    }

    #[gpui::test]
    async fn test_terminal_external_image_drop_writes_path(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.update(|_, cx| {
            cx.update_flags(true, vec!["agent-panel-terminal".to_string()]);
        });

        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Image Upload", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        let terminal = panel.read_with(&cx, |panel, cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel")
                .view
                .read(cx)
                .terminal()
                .clone()
        });
        terminal.update(&mut cx, |terminal, _cx| terminal.take_input_log());

        let image_path = PathBuf::from("/tmp/dropped-image.png");
        panel.update_in(&mut cx, |panel, window, cx| {
            let external_paths = ExternalPaths(vec![image_path.clone()].into());
            panel.paste_external_paths_into_active_terminal(&external_paths, window, cx);
        });

        let mut input_log = terminal.update(&mut cx, |terminal, _cx| terminal.take_input_log());
        assert_eq!(input_log.len(), 1, "expected one write to the terminal");
        let written =
            String::from_utf8(input_log.remove(0)).expect("terminal write should be valid UTF-8");
        assert_eq!(
            written,
            expected_terminal_drop_text(std::slice::from_ref(&image_path))
        );
    }

    #[gpui::test]
    async fn test_terminal_external_paths_drop_handler_writes_image_path(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.update(|_, cx| {
            cx.update_flags(true, vec!["agent-panel-terminal".to_string()]);
        });

        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Image Upload", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        let terminal = panel.read_with(&cx, |panel, cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel")
                .view
                .read(cx)
                .terminal()
                .clone()
        });
        terminal.update(&mut cx, |terminal, _cx| terminal.take_input_log());

        let image_path = PathBuf::from("/tmp/dropped-image.png");
        panel.update_in(&mut cx, |panel, window, cx| {
            let external_paths = ExternalPaths(vec![image_path.clone()].into());
            panel.handle_external_paths_drop(&external_paths, window, cx);
        });

        let mut input_log = terminal.update(&mut cx, |terminal, _cx| terminal.take_input_log());
        assert_eq!(input_log.len(), 1, "expected one write to the terminal");
        let written =
            String::from_utf8(input_log.remove(0)).expect("terminal write should be valid UTF-8");
        assert_eq!(
            written,
            expected_terminal_drop_text(std::slice::from_ref(&image_path))
        );
    }

    #[gpui::test]
    async fn test_external_file_drop_on_thread_does_not_paste_into_later_terminal(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            cx.update_flags(true, vec!["agent-panel-terminal".to_string()]);
        });

        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        fs.insert_tree("/project", json!({ "file.txt": "content" }))
            .await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        open_thread_with_connection(&panel, StubAgentConnection::new(), &mut cx);
        let thread_id = active_thread_id(&panel, &cx);

        let file_path = PathBuf::from("/project/file.txt");
        panel.update_in(&mut cx, |panel, window, cx| {
            let external_paths = ExternalPaths(vec![file_path.clone()].into());
            panel.handle_external_paths_drop(&external_paths, window, cx);
        });

        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Drop Target", true, window, cx)
            })
            .expect("test terminal should be inserted");
        let terminal = panel.read_with(&cx, |panel, cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel")
                .view
                .read(cx)
                .terminal()
                .clone()
        });
        terminal.update(&mut cx, |terminal, _cx| terminal.take_input_log());

        cx.run_until_parked();

        let input_log = terminal.update(&mut cx, |terminal, _cx| terminal.take_input_log());
        assert!(
            input_log.is_empty(),
            "thread drop completion should not write to the active terminal"
        );

        let expected_uri = MentionUri::File {
            abs_path: file_path,
        }
        .to_uri()
        .to_string();
        let expected_text = format!("[@file.txt]({expected_uri}) ");
        let actual_text = panel.read_with(&cx, |panel, cx| panel.editor_text(thread_id, cx));
        assert_eq!(actual_text.as_deref(), Some(expected_text.as_str()));
    }

    #[gpui::test]
    async fn test_terminal_entry_kind_controls_new_entry(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        panel.read_with(&cx, |panel, cx| {
            assert!(panel.project.read(cx).supports_terminal(cx));
            assert!(!panel.should_create_terminal_for_new_entry(cx));
        });

        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Dev Server", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            assert_eq!(panel.active_terminal_id(), Some(terminal_id));
            assert!(panel.has_terminal(terminal_id));
            assert!(panel.should_create_terminal_for_new_entry(cx));
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "Dev Server");
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.activate_new_thread(false, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            assert_eq!(panel.active_terminal_id(), None);
            assert!(panel.has_terminal(terminal_id));
            assert!(!panel.should_create_terminal_for_new_entry(cx));
        });
    }

    #[gpui::test]
    async fn test_skills_menu_entry_shows_manage_skills_shortcut(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            let default_key_bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-macos.json",
                cx,
            )
            .unwrap();
            cx.bind_keys(default_key_bindings);
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        open_thread_with_connection(&panel, StubAgentConnection::new(), &mut cx);
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.focus_panel::<AgentPanel>(window, cx);
        });
        cx.run_until_parked();

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.toggle_options_menu(&ToggleOptionsMenu, window, cx);
        });
        cx.run_until_parked();

        assert!(
            cx.debug_bounds("MENU_ITEM-Skills").is_some(),
            "Skills menu item should be visible"
        );
        assert!(
            cx.debug_bounds("KEY_BINDING-l").is_some(),
            "Skills menu item should show the ManageSkills shortcut"
        );
    }

    #[gpui::test]
    async fn test_terminal_title_omits_placeholder_title(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "");
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            assert_eq!(terminal.title(cx).as_ref(), "");
        });

        let terminal_view = panel.read_with(&cx, |panel, _cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel")
                .view
                .clone()
        });
        let terminal_entity =
            terminal_view.read_with(&cx, |terminal_view, _cx| terminal_view.terminal().clone());
        terminal_entity.update(&mut cx, |_terminal, cx| {
            cx.emit(TerminalEvent::TitleChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "");
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            assert_eq!(terminal.title(cx).as_ref(), "");
        });

        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "Shell Breadcrumb".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "Shell Breadcrumb");
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            assert_eq!(terminal.title(cx).as_ref(), "Shell Breadcrumb");
        });
    }

    #[gpui::test]
    async fn test_title_edit_affordance_matches_threads_and_terminals(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        panel.update_in(&mut cx, |panel, window, cx| {
            assert!(matches!(
                panel.visible_surface(),
                VisibleSurface::AgentThread(_)
            ));
            assert!(panel.should_show_title_edit(window, cx));
        });

        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Dev Server", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        panel.update_in(&mut cx, |panel, window, cx| {
            assert!(matches!(
                panel.visible_surface(),
                VisibleSurface::Terminal(_)
            ));
            assert!(panel.should_show_title_edit(window, cx));

            panel.edit_terminal_title(terminal_id, window, cx);
            assert!(!panel.should_show_title_edit(window, cx));
        });
    }

    #[gpui::test]
    async fn test_restored_terminal_uses_metadata_title_until_shell_title_arrives(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        let terminal_id = TerminalId::new();
        let now = Utc::now();
        let metadata = TerminalThreadMetadata {
            terminal_id,
            title: "Persisted Shell Title".into(),
            custom_title: None,
            created_at: now,
            worktree_paths: WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from(
                "/project",
            )])),
            remote_connection: None,
            working_directory: None,
            inline_context: None,
        };

        panel.update_in(&mut cx, |panel, window, cx| {
            panel
                .restore_test_terminal(metadata, true, AgentThreadSource::Sidebar, None, window, cx)
                .expect("test terminal should be restored");
        });
        cx.run_until_parked();

        let terminal_view = panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "Persisted Shell Title");
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should be restored")
                .view
                .clone()
        });

        let terminal_entity =
            terminal_view.read_with(&cx, |terminal_view, _cx| terminal_view.terminal().clone());
        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "Fresh Shell Title".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "Fresh Shell Title");
        });
    }

    #[gpui::test]
    async fn test_restored_terminal_selects_without_focusing(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        let terminal_id = TerminalId::new();
        let now = Utc::now();
        let metadata = TerminalThreadMetadata {
            terminal_id,
            title: "Persisted Shell Title".into(),
            custom_title: None,
            created_at: now,
            worktree_paths: WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from(
                "/project",
            )])),
            remote_connection: None,
            working_directory: None,
            inline_context: None,
        };

        panel.update_in(&mut cx, |panel, window, cx| {
            panel
                .restore_test_terminal(
                    metadata,
                    false,
                    AgentThreadSource::Sidebar,
                    None,
                    window,
                    cx,
                )
                .expect("test terminal should be restored");
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.active_terminal_id(), Some(terminal_id));
        });
    }

    #[gpui::test]
    async fn test_terminal_working_directory_uses_active_workspace_while_workspace_is_updating(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, mut cx) = setup_workspace_panel(cx).await;
        panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Dev Server", false, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            assert_eq!(panel.last_created_entry_kind, AgentPanelEntryKind::Terminal);
            assert!(panel.should_create_terminal_for_new_entry(cx));
        });

        workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = workspace
                .panel::<AgentPanel>(cx)
                .expect("agent panel should be registered in workspace");
            panel.read_with(cx, |panel, cx| {
                panel.terminal_working_directory(Some(workspace), cx);
            });
            workspace.focus_panel::<AgentPanel>(window, cx);
        });

        panel.read_with(&cx, |panel, cx| {
            assert_eq!(panel.last_created_entry_kind, AgentPanelEntryKind::Terminal);
            assert!(panel.should_create_terminal_for_new_entry(cx));
        });
    }

    #[gpui::test]
    async fn test_terminal_title_editor_is_created_only_while_editing(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Dev Server", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            assert!(terminal.title_editor.is_none());
        });

        panel.update(&mut cx, |panel, cx| {
            panel.refresh_terminal_metadata(terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            assert!(terminal.title_editor.is_none());
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.edit_terminal_title(terminal_id, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            let title_editor = terminal
                .title_editor
                .as_ref()
                .expect("terminal title editor should be active while editing");
            assert_eq!(title_editor.read(cx).text(cx), "Dev Server");
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.stop_editing_terminal_title(terminal_id, false, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            assert!(terminal.title_editor.is_none());
        });
    }

    #[gpui::test]
    async fn test_terminal_title_editor_does_not_set_custom_title_when_unchanged(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Initial Custom Title", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        let terminal_view = panel.read_with(&cx, |panel, _cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel")
                .view
                .clone()
        });
        terminal_view.update(&mut cx, |terminal_view, cx| {
            terminal_view.set_custom_title(None, cx);
        });
        let terminal_entity =
            terminal_view.read_with(&cx, |terminal_view, _cx| terminal_view.terminal().clone());
        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "Shell Breadcrumb".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "Shell Breadcrumb");
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.edit_terminal_title(terminal_id, window, cx);
        });
        cx.run_until_parked();

        let title_editor = panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            let title_editor = terminal
                .title_editor
                .as_ref()
                .expect("terminal title editor should be active while editing")
                .clone();
            assert_eq!(title_editor.read(cx).text(cx), "Shell Breadcrumb");
            title_editor
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.handle_terminal_title_editor_event(
                terminal_id,
                &title_editor,
                &editor::EditorEvent::BufferEdited,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        terminal_view.read_with(&cx, |terminal_view, _cx| {
            assert!(terminal_view.custom_title().is_none());
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.stop_editing_terminal_title(terminal_id, false, window, cx);
        });
        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "Updated Shell Breadcrumb".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "Updated Shell Breadcrumb");
        });
    }

    #[gpui::test]
    async fn test_terminal_custom_title_recomposes_with_live_spinner(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Fix bug", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        let terminal_entity = panel.read_with(&cx, |panel, _cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel")
                .view
                .clone()
        });
        let terminal_entity =
            terminal_entity.read_with(&cx, |terminal_view, _cx| terminal_view.terminal().clone());

        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "⠋ Thinking".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "⠋ Fix bug");
            let metadata = panel
                .terminal_metadata(terminal_id, cx)
                .expect("terminal metadata should be available");
            assert_eq!(metadata.title.as_ref(), "⠋ Thinking");
            assert_eq!(
                metadata.custom_title.as_ref().map(|title| title.as_ref()),
                Some("Fix bug")
            );
            assert_eq!(metadata.display_title().as_ref(), "⠋ Fix bug");
        });

        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "⠙ Thinking".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "⠙ Fix bug");
            let metadata = panel
                .terminal_metadata(terminal_id, cx)
                .expect("terminal metadata should be available");
            assert_eq!(metadata.title.as_ref(), "⠙ Thinking");
            assert_eq!(metadata.display_title().as_ref(), "⠙ Fix bug");
        });

        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "Thinking".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "Fix bug");
            let metadata = panel
                .terminal_metadata(terminal_id, cx)
                .expect("terminal metadata should be available");
            assert_eq!(metadata.title.as_ref(), "Thinking");
            assert_eq!(metadata.display_title().as_ref(), "Fix bug");
        });
    }

    #[gpui::test]
    async fn test_terminal_title_editor_excludes_spinner_prefix(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Initial Custom Title", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        let terminal_view = panel.read_with(&cx, |panel, _cx| {
            panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel")
                .view
                .clone()
        });
        terminal_view.update(&mut cx, |terminal_view, cx| {
            terminal_view.set_custom_title(None, cx);
        });
        let terminal_entity =
            terminal_view.read_with(&cx, |terminal_view, _cx| terminal_view.terminal().clone());
        terminal_entity.update(&mut cx, |terminal, cx| {
            terminal.breadcrumb_text = "⠋ Thinking".to_string();
            cx.emit(TerminalEvent::BreadcrumbsChanged);
        });
        cx.run_until_parked();

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.edit_terminal_title(terminal_id, window, cx);
        });
        cx.run_until_parked();

        let title_editor = panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            let title_editor = terminal
                .title_editor
                .as_ref()
                .expect("terminal title editor should be active while editing")
                .clone();
            assert_eq!(title_editor.read(cx).text(cx), "Thinking");
            title_editor
        });

        title_editor.update_in(&mut cx, |editor, window, cx| {
            editor.set_text("Fix bug", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.handle_terminal_title_editor_event(
                terminal_id,
                &title_editor,
                &editor::EditorEvent::BufferEdited,
                window,
                cx,
            );
        });
        cx.run_until_parked();

        terminal_view.read_with(&cx, |terminal_view, _cx| {
            assert_eq!(terminal_view.custom_title(), Some("Fix bug"));
        });
        panel.read_with(&cx, |panel, cx| {
            let terminals = panel.terminals(cx);
            assert_eq!(terminals.len(), 1);
            assert_eq!(terminals[0].title.as_ref(), "⠋ Fix bug");
            let metadata = panel
                .terminal_metadata(terminal_id, cx)
                .expect("terminal metadata should be available");
            assert_eq!(metadata.title.as_ref(), "⠋ Thinking");
            assert_eq!(
                metadata.custom_title.as_ref().map(|title| title.as_ref()),
                Some("Fix bug")
            );
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.stop_editing_terminal_title(terminal_id, false, window, cx);
            panel.edit_terminal_title(terminal_id, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals
                .get(&terminal_id)
                .expect("terminal should remain in the panel");
            let title_editor = terminal
                .title_editor
                .as_ref()
                .expect("terminal title editor should be active while editing");
            assert_eq!(title_editor.read(cx).text(cx), "Fix bug");
        });
    }

    #[gpui::test]
    async fn test_terminal_bell_marks_and_activation_clears_notification(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        let first_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Build", true, window, cx)
            })
            .expect("first test terminal should be inserted");
        let second_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Server", true, window, cx)
            })
            .expect("second test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.active_terminal_id(), Some(second_terminal_id));
        });

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(first_terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let first_terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == first_terminal_id)
                .expect("first terminal should remain in the panel");
            assert!(first_terminal.has_notification);
        });

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.activate_terminal(first_terminal_id, true, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let first_terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == first_terminal_id)
                .expect("first terminal should remain in the panel");
            assert!(!first_terminal.has_notification);
        });
    }

    #[gpui::test]
    async fn test_visible_terminal_bell_is_suppressed(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Claude", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        cx.update(|window, cx| {
            assert!(window.is_window_active());
            assert!(panel.read(cx).focus_handle(cx).contains_focused(window, cx));
        });

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == terminal_id)
                .expect("terminal should remain in the panel");
            assert!(!terminal.has_notification);
        });
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
    }

    #[gpui::test]
    async fn test_visible_terminal_bell_is_suppressed_without_focus(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Claude", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        let workspace = cx.update(|window, cx| {
            window
                .root::<MultiWorkspace>()
                .flatten()
                .expect("test window should have a MultiWorkspace root")
                .read(cx)
                .workspace()
                .clone()
        });
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.focus_handle(cx).focus(window, cx);
        });
        cx.update(|window, cx| {
            assert!(window.is_window_active());
            assert!(workspace.read(cx).focus_handle(cx).is_focused(window));
            assert!(!panel.read(cx).focus_handle(cx).contains_focused(window, cx));
        });

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == terminal_id)
                .expect("terminal should remain in the panel");
            assert!(!terminal.has_notification);
        });
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
    }

    #[gpui::test]
    async fn test_terminal_bell_marks_without_popup_when_sidebar_open(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let first_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Build", true, window, cx)
            })
            .expect("first test terminal should be inserted");
        let second_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Server", true, window, cx)
            })
            .expect("second test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.active_terminal_id(), Some(second_terminal_id));
        });
        cx.update(|window, cx| {
            let multi_workspace = window
                .root::<MultiWorkspace>()
                .flatten()
                .expect("test window should have a MultiWorkspace root");
            multi_workspace.update(cx, |multi_workspace, cx| {
                multi_workspace.open_sidebar(cx);
            });
        });
        cx.run_until_parked();

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(first_terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let first_terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == first_terminal_id)
                .expect("first terminal should remain in the panel");
            assert!(first_terminal.has_notification);
        });
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
    }

    #[gpui::test]
    async fn test_terminal_bell_notifies_when_sidebar_history_open(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel_with_sidebar(cx, false).await;
        let first_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Build", true, window, cx)
            })
            .expect("first test terminal should be inserted");
        let second_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Server", true, window, cx)
            })
            .expect("second test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.active_terminal_id(), Some(second_terminal_id));
        });
        cx.update(|window, cx| {
            let multi_workspace = window
                .root::<MultiWorkspace>()
                .flatten()
                .expect("test window should have a MultiWorkspace root");
            multi_workspace.update(cx, |multi_workspace, cx| {
                multi_workspace.open_sidebar(cx);
            });
        });
        cx.run_until_parked();

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(first_terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let first_terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == first_terminal_id)
                .expect("first terminal should remain in the panel");
            assert!(first_terminal.has_notification);
        });
        cx.windows()
            .iter()
            .find_map(|window| window.downcast::<AgentNotification>())
            .expect("terminal bell should notify when the sidebar thread list is hidden");
    }

    #[gpui::test]
    async fn test_terminal_notification_dismissed_when_sidebar_opens(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let first_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Build", true, window, cx)
            })
            .expect("first test terminal should be inserted");
        let second_terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Server", true, window, cx)
            })
            .expect("second test terminal should be inserted");
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.active_terminal_id(), Some(second_terminal_id));
        });
        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(first_terminal_id, cx);
        });
        cx.run_until_parked();

        cx.windows()
            .iter()
            .find_map(|window| window.downcast::<AgentNotification>())
            .expect("inactive terminal bell should show a notification");

        cx.update(|window, cx| {
            let multi_workspace = window
                .root::<MultiWorkspace>()
                .flatten()
                .expect("test window should have a MultiWorkspace root");
            multi_workspace.update(cx, |multi_workspace, cx| {
                multi_workspace.open_sidebar(cx);
            });
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let first_terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == first_terminal_id)
                .expect("first terminal should remain in the panel");
            assert!(first_terminal.has_notification);
        });
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
    }

    #[gpui::test]
    async fn test_focused_terminal_bell_notifies_when_window_inactive(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Claude", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        cx.update(|window, cx| {
            assert!(window.is_window_active());
            assert!(panel.read(cx).focus_handle(cx).contains_focused(window, cx));
        });
        cx.deactivate_window();
        cx.update(|window, _cx| {
            assert!(!window.is_window_active());
        });

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == terminal_id)
                .expect("terminal should remain in the panel");
            assert!(terminal.has_notification);
        });
        cx.windows()
            .iter()
            .find_map(|window| window.downcast::<AgentNotification>())
            .expect("background terminal bell should show a notification");
    }

    #[gpui::test]
    async fn test_active_terminal_notification_clears_when_window_reactivates(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Claude", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        cx.deactivate_window();
        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == terminal_id)
                .expect("terminal should remain in the panel");
            assert!(terminal.has_notification);
        });
        cx.windows()
            .iter()
            .find_map(|window| window.downcast::<AgentNotification>())
            .expect("background terminal bell should show a notification");

        cx.update(|window, _cx| {
            window.activate_window();
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == terminal_id)
                .expect("terminal should remain in the panel");
            assert!(!terminal.has_notification);
        });
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
    }

    #[gpui::test]
    async fn test_terminal_notification_dismissed_when_active_terminal_becomes_visible(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.update(|_window, cx| {
            AgentSettings::override_global(
                AgentSettings {
                    notify_when_agent_waiting: NotifyWhenAgentWaiting::PrimaryScreen,
                    ..AgentSettings::get_global(cx).clone()
                },
                cx,
            );
        });
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Claude", true, window, cx)
            })
            .expect("test terminal should be inserted");
        cx.run_until_parked();

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(terminal_id, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == terminal_id)
                .expect("terminal should remain in the panel");
            assert!(terminal.has_notification);
        });
        cx.windows()
            .iter()
            .find_map(|window| window.downcast::<AgentNotification>())
            .expect("hidden terminal bell should show a notification");

        let workspace = cx.update(|window, cx| {
            window
                .root::<MultiWorkspace>()
                .flatten()
                .expect("test window should have a MultiWorkspace root")
                .read(cx)
                .workspace()
                .clone()
        });
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace.focus_panel::<AgentPanel>(window, cx);
        });
        cx.run_until_parked();

        panel.read_with(&cx, |panel, cx| {
            let terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == terminal_id)
                .expect("terminal should remain in the panel");
            assert!(!terminal.has_notification);
        });
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
    }

    #[gpui::test]
    async fn test_terminal_notification_closed_when_panel_dropped(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.update(|_window, cx| {
            AgentSettings::override_global(
                AgentSettings {
                    notify_when_agent_waiting: NotifyWhenAgentWaiting::PrimaryScreen,
                    ..AgentSettings::get_global(cx).clone()
                },
                cx,
            );
        });
        let terminal_id = panel
            .update_in(&mut cx, |panel, window, cx| {
                panel.insert_test_terminal("Claude", true, window, cx)
            })
            .expect("test terminal should be inserted");
        let weak_panel = panel.downgrade();
        cx.run_until_parked();

        panel.update(&mut cx, |panel, cx| {
            panel.emit_test_terminal_bell(terminal_id, cx);
        });
        cx.run_until_parked();

        cx.windows()
            .iter()
            .find_map(|window| window.downcast::<AgentNotification>())
            .expect("hidden terminal bell should show a notification");

        drop(panel);
        cx.update(|_window, _cx| {});
        cx.run_until_parked();

        assert!(
            !weak_panel.is_upgradable(),
            "agent panel should be released after dropping the last handle"
        );
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
    }

    #[gpui::test]
    async fn test_terminal_notification_view_activates_terminal_workspace(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            cx.update_flags(true, vec!["agent-panel-terminal".to_string()]);
            AgentSettings::override_global(
                AgentSettings {
                    notify_when_agent_waiting: NotifyWhenAgentWaiting::PrimaryScreen,
                    ..AgentSettings::get_global(cx).clone()
                },
                cx,
            );
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        fs.insert_tree("/project_b", json!({ "file.txt": "" }))
            .await;
        let project_a = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;
        let project_b = Project::test(fs, [Path::new("/project_b")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));
        let workspace_a = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        let first_terminal_id = panel_a
            .update_in(cx, |panel, window, cx| {
                panel.insert_test_terminal("Build", true, window, cx)
            })
            .expect("first test terminal should be inserted");
        let second_terminal_id = panel_a
            .update_in(cx, |panel, window, cx| {
                panel.insert_test_terminal("Server", true, window, cx)
            })
            .expect("second test terminal should be inserted");
        cx.run_until_parked();

        multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                assert_eq!(multi_workspace.workspace(), &workspace_b);
            })
            .unwrap();
        panel_a.read_with(cx, |panel, _cx| {
            assert_eq!(panel.active_terminal_id(), Some(second_terminal_id));
        });

        panel_a.update(cx, |panel, cx| {
            panel.emit_test_terminal_bell(first_terminal_id, cx);
        });
        cx.run_until_parked();

        let notification = cx
            .windows()
            .iter()
            .find_map(|window| window.downcast::<AgentNotification>())
            .expect("terminal bell should show a notification");
        notification
            .update(cx, |notification, _window, cx| notification.accept(cx))
            .unwrap();
        assert!(
            cx.windows()
                .iter()
                .all(|window| window.downcast::<AgentNotification>().is_none())
        );
        cx.run_until_parked();

        multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                assert_eq!(multi_workspace.workspace(), &workspace_a);
            })
            .unwrap();
        panel_a.read_with(cx, |panel, cx| {
            assert_eq!(panel.active_terminal_id(), Some(first_terminal_id));
            let first_terminal = panel
                .terminals(cx)
                .into_iter()
                .find(|terminal| terminal.id == first_terminal_id)
                .expect("first terminal should remain in the panel");
            assert!(!first_terminal.has_notification);
        });
    }

    #[gpui::test]
    async fn test_running_thread_retained_when_navigating_away(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;

        let connection_a = StubAgentConnection::new();
        open_thread_with_connection(&panel, connection_a.clone(), &mut cx);
        send_message(&panel, &mut cx);

        let session_id_a = active_session_id(&panel, &cx);
        let thread_id_a = active_thread_id(&panel, &cx);

        // Send a chunk to keep thread A generating (don't end the turn).
        cx.update(|_, cx| {
            connection_a.send_update(
                session_id_a.clone(),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new("chunk".into())),
                cx,
            );
        });
        cx.run_until_parked();

        // Verify thread A is generating.
        panel.read_with(&cx, |panel, cx| {
            let thread = panel.active_agent_thread(cx).unwrap();
            assert_eq!(thread.read(cx).status(), ThreadStatus::Generating);
            assert!(panel.retained_threads.is_empty());
        });

        // Open a new thread B — thread A should be retained in background.
        let connection_b = StubAgentConnection::new();
        open_thread_with_connection(&panel, connection_b, &mut cx);

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(
                panel.retained_threads.len(),
                1,
                "Running thread A should be retained in retained_threads"
            );
            assert!(
                panel.retained_threads.contains_key(&thread_id_a),
                "Retained thread should be keyed by thread A's thread ID"
            );
        });
    }

    #[gpui::test]
    async fn test_idle_non_loadable_thread_retained_when_navigating_away(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;

        let connection_a = StubAgentConnection::new();
        connection_a.set_next_prompt_updates(vec![acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new("Response".into()),
        )]);
        open_thread_with_connection(&panel, connection_a, &mut cx);
        send_message(&panel, &mut cx);

        let weak_view_a = panel.read_with(&cx, |panel, _cx| {
            panel.active_conversation_view().unwrap().downgrade()
        });
        let thread_id_a = active_thread_id(&panel, &cx);

        // Thread A should be idle (auto-completed via set_next_prompt_updates).
        panel.read_with(&cx, |panel, cx| {
            let thread = panel.active_agent_thread(cx).unwrap();
            assert_eq!(thread.read(cx).status(), ThreadStatus::Idle);
        });

        // Open a new thread B — thread A should be retained because it is not loadable.
        let connection_b = StubAgentConnection::new();
        open_thread_with_connection(&panel, connection_b, &mut cx);

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(
                panel.retained_threads.len(),
                1,
                "Idle non-loadable thread A should be retained in retained_threads"
            );
            assert!(
                panel.retained_threads.contains_key(&thread_id_a),
                "Retained thread should be keyed by thread A's thread ID"
            );
        });

        assert!(
            weak_view_a.upgrade().is_some(),
            "Idle non-loadable ConnectionView should still be retained"
        );
    }

    #[gpui::test]
    async fn test_background_thread_promoted_via_load(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;

        let connection_a = StubAgentConnection::new();
        open_thread_with_connection(&panel, connection_a.clone(), &mut cx);
        send_message(&panel, &mut cx);

        let session_id_a = active_session_id(&panel, &cx);
        let thread_id_a = active_thread_id(&panel, &cx);

        // Keep thread A generating.
        cx.update(|_, cx| {
            connection_a.send_update(
                session_id_a.clone(),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new("chunk".into())),
                cx,
            );
        });
        cx.run_until_parked();

        // Open thread B — thread A goes to background.
        let connection_b = StubAgentConnection::new();
        open_thread_with_connection(&panel, connection_b, &mut cx);
        send_message(&panel, &mut cx);

        let thread_id_b = active_thread_id(&panel, &cx);

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.retained_threads.len(), 1);
            assert!(panel.retained_threads.contains_key(&thread_id_a));
        });

        // Load thread A back via load_agent_thread — should promote from background.
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.load_agent_thread(
                panel.selected_agent(cx),
                thread_id_a,
                None,
                None,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });

        // Thread A should now be the active view, promoted from background.
        let active_session = active_session_id(&panel, &cx);
        assert_eq!(
            active_session, session_id_a,
            "Thread A should be the active thread after promotion"
        );

        panel.read_with(&cx, |panel, _cx| {
            assert!(
                !panel.retained_threads.contains_key(&thread_id_a),
                "Promoted thread A should no longer be in retained_threads"
            );
            assert!(
                panel.retained_threads.contains_key(&thread_id_b),
                "Thread B (idle, non-loadable) should remain retained in retained_threads"
            );
        });
    }

    #[gpui::test]
    async fn test_reopening_visible_thread_keeps_thread_usable(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        cx.run_until_parked();

        panel.update(&mut cx, |panel, cx| {
            panel.connection_store.update(cx, |store, cx| {
                store.restart_connection(
                    Agent::NativeAgent,
                    Rc::new(StubAgentServer::new(SessionTrackingConnection::new())),
                    cx,
                );
            });
        });
        cx.run_until_parked();

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.external_thread(
                Some(Agent::NativeAgent),
                None,
                None,
                None,
                None,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        send_message(&panel, &mut cx);

        let session_id = active_session_id(&panel, &cx);

        panel.update_in(&mut cx, |panel, window, cx| {
            panel.open_thread(session_id.clone(), None, None, window, cx);
        });
        cx.run_until_parked();

        send_message(&panel, &mut cx);

        panel.read_with(&cx, |panel, cx| {
            let active_view = panel
                .active_conversation_view()
                .expect("visible conversation should remain open after reopening");
            let connected = active_view
                .read(cx)
                .as_connected()
                .expect("visible conversation should still be connected in the UI");
            assert!(
                !connected.has_thread_error(cx),
                "reopening an already-visible session should keep the thread usable"
            );
        });
    }

    #[gpui::test]
    async fn test_initial_content_for_thread_summary_uses_own_session_id(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let source_session_id = acp::SessionId::new("source-thread-session");
        let source_title: SharedString = "Source Thread Title".into();
        let db_thread = agent::DbThread {
            recipe_scan: false,
            doctor_repair_proposal: false,
            goal: None,
            title: source_title.clone(),
            messages: Vec::new(),
            updated_at: Utc::now(),
            detailed_summary: None,
            initial_project_snapshot: None,
            cumulative_token_usage: Default::default(),
            request_token_usage: HashMap::default(),
            model: None,
            profile: None,
            subagent_context: None,
            speed: None,
            thinking_enabled: false,
            thinking_effort: None,
            draft_prompt: None,
            ui_scroll_position: None,
            sandboxed_terminal_temp_dir: None,
            sandbox_grants: Default::default(),
        };

        let thread_store = cx.update(|cx| ThreadStore::global(cx));
        thread_store
            .update(cx, |store, cx| {
                store.save_thread(
                    source_session_id.clone(),
                    db_thread,
                    PathList::default(),
                    cx,
                )
            })
            .await
            .expect("saving source thread should succeed");
        cx.run_until_parked();

        thread_store.read_with(cx, |store, _cx| {
            let entry = store
                .thread_from_session_id(&source_session_id)
                .expect("saved thread should be listed in the store");
            assert!(
                entry.parent_session_id.is_none(),
                "saved thread is a root thread with no parent session"
            );
        });

        let content = cx
            .update(|cx| {
                AgentPanel::initial_content_for_thread_summary(source_session_id.clone(), cx)
            })
            .expect("initial content should be produced for a root thread");

        match content {
            AgentInitialContent::ThreadSummary { session_id, title } => {
                assert_eq!(
                    session_id, source_session_id,
                    "thread-summary mention should use the source thread's own session id"
                );
                assert_eq!(title, Some(source_title.clone()));
            }
            _ => panic!("expected AgentInitialContent::ThreadSummary"),
        }

        // Unknown session ids should still produce no content.
        let missing = cx.update(|cx| {
            AgentPanel::initial_content_for_thread_summary(
                acp::SessionId::new("does-not-exist"),
                cx,
            )
        });
        assert!(
            missing.is_none(),
            "unknown session ids should not produce initial content"
        );
    }

    #[gpui::test]
    async fn review_returns_to_its_own_composer_without_toggling_focus(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let connection = StubAgentConnection::new().with_supports_load_session(true);
        let (review_session, review_id) =
            open_generating_thread_with_loadable_connection(&panel, &connection, &mut cx);
        connection.end_turn(review_session, acp::StopReason::EndTurn);
        cx.run_until_parked();
        let reviewed = panel.read_with(&cx, |panel, cx| {
            panel.active_agent_thread(cx).expect("reviewed thread")
        });
        let (active_session, active_id) =
            open_generating_thread_with_loadable_connection(&panel, &connection, &mut cx);
        connection.end_turn(active_session, acp::StopReason::EndTurn);
        cx.run_until_parked();
        assert_ne!(active_id, review_id);
        for _ in 0..2 {
            panel
                .update_in(&mut cx, |panel, window, cx| {
                    panel.focus_review_composer(&reviewed, window, cx)
                })
                .expect("canonical composer opens");
            cx.run_until_parked();
            assert_eq!(active_thread_id(&panel, &mut cx), review_id);
            panel.read_with(&cx, |panel, cx| {
                assert_eq!(panel.active_agent_thread(cx), Some(reviewed.clone()))
            });
            let composer = panel.read_with(&cx, |panel, cx| {
                panel
                    .active_thread_view(cx)
                    .expect("thread view")
                    .read(cx)
                    .message_editor
                    .clone()
            });
            cx.update(|window, cx| assert!(composer.focus_handle(cx).contains_focused(window, cx)));
        }
    }

    #[gpui::test]
    async fn test_export_loads_one_retained_transcript_without_switching_tasks(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        let connection = StubAgentConnection::new()
            .with_supports_load_session(true)
            .with_agent_id("export-provider".into());
        let (session_id, thread_id) =
            open_generating_thread_with_loadable_connection(&panel, &connection, &mut cx);
        connection.end_turn(session_id.clone(), acp::StopReason::EndTurn);
        cx.run_until_parked();
        let other_connection = StubAgentConnection::new()
            .with_supports_load_session(true)
            .with_agent_id("active-provider".into());
        let (active_session, active_id) =
            open_generating_thread_with_loadable_connection(&panel, &other_connection, &mut cx);
        other_connection.end_turn(active_session.clone(), acp::StopReason::EndTurn);
        cx.run_until_parked();
        let selected_agent = panel.read_with(&cx, |panel, _| panel.selected_agent.clone());
        panel.update(&mut cx, |panel, _| {
            assert!(panel.retained_threads.remove(&thread_id).is_some());
        });
        cx.update(|_, cx| {
            ThreadMetadataStore::global(cx).update(cx, |store, cx| {
                let mut metadata = store.entry(thread_id).expect("saved task metadata").clone();
                metadata.archived = true;
                store.save(metadata, cx);
            });
        });
        let first = panel.update_in(&mut cx, |panel, window, cx| {
            panel.load_transcript_for_export(thread_id, window, cx)
        });
        panel.update(&mut cx, |panel, cx| {
            cx.set_global(MaxIdleRetainedThreads(0));
            panel.cleanup_retained_threads(cx);
            assert!(
                panel.retained_threads.contains_key(&thread_id),
                "cache cleanup must not evict a canonical session while export loads it"
            );
            cx.set_global(MaxIdleRetainedThreads(5));
        });
        let second = panel.update_in(&mut cx, |panel, window, cx| {
            panel.load_transcript_for_export(thread_id, window, cx)
        });
        let first = first.await.expect("closed provider transcript loads");
        let second = second.await.expect("concurrent reader reuses transcript");
        assert_eq!(first.entity_id(), second.entity_id());
        first.read_with(&cx, |view, cx| {
            assert_eq!(view.thread.read(cx).session_id(), &session_id);
        });
        assert_eq!(active_thread_id(&panel, &mut cx), active_id);
        assert_eq!(active_session_id(&panel, &mut cx), active_session);
        panel.read_with(&cx, |panel, cx| {
            assert_eq!(panel.selected_agent, selected_agent);
            assert_eq!(
                panel
                    .conversation_view_for_id(&thread_id, cx)
                    .expect("retained canonical conversation")
                    .read(cx)
                    .root_thread_view()
                    .expect("loaded root")
                    .entity_id(),
                first.entity_id()
            );
            assert!(
                ThreadMetadataStore::global(cx)
                    .read(cx)
                    .entry(thread_id)
                    .expect("task remains")
                    .archived
            );
        });
    }

    #[gpui::test]
    async fn test_export_load_failure_is_retryable_without_switching_tasks(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        let connection = StubAgentConnection::new().with_agent_id("non-loadable-export".into());
        let (session_id, thread_id) =
            open_generating_thread_with_loadable_connection(&panel, &connection, &mut cx);
        connection.end_turn(session_id, acp::StopReason::EndTurn);
        cx.run_until_parked();
        let other_connection = StubAgentConnection::new().with_agent_id("other-export".into());
        let (active_session, active_id) =
            open_generating_thread_with_loadable_connection(&panel, &other_connection, &mut cx);
        other_connection.end_turn(active_session, acp::StopReason::EndTurn);
        cx.run_until_parked();
        panel.update(&mut cx, |panel, _| {
            assert!(panel.retained_threads.remove(&thread_id).is_some());
        });
        for _ in 0..2 {
            let export = panel.update_in(&mut cx, |panel, window, cx| {
                panel.load_transcript_for_export(thread_id, window, cx)
            });
            assert!(export.await.is_err());
            assert_eq!(active_thread_id(&panel, &mut cx), active_id);
        }
    }

    #[gpui::test]
    async fn test_cleanup_retained_threads_keeps_five_most_recent_idle_loadable_threads(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        let connection = StubAgentConnection::new()
            .with_supports_load_session(true)
            .with_agent_id("loadable-stub".into())
            .with_telemetry_id("loadable-stub".into());
        let mut session_ids = Vec::new();
        let mut thread_ids = Vec::new();

        for _ in 0..7 {
            let (session_id, thread_id) =
                open_generating_thread_with_loadable_connection(&panel, &connection, &mut cx);
            session_ids.push(session_id);
            thread_ids.push(thread_id);
        }

        let base_time = Instant::now();

        for session_id in session_ids.iter().take(6) {
            connection.end_turn(session_id.clone(), acp::StopReason::EndTurn);
        }
        cx.run_until_parked();

        panel.update(&mut cx, |panel, cx| {
            for (index, thread_id) in thread_ids.iter().take(6).enumerate() {
                let conversation_view = panel
                    .retained_threads
                    .get(thread_id)
                    .expect("retained thread should exist")
                    .clone();
                conversation_view.update(cx, |view, cx| {
                    view.set_updated_at(base_time + Duration::from_secs(index as u64), cx);
                });
            }
            panel.cleanup_retained_threads(cx);
        });

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(
                panel.retained_threads.len(),
                5,
                "cleanup should keep at most five idle loadable retained threads"
            );
            assert!(
                !panel.retained_threads.contains_key(&thread_ids[0]),
                "oldest idle loadable retained thread should be removed"
            );
            for thread_id in &thread_ids[1..6] {
                assert!(
                    panel.retained_threads.contains_key(thread_id),
                    "more recent idle loadable retained threads should be retained"
                );
            }
            assert!(
                !panel.retained_threads.contains_key(&thread_ids[6]),
                "the active thread should not also be stored as a retained thread"
            );
        });
    }

    #[gpui::test]
    async fn test_cleanup_retained_threads_preserves_idle_non_loadable_threads(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;

        let non_loadable_connection = StubAgentConnection::new();
        let (_non_loadable_session_id, non_loadable_thread_id) =
            open_idle_thread_with_non_loadable_connection(
                &panel,
                &non_loadable_connection,
                &mut cx,
            );

        let loadable_connection = StubAgentConnection::new()
            .with_supports_load_session(true)
            .with_agent_id("loadable-stub".into())
            .with_telemetry_id("loadable-stub".into());
        let mut loadable_session_ids = Vec::new();
        let mut loadable_thread_ids = Vec::new();

        for _ in 0..7 {
            let (session_id, thread_id) = open_generating_thread_with_loadable_connection(
                &panel,
                &loadable_connection,
                &mut cx,
            );
            loadable_session_ids.push(session_id);
            loadable_thread_ids.push(thread_id);
        }

        let base_time = Instant::now();

        for session_id in loadable_session_ids.iter().take(6) {
            loadable_connection.end_turn(session_id.clone(), acp::StopReason::EndTurn);
        }
        cx.run_until_parked();

        panel.update(&mut cx, |panel, cx| {
            for (index, thread_id) in loadable_thread_ids.iter().take(6).enumerate() {
                let conversation_view = panel
                    .retained_threads
                    .get(thread_id)
                    .expect("retained thread should exist")
                    .clone();
                conversation_view.update(cx, |view, cx| {
                    view.set_updated_at(base_time + Duration::from_secs(index as u64), cx);
                });
            }
            panel.cleanup_retained_threads(cx);
        });

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(
                panel.retained_threads.len(),
                6,
                "cleanup should keep the non-loadable idle thread in addition to five loadable ones"
            );
            assert!(
                panel.retained_threads.contains_key(&non_loadable_thread_id),
                "idle non-loadable retained threads should not be cleanup candidates"
            );
            assert!(
                !panel.retained_threads.contains_key(&loadable_thread_ids[0]),
                "oldest idle loadable retained thread should still be removed"
            );
            for thread_id in &loadable_thread_ids[1..6] {
                assert!(
                    panel.retained_threads.contains_key(thread_id),
                    "more recent idle loadable retained threads should be retained"
                );
            }
            assert!(
                !panel.retained_threads.contains_key(&loadable_thread_ids[6]),
                "the active loadable thread should not also be stored as a retained thread"
            );
        });
    }

    #[test]
    fn test_deserialize_agent_variants() {
        // PascalCase (legacy AgentType format, persisted in panel state)
        assert_eq!(
            serde_json::from_str::<Agent>(r#""NativeAgent""#).unwrap(),
            Agent::NativeAgent,
        );
        assert_eq!(
            serde_json::from_str::<Agent>(r#"{"Custom":{"name":"my-agent"}}"#).unwrap(),
            Agent::Custom {
                id: "my-agent".into(),
            },
        );

        // Legacy TextThread variant deserializes to NativeAgent
        assert_eq!(
            serde_json::from_str::<Agent>(r#""TextThread""#).unwrap(),
            Agent::NativeAgent,
        );

        // snake_case (canonical format)
        assert_eq!(
            serde_json::from_str::<Agent>(r#""native_agent""#).unwrap(),
            Agent::NativeAgent,
        );
        assert_eq!(
            serde_json::from_str::<Agent>(r#"{"custom":{"name":"my-agent"}}"#).unwrap(),
            Agent::Custom {
                id: "my-agent".into(),
            },
        );

        // Serialization uses snake_case
        assert_eq!(
            serde_json::to_string(&Agent::NativeAgent).unwrap(),
            r#""native_agent""#,
        );
        assert_eq!(
            serde_json::to_string(&Agent::Custom {
                id: "my-agent".into()
            })
            .unwrap(),
            r#"{"custom":{"name":"my-agent"}}"#,
        );
    }

    #[gpui::test]
    fn test_resolve_worktree_branch_target() {
        let resolved = git_ui_core::worktree_service::resolve_worktree_branch_target(
            &NewWorktreeBranchTarget::ExistingBranch {
                name: "feature".to_string(),
            },
        );
        assert_eq!(resolved, Some("feature".to_string()));

        let resolved = git_ui_core::worktree_service::resolve_worktree_branch_target(
            &NewWorktreeBranchTarget::CurrentBranch,
        );
        assert_eq!(resolved, None);

        let resolved = git_ui_core::worktree_service::resolve_worktree_branch_target(
            &NewWorktreeBranchTarget::RemoteBranch {
                remote_name: "origin".to_string(),
                branch_name: "main".to_string(),
            },
        );
        assert_eq!(resolved, Some("refs/remotes/origin/main".to_string()));
    }

    #[gpui::test]
    async fn test_work_dirs_update_when_worktrees_change(cx: &mut TestAppContext) {
        use crate::thread_metadata_store::ThreadMetadataStore;

        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        // Set up a project with one worktree.
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        let project = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();
        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        // Open thread A and send a message. With empty next_prompt_updates it
        // stays generating, so opening B will move A to retained_threads.
        let connection_a = StubAgentConnection::new().with_agent_id("agent-a".into());
        open_thread_with_custom_connection(&panel, connection_a.clone(), &mut cx);
        send_message(&panel, &mut cx);
        let session_id_a = active_session_id(&panel, &cx);
        let thread_id_a = active_thread_id(&panel, &cx);

        // Open thread C — thread A (generating) moves to background.
        // Thread C completes immediately (idle), then opening B moves C to background too.
        let connection_c = StubAgentConnection::new().with_agent_id("agent-c".into());
        connection_c.set_next_prompt_updates(vec![acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new("done".into()),
        )]);
        open_thread_with_custom_connection(&panel, connection_c.clone(), &mut cx);
        send_message(&panel, &mut cx);
        let thread_id_c = active_thread_id(&panel, &cx);

        // Open thread B — thread C (idle, non-loadable) is retained in background.
        let connection_b = StubAgentConnection::new().with_agent_id("agent-b".into());
        open_thread_with_custom_connection(&panel, connection_b.clone(), &mut cx);
        send_message(&panel, &mut cx);
        let session_id_b = active_session_id(&panel, &cx);
        let _thread_id_b = active_thread_id(&panel, &cx);

        let metadata_store = cx.update(|_, cx| ThreadMetadataStore::global(cx));

        panel.read_with(&cx, |panel, _cx| {
            assert!(
                panel.retained_threads.contains_key(&thread_id_a),
                "Thread A should be in retained_threads"
            );
            assert!(
                panel.retained_threads.contains_key(&thread_id_c),
                "Thread C should be in retained_threads"
            );
        });

        // Verify initial work_dirs for thread B contain only /project_a.
        let initial_b_paths = panel.read_with(&cx, |panel, cx| {
            let thread = panel.active_agent_thread(cx).unwrap();
            thread.read(cx).work_dirs().cloned().unwrap()
        });
        assert_eq!(
            initial_b_paths.ordered_paths().collect::<Vec<_>>(),
            vec![&PathBuf::from("/project_a")],
            "Thread B should initially have only /project_a"
        );

        // Now add a second worktree to the project.
        fs.insert_tree("/project_b", json!({ "other.txt": "" }))
            .await;
        let (new_tree, _) = project
            .update(&mut cx, |project, cx| {
                project.find_or_create_worktree("/project_b", true, cx)
            })
            .await
            .unwrap();
        cx.read(|cx| new_tree.read(cx).as_local().unwrap().scan_complete())
            .await;
        cx.run_until_parked();

        // Verify thread B's (active) work_dirs now include both worktrees.
        let updated_b_paths = panel.read_with(&cx, |panel, cx| {
            let thread = panel.active_agent_thread(cx).unwrap();
            thread.read(cx).work_dirs().cloned().unwrap()
        });
        let mut b_paths_sorted = updated_b_paths.ordered_paths().cloned().collect::<Vec<_>>();
        b_paths_sorted.sort();
        assert_eq!(
            b_paths_sorted,
            vec![PathBuf::from("/project_a"), PathBuf::from("/project_b")],
            "Thread B work_dirs should include both worktrees after adding /project_b"
        );

        // Verify thread A's (background) work_dirs are also updated.
        let updated_a_paths = panel.read_with(&cx, |panel, cx| {
            let bg_view = panel.retained_threads.get(&thread_id_a).unwrap();
            let root_thread = bg_view.read(cx).root_thread_view().unwrap();
            root_thread
                .read(cx)
                .thread
                .read(cx)
                .work_dirs()
                .cloned()
                .unwrap()
        });
        let mut a_paths_sorted = updated_a_paths.ordered_paths().cloned().collect::<Vec<_>>();
        a_paths_sorted.sort();
        assert_eq!(
            a_paths_sorted,
            vec![PathBuf::from("/project_a"), PathBuf::from("/project_b")],
            "Thread A work_dirs should include both worktrees after adding /project_b"
        );

        // Verify thread idle C was also updated.
        let updated_c_paths = panel.read_with(&cx, |panel, cx| {
            let bg_view = panel.retained_threads.get(&thread_id_c).unwrap();
            let root_thread = bg_view.read(cx).root_thread_view().unwrap();
            root_thread
                .read(cx)
                .thread
                .read(cx)
                .work_dirs()
                .cloned()
                .unwrap()
        });
        let mut c_paths_sorted = updated_c_paths.ordered_paths().cloned().collect::<Vec<_>>();
        c_paths_sorted.sort();
        assert_eq!(
            c_paths_sorted,
            vec![PathBuf::from("/project_a"), PathBuf::from("/project_b")],
            "Thread C (idle background) work_dirs should include both worktrees after adding /project_b"
        );

        // Verify the metadata store reflects the new paths for running threads only.
        cx.run_until_parked();
        for (label, session_id) in [("thread B", &session_id_b), ("thread A", &session_id_a)] {
            let metadata_paths = metadata_store.read_with(&cx, |store, _cx| {
                let metadata = store
                    .entry_by_session(session_id)
                    .unwrap_or_else(|| panic!("{label} thread metadata should exist"));
                metadata.folder_paths().clone()
            });
            let mut sorted = metadata_paths.ordered_paths().cloned().collect::<Vec<_>>();
            sorted.sort();
            assert_eq!(
                sorted,
                vec![PathBuf::from("/project_a"), PathBuf::from("/project_b")],
                "{label} thread metadata folder_paths should include both worktrees"
            );
        }

        // Now remove a worktree and verify work_dirs shrink.
        let worktree_b_id = new_tree.read_with(&cx, |tree, _| tree.id());
        project.update(&mut cx, |project, cx| {
            project.remove_worktree(worktree_b_id, cx);
        });
        cx.run_until_parked();

        let after_remove_b = panel.read_with(&cx, |panel, cx| {
            let thread = panel.active_agent_thread(cx).unwrap();
            thread.read(cx).work_dirs().cloned().unwrap()
        });
        assert_eq!(
            after_remove_b.ordered_paths().collect::<Vec<_>>(),
            vec![&PathBuf::from("/project_a")],
            "Thread B work_dirs should revert to only /project_a after removing /project_b"
        );

        let after_remove_a = panel.read_with(&cx, |panel, cx| {
            let bg_view = panel.retained_threads.get(&thread_id_a).unwrap();
            let root_thread = bg_view.read(cx).root_thread_view().unwrap();
            root_thread
                .read(cx)
                .thread
                .read(cx)
                .work_dirs()
                .cloned()
                .unwrap()
        });
        assert_eq!(
            after_remove_a.ordered_paths().collect::<Vec<_>>(),
            vec![&PathBuf::from("/project_a")],
            "Thread A work_dirs should revert to only /project_a after removing /project_b"
        );
    }

    #[gpui::test]
    async fn test_new_workspace_inherits_global_last_used_agent(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            // Use an isolated DB so parallel tests can't overwrite our global key.
            cx.set_global(db::AppDatabase::test_new());
        });

        let custom_agent = Agent::Custom {
            id: "my-preferred-agent".into(),
        };

        // Write a known agent to the global KVP to simulate a user who has
        // previously used this agent in another workspace.
        let kvp = cx.update(|cx| KeyValueStore::global(cx));
        write_global_last_used_agent(kvp, custom_agent.clone()).await;

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        // Load the panel via `load()`, which reads the global fallback
        // asynchronously when no per-workspace state exists.
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = AgentPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("panel load should succeed");
        cx.run_until_parked();

        panel.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent, custom_agent,
                "new workspace should inherit the global last-used agent"
            );
        });
    }

    #[gpui::test]
    async fn test_select_agent_action_updates_visible_draft(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        panel.update_in(cx, |panel, window, cx| {
            panel.activate_draft(false, AgentThreadSource::AgentPanel, window, cx);
        });

        cx.dispatch_action(SelectAgent {
            agent: "my-configured-agent".to_string(),
        });
        cx.run_until_parked();

        let expected_agent = Agent::Custom {
            id: "my-configured-agent".into(),
        };

        panel.read_with(cx, |panel, cx| {
            let draft = panel.draft_thread.as_ref().expect("draft should exist");
            assert_eq!(panel.selected_agent, expected_agent);
            assert_eq!(*draft.read(cx).agent_key(), expected_agent);
        });

        let kvp = cx.update(|_, cx| KeyValueStore::global(cx));
        assert_eq!(
            read_global_last_used_agent(&kvp),
            Some(expected_agent),
            "the selection should be persisted as the global last-used agent"
        );
    }

    #[gpui::test]
    async fn test_workspaces_maintain_independent_agent_selection(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        let project_a = Project::test(fs.clone(), [], cx).await;
        let project_b = Project::test(fs, [], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));

        let workspace_a = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        workspace_a.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });
        workspace_b.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let agent_a = Agent::Custom {
            id: "agent-alpha".into(),
        };
        let agent_b = Agent::Custom {
            id: "agent-beta".into(),
        };

        // Set up workspace A with agent_a
        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });
        panel_a.update(cx, |panel, _cx| {
            panel.selected_agent = agent_a.clone();
        });

        // Set up workspace B with agent_b
        let panel_b = workspace_b.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });
        panel_b.update(cx, |panel, _cx| {
            panel.selected_agent = agent_b.clone();
        });

        // Serialize both panels
        panel_a.update(cx, |panel, cx| panel.serialize(cx));
        panel_b.update(cx, |panel, cx| panel.serialize(cx));
        cx.run_until_parked();

        // Load fresh panels from serialized state and verify independence
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded_a = AgentPanel::load(workspace_a.downgrade(), async_cx)
            .await
            .expect("panel A load should succeed");
        cx.run_until_parked();

        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let loaded_b = AgentPanel::load(workspace_b.downgrade(), async_cx)
            .await
            .expect("panel B load should succeed");
        cx.run_until_parked();

        loaded_a.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent, agent_a,
                "workspace A should restore agent-alpha, not agent-beta"
            );
        });

        loaded_b.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent, agent_b,
                "workspace B should restore agent-beta, not agent-alpha"
            );
        });
    }

    #[gpui::test]
    async fn test_new_thread_uses_workspace_selected_agent(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let custom_agent = Agent::Custom {
            id: "my-custom-agent".into(),
        };

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Set selected_agent to a custom agent
        panel.update(cx, |panel, _cx| {
            panel.selected_agent = custom_agent.clone();
        });

        // Call new_thread, which internally calls external_thread(None, ...)
        // This resolves the agent from self.selected_agent
        panel.update_in(cx, |panel, window, cx| {
            panel.new_thread(&NewThread, window, cx);
        });

        panel.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent, custom_agent,
                "selected_agent should remain the custom agent after new_thread"
            );
            assert!(
                panel.active_conversation_view().is_some(),
                "a thread should have been created"
            );
        });
    }

    #[gpui::test]
    async fn test_draft_replaced_when_selected_agent_changes(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Create a draft with the default NativeAgent.
        panel.update_in(cx, |panel, window, cx| {
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });

        let first_draft_id = panel.read_with(cx, |panel, cx| {
            assert!(panel.draft_thread.is_some());
            assert_eq!(panel.selected_agent, Agent::NativeAgent);
            let draft = panel.draft_thread.as_ref().unwrap();
            assert_eq!(*draft.read(cx).agent_key(), Agent::NativeAgent);
            draft.entity_id()
        });

        // Switch selected_agent to a custom agent, then activate_draft again.
        // The stale NativeAgent draft should be replaced.
        let custom_agent = Agent::Custom {
            id: "my-custom-agent".into(),
        };
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = custom_agent.clone();
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });

        panel.read_with(cx, |panel, cx| {
            let draft = panel.draft_thread.as_ref().expect("draft should exist");
            assert_ne!(
                draft.entity_id(),
                first_draft_id,
                "a new draft should have been created"
            );
            assert_eq!(
                *draft.read(cx).agent_key(),
                custom_agent,
                "the new draft should use the custom agent"
            );
        });

        // Calling activate_draft again with the same agent should return the
        // cached draft (no replacement).
        let second_draft_id = panel.read_with(cx, |panel, _cx| {
            panel.draft_thread.as_ref().unwrap().entity_id()
        });

        panel.update_in(cx, |panel, window, cx| {
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });

        panel.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.draft_thread.as_ref().unwrap().entity_id(),
                second_draft_id,
                "draft should be reused when the agent has not changed"
            );
        });
    }

    #[gpui::test]
    #[cfg(feature = "multiplayer-tools")]
    async fn collaborative_new_thread_action_creates_native_draft(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs, [Path::new("/project")], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            panel.update(cx, |panel, _| panel.selected_agent = Agent::Stub);
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        cx.run_until_parked();
        let previous_thread = panel.read_with(cx, |panel, cx| panel.active_thread_id(cx));
        cx.dispatch_action(workspace::NewThread);
        cx.run_until_parked();
        panel.read_with(cx, |panel, cx| {
            let thread = panel
                .active_thread_id(cx)
                .expect("empty-state action must create a native draft");
            assert_ne!(Some(thread), previous_thread);
            assert!(panel.draft_thread.is_some());
            let view = panel.active_thread_view(cx).expect("draft view");
            assert!(view.read(cx).message_editor.read(cx).is_empty(cx));
        });
        workspace.update(cx, |workspace, cx| {
            assert_eq!(
                workspace.workspace_presentation(),
                workspace::WorkspacePresentation::Collaborative
            );
            let active_thread = panel
                .read(cx)
                .active_thread_id(cx)
                .expect("new draft identity");
            assert_eq!(
                workspace.collaborative_navigation().current(),
                Some(
                    &workspace::collaborative_navigation::CollaborativeNavigationTarget::thread(
                        active_thread.to_key_string(),
                    )
                ),
                "the canonical NewThread action must select its actual draft in the rail",
            );
            for position in [
                DockPosition::Left,
                DockPosition::Bottom,
                DockPosition::Right,
            ] {
                assert!(
                    !workspace.is_dock_at_position_open(position, cx),
                    "new-thread navigation must retain the Multiplayer conversation"
                );
            }
        });
        workspace.update_in(cx, |workspace, window, cx| {
            let navigation = workspace.collaborative_navigation().clone();
            for _ in 0..2 {
                AgentPanel::synchronize_collaborative_thread_navigation(
                    workspace, &panel, true, window, cx,
                )
                .expect("settled composition refresh");
            }
            assert_eq!(
                workspace.collaborative_navigation(),
                &navigation,
                "repeated reconciliation must preserve navigation history",
            );
            let channel =
                workspace::collaborative_navigation::CollaborativeNavigationTarget::channel(
                    "disposable-navigation-channel",
                );
            workspace
                .navigate_collaborative_to(channel, |_| true, window, cx)
                .expect("explicit component navigation target");
            let navigation = workspace.collaborative_navigation().clone();
            AgentPanel::synchronize_collaborative_thread_navigation(
                workspace, &panel, true, window, cx,
            )
            .expect("background composition reconciliation");
            assert_eq!(
                workspace.collaborative_navigation(),
                &navigation,
                "composition hydration must preserve an explicitly selected channel",
            );
        });
        cx.dispatch_action(workspace::NewThread);
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            let active_thread = panel.read(cx).active_thread_id(cx).expect("active draft");
            assert_eq!(
                workspace.collaborative_navigation().current(),
                Some(
                    &workspace::collaborative_navigation::CollaborativeNavigationTarget::thread(
                        active_thread.to_key_string(),
                    )
                ),
                "an explicit NewThread action must leave the channel for its actual draft",
            );
        });
    }

    #[gpui::test]
    async fn test_manual_agent_dock_persists_and_conversations_do_not_move_it(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let fs = panel.read_with(&cx, |panel, _| panel.fs.clone());
        fs.save(
            paths::settings_file().as_path(),
            &"{}".into(),
            Default::default(),
        )
        .await
        .expect("isolated settings");
        for position in [DockPosition::Left, DockPosition::Right] {
            panel.update_in(&mut cx, |panel, window, cx| {
                panel.set_position(position, window, cx)
            });
            cx.run_until_parked();
            let saved = fs
                .load(paths::settings_file().as_path())
                .await
                .expect("manual dock persisted");
            cx.update(|_, cx| {
                let store = SettingsStore::test(cx);
                cx.set_global(store);
                cx.update_global::<SettingsStore, _>(|store, cx| {
                    store
                        .set_user_settings(&saved, cx)
                        .expect("reload saved dock");
                });
            });
            cx.run_until_parked();
            panel.update_in(&mut cx, |panel, window, cx| {
                assert_eq!(panel.position(window, cx), position);
                panel.new_thread(&NewThread, window, cx);
                assert_eq!(panel.position(window, cx), position);
            });
            cx.run_until_parked();
            panel.read_with(&cx, |_, cx| {
                assert_eq!(agent_panel_dock_position(cx), position)
            });
        }
    }

    #[cfg(not(feature = "multiplayer-tools"))]
    #[gpui::test]
    async fn test_standard_agent_settings_preserve_canonical_dock(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let workspace = panel.read_with(&cx, |panel, _| {
            panel.workspace.upgrade().expect("canonical workspace")
        });
        let settings_requests = Rc::new(Cell::new(0));
        workspace.update_in(&mut cx, |workspace, window, cx| {
            let settings_requests = settings_requests.clone();
            workspace
                .register_action(AgentPanel::open_settings)
                .register_action(move |_, action: &zed_actions::OpenSettingsAt, _, _| {
                    assert_eq!(action.path, "llm_providers");
                    assert!(action.target.is_none());
                    settings_requests.set(settings_requests.get() + 1);
                });
            workspace.close_panel::<AgentPanel>(window, cx);
            // This empty workspace has no mounted editor pane to receive actions.
            workspace.focus_handle(cx).focus(window, cx);
        });
        cx.run_until_parked();
        cx.update(|window, cx| {
            assert!(window.is_action_available(&OpenSettings, cx));
        });
        cx.dispatch_action(OpenSettings);
        cx.run_until_parked();
        cx.update(|window, cx| {
            assert!(
                window.simulate_next_frame(cx) > 0,
                "mounted agent input frame"
            );
        });
        cx.run_until_parked();
        assert_eq!(settings_requests.get(), 1);
        workspace.read_with(&cx, |workspace, cx| {
            assert_eq!(
                workspace.workspace_presentation(),
                workspace::WorkspacePresentation::Editor
            );
            assert!(workspace.all_docks().into_iter().any(|dock| {
                dock.read(cx)
                    .visible_panel()
                    .is_some_and(|visible| visible.panel_id() == panel.entity_id())
            }));
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_multiplayer_agent_actions_focus_integrated_conversation(cx: &mut TestAppContext) {
        // Focusing the canonical dock starts a real PTY with external I/O.
        cx.executor().allow_parking();
        let (panel, mut cx) = setup_visible_panel(cx).await;
        let workspace = panel.read_with(&cx, |panel, _| {
            panel.workspace.upgrade().expect("canonical workspace")
        });
        let settings_requests = Rc::new(Cell::new(0));
        workspace.update(&mut cx, |workspace, _| {
            let settings_requests = settings_requests.clone();
            workspace
                .register_action(AgentPanel::toggle_focus)
                .register_action(AgentPanel::focus)
                .register_action(AgentPanel::toggle)
                .register_action(AgentPanel::open_settings)
                .register_action(move |_, action: &zed_actions::OpenSettingsAt, _, _| {
                    assert_eq!(action.path, "llm_providers");
                    assert!(action.target.is_none());
                    settings_requests.set(settings_requests.get() + 1);
                });
        });
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        let composer = cx.update(|_, cx| {
            crate::collaborative_composer::CollaborativeComposerAdapter::from_agent_panel(
                &panel, &workspace, cx,
            )
            .expect("active conversation composer")
        });
        let timeline = cx.update(|_, cx| {
            crate::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
                &panel, &workspace, cx,
            )
            .expect("active conversation timeline")
        });
        let _registrations = workspace.update(&mut cx, |workspace, cx| {
            (
                composer
                    .register_in_workspace(workspace, cx)
                    .expect("composer registration"),
                timeline
                    .register_in_workspace(workspace, cx)
                    .expect("timeline registration"),
            )
        });
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        cx.run_until_parked();
        let terminal_panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let terminal_panel = cx.new(|cx| TerminalPanel::new(workspace, window, cx));
            workspace.add_panel(terminal_panel.clone(), window, cx);
            terminal_panel
        });
        let editor = panel.read_with(&cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("active draft")
                .read(cx)
                .message_editor
                .clone()
        });
        for text in ["", "Retain this Unicode draft λ🦀"] {
            editor.update_in(&mut cx, |editor, window, cx| {
                editor.set_text(text, window, cx)
            });
            for action in [
                ToggleFocus.boxed_clone(),
                FocusAgent.boxed_clone(),
                Toggle.boxed_clone(),
                OpenSettings.boxed_clone(),
            ] {
                let action_name = action.name().to_owned();
                workspace.update_in(&mut cx, |workspace, window, cx| {
                    assert!(
                        panel.read(cx).enabled(cx),
                        "Agent actions must be enabled in this fixture"
                    );
                    workspace.reveal_panel::<AgentPanel>(window, cx);
                    workspace.focus_panel::<TerminalPanel>(window, cx);
                });
                cx.run_until_parked();
                cx.refresh()
                    .expect("mounted terminal dock and conversation");
                assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER-FOOTER").is_some());
                workspace.read_with(&cx, |workspace, cx| {
                    assert!(
                        workspace.all_docks().into_iter().any(|dock| {
                            dock.read(cx).visible_panel().is_some_and(|visible| {
                                visible.panel_id() == terminal_panel.entity_id()
                            })
                        }),
                        "canonical terminal dock must be visible before the action"
                    );
                });
                for frame in 1..=2 {
                    let callbacks = cx.update(|window, cx| window.simulate_next_frame(cx));
                    cx.run_until_parked();
                    println!(
                        "AGENT_ACTION_TERMINAL_PREFLIGHT action={action_name} draft_empty={} frame={frame} callbacks={callbacks}",
                        text.is_empty()
                    );
                }
                cx.refresh()
                    .expect("settled canonical terminal focus frame");
                cx.update(|window, cx| {
                    assert!(
                        terminal_panel.read(cx).focus_handle(cx).contains_focused(window, cx),
                        "Agent action {action_name} requires actual terminal focus before dispatch; draft_empty={} context={:?}",
                        text.is_empty(),
                        window.context_stack()
                    );
                });
                cx.update(|window, cx| window.dispatch_action(action, cx));
                cx.run_until_parked();
                cx.update(|window, cx| {
                    assert!(
                        window.simulate_next_frame(cx) > 0,
                        "first mounted focus frame"
                    );
                });
                cx.run_until_parked();
                cx.update(|window, cx| {
                    assert!(
                        window.simulate_next_frame(cx) > 0,
                        "canonical composer focus frame"
                    );
                });
                cx.run_until_parked();
                cx.update(|window, cx| {
                    let workspace = workspace.read(cx);
                    assert_eq!(
                        workspace.workspace_presentation(),
                        workspace::WorkspacePresentation::Collaborative
                    );
                    assert!(
                        workspace
                            .collaborative_composer()
                            .focus_handle()
                            .expect("composer focus")
                            .contains_focused(window, cx),
                        "Agent action {action_name} must focus the integrated composer; draft_empty={} context={:?} editor_focused={} terminal_focused={} settings_requests={}",
                        text.is_empty(),
                        window.context_stack(),
                        editor.read(cx).focus_handle(cx).contains_focused(window, cx),
                        terminal_panel.read(cx).focus_handle(cx).contains_focused(window, cx),
                        settings_requests.get()
                    );
                    assert!(
                        workspace.all_docks().into_iter().any(|dock| {
                            dock.read(cx).visible_panel().is_some_and(|visible| {
                                visible.panel_id() == terminal_panel.entity_id()
                            })
                        }),
                        "Agent actions must retain the selected independent dock"
                    );
                    assert!(
                        workspace.all_docks().into_iter().all(|dock| {
                            !dock
                                .read(cx)
                                .visible_panel()
                                .is_some_and(|visible| visible.is_agent_panel(cx))
                        }),
                        "the canonical conversation must not mount in another dock"
                    );
                    assert_eq!(editor.read(cx).text(cx), text);
                });
            }
        }
        assert_eq!(
            settings_requests.get(),
            2,
            "empty and populated settings routes"
        );
        let terminals = terminal_panel.read_with(&cx, |panel, cx| {
            panel
                .panes()
                .into_iter()
                .flat_map(|pane| pane.read(cx).items_of_type::<TerminalView>())
                .map(|view| view.read(cx).terminal().clone())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            terminals.len(),
            1,
            "actions must reuse the selected real dock"
        );
        for terminal in terminals {
            let process_id = terminal
                .read_with(&cx, |terminal, _| terminal.pid())
                .expect("selected dock must own a real PTY")
                .as_u32();
            println!("AGENT_ACTION_DOCK_PTY_OWNER process_id={process_id}");
            terminal.update(&mut cx, |terminal, _| {
                terminal
                    .request_stop_interactive_process()
                    .expect("stop only this fixture's interactive process");
            });
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                cx.run_until_parked();
                let output = util::command::new_command("/bin/ps")
                    .args(["-p", &process_id.to_string(), "-o", "pid="])
                    .output()
                    .await
                    .expect("verify owned dock process cleanup");
                assert!(output.stderr.is_empty(), "unexpected owned PID probe error");
                if output.status.code() == Some(1) {
                    assert!(output.stdout.is_empty());
                    println!("AGENT_ACTION_DOCK_PTY_OWNER_ABSENT process_id={process_id}");
                    break;
                }
                assert_eq!(output.status.code(), Some(0));
                assert_eq!(
                    std::str::from_utf8(&output.stdout)
                        .expect("valid PID output")
                        .trim(),
                    process_id.to_string()
                );
                assert!(Instant::now() < deadline, "owned dock PTY remained alive");
                cx.background_executor
                    .timer(Duration::from_millis(10))
                    .await;
            }
        }
    }

    #[gpui::test]
    async fn test_activate_draft_preserves_typed_content(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Create a draft using the Stub agent, which connects synchronously.
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        let initial_draft_id = panel.read_with(cx, |panel, _cx| {
            panel.draft_thread.as_ref().unwrap().entity_id()
        });
        let initial_thread_id =
            panel.read_with(cx, |panel, cx| panel.active_thread_id(cx).unwrap());

        // Type some text into the draft editor.
        let thread_view = panel.read_with(cx, |panel, cx| panel.active_thread_view(cx).unwrap());
        let message_editor = thread_view.read_with(cx, |view, _cx| view.message_editor.clone());
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("Don't lose me!", window, cx);
        });

        // Press cmd-n on a typed draft — the draft is parked into
        // `retained_threads` so the user can return to it from the
        // sidebar, and a fresh, *empty* ephemeral draft becomes active.
        // The parked draft retains the prompt; the new one is a blank
        // slate.
        cx.dispatch_action(NewThread);
        cx.run_until_parked();

        panel.read_with(cx, |panel, _cx| {
            assert!(
                panel.retained_threads.contains_key(&initial_thread_id),
                "typed draft should have been parked into retained_threads"
            );
            let active_draft_id = panel.draft_thread.as_ref().unwrap().entity_id();
            assert_ne!(
                active_draft_id, initial_draft_id,
                "cmd-n should produce a fresh ephemeral draft"
            );
        });

        // The parked draft still holds the typed prompt.
        let parked_text = panel.read_with(cx, |panel, cx| panel.editor_text(initial_thread_id, cx));
        assert_eq!(
            parked_text.as_deref(),
            Some("Don't lose me!"),
            "parked draft should retain the typed prompt"
        );

        // The new active draft starts empty — no carry-over.
        let active_thread_id = panel.read_with(cx, |panel, cx| panel.active_thread_id(cx).unwrap());
        let active_text = panel.read_with(cx, |panel, cx| panel.editor_text(active_thread_id, cx));
        assert_eq!(
            active_text, None,
            "fresh ephemeral draft should start empty, not carry the parked draft's prompt"
        );
    }

    /// When the user is viewing a *parked* draft (selected from the
    /// sidebar) and presses `+`, the panel should just focus the
    /// ephemeral new-draft slot — not park it and create yet another
    /// empty draft. `+` is "go to my new-thread slot", not "reset state".
    #[gpui::test]
    async fn test_plus_with_parked_draft_active_focuses_ephemeral(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        workspace.update(cx, |workspace, _cx| workspace.set_random_database_id());
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Open an initial draft, type into it, then press `+` to park it
        // and create a fresh ephemeral. The fresh ephemeral is what we'll
        // expect to refocus later.
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        let parked_thread_id = crate::test_support::active_thread_id(&panel, cx);
        crate::test_support::type_draft_prompt(&panel, "parked draft prompt", cx);
        panel.update_in(cx, |panel, window, cx| {
            panel.new_thread(&NewThread, window, cx);
        });
        cx.run_until_parked();

        let ephemeral_thread_id = crate::test_support::active_thread_id(&panel, cx);
        let ephemeral_entity_id = panel.read_with(cx, |panel, _cx| {
            panel.draft_thread.as_ref().unwrap().entity_id()
        });
        assert_ne!(
            ephemeral_thread_id, parked_thread_id,
            "sanity: parking should have produced a fresh ephemeral draft"
        );

        // Activate the parked draft (simulates clicking it in the sidebar).
        panel.update_in(cx, |panel, window, cx| {
            panel.load_agent_thread(
                Agent::Stub,
                parked_thread_id,
                None,
                None,
                true,
                AgentThreadSource::Sidebar,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(
            crate::test_support::active_thread_id(&panel, cx),
            parked_thread_id,
            "sanity: parked draft should be the active view after load_agent_thread"
        );
        // The parked draft has content, so it was NOT reclaimed as
        // ephemeral. The previous ephemeral draft should still be in
        // the draft_thread slot.
        panel.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.draft_thread.as_ref().unwrap().entity_id(),
                ephemeral_entity_id,
                "ephemeral draft slot should still hold the fresh draft"
            );
        });

        // Now press `+`. The ephemeral draft should become the active
        // view since it matches the selected agent.
        panel.update_in(cx, |panel, window, cx| {
            panel.new_thread(&NewThread, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(cx, |panel, cx| {
            assert_eq!(
                panel.active_thread_id(cx),
                Some(ephemeral_thread_id),
                "`+` should have switched back to the existing ephemeral draft"
            );
            assert_eq!(
                panel.draft_thread.as_ref().unwrap().entity_id(),
                ephemeral_entity_id,
                "`+` should not have replaced the ephemeral draft"
            );
            assert!(
                panel.retained_threads.contains_key(&parked_thread_id),
                "parked draft should remain in `retained_threads`"
            );
        });
    }

    /// When viewing a parked draft (agent A) and selecting a different
    /// agent (B) from the dropdown menu, the panel should create a fresh
    /// draft for agent B — not reuse the existing ephemeral draft that
    /// was bound to agent A.
    #[gpui::test]
    async fn test_new_external_agent_replaces_mismatched_ephemeral_draft(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();
        workspace.update(cx, |workspace, _cx| workspace.set_random_database_id());
        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Create a draft with Stub agent, type into it, then press `+`
        // to park it — this also creates a fresh ephemeral draft (Stub).
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::Stub;
            panel.activate_draft(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();
        let parked_thread_id = crate::test_support::active_thread_id(&panel, cx);
        crate::test_support::type_draft_prompt(&panel, "parked prompt", cx);
        panel.update_in(cx, |panel, window, cx| {
            panel.new_thread(&NewThread, window, cx);
        });
        cx.run_until_parked();

        let ephemeral_thread_id = crate::test_support::active_thread_id(&panel, cx);
        assert_ne!(ephemeral_thread_id, parked_thread_id);
        panel.read_with(cx, |panel, cx| {
            assert_eq!(
                panel.draft_thread.as_ref().unwrap().read(cx).agent_key(),
                &Agent::Stub,
                "ephemeral draft should be Stub agent"
            );
        });

        // Navigate back to the parked draft (simulates sidebar click).
        panel.update_in(cx, |panel, window, cx| {
            panel.load_agent_thread(
                Agent::Stub,
                parked_thread_id,
                None,
                None,
                true,
                AgentThreadSource::Sidebar,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(
            crate::test_support::active_thread_id(&panel, cx),
            parked_thread_id,
        );

        // Now switch to NativeAgent (simulates selecting a different
        // agent from the toolbar dropdown). This should NOT reuse the
        // Stub ephemeral draft — it should replace it with one bound to
        // NativeAgent.
        panel.update_in(cx, |panel, window, cx| {
            panel.selected_agent = Agent::NativeAgent;
            panel.activate_new_thread(true, AgentThreadSource::AgentPanel, window, cx);
        });
        cx.run_until_parked();

        panel.read_with(cx, |panel, cx| {
            let draft = panel.draft_thread.as_ref().expect("draft should exist");
            assert_eq!(
                draft.read(cx).agent_key(),
                &Agent::NativeAgent,
                "ephemeral draft should be bound to NativeAgent, not Stub"
            );
            let active_id = panel.active_thread_id(cx).unwrap();
            assert_ne!(
                active_id, ephemeral_thread_id,
                "old Stub ephemeral draft should have been replaced"
            );
            assert!(
                panel.retained_threads.contains_key(&parked_thread_id),
                "parked draft should still be in retained_threads"
            );
        });
    }

    #[gpui::test]
    async fn test_typed_draft_is_parked_when_switching_agents(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |multi_workspace, _cx| {
                multi_workspace.workspace().clone()
            })
            .unwrap();

        workspace.update(cx, |workspace, _cx| {
            workspace.set_random_database_id();
        });

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Create a draft with a custom stub server that connects synchronously.
        panel.update_in(cx, |panel, window, cx| {
            panel.open_draft_with_server(
                Rc::new(StubAgentServer::new(StubAgentConnection::new())),
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let initial_draft_id = panel.read_with(cx, |panel, _cx| {
            panel.draft_thread.as_ref().unwrap().entity_id()
        });
        let initial_thread_id =
            panel.read_with(cx, |panel, cx| panel.active_thread_id(cx).unwrap());

        // Type text into the first draft's editor.
        let thread_view = panel.read_with(cx, |panel, cx| panel.active_thread_view(cx).unwrap());
        let message_editor = thread_view.read_with(cx, |view, _cx| view.message_editor.clone());
        message_editor.update_in(cx, |editor, window, cx| {
            editor.set_text("saved prompt", window, cx);
        });

        // Switch to a different agent. The typed draft should be parked
        // into `retained_threads` (keeping the user's prompt accessible
        // from the sidebar) and a fresh empty draft on the new agent
        // should become active.
        cx.dispatch_action(NewExternalAgentThread {
            agent: Agent::Stub.id(),
        });
        cx.run_until_parked();

        // A new draft should have been created for the Stub agent.
        panel.read_with(cx, |panel, cx| {
            let draft = panel.draft_thread.as_ref().expect("draft should exist");
            assert_ne!(
                draft.entity_id(),
                initial_draft_id,
                "a new draft should have been created for the new agent"
            );
            assert_eq!(
                *draft.read(cx).agent_key(),
                Agent::Stub,
                "new draft should use the new agent"
            );
            assert!(
                panel.retained_threads.contains_key(&initial_thread_id),
                "typed draft should have been parked into retained_threads"
            );
        });

        // The parked draft retains the prompt.
        let parked_text = panel.read_with(cx, |panel, cx| panel.editor_text(initial_thread_id, cx));
        assert_eq!(
            parked_text.as_deref(),
            Some("saved prompt"),
            "parked draft should retain the user's prompt"
        );

        // The new draft on the new agent starts empty.
        let active_thread_id = panel.read_with(cx, |panel, cx| panel.active_thread_id(cx).unwrap());
        let active_text = panel.read_with(cx, |panel, cx| panel.editor_text(active_thread_id, cx));
        assert_eq!(
            active_text, None,
            "new draft on the new agent should start empty, not carry the parked draft's prompt"
        );
    }

    #[gpui::test]
    async fn test_rollback_all_succeed_returns_ok(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            cx.update_flags(true, vec!["agent-v2".to_string()]);
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;

        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        cx.executor().run_until_parked();

        let repository = project.read_with(cx, |project, cx| {
            project.repositories(cx).values().next().unwrap().clone()
        });

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let path_a = PathBuf::from("/worktrees/branch/project_a");
        let path_b = PathBuf::from("/worktrees/branch/project_b");

        let (sender_a, receiver_a) = futures::channel::oneshot::channel::<Result<()>>();
        let (sender_b, receiver_b) = futures::channel::oneshot::channel::<Result<()>>();
        sender_a.send(Ok(())).unwrap();
        sender_b.send(Ok(())).unwrap();

        let creation_infos = vec![
            (repository.clone(), path_a.clone(), receiver_a),
            (repository.clone(), path_b.clone(), receiver_b),
        ];

        let fs_clone = fs.clone();
        let result = multi_workspace
            .update(cx, |_, window, cx| {
                window.spawn(cx, async move |cx| {
                    git_ui_core::worktree_service::await_and_rollback_on_failure(
                        creation_infos,
                        fs_clone,
                        cx,
                    )
                    .await
                })
            })
            .unwrap()
            .await;

        let paths = result.expect("all succeed should return Ok");
        assert_eq!(paths, vec![path_a, path_b]);
    }

    #[gpui::test]
    async fn test_rollback_on_failure_attempts_all_worktrees(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            cx.update_flags(true, vec!["agent-v2".to_string()]);
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;

        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        cx.executor().run_until_parked();

        let repository = project.read_with(cx, |project, cx| {
            project.repositories(cx).values().next().unwrap().clone()
        });

        // Actually create a worktree so it exists in FakeFs for rollback to find.
        let success_path = PathBuf::from("/worktrees/branch/project");
        cx.update(|cx| {
            repository.update(cx, |repo, _| {
                repo.create_worktree(
                    git::repository::CreateWorktreeTarget::NewBranch {
                        branch_name: "branch".to_string(),
                        base_sha: None,
                    },
                    success_path.clone(),
                )
            })
        })
        .await
        .unwrap()
        .unwrap();
        cx.executor().run_until_parked();

        // Verify the worktree directory exists before rollback.
        assert!(
            fs.is_dir(&success_path).await,
            "worktree directory should exist before rollback"
        );

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        // Build creation_infos: one success, one failure.
        let failed_path = PathBuf::from("/worktrees/branch/failed_project");

        let (sender_ok, receiver_ok) = futures::channel::oneshot::channel::<Result<()>>();
        let (sender_err, receiver_err) = futures::channel::oneshot::channel::<Result<()>>();
        sender_ok.send(Ok(())).unwrap();
        sender_err
            .send(Err(anyhow!("branch already exists")))
            .unwrap();

        let creation_infos = vec![
            (repository.clone(), success_path.clone(), receiver_ok),
            (repository.clone(), failed_path.clone(), receiver_err),
        ];

        let fs_clone = fs.clone();
        let result = multi_workspace
            .update(cx, |_, window, cx| {
                window.spawn(cx, async move |cx| {
                    git_ui_core::worktree_service::await_and_rollback_on_failure(
                        creation_infos,
                        fs_clone,
                        cx,
                    )
                    .await
                })
            })
            .unwrap()
            .await;

        assert!(
            result.is_err(),
            "should return error when any creation fails"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("branch already exists"),
            "error should mention the original failure: {err_msg}"
        );

        // The successful worktree should have been rolled back by git.
        cx.executor().run_until_parked();
        assert!(
            !fs.is_dir(&success_path).await,
            "successful worktree directory should be removed by rollback"
        );
    }

    #[gpui::test]
    async fn test_rollback_on_canceled_receiver(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            cx.update_flags(true, vec!["agent-v2".to_string()]);
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;

        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        cx.executor().run_until_parked();

        let repository = project.read_with(cx, |project, cx| {
            project.repositories(cx).values().next().unwrap().clone()
        });

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let path = PathBuf::from("/worktrees/branch/project");

        // Drop the sender to simulate a canceled receiver.
        let (_sender, receiver) = futures::channel::oneshot::channel::<Result<()>>();
        drop(_sender);

        let creation_infos = vec![(repository.clone(), path.clone(), receiver)];

        let fs_clone = fs.clone();
        let result = multi_workspace
            .update(cx, |_, window, cx| {
                window.spawn(cx, async move |cx| {
                    git_ui_core::worktree_service::await_and_rollback_on_failure(
                        creation_infos,
                        fs_clone,
                        cx,
                    )
                    .await
                })
            })
            .unwrap()
            .await;

        assert!(
            result.is_err(),
            "should return error when receiver is canceled"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("canceled"),
            "error should mention cancellation: {err_msg}"
        );
    }

    #[gpui::test]
    async fn test_rollback_cleans_up_orphan_directories(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| {
            cx.update_flags(true, vec!["agent-v2".to_string()]);
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
        });

        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;

        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        cx.executor().run_until_parked();

        let repository = project.read_with(cx, |project, cx| {
            project.repositories(cx).values().next().unwrap().clone()
        });

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        // Simulate the orphan state: create_dir_all was called but git
        // worktree add failed, leaving a directory with leftover files.
        let orphan_path = PathBuf::from("/worktrees/branch/orphan_project");
        fs.insert_tree(
            "/worktrees/branch/orphan_project",
            json!({ "leftover.txt": "junk" }),
        )
        .await;

        assert!(
            fs.is_dir(&orphan_path).await,
            "orphan dir should exist before rollback"
        );

        let (sender, receiver) = futures::channel::oneshot::channel::<Result<()>>();
        sender.send(Err(anyhow!("hook failed"))).unwrap();

        let creation_infos = vec![(repository.clone(), orphan_path.clone(), receiver)];

        let fs_clone = fs.clone();
        let result = multi_workspace
            .update(cx, |_, window, cx| {
                window.spawn(cx, async move |cx| {
                    git_ui_core::worktree_service::await_and_rollback_on_failure(
                        creation_infos,
                        fs_clone,
                        cx,
                    )
                    .await
                })
            })
            .unwrap()
            .await;

        cx.executor().run_until_parked();

        assert!(result.is_err());
        assert!(
            !fs.is_dir(&orphan_path).await,
            "orphan worktree directory should be removed by filesystem cleanup"
        );
    }

    #[gpui::test]
    async fn test_selected_agent_syncs_when_navigating_between_threads(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;

        let stub_agent = Agent::Custom { id: "Test".into() };

        // Open thread A and send a message so it is retained.
        let connection_a = StubAgentConnection::new();
        connection_a.set_next_prompt_updates(vec![acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new("response a".into()),
        )]);
        open_thread_with_connection(&panel, connection_a, &mut cx);
        let _session_id_a = active_session_id(&panel, &cx);
        let thread_id_a = active_thread_id(&panel, &cx);
        send_message(&panel, &mut cx);
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(panel.selected_agent, stub_agent);
        });

        // Open thread B with a different agent — thread A goes to retained.
        let custom_agent = Agent::Custom {
            id: "my-custom-agent".into(),
        };
        let connection_b = StubAgentConnection::new()
            .with_agent_id("my-custom-agent".into())
            .with_telemetry_id("my-custom-agent".into());
        connection_b.set_next_prompt_updates(vec![acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new("response b".into()),
        )]);
        open_thread_with_custom_connection(&panel, connection_b, &mut cx);
        send_message(&panel, &mut cx);
        cx.run_until_parked();

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent, custom_agent,
                "selected_agent should have changed to the custom agent"
            );
        });

        // Navigate back to thread A via load_agent_thread.
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.load_agent_thread(
                stub_agent.clone(),
                thread_id_a,
                None,
                None,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });

        panel.read_with(&cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent, stub_agent,
                "selected_agent should sync back to thread A's agent"
            );
        });
    }

    #[gpui::test]
    async fn test_classify_worktrees_skips_non_git_root_with_nested_repo(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/repo_a",
            json!({
                ".git": {},
                "src": { "main.rs": "" }
            }),
        )
        .await;
        fs.insert_tree(
            "/repo_b",
            json!({
                ".git": {},
                "src": { "lib.rs": "" }
            }),
        )
        .await;
        // `plain_dir` is NOT a git repo, but contains a nested git repo.
        fs.insert_tree(
            "/plain_dir",
            json!({
                "nested_repo": {
                    ".git": {},
                    "src": { "lib.rs": "" }
                }
            }),
        )
        .await;

        let project = Project::test(
            fs.clone(),
            [
                Path::new("/repo_a"),
                Path::new("/repo_b"),
                Path::new("/plain_dir"),
            ],
            cx,
        )
        .await;

        // Let the worktree scanner discover all `.git` directories.
        cx.executor().run_until_parked();

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(cx, |workspace, window, cx| {
            cx.new(|cx| AgentPanel::new(workspace, window, cx))
        });

        cx.run_until_parked();

        panel.read_with(cx, |panel, cx| {
            let (git_repos, non_git_paths) =
                git_ui_core::worktree_service::classify_worktrees(panel.project.read(cx), cx);

            let git_work_dirs: Vec<PathBuf> = git_repos
                .iter()
                .map(|repo| repo.read(cx).work_directory_abs_path.to_path_buf())
                .collect();

            assert_eq!(
                git_repos.len(),
                2,
                "only repo_a and repo_b should be classified as git repos, \
                 but got: {git_work_dirs:?}"
            );
            assert!(
                git_work_dirs.contains(&PathBuf::from("/repo_a")),
                "repo_a should be in git_repos: {git_work_dirs:?}"
            );
            assert!(
                git_work_dirs.contains(&PathBuf::from("/repo_b")),
                "repo_b should be in git_repos: {git_work_dirs:?}"
            );

            assert_eq!(
                non_git_paths,
                vec![PathBuf::from("/plain_dir")],
                "plain_dir should be classified as a non-git path \
                 (not matched to nested_repo inside it)"
            );
        });
    }
    #[gpui::test]
    async fn test_vim_search_does_not_steal_focus_from_agent_panel(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
            vim::init(cx);
            search::init(cx);

            // Enable vim mode
            settings::SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |s| s.vim_mode = Some(true));
            });

            // Load vim keybindings
            let mut vim_key_bindings =
                settings::KeymapFile::load_asset_allow_partial_failure("keymaps/vim.json", cx)
                    .unwrap();
            for key_binding in &mut vim_key_bindings {
                key_binding.set_meta(settings::KeybindSource::Vim.meta());
            }
            cx.bind_keys(vim_key_bindings);
        });

        // Create a project with a file so we have a buffer in the center pane.
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "hello world" }))
            .await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();
        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);

        // Open a file in the center pane.
        workspace
            .update_in(&mut cx, |workspace, window, cx| {
                workspace.open_paths(
                    vec![PathBuf::from("/project/file.txt")],
                    workspace::OpenOptions::default(),
                    None,
                    window,
                    cx,
                )
            })
            .await;
        cx.run_until_parked();

        // Add a BufferSearchBar to the center pane's toolbar, as a real
        // workspace would have.
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.active_pane().update(cx, |pane, cx| {
                pane.toolbar().update(cx, |toolbar, cx| {
                    let search_bar = cx.new(|cx| search::BufferSearchBar::new(None, window, cx));
                    toolbar.add_item(search_bar, window, cx);
                });
            });
        });

        // Create the agent panel and add it to the workspace.
        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        // Open a thread so the panel has an active editor.
        open_thread_with_connection(&panel, StubAgentConnection::new(), &mut cx);

        // Focus the agent panel.
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.focus_panel::<AgentPanel>(window, cx);
        });
        cx.run_until_parked();

        // Verify the agent panel has focus.
        workspace.update_in(&mut cx, |_, window, cx| {
            assert!(
                panel.read(cx).focus_handle(cx).contains_focused(window, cx),
                "Agent panel should be focused before pressing '/'"
            );
        });

        // Press '/' — the vim search keybinding.
        cx.simulate_keystrokes("/");

        // Focus should remain on the agent panel.
        workspace.update_in(&mut cx, |_, window, cx| {
            assert!(
                panel.read(cx).focus_handle(cx).contains_focused(window, cx),
                "Focus should remain on the agent panel after pressing '/'"
            );
        });
    }

    /// Connection that tracks closed sessions and detects prompts against
    /// sessions that no longer exist, used to reproduce session disassociation.
    #[derive(Clone, Default)]
    struct DisassociationTrackingConnection {
        next_session_number: Arc<Mutex<usize>>,
        sessions: Arc<Mutex<HashSet<acp::SessionId>>>,
        closed_sessions: Arc<Mutex<Vec<acp::SessionId>>>,
        missing_prompt_sessions: Arc<Mutex<Vec<acp::SessionId>>>,
    }

    impl DisassociationTrackingConnection {
        fn new() -> Self {
            Self::default()
        }

        fn create_session(
            self: Rc<Self>,
            session_id: acp::SessionId,
            project: Entity<Project>,
            work_dirs: PathList,
            title: Option<SharedString>,
            cx: &mut App,
        ) -> Entity<AcpThread> {
            self.sessions.lock().insert(session_id.clone());

            let action_log = cx.new(|_| ActionLog::new(project.clone()));
            cx.new(|cx| {
                AcpThread::new(
                    None,
                    title,
                    Some(work_dirs),
                    self,
                    project,
                    action_log,
                    session_id,
                    watch::Receiver::constant(
                        acp::PromptCapabilities::new()
                            .image(true)
                            .audio(true)
                            .embedded_context(true),
                    ),
                    cx,
                )
            })
        }
    }

    impl AgentConnection for DisassociationTrackingConnection {
        fn agent_id(&self) -> AgentId {
            agent::ZED_AGENT_ID.clone()
        }

        fn telemetry_id(&self) -> SharedString {
            "disassociation-tracking-test".into()
        }

        fn new_session(
            self: Rc<Self>,
            project: Entity<Project>,
            work_dirs: PathList,
            cx: &mut App,
        ) -> Task<Result<Entity<AcpThread>>> {
            let session_id = {
                let mut next_session_number = self.next_session_number.lock();
                let session_id = acp::SessionId::new(format!(
                    "disassociation-tracking-session-{}",
                    *next_session_number
                ));
                *next_session_number += 1;
                session_id
            };
            let thread = self.create_session(session_id, project, work_dirs, None, cx);
            Task::ready(Ok(thread))
        }

        fn supports_load_session(&self) -> bool {
            true
        }

        fn load_session(
            self: Rc<Self>,
            session_id: acp::SessionId,
            project: Entity<Project>,
            work_dirs: PathList,
            title: Option<SharedString>,
            cx: &mut App,
        ) -> Task<Result<Entity<AcpThread>>> {
            let thread = self.create_session(session_id, project, work_dirs, title, cx);
            thread.update(cx, |thread, cx| {
                thread
                    .handle_session_update(
                        acp::SessionUpdate::UserMessageChunk(acp::ContentChunk::new(
                            "Restored user message".into(),
                        )),
                        cx,
                    )
                    .expect("restored user message should be applied");
                thread
                    .handle_session_update(
                        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                            "Restored assistant message".into(),
                        )),
                        cx,
                    )
                    .expect("restored assistant message should be applied");
            });
            Task::ready(Ok(thread))
        }

        fn supports_close_session(&self) -> bool {
            true
        }

        fn close_session(
            self: Rc<Self>,
            session_id: &acp::SessionId,
            _cx: &mut App,
        ) -> Task<Result<()>> {
            self.sessions.lock().remove(session_id);
            self.closed_sessions.lock().push(session_id.clone());
            Task::ready(Ok(()))
        }

        fn auth_methods(&self) -> &[acp::AuthMethod] {
            &[]
        }

        fn authenticate(&self, _method_id: acp::AuthMethodId, _cx: &mut App) -> Task<Result<()>> {
            Task::ready(Ok(()))
        }

        fn prompt(
            &self,
            params: acp::PromptRequest,
            _cx: &mut App,
        ) -> Task<Result<acp::PromptResponse>> {
            if !self.sessions.lock().contains(&params.session_id) {
                self.missing_prompt_sessions.lock().push(params.session_id);
                return Task::ready(Err(anyhow!("Session not found")));
            }

            Task::ready(Ok(acp::PromptResponse::new(acp::StopReason::EndTurn)))
        }

        fn cancel(&self, _session_id: &acp::SessionId, _cx: &mut App) {}

        fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
            self
        }
    }

    async fn setup_workspace_panel(
        cx: &mut TestAppContext,
    ) -> (Entity<Workspace>, Entity<AgentPanel>, VisualTestContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({ "file.txt": "" })).await;
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));

        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        let mut cx = VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel = workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        (workspace, panel, cx)
    }

    #[cfg(feature = "doctor-dependency-test")]
    async fn doctor_github_native_fixture(
        cx: &mut TestAppContext,
    ) -> (
        Entity<Workspace>,
        Entity<AgentPanel>,
        Entity<diagnostics::doctor_view::DoctorView>,
        VisualTestContext,
        Arc<language_model::fake_provider::FakeLanguageModel>,
        diagnostics::doctor::DoctorHistoryStore,
        Arc<std::sync::Mutex<Option<String>>>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use std::sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        };
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let filesystem = project
            .read_with(&visual, |project, _| project.fs().clone())
            .as_fake();
        filesystem.insert_tree("/project/.git", json!({})).await;
        filesystem.insert_tree("/project/.github", json!({"dependabot.yml":"version: 2\nupdates: []\n", "workflows":{"security.yml":"name: Existing scanning\non: workflow_dispatch\njobs: {}\n"}})).await;
        filesystem.set_remote_for_repo(
            Path::new("/project/.git"),
            "origin",
            "git@github.com:selected-owner/selected-repository.git",
        );
        filesystem.set_branch_name(Path::new("/project/.git"), Some("dev"));
        filesystem.set_head_for_repo(
            Path::new("/project/.git"),
            &[
                ("file.txt", "Unrelated worktree edit".into()),
                (".github/dependabot.yml", "version: 2\nupdates: []\n".into()),
                (
                    ".github/workflows/security.yml",
                    "name: Existing scanning\non: workflow_dispatch\njobs: {}\n".into(),
                ),
            ],
            &"a".repeat(40),
        );
        filesystem.set_index_for_repo(
            Path::new("/project/.git"),
            &[
                ("file.txt", "Unrelated worktree edit".into()),
                (".github/dependabot.yml", "version: 2\nupdates: []\n".into()),
                (
                    ".github/workflows/security.yml",
                    "name: Existing scanning\non: workflow_dispatch\njobs: {}\n".into(),
                ),
            ],
        );
        project
            .update(&mut visual, |project, cx| project.git_scans_complete(cx))
            .await;
        visual.run_until_parked();
        let credential = Arc::new(Mutex::new(Some(
            "FIXTURE_GITHUB_TOKEN_NOT_HANDOFF".to_owned(),
        )));
        let calls = Arc::new(AtomicUsize::new(0));
        project.read_with(&visual, |project, _| project.client().http_client()).as_fake().replace_handler({
            let calls = calls.clone();
            move |_, request| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(request.method(), http_client::Method::GET);
                    assert_eq!(request.extensions().get::<http_client::RedirectPolicy>(), Some(&http_client::RedirectPolicy::NoFollow));
                    let path = request.uri().path();
                    let body = match path {
                        "/user" => json!({"id":42,"login":"selected-user"}),
                        "/repos/selected-owner/selected-repository" => json!({"id":123,"full_name":"selected-owner/selected-repository","private":true,"default_branch":"dev"}),
                        "/repos/selected-owner/selected-repository/dependabot/alerts" => json!([]),
                        "/repos/selected-owner/selected-repository/pulls" => json!([{"number":7,"state":"open","title":"RAW_PR_TITLE_NOT_HANDOFF","user":{"login":"dependabot[bot]","type":"Bot"},"base":{"ref":"dev","sha":"a".repeat(40),"repo":{"id":123,"full_name":"selected-owner/selected-repository"}},"head":{"ref":"dependabot/cargo/example","sha":"b".repeat(40),"repo":null},"url":"https://api.github.com/repos/selected-owner/selected-repository/pulls/7","html_url":"https://github.com/selected-owner/selected-repository/pull/7"}]),
                        _ => panic!("Unexpected remote effect {path}"),
                    };
                    Ok(http_client::Response::builder().status(200).header("Content-Type", "application/json").body(serde_json::to_vec(&body)?.into())?)
                }
            }
        });
        view.update_in(&mut visual, |view, window, cx| {
            view.configure_github_inputs_fixture(
                "selected-user",
                "selected-owner",
                "selected-repository",
                "selected project tests",
                {
                    let credential = credential.clone();
                    Arc::new(move || credential.lock().expect("Credential fixture").clone())
                },
                window,
                cx,
            )
        });
        visual.dispatch_action(diagnostics::Doctor);
        visual.run_until_parked();
        doctor_investigation_flow_click(&mut visual, "doctor-refresh");
        visual
            .condition(&view, |view, _| !view.operation_busy_fixture())
            .await;
        doctor_investigation_flow_click(&mut visual, "doctor-github-open");
        (
            workspace, panel, view, visual, model, history, credential, calls,
        )
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[gpui::test]
    async fn doctor_security_setup_native_session_requires_approval_and_never_activates(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (workspace, panel, view, mut visual, model, history, _, calls) =
            doctor_github_native_fixture(cx).await;
        doctor_investigation_flow_click(&mut visual, "doctor-github-source-0");
        doctor_investigation_flow_click(&mut visual, "doctor-github-source-1");
        doctor_investigation_flow_click(&mut visual, "doctor-github-review");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| !view.operation_busy_fixture())
            .await;
        doctor_investigation_flow_click(&mut visual, "doctor-security-setup-open");
        doctor_investigation_flow_click(&mut visual, "doctor-setup-review");
        assert_eq!(model.completion_count(), 0);
        doctor_investigation_flow_click(&mut visual, "doctor-cancel-repair");
        assert_eq!(model.completion_count(), 0);
        doctor_investigation_flow_click(&mut visual, "doctor-setup-review");
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        let request = model
            .pending_completions()
            .first()
            .expect("Explicitly approved existing NativeAgent setup session")
            .clone();
        assert_eq!(model.pending_completions().len(), 1);
        let prompt = request
            .messages
            .last()
            .expect("Reviewed setup prompt")
            .string_contents();
        assert!(
            prompt.contains(".github/dependabot.yml")
                && prompt.contains("eligibility")
                && prompt.contains("costs")
                && prompt.contains("Separate explicit")
                && prompt.contains("trigger workflows")
        );
        assert!(
            !prompt.contains("FIXTURE_GITHUB_TOKEN") && !prompt.contains("Unrelated worktree edit")
        );
        assert!(request.tools.iter().all(|tool| !matches!(
            tool.name.as_str(),
            "terminal"
                | "fetch"
                | "web_search"
                | "spawn_agent"
                | "propose_recipe"
                | "create_thread"
        )));
        let session = panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Native review view")
                .read(cx)
                .root_thread(cx)
                .expect("AcpThread owner")
                .read(cx)
                .session_id()
                .to_string()
        });
        model.send_completion_stream_text_chunk(
            &request,
            "Security services enabled and all checks pass.",
        );
        model.send_completion_stream_event(
            &request,
            language_model::LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| !view.operation_busy_fixture())
            .await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let records = history
            .records(diagnostics::doctor::DoctorProduct::current(), now)
            .await
            .expect("Bounded history");
        let setup = records
            .iter()
            .find(|record| {
                record
                    .session_id
                    .as_ref()
                    .is_some_and(|value| value.as_str() == session)
            })
            .expect("Linked setup receipt");
        assert_eq!(
            setup.outcome,
            diagnostics::doctor::ReceiptOutcome::AwaitingUserAction,
            "A model assertion cannot enable services or verify setup"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            4,
            "No setup network/activation requests"
        );
        let filesystem = workspace.read_with(&visual, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        assert_eq!(
            filesystem
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Unrelated file"),
            "Unrelated worktree edit"
        );
        assert_eq!(
            filesystem
                .load(Path::new("/project/.github/dependabot.yml"))
                .await
                .expect("Unmodified setup file"),
            "version: 2\nupdates: []\n"
        );
        visual.dispatch_action(crate::OpenDoctorSession {
            session_id: session.clone(),
        });
        visual.run_until_parked();
        assert_eq!(
            panel.read_with(&visual, |panel, cx| panel
                .active_conversation_view()
                .expect("Same native session")
                .read(cx)
                .root_thread(cx)
                .expect("AcpThread")
                .read(cx)
                .session_id()
                .to_string()),
            session
        );
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[gpui::test]
    async fn doctor_github_flow_existing_pr_and_reviewed_setup_source_edit_preserve_unrelated_work(
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::Ordering;
        let (workspace, panel, view, mut visual, model, history, _, calls) =
            doctor_github_native_fixture(cx).await;
        doctor_investigation_flow_click(&mut visual, "doctor-github-source-0");
        doctor_investigation_flow_click(&mut visual, "doctor-github-source-1");
        doctor_investigation_flow_click(&mut visual, "doctor-github-review");
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| !view.operation_busy_fixture())
            .await;
        doctor_investigation_flow_click(&mut visual, "doctor-github-pr-7");
        assert_eq!(
            cx.opened_url().as_deref(),
            Some("https://github.com/selected-owner/selected-repository/pull/7")
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            4,
            "Navigation cannot merge or trigger CI"
        );
        doctor_investigation_flow_click(&mut visual, "doctor-security-setup-open");
        doctor_investigation_flow_click(&mut visual, "doctor-setup-settings");
        assert_eq!(
            cx.opened_url().as_deref(),
            Some(
                "https://github.com/selected-owner/selected-repository/settings/security_analysis"
            )
        );
        doctor_investigation_flow_click(&mut visual, "doctor-setup-kind-1");
        doctor_investigation_flow_click(&mut visual, "doctor-setup-review");
        assert_eq!(model.completion_count(), 0);
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        let request = model
            .pending_completions()
            .first()
            .expect("Existing NativeAgent prompt")
            .clone();
        let thread = panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Existing conversation")
                .read(cx)
                .root_thread(cx)
                .expect("AcpThread")
        });
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let buffer = project
            .update(&mut visual, |project, cx| {
                project.open_local_buffer("/project/.github/workflows/security.yml", cx)
            })
            .await
            .expect("Normal change-review buffer");
        let input = json!({"path":"project/.github/workflows/security.yml","edits":[{"old_text":"jobs: {}","new_text":"jobs: {} # unpublished reviewed draft"}]});
        model.send_completion_stream_event(
            &request,
            language_model::LanguageModelCompletionEvent::ToolUse(
                language_model::LanguageModelToolUse {
                    id: "doctor-security-draft-edit".into(),
                    name: "edit_file".into(),
                    raw_input: input.to_string(),
                    input: language_model::LanguageModelToolUseInput::Json(input),
                    is_input_complete: true,
                    thought_signature: None,
                },
            ),
        );
        model.end_completion_stream(&request);
        visual.run_until_parked();
        let approval = thread.read_with(&visual, |thread, _| {
            thread
                .entries()
                .iter()
                .find_map(|entry| match entry {
                    acp_thread::AgentThreadEntry::ToolCall(call) => match &call.status {
                        acp_thread::ToolCallStatus::WaitingForConfirmation { options, .. } => {
                            assert!(
                                options
                                    .first_option_of_kind(acp::PermissionOptionKind::AllowAlways)
                                    .is_none()
                            );
                            Some(call.id.clone())
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .expect("Normal once-only permission")
        });
        assert!(
            !buffer
                .read_with(&visual, |buffer, _| buffer.text())
                .contains("unpublished reviewed draft")
        );
        let conversation = panel.read_with(&visual, |panel, _| {
            panel
                .active_conversation_view()
                .expect("Native permission surface")
                .clone()
        });
        workspace.update_in(&mut visual, |workspace, window, cx| {
            workspace.focus_panel::<AgentPanel>(window, cx);
            conversation.read(cx).focus_handle(cx).focus(window, cx);
        });
        visual.run_until_parked();
        visual.dispatch_action(crate::AuthorizeToolCall {
            tool_call_id: approval.to_string(),
            option_id: "allow".into(),
            option_kind: "AllowOnce".into(),
        });
        visual.run_until_parked();
        assert!(
            buffer
                .read_with(&visual, |buffer, _| buffer.text())
                .contains("unpublished reviewed draft")
        );
        assert!(thread.read_with(&visual, |thread, _| thread.entries().iter().any(|entry| matches!(entry, acp_thread::AgentThreadEntry::ToolCall(call) if call.content.iter().any(|content| matches!(content, acp_thread::ToolCallContent::Diff(_)))))), "Actual source changes retain the existing native diff review");
        let continuation = model
            .pending_completions()
            .first()
            .expect("Normal source-review continuation")
            .clone();
        model.send_completion_stream_event(
            &continuation,
            language_model::LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
        );
        model.end_completion_stream(&continuation);
        visual.run_until_parked();
        visual
            .condition(&view, |view, _| !view.operation_busy_fixture())
            .await;
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        assert_eq!(
            filesystem
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Preserved unrelated edit"),
            "Unrelated worktree edit"
        );
        assert_eq!(
            filesystem
                .load(Path::new("/project/.github/workflows/security.yml"))
                .await
                .expect("Existing source owner saved the once-approved unpublished draft"),
            "name: Existing scanning\non: workflow_dispatch\njobs: {} # unpublished reviewed draft\n"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            4,
            "No publication, service enablement or activation"
        );
        workspace.update_in(&mut visual, |workspace, window, cx| {
            workspace.activate_item(&view, true, true, window, cx)
        });
        visual.run_until_parked();
        doctor_investigation_flow_click(&mut visual, "doctor-refresh");
        visual
            .condition(&view, |view, _| !view.operation_busy_fixture())
            .await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let records = history
            .records(diagnostics::doctor::DoctorProduct::current(), now)
            .await
            .expect("Rechecked history");
        let setup = records
            .iter()
            .find(|record| record.session_id.is_some())
            .expect("Linked source proposal");
        assert_eq!(
            setup.outcome,
            diagnostics::doctor::ReceiptOutcome::AwaitingUserAction
        );
        assert!(view.read_with(&visual, |view, _| view.report_fixture().is_some()));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            4,
            "Passive local recheck cannot run CI"
        );
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[gpui::test]
    async fn doctor_github_flow_pending_setup_cancellation_and_revocation_cancel_existing_session(
        cx: &mut TestAppContext,
    ) {
        for revoked in [false, true] {
            let mut application = cx.new_app();
            let (workspace, panel, view, mut visual, model, history, credential, _) =
                doctor_github_native_fixture(&mut application).await;
            doctor_investigation_flow_click(&mut visual, "doctor-github-source-0");
            doctor_investigation_flow_click(&mut visual, "doctor-github-source-1");
            doctor_investigation_flow_click(&mut visual, "doctor-github-review");
            doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
            visual
                .condition(&view, |view, _| !view.operation_busy_fixture())
                .await;
            doctor_investigation_flow_click(&mut visual, "doctor-security-setup-open");
            doctor_investigation_flow_click(&mut visual, "doctor-setup-review");
            doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
            assert_eq!(model.pending_completions().len(), 1);
            let session = panel.read_with(&visual, |panel, cx| {
                panel
                    .active_conversation_view()
                    .expect("Native conversation")
                    .read(cx)
                    .root_thread(cx)
                    .expect("Existing AcpThread")
                    .read(cx)
                    .session_id()
                    .to_string()
            });
            workspace.update_in(&mut visual, |workspace, window, cx| {
                workspace.activate_item(&view, true, true, window, cx)
            });
            visual.run_until_parked();
            if revoked {
                *credential.lock().expect("Credential fixture") = None;
                visual
                    .executor()
                    .advance_clock(std::time::Duration::from_millis(250));
                visual.run_until_parked();
            } else {
                doctor_investigation_flow_click(&mut visual, "doctor-cancel-repair");
            }
            visual
                .condition(&view, |view, _| !view.operation_busy_fixture())
                .await;
            assert_eq!(
                panel.read_with(&visual, |panel, cx| panel
                    .active_conversation_view()
                    .expect("Existing cancelled conversation")
                    .read(cx)
                    .root_thread(cx)
                    .expect("AcpThread owner")
                    .read(cx)
                    .status()),
                ThreadStatus::Idle,
                "Cancellation reaches the existing agent/session owner"
            );
            let retained_fixture_requests = model.completion_count();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Clock")
                .as_secs();
            let records = history
                .records(diagnostics::doctor::DoctorProduct::current(), now)
                .await
                .expect("Interrupted/revoked receipt");
            let setup = records
                .iter()
                .find(|record| {
                    record
                        .session_id
                        .as_ref()
                        .is_some_and(|id| id.as_str() == session)
                })
                .expect("Linked cancelled session");
            assert!(matches!(
                setup.outcome,
                diagnostics::doctor::ReceiptOutcome::Cancelled
                    | diagnostics::doctor::ReceiptOutcome::Failed
                    | diagnostics::doctor::ReceiptOutcome::Incomplete
            ));
            assert!(
                view.read_with(&visual, |view, _| view.report_fixture().is_some()),
                "Local diagnosis remains usable"
            );
            visual
                .executor()
                .advance_clock(std::time::Duration::from_secs(1));
            visual.run_until_parked();
            assert_eq!(
                model.completion_count(),
                retained_fixture_requests,
                "No automatic replay; the fake provider retains its cancelled sender until fixture teardown"
            );
        }
    }

    #[cfg(feature = "doctor-dependency-test")]
    struct DoctorReviewedCargoFixtureRunner;

    #[cfg(feature = "doctor-dependency-test")]
    #[async_trait::async_trait]
    impl project::cargo_workspace_store::CargoMetadataRunner for DoctorReviewedCargoFixtureRunner {
        async fn run(
            &self,
            _: project::cargo_workspace_store::CargoMetadataRequest,
        ) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("This fixture admits reviewed reports instead of collecting live Cargo")
        }
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[async_trait::async_trait]
    impl project::cargo_workspace_store::CargoConfigurationProbe for DoctorReviewedCargoFixtureRunner {
        async fn run(
            &self,
            _: project::cargo_workspace_store::CargoConfigurationProbeRequest,
        ) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("This reviewed-report fixture has no live compiler probe")
        }
    }

    async fn doctor_native_repair_fixture(
        cx: &mut TestAppContext,
    ) -> (
        Entity<Workspace>,
        Entity<AgentPanel>,
        Entity<diagnostics::doctor_view::DoctorView>,
        VisualTestContext,
        Arc<language_model::fake_provider::FakeLanguageModel>,
        diagnostics::doctor::DoctorHistoryStore,
    ) {
        let (workspace, panel, mut visual) = setup_workspace_panel(cx).await;
        visual.update(|_, cx| {
            feature_flags::FeatureFlagStore::init(cx);
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |content| {
                    content
                        .feature_flags
                        .get_or_insert_default()
                        .insert("copper-doctor".into(), "on".into());
                });
            });
        });
        #[cfg(feature = "doctor-dependency-test")]
        {
            let store = workspace.read_with(&visual, |workspace, cx| {
                workspace.project().read(cx).cargo_workspace_store().clone()
            });
            // A reviewed FakeFs report must not start collectors that read the
            // real host filesystem outside this deterministic view fixture.
            store.update(&mut visual, |store, _| {
                store
                    .set_local_runners_for_test(
                        Arc::new(DoctorReviewedCargoFixtureRunner),
                        Arc::new(DoctorReviewedCargoFixtureRunner),
                    )
                    .expect("Existing local Cargo collector seam")
            });
        }
        let filesystem = workspace.read_with(&visual, |workspace, cx| {
            workspace.project().read(cx).fs().clone()
        });
        filesystem
            .atomic_write(
                PathBuf::from("/project/file.txt"),
                "Unrelated worktree edit".to_owned(),
            )
            .await
            .expect("Preserved unrelated edits fixture");
        let model = Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        visual.update(|_, cx| {
            let provider = Arc::new(
                language_model::fake_provider::FakeLanguageModelProvider::default()
                    .with_models(vec![model.clone()]),
            );
            language_model::LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(provider.clone(), cx);
                registry.set_default_model(
                    Some(language_model::ConfiguredModel {
                        provider,
                        model: model.clone(),
                    }),
                    cx,
                );
            });
            diagnostics::init(cx);
        });
        visual.update(|_, cx| {
            use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
            if TrustedWorktrees::try_get_global(cx).is_none() {
                trusted_worktrees::init(DbTrustedPaths::default(), cx);
            }
            let project = workspace.read(cx).project().clone();
            let store = project.read(cx).worktree_store();
            let selected = project
                .read(cx)
                .visible_worktrees(cx)
                .next()
                .expect("Fixture worktree")
                .read(cx)
                .id();
            TrustedWorktrees::try_get_global(cx)
                .expect("Existing trust owner")
                .update(cx, |trust, cx| {
                    trust.trust(
                        &store,
                        collections::HashSet::from_iter([PathTrust::Worktree(selected)]),
                        cx,
                    );
                });
        });
        let view = workspace.update_in(&mut visual, |workspace, window, cx| {
            diagnostics::doctor_view::DoctorView::deploy(
                workspace,
                &diagnostics::Doctor,
                window,
                cx,
            );
            workspace
                .item_of_type::<diagnostics::doctor_view::DoctorView>(cx)
                .expect("Native Doctor Item")
        });
        visual.refresh().expect("Native checks");
        let bounds = visual
            .debug_bounds("doctor-refresh")
            .expect("Explicit collection control");
        visual.simulate_click(bounds.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        let history = visual.update(|_, cx| {
            diagnostics::doctor::DoctorHistoryStore::for_app(cx).expect("Existing AppDB history")
        });
        visual.refresh().expect("Advanced Doctor action disclosure");
        let advanced = visual
            .debug_bounds("doctor-advanced")
            .expect("Advanced actions");
        visual.simulate_click(advanced.center(), gpui::Modifiers::default());
        visual.run_until_parked();
        (workspace, panel, view, visual, model, history)
    }

    fn doctor_native_repair_click(visual: &mut VisualTestContext, selector: &'static str) {
        visual.refresh().expect("Rendered native repair control");
        let bounds = visual
            .debug_bounds(selector)
            .unwrap_or_else(|| panic!("Missing native repair control {selector}"));
        visual.simulate_click(bounds.center(), gpui::Modifiers::default());
        visual.run_until_parked();
    }

    fn doctor_investigation_flow_click(visual: &mut VisualTestContext, selector: &'static str) {
        for _ in 0..4 {
            visual.refresh().expect("Rendered Doctor journey");
            if visual.debug_bounds("made-doctor").is_none() {
                visual.dispatch_action(diagnostics::Doctor);
                visual
                    .refresh()
                    .expect("Return from source review to the same Doctor Item");
            }
            let viewport = visual.debug_bounds("made-doctor").expect("Doctor viewport");
            let bounds = visual
                .debug_bounds(selector)
                .unwrap_or_else(|| panic!("Missing control {selector}"));
            if viewport.contains(&bounds.center()) {
                visual.simulate_click(bounds.center(), gpui::Modifiers::default());
                visual.run_until_parked();
                return;
            }
            visual.simulate_event(gpui::ScrollWheelEvent {
                position: viewport.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(
                    gpui::px(0.),
                    viewport.center().y - bounds.center().y,
                )),
                modifiers: gpui::Modifiers::default(),
                touch_phase: gpui::TouchPhase::Moved,
            });
            visual.run_until_parked();
        }
        panic!("Control {selector} did not become visible through normal scrolling");
    }

    async fn doctor_investigation_flow_source_review(
        product: diagnostics::doctor::DoctorProduct,
        cx: &mut TestAppContext,
    ) {
        use diagnostics::doctor::*;
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        doctor_investigation_flow_click(
            &mut visual,
            if product == DoctorProduct::Copper {
                "doctor-product-1"
            } else {
                "doctor-product-2"
            },
        );
        doctor_investigation_flow_click(&mut visual, "doctor-refresh");
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        filesystem
            .atomic_write(
                PathBuf::from("/project/unrelated.txt"),
                "Preserve independent edits".into(),
            )
            .await
            .expect("Independent sentinel");
        let buffer = project
            .update(&mut visual, |project, cx| {
                project.open_local_buffer("/project/file.txt", cx)
            })
            .await
            .expect("Existing source owner");
        let selection = DoctorInvestigationSelection {
            input_identity: DoctorInvestigationInputIdentity::from_fields(&[
                product.name(),
                "selected fixture",
                "debug",
            ])
            .expect("Identity"),
            scope: DoctorInvestigationScope::Application,
            problem: DoctorInvestigationProblem::Test,
            symptom: DiagnosticText::new("Selected regression fails"),
            validation_target: DiagnosticText::new("selected regression target"),
            workload: DiagnosticText::new("one selected test"),
            configuration: DiagnosticText::new("debug"),
            source_revision: DiagnosticText::new("fixture source"),
            cache_conditions: DiagnosticText::new("warm"),
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let observation = |failure| DoctorEvidenceInput {
            owner: DoctorEvidenceOwner::Test,
            owner_reference: DiagnosticText::new("fixture:selected-test"),
            classification: DoctorEvidenceClass::Observed,
            observation: if failure {
                DoctorEvidenceObservation::FunctionalFailure {
                    assertion: DiagnosticText::new("fixture:selected-assertion"),
                }
            } else {
                DoctorEvidenceObservation::FunctionalSuccess {
                    assertion: DiagnosticText::new("fixture:selected-assertion"),
                }
            },
            detail: DiagnosticText::new(
                "Deterministic selected-test fixture, not live application proof; API_KEY=DOCTOR_JOURNEY_SECRET",
            ),
            observed_at: now,
            historical: false,
        };
        let enabled = view
            .update_in(&mut visual, |view, window, cx| {
                view.configure_investigation_fixture(
                    selection.clone(),
                    vec![observation(true)],
                    window,
                    cx,
                )
            })
            .expect("Selected baseline fixture");
        if !enabled {
            assert_eq!(model.completion_count(), 0);
            return;
        }
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-scope-1");
        assert!(visual.debug_bounds("doctor-native-review").is_none());
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-native-select");
        assert!(
            visual.debug_bounds("doctor-native-review").is_some(),
            "Selecting investigation observations must reveal their Advanced repair controls"
        );
        assert_eq!(
            model.completion_count(),
            0,
            "Navigation never starts inference"
        );
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-series-0");
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-evidence-0");
        doctor_investigation_flow_click(&mut visual, "doctor-native-review");
        assert_eq!(model.completion_count(), 0);
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        let request = model
            .pending_completions()
            .first()
            .expect("Approved NativeAgent handoff")
            .clone();
        assert_eq!(model.pending_completions().len(), 1);
        assert!(
            !request
                .messages
                .last()
                .expect("Selected prompt")
                .string_contents()
                .contains("DOCTOR_JOURNEY_SECRET")
        );
        let conversation = panel.read_with(&visual, |panel, _| {
            panel
                .active_conversation_view()
                .expect("Normal conversation")
                .clone()
        });
        let thread = conversation.read_with(&visual, |conversation, cx| {
            conversation.root_thread(cx).expect("Existing AcpThread")
        });
        let input = json!({"path": "project/file.txt", "edits": [{"old_text": "Unrelated worktree edit", "new_text": "Reviewed Doctor source proposal"}]});
        model.send_completion_stream_event(
            &request,
            language_model::LanguageModelCompletionEvent::ToolUse(
                language_model::LanguageModelToolUse {
                    id: "doctor-journey-source-edit".into(),
                    name: "edit_file".into(),
                    raw_input: input.to_string(),
                    input: language_model::LanguageModelToolUseInput::Json(input),
                    is_input_complete: true,
                    thought_signature: None,
                },
            ),
        );
        model.end_completion_stream(&request);
        visual.run_until_parked();
        assert!(thread.read_with(&visual, |thread, _| thread.entries().iter().any(|entry| matches!(entry, acp_thread::AgentThreadEntry::ToolCall(call) if matches!(&call.status, acp_thread::ToolCallStatus::WaitingForConfirmation { .. })))));
        assert_eq!(
            buffer.read_with(&visual, |buffer, _| buffer.text()),
            "Unrelated worktree edit"
        );
        workspace.update_in(&mut visual, |workspace, window, cx| {
            workspace.focus_panel::<AgentPanel>(window, cx);
            conversation.read(cx).focus_handle(cx).focus(window, cx);
        });
        visual.run_until_parked();
        assert!(conversation.read_with(&visual, |conversation, cx| {
            conversation.root_thread_has_pending_tool_call(cx)
        }));
        visual.dispatch_action(crate::AllowOnce);
        visual.run_until_parked();
        assert_eq!(
            buffer.read_with(&visual, |buffer, _| buffer.text()),
            "Reviewed Doctor source proposal\n"
        );
        let continuation = model
            .pending_completions()
            .first()
            .expect("Normal tool completion")
            .clone();
        model.send_completion_stream_text_chunk(&continuation, "Fixed and verified.");
        model.send_completion_stream_event(
            &continuation,
            language_model::LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
        );
        model.end_completion_stream(&continuation);
        visual.run_until_parked();
        model.forbid_requests();
        visual.update(|window, cx| conversation.read(cx).focus_handle(cx).focus(window, cx));
        visual.dispatch_action(crate::OpenAgentDiff);
        visual.run_until_parked();
        assert!(workspace.read_with(&visual, |workspace, cx| {
            workspace.item_of_type::<crate::AgentDiffPane>(cx).is_some()
        }));
        let records = history
            .records(DoctorProduct::current(), now + 1)
            .await
            .expect("Native receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Proposal receipt").outcome,
            ReceiptOutcome::AwaitingUserAction
        );
        let session = records
            .first()
            .expect("Session receipt")
            .session_id
            .as_ref()
            .expect("Real session")
            .as_str()
            .to_owned();
        assert!(visual.read(|cx| doctor_history_session_available(&project, &session, cx)));
        workspace.update_in(&mut visual, |workspace, window, cx| {
            diagnostics::doctor_view::DoctorView::deploy(
                workspace,
                &diagnostics::Doctor,
                window,
                cx,
            )
        });
        doctor_investigation_flow_click(&mut visual, "doctor-native-session");
        assert_eq!(
            panel.read_with(&visual, |panel, cx| panel
                .active_conversation_view()
                .expect("Existing session link")
                .read(cx)
                .root_thread(cx)),
            Some(thread.clone())
        );
        workspace.update_in(&mut visual, |workspace, window, cx| {
            diagnostics::doctor_view::DoctorView::deploy(
                workspace,
                &diagnostics::Doctor,
                window,
                cx,
            )
        });
        project
            .update(&mut visual, |project, cx| {
                project.save_buffer(buffer.clone(), cx)
            })
            .await
            .expect("Explicit reviewed fixture source save before fresh selected observation");
        visual.run_until_parked();
        let mut after_selection = selection;
        after_selection.input_identity = DoctorInvestigationInputIdentity::from_fields(&[
            product.name(),
            "selected fixture after reviewed source change",
            "debug",
        ])
        .expect("Changed source input identity");
        after_selection.source_revision = DiagnosticText::new("reviewed fixture source");
        view.update_in(&mut visual, |view, window, cx| {
            view.configure_investigation_fixture(
                after_selection,
                vec![observation(false)],
                window,
                cx,
            )
        })
        .expect("Selected after-change fixture");
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-series-2");
        doctor_investigation_flow_click(&mut visual, "doctor-comparison-before-functional-0");
        doctor_investigation_flow_click(&mut visual, "doctor-comparison-after-functional-0");
        view.update_in(&mut visual, |view, window, cx| {
            view.configure_comparison_change_fixture(
                "Reviewed source edit in the linked native session",
                window,
                cx,
            )
        });
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-compare");
        view.read_with(&visual, |view, _| {
            let comparison = view
                .report_fixture()
                .expect("Current report")
                .comparison()
                .expect("Explicit relevant comparison");
            assert_eq!(
                comparison.outcome,
                DoctorComparisonOutcome::VerifiedSelectedTarget
            );
            assert!(comparison.change.session_id.is_none(), "Fixture observations do not invent a reproduction-linked session; the actual native proposal receipt retains its real session");
        });
        doctor_investigation_flow_click(&mut visual, "doctor-history-save");
        doctor_investigation_flow_click(&mut visual, "doctor-history-confirm");
        visual.run_until_parked();
        let records = history
            .records(DoctorProduct::current(), now + 2)
            .await
            .expect("Durable reviewed comparison");
        assert!(
            records.iter().any(|record| record
                .report
                .comparison()
                .is_some_and(|comparison| comparison.outcome
                    == DoctorComparisonOutcome::VerifiedSelectedTarget))
        );
        assert!(
            !serde_json::to_string(&records)
                .expect("Redacted history")
                .contains("DOCTOR_JOURNEY_SECRET")
        );
        assert_eq!(
            filesystem
                .load(Path::new("/project/unrelated.txt"))
                .await
                .expect("Independent edits retained"),
            "Preserve independent edits"
        );
        assert_eq!(
            filesystem
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Explicitly reviewed saved source"),
            "Reviewed Doctor source proposal\n"
        );
        assert!(
            model.pending_completions().is_empty(),
            "Rechecking never launches an extra provider turn"
        );
        buffer.update(&mut visual, |buffer, cx| {
            buffer.edit([(0..0, "later independent edit\n")], None, cx)
        });
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-compare");
        assert_eq!(
            history
                .records(DoctorProduct::current(), now + 3)
                .await
                .expect("No automatic replay")
                .len(),
            records.len()
        );
    }

    #[gpui::test]
    async fn doctor_investigation_flow_copper_native_source_review_and_recheck(
        cx: &mut TestAppContext,
    ) {
        doctor_investigation_flow_source_review(diagnostics::doctor::DoctorProduct::Copper, cx)
            .await;
    }

    #[gpui::test]
    async fn doctor_investigation_flow_groovy_native_source_review_and_recheck(
        cx: &mut TestAppContext,
    ) {
        doctor_investigation_flow_source_review(diagnostics::doctor::DoctorProduct::Groovy, cx)
            .await;
    }

    #[gpui::test]
    async fn doctor_investigation_repair_desktop_selected_evidence_review_session_and_unverified_result(
        cx: &mut TestAppContext,
    ) {
        use diagnostics::doctor::*;
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let selection = DoctorInvestigationSelection {
            input_identity: DoctorInvestigationInputIdentity::from_fields(&[
                "selected test",
                "fixture source",
                "debug",
            ])
            .expect("Bounded identity"),
            scope: DoctorInvestigationScope::Application,
            problem: DoctorInvestigationProblem::Test,
            symptom: DiagnosticText::new("Selected regression fails"),
            validation_target: DiagnosticText::new("selected regression target"),
            workload: DiagnosticText::new("one selected test"),
            configuration: DiagnosticText::new("debug"),
            source_revision: DiagnosticText::new("fixture source"),
            cache_conditions: DiagnosticText::new("warm"),
        };
        let enabled = view
            .update_in(&mut visual, |view, window, cx| {
                view.configure_investigation_fixture(
                    selection,
                    vec![
                        DoctorEvidenceInput {
                            owner: DoctorEvidenceOwner::Test,
                            owner_reference: DiagnosticText::new("selected owner"),
                            classification: DoctorEvidenceClass::Observed,
                            observation: DoctorEvidenceObservation::FunctionalFailure {
                                assertion: DiagnosticText::new("selected regression assertion"),
                            },
                            detail: DiagnosticText::new(
                                "API_KEY=UI_HANDOFF_PRIVATE observed baseline",
                            ),
                            observed_at: now + 1,
                            historical: false,
                        },
                        DoctorEvidenceInput {
                            owner: DoctorEvidenceOwner::User,
                            owner_reference: DiagnosticText::new("user report"),
                            classification: DoctorEvidenceClass::UserReported,
                            observation: DoctorEvidenceObservation::Unverified,
                            detail: DiagnosticText::new("User symptom remains unverified"),
                            observed_at: now + 1,
                            historical: false,
                        },
                        DoctorEvidenceInput {
                            owner: DoctorEvidenceOwner::NativeAgent,
                            owner_reference: DiagnosticText::new("unselected model hypothesis"),
                            classification: DoctorEvidenceClass::Hypothesis,
                            observation: DoctorEvidenceObservation::Unverified,
                            detail: DiagnosticText::new("UNSELECTED_UI_SENTINEL"),
                            observed_at: now + 1,
                            historical: false,
                        },
                    ],
                    window,
                    cx,
                )
            })
            .expect("Owner-backed fixture evidence");
        if !enabled {
            assert_eq!(model.completion_count(), 0);
            return;
        }
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-evidence-0");
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-evidence-1");
        doctor_investigation_flow_click(&mut visual, "doctor-native-review");
        assert_eq!(
            model.completion_count(),
            0,
            "Review has no provider effects"
        );
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        assert!(
            view.read_with(&visual, |view, _| view.error_fixture().is_none()),
            "Selected evidence review failed: {:?}",
            view.read_with(&visual, |view, _| view.error_fixture().map(str::to_owned))
        );
        let request = model
            .pending_completions()
            .first()
            .expect("One approved handoff")
            .clone();
        assert_eq!(model.pending_completions().len(), 1);
        let prompt = request
            .messages
            .last()
            .expect("Selected prompt")
            .string_contents();
        assert!(
            prompt.contains("selected regression target")
                && prompt.contains("baseline_policy")
                && prompt.contains("UserReported")
        );
        assert!(
            !prompt.contains("UI_HANDOFF_PRIVATE") && !prompt.contains("UNSELECTED_UI_SENTINEL")
        );
        assert!(panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Existing conversation")
                .read(cx)
                .as_native_thread(cx)
                .expect("Existing Thread")
                .read(cx)
                .is_doctor_repair_proposal()
        }));
        model.send_completion_stream_text_chunk(&request, "Fixed and verified.");
        model.send_completion_stream_event(
            &request,
            language_model::LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        visual.run_until_parked();
        let records = history
            .records(DoctorProduct::current(), now + 2)
            .await
            .expect("Durable history");
        assert_eq!(records.len(), 1);
        let record = records.first().expect("Receipt");
        assert_eq!(
            record.outcome,
            ReceiptOutcome::AwaitingUserAction,
            "Assertion is not verification"
        );
        let session = record
            .session_id
            .as_ref()
            .expect("Existing session link")
            .as_str();
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        assert!(visual.read(|cx| doctor_history_session_available(&project, session, cx)));
        assert_eq!(
            project
                .read_with(&visual, |project, _| project.fs().clone())
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Preserved unrelated file"),
            "Unrelated worktree edit"
        );
    }

    #[cfg(feature = "doctor-dependency-test")]
    fn doctor_dependency_repair_click(visual: &mut VisualTestContext, selector: &'static str) {
        for _ in 0..16 {
            visual
                .refresh()
                .expect("Rendered bounded dependency review");
            let viewport = visual.debug_bounds("made-doctor").expect("Doctor viewport");
            if let Some(bounds) = visual.debug_bounds(selector) {
                if viewport.contains(&bounds.center()) {
                    visual.simulate_click(bounds.center(), gpui::Modifiers::default());
                    visual.run_until_parked();
                    return;
                }
            }
            visual.simulate_event(gpui::ScrollWheelEvent {
                position: viewport.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(
                    gpui::px(0.),
                    -viewport.size.height * 0.8,
                )),
                modifiers: gpui::Modifiers::default(),
                touch_phase: gpui::TouchPhase::Moved,
            });
            visual.run_until_parked();
        }
        panic!("Dependency review control {selector} remained unavailable after native scrolling");
    }

    #[cfg(feature = "doctor-dependency-test")]
    async fn doctor_dependency_flow_owner_graph(
        project: &Entity<Project>,
        visual: &mut VisualTestContext,
        maven: bool,
        version: &str,
    ) {
        if maven {
            let filesystem = project.read_with(visual, |project, _| project.fs().clone());
            filesystem.as_fake().insert_tree("/project", json!({
                "pom.xml":"<project><groupId>evidence</groupId><artifactId>doctor-report</artifactId><version>1.0</version></project>",
                "src":{"Main.java":"preserved application source"}
            })).await;
            visual.run_until_parked();
            let root = project.read_with(visual, |project, cx| {
                project
                    .find_project_path(Path::new("/project"), cx)
                    .expect("Selected Maven root")
            });
            let store = project.read_with(visual, |project, _| project.lsp_store());
            let inputs = store
                .read_with(visual, |store, cx| {
                    store.prepare_reviewed_jvm_dependency_import(
                        root,
                        project::jvm_project::JvmReviewedDependencyFormat::MavenSelectedTree,
                        cx,
                    )
                })
                .expect("Reviewed selected JVM import");
            store.update(visual, |store,cx| store.admit_reviewed_jvm_dependency_import(inputs,include_bytes!("../../project/test_data/groovy_jvm_maven/doctor-selected-dependencies-v1.json").to_vec(),cx)).await.expect("Actual JVM owner admission");
            return;
        }
        let filesystem = project.read_with(visual, |project, _| project.fs().clone());
        if !filesystem.is_file(Path::new("/project/Cargo.toml")).await {
            filesystem.as_fake().insert_tree("/project",serde_json::json!({
            "Cargo.toml":"[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n",
            "member-one":{"Cargo.toml":"[package]\nname = \"member-one\"\nversion = \"0.1.0\"\n","src":{"lib.rs":"preserved source"}},
            "member-two":{"Cargo.toml":"[package]\nname = \"member-two\"\nversion = \"0.2.0\"\n","src":{"lib.rs":"other source"}}
        })).await;
        }
        visual.run_until_parked();
        let manifest = project.read_with(visual, |project, cx| {
            project
                .find_project_path(Path::new("/project/Cargo.toml"), cx)
                .expect("Selected Cargo manifest")
        });
        let store = project.read_with(visual, |project, _| project.cargo_workspace_store().clone());
        let tests = project.read_with(visual, |project, _| {
            project.rust_test_provider_store().clone()
        });
        drop(tests.update(visual, |tests, cx| tests.refresh(cx)));
        visual.run_until_parked();
        let inputs = store
            .read_with(visual, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest, cx)
            })
            .expect("Existing importer authority");
        inputs
            .preflight_filesystem(filesystem.as_ref())
            .await
            .expect("Existing confinement");
        let bytes = String::from_utf8_lossy(include_bytes!(
            "../../project/test_data/cargo_workspace/workspace-v1.json"
        ))
        .replace("/workspace", "/project");
        let mut metadata: serde_json::Value =
            serde_json::from_str(&bytes).expect("Owner metadata fixture");
        let mut package = metadata["packages"][1].clone();
        let identity = format!(
            "registry+https://github.com/rust-lang/crates.io-index#public-example@{version}"
        );
        package["name"] = "public-example".into();
        package["version"] = version.into();
        package["id"] = identity.clone().into();
        package["source"] = "registry+https://github.com/rust-lang/crates.io-index".into();
        package["manifest_path"] = "/cargo-cache/public-example/Cargo.toml".into();
        metadata["packages"]
            .as_array_mut()
            .expect("Packages")
            .push(package);
        metadata["resolve"]["nodes"]
            .as_array_mut()
            .expect("Resolution")
            .push(serde_json::json!({"id":identity,"dependencies":[],"deps":[],"features":[]}));
        metadata["resolve"]["nodes"][0]["deps"].as_array_mut().expect("Introducing edge").push(serde_json::json!({"name":"public_example","pkg":identity,"dep_kinds":[{"kind":null,"target":null}]}));
        store
            .update(visual, |store, cx| {
                store.admit_reviewed_metadata_import(
                    inputs,
                    serde_json::to_vec(&metadata).expect("Metadata bytes"),
                    cx,
                )
            })
            .await
            .expect("Current ecosystem-owned graph");
    }

    #[cfg(feature = "doctor-dependency-test")]
    async fn doctor_dependency_flow_selected_catalog_native_review(
        maven: bool,
        repair: bool,
        cx: &mut TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        workspace.update_in(&mut visual, |workspace, window, cx| {
            diagnostics::doctor_view::DoctorView::register(workspace, Some(window), cx)
        });
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        doctor_dependency_flow_owner_graph(&project, &mut visual, maven, "1.0.0").await;
        view.update_in(&mut visual, |view, window, cx| {
            view.configure_dependency_security_inputs_fixture(
                maven,
                "selected project build and affected tests",
                "org.junit.jupiter:junit-jupiter-api",
                window,
                cx,
            )
        });
        visual.run_until_parked();
        let archive =
            diagnostics::doctor_view::DoctorView::dependency_security_archive_fixture(maven, false)
                .await
                .expect("Public fixture archive");
        let withdrawals =
            diagnostics::doctor_view::DoctorView::dependency_security_archive_fixture(maven, true)
                .await
                .expect("Withdrawals archive");
        let requests = Arc::new(AtomicUsize::new(0));
        project.read_with(&visual,|project,_|project.client().http_client()).as_fake().replace_handler({
            let requests=requests.clone();
            move |_,request| {
                let requests=requests.clone();let archive=archive.clone();let withdrawals=withdrawals.clone();
                async move {
                    assert_eq!(request.method(),"GET");
                    assert!(request.headers().get("Authorization").is_none()&&request.headers().get("Cookie").is_none());
                    let url=request.uri().to_string();
                    let body=if url=="https://codeload.github.com/RustSec/advisory-db/zip/1111111111111111111111111111111111111111" { assert!(!maven);archive }
                    else if url=="https://storage.googleapis.com/osv-vulnerabilities/Maven/all.zip" {assert!(maven);archive}
                    else if url=="https://storage.googleapis.com/osv-vulnerabilities/%5BEMPTY%5D/all.zip" {assert!(maven);withdrawals}
                    else if url=="https://crates.io/api/v1/crates/public-example/versions" {assert!(!maven);br#"{"versions":[{"crate":"public-example","num":"2.0.0","yanked":false}],"meta":{"total":1,"next_page":null}}"#.to_vec()}
                    else if url=="https://repo.maven.apache.org/maven2/org/junit/jupiter/junit-jupiter-api/maven-metadata.xml" {assert!(maven);br#"<metadata><groupId>org.junit.jupiter</groupId><artifactId>junit-jupiter-api</artifactId><versioning><versions><version>5.12.0</version></versions></versioning></metadata>"#.to_vec()}
                    else {anyhow::bail!("Unexpected non-reviewed destination: {url}")};
                    requests.fetch_add(1,Ordering::SeqCst);
                    Ok(http_client::Response::builder().status(200).header("Date",chrono::Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()).header("Last-Modified",chrono::Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()).body(http_client::AsyncBody::from(body))?)
                }
            }
        });
        doctor_investigation_flow_click(&mut visual, "doctor-investigation-development-only");
        doctor_investigation_flow_click(&mut visual, "doctor-security-open");
        doctor_investigation_flow_click(&mut visual, "doctor-security-inventory");
        let name = if maven {
            "org.junit.jupiter:junit-jupiter-api"
        } else {
            "public-example"
        };
        let selected = view
            .read_with(&visual, |view, _| {
                view.dependency_security_instance_fixture(name)
            })
            .expect("Actual selected instance");
        let selector: &'static str =
            Box::leak(format!("doctor-security-instance-{selected}").into_boxed_str());
        doctor_investigation_flow_click(&mut visual, selector);
        doctor_investigation_flow_click(
            &mut visual,
            if maven {
                "doctor-security-osv-refresh"
            } else {
                "doctor-security-rustsec-refresh"
            },
        );
        visual
            .condition(&view, |view, _| !view.dependency_security_busy_fixture())
            .await;
        assert_eq!(
            requests.load(Ordering::SeqCst),
            0,
            "No refresh before explicit concrete approval"
        );
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| !view.dependency_security_busy_fixture())
            .await;
        let archive_requests = if maven { 2 } else { 1 };
        assert_eq!(
            requests.load(Ordering::SeqCst),
            archive_requests,
            "Exact reviewed public archives"
        );
        assert!(view.read_with(&visual, |view, _| view.error_fixture().is_none()));
        doctor_investigation_flow_click(&mut visual, "doctor-security-scan");
        visual
            .condition(&view, |view, _| !view.dependency_security_busy_fixture())
            .await;
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| !view.dependency_security_busy_fixture())
            .await;
        assert_eq!(
            requests.load(Ordering::SeqCst),
            archive_requests,
            "Matching stays local"
        );
        doctor_investigation_flow_click(
            &mut visual,
            if maven {
                "doctor-security-public-catalog"
            } else {
                "doctor-security-query"
            },
        );
        visual
            .condition(&view, |view, _| !view.dependency_security_busy_fixture())
            .await;
        assert_eq!(
            requests.load(Ordering::SeqCst),
            archive_requests,
            "Catalog disclosure needs separate approval"
        );
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        visual
            .condition(&view, |view, _| !view.dependency_security_busy_fixture())
            .await;
        assert_eq!(requests.load(Ordering::SeqCst), archive_requests + 1);
        assert_eq!(model.completion_count(), 0);
        doctor_investigation_flow_click(&mut visual, "doctor-security-fix-0");
        assert!(
            view.read_with(&visual, |view, _| view.error_fixture().is_none()),
            "Native candidate review: {:?}",
            view.read_with(&visual, |view, _| view.error_fixture().map(str::to_owned))
        );
        assert_eq!(
            model.completion_count(),
            0,
            "Selecting an alternative is not permission to submit it"
        );
        doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
        let request = model
            .pending_completions()
            .first()
            .expect("Existing NativeAgent session")
            .clone();
        let prompt = request
            .messages
            .last()
            .expect("Exact selection")
            .string_contents();
        assert!(prompt.contains(name) && prompt.contains(if maven { "5.12.0" } else { "2.0.0" }));
        assert!(
            !prompt.contains("PUBLIC_BODY_NOT_HISTORY")
                && !prompt.contains("Unrelated worktree edit")
                && !prompt.contains("/cargo-cache")
        );
        if maven {
            assert!(
                prompt.contains("catalog") && prompt.contains("Unknown"),
                "Catalog request cannot establish artifact origin"
            );
        }
        assert!(request.tools.iter().all(|tool| !matches!(
            tool.name.as_str(),
            "terminal" | "fetch" | "web_search" | "spawn_agent" | "propose_recipe"
        )));
        let session = panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Existing conversation")
                .read(cx)
                .root_thread(cx)
                .expect("AcpThread")
                .read(cx)
                .session_id()
                .to_string()
        });
        if repair {
            assert!(!maven);
            let thread = panel.read_with(&visual, |panel, cx| {
                panel
                    .active_conversation_view()
                    .expect("Native proposal conversation")
                    .read(cx)
                    .root_thread(cx)
                    .expect("Existing AcpThread")
            });
            let input = json!({"path":"project/member-one/Cargo.toml","edits":[{"old_text":"version = \"0.1.0\"","new_text":"version = \"0.1.0\"\n\n[dependencies]\npublic-example = \"2.0.0\""}]});
            model.send_completion_stream_event(
                &request,
                language_model::LanguageModelCompletionEvent::ToolUse(
                    language_model::LanguageModelToolUse {
                        id: "doctor-selected-dependency-edit".into(),
                        name: "edit_file".into(),
                        raw_input: input.to_string(),
                        input: language_model::LanguageModelToolUseInput::Json(input),
                        is_input_complete: true,
                        thought_signature: None,
                    },
                ),
            );
            model.end_completion_stream(&request);
            visual.run_until_parked();
            let approval = thread.read_with(&visual, |thread, _| {
                thread
                    .entries()
                    .iter()
                    .find_map(|entry| match entry {
                        acp_thread::AgentThreadEntry::ToolCall(call) => match &call.status {
                            acp_thread::ToolCallStatus::WaitingForConfirmation {
                                options, ..
                            } => {
                                assert!(
                                    options
                                        .first_option_of_kind(
                                            acp::PermissionOptionKind::AllowAlways
                                        )
                                        .is_none()
                                );
                                Some(call.id.clone())
                            }
                            _ => None,
                        },
                        _ => None,
                    })
                    .expect("Normal once-only source permission")
            });
            let buffer = project
                .update(&mut visual, |project, cx| {
                    project.open_local_buffer("/project/member-one/Cargo.toml", cx)
                })
                .await
                .expect("Existing change-review buffer");
            assert!(
                !buffer
                    .read_with(&visual, |buffer, _| buffer.text())
                    .contains("public-example")
            );
            let conversation = panel.read_with(&visual, |panel, _| {
                panel
                    .active_conversation_view()
                    .expect("Native permission surface")
                    .clone()
            });
            workspace.update_in(&mut visual, |workspace, window, cx| {
                workspace.focus_panel::<AgentPanel>(window, cx);
                conversation.read(cx).focus_handle(cx).focus(window, cx);
            });
            visual.run_until_parked();
            visual.dispatch_action(crate::AuthorizeToolCall {
                tool_call_id: approval.to_string(),
                option_id: "allow".into(),
                option_kind: "AllowOnce".into(),
            });
            visual.run_until_parked();
            assert!(
                buffer
                    .read_with(&visual, |buffer, _| buffer.text())
                    .contains("public-example = \"2.0.0\"")
            );
            let continuation = model
                .pending_completions()
                .first()
                .expect("Reviewed proposal continuation")
                .clone();
            model.send_completion_stream_event(
                &continuation,
                language_model::LanguageModelCompletionEvent::Stop(
                    language_model::StopReason::EndTurn,
                ),
            );
            model.end_completion_stream(&continuation);
            visual.run_until_parked();
            visual
                .condition(&view, |view, _| !view.dependency_security_busy_fixture())
                .await;
            project
                .update(&mut visual, |project, cx| {
                    project.save_buffer(buffer.clone(), cx)
                })
                .await
                .expect("Explicit normal source save");
            visual.run_until_parked();
            doctor_dependency_flow_owner_graph(&project, &mut visual, false, "2.0.0").await;
            visual.dispatch_action(diagnostics::Doctor);
            visual.run_until_parked();
            doctor_investigation_flow_click(&mut visual, "doctor-security-inventory");
            doctor_investigation_flow_click(&mut visual, "doctor-security-scan");
            visual
                .condition(&view, |view, _| !view.dependency_security_busy_fixture())
                .await;
            doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
            visual
                .condition(&view, |view, _| !view.dependency_security_busy_fixture())
                .await;
            let (started, launches) = async_channel::bounded(1);
            workspace.update(&mut visual, |workspace, _| {
                workspace.set_terminal_provider(DoctorDependencyFlowTaskFixture { started })
            });
            doctor_investigation_flow_click(&mut visual, "doctor-investigation-review-command");
            visual
                .condition(&view, |view, _| !view.dependency_security_busy_fixture())
                .await;
            assert!(
                launches.is_empty(),
                "Relevant tests need their own concrete approval"
            );
            doctor_investigation_flow_click(&mut visual, "doctor-approve-repair");
            let launch = launches.recv().await.expect("Existing selected Task owner");
            assert_eq!(launch.save, task::SaveStrategy::None);
            assert!(
                launch
                    .command
                    .as_deref()
                    .is_some_and(|command| command.contains("cargo test")
                        && command.contains("--locked --offline"))
            );
            visual
                .condition(&view, |view, _| !view.dependency_security_busy_fixture())
                .await;
            doctor_investigation_flow_click(&mut visual, "doctor-investigation-series-2");
            doctor_investigation_flow_click(&mut visual, "doctor-comparison-after-functional-0");
            doctor_investigation_flow_click(&mut visual, "doctor-security-verify");
            visual
                .condition(&view, |view, _| !view.dependency_security_busy_fixture())
                .await;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Clock")
                .as_secs();
            let records = history
                .records(diagnostics::doctor::DoctorProduct::current(), now)
                .await
                .expect("Source proposal and independent recheck receipts");
            assert!(records.iter().any(|record| {
                record
                    .session_id
                    .as_ref()
                    .is_some_and(|id| id.as_str() == session)
            }));
            assert!(
                records.iter().any(|record| record.outcome
                    == diagnostics::doctor::ReceiptOutcome::AwaitingUserAction)
            );
            assert!(
                records.last().expect("Relevant Task receipt").outcome
                    == diagnostics::doctor::ReceiptOutcome::Incomplete,
                "No test-harness observation and no overall security proof"
            );
            assert!(view.read_with(&visual, |view, _| {
                view.report_fixture()
                    .expect("Bounded observations")
                    .checks()
                    .iter()
                    .any(|check| {
                        check.id.as_str() == "doctor.dependency.verification"
                            && check.outcome != diagnostics::doctor::CheckOutcome::Ready
                    })
            }));
            assert!(
                launches.is_empty(),
                "Recheck comparison and navigation never replay a Task"
            );
            assert!(visual.read(|cx| doctor_history_session_available(&project, &session, cx)));
            assert_eq!(
                requests.load(Ordering::SeqCst),
                archive_requests + 1,
                "Fresh advisory rechecks stay local"
            );
            assert_eq!(
                project
                    .read_with(&visual, |project, _| project.fs().clone())
                    .load(Path::new("/project/file.txt"))
                    .await
                    .expect("Unrelated edits"),
                "Unrelated worktree edit"
            );
            return;
        }
        visual.dispatch_action(diagnostics::Doctor);
        visual.run_until_parked();
        assert_eq!(
            workspace.read_with(&visual, |workspace, cx| workspace
                .item_of_type::<diagnostics::doctor_view::DoctorView>(cx)
                .expect("Reopened existing Doctor")
                .entity_id()),
            view.entity_id()
        );
        doctor_dependency_repair_click(&mut visual, "doctor-cancel-repair");
        visual.executor().advance_clock(Duration::from_millis(200));
        visual.run_until_parked();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let records = history
            .records(diagnostics::doctor::DoctorProduct::current(), now)
            .await
            .expect("Existing durable receipts");
        assert_eq!(
            records.last().expect("Native receipt").outcome,
            diagnostics::doctor::ReceiptOutcome::Cancelled
        );
        assert_eq!(
            records
                .last()
                .expect("Session link")
                .session_id
                .as_ref()
                .map(diagnostics::doctor::DiagnosticText::as_str),
            Some(session.as_str())
        );
        assert!(visual.read(|cx| doctor_history_session_available(&project, &session, cx)));
        doctor_investigation_flow_click(&mut visual, "doctor-security-verify");
        assert!(
            view.read_with(&visual, |view, _| view.error_fixture().is_some()),
            "Cancelled proposal and missing relevant tests never verify a fix"
        );
        assert_eq!(requests.load(Ordering::SeqCst), archive_requests + 1);
        assert_eq!(
            project
                .read_with(&visual, |project, _| project.fs().clone())
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Unrelated edits"),
            "Unrelated worktree edit"
        );
        let persisted = serde_json::to_string(&records).expect("Bounded receipts");
        assert!(
            !persisted.contains("PUBLIC_BODY_NOT_HISTORY")
                && !persisted.contains("/cargo-cache")
                && !persisted.contains("maven-metadata.xml")
        );
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[gpui::test]
    async fn doctor_dependency_flow_copper_catalog_review_native_session_cancel_and_unverified(
        cx: &mut TestAppContext,
    ) {
        doctor_dependency_flow_selected_catalog_native_review(false, false, cx).await;
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[gpui::test]
    async fn doctor_dependency_flow_groovy_unknown_origin_catalog_native_review_and_cancel(
        cx: &mut TestAppContext,
    ) {
        doctor_dependency_flow_selected_catalog_native_review(true, false, cx).await;
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[gpui::test]
    async fn doctor_dependency_flow_copper_reviewed_source_change_current_graph_and_relevant_task_recheck(
        cx: &mut TestAppContext,
    ) {
        doctor_dependency_flow_selected_catalog_native_review(false, true, cx).await;
    }

    #[cfg(feature = "doctor-dependency-test")]
    struct DoctorDependencyFlowTaskFixture {
        started: async_channel::Sender<task::SpawnInTerminal>,
    }

    #[cfg(feature = "doctor-dependency-test")]
    impl workspace::TerminalProvider for DoctorDependencyFlowTaskFixture {
        fn spawn(
            &self,
            _: task::SpawnInTerminal,
            _: &mut Window,
            _: &mut App,
        ) -> Task<Option<Result<std::process::ExitStatus>>> {
            panic!("The Doctor journey must use its existing structured Task handle");
        }
        fn spawn_structured(
            &self,
            task: task::SpawnInTerminal,
            handle: task::StructuredTaskHandle,
            _: &mut Window,
            cx: &mut App,
        ) -> Task<Option<Result<std::process::ExitStatus>>> {
            #[cfg(unix)]
            use std::os::unix::process::ExitStatusExt as _;
            #[cfg(windows)]
            use std::os::windows::process::ExitStatusExt as _;
            self.started
                .try_send(task)
                .expect("One explicitly reviewed relevant Task");
            handle.bind_terminal(task::StructuredTerminalId(900), |_| true, cx);
            Task::ready(Some(Ok(std::process::ExitStatus::from_raw(0))))
        }
    }

    #[cfg(feature = "doctor-dependency-test")]
    #[gpui::test]
    async fn doctor_dependency_repair_desktop_selected_prompt_native_session_cancel_and_history(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        filesystem.as_fake().insert_tree("/project",serde_json::json!({
            "Cargo.toml":"[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n",
            "member-one":{"Cargo.toml":"[package]\nname = \"member-one\"\nversion = \"0.1.0\"\n","src":{"lib.rs":"preserved source"}},
            "member-two":{"Cargo.toml":"[package]\nname = \"member-two\"\nversion = \"0.2.0\"\n","src":{"lib.rs":"other source"}}
        })).await;
        visual.run_until_parked();
        let manifest = project.read_with(&visual, |project, cx| {
            project
                .find_project_path(Path::new("/project/Cargo.toml"), cx)
                .expect("Selected Cargo manifest")
        });
        let store = project.read_with(&visual, |project, _| {
            project.cargo_workspace_store().clone()
        });
        let inputs = store
            .read_with(&visual, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest, cx)
            })
            .expect("Existing importer authority");
        inputs
            .preflight_filesystem(filesystem.as_ref())
            .await
            .expect("Existing confinement");
        let bytes = String::from_utf8_lossy(include_bytes!(
            "../../project/test_data/cargo_workspace/workspace-v1.json"
        ))
        .replace("/workspace", "/project");
        let mut metadata: serde_json::Value =
            serde_json::from_str(&bytes).expect("Owner metadata fixture");
        let mut package = metadata["packages"][1].clone();
        let identity = "registry+https://github.com/rust-lang/crates.io-index#public-example@1.0.0";
        package["name"] = "public-example".into();
        package["version"] = "1.0.0".into();
        package["id"] = identity.into();
        package["source"] = "registry+https://github.com/rust-lang/crates.io-index".into();
        package["manifest_path"] = "/cargo-cache/public-example/Cargo.toml".into();
        metadata["packages"]
            .as_array_mut()
            .expect("Packages")
            .push(package);
        metadata["resolve"]["nodes"]
            .as_array_mut()
            .expect("Resolution")
            .push(serde_json::json!({"id":identity,"dependencies":[],"deps":[],"features":[]}));
        metadata["resolve"]["nodes"][0]["deps"].as_array_mut().expect("Introducing edge").push(serde_json::json!({"name":"public_example","pkg":identity,"dep_kinds":[{"kind":null,"target":null}]}));
        store
            .update(&mut visual, |store, cx| {
                store.admit_reviewed_metadata_import(
                    inputs,
                    serde_json::to_vec(&metadata).expect("Metadata bytes"),
                    cx,
                )
            })
            .await
            .expect("Current ecosystem-owned graph");
        let enabled = view
            .update_in(&mut visual, |view, window, cx| {
                view.configure_dependency_repair_fixture(
                    "selected workspace build and affected tests",
                    window,
                    cx,
                )
            })
            .expect("Concrete selection and production handoff");
        assert!(
            enabled,
            "The explicit Doctor dependency test feature must enable its production review"
        );
        assert_eq!(
            model.completion_count(),
            0,
            "No prompt before explicit desktop approval"
        );
        doctor_dependency_repair_click(&mut visual, "doctor-approve-repair");
        let request = model
            .pending_completions()
            .first()
            .expect("Approved normal native request")
            .clone();
        let prompt = request
            .messages
            .last()
            .expect("Selected evidence")
            .string_contents();
        assert!(
            prompt.contains("public-example 2.0.0")
                && prompt.contains("only the exact explicitly selected dependency")
        );
        assert!(!prompt.contains("/cargo-cache") && !prompt.contains("Unrelated worktree edit"));
        assert!(request.tools.iter().all(|tool| !matches!(
            tool.name.as_str(),
            "terminal"
                | "fetch"
                | "web_search"
                | "spawn_agent"
                | "create_thread"
                | "propose_recipe"
        )));
        let session = panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Existing conversation")
                .read(cx)
                .root_thread(cx)
                .expect("Existing AcpThread")
                .read(cx)
                .session_id()
                .to_string()
        });
        assert!(
            view.read_with(&visual, |view, cx| workspace::item::Item::restart_blocker(
                view, cx
            )
            .is_some()),
            "Pending dependency proposal remains owned by Doctor until cancellation or completion"
        );
        view.update(&mut visual, |_, cx| cx.notify());
        doctor_dependency_repair_click(&mut visual, "doctor-cancel-repair");
        visual.executor().advance_clock(Duration::from_millis(200));
        visual.run_until_parked();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let records = history
            .records(diagnostics::doctor::DoctorProduct::current(), now)
            .await
            .expect("Durable cancellation receipt");
        let record = records.last().expect("Receipt");
        assert_eq!(
            record.outcome,
            diagnostics::doctor::ReceiptOutcome::Cancelled
        );
        assert_eq!(
            record
                .session_id
                .as_ref()
                .map(diagnostics::doctor::DiagnosticText::as_str),
            Some(session.as_str())
        );
        assert!(visual.read(|cx| doctor_history_session_available(&project, &session, cx)));
        assert_eq!(
            filesystem
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Preserved unrelated file"),
            "Unrelated worktree edit"
        );
        let retained = serde_json::to_string(&records).expect("Redacted history");
        assert!(
            retained.contains("public-example"),
            "Approved action metadata identifies the selected public dependency"
        );
        assert!(
            !retained.contains("introducing_paths")
                && !retained.contains("registry_revision")
                && !retained.contains("/cargo-cache"),
            "History excludes graph and prompt payloads"
        );
    }

    #[gpui::test]
    async fn doctor_native_repair_desktop_explicit_approval_session_history_and_unverified_completion(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        let enabled = view
            .update_in(&mut visual, |view, window, cx| {
                view.configure_native_repair_fixture("selected functional test", window, cx)
            })
            .expect("Fixture selection");
        if !enabled {
            assert!(visual.debug_bounds("doctor-native-review").is_none());
            assert_eq!(model.completion_count(), 0);
            return;
        }
        doctor_native_repair_click(&mut visual, "doctor-native-review");
        assert_eq!(model.completion_count(), 0, "Review never submits a prompt");
        assert!(
            history
                .records(
                    diagnostics::doctor::DoctorProduct::current(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("Clock")
                        .as_secs()
                )
                .await
                .expect("No receipt before approval")
                .is_empty()
        );
        doctor_native_repair_click(&mut visual, "doctor-approve-repair");
        let request = model
            .pending_completions()
            .first()
            .expect("One approved native request")
            .clone();
        assert_eq!(model.pending_completions().len(), 1);
        assert!(request.tools.iter().all(|tool| {
            !matches!(
                tool.name.as_str(),
                "terminal"
                    | "fetch"
                    | "web_search"
                    | "spawn_agent"
                    | "create_thread"
                    | "propose_recipe"
            )
        }));
        assert!(panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Native view")
                .read(cx)
                .as_native_thread(cx)
                .expect("Native Thread")
                .read(cx)
                .is_doctor_repair_proposal()
        }));
        panel.read_with(&visual, |panel, cx| {
            let conversation = panel
                .active_conversation_view()
                .expect("Native proposal view")
                .read(cx);
            let thread = conversation.root_thread(cx).expect("Existing ACP session");
            let session_id = thread.read(cx).session_id().clone();
            assert!(
                thread.read(cx).available_commands().is_empty(),
                "Source proposals do not advertise command execution"
            );
            assert!(
                conversation
                    .as_native_connection(cx)
                    .expect("Native connection")
                    .available_skills(&session_id, cx)
                    .is_empty(),
                "Existing skill autocomplete respects the same purpose"
            );
        });
        assert!(
            request
                .messages
                .last()
                .expect("Selected prompt")
                .string_contents()
                .contains("selected functional test")
        );
        let session = panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Normal native conversation")
                .read(cx)
                .root_thread(cx)
                .expect("Existing AcpThread owner")
                .read(cx)
                .session_id()
                .to_string()
        });
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let records = history
            .records(diagnostics::doctor::DoctorProduct::current(), now)
            .await
            .expect("Durable running receipt");
        assert_eq!(records.len(), 1);
        view.update(&mut visual, |_, cx| cx.notify());
        visual.refresh().expect("Live native session navigation");
        assert!(
            visual.debug_bounds("doctor-native-session").is_some(),
            "The running session remains explicitly available for tool permissions and change review"
        );
        assert_eq!(
            records
                .first()
                .expect("Receipt")
                .session_id
                .as_ref()
                .map(diagnostics::doctor::DiagnosticText::as_str),
            Some(session.as_str())
        );
        model.send_completion_stream_text_chunk(&request, "The issue is fixed; all tests pass.");
        model.send_completion_stream_event(
            &request,
            language_model::LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        visual.run_until_parked();
        let records = history
            .records(diagnostics::doctor::DoctorProduct::current(), now + 1)
            .await
            .expect("Observed handoff outcome");
        assert_eq!(
            records.first().expect("Receipt").outcome,
            diagnostics::doctor::ReceiptOutcome::AwaitingUserAction,
            "Agent assertion is not verification"
        );
        assert_eq!(records.len(), 1);
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        assert!(visual.read(|cx| doctor_history_session_available(&project, &session, cx)));
        visual.dispatch_action(crate::OpenDoctorSession {
            session_id: session.clone(),
        });
        visual.run_until_parked();
        assert_eq!(
            panel.read_with(&visual, |panel, cx| panel
                .active_conversation_view()
                .expect("Reused existing native session")
                .read(cx)
                .root_thread(cx)
                .expect("Thread")
                .read(cx)
                .session_id()
                .to_string()),
            session
        );
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        assert_eq!(
            filesystem
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Preserved unrelated fixture file"),
            "Unrelated worktree edit"
        );
    }

    #[gpui::test]
    async fn doctor_native_repair_desktop_uncommitted_purpose_never_exposes_session_or_submits(
        cx: &mut TestAppContext,
    ) {
        for timeout in [false, true] {
            let (_workspace, panel, view, mut visual, model, history) =
                doctor_native_repair_fixture(cx).await;
            if !view
                .update_in(&mut visual, |view, window, cx| {
                    view.configure_native_repair_fixture("selected test", window, cx)
                })
                .expect("Reviewed native selection")
            {
                assert_eq!(model.completion_count(), 0);
                return;
            }
            model.forbid_requests();
            let previous = panel.read_with(&visual, |panel, _| {
                panel.active_conversation_view().map(Entity::entity_id)
            });
            let store = panel.read_with(&visual, |panel, _| panel.thread_store().clone());
            let (release, blocked) = futures::channel::oneshot::channel::<()>();
            store.update(&mut visual, |store, _| {
                store.set_save_barrier_for_test(Some(
                    async move {
                        blocked.await.expect("Release retained persistence barrier");
                    }
                    .boxed()
                    .shared(),
                ));
            });
            doctor_native_repair_click(&mut visual, "doctor-native-review");
            doctor_native_repair_click(&mut visual, "doctor-approve-repair");
            assert_eq!(model.completion_count(), 0);
            assert_eq!(
                panel.read_with(&visual, |panel, _| {
                    panel.active_conversation_view().map(Entity::entity_id)
                }),
                previous,
                "Pending source-purpose persistence never replaces the visible session"
            );
            visual.refresh().expect("Pending Doctor view");
            assert!(visual.debug_bounds("doctor-native-session").is_none());
            if timeout {
                visual.executor().advance_clock(Duration::from_secs(31));
            } else {
                doctor_native_repair_click(&mut visual, "doctor-cancel-repair");
                visual.executor().advance_clock(Duration::from_millis(200));
            }
            visual.run_until_parked();
            visual
                .condition(&view, |view, _| !view.operation_busy_fixture())
                .await;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Clock")
                .as_secs();
            let records = history
                .records(diagnostics::doctor::DoctorProduct::current(), now)
                .await
                .expect("Observed admission failure receipt");
            let record = records.last().expect("Approved action remains recorded");
            assert!(record.session_id.is_none());
            assert!(matches!(
                record.outcome,
                diagnostics::doctor::ReceiptOutcome::Cancelled
                    | diagnostics::doctor::ReceiptOutcome::Failed
                    | diagnostics::doctor::ReceiptOutcome::Incomplete
            ));
            release.send(()).expect("Release cancelled persistence");
            store.update(&mut visual, |store, _| {
                store.set_save_barrier_for_test(None)
            });
            visual.run_until_parked();
            visual.refresh().expect("Final Doctor view");
            assert!(visual.debug_bounds("doctor-native-session").is_none());
            assert_eq!(model.completion_count(), 0, "No late request or replay");
            assert_eq!(
                panel.read_with(&visual, |panel, _| {
                    panel.active_conversation_view().map(Entity::entity_id)
                }),
                previous
            );
        }
    }

    #[gpui::test]
    async fn doctor_native_repair_desktop_cancel_and_provider_revocation_use_existing_thread(
        cx: &mut TestAppContext,
    ) {
        for revoke in [false, true] {
            let (_workspace, panel, view, mut visual, model, history) =
                doctor_native_repair_fixture(cx).await;
            let enabled = view
                .update_in(&mut visual, |view, window, cx| {
                    view.configure_native_repair_fixture("selected test", window, cx)
                })
                .expect("Fixture selection");
            if !enabled {
                assert_eq!(model.completion_count(), 0);
                return;
            }
            doctor_native_repair_click(&mut visual, "doctor-native-review");
            doctor_native_repair_click(&mut visual, "doctor-approve-repair");
            assert_eq!(model.pending_completions().len(), 1);
            if revoke {
                visual.update(|_, cx| {
                    language_model::LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                        registry.unregister_provider(
                            language_model::LanguageModelProviderId::from("fake".to_owned()),
                            cx,
                        )
                    })
                });
                visual.executor().advance_clock(Duration::from_millis(200));
                visual.run_until_parked();
            } else {
                assert!(
                    view.read_with(&visual, |view, cx| workspace::item::Item::restart_blocker(
                        view, cx
                    )
                    .is_some()),
                    "Pending native turn retains Doctor lifecycle ownership"
                );
                view.update(&mut visual, |_, cx| cx.notify());
                doctor_native_repair_click(&mut visual, "doctor-cancel-repair");
                visual.executor().advance_clock(Duration::from_millis(200));
                visual.run_until_parked();
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Clock")
                .as_secs();
            let records = history
                .records(diagnostics::doctor::DoctorProduct::current(), now)
                .await
                .expect("Cancelled receipt");
            assert_eq!(
                records.len(),
                if revoke { 2 } else { 1 },
                "Previous independently approved history is preserved"
            );
            assert_eq!(
                records.last().expect("Outcome").outcome,
                if revoke {
                    diagnostics::doctor::ReceiptOutcome::Incomplete
                } else {
                    diagnostics::doctor::ReceiptOutcome::Cancelled
                }
            );
            assert!(
                records
                    .last()
                    .expect("Session provenance retained")
                    .session_id
                    .is_some()
            );
            assert_eq!(
                panel.read_with(&visual, |panel, cx| panel
                    .active_conversation_view()
                    .expect("Existing cancelled session")
                    .read(cx)
                    .root_thread(cx)
                    .expect("Thread")
                    .read(cx)
                    .status()),
                ThreadStatus::Idle
            );
        }
    }
    #[gpui::test]
    async fn doctor_native_repair_desktop_denied_source_edit_preserves_permissions_and_files(
        cx: &mut TestAppContext,
    ) {
        use settings::Settings as _;
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        visual.update(|_, cx| {
            let mut settings = agent_settings::AgentSettings::get_global(cx).clone();
            settings.tool_permissions.default = settings::ToolPermissionMode::Deny;
            settings.tool_permissions.tools.remove("edit_file");
            agent_settings::AgentSettings::override_global(settings, cx);
        });
        if !view
            .update_in(&mut visual, |view, window, cx| {
                view.configure_native_repair_fixture("review selected tests", window, cx)
            })
            .expect("Selected evidence")
        {
            return;
        }
        doctor_native_repair_click(&mut visual, "doctor-native-review");
        assert_eq!(model.completion_count(), 0);
        doctor_native_repair_click(&mut visual, "doctor-approve-repair");
        let first = model
            .pending_completions()
            .first()
            .expect("Approved request")
            .clone();
        let input = json!({"path": "project/file.txt", "edits": [{"old_text": "Unrelated worktree edit", "new_text": "DOCTOR_DENIED_EDIT"}]});
        model.send_completion_stream_event(
            &first,
            language_model::LanguageModelCompletionEvent::ToolUse(
                language_model::LanguageModelToolUse {
                    id: "doctor-denied-edit".into(),
                    name: "edit_file".into(),
                    raw_input: input.to_string(),
                    input: language_model::LanguageModelToolUseInput::Json(input),
                    is_input_complete: true,
                    thought_signature: None,
                },
            ),
        );
        model.end_completion_stream(&first);
        visual.run_until_parked();
        let continuation = model
            .pending_completions()
            .first()
            .expect("Permission failure returned to native session")
            .clone();
        assert!(
            continuation.messages.iter().any(|message| message
                .string_contents()
                .contains("Blocked by global default: deny")),
            "Existing permission owner must report its configured refusal: {}",
            continuation
                .messages
                .last()
                .expect("Tool outcome")
                .string_contents()
                .chars()
                .take(1024)
                .collect::<String>()
        );
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        assert!(
            project.read_with(&visual, |project, _| project
                .local_terminal_handles()
                .is_empty()),
            "Doctor handoff never grants terminal permission"
        );
        let filesystem = project.read_with(&visual, |project, _| project.fs().clone());
        assert_eq!(
            filesystem
                .load(Path::new("/project/file.txt"))
                .await
                .expect("Preserved source"),
            "Unrelated worktree edit"
        );
        model.send_completion_stream_event(
            &continuation,
            language_model::LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
        );
        model.end_completion_stream(&continuation);
        visual.run_until_parked();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        assert_eq!(
            history
                .records(diagnostics::doctor::DoctorProduct::current(), now)
                .await
                .expect("Observed history")
                .first()
                .expect("Receipt")
                .outcome,
            diagnostics::doctor::ReceiptOutcome::AwaitingUserAction
        );
        assert_eq!(
            panel.read_with(&visual, |panel, cx| panel
                .active_conversation_view()
                .expect("Normal session")
                .read(cx)
                .root_thread(cx)
                .expect("Thread")
                .read(cx)
                .status()),
            ThreadStatus::Idle
        );
    }

    #[gpui::test]
    async fn doctor_proposal_desktop_ambient_allow_cannot_execute_and_source_edits_need_review(
        cx: &mut TestAppContext,
    ) {
        use settings::Settings as _;
        let (workspace, panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        visual.update(|_, cx| {
            let mut settings = agent_settings::AgentSettings::get_global(cx).clone();
            settings.tool_permissions.default = settings::ToolPermissionMode::Allow;
            settings.tool_permissions.tools.clear();
            agent_settings::AgentSettings::override_global(settings, cx);
        });
        if !view
            .update_in(&mut visual, |view, window, cx| {
                view.configure_native_repair_fixture("selected functional test", window, cx)
            })
            .expect("Selected source proposal")
        {
            return;
        }
        doctor_native_repair_click(&mut visual, "doctor-native-review");
        doctor_native_repair_click(&mut visual, "doctor-approve-repair");
        let thread = panel.read_with(&visual, |panel, cx| {
            panel
                .active_conversation_view()
                .expect("Native conversation")
                .read(cx)
                .root_thread(cx)
                .expect("Existing AcpThread")
        });
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        for (name, input) in [
            (
                "terminal",
                json!({"command": "echo DOCTOR_MUST_NOT_EXECUTE", "cd": "/project", "timeout_ms": 1000}),
            ),
            ("fetch", json!({"url": "https://example.invalid/private"})),
            ("spawn_agent", json!({"task": "Collect a heap dump"})),
            (
                "propose_recipe",
                json!({"prompt": "Enable an unattended schedule"}),
            ),
        ] {
            let request = model
                .pending_completions()
                .first()
                .expect("Active native turn")
                .clone();
            model.send_completion_stream_event(
                &request,
                language_model::LanguageModelCompletionEvent::ToolUse(
                    language_model::LanguageModelToolUse {
                        id: format!("doctor-forbidden-{name}").into(),
                        name: name.into(),
                        raw_input: input.to_string(),
                        input: language_model::LanguageModelToolUseInput::Json(input),
                        is_input_complete: true,
                        thought_signature: None,
                    },
                ),
            );
            model.end_completion_stream(&request);
            visual.run_until_parked();
            let continuation = model
                .pending_completions()
                .first()
                .expect("Explicit tool refusal")
                .clone();
            assert!(continuation.messages.iter().any(|message| {
                message
                    .string_contents()
                    .contains("unavailable in a Doctor source-proposal")
            }));
            assert!(!thread.read_with(&visual, |thread, _| thread.has_in_progress_tool_calls()));
            assert!(project.read_with(&visual, |project, _| {
                project.local_terminal_handles().is_empty()
            }));
        }
        let request = model
            .pending_completions()
            .first()
            .expect("Source proposal turn")
            .clone();
        let input = json!({"path": "project/file.txt", "edits": [{"old_text": "Unrelated worktree edit", "new_text": "Reviewed Doctor source proposal"}]});
        model.send_completion_stream_event(
            &request,
            language_model::LanguageModelCompletionEvent::ToolUse(
                language_model::LanguageModelToolUse {
                    id: "doctor-reviewed-source-edit".into(),
                    name: "edit_file".into(),
                    raw_input: input.to_string(),
                    input: language_model::LanguageModelToolUseInput::Json(input),
                    is_input_complete: true,
                    thought_signature: None,
                },
            ),
        );
        model.end_completion_stream(&request);
        visual.run_until_parked();
        let approval = thread.read_with(&visual, |thread, _| {
            thread
                .entries()
                .iter()
                .find_map(|entry| match entry {
                    acp_thread::AgentThreadEntry::ToolCall(call) => match &call.status {
                        acp_thread::ToolCallStatus::WaitingForConfirmation { options, .. } => {
                            assert!(
                                options
                                    .first_option_of_kind(acp::PermissionOptionKind::AllowAlways)
                                    .is_none()
                            );
                            assert!(
                                options
                                    .first_option_of_kind(acp::PermissionOptionKind::AllowOnce)
                                    .is_some()
                            );
                            Some(call.id.clone())
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .expect("Ambient allow cannot approve a source edit")
        });
        let buffer = project
            .update(&mut visual, |project, cx| {
                project.open_local_buffer("/project/file.txt", cx)
            })
            .await
            .expect("Existing source buffer");
        assert_eq!(
            buffer.read_with(&visual, |buffer, _| buffer.text()),
            "Unrelated worktree edit"
        );
        thread.update(&mut visual, |thread, cx| {
            thread.authorize_tool_call(
                approval,
                acp_thread::SelectedPermissionOutcome::new(
                    "allow".into(),
                    acp::PermissionOptionKind::AllowOnce,
                ),
                cx,
            )
        });
        visual.run_until_parked();
        assert_eq!(
            buffer.read_with(&visual, |buffer, _| buffer.text()),
            "Reviewed Doctor source proposal\n"
        );
        let continuation = model
            .pending_completions()
            .first()
            .expect("Reviewed source edit completion")
            .clone();
        model.send_completion_stream_event(
            &continuation,
            language_model::LanguageModelCompletionEvent::Stop(language_model::StopReason::EndTurn),
        );
        model.end_completion_stream(&continuation);
        visual.run_until_parked();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        let records = history
            .records(diagnostics::doctor::DoctorProduct::current(), now)
            .await
            .expect("Native receipt");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records.first().expect("Unverified repair proposal").outcome,
            diagnostics::doctor::ReceiptOutcome::AwaitingUserAction
        );
    }

    #[gpui::test]
    async fn doctor_native_repair_desktop_review_cancellation_and_stale_inputs_never_submit(
        cx: &mut TestAppContext,
    ) {
        let (_workspace, _panel, view, mut visual, model, history) =
            doctor_native_repair_fixture(cx).await;
        model.forbid_requests();
        if !view
            .update_in(&mut visual, |view, window, cx| {
                view.configure_native_repair_fixture("selected test", window, cx)
            })
            .expect("Selected evidence")
        {
            return;
        }
        doctor_native_repair_click(&mut visual, "doctor-native-review");
        doctor_native_repair_click(&mut visual, "doctor-cancel-repair");
        assert_eq!(model.completion_count(), 0);
        doctor_native_repair_click(&mut visual, "doctor-native-review");
        visual.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| settings.vim_mode = Some(true))
            })
        });
        doctor_native_repair_click(&mut visual, "doctor-approve-repair");
        assert_eq!(
            model.completion_count(),
            0,
            "Changed reviewed settings must invalidate approval"
        );
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Clock")
            .as_secs();
        assert!(
            history
                .records(diagnostics::doctor::DoctorProduct::current(), now)
                .await
                .expect("No effects were admitted")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn doctor_command_full_native_workspace_flow_reuses_view_without_collection(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, mut visual) = setup_workspace_panel(cx).await;
        visual.update(|_, cx| {
            feature_flags::FeatureFlagStore::init(cx);
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |content| {
                    content
                        .feature_flags
                        .get_or_insert_default()
                        .insert("copper-doctor".into(), "on".into());
                });
            });
        });
        visual.run_until_parked();
        panel.update_in(&mut visual, |panel, window, cx| {
            panel.external_thread(
                Some(Agent::NativeAgent),
                None,
                None,
                None,
                None,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        visual.run_until_parked();
        let conversation = panel.read_with(&visual, |panel, _| {
            panel
                .active_conversation_view()
                .expect("Native conversation")
                .clone()
        });
        let (connection, thread) = conversation.read_with(&visual, |conversation, cx| {
            let connection = conversation
                .as_native_connection(cx)
                .expect("Existing actual NativeAgentConnection");
            let thread = conversation
                .active_thread()
                .expect("Actual ACP thread")
                .read(cx)
                .thread
                .clone();
            (connection, thread)
        });
        let session = thread.read_with(&visual, |thread, _| thread.session_id().clone());
        let model =
            std::sync::Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        model.forbid_requests();
        let native = conversation.read_with(&visual, |conversation, cx| {
            conversation
                .as_native_thread(cx)
                .expect("Native session owner")
        });
        native.update(&mut visual, |thread, cx| {
            thread.set_model(model.clone(), cx)
        });
        workspace.update_in(&mut visual, |workspace, window, cx| {
            workspace.focus_panel::<AgentPanel>(window, cx);
        });
        panel.update_in(&mut visual, |panel, window, cx| {
            panel.toggle_zoom(&ToggleZoom, window, cx);
        });
        visual.run_until_parked();
        assert!(panel.read_with(&visual, |panel, _| panel.zoomed));
        assert!(workspace.read_with(&visual, |workspace, _| workspace.zoomed_item().is_some()));
        let mut observed = None;
        for _ in 0..2 {
            let command = thread.update(&mut visual, |thread, cx| {
                thread.send_command(vec!["/doctor".into()], cx)
            });
            let command = visual.foreground_executor().spawn(command);
            visual.run_until_parked();
            for _ in 0..2 {
                visual.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    window.simulate_next_frame(cx);
                });
                visual.run_until_parked();
            }
            command
                .await
                .expect("Native desktop command")
                .expect("Local response");
            visual
                .refresh()
                .expect("Paint the revealed editor after command completion");
            visual.run_until_parked();
            assert!(
                workspace.read_with(&visual, |workspace, _| workspace.zoomed_item().is_none()),
                "Doctor must replace the zoomed overlay"
            );
            let view = workspace.read_with(&visual, |workspace, cx| {
                workspace
                    .item_of_type::<diagnostics::doctor_view::DoctorView>(cx)
                    .expect("Existing Doctor Item opened")
            });
            if let Some(previous) = observed {
                assert_eq!(view.entity_id(), previous);
            }
            observed = Some(view.entity_id());
            assert_eq!(
                workspace.read_with(&visual, |workspace, cx| workspace
                    .active_item(cx)
                    .map(|item| item.item_id())),
                Some(view.entity_id()),
                "Doctor is the active center item"
            );
            view.update(&mut visual, |_, cx| cx.notify());
            visual.refresh().expect("Paint Doctor controls");
            visual.run_until_parked();
            assert!(
                visual.debug_bounds("made-doctor").is_some(),
                "Actual native Doctor surface"
            );
            assert!(workspace.read_with(&visual, |workspace, _| workspace.zoomed_item().is_none()));
            visual.update(|window, cx| {
                assert!(view.focus_handle(cx).contains_focused(window, cx));
            });
            assert_eq!(model.completion_count(), 0);
        }
        assert_eq!(
            thread.read_with(&visual, |thread, _| thread.session_id().clone()),
            session
        );
        assert_eq!(
            panel.read_with(&visual, |panel, _| panel
                ._doctor_subscription
                .as_ref()
                .expect("Connection-owned Doctor subscription")
                .0),
            connection.0.entity_id()
        );
        assert!(
            visual.debug_bounds("doctor-refresh").is_some(),
            "Collection remains explicit"
        );
        #[cfg(feature = "doctor-dependency-test")]
        {
            let view = workspace
                .read_with(&visual, |workspace, cx| {
                    workspace.item_of_type::<diagnostics::doctor_view::DoctorView>(cx)
                })
                .expect("Reused Doctor");
            let collect = visual
                .debug_bounds("doctor-refresh")
                .expect("Actual collection button");
            visual.simulate_click(collect.center(), gpui::Modifiers::default());
            visual
                .condition(&view, |view, _| view.report_fixture().is_some())
                .await;
            visual.run_until_parked();
            let history = native.read_with(&visual, |thread, _| thread.to_markdown());
            assert!(history.contains("/doctor collection"));
            assert!(history.contains("collection status: Collected"));
            assert!(history.contains("unperformed application checks remain Not run"));
            visual.update(|_, cx| {
                SettingsStore::update_global(cx, |store, cx| {
                    store.update_user_settings(cx, |settings| settings.vim_mode = Some(true))
                });
            });
            visual.run_until_parked();
            let history = native.read_with(&visual, |thread, _| thread.to_markdown());
            assert!(
                history.contains("collection status: Stale"),
                "The originating conversation must not retain only a current-looking status after settings change"
            );
            assert_eq!(
                model.completion_count(),
                0,
                "Report context does not send a model request"
            );
        }
    }

    #[gpui::test]
    async fn doctor_history_session_reuses_existing_native_thread_and_rejects_missing_metadata(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, mut visual) = setup_workspace_panel(cx).await;
        visual.run_until_parked();
        let connection = DisassociationTrackingConnection::new();
        panel.update(&mut visual, |panel, cx| {
            panel.connection_store.update(cx, |store, cx| {
                store.restart_connection(
                    Agent::NativeAgent,
                    Rc::new(StubAgentServer::new(connection.clone())),
                    cx,
                );
            });
        });
        visual.run_until_parked();
        panel.update_in(&mut visual, |panel, window, cx| {
            panel.external_thread(
                Some(Agent::NativeAgent),
                None,
                None,
                None,
                None,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        visual.run_until_parked();
        send_message(&panel, &mut visual);
        let session = active_session_id(&panel, &visual);
        let thread = active_thread_id(&panel, &visual);
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        assert!(visual.read(|cx| doctor_history_session_available(
            &project,
            session.0.as_ref(),
            cx
        )));
        let created = *connection.next_session_number.lock();
        for _ in 0..2 {
            visual.dispatch_action(crate::OpenDoctorSession {
                session_id: session.0.to_string(),
            });
            visual.run_until_parked();
        }
        assert_eq!(active_session_id(&panel, &visual), session);
        assert_eq!(active_thread_id(&panel, &visual), thread);
        assert_eq!(
            *connection.next_session_number.lock(),
            created,
            "Navigation must not start another session"
        );
        visual.update(|_, cx| {
            ThreadMetadataStore::global(cx).update(cx, |store, cx| store.delete(thread, cx))
        });
        visual.dispatch_action(crate::OpenDoctorSession {
            session_id: session.0.to_string(),
        });
        visual.run_until_parked();
        assert_eq!(*connection.next_session_number.lock(), created);
        assert_eq!(active_thread_id(&panel, &visual), thread);
        assert!(!visual.read(|cx| doctor_history_session_available(
            &project,
            session.0.as_ref(),
            cx
        )));
    }

    #[gpui::test]
    async fn doctor_history_session_rejects_other_project_external_agent_and_invalid_id(
        cx: &mut TestAppContext,
    ) {
        let (workspace, _panel, mut visual) = setup_workspace_panel(cx).await;
        visual.run_until_parked();
        let project = workspace.read_with(&visual, |workspace, _| workspace.project().clone());
        let session = acp::SessionId::new("doctor-native-metadata");
        let metadata = crate::thread_metadata_store::ThreadMetadata {
            thread_id: ThreadId::new(),
            session_id: Some(session.clone()),
            agent_id: agent::ZED_AGENT_ID.clone(),
            title: Some("Existing reviewed repair".into()),
            title_override: None,
            updated_at: Utc::now(),
            created_at: Some(Utc::now()),
            interacted_at: None,
            worktree_paths: WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from(
                "/project",
            )])),
            remote_connection: None,
            archived: false,
            execution_outcome: acp_thread::ThreadExecutionOutcome::Unknown,
        };
        let save = |metadata, visual: &mut VisualTestContext| {
            visual.update(|_, cx| {
                ThreadMetadataStore::global(cx).update(cx, |store, cx| store.save(metadata, cx))
            });
        };
        save(metadata.clone(), &mut visual);
        assert!(visual.read(|cx| doctor_history_session_available(
            &project,
            session.0.as_ref(),
            cx
        )));
        let mut other = metadata.clone();
        other.worktree_paths =
            WorktreePaths::from_folder_paths(&PathList::new(&[PathBuf::from("/other-project")]));
        save(other, &mut visual);
        assert!(!visual.read(|cx| doctor_history_session_available(
            &project,
            session.0.as_ref(),
            cx
        )));
        let mut external = metadata;
        external.agent_id = "External".into();
        save(external, &mut visual);
        assert!(!visual.read(|cx| doctor_history_session_available(
            &project,
            session.0.as_ref(),
            cx
        )));
        for invalid in ["", "unknown-session", "bad\nsession"] {
            assert!(!visual.read(|cx| doctor_history_session_available(&project, invalid, cx)));
            visual.dispatch_action(crate::OpenDoctorSession {
                session_id: invalid.into(),
            });
            visual.run_until_parked();
        }
    }

    /// Reproduces the retained-thread reset race:
    ///
    /// 1. Thread A is active and Connected.
    /// 2. User switches to thread B → A goes to retained_threads.
    /// 3. A thread_error is set on retained A's thread view.
    /// 4. AgentServersUpdated fires → retained A's handle_agent_servers_updated
    ///    sees has_thread_error=true → calls reset() → close_all_sessions →
    ///    session X removed, state = Loading.
    /// 5. User reopens thread X via open_thread → load_agent_thread checks
    ///    retained A's has_session → returns false (state is Loading) →
    ///    creates new ConversationView C.
    /// 6. Both A's reload task and C's load task complete → both call
    ///    load_session(X) → both get Connected with session X.
    /// 7. A is eventually cleaned up → on_release → close_all_sessions →
    ///    removes session X.
    /// 8. C sends → "Session not found".
    #[gpui::test]
    async fn test_retained_thread_reset_race_disassociates_session(cx: &mut TestAppContext) {
        let (_workspace, panel, mut cx) = setup_workspace_panel(cx).await;
        cx.run_until_parked();

        let connection = DisassociationTrackingConnection::new();
        panel.update(&mut cx, |panel, cx| {
            panel.connection_store.update(cx, |store, cx| {
                store.restart_connection(
                    Agent::Stub,
                    Rc::new(StubAgentServer::new(connection.clone())),
                    cx,
                );
            });
        });
        cx.run_until_parked();

        // Step 1: Open thread A and send a message.
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.external_thread(
                Some(Agent::Stub),
                None,
                None,
                None,
                None,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        send_message(&panel, &mut cx);

        let session_id_a = active_session_id(&panel, &cx);
        let _thread_id_a = active_thread_id(&panel, &cx);

        // Step 2: Open thread B → A goes to retained_threads.
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.external_thread(
                Some(Agent::Stub),
                None,
                None,
                None,
                None,
                true,
                AgentThreadSource::AgentPanel,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        send_message(&panel, &mut cx);

        // Confirm A is retained.
        panel.read_with(&cx, |panel, _cx| {
            assert!(
                panel.retained_threads.contains_key(&_thread_id_a),
                "thread A should be in retained_threads after switching to B"
            );
        });

        // Step 3: Set a thread_error on retained A's active thread view.
        // This simulates an API error that occurred before the user switched
        // away, or a transient failure.
        let retained_conversation_a = panel.read_with(&cx, |panel, _cx| {
            panel
                .retained_threads
                .get(&_thread_id_a)
                .expect("thread A should be retained")
                .clone()
        });
        retained_conversation_a.update(&mut cx, |conversation, cx| {
            if let Some(thread_view) = conversation.active_thread() {
                thread_view.update(cx, |view, cx| {
                    view.handle_thread_error(
                        crate::conversation_view::ThreadError::Other {
                            message: "simulated error".into(),
                            acp_error_code: None,
                        },
                        cx,
                    );
                });
            }
        });

        // Confirm the thread error is set.
        retained_conversation_a.read_with(&cx, |conversation, cx| {
            let connected = conversation.as_connected().expect("should be connected");
            assert!(
                connected.has_thread_error(cx),
                "retained A should have a thread error"
            );
        });

        // Step 4: Emit AgentServersUpdated → retained A's
        // handle_agent_servers_updated sees has_thread_error=true,
        // calls reset(), which closes session X and sets state=Loading.
        //
        // Critically, we do NOT call run_until_parked between the emit
        // and open_thread. The emit's synchronous effects (event delivery
        // → reset() → close_all_sessions → state=Loading) happen during
        // the update's flush_effects. But the async reload task spawned
        // by initial_state has NOT been polled yet.
        panel.update(&mut cx, |panel, cx| {
            panel.project.update(cx, |project, cx| {
                project
                    .agent_server_store()
                    .update(cx, |_store, cx| cx.emit(project::AgentServersUpdated));
            });
        });
        // After this update returns, the retained ConversationView is in
        // Loading state (reset ran synchronously), but its async reload
        // task hasn't executed yet.

        // Step 5: Immediately open thread X via open_thread, BEFORE
        // the retained view's async reload completes. load_agent_thread
        // checks retained A's has_session → returns false (state is
        // Loading) → creates a NEW ConversationView C for session X.
        panel.update_in(&mut cx, |panel, window, cx| {
            panel.open_thread(session_id_a.clone(), None, None, window, cx);
        });

        // NOW settle everything: both async tasks (A's reload and C's load)
        // complete, both register session X.
        cx.run_until_parked();

        // Verify session A is the active session via C.
        panel.read_with(&cx, |panel, cx| {
            let active_session = panel
                .active_agent_thread(cx)
                .map(|t| t.read(cx).session_id().clone());
            assert_eq!(
                active_session,
                Some(session_id_a.clone()),
                "session A should be the active session after open_thread"
            );
        });

        // Step 6: Force the retained ConversationView A to be dropped
        // while the active view (C) still has the same session.
        // We can't use remove_thread because C shares the same ThreadId
        // and remove_thread would kill the active view too. Instead,
        // directly remove from retained_threads and drop the handle
        // so on_release → close_all_sessions fires only on A.
        drop(retained_conversation_a);
        panel.update(&mut cx, |panel, _cx| {
            panel.retained_threads.remove(&_thread_id_a);
        });
        cx.run_until_parked();

        // The key assertion: sending messages on the ACTIVE view (C)
        // must succeed. If the session was disassociated by A's cleanup,
        // this will fail with "Session not found".
        send_message(&panel, &mut cx);
        send_message(&panel, &mut cx);

        let missing = connection.missing_prompt_sessions.lock().clone();
        assert!(
            missing.is_empty(),
            "session should not be disassociated after retained thread reset race, \
             got missing prompt sessions: {:?}",
            missing
        );

        panel.read_with(&cx, |panel, cx| {
            let active_view = panel
                .active_conversation_view()
                .expect("conversation should remain open");
            let connected = active_view
                .read(cx)
                .as_connected()
                .expect("conversation should be connected");
            assert!(
                !connected.has_thread_error(cx),
                "conversation should not have a thread error"
            );
        });
    }

    #[gpui::test]
    async fn test_initialize_from_source_transfers_draft_to_fresh_panel(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        fs.insert_tree("/project_b", json!({ "file.txt": "" }))
            .await;
        let project_a = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;
        let project_b = Project::test(fs.clone(), [Path::new("/project_b")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));

        let workspace_a = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        // Set up panel_a with an active thread and type draft text.
        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        cx.run_until_parked();

        panel_a.update_in(cx, |panel, window, cx| {
            panel.open_external_thread_with_server(
                Rc::new(StubAgentServer::default_response()),
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let thread_view_a =
            panel_a.read_with(cx, |panel, cx| panel.active_thread_view(cx).unwrap());
        let editor_a = thread_view_a.read_with(cx, |view, _cx| view.message_editor.clone());
        editor_a.update_in(cx, |editor, window, cx| {
            editor.set_text("Draft from workspace A", window, cx);
        });

        // Set up panel_b on workspace_b — starts as a fresh, empty panel.
        let panel_b = workspace_b.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        cx.run_until_parked();

        // Initializing panel_b from workspace_a should transfer the draft,
        // even if panel_b already has an auto-created empty draft thread
        // (which set_active creates during add_panel).
        let transferred = panel_b.update_in(cx, |panel, window, cx| {
            panel.initialize_from_source_workspace_if_needed(workspace_a.downgrade(), window, cx)
        });
        assert!(
            transferred,
            "fresh destination panel should accept source content"
        );

        // Verify the panel was initialized: the base_view should now be an
        // AgentThread (not Uninitialized) and a draft_thread should be set.
        // We can't check the message editor text directly because the thread
        // needs a connected server session (not available in unit tests without
        // a stub server). The `transferred == true` return already proves that
        // source_panel_initialization read the content successfully.
        panel_b.read_with(cx, |panel, _cx| {
            assert!(
                panel.active_conversation_view().is_some(),
                "panel_b should have a conversation view after initialization"
            );
            assert!(
                panel.draft_thread.is_some(),
                "panel_b should have a draft_thread set after initialization"
            );
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn test_source_transfer_preserves_terminal_only_destination(cx: &mut TestAppContext) {
        let (panel, mut cx) = setup_panel(cx).await;
        let workspace = panel.read_with(&cx, |panel, _| {
            panel.workspace.upgrade().expect("workspace")
        });
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
        });
        crate::test_support::open_draft_with_connection(
            &panel,
            StubAgentConnection::new(),
            &mut cx,
        );
        workspace.update_in(&mut cx, |workspace, window, cx| {
            workspace.focus_panel::<AgentPanel>(window, cx);
        });
        cx.run_until_parked();
        let parent = panel.read_with(&cx, |panel, cx| {
            panel.active_thread_view(cx).expect("parent")
        });
        let editor = parent.read_with(&cx, |parent, _| parent.message_editor.clone());
        let conversation = panel.read_with(&cx, |panel, _| {
            panel
                .active_conversation_view()
                .expect("conversation")
                .clone()
        });
        let workspace = panel.read_with(&cx, |panel, _| {
            panel.workspace.upgrade().expect("workspace")
        });
        let terminal_id = TerminalId::new();
        let position =
            crate::terminal_thread_metadata_store::InlineTerminalDraftPosition::new("", 0)
                .expect("empty source boundary");
        let (terminal, block_id) = cx.update(|window, cx| {
            TerminalThreadMetadataStore::init_global(cx);
            let history = TerminalThreadMetadataStore::global(cx);
            let mut context = parent
                .read(cx)
                .inline_terminal_context(cx)
                .expect("canonical scope");
            context.draft_position = Some(position.clone());
            context.output = "retained terminal-only output".into();
            history
                .update(cx, |history, cx| {
                    history.save_inline(
                        TerminalThreadMetadata {
                            terminal_id,
                            title: "Retained terminal".into(),
                            custom_title: None,
                            created_at: Utc::now(),
                            worktree_paths: WorktreePaths::from_folder_paths(
                                &workspace::PathList::new(&[PathBuf::from("/project")]),
                            ),
                            remote_connection: None,
                            working_directory: None,
                            inline_context: Some(context),
                        },
                        cx,
                    )
                })
                .expect("canonical history admission");
            let project = workspace.read(cx).project().clone();
            let settings = TerminalSettings::get_global(cx);
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
                terminal.write_output(b"retained terminal-only output\r\n", cx)
            });
            let view = cx.new(|cx| {
                TerminalView::new(
                    terminal.clone(),
                    workspace.downgrade(),
                    None,
                    project.downgrade(),
                    window,
                    cx,
                )
            });
            let inline = cx.new(|cx| {
                crate::inline_terminal_view::InlineTerminalView::new(
                    terminal_id,
                    None,
                    view.clone(),
                    workspace.downgrade(),
                    history.downgrade(),
                    editor.downgrade(),
                    false,
                    cx,
                )
            });
            let block_id = editor
                .update(cx, |editor, cx| {
                    editor.mount_inline_terminal_view(view, Some(inline.clone()), &position, cx)
                })
                .expect("actual editor character block");
            inline.update(cx, |inline, _| inline.set_block_id(block_id));
            (terminal, block_id)
        });
        assert!(
            editor
                .read_with(&cx, |editor, cx| editor.text(cx))
                .is_empty()
        );
        assert!(
            editor
                .read_with(&cx, |editor, cx| editor.draft_content_blocks_snapshot(cx))
                .expect("valid retained terminal snapshot")
                .iter()
                .any(|block| matches!(block, acp::ContentBlock::Resource(_)))
        );
        for invalid_scope in [false, true] {
            if invalid_scope {
                editor.update(&mut cx, |editor, _| {
                    editor.set_inline_terminal_scope(
                        ThreadId::new(),
                        acp::SessionId::new("foreign-session"),
                    )
                });
                assert!(
                    editor
                        .read_with(&cx, |editor, cx| editor.draft_content_blocks_snapshot(cx))
                        .is_err()
                );
            }
            panel.update_in(&mut cx, |panel, window, cx| {
                assert!(panel.terminals.is_empty());
                assert!(panel.retained_threads.is_empty());
                assert!(parent.read(cx).thread.read(cx).entries().is_empty());
                assert!(
                    parent
                        .read(cx)
                        .thread
                        .read(cx)
                        .draft_prompt()
                        .is_none_or(|prompt| prompt.is_empty())
                );
                assert!(
                    panel.destination_has_meaningful_state(cx),
                    "terminal-only input is meaningful even when its owner snapshot is unavailable"
                );
                let agent = panel.selected_agent.clone();
                assert!(!panel.initialize_from_source_workspace_if_needed(
                    workspace.downgrade(),
                    window,
                    cx
                ));
                assert_eq!(panel.selected_agent, agent);
                assert_eq!(
                    panel
                        .active_conversation_view()
                        .expect("preserved conversation")
                        .entity_id(),
                    conversation.entity_id()
                );
                assert_eq!(
                    parent.read(cx).message_editor.entity_id(),
                    editor.entity_id()
                );
                assert_eq!(
                    editor
                        .read(cx)
                        .inline_terminal_position(block_id, cx)
                        .expect("preserved boundary"),
                    position
                );
            });
        }
        assert!(
            !terminal.read_with(&cx, |terminal, _| terminal.is_pty()),
            "retained history guard cannot allocate a replacement process"
        );
    }

    #[gpui::test]
    async fn test_initialize_from_source_inherits_agent_without_draft_content(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        fs.insert_tree("/project_b", json!({ "file.txt": "" }))
            .await;
        let project_a = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;
        let project_b = Project::test(fs.clone(), [Path::new("/project_b")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));

        let workspace_a = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        panel_a.update(cx, |panel, _cx| {
            panel.selected_agent = Agent::Stub;
        });

        let panel_b = workspace_b.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        let initialized = panel_b.update_in(cx, |panel, window, cx| {
            panel.initialize_from_source_workspace_if_needed(workspace_a.downgrade(), window, cx)
        });
        assert!(
            initialized,
            "fresh destination panel should inherit the source agent"
        );

        panel_b.read_with(cx, |panel, _cx| {
            assert_eq!(
                panel.selected_agent,
                Agent::Stub,
                "destination panel should inherit the source panel's selected agent"
            );
            assert!(
                panel.active_conversation_view().is_none(),
                "agent-only initialization should not create a draft thread"
            );
        });
    }

    #[gpui::test]
    async fn test_initialize_from_source_retargets_empty_destination_draft_agent(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        fs.insert_tree("/project_b", json!({ "file.txt": "" }))
            .await;
        let project_a = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;
        let project_b = Project::test(fs.clone(), [Path::new("/project_b")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));

        let workspace_a = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });

        panel_a.update(cx, |panel, _cx| {
            panel.selected_agent = Agent::Stub;
        });

        let panel_b = workspace_b.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        panel_b.update_in(cx, |panel, window, cx| {
            panel.activate_new_thread(false, AgentThreadSource::AgentPanel, window, cx);
        });

        let original_draft = panel_b.read_with(cx, |panel, cx| {
            let draft = panel.draft_thread.as_ref().expect("draft should exist");
            assert_eq!(
                *draft.read(cx).agent_key(),
                Agent::NativeAgent,
                "destination draft should start on the default agent"
            );
            draft.entity_id()
        });

        let initialized = panel_b.update_in(cx, |panel, window, cx| {
            panel.initialize_from_source_workspace_if_needed(workspace_a.downgrade(), window, cx)
        });
        assert!(
            initialized,
            "fresh destination draft should inherit the source agent"
        );

        panel_b.read_with(cx, |panel, cx| {
            let draft = panel.draft_thread.as_ref().expect("draft should exist");
            assert_ne!(
                draft.entity_id(),
                original_draft,
                "empty destination draft should be replaced when the inherited agent differs"
            );
            assert_eq!(
                *draft.read(cx).agent_key(),
                Agent::Stub,
                "empty destination draft should be rebound to the inherited agent"
            );
        });
    }

    #[gpui::test]
    async fn test_initialize_from_source_does_not_overwrite_existing_content(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            agent::ThreadStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project_a", json!({ "file.txt": "" }))
            .await;
        fs.insert_tree("/project_b", json!({ "file.txt": "" }))
            .await;
        let project_a = Project::test(fs.clone(), [Path::new("/project_a")], cx).await;
        let project_b = Project::test(fs.clone(), [Path::new("/project_b")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project_a.clone(), window, cx));

        let workspace_a = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        let workspace_b = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b.clone(), window, cx)
            })
            .unwrap();

        let cx = &mut VisualTestContext::from_window(multi_workspace.into(), cx);

        // Set up panel_a with draft text.
        let panel_a = workspace_a.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        cx.run_until_parked();

        panel_a.update_in(cx, |panel, window, cx| {
            panel.open_external_thread_with_server(
                Rc::new(StubAgentServer::default_response()),
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let thread_view_a =
            panel_a.read_with(cx, |panel, cx| panel.active_thread_view(cx).unwrap());
        let editor_a = thread_view_a.read_with(cx, |view, _cx| view.message_editor.clone());
        editor_a.update_in(cx, |editor, window, cx| {
            editor.set_text("Draft from workspace A", window, cx);
        });

        // Set up panel_b with its OWN content — this is a non-fresh panel.
        let panel_b = workspace_b.update_in(cx, |workspace, window, cx| {
            let panel = cx.new(|cx| AgentPanel::new(workspace, window, cx));
            workspace.add_panel(panel.clone(), window, cx);
            panel
        });
        cx.run_until_parked();

        panel_b.update_in(cx, |panel, window, cx| {
            panel.open_external_thread_with_server(
                Rc::new(StubAgentServer::default_response()),
                window,
                cx,
            );
        });
        cx.run_until_parked();

        let thread_view_b =
            panel_b.read_with(cx, |panel, cx| panel.active_thread_view(cx).unwrap());
        let editor_b = thread_view_b.read_with(cx, |view, _cx| view.message_editor.clone());
        editor_b.update_in(cx, |editor, window, cx| {
            editor.set_text("Existing work in workspace B", window, cx);
        });

        // Attempting to initialize panel_b from workspace_a should be rejected
        // because panel_b already has meaningful content.
        let transferred = panel_b.update_in(cx, |panel, window, cx| {
            panel.initialize_from_source_workspace_if_needed(workspace_a.downgrade(), window, cx)
        });
        assert!(
            !transferred,
            "destination panel with existing content should not be overwritten"
        );

        // Verify panel_b still has its original content.
        panel_b.read_with(cx, |panel, cx| {
            let thread_view = panel
                .active_thread_view(cx)
                .expect("panel_b should still have its thread view");
            let text = thread_view.read(cx).message_editor.read(cx).text(cx);
            assert_eq!(
                text, "Existing work in workspace B",
                "destination panel's content should be preserved"
            );
        });
    }

    #[gpui::test]
    async fn test_create_thread_with_options_retains_thread_and_restores_agent(
        cx: &mut TestAppContext,
    ) {
        let (panel, mut cx) = setup_panel(cx).await;
        let _stub_connection =
            crate::test_support::set_stub_agent_connection(StubAgentConnection::new());

        // Baseline: panel's selected_agent is the stub.
        panel.update(&mut cx, |panel, _cx| {
            panel.selected_agent = Agent::Stub;
        });

        // Case 1: no agent override. The new thread should land in
        // `retained_threads` and `selected_agent` should be unchanged.
        let no_override_id = panel.update_in(&mut cx, |panel, window, cx| {
            panel.create_thread_with_options(
                CreateThreadOptions::default(),
                AgentThreadSource::AgentPanel,
                window,
                cx,
            )
        });

        panel.read_with(&cx, |panel, _cx| {
            assert!(
                panel.retained_threads.contains_key(&no_override_id),
                "thread created via create_thread_with_options should be retained"
            );
            assert_eq!(
                panel.selected_agent,
                Agent::Stub,
                "selected_agent should be unchanged when no agent override is requested"
            );
        });

        // Case 2: an explicit agent override that differs from the panel's
        // selection. `create_agent_thread_inner` updates `selected_agent` as a
        // side effect; `create_thread_with_options` must restore it so the
        // user's last-used agent isn't silently flipped by an agent-initiated
        // call.
        let override_agent = Agent::Custom {
            id: "override-agent".into(),
        };
        let override_id = panel.update_in(&mut cx, |panel, window, cx| {
            panel.create_thread_with_options(
                CreateThreadOptions {
                    agent: Some(override_agent.clone()),
                    ..CreateThreadOptions::default()
                },
                AgentThreadSource::AgentPanel,
                window,
                cx,
            )
        });

        panel.read_with(&cx, |panel, _cx| {
            assert!(
                panel.retained_threads.contains_key(&override_id),
                "thread created with an agent override should also be retained"
            );
            assert_ne!(
                no_override_id, override_id,
                "each call should produce a distinct ThreadId"
            );
            assert_eq!(
                panel.selected_agent,
                Agent::Stub,
                "selected_agent should be restored to the original after an agent override"
            );
        });
    }
}
