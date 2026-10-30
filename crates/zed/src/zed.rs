mod app_menus;
pub mod edit_prediction_registry;
#[cfg(target_os = "macos")]
pub(crate) mod mac_only_instance;
mod migrate;
#[cfg(feature = "multiplayer-tools")]
#[path = "migration.rs"]
#[allow(dead_code)]
pub mod migration;
#[cfg(target_os = "macos")]
pub(crate) mod move_to_applications;
mod open_listener;
mod open_url_modal;
mod quick_action_bar;
pub mod remote_debug;
#[cfg(feature = "sim")]
#[path = "sim_plugin_services.rs"]
pub mod sim_plugin_services;
pub mod telemetry_log;
#[cfg(all(target_os = "macos", feature = "visual-tests"))]
pub mod visual_tests;
#[cfg(target_os = "windows")]
pub(crate) mod windows_only_instance;

#[cfg(feature = "agentic-tools")]
use agent_settings::{UserAgentsMdState, init_user_agents_md};
#[cfg(feature = "agentic-tools")]
use agent_ui::AgentDiffToolbar;
#[cfg(feature = "multiplayer-tools")]
use agent_ui::AgentPanel;
#[cfg(feature = "multiplayer-tools")]
use agent_ui::AgentPanelEvent;
use anyhow::Context as _;
pub use app_menus::*;
use assets::Assets;

use breadcrumbs::Breadcrumbs;
#[cfg(feature = "rust-tools")]
use cargo_ui::CargoPanel;
use client::zed_urls;
use collections::VecDeque;
use debugger_ui::debugger_panel::DebugPanel;
use editor::{Editor, MultiBuffer};
use extension_host::ExtensionStore;
use feature_flags::{FeatureFlagAppExt as _, PanicFeatureFlag};
use fs::Fs;
#[cfg(feature = "sim")]
use futures::FutureExt;
use futures::{StreamExt, channel::mpsc, select_biased};
use git_ui::branch_diff::BranchDiffToolbar;
use git_ui::commit_view::CommitViewToolbar;
use git_ui::git_panel::GitPanel;
#[cfg(feature = "multiplayer-tools")]
use git_ui::project_diff::ProjectDiff;
use git_ui::project_diff::ProjectDiffToolbar;
use git_ui::solo_diff_view::{SoloDiffGitToolbar, SoloDiffStyleToolbar};
use git_ui::staged_diff::StagedDiffToolbar;
use git_ui::unstaged_diff::UnstagedDiffToolbar;
#[cfg(feature = "agentic-tools")]
use gpui::AsyncWindowContext;
use gpui::{
    Action, App, AppContext as _, ClipboardItem, Context, DismissEvent, Element, Entity,
    FocusHandle, Focusable, KeyBinding, ParentElement, PathPromptOptions, PromptLevel, ReadGlobal,
    SharedString, Size, Task, TaskExt, TitlebarOptions, UpdateGlobal, WeakEntity, Window,
    WindowBounds, WindowHandle, WindowKind, WindowOptions, actions, image_cache, img, point, px,
    retain_all,
};
#[cfg(feature = "multiplayer-tools")]
use gpui::{EntityId, Subscription};
#[cfg(feature = "jvm-tools")]
use jvm_ui::DependenciesPanel;
use language::Capability;
use language_onboarding::BasedPyrightBanner;
use language_tools::lsp_log_view::LspLogToolbarItemView;
use markdown::{Markdown, MarkdownElement, MarkdownFont, MarkdownStyle};
use migrate::{MigrationBanner, MigrationEvent, MigrationNotification, MigrationType};
use migrator::migrate_keymap;
use onboarding::multibuffer_hint::MultibufferHint;
pub use open_listener::*;
use outline_panel::OutlinePanel;
use paths::{
    local_debug_file_relative_path, local_settings_file_relative_path,
    local_tasks_file_relative_path,
};
use project::{
    DirectoryLister, DisableAiSettings, ProjectItem,
    project_settings::{SettingsObserver, SettingsObserverEvent},
};
use project_panel::ProjectPanel;
use quick_action_bar::QuickActionBar;
use recent_projects::open_remote_project;
use release_channel::{AppCommitSha, AppVersion, ReleaseChannel};
use rope::Rope;
use search::project_search::ProjectSearchBar;
use settings::{
    BaseKeymap, DEFAULT_KEYMAP_PATH, DefaultOpenBehavior, InvalidSettingsError, KeybindSource,
    KeymapFile, KeymapFileLoadResult, MigrationStatus, SPECIFIC_OVERRIDES_KEYMAP_PATH, Settings,
    SettingsFile, SettingsStore, VIM_KEYMAP_PATH, initial_local_debug_tasks_content,
    initial_project_settings_content, initial_tasks_content, update_settings_file,
};
#[cfg(feature = "agentic-tools")]
use sidebar::Sidebar;
#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
use tasks_ui::TestsPanel;
#[cfg(debug_assertions)]
use workspace::workspace_error::{ErrorAction, ErrorSeverity, WorkspaceError};

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::Arc,
    sync::atomic::{self, AtomicBool},
};
#[cfg(feature = "multiplayer-tools")]
use std::{cell::RefCell, rc::Rc};
use terminal_view::terminal_panel::{self, TerminalPanel};
use theme::{ActiveTheme, SystemAppearance, ThemeRegistry, deserialize_icon_theme};
use theme_settings::{ThemeSettings, load_user_theme};
use ui::{Navigable, NavigableEntry, TintColor, prelude::*};
use util::markdown::MarkdownString;
use util::rel_path::RelPath;
use util::{ResultExt, asset_str, maybe};
use uuid::Uuid;
use vim_mode_setting::VimModeSetting;
use workspace::notifications::{NotificationId, dismiss_app_notification, show_app_notification};

#[cfg(feature = "agentic-tools")]
use workspace::Panel;
#[cfg(feature = "multiplayer-tools")]
use workspace::collaborative_composer::CollaborativeComposerRegistration;
#[cfg(feature = "multiplayer-tools")]
use workspace::collaborative_participants::{
    CollaborativeConnectionState, CollaborativeParticipant, CollaborativeParticipantPresence,
    CollaborativeParticipantProvider, CollaborativeParticipantRegistration,
};
#[cfg(all(test, feature = "multiplayer-tools"))]
use workspace::collaborative_participants::{
    CollaborativeParticipantProviderState, CollaborativeParticipantViewData,
};
#[cfg(feature = "multiplayer-tools")]
use workspace::collaborative_review::CollaborativeReviewRegistration;
#[cfg(feature = "multiplayer-tools")]
use workspace::collaborative_timeline::CollaborativeTimelineRegistration;
use workspace::{
    AppState, MultiWorkspace, NewFile, NewWindow, OpenLog, Toast, Workspace, WorkspaceSettings,
    create_and_open_local_file, notifications::simple_message_notification::MessageNotification,
    open_new,
};
use workspace::{
    CloseIntent, CloseProject, CloseWindow, RestoreBanner, with_active_or_new_workspace,
};
use workspace::{Pane, notifications::DetachAndPromptErr};
use zed_actions::{
    About, GetMerch, OpenAccountSettings, OpenBrowser, OpenDocs, OpenProjectTasks,
    OpenServerSettings, OpenSettingsFile, OpenStatusPage, OpenZedUrl, Quit,
};
const DOCS_URL: &str = "https://zed.dev/docs/";
const STATUS_URL: &str = "https://status.zed.dev";
const MERCH_URL: &str = "https://merch.zed.dev/";

pub struct CrashHandler(pub Arc<crashes::Client>);

impl gpui::Global for CrashHandler {}

#[cfg(feature = "multiplayer-tools")]
#[derive(Default)]
struct CollaborativeReviewCompositionState {
    agent_panel_id: Option<EntityId>,
    agent_thread_id: Option<EntityId>,
    agent_registration: Option<CollaborativeReviewRegistration>,
    composer_thread_view_id: Option<EntityId>,
    composer_registration: Option<CollaborativeComposerRegistration>,
    participant_thread_view_id: Option<EntityId>,
    participant_registration: Option<CollaborativeParticipantRegistration>,
    participant_observed_thread_view_id: Option<EntityId>,
    participant_thread_observation: Option<Subscription>,
    participant_active_call_observation: Option<Subscription>,
    participant_room_id: Option<EntityId>,
    participant_room_observation: Option<Subscription>,
    timeline_conversation_view_id: Option<EntityId>,
    timeline_registration: Option<CollaborativeTimelineRegistration>,
    project_diff_id: Option<EntityId>,
    project_registration: Option<CollaborativeReviewRegistration>,
}

#[cfg(feature = "multiplayer-tools")]
struct CollaborativeParticipantProjection {
    thread_view_id: EntityId,
    provider: CollaborativeParticipantProvider,
}

#[cfg(feature = "multiplayer-tools")]
impl CollaborativeParticipantProjection {
    fn from_adapter(
        adapter: agent_ui::collaborative_participants::CollaborativeParticipantAdapter,
    ) -> Self {
        let thread_view_id = adapter.thread_view_id();
        let provider = adapter.into_provider();
        Self {
            thread_view_id,
            provider,
        }
    }
}

#[cfg(feature = "multiplayer-tools")]
fn reconcile_collaborative_composer(
    workspace_handle: &Entity<Workspace>,
    agent_panel: &Entity<AgentPanel>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut App,
) {
    let thread_view_id = agent_panel
        .read(cx)
        .active_conversation_view()
        .and_then(|conversation| conversation.read(cx).collaborative_active_thread().cloned())
        .map(|thread_view| thread_view.entity_id());
    if state.borrow().composer_thread_view_id == thread_view_id {
        return;
    }

    let adapter = agent_ui::collaborative_composer::CollaborativeComposerAdapter::from_agent_panel(
        agent_panel,
        workspace_handle,
        cx,
    );
    let state = state.clone();
    workspace_handle.update(cx, |workspace, cx| {
        if let Some(registration) = state.borrow_mut().composer_registration.take() {
            workspace.unregister_collaborative_composer_provider(registration, cx);
        }
        state.borrow_mut().composer_thread_view_id = None;

        let adapter = match adapter {
            Ok(adapter) => adapter,
            Err(
                agent_ui::collaborative_composer::CollaborativeComposerAdapterError::ThreadUnavailable,
            ) => return,
            Err(error) => {
                log::warn!("failed to adapt collaborative composer: {error}");
                return;
            }
        };
        let thread_view_id = adapter.thread_view_id();
        match adapter.register_in_workspace(workspace, cx) {
            Ok(registration) => {
                let mut state = state.borrow_mut();
                state.composer_thread_view_id = Some(thread_view_id);
                state.composer_registration = Some(registration);
            }
            Err(error) => {
                log::warn!("failed to register collaborative composer: {error}");
            }
        }
    });
}

#[cfg(feature = "multiplayer-tools")]
fn reconcile_collaborative_timeline(
    workspace_handle: &Entity<Workspace>,
    agent_panel: &Entity<AgentPanel>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut App,
) {
    let conversation_view_id = agent_panel
        .read(cx)
        .active_conversation_view()
        .map(|conversation| conversation.entity_id());
    if state.borrow().timeline_conversation_view_id == conversation_view_id {
        return;
    }

    let adapter = agent_ui::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(
        agent_panel,
        workspace_handle,
        cx,
    );
    let state = state.clone();
    workspace_handle.update(cx, |workspace, cx| {
        if let Some(registration) = state.borrow_mut().timeline_registration.take() {
            workspace.unregister_collaborative_timeline_provider(registration, cx);
        }
        state.borrow_mut().timeline_conversation_view_id = None;

        let adapter = match adapter {
            Ok(adapter) => adapter,
            Err(
                agent_ui::collaborative_timeline::CollaborativeTimelineAdapterError::ConversationUnavailable,
            ) => return,
            Err(error) => {
                log::warn!("failed to adapt collaborative timeline: {error}");
                return;
            }
        };
        let conversation_view_id = adapter.conversation_view_id();
        match adapter.register_in_workspace(workspace, cx) {
            Ok(registration) => {
                let mut state = state.borrow_mut();
                state.timeline_conversation_view_id = Some(conversation_view_id);
                state.timeline_registration = Some(registration);
            }
            Err(error) => log::warn!("failed to register collaborative timeline: {error}"),
        }
    });
}

#[cfg(feature = "multiplayer-tools")]
fn apply_collaborative_participant_projection(
    workspace_handle: &Entity<Workspace>,
    projection: Option<CollaborativeParticipantProjection>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut App,
) {
    let projection_is_current = projection.as_ref().is_some_and(|projection| {
        let state = state.borrow();
        state.participant_thread_view_id == Some(projection.thread_view_id)
            && state.participant_registration.is_some()
    });
    if projection_is_current {
        workspace_handle.update(cx, |_, cx| cx.notify());
        return;
    }
    if projection.is_none() && state.borrow().participant_registration.is_none() {
        return;
    }

    let state = state.clone();
    workspace_handle.update(cx, |workspace, cx| {
        if let Some(projection) = projection {
            if let Some(registration) = state.borrow_mut().participant_registration.take() {
                workspace.unregister_collaborative_participant_provider(registration, cx);
            }
            state.borrow_mut().participant_thread_view_id = None;
            match workspace.register_collaborative_participant_provider(projection.provider, cx) {
                Ok(registration) => {
                    let mut state = state.borrow_mut();
                    state.participant_thread_view_id = Some(projection.thread_view_id);
                    state.participant_registration = Some(registration);
                }
                Err(error) => {
                    log::warn!("failed to register collaborative participant provider: {error}");
                }
            }
        } else {
            if let Some(registration) = state.borrow_mut().participant_registration.take() {
                workspace.unregister_collaborative_participant_provider(registration, cx);
            }
            let mut state = state.borrow_mut();
            state.participant_thread_view_id = None;
        }
    });
}

#[cfg(feature = "multiplayer-tools")]
fn reconcile_collaborative_participants(
    workspace_handle: &Entity<Workspace>,
    agent_panel: &Entity<AgentPanel>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut App,
) {
    let projection = match agent_ui::collaborative_participants::CollaborativeParticipantAdapter::from_agent_panel(
        agent_panel,
        workspace_handle,
        cx,
    ) {
        Ok(adapter) => {
            let adapter = adapter.with_room_state_reader(|cx| {
                call::ActiveCall::try_global(cx)
                    .and_then(|active_call| active_call.read(cx).room().cloned())
                    .map(|room| {
                        room.read_with(cx, |room, cx| {
                            let mut participants = Vec::new();
                            if let Some(user) = room.local_participant_user(cx) {
                                participants.push(CollaborativeParticipant::human(
                                    &user,
                                    CollaborativeParticipantPresence::Online,
                                ));
                            }
                            participants.extend(
                                room.remote_participants().values().map(|participant| {
                                    CollaborativeParticipant::human(
                                        &participant.user,
                                        CollaborativeParticipantPresence::Online,
                                    )
                                }),
                            );
                            let connection = if room.status().is_online() {
                                CollaborativeConnectionState::Connected
                            } else if room.status().is_offline() {
                                CollaborativeConnectionState::Failed
                            } else {
                                CollaborativeConnectionState::Connecting
                            };
                            (participants, connection)
                        })
                    })
                    .unwrap_or_default()
            });
            Some(CollaborativeParticipantProjection::from_adapter(adapter))
        }
        Err(
            agent_ui::collaborative_participants::CollaborativeParticipantAdapterError::ThreadUnavailable,
        ) => None,
        Err(error) => {
            log::warn!("failed to adapt collaborative participants: {error}");
            None
        }
    };
    apply_collaborative_participant_projection(workspace_handle, projection, state, cx);
}

#[cfg(feature = "multiplayer-tools")]
fn reconcile_collaborative_project_review(
    workspace_handle: &Entity<Workspace>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    window: &mut Window,
    cx: &mut App,
) {
    let project_diff_id = workspace_handle
        .read(cx)
        .item_of_type::<ProjectDiff>(cx)
        .map(|project_diff| project_diff.entity_id());
    {
        let state = state.borrow();
        if state.project_registration.is_some()
            && (state.project_diff_id == project_diff_id || project_diff_id.is_none())
        {
            return;
        }
    }

    let adapter =
        git_ui::collaborative_review::CollaborativeProjectReviewAdapter::from_workspace_or_create(
            workspace_handle,
            window,
            cx,
        );
    let state = state.clone();
    workspace_handle.update(cx, |workspace, cx| {
        if let Some(registration) = state.borrow_mut().project_registration.take() {
            workspace.unregister_collaborative_review_provider(registration, cx);
        }
        state.borrow_mut().project_diff_id = None;

        let adapter = match adapter {
            Ok(adapter) => adapter,
            Err(
                git_ui::collaborative_review::CollaborativeProjectReviewError::ProjectDiffUnavailable,
            ) => return,
            Err(error) => {
                log::warn!("failed to adapt collaborative project review: {error}");
                return;
            }
        };
        let project_diff_id = adapter.project_diff().entity_id();
        match adapter.register_in_workspace(workspace, cx) {
            Ok(registration) => {
                let mut state = state.borrow_mut();
                state.project_diff_id = Some(project_diff_id);
                state.project_registration = Some(registration);
            }
            Err(error) => {
                log::warn!("failed to register collaborative project review: {error}");
            }
        }
    });
}

#[cfg(feature = "multiplayer-tools")]
fn reconcile_collaborative_agent_review(
    workspace_handle: &Entity<Workspace>,
    thread: Option<Entity<acp_thread::AcpThread>>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    window: &mut Window,
    cx: &mut App,
) {
    let thread_id = thread.as_ref().map(Entity::entity_id);
    if state.borrow().agent_thread_id == thread_id {
        return;
    }

    let adapter = agent_ui::collaborative_review::CollaborativeAgentReviewAdapter::new(
        thread,
        workspace_handle,
        window,
        cx,
    );
    let state = state.clone();
    workspace_handle.update(cx, |workspace, cx| {
        if let Some(registration) = state.borrow_mut().agent_registration.take() {
            workspace.unregister_collaborative_review_provider(registration, cx);
        }
        state.borrow_mut().agent_thread_id = None;

        let adapter = match adapter {
            Ok(adapter) => adapter,
            Err(
                agent_ui::collaborative_review::CollaborativeAgentReviewError::ThreadUnavailable,
            ) => {
                return;
            }
            Err(error) => {
                log::warn!("failed to adapt collaborative agent review: {error}");
                return;
            }
        };
        match adapter.register_in_workspace(workspace, cx) {
            Ok(registration) => {
                let mut state = state.borrow_mut();
                state.agent_thread_id = thread_id;
                state.agent_registration = Some(registration);
            }
            Err(error) => {
                log::warn!("failed to register collaborative agent review: {error}");
            }
        }
    });
}

#[cfg(feature = "multiplayer-tools")]
fn schedule_collaborative_project_review_reconciliation(
    workspace_handle: Entity<Workspace>,
    state: Rc<RefCell<CollaborativeReviewCompositionState>>,
    window: &Window,
    cx: &mut Context<Workspace>,
) {
    let window_handle = window.window_handle();
    cx.defer(move |cx| {
        if let Err(error) = window_handle.update(cx, |_, window, cx| {
            reconcile_collaborative_project_review(&workspace_handle, &state, window, cx);
        }) {
            log::warn!("failed to reconcile collaborative project review: {error}");
        }
    });
}

#[cfg(feature = "multiplayer-tools")]
fn schedule_collaborative_agent_review_reconciliation(
    workspace_handle: Entity<Workspace>,
    thread: Option<Entity<acp_thread::AcpThread>>,
    state: Rc<RefCell<CollaborativeReviewCompositionState>>,
    window: &Window,
    cx: &mut Context<Workspace>,
) {
    let window_handle = window.window_handle();
    cx.defer(move |cx| {
        if let Err(error) = window_handle.update(cx, |_, window, cx| {
            reconcile_collaborative_agent_review(&workspace_handle, thread, &state, window, cx);
        }) {
            log::warn!("failed to reconcile collaborative agent review: {error}");
        }
    });
}

#[cfg(feature = "multiplayer-tools")]
fn schedule_collaborative_composer_reconciliation(
    workspace_handle: Entity<Workspace>,
    agent_panel: Entity<AgentPanel>,
    state: Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut Context<Workspace>,
) {
    cx.defer(move |cx| {
        reconcile_collaborative_composer(&workspace_handle, &agent_panel, &state, cx);
    });
}

#[cfg(feature = "multiplayer-tools")]
fn schedule_collaborative_timeline_reconciliation(
    workspace_handle: Entity<Workspace>,
    agent_panel: Entity<AgentPanel>,
    state: Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut Context<Workspace>,
) {
    cx.defer(move |cx| {
        reconcile_collaborative_timeline(&workspace_handle, &agent_panel, &state, cx);
    });
}

#[cfg(feature = "multiplayer-tools")]
fn schedule_collaborative_participant_reconciliation(
    workspace_handle: Entity<Workspace>,
    agent_panel: Entity<AgentPanel>,
    state: Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut Context<Workspace>,
) {
    cx.defer(move |cx| {
        reconcile_collaborative_participants(&workspace_handle, &agent_panel, &state, cx);
    });
}

#[cfg(feature = "multiplayer-tools")]
fn observe_collaborative_participant_thread(
    workspace_handle: &Entity<Workspace>,
    agent_panel: &Entity<AgentPanel>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut Context<Workspace>,
) {
    let thread_view = agent_panel.read(cx).active_thread_view(cx);
    let thread_view_id = thread_view.as_ref().map(Entity::entity_id);
    if state.borrow().participant_observed_thread_view_id == thread_view_id {
        return;
    }

    {
        let mut state = state.borrow_mut();
        state.participant_observed_thread_view_id = thread_view_id;
        state.participant_thread_observation = None;
    }
    let Some(thread_view) = thread_view else {
        return;
    };

    let workspace_handle = workspace_handle.clone();
    let agent_panel = agent_panel.clone();
    let weak_state = Rc::downgrade(state);
    let observation = cx.observe(&thread_view, move |_, _, cx| {
        let Some(state) = weak_state.upgrade() else {
            return;
        };
        schedule_collaborative_participant_reconciliation(
            workspace_handle.clone(),
            agent_panel.clone(),
            state,
            cx,
        );
    });
    state.borrow_mut().participant_thread_observation = Some(observation);
}

#[cfg(feature = "multiplayer-tools")]
fn observe_collaborative_participant_room(
    workspace_handle: &Entity<Workspace>,
    agent_panel: &Entity<AgentPanel>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    cx: &mut Context<Workspace>,
) {
    let room = call::ActiveCall::try_global(cx)
        .and_then(|active_call| active_call.read(cx).room().cloned());
    let room_id = room.as_ref().map(Entity::entity_id);
    if state.borrow().participant_room_id == room_id {
        return;
    }
    {
        let mut state = state.borrow_mut();
        state.participant_room_id = room_id;
        state.participant_room_observation = None;
    }
    let Some(room) = room else {
        return;
    };
    let workspace_handle = workspace_handle.clone();
    let agent_panel = agent_panel.clone();
    let weak_state = Rc::downgrade(state);
    let observation = cx.observe(&room, move |_, _, cx| {
        let Some(state) = weak_state.upgrade() else {
            return;
        };
        schedule_collaborative_participant_reconciliation(
            workspace_handle.clone(),
            agent_panel.clone(),
            state,
            cx,
        );
    });
    state.borrow_mut().participant_room_observation = Some(observation);
}

#[cfg(feature = "multiplayer-tools")]
fn subscribe_to_collaborative_review_agent_panel(
    workspace_handle: &Entity<Workspace>,
    agent_panel: Entity<AgentPanel>,
    state: &Rc<RefCell<CollaborativeReviewCompositionState>>,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let agent_panel_id = agent_panel.entity_id();
    if state.borrow().agent_panel_id == Some(agent_panel_id) {
        return;
    }
    state.borrow_mut().agent_panel_id = Some(agent_panel_id);

    let panel_for_navigation = agent_panel.clone();
    cx.defer_in(window, move |workspace, window, cx| {
        AgentPanel::synchronize_collaborative_thread_navigation(
            workspace,
            &panel_for_navigation,
            true,
            window,
            cx,
        )
        .log_err();
    });

    schedule_collaborative_agent_review_reconciliation(
        workspace_handle.clone(),
        agent_panel.read(cx).active_agent_thread(cx),
        state.clone(),
        window,
        cx,
    );
    schedule_collaborative_composer_reconciliation(
        workspace_handle.clone(),
        agent_panel.clone(),
        state.clone(),
        cx,
    );
    schedule_collaborative_timeline_reconciliation(
        workspace_handle.clone(),
        agent_panel.clone(),
        state.clone(),
        cx,
    );
    observe_collaborative_participant_thread(workspace_handle, &agent_panel, state, cx);
    schedule_collaborative_participant_reconciliation(
        workspace_handle.clone(),
        agent_panel.clone(),
        state.clone(),
        cx,
    );
    observe_collaborative_participant_room(workspace_handle, &agent_panel, state, cx);
    if state.borrow().participant_active_call_observation.is_none()
        && let Some(active_call) = call::ActiveCall::try_global(cx)
    {
        let workspace_handle = workspace_handle.clone();
        let agent_panel = agent_panel.clone();
        let weak_state = Rc::downgrade(state);
        let observation = cx.observe(&active_call, move |_, _, cx| {
            let Some(state) = weak_state.upgrade() else {
                return;
            };
            observe_collaborative_participant_room(&workspace_handle, &agent_panel, &state, cx);
            schedule_collaborative_participant_reconciliation(
                workspace_handle.clone(),
                agent_panel.clone(),
                state,
                cx,
            );
        });
        state.borrow_mut().participant_active_call_observation = Some(observation);
    }

    let workspace_handle = workspace_handle.clone();
    let state = state.clone();
    cx.subscribe_in(
        &agent_panel,
        window,
        move |_, agent_panel, event: &AgentPanelEvent, window, cx| {
            if matches!(event, AgentPanelEvent::ActiveViewChanged) {
                schedule_collaborative_agent_review_reconciliation(
                    workspace_handle.clone(),
                    agent_panel.read(cx).active_agent_thread(cx),
                    state.clone(),
                    window,
                    cx,
                );
                schedule_collaborative_composer_reconciliation(
                    workspace_handle.clone(),
                    agent_panel.clone(),
                    state.clone(),
                    cx,
                );
                schedule_collaborative_timeline_reconciliation(
                    workspace_handle.clone(),
                    agent_panel.clone(),
                    state.clone(),
                    cx,
                );
                observe_collaborative_participant_thread(
                    &workspace_handle,
                    agent_panel,
                    &state,
                    cx,
                );
                schedule_collaborative_participant_reconciliation(
                    workspace_handle.clone(),
                    agent_panel.clone(),
                    state.clone(),
                    cx,
                );
            }
        },
    )
    .detach();
}

#[cfg(feature = "sim")]
#[derive(Clone)]
struct DesktopComponentLifecycleAdapter {
    router: sim_plugin_host::ComponentHostRouter,
    accepted_candidate_identity:
        Arc<std::sync::Mutex<Option<extension_host::ComponentInventoryCandidateIdentity>>>,
}

#[cfg(feature = "sim")]
impl extension_host::ComponentLifecycleAdapter for DesktopComponentLifecycleAdapter {
    fn adapter_id(&self) -> &'static str {
        "sim.desktop-component-lifecycle"
    }

    fn synchronize_candidate(
        &self,
        candidate: extension_host::ComponentInventoryCandidate,
    ) -> futures::future::BoxFuture<'static, Result<(), String>> {
        let router = self.router.clone();
        let accepted_candidate_identity = self.accepted_candidate_identity.clone();
        async move {
            let candidate_identity = candidate.identity().clone();
            let previous_identity = accepted_candidate_identity
                .lock()
                .map_err(|_| "desktop component candidate identity is unavailable".to_owned())?
                .replace(candidate_identity);
            if let Err(error) = extension_host::ComponentLifecycleAdapter::synchronize(
                &router,
                candidate.into_components(),
            )
            .await
            {
                *accepted_candidate_identity.lock().map_err(|_| {
                    "desktop component candidate identity is unavailable".to_owned()
                })? = previous_identity;
                return Err(error);
            }
            Ok(())
        }
        .boxed()
    }

    fn synchronize(
        &self,
        components: Vec<extension_host::InstalledComponent>,
    ) -> futures::future::BoxFuture<'static, Result<(), String>> {
        extension_host::ComponentLifecycleAdapter::synchronize(&self.router, components)
    }
}

#[cfg(feature = "sim")]
#[derive(Clone)]
struct DeferredDesktopComponentLifecycleAdapter {
    active: Arc<std::sync::Mutex<Option<DesktopComponentLifecycleAdapter>>>,
    accepted_candidate_identity:
        Arc<std::sync::Mutex<Option<extension_host::ComponentInventoryCandidateIdentity>>>,
}

#[cfg(feature = "sim")]
impl DeferredDesktopComponentLifecycleAdapter {
    fn active_adapter(&self) -> Result<DesktopComponentLifecycleAdapter, String> {
        self.active
            .lock()
            .map_err(|_| "desktop component adapter state is unavailable".to_owned())?
            .clone()
            .ok_or_else(|| "native Sim component host is unavailable".to_owned())
    }
}

#[cfg(feature = "sim")]
impl extension_host::ComponentLifecycleAdapter for DeferredDesktopComponentLifecycleAdapter {
    fn adapter_id(&self) -> &'static str {
        "sim.desktop-component-lifecycle"
    }

    fn synchronize_candidate(
        &self,
        candidate: extension_host::ComponentInventoryCandidate,
    ) -> futures::future::BoxFuture<'static, Result<(), String>> {
        match self.active_adapter() {
            Ok(adapter) => extension_host::ComponentLifecycleAdapter::synchronize_candidate(
                &adapter, candidate,
            ),
            Err(error) => futures::future::ready(Err(error)).boxed(),
        }
    }

    fn synchronize(
        &self,
        components: Vec<extension_host::InstalledComponent>,
    ) -> futures::future::BoxFuture<'static, Result<(), String>> {
        match self.active_adapter() {
            Ok(adapter) => {
                extension_host::ComponentLifecycleAdapter::synchronize(&adapter, components)
            }
            Err(error) => futures::future::ready(Err(error)).boxed(),
        }
    }
}

#[cfg(feature = "sim")]
struct DesktopComponentLifecycleRegistration(DeferredDesktopComponentLifecycleAdapter);

#[cfg(feature = "sim")]
impl gpui::Global for DesktopComponentLifecycleRegistration {}

#[cfg(feature = "sim")]
fn ensure_desktop_component_lifecycle_registration(
    cx: &mut App,
) -> anyhow::Result<DeferredDesktopComponentLifecycleAdapter> {
    if let Some(registration) = cx.try_global::<DesktopComponentLifecycleRegistration>() {
        return Ok(registration.0.clone());
    }
    let adapter = DeferredDesktopComponentLifecycleAdapter {
        active: Arc::new(std::sync::Mutex::new(None)),
        accepted_candidate_identity: Arc::new(std::sync::Mutex::new(None)),
    };
    // ExtensionStore snapshots adapters before fallible profile/worker bootstrap.
    // Returning unavailable rejects the candidate until a real host can accept it.
    extension_host::register_component_lifecycle_adapter(Arc::new(adapter.clone()), cx)?;
    cx.set_global(DesktopComponentLifecycleRegistration(adapter.clone()));
    Ok(adapter)
}

#[cfg(feature = "sim")]
struct SimComponentHostGlobal {
    profile_id: String,
    component_generation: u64,
    #[cfg(not(test))]
    plugin_security: sim_runtime::NativePluginSecurityPolicy,
    router: sim_plugin_host::ComponentHostRouter,
    #[cfg(not(test))]
    provider_invocation_authority: Option<Arc<dyn sim_runtime::NativeProviderInvocationAuthority>>,
    #[cfg(not(test))]
    private_worker_executor: Arc<sim_plugin_host::PrivateWorkerPluginExecutor>,
    #[cfg(not(test))]
    provider_cost_authority: Arc<sim_runtime::ProviderCostApprovalAuthority>,
    accepted_inventory: bool,
    accepted_candidate_identity:
        Arc<std::sync::Mutex<Option<extension_host::ComponentInventoryCandidateIdentity>>>,
    #[cfg(not(test))]
    active_provider_deployment: Option<sim_api::NativeProviderDeploymentIdentity>,
}

#[cfg(all(test, feature = "rust-tools"))]
mod cargo_panel_feature_tests {
    use super::*;

    #[test]
    fn cargo_panel_is_registered_in_rust_tools_builds() {
        assert_eq!(<CargoPanel as workspace::Panel>::persistent_name(), "Cargo");
        assert_eq!(<CargoPanel as workspace::Panel>::panel_key(), "CargoPanel");
    }

    #[test]
    fn cargo_actions_are_registered_only_with_rust_tools() {
        let actions = cargo_ui::CargoAction::ALL.map(cargo_ui::CargoAction::label);
        assert_eq!(
            actions,
            [
                "Build",
                "Check",
                "Run",
                "Run with Coverage",
                "Profile",
                "Test",
                "Bench",
                "Debug",
                "Doc",
                "Clippy",
                "Fmt",
                "Clean",
                "Tree",
            ]
        );
        let _ = cargo_ui::BuildSelected;
        let _ = cargo_ui::CheckSelected;
        let _ = cargo_ui::RunSelected;
        let _ = cargo_ui::ProfileSelected;
        let _ = cargo_ui::TestSelected;
        let _ = cargo_ui::BenchSelected;
        let _ = cargo_ui::DebugSelected;
    }

    #[test]
    fn rust_test_explorer_actions_are_registered_only_with_rust_tools() {
        let _ = tasks_ui::ToggleTestsPanel;
        let _ = tasks_ui::RunSelectedTests;
        let _ = tasks_ui::DebugSelectedTests;
        let _ = tasks_ui::CancelTestRun;
        let _ = tasks_ui::RerunFailedTests;
        let _ = tasks_ui::RevealTestTerminal;
    }

    #[test]
    fn rust_workspace_enabled_desktop_registers_cargo_and_tests_panels() {
        assert!(cfg!(feature = "rust-tools"));
        assert_eq!(<CargoPanel as workspace::Panel>::persistent_name(), "Cargo");
        let _ = tasks_ui::ToggleTestsPanel;
        let _ = cargo_ui::BuildSelected;
    }
}

#[cfg(all(test, not(feature = "rust-tools")))]
mod cargo_panel_disabled_feature_tests {
    #[test]
    fn cargo_panel_disabled_build_has_no_rust_tools_capability() {
        assert!(!cfg!(feature = "rust-tools"));
    }

    #[test]
    fn rust_workspace_disabled_desktop_has_no_rust_tools_capability() {
        assert!(!cfg!(feature = "rust-tools"));
    }
}

#[cfg(feature = "sim")]
impl gpui::Global for SimComponentHostGlobal {}

#[cfg(feature = "sim")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeSimRuntimeBinding {
    profile_id: Uuid,
    model_roots: Vec<PathBuf>,
    device: sim_types::DeviceKind,
    memory_policy: sim_runtime::MemoryPolicy,
    api_host: sim_runtime::NativeApiHostPolicy,
    plugin_policy: sim_runtime::PluginPolicy,
    rocm_package: Option<sim_runtime::NativeRocmPackageSettings>,
    metal_package: Option<sim_runtime::NativeMetalPackageSettings>,
    mlu_package: Option<sim_runtime::NativeMluPackageSettings>,
    npu_package: Option<sim_runtime::NativeNpuPackageSettings>,
    cuda_package: Option<sim_runtime::NativeCudaPackageSettings>,
    xpu_package: Option<sim_runtime::NativeXpuPackageSettings>,
    directml_package: Option<sim_runtime::NativeDirectMlPackageSettings>,
    provider_scope: String,
    compatibility_version: u16,
    plugin_security: sim_runtime::NativePluginSecurityPolicy,
}

#[cfg(feature = "sim")]
impl NativeSimRuntimeBinding {
    fn new(
        profile: &sim_runtime::NativeRuntimeProfile,
        plugin_security: &sim_runtime::NativePluginSecurityPolicy,
    ) -> Self {
        Self {
            profile_id: profile.id,
            model_roots: profile.model_roots.clone(),
            device: profile.device,
            memory_policy: profile.memory_policy,
            api_host: profile.api_host.clone(),
            plugin_policy: profile.plugin_policy,
            rocm_package: profile.rocm_package.clone(),
            metal_package: profile.metal_package.clone(),
            mlu_package: profile.mlu_package.clone(),
            npu_package: profile.npu_package.clone(),
            cuda_package: profile.cuda_package.clone(),
            xpu_package: profile.xpu_package.clone(),
            directml_package: profile.directml_package.clone(),
            provider_scope: profile.provider_scope.clone(),
            compatibility_version: profile.compatibility_version,
            plugin_security: plugin_security.clone(),
        }
    }
}

#[cfg(feature = "sim")]
impl gpui::Global for NativeSimRuntimeBinding {}

#[cfg(feature = "sim")]
#[derive(Clone)]
struct SimPluginContributionSource {
    router: sim_plugin_host::ComponentHostRouter,
}

#[cfg(feature = "sim")]
impl sim_ui::PluginContributionSource for SimPluginContributionSource {
    fn verified_contributions(&self) -> anyhow::Result<Vec<sim_ui::PluginContributionInput>> {
        let plugins = self.router.current()?.installed_plugins()?;
        let mut contributions = Vec::new();
        for plugin in plugins {
            let binding = plugin.binding();
            for contribution in &plugin.manifest().ui {
                contributions.push(sim_ui::PluginContributionInput::from_verified_manifest(
                    binding.signed_plugin_identifier(),
                    binding.signed_digest_sha256(),
                    contribution.id.as_str(),
                    contribution.surface.as_str(),
                    contribution.state_schema.as_str(),
                )?);
            }
        }
        Ok(contributions)
    }
}

#[cfg(feature = "sim")]
fn register_sim_plugin_contribution_source(
    router: sim_plugin_host::ComponentHostRouter,
    cx: &mut App,
) {
    sim_ui::register_plugin_contribution_source(
        Arc::new(SimPluginContributionSource { router }),
        cx,
    );
}

#[cfg(feature = "sim")]
fn init_sim_component_host(replace_assets: bool, cx: &mut App) -> anyhow::Result<()> {
    let lifecycle_adapter = ensure_desktop_component_lifecycle_registration(cx)?;
    #[cfg(not(test))]
    let (profile, plugin_security) = active_native_sim_configuration(cx)?;
    #[cfg(not(test))]
    let profile_id = profile.id.to_string();
    #[cfg(not(test))]
    {
        let profile_root = sim_runtime::native_profile_asset_root(paths::data_dir(), profile.id);
        // Private worker services and the execution controller must retain the
        // same checked service, including after a profile configuration change.
        ensure_native_sim_assets(&profile, replace_assets, &profile_root, cx)?;
    }
    #[cfg(test)]
    let profile_id = sim_ui::LOCAL_EXECUTION_PROFILE_ID.0.to_string();

    #[cfg(not(test))]
    let (trust_policy, permission_policy, component_generation) = (
        plugin_security.trust_policy().clone(),
        plugin_security.permission_policy().clone(),
        plugin_security.component_registry_generation(),
    );
    #[cfg(test)]
    let (trust_policy, permission_policy, component_generation) = (
        sim_runtime::PluginTrustPolicy::default(),
        sim_runtime::PermissionPolicy::new(profile_id.clone(), std::iter::empty())?,
        sim_runtime::DEFAULT_COMPONENT_REGISTRY_GENERATION,
    );

    if let Some(component_host) = cx.try_global::<SimComponentHostGlobal>() {
        let matches_current = !replace_assets
            && component_host.profile_id == profile_id
            && component_host.component_generation == component_generation
            && {
                #[cfg(not(test))]
                {
                    component_host.plugin_security == plugin_security
                }
                #[cfg(test)]
                {
                    true
                }
            };
        if matches_current {
            let router = component_host.router.clone();
            router.current()?.installed_plugins()?;
            register_sim_plugin_contribution_source(router, cx);
            return Ok(());
        }
    }

    #[cfg(not(test))]
    let plugin_services = {
        let asset_service = sim_ui::native_asset_services(cx)
            .ok_or_else(|| anyhow::anyhow!("native Sim asset service is unavailable"))?;
        anyhow::ensure!(
            asset_service.profile_id() == profile_id,
            "native Sim component and asset profiles differ"
        );
        let worker = native_sim_worker_launch(&profile, sim_types::WorkerId(Uuid::new_v4()))?;
        let profile_bits = profile.id.as_u128();
        let profile_seed = (profile_bits as u64) ^ ((profile_bits >> 64) as u64);
        sim_plugin_services::private_worker_services(
            worker,
            asset_service.assets(),
            plugin_security.provider_policy().clone(),
            profile_seed,
            cx,
        )?
    };
    #[cfg(not(test))]
    let execution_boundary = plugin_services.boundary.clone();
    #[cfg(not(test))]
    let private_worker_executor = plugin_services.private_worker_executor();
    #[cfg(test)]
    let execution_boundary = sim_plugin_host::ComponentExecutionBoundary::conformance_in_process(
        Arc::new(sim_plugin_host::UnavailablePluginCapabilityServices),
    );

    let runtime = extension_host::ComponentRuntime::no_wasi()?;
    let replacement_host = sim_plugin_host::ComponentHost::new(
        runtime,
        trust_policy,
        permission_policy,
        execution_boundary,
        sim_plugin_host::ComponentLimits::default(),
        sim_runtime::generated_native_node_registry_projection(None)?,
    )?;
    #[cfg(not(test))]
    let provider_invocation_authority = Some(
        plugin_services.invocation_authority(replacement_host.clone())?
            as Arc<dyn sim_runtime::NativeProviderInvocationAuthority>,
    );
    #[cfg(not(test))]
    let provider_cost_authority = plugin_services.cost_authority();
    if let Some(component_host) = cx.try_global::<SimComponentHostGlobal>() {
        let router = component_host.router.clone();
        router.replace_with_initial_generation(replacement_host, component_generation)?;
        register_sim_plugin_contribution_source(router, cx);
        let component_host = cx.global_mut::<SimComponentHostGlobal>();
        component_host.profile_id = profile_id;
        component_host.component_generation = component_generation;
        #[cfg(not(test))]
        {
            component_host.plugin_security = plugin_security;
        }
        #[cfg(not(test))]
        {
            component_host.provider_invocation_authority = provider_invocation_authority;
            component_host.active_provider_deployment = None;
            component_host.private_worker_executor = private_worker_executor;
            component_host.provider_cost_authority = provider_cost_authority;
        }
        return Ok(());
    }
    let router = sim_plugin_host::ComponentHostRouter::with_initial_generation(
        replacement_host,
        component_generation,
    )?;
    let accepted_candidate_identity = lifecycle_adapter.accepted_candidate_identity.clone();
    register_sim_plugin_contribution_source(router.clone(), cx);
    #[cfg(not(test))]
    let receiver = {
        let receiver = router.subscribe_execution_registry_bundles()?;
        receiver.try_recv()?;
        receiver
    };
    *lifecycle_adapter
        .active
        .lock()
        .map_err(|_| anyhow::anyhow!("desktop component adapter state is unavailable"))? =
        Some(DesktopComponentLifecycleAdapter {
            router: router.clone(),
            accepted_candidate_identity: accepted_candidate_identity.clone(),
        });
    cx.set_global(SimComponentHostGlobal {
        profile_id,
        component_generation,
        #[cfg(not(test))]
        plugin_security,
        router,
        #[cfg(not(test))]
        provider_invocation_authority,
        #[cfg(not(test))]
        private_worker_executor,
        #[cfg(not(test))]
        provider_cost_authority,
        accepted_inventory: false,
        accepted_candidate_identity,
        #[cfg(not(test))]
        active_provider_deployment: None,
    });
    #[cfg(not(test))]
    {
        cx.spawn(async move |cx| {
            while let Ok(bundle) = receiver.recv().await {
                let result = cx.update(|cx| {
                    let selection = (|| {
                        let settings_store = cx
                            .try_global::<SettingsStore>()
                            .ok_or_else(|| anyhow::anyhow!("Zed settings store is unavailable"))?;
                        let component_host =
                            cx.try_global::<SimComponentHostGlobal>().ok_or_else(|| {
                                anyhow::anyhow!("native Sim component host is unavailable")
                            })?;
                        let current_bundle =
                            component_host.router.active_execution_registry_bundle()?;
                        native_sim_registry_event_profile(
                            settings_store.merged_settings(),
                            cx.try_global::<NativeSimRuntimeBinding>(),
                            &component_host.profile_id,
                            &component_host.plugin_security,
                            &bundle,
                            &current_bundle,
                        )
                    })();
                    rebind_native_sim_registry_event(selection, cx, register_native_sim_execution)
                });
                if let Err(error) = result {
                    eprintln!("native Sim component lifecycle rebind failed: {error}");
                }
            }
        })
        .detach();
    }
    Ok(())
}

#[cfg(feature = "sim")]
pub(crate) fn init_sim_ui(cx: &mut App) {
    #[cfg(test)]
    sim_ui::init(cx);

    #[cfg(not(test))]
    {
        match active_native_sim_profile(cx) {
            Ok(profile) => {
                sim_ui::init_for_profile(sim_types::ProfileId(profile.id), cx);
            }
            Err(error) => {
                let profile_id = configured_native_sim_profile_id(cx)
                    .unwrap_or(sim_types::ProfileId(Uuid::nil()));
                sim_ui::init_for_profile(profile_id, cx);
                if let Err(clear_error) = sim_ui::clear_native_execution_services(cx) {
                    log::error!("native Sim execution shutdown failed: {clear_error}");
                }
                let message = format!("native Sim settings are invalid: {error}");
                log::error!("{message}");
                sim_ui::set_initialization_error(message, cx);
            }
        }
        cx.observe_global::<SettingsStore>(sync_active_native_sim_profile)
            .detach();
    }
}

#[cfg(not(test))]
#[cfg(feature = "sim")]
fn sync_active_native_sim_profile(cx: &mut App) {
    match active_native_sim_configuration(cx) {
        Ok((profile, plugin_security)) => {
            let profile_id = sim_types::ProfileId(profile.id);
            let requires_activation = sim_ui::initialized_profile_id(cx) != Some(profile_id);
            let requires_recovery = sim_ui::initialization_error(cx).is_some();
            let next_binding = NativeSimRuntimeBinding::new(&profile, &plugin_security);
            let requires_rebind = cx.try_global::<NativeSimRuntimeBinding>() != Some(&next_binding);
            if !requires_activation && !requires_recovery && !requires_rebind {
                return;
            }
            if requires_activation || requires_rebind {
                clear_native_graphite_assets(cx);
            }
            sim_ui::clear_initialization_error(cx);
            sim_ui::init_for_profile(profile_id, cx);
            if let Err(error) = init_sim_component_host(requires_rebind, cx) {
                let message = format!("native Sim component host initialization failed: {error}");
                fail_native_sim_activation(message, cx);
            } else {
                match bind_native_sim_host_or_wait_for_inventory(
                    &profile,
                    next_binding.clone(),
                    cx,
                    register_native_sim_execution,
                ) {
                    Err(error) => fail_native_sim_activation(
                        format!("native Sim worker registry initialization failed: {error}"),
                        cx,
                    ),
                    Ok(Some(reload)) => {
                        cx.spawn(async move |cx| {
                            if let Err(error) = reload.await {
                                cx.update(|cx| {
                                    if cx.try_global::<NativeSimRuntimeBinding>()
                                        == Some(&next_binding)
                                        && cx
                                            .try_global::<SimComponentHostGlobal>()
                                            .is_some_and(|host| !host.accepted_inventory)
                                    {
                                        fail_native_sim_activation(
                                            format!(
                                                "native Sim inventory recovery failed: {error}"
                                            ),
                                            cx,
                                        );
                                    }
                                });
                            }
                            anyhow::Ok(())
                        })
                        .detach_and_log_err(cx);
                    }
                    Ok(None) => {}
                }
            }
        }
        Err(error) => {
            clear_native_graphite_assets(cx);
            if cx.has_global::<NativeSimRuntimeBinding>() {
                let _removed_binding = cx.remove_global::<NativeSimRuntimeBinding>();
            }
            if let Err(clear_error) = sim_ui::clear_native_execution_services(cx) {
                log::error!("native Sim execution shutdown failed: {clear_error}");
            }
            let profile_id =
                configured_native_sim_profile_id(cx).unwrap_or(sim_types::ProfileId(Uuid::nil()));
            if sim_ui::initialized_profile_id(cx) != Some(profile_id) {
                sim_ui::init_for_profile(profile_id, cx);
            }
            sim_ui::clear_initialization_error(cx);
            let message = format!("native Sim settings are invalid: {error}");
            log::error!("{message}");
            sim_ui::set_initialization_error(message, cx);
        }
    }
}

#[cfg(feature = "sim")]
fn bind_native_sim_host_or_wait_for_inventory(
    profile: &sim_runtime::NativeRuntimeProfile,
    next_binding: NativeSimRuntimeBinding,
    cx: &mut App,
    register_execution: impl FnOnce(&sim_runtime::NativeRuntimeProfile, &mut App) -> anyhow::Result<()>,
) -> anyhow::Result<Option<futures::future::BoxFuture<'static, anyhow::Result<()>>>> {
    let accepted_inventory = cx
        .try_global::<SimComponentHostGlobal>()
        .ok_or_else(|| anyhow::anyhow!("native Sim component host is unavailable"))?
        .accepted_inventory;
    cx.set_global(next_binding);
    if accepted_inventory {
        register_execution(profile, cx)?;
        return Ok(None);
    }
    // The subscriber needs this exact binding to validate the recovered candidate.
    // Pending inventory never supplies execution authority or queue eligibility.
    sim_ui::clear_native_execution_services(cx)?;
    Ok(ExtensionStore::try_global(cx)
        .map(|store| store.update(cx, |store, cx| store.reload(None, cx)).boxed()))
}

#[cfg(feature = "sim")]
fn fail_native_sim_activation(message: String, cx: &mut App) {
    clear_native_graphite_assets(cx);
    if cx.has_global::<NativeSimRuntimeBinding>() {
        let _removed_binding = cx.remove_global::<NativeSimRuntimeBinding>();
    }
    let message = match sim_ui::clear_native_execution_services(cx) {
        Ok(()) => message,
        Err(error) => format!("{message}; native execution shutdown failed: {error}"),
    };
    log::error!("{message}");
    sim_ui::set_initialization_error(message, cx);
}

#[cfg(all(feature = "sim", not(test)))]
fn active_native_sim_profile(cx: &App) -> anyhow::Result<sim_runtime::NativeRuntimeProfile> {
    let settings_store = cx
        .try_global::<SettingsStore>()
        .ok_or_else(|| anyhow::anyhow!("Zed settings store is unavailable"))?;
    active_native_sim_profile_from_settings(settings_store.merged_settings())
}

#[cfg(all(feature = "sim", not(test)))]
fn active_native_sim_configuration(
    cx: &App,
) -> anyhow::Result<(
    sim_runtime::NativeRuntimeProfile,
    sim_runtime::NativePluginSecurityPolicy,
)> {
    let settings_store = cx
        .try_global::<SettingsStore>()
        .ok_or_else(|| anyhow::anyhow!("Zed settings store is unavailable"))?;
    active_native_sim_configuration_from_settings(settings_store.merged_settings())
}

#[cfg(feature = "sim")]
fn rebind_native_sim_registry_event(
    selection: anyhow::Result<sim_runtime::NativeRuntimeProfile>,
    cx: &mut App,
    register_execution: impl FnOnce(&sim_runtime::NativeRuntimeProfile, &mut App) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let profile = selection?;
    cx.global_mut::<SimComponentHostGlobal>().accepted_inventory = true;
    if let Err(error) = register_execution(&profile, cx) {
        fail_native_sim_activation(
            format!("native Sim component lifecycle rebind failed: {error}"),
            cx,
        );
        return Err(error);
    }
    Ok(())
}

#[cfg(feature = "sim")]
fn native_sim_registry_event_profile(
    settings: &settings::SettingsContent,
    retained_binding: Option<&NativeSimRuntimeBinding>,
    host_profile_id: &str,
    host_security: &sim_runtime::NativePluginSecurityPolicy,
    received_bundle: &sim_runtime::NativeExecutionRegistryBundle,
    current_bundle: &sim_runtime::NativeExecutionRegistryBundle,
) -> anyhow::Result<sim_runtime::NativeRuntimeProfile> {
    let (profile, plugin_security) = active_native_sim_configuration_from_settings(settings)?;
    let binding = NativeSimRuntimeBinding::new(&profile, &plugin_security);
    anyhow::ensure!(
        retained_binding == Some(&binding),
        "native Sim registry event does not match the active runtime binding"
    );
    anyhow::ensure!(
        host_profile_id == profile.id.to_string() && host_security == &plugin_security,
        "native Sim registry event does not match the active component host"
    );
    anyhow::ensure!(
        received_bundle.profile_id().0 == profile.id
            && current_bundle.profile_id().0 == profile.id
            && received_bundle.identity_sha256() == current_bundle.identity_sha256(),
        "native Sim registry event is stale or belongs to another profile"
    );
    Ok(profile)
}

#[cfg(all(feature = "sim", not(test)))]
fn configured_native_sim_profile_id(cx: &App) -> Option<sim_types::ProfileId> {
    cx.try_global::<SettingsStore>()
        .and_then(|store| native_sim_profile_id_from_settings(store.merged_settings()))
}

#[cfg(feature = "sim")]
fn native_sim_profile_id_from_settings(
    settings: &settings::SettingsContent,
) -> Option<sim_types::ProfileId> {
    let fallback = default_native_sim_settings_content().ok();
    let runtime = settings
        .sim_runtime
        .as_ref()
        .or_else(|| fallback.as_ref()?.sim_runtime.as_ref())?;
    let active_profile = runtime.active_profile.as_deref()?;
    Uuid::parse_str(active_profile)
        .ok()
        .map(sim_types::ProfileId)
}

#[cfg(feature = "sim")]
fn default_native_sim_settings_content() -> anyhow::Result<settings::SettingsContent> {
    <settings::SettingsContent as settings::RootUserSettings>::parse_json_with_comments(
        include_str!("../../../assets/settings/default-sim.json"),
    )
}

#[cfg(feature = "sim")]
fn active_native_sim_configuration_from_settings(
    settings: &settings::SettingsContent,
) -> anyhow::Result<(
    sim_runtime::NativeRuntimeProfile,
    sim_runtime::NativePluginSecurityPolicy,
)> {
    let fallback = default_native_sim_settings_content()?;
    let content = settings
        .sim_runtime
        .as_ref()
        .or(fallback.sim_runtime.as_ref())
        .ok_or_else(|| anyhow::anyhow!("native Sim default settings are missing"))?;
    let settings = sim_runtime::parse_runtime_settings(content)?;
    let profile = settings
        .active_profile()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("native Sim active profile is unavailable"))?;
    let plugin_security = settings
        .active_plugin_security_policy()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("native Sim plugin security policy is unavailable"))?;
    Ok((profile, plugin_security))
}

#[cfg(feature = "sim")]
fn active_native_sim_profile_from_settings(
    settings: &settings::SettingsContent,
) -> anyhow::Result<sim_runtime::NativeRuntimeProfile> {
    active_native_sim_configuration_from_settings(settings).map(|(profile, _)| profile)
}

#[cfg(all(feature = "sim", not(test)))]
fn native_sim_worker_launch(
    profile: &sim_runtime::NativeRuntimeProfile,
    worker_id: sim_types::WorkerId,
) -> anyhow::Result<sim_runtime::WorkerLaunchConfig> {
    Ok(
        sim_runtime::WorkerLaunchConfig::for_packaged_worker_profile(
            profile,
            worker_id,
            sim_runtime::NATIVE_IMAGE_REGISTRY_VERSION,
            8 * 1024 * 1024 * 1024,
        )?,
    )
}

#[cfg(feature = "sim")]
struct AcceptedDesktopComponentDeployment {
    registry_bundle: Arc<sim_runtime::NativeExecutionRegistryBundle>,
    candidate_identity: extension_host::ComponentInventoryCandidateIdentity,
}

#[cfg(feature = "sim")]
fn accepted_desktop_component_registry_bundle(
    accepted_inventory: bool,
    candidate_identity: Option<extension_host::ComponentInventoryCandidateIdentity>,
    router: &sim_plugin_host::ComponentHostRouter,
) -> anyhow::Result<AcceptedDesktopComponentDeployment> {
    anyhow::ensure!(
        accepted_inventory,
        "native Sim component inventory has not produced an accepted candidate"
    );
    let candidate_identity = candidate_identity.ok_or_else(|| {
        anyhow::anyhow!("native Sim component inventory candidate identity is unavailable")
    })?;
    Ok(AcceptedDesktopComponentDeployment {
        registry_bundle: Arc::new(router.active_execution_registry_bundle()?),
        candidate_identity,
    })
}

#[cfg(feature = "sim")]
#[cfg_attr(
    test,
    expect(
        dead_code,
        reason = "desktop startup owns the concrete controller attachment transition"
    )
)]
fn activate_desktop_component_deployment<RollbackError>(
    deployment: AcceptedDesktopComponentDeployment,
    provider_worker_bridge: sim_runtime::NativeProviderWorkerBridgeAttachment,
    private_worker_executor: Arc<sim_plugin_host::PrivateWorkerPluginExecutor>,
    cx: &mut App,
    rollback_controller: impl FnOnce(&mut App) -> Result<(), RollbackError>,
) -> anyhow::Result<sim_api::NativeProviderDeploymentIdentity>
where
    RollbackError: std::fmt::Display,
{
    let provider_deployment = sim_api::NativeProviderDeploymentIdentity::from_registry_bundle(
        &deployment.registry_bundle,
        &deployment.candidate_identity,
    )?;
    if let Err(error) =
        private_worker_executor.attach_provider_worker_bridge(provider_worker_bridge)
    {
        if let Err(rollback_error) = rollback_controller(cx) {
            return Err(anyhow::anyhow!(
                "native provider bridge attachment failed: {error}; controller rollback failed: {rollback_error}"
            ));
        }
        return Err(anyhow::anyhow!(
            "native provider bridge attachment failed: {error}"
        ));
    }
    Ok(provider_deployment)
}

#[cfg(feature = "sim")]
fn ensure_native_sim_assets(
    profile: &sim_runtime::NativeRuntimeProfile,
    replace_assets: bool,
    profile_root: &std::path::Path,
    cx: &mut App,
) -> anyhow::Result<sim_runtime::SharedAssetService> {
    if let Some(services) = sim_ui::native_asset_services(cx)
        .filter(|services| !replace_assets && services.profile_id() == profile.id.to_string())
    {
        let assets = services.assets();
        bind_native_graphite_assets(assets.clone(), profile, cx)?;
        return Ok(assets);
    }
    let assets = sim_runtime::open_native_profile_asset_service(
        profile.id.to_string(),
        profile_root,
        &profile.model_roots,
    )?;
    sim_ui::register_native_asset_services(assets.clone(), cx)?;
    bind_native_graphite_assets(assets.clone(), profile, cx)?;
    Ok(assets)
}

#[cfg(feature = "sim")]
struct NativeProjectPublisher {
    pool: sim_runtime::ProjectOutputCommitterPool,
}

#[cfg(feature = "sim")]
impl gpui::Global for NativeProjectPublisher {}

#[cfg(feature = "sim")]
fn native_project_publisher(
    assets: &sim_runtime::SharedAssetService,
    cx: &mut App,
) -> anyhow::Result<sim_runtime::SharedOutputCommitter> {
    let roots = assets
        .lock()
        .map_err(|error| anyhow::anyhow!("The project asset service is unavailable: {error}"))?
        .roots()
        .clone();
    if cx.try_global::<NativeProjectPublisher>().is_none() {
        cx.set_global(NativeProjectPublisher {
            pool: sim_runtime::ProjectOutputCommitterPool::default(),
        });
    }
    Ok(cx
        .global_mut::<NativeProjectPublisher>()
        .pool
        .for_roots(roots)?)
}

#[cfg(feature = "sim")]
fn bind_native_graphite_assets(
    assets: sim_runtime::SharedAssetService,
    profile: &sim_runtime::NativeRuntimeProfile,
    cx: &mut App,
) -> anyhow::Result<()> {
    let authorization = sim_runtime::authorize_native_input_reader(profile.id.to_string())?;
    let policy = sim_runtime::PermissionPolicy::native_runtime_services(profile.id.to_string())?;
    let output_authorization = policy.authorize_one(
        sim_runtime::NATIVE_API_ASSET_READER_SUBJECT,
        sim_runtime::Capability::Asset {
            namespace: "output".to_owned(),
            action: sim_runtime::AssetOperation::Read,
        },
    )?;
    let committer = native_project_publisher(&assets, cx)?;
    let publisher_authorization =
        sim_runtime::authorize_native_output_committer(profile.id.to_string())?;
    graphics_ui::register_asset_binding_with_output_reader(
        assets,
        authorization,
        Some(output_authorization),
        cx,
    )?;
    graphics_ui::register_asset_publisher(committer, publisher_authorization, cx)?;
    Ok(())
}

#[cfg(feature = "sim")]
fn clear_native_graphite_assets(cx: &mut App) {
    if let Err(error) = graphics_ui::clear_asset_binding(cx) {
        log::error!("native Graphite asset shutdown failed: {error}");
    }
}

#[cfg(all(feature = "sim", not(test)))]
fn register_native_sim_execution(
    profile: &sim_runtime::NativeRuntimeProfile,
    cx: &mut App,
) -> anyhow::Result<()> {
    use sim_runtime::NativeExecutionControllerConfig;
    use sim_types::{ProfileId, WorkerId};

    let profile_id = ProfileId(profile.id);
    let assets = sim_ui::native_asset_services(cx)
        .filter(|services| services.profile_id() == profile_id.0.to_string())
        .ok_or_else(|| anyhow::anyhow!("native Sim asset service is unavailable"))?
        .assets();
    let mut worker = native_sim_worker_launch(profile, WorkerId(Uuid::new_v4()))?;
    let (deployment, provider_invocation_authority, private_worker_executor) = {
        let component_host = cx
            .try_global::<SimComponentHostGlobal>()
            .filter(|component_host| component_host.profile_id == profile_id.0.to_string())
            .ok_or_else(|| anyhow::anyhow!("native Sim component host is unavailable"))?;
        let candidate_identity = component_host
            .accepted_candidate_identity
            .lock()
            .map_err(|_| anyhow::anyhow!("desktop component candidate identity is unavailable"))?
            .clone();
        let deployment = accepted_desktop_component_registry_bundle(
            component_host.accepted_inventory,
            candidate_identity,
            &component_host.router,
        )?;
        (
            deployment,
            component_host.provider_invocation_authority.clone(),
            component_host.private_worker_executor.clone(),
        )
    };
    let registry_bundle = deployment.registry_bundle.clone();
    worker = worker.with_registry_deployment(registry_bundle.worker_deployment().clone());
    let presentation = sim_ui::execution_ui_model(cx)
        .ok_or_else(|| anyhow::anyhow!("native execution UI model is not initialized"))?
        .read(cx)
        .shared_service();
    let committer = native_project_publisher(&assets, cx)?;
    let mut config = NativeExecutionControllerConfig::new_with_output_committer(
        assets,
        presentation,
        worker,
        true,
        committer,
    )?
    .with_memory_policy(profile.memory_policy);
    if let Some(provider_registry) = registry_bundle.provider_registry() {
        config = config.with_provider_registry(provider_registry.clone())?;
        config =
            config.with_provider_invocation_authority(provider_invocation_authority.ok_or_else(
                || anyhow::anyhow!("native provider invocation authority is unavailable"),
            )?);
    }
    let provider_deployment = sim_api::NativeProviderDeploymentIdentity::from_registry_bundle(
        &deployment.registry_bundle,
        &deployment.candidate_identity,
    )?;
    if cx
        .try_global::<SimComponentHostGlobal>()
        .and_then(|component_host| component_host.active_provider_deployment.as_ref())
        == Some(&provider_deployment)
    {
        return Ok(());
    }
    let provider_worker_bridge =
        sim_ui::register_native_execution_services(config, registry_bundle, cx)?;
    let provider_deployment = activate_desktop_component_deployment(
        deployment,
        provider_worker_bridge,
        private_worker_executor,
        cx,
        sim_ui::clear_native_execution_services,
    )?;
    cx.global_mut::<SimComponentHostGlobal>()
        .active_provider_deployment = Some(provider_deployment);
    Ok(())
}

actions!(
    zed,
    [
        /// Opens the element inspector for debugging UI.
        DebugElements,
        /// Hides the application window.
        Hide,
        /// Hides all other application windows.
        HideOthers,
        /// Minimizes the current window.
        Minimize,
        /// Opens the default settings file.
        OpenDefaultSettings,
        /// Opens project-specific settings file.
        OpenProjectSettingsFile,
        /// Opens the tasks panel.
        OpenTasks,
        /// Opens debug tasks configuration.
        OpenDebugTasks,
        /// Shows the default semantic token rules (read-only).
        ShowDefaultSemanticTokenRules,
        /// Resets the application database.
        ResetDatabase,
        /// Shows all hidden windows.
        ShowAll,
        /// Toggles fullscreen mode.
        ToggleFullScreen,
        /// Zooms the window.
        Zoom,
        /// Triggers a test panic for debugging.
        TestPanic,
        /// Triggers a hard crash for debugging.
        TestCrash,
    ]
);

actions!(
    dev,
    [
        /// Opens a prompt to enter a URL to open.
        OpenUrlPrompt,
        /// Dumps the current accessibility tree (the last update sent to the
        /// platform adapter) to a new buffer as JSON, for debugging what is
        /// exposed to assistive technology.
        DumpAccessibilityTree,
        /// Copies the current accessibility tree to the clipboard as JSON,
        /// without opening a buffer. See [`DumpAccessibilityTree`].
        CopyAccessibilityTree,
    ]
);

/// Serializes the window's most recent accessibility tree to JSON for the
/// `dev: dump/copy accessibility tree` actions, falling back to a friendly
/// placeholder when no tree has been built yet.
fn accessibility_tree_dump(window: &Window) -> String {
    window.debug_a11y_tree_json().unwrap_or_else(|| {
        "No accessibility tree has been built yet. The tree is only \
         produced once assistive technology (e.g. a screen reader) is \
         active for this window."
            .to_string()
    })
}

#[cfg(debug_assertions)]
actions!(
    dev,
    [
        /// Show an error on the workspace level.
        ShowWorkspaceError
    ]
);

pub fn init(cx: &mut App) {
    #[cfg(feature = "sim")]
    {
        graphics_ui::init(cx);
        let lifecycle_registration = ensure_desktop_component_lifecycle_registration(cx);
        init_sim_ui(cx);
        match lifecycle_registration.and_then(|_| init_sim_component_host(false, cx)) {
            Err(error) => {
                let message = format!("native Sim component host initialization failed: {error}");
                log::error!("{message}");
                sim_ui::set_initialization_error(message, cx);
            }
            #[cfg(not(test))]
            Ok(()) => {
                if let Ok((profile, plugin_security)) = active_native_sim_configuration(cx) {
                    cx.set_global(NativeSimRuntimeBinding::new(&profile, &plugin_security));
                }
            }
            #[cfg(test)]
            Ok(()) => {}
        }
    }

    #[cfg(target_os = "macos")]
    cx.on_action(|_: &Hide, cx| cx.hide());
    #[cfg(target_os = "macos")]
    cx.on_action(|_: &HideOthers, cx| cx.hide_other_apps());
    #[cfg(target_os = "macos")]
    cx.on_action(|_: &ShowAll, cx| cx.unhide_other_apps());
    cx.on_action(quit);

    cx.on_action(|_: &RestoreBanner, cx| title_bar::restore_banner(cx));

    cx.observe_flag::<PanicFeatureFlag, _>({
        let mut added = false;
        move |flag, cx| {
            if added || !*flag {
                return;
            }
            added = true;
            cx.on_action(|_: &TestPanic, _| panic!("Ran the TestPanic action"))
                .on_action(|_: &TestCrash, _| {
                    unsafe extern "C" {
                        fn puts(s: *const i8);
                    }
                    unsafe {
                        puts(0xabad1d3a as *const i8);
                    }
                });
        }
    })
    .detach();

    // When Zed logs to stdout rather than the log file, avoid registering
    // handlers for both `OpenLog` and `RevealLogInFileManager`, as the log file
    // does not exist in that scenario and these actions would error.
    if !crate::stdout_is_a_pty() {
        cx.on_action(|_: &OpenLog, cx| {
            with_active_or_new_workspace(cx, |workspace, window, cx| {
                open_log_file(workspace, window, cx);
            });
        })
        .on_action(|_: &workspace::RevealLogInFileManager, cx| {
            cx.reveal_path(paths::log_file().as_path());
        });
    }

    cx.on_action(|_: &zed_actions::OpenLicenses, cx| {
        with_active_or_new_workspace(cx, |workspace, window, cx| {
            open_bundled_file(
                workspace,
                asset_str::<Assets>("licenses.md"),
                "Open Source License Attribution",
                "Markdown",
                window,
                cx,
            );
        });
    })
    .on_action(|&zed_actions::OpenKeymapFile, cx| {
        with_active_or_new_workspace(cx, |_, window, cx| {
            open_settings_file(
                paths::keymap_file(),
                || settings::initial_keymap_content().as_ref().into(),
                window,
                cx,
            );
        });
    })
    .on_action(|_: &OpenSettingsFile, cx| {
        with_active_or_new_workspace(cx, |_, window, cx| {
            open_settings_file(
                paths::settings_file(),
                || settings::initial_user_settings_content().as_ref().into(),
                window,
                cx,
            );
        });
    })
    .on_action(|_: &OpenAccountSettings, cx| {
        with_active_or_new_workspace(cx, |_, _, cx| {
            cx.open_url(&zed_urls::account_url(cx));
        });
    })
    .on_action(|_: &OpenTasks, cx| {
        with_active_or_new_workspace(cx, |_, window, cx| {
            open_settings_file(
                paths::tasks_file(),
                || settings::initial_tasks_content().as_ref().into(),
                window,
                cx,
            );
        });
    })
    .on_action(|_: &OpenDebugTasks, cx| {
        with_active_or_new_workspace(cx, |_, window, cx| {
            open_settings_file(
                paths::debug_scenarios_file(),
                || settings::initial_debug_tasks_content().as_ref().into(),
                window,
                cx,
            );
        });
    })
    .on_action(|_: &ShowDefaultSemanticTokenRules, cx| {
        with_active_or_new_workspace(cx, |workspace, window, cx| {
            open_bundled_file(
                workspace,
                settings::default_semantic_token_rules(),
                "Default Semantic Token Rules",
                "JSONC",
                window,
                cx,
            );
        });
    })
    .on_action(|_: &OpenDefaultSettings, cx| {
        with_active_or_new_workspace(cx, |workspace, window, cx| {
            open_bundled_file(
                workspace,
                settings::default_settings(),
                "Default Settings",
                "JSON",
                window,
                cx,
            );
        });
    })
    .on_action(|_: &zed_actions::OpenDefaultKeymap, cx| {
        with_active_or_new_workspace(cx, |workspace, window, cx| {
            open_bundled_file(
                workspace,
                settings::default_keymap(),
                "Default Key Bindings",
                "JSON",
                window,
                cx,
            );
        });
    })
    .on_action(|_: &About, cx| {
        open_about_window(cx);
    });
}

fn bind_on_window_closed(cx: &mut App) -> Option<gpui::Subscription> {
    #[cfg(target_os = "macos")]
    {
        WorkspaceSettings::get_global(cx)
            .on_last_window_closed
            .is_quit_app()
            .then(|| {
                cx.on_window_closed(|cx, _window_id| {
                    if cx.windows().is_empty() {
                        cx.quit();
                    }
                })
            })
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some(cx.on_window_closed(|cx, _window_id| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        }))
    }
}

pub fn build_window_options(display_uuid: Option<Uuid>, cx: &mut App) -> WindowOptions {
    let display = display_uuid.and_then(|uuid| {
        cx.displays()
            .into_iter()
            .find(|display| display.uuid().ok() == Some(uuid))
    });
    let app_id = ReleaseChannel::global(cx).app_id();
    let window_decorations = match std::env::var("ZED_WINDOW_DECORATIONS") {
        Ok(val) if val == "server" => gpui::WindowDecorations::Server,
        Ok(val) if val == "client" => gpui::WindowDecorations::Client,
        _ => match WorkspaceSettings::get_global(cx).window_decorations {
            settings::WindowDecorations::Server => gpui::WindowDecorations::Server,
            settings::WindowDecorations::Client => gpui::WindowDecorations::Client,
        },
    };

    let use_system_window_tabs = WorkspaceSettings::get_global(cx).use_system_window_tabs;

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    static APP_ICON: std::sync::LazyLock<Option<std::sync::Arc<image::RgbaImage>>> =
        std::sync::LazyLock::new(|| {
            // this shouldn't fail since decode is checked in build.rs
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/app_icon.png"));
            util::maybe!({
                let image = image::ImageReader::new(std::io::Cursor::new(BYTES))
                    .with_guessed_format()?
                    .decode()?
                    .into();
                anyhow::Ok(Arc::new(image))
            })
            .log_err()
        });

    WindowOptions {
        titlebar: Some(TitlebarOptions {
            title: None,
            appears_transparent: true,
            traffic_light_position: Some(point(px(9.0), px(9.0))),
        }),
        window_bounds: None,
        focus: false,
        show: false,
        kind: WindowKind::Normal,
        is_movable: true,
        // Zed draws its own titlebar and moves the window via [`Window::start_window_move`],
        // so on macOS AppKit should not own titlebar dragging. This avoids the titlebar
        // click delay from AppKit's drag disambiguation (first observed on macOS 27) while
        // keeping the window movable and the Window-menu tiling items enabled. No-op on
        // other platforms.
        app_owns_titlebar_drag: true,
        display_id: display.map(|display| display.id()),
        window_background: cx.theme().window_background_appearance(),
        app_id: Some(app_id.to_owned()),
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        icon: APP_ICON.as_ref().cloned(),
        window_decorations: Some(window_decorations),
        window_min_size: Some(gpui::Size {
            width: px(360.0),
            height: px(240.0),
        }),
        tabbing_identifier: if use_system_window_tabs {
            Some(String::from("zed"))
        } else {
            None
        },
        ..Default::default()
    }
}

pub fn initialize_workspace(app_state: Arc<AppState>, cx: &mut App) {
    let mut _on_close_subscription = bind_on_window_closed(cx);
    cx.observe_global::<SettingsStore>(move |cx| {
        // A 1.92 regression causes unused-assignment to trigger on this variable.
        _ = _on_close_subscription.is_some();
        _on_close_subscription = bind_on_window_closed(cx);
    })
    .detach();

    init_cursor_hide_mode(cx);
    init_app_appearance(cx);
    init_reduce_motion(cx);
    init_global_config_error_notifications(cx);

    cx.observe_new(|_multi_workspace: &mut MultiWorkspace, window, cx| {
        let Some(window) = window else {
            return;
        };

        #[cfg(feature = "track-project-leak")]
        {
            let multi_workspace_handle = cx.weak_entity();
            let workspace_handle = _multi_workspace.workspace().downgrade();
            let project_handle = _multi_workspace.workspace().read(cx).project().downgrade();
            let window_id_2 = window.window_handle().window_id();
            cx.on_window_closed(move |cx, window_id| {
                let multi_workspace_handle = multi_workspace_handle.clone();
                let workspace_handle = workspace_handle.clone();
                let project_handle = project_handle.clone();
                if window_id != window_id_2 {
                    return;
                }
                cx.spawn(async move |cx| {
                    cx.background_executor()
                        .timer(std::time::Duration::from_millis(1500))
                        .await;

                    multi_workspace_handle.assert_released();
                    workspace_handle.assert_released();
                    project_handle.assert_released();
                })
                .detach();
            })
            .detach();
        }

        cx.spawn_in(window, async move |_this, cx| {
            const TELEMETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5 * 60);
            loop {
                cx.background_executor().timer(TELEMETRY_INTERVAL).await;
                if cx
                    .update(|window, cx| {
                        input_latency_ui::report_input_latency_telemetry(window, cx);
                        input_latency_ui::report_frame_duration_telemetry(window, cx);
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();

        let multi_workspace_handle = cx.entity().downgrade();
        window.on_window_should_close(cx, move |window, cx| {
            multi_workspace_handle
                .update(cx, |multi_workspace, cx| {
                    // We'll handle closing asynchronously
                    multi_workspace.close_window(&CloseWindow, window, cx);
                    false
                })
                .unwrap_or(true)
        });

        #[cfg(feature = "agentic-tools")]
        {
            let window_handle = window.window_handle();
            let multi_workspace_handle = cx.entity();
            cx.subscribe_in(
                &multi_workspace_handle,
                window,
                |this, _multi_workspace, event: &workspace::MultiWorkspaceEvent, window, cx| {
                    let workspace::MultiWorkspaceEvent::ActiveWorkspaceChanged {
                        source_workspace,
                    } = event
                    else {
                        return;
                    };

                    let active_workspace = this.workspace().clone();
                    let source_workspace = source_workspace.clone();
                    active_workspace.update(cx, |workspace, cx| {
                        if let Some(ref source) = source_workspace
                            && let Some(panel) = workspace.panel::<agent_ui::AgentPanel>(cx)
                        {
                            panel.update(cx, |panel, cx| {
                                panel.initialize_from_source_workspace_if_needed(
                                    source.clone(),
                                    window,
                                    cx,
                                );
                            });
                        }

                        ensure_agent_panel_for_workspace(workspace, source_workspace, window, cx)
                            .detach_and_log_err(cx);
                    });
                },
            )
            .detach();

            cx.defer(move |cx| {
                window_handle
                    .update(cx, |_, window, cx| {
                        let sidebar =
                            cx.new(|cx| Sidebar::new(multi_workspace_handle.clone(), window, cx));
                        multi_workspace_handle.update(cx, |multi_workspace, cx| {
                            multi_workspace.register_sidebar(sidebar, cx);
                        });
                    })
                    .ok();
            });
        }
    })
    .detach();

    cx.observe_new(move |workspace: &mut Workspace, window, cx| {
        let Some(window) = window else {
            return;
        };

        let workspace_handle = cx.entity();
        #[cfg(feature = "multiplayer-tools")]
        let collaborative_review_state =
            Rc::new(RefCell::new(CollaborativeReviewCompositionState::default()));
        #[cfg(feature = "multiplayer-tools")]
        schedule_collaborative_project_review_reconciliation(
            workspace_handle.clone(),
            collaborative_review_state.clone(),
            window,
            cx,
        );
        #[cfg(feature = "multiplayer-tools")]
        if let Some(agent_panel) = workspace.panel::<AgentPanel>(cx) {
            subscribe_to_collaborative_review_agent_panel(
                &workspace_handle,
                agent_panel,
                &collaborative_review_state,
                window,
                cx,
            );
        }
        let center_pane = workspace.active_pane().clone();
        initialize_pane(workspace, &center_pane, window, cx);

        cx.subscribe_in(&workspace_handle, window, {
            #[cfg(feature = "multiplayer-tools")]
            let workspace_handle = workspace_handle.clone();
            #[cfg(feature = "multiplayer-tools")]
            let collaborative_review_state = collaborative_review_state.clone();
            move |workspace, _, event, window, cx| match event {
                workspace::Event::PaneAdded(pane) => {
                    initialize_pane(workspace, pane, window, cx);
                }
                #[cfg(feature = "multiplayer-tools")]
                workspace::Event::ItemAdded { .. } | workspace::Event::ItemRemoved { .. } => {
                    schedule_collaborative_project_review_reconciliation(
                        workspace_handle.clone(),
                        collaborative_review_state.clone(),
                        window,
                        cx,
                    );
                }
                #[cfg(feature = "multiplayer-tools")]
                workspace::Event::PanelAdded(panel) => {
                    if let Ok(agent_panel) = panel.clone().downcast::<AgentPanel>() {
                        subscribe_to_collaborative_review_agent_panel(
                            &workspace_handle,
                            agent_panel,
                            &collaborative_review_state,
                            window,
                            cx,
                        );
                    }
                }
                #[cfg(feature = "multiplayer-tools")]
                workspace::Event::PresentationChanged => {
                    if let Some(panel) = workspace.panel::<AgentPanel>(cx) {
                        AgentPanel::synchronize_collaborative_thread_navigation(
                            workspace, &panel, true, window, cx,
                        )
                        .log_err();
                    }
                }
                workspace::Event::OpenBundledFile {
                    text,
                    title,
                    language,
                } => open_bundled_file(workspace, text.clone(), title, language, window, cx),
                _ => {}
            }
        })
        .detach();

        #[cfg(not(any(test, target_os = "macos")))]
        initialize_file_watcher(window, cx);

        if let Some(specs) = window.gpu_specs() {
            log::info!("Using GPU: {:?}", specs);
            show_software_emulation_warning_if_needed(specs.clone(), window, cx);
            if let Some(crash_client) = cx.try_global::<CrashHandler>() {
                crashes::set_gpu_info(&crash_client.0, specs);
            }
        }

        crate::workspace_status::initialize(&app_state, workspace, window, cx);

        let panels_task = initialize_panels(window, cx);
        workspace.set_panels_task(panels_task);
        register_actions(app_state.clone(), workspace, window, cx);

        if !workspace.has_active_modal(window, cx) {
            workspace.focus_handle(cx).focus(window, cx);
        }
    })
    .detach();
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[allow(unused)]
fn initialize_file_watcher(window: &mut Window, cx: &mut Context<Workspace>) {
    if let Err(e) = fs::fs_watcher::global(|_| {}) {
        let message = format!(
            db::indoc! {r#"
            inotify_init returned {}

            This may be due to system-wide limits on inotify instances. For troubleshooting see: https://zed.dev/docs/linux
            "#},
            e
        );
        let prompt = window.prompt(
            PromptLevel::Critical,
            "Could not start inotify",
            Some(&message),
            &["Troubleshoot and Quit"],
            cx,
        );
        cx.spawn(async move |_, cx| {
            if prompt.await == Ok(0) {
                cx.update(|cx| {
                    cx.open_url("https://zed.dev/docs/linux#could-not-start-inotify");
                    cx.quit();
                });
            }
        })
        .detach()
    }
}

#[cfg(target_os = "windows")]
#[allow(unused)]
fn initialize_file_watcher(window: &mut Window, cx: &mut Context<Workspace>) {
    if let Err(e) = fs::fs_watcher::global(|_| {}) {
        let message = format!(
            db::indoc! {r#"
            ReadDirectoryChangesW initialization failed: {}

            This may occur on network filesystems and WSL paths. For troubleshooting see: https://zed.dev/docs/windows
            "#},
            e
        );
        let prompt = window.prompt(
            PromptLevel::Critical,
            "Could not start ReadDirectoryChangesW",
            Some(&message),
            &["Troubleshoot and Quit"],
            cx,
        );
        cx.spawn(async move |_, cx| {
            if prompt.await == Ok(0) {
                cx.update(|cx| {
                    cx.open_url("https://zed.dev/docs/windows");
                    cx.quit()
                });
            }
        })
        .detach()
    }
}

fn show_software_emulation_warning_if_needed(
    specs: gpui::GpuSpecs,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    if specs.is_software_emulated && std::env::var("ZED_ALLOW_EMULATED_GPU").is_err() {
        let (graphics_api, docs_url, open_url) = if cfg!(target_os = "windows") {
            (
                "DirectX",
                "https://zed.dev/docs/windows",
                "https://zed.dev/docs/windows",
            )
        } else {
            (
                "Vulkan",
                "https://zed.dev/docs/linux",
                "https://zed.dev/docs/linux#zed-fails-to-open-windows",
            )
        };
        let message = format!(
            db::indoc! {r#"
            Zed uses {} for rendering and requires a compatible GPU.

            Currently you are using a software emulated GPU ({}) which
            will result in awful performance.

            For troubleshooting see: {}
            Set ZED_ALLOW_EMULATED_GPU=1 env var to permanently override.
            "#},
            graphics_api, specs.device_name, docs_url
        );
        let prompt = window.prompt(
            PromptLevel::Critical,
            "Unsupported GPU",
            Some(&message),
            &["Skip", "Troubleshoot and Quit"],
            cx,
        );
        cx.spawn(async move |_, cx| {
            if prompt.await == Ok(1) {
                cx.update(|cx| {
                    cx.open_url(open_url);
                    cx.quit();
                });
            }
        })
        .detach()
    }
}

fn initialize_panels(window: &mut Window, cx: &mut Context<Workspace>) -> Task<anyhow::Result<()>> {
    cx.spawn_in(window, async move |workspace_handle, cx| {
        let project_panel = ProjectPanel::load(workspace_handle.clone(), cx.clone());
        let outline_panel = OutlinePanel::load(workspace_handle.clone(), cx.clone());
        #[cfg(feature = "rust-tools")]
        let cargo_panel = CargoPanel::load(workspace_handle.clone(), cx.clone());
        #[cfg(feature = "jvm-tools")]
        let dependencies_panel = DependenciesPanel::load(workspace_handle.clone(), cx.clone());
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        let tests_panel = TestsPanel::load(workspace_handle.clone(), cx.clone());
        let terminal_panel = TerminalPanel::load(workspace_handle.clone(), cx.clone());
        let git_panel = GitPanel::load(workspace_handle.clone(), cx.clone());
        let channels_panel =
            collab_ui::collab_panel::CollabPanel::load(workspace_handle.clone(), cx.clone());
        let debug_panel = DebugPanel::load(workspace_handle.clone(), cx);

        async fn add_panel_when_ready(
            panel_task: impl Future<Output = anyhow::Result<Entity<impl workspace::Panel>>> + 'static,
            workspace_handle: WeakEntity<Workspace>,
            mut cx: gpui::AsyncWindowContext,
        ) {
            if let Some(panel) = panel_task.await.context("failed to load panel").log_err()
            {
                workspace_handle
                    .update_in(&mut cx, |workspace, window, cx| {
                        workspace.add_panel(panel, window, cx);
                    })
                    .log_err();
            }
        }

        #[cfg(feature = "rust-tools")]
        let cargo_panel_task =
            add_panel_when_ready(cargo_panel, workspace_handle.clone(), cx.clone());
        #[cfg(not(feature = "rust-tools"))]
        let cargo_panel_task = std::future::ready(());
        #[cfg(feature = "jvm-tools")]
        let dependencies_panel_task =
            add_panel_when_ready(dependencies_panel, workspace_handle.clone(), cx.clone());
        #[cfg(not(feature = "jvm-tools"))]
        let dependencies_panel_task = std::future::ready(());
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        let tests_panel_task =
            add_panel_when_ready(tests_panel, workspace_handle.clone(), cx.clone());
        #[cfg(not(any(feature = "rust-tools", feature = "jvm-tools")))]
        let tests_panel_task = std::future::ready(());

        #[cfg(feature = "sim")]
        let sim_workspace_handle = workspace_handle.clone();
        let sim_panels = async {
            #[cfg(feature = "sim")]
            {
                let mut execution_context = cx.clone();
                let execution_panel = sim_ui::ExecutionPanel::load(
                    sim_workspace_handle.clone(),
                    &mut execution_context,
                );
                let mut properties_context = cx.clone();
                let graph_properties_panel = sim_ui::GraphPropertiesPanel::load(
                    sim_workspace_handle.clone(),
                    &mut properties_context,
                );
                futures::join!(
                    add_panel_when_ready(
                        execution_panel,
                        sim_workspace_handle.clone(),
                        cx.clone()
                    ),
                    add_panel_when_ready(
                        graph_properties_panel,
                        sim_workspace_handle.clone(),
                        cx.clone()
                    ),
                );
            }
        };

        futures::join!(
            add_panel_when_ready(project_panel, workspace_handle.clone(), cx.clone()),
            add_panel_when_ready(outline_panel, workspace_handle.clone(), cx.clone()),
            cargo_panel_task,
            dependencies_panel_task,
            tests_panel_task,
            add_panel_when_ready(terminal_panel, workspace_handle.clone(), cx.clone()),
            add_panel_when_ready(git_panel, workspace_handle.clone(), cx.clone()),
            add_panel_when_ready(channels_panel, workspace_handle.clone(), cx.clone()),
            add_panel_when_ready(debug_panel, workspace_handle.clone(), cx.clone()),
            sim_panels,
        );

        #[cfg(feature = "agentic-tools")]
        initialize_agent_panel(workspace_handle, cx.clone())
            .await
            .log_err();

        anyhow::Ok(())
    })
}

#[cfg(feature = "agentic-tools")]
fn setup_or_teardown_ai_panel<P: Panel>(
    workspace: &mut Workspace,
    window: &mut Window,
    cx: &mut Context<Workspace>,
    load_panel: impl FnOnce(
        WeakEntity<Workspace>,
        AsyncWindowContext,
    ) -> Task<anyhow::Result<Entity<P>>>
    + 'static,
) -> Task<anyhow::Result<()>> {
    let disable_ai = SettingsStore::global(cx)
        .get::<DisableAiSettings>(None)
        .disable_ai;
    let existing_panel = workspace.panel::<P>(cx);
    // Tests mount explicit panels; suppress automatic loading without removing those owners.
    match (disable_ai, existing_panel) {
        (false, None) if !cfg!(test) => cx.spawn_in(window, async move |workspace, cx| {
            let panel = load_panel(workspace.clone(), cx.clone()).await?;
            workspace.update_in(cx, |workspace, window, cx| {
                let disable_ai = SettingsStore::global(cx)
                    .get::<DisableAiSettings>(None)
                    .disable_ai;
                let have_panel = workspace.panel::<P>(cx).is_some();
                if !disable_ai && !have_panel {
                    workspace.add_panel(panel, window, cx);
                }
            })
        }),
        (true, Some(existing_panel)) => {
            workspace.remove_panel::<P>(&existing_panel, window, cx);
            Task::ready(Ok(()))
        }
        _ => Task::ready(Ok(())),
    }
}

#[cfg(feature = "agentic-tools")]
fn ensure_agent_panel_for_workspace(
    workspace: &mut Workspace,
    source_workspace: Option<WeakEntity<Workspace>>,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) -> Task<anyhow::Result<()>> {
    let task = setup_or_teardown_ai_panel(workspace, window, cx, move |workspace, cx| {
        agent_ui::AgentPanel::load(workspace, cx)
    });

    cx.spawn_in(window, async move |workspace, cx| {
        task.await?;
        workspace.update_in(cx, |workspace, window, cx| {
            if let Some(source_workspace) = source_workspace.clone()
                && let Some(panel) = workspace.panel::<agent_ui::AgentPanel>(cx)
            {
                panel.update(cx, |panel, cx| {
                    panel.initialize_from_source_workspace_if_needed(source_workspace, window, cx);
                });
            }
        })
    })
}

#[cfg(feature = "agentic-tools")]
async fn initialize_agent_panel(
    workspace_handle: WeakEntity<Workspace>,
    mut cx: AsyncWindowContext,
) -> anyhow::Result<()> {
    workspace_handle
        .update_in(&mut cx, |workspace, window, cx| {
            ensure_agent_panel_for_workspace(workspace, None, window, cx)
        })?
        .await?;

    workspace_handle.update_in(&mut cx, |workspace, window, cx| {
        cx.observe_global_in::<SettingsStore>(window, move |workspace, window, cx| {
            ensure_agent_panel_for_workspace(workspace, None, window, cx).detach_and_log_err(cx);
        })
        .detach();

        // Register the actions that are shared between `assistant` and `assistant2`.
        //
        // We need to do this here instead of within the individual `init`
        // functions so that we only register the actions once.
        //
        // Once we ship `assistant2` we can push this back down into `agent::agent_panel::init`.
        if !cfg!(test) {
            workspace
                .register_action(agent_ui::AgentPanel::toggle_focus)
                .register_action(agent_ui::AgentPanel::focus)
                .register_action(agent_ui::AgentPanel::toggle)
                .register_action(agent_ui::InlineAssistant::inline_assist);
        }
    })?;

    anyhow::Ok(())
}

fn register_actions(
    app_state: Arc<AppState>,
    workspace: &mut Workspace,
    _: &mut Window,
    cx: &mut Context<Workspace>,
) {
    workspace
        .register_action(|_, _: &OpenDocs, _, cx| cx.open_url(DOCS_URL))
        .register_action(|_, _: &OpenStatusPage, _, cx| cx.open_url(STATUS_URL))
        .register_action(|_, _: &GetMerch, _, cx| cx.open_url(MERCH_URL))
        .register_action(
            |workspace: &mut Workspace,
             _: &input_latency_ui::DumpInputLatencyHistogram,
             window: &mut Window,
             cx: &mut Context<Workspace>| {
                let project = workspace.project().clone();
                // In a collab session the report buffer is visible to other
                // participants, so attribute the data to this user's machine.
                let reported_by = if project.read(cx).is_shared()
                    || project.read(cx).is_via_collab()
                {
                    workspace
                        .user_store()
                        .read(cx)
                        .current_user()
                        .map(|user| user.username.to_string())
                } else {
                    None
                };
                let report_data = input_latency_ui::snapshot_input_latency_report(
                    window,
                    reported_by,
                    cx,
                );
                cx.spawn_in(window, async move |workspace, cx| {
                    let report = cx
                        .background_spawn(async move {
                            input_latency_ui::format_input_latency_report(&report_data)
                        })
                        .await;
                    let buffer = project
                        .update(cx, |project, cx| project.create_buffer(None, true, cx))
                        .await?;
                    buffer.update(cx, |buffer, cx| {
                        buffer.set_text(report, cx);
                    });
                    workspace.update_in(cx, |workspace, window, cx| {
                        let editor = cx
                            .new(|cx| Editor::for_buffer(buffer, Some(project), window, cx));
                        workspace
                            .add_item_to_active_pane(Box::new(editor), None, true, window, cx);
                    })
                })
                .detach_and_log_err(cx);
            },
        )
        .register_action(
            |workspace: &mut Workspace,
             _: &DumpAccessibilityTree,
             window: &mut Window,
             cx: &mut Context<Workspace>| {
                let json = accessibility_tree_dump(window);
                let language = workspace.app_state().languages.language_for_name("JSON");
                let project = workspace.project().clone();
                cx.spawn_in(window, async move |workspace, cx| {
                    let language = language.await.log_err();
                    let buffer = project
                        .update(cx, |project, cx| {
                            project.create_buffer(language, true, cx)
                        })
                        .await?;
                    buffer.update(cx, |buffer, cx| {
                        buffer.set_text(json, cx);
                    });
                    workspace.update_in(cx, |workspace, window, cx| {
                        let title = "Accessibility Tree".to_string();
                        let buffer = cx.new(|cx| {
                            MultiBuffer::singleton(buffer, cx).with_title(title.clone())
                        });
                        let editor = cx.new(|cx| {
                            let mut editor = Editor::for_multibuffer(
                                buffer,
                                Some(project),
                                window,
                                cx,
                            );
                            editor.set_breadcrumb_header(title);
                            editor
                        });
                        workspace.add_item_to_active_pane(
                            Box::new(editor),
                            None,
                            true,
                            window,
                            cx,
                        );
                    })
                })
                .detach_and_log_err(cx);
            },
        )
        .register_action(
            |_workspace: &mut Workspace,
             _: &CopyAccessibilityTree,
             window: &mut Window,
             cx: &mut Context<Workspace>| {
                let json = accessibility_tree_dump(window);
                cx.write_to_clipboard(ClipboardItem::new_string(json));
            },
        )
        .register_action(|_, _: &Minimize, window, _| {
            window.minimize_window();
        })
        .register_action(|_, _: &Zoom, window, _| {
            window.zoom_window();
        })
        .register_action(|_, _: &ToggleFullScreen, window, _| {
            window.toggle_fullscreen();
        })
        .register_action(|_, action: &OpenZedUrl, _, cx| {
            OpenListener::global(cx).open(RawOpenRequest {
                urls: vec![String::from(&*action.url)],
                ..Default::default()
            })
        })
        .register_action(|workspace, _: &OpenUrlPrompt, window, cx| {
            workspace.toggle_modal(window, cx, |window, cx| {
                open_url_modal::OpenUrlModal::new(window, cx)
            });
        })
        .register_action(|workspace, action: &OpenBrowser, _window, cx| {
            // Parse and validate the URL to ensure it's properly formatted
            match url::Url::parse(&action.url) {
                Ok(parsed_url) => {
                    // Use the parsed URL's string representation which is properly escaped
                    cx.open_url(parsed_url.as_str());
                }
                Err(e) => {
                    workspace.show_error(
                        format!(
                            "Opening this URL in a browser failed because the URL is invalid: {}\n\nError was: {e}",
                            action.url
                        ),
                        cx,
                    );
                }
            }
        })
        .register_action(|workspace, action: &workspace::Open, window, cx| {
            telemetry::event!("Project Opened");
            workspace::prompt_for_open_path_and_open(
                workspace,
                workspace.app_state().clone(),
                PathPromptOptions {
                    files: true,
                    directories: true,
                    multiple: true,
                    prompt: None,
                },
                action.create_new_window.unwrap_or_else(|| {
                    matches!(
                        WorkspaceSettings::get_global(cx).default_open_behavior,
                        DefaultOpenBehavior::NewWindow
                    )
                }),
                window,
                cx,
            );
        })
        .register_action(|workspace, _: &workspace::OpenFiles, window, cx| {
            let directories = cx.can_select_mixed_files_and_dirs();
            workspace::prompt_for_open_path_and_open(
                workspace,
                workspace.app_state().clone(),
                PathPromptOptions {
                    files: true,
                    directories,
                    multiple: true,
                    prompt: None,
                },
                true,
                window,
                cx,
            );
        })
        .register_action(|workspace, action: &zed_actions::OpenRemote, window, cx| {
            if !action.from_existing_connection {
                cx.propagate();
                return;
            }
            // You need existing remote connection to open it this way
            if workspace.project().read(cx).is_local() {
                return;
            }
            let create_new_window = action.create_new_window.unwrap_or_else(|| {
                matches!(
                    WorkspaceSettings::get_global(cx).default_open_behavior,
                    DefaultOpenBehavior::NewWindow
                )
            });
            telemetry::event!("Project Opened");
            let paths = workspace.prompt_for_open_path(
                PathPromptOptions {
                    files: true,
                    directories: true,
                    multiple: true,
                    prompt: None,
                },
                DirectoryLister::Project(workspace.project().clone()),
                window,
                cx,
            );
            cx.spawn_in(window, async move |this, cx| {
                let Some(paths) = paths.await.log_err().flatten() else {
                    return;
                };
                if let Some(task) = this
                    .update_in(cx, |this, window, cx| {
                        open_new_ssh_project_from_project(
                            this,
                            paths,
                            create_new_window,
                            window,
                            cx,
                        )
                    })
                    .log_err()
                {
                    task.await.log_err();
                }
            })
            .detach()
        })
        .register_action({
            let fs = app_state.fs.clone();
            move |_, action: &zed_actions::IncreaseUiFontSize, _window, cx| {
                if action.persist {
                    update_settings_file(fs.clone(), cx, move |settings, cx| {
                        let ui_font_size = ThemeSettings::get_global(cx).ui_font_size(cx) + px(1.0);
                        let _ = settings
                            .theme
                            .ui_font_size
                            .insert(f32::from(theme_settings::clamp_font_size(ui_font_size)).into());
                    });
                } else {
                    theme_settings::adjust_ui_font_size(cx, |size| size + px(1.0));
                }
            }
        })
        .register_action({
            let fs = app_state.fs.clone();
            move |_, action: &zed_actions::DecreaseUiFontSize, _window, cx| {
                if action.persist {
                    update_settings_file(fs.clone(), cx, move |settings, cx| {
                        let ui_font_size = ThemeSettings::get_global(cx).ui_font_size(cx) - px(1.0);
                        let _ = settings
                            .theme
                            .ui_font_size
                            .insert(f32::from(theme_settings::clamp_font_size(ui_font_size)).into());
                    });
                } else {
                    theme_settings::adjust_ui_font_size(cx, |size| size - px(1.0));
                }
            }
        })
        .register_action({
            let fs = app_state.fs.clone();
            move |_, action: &zed_actions::ResetUiFontSize, _window, cx| {
                if action.persist {
                    update_settings_file(fs.clone(), cx, move |settings, _| {
                        settings.theme.ui_font_size = None;
                    });
                } else {
                    theme_settings::reset_ui_font_size(cx);
                }
            }
        })
        .register_action({
            let fs = app_state.fs.clone();
            move |_, action: &zed_actions::IncreaseBufferFontSize, _window, cx| {
                if action.persist {
                    update_settings_file(fs.clone(), cx, move |settings, cx| {
                        let buffer_font_size =
                            ThemeSettings::get_global(cx).buffer_font_size(cx) + px(1.0);
                        let _ = settings
                            .theme
                            .buffer_font_size
                            .insert(f32::from(theme_settings::clamp_font_size(buffer_font_size)).into());
                    });
                } else {
                    theme_settings::increase_buffer_font_size(cx);
                }
            }
        })
        .register_action({
            let fs = app_state.fs.clone();
            move |_, action: &zed_actions::DecreaseBufferFontSize, _window, cx| {
                if action.persist {
                    update_settings_file(fs.clone(), cx, move |settings, cx| {
                        let buffer_font_size =
                            ThemeSettings::get_global(cx).buffer_font_size(cx) - px(1.0);
                        let _ = settings
                            .theme
                            .buffer_font_size
                            .insert(f32::from(theme_settings::clamp_font_size(buffer_font_size)).into());
                    });
                } else {
                    theme_settings::decrease_buffer_font_size(cx);
                }
            }
        })
        .register_action({
            let fs = app_state.fs.clone();
            move |_, action: &zed_actions::ResetBufferFontSize, _window, cx| {
                if action.persist {
                    update_settings_file(fs.clone(), cx, move |settings, _| {
                        settings.theme.buffer_font_size = None;
                    });
                } else {
                    theme_settings::reset_buffer_font_size(cx);
                }
            }
        })
        .register_action({
            let fs = app_state.fs.clone();
            move |_, action: &zed_actions::ResetAllZoom, _window, cx| {
                if action.persist {
                    update_settings_file(fs.clone(), cx, move |settings, _| {
                        settings.theme.ui_font_size = None;
                        settings.theme.buffer_font_size = None;
                        settings.theme.agent_ui_font_size = None;
                        settings.theme.agent_buffer_font_size = None;
                    });
                } else {
                    theme_settings::reset_ui_font_size(cx);
                    theme_settings::reset_buffer_font_size(cx);
                    theme_settings::reset_agent_ui_font_size(cx);
                    theme_settings::reset_agent_buffer_font_size(cx);
                }
            }
        })
        .register_action(|_, _: &install_cli::RegisterZedScheme, window, cx| {
            cx.spawn_in(window, async move |workspace, cx| {
                install_cli::register_zed_scheme(cx).await?;
                workspace.update_in(cx, |workspace, _, cx| {
                    struct RegisterZedScheme;

                    workspace.show_toast(
                        Toast::new(
                            NotificationId::unique::<RegisterZedScheme>(),
                            format!(
                                "{} links will now open in {}.",
                                product_flavor::URL_PREFIX,
                                ReleaseChannel::global(cx).display_name()
                            ),
                        ),
                        cx,
                    )
                })?;
                Ok(())
            })
            .detach_and_prompt_err(
                product_flavor::REGISTER_SCHEME_ERROR_TITLE,
                window,
                cx,
                |_, _, _| None,
            );
        })
        .register_action(open_project_settings_file)
        .register_action(open_project_tasks_file)
        .register_action(open_worktree_setup_tasks_file)
        .register_action(open_project_debug_tasks_file)
        .register_action(
            |workspace: &mut Workspace,
             _: &zed_actions::project_panel::ToggleFocus,
             window: &mut Window,
             cx: &mut Context<Workspace>| {
                workspace.toggle_panel_focus::<ProjectPanel>(window, cx);
            },
        )
        .register_action(
            |workspace: &mut Workspace,
             _: &outline_panel::ToggleFocus,
             window: &mut Window,
             cx: &mut Context<Workspace>| {
                workspace.toggle_panel_focus::<OutlinePanel>(window, cx);
            },
        )
        .register_action(
            |workspace: &mut Workspace,
             _: &collab_ui::collab_panel::ToggleFocus,
             window: &mut Window,
             cx: &mut Context<Workspace>| {
                workspace.toggle_panel_focus::<collab_ui::collab_panel::CollabPanel>(window, cx);
            },
        )
        .register_action(
            |workspace: &mut Workspace,
             _: &terminal_panel::ToggleFocus,
             window: &mut Window,
             cx: &mut Context<Workspace>| {
                workspace.toggle_panel_focus::<TerminalPanel>(window, cx);
            },
        )
        .register_action({
            let app_state = app_state.clone();
            move |_, _: &NewWindow, _, cx| {
                open_new(
                    Default::default(),
                    app_state.clone(),
                    cx,
                    |workspace, window, cx| {
                        cx.activate(true);
                        // Create buffer synchronously to avoid flicker
                        let project = workspace.project().clone();
                        let buffer = project.update(cx, |project, cx| {
                            project.create_local_buffer("", None, true, cx)
                        });
                        let editor = cx.new(|cx| {
                            Editor::for_buffer(buffer, Some(project), window, cx)
                        });
                        workspace.add_item_to_active_pane(
                            Box::new(editor),
                            None,
                            true,
                            window,
                            cx,
                        );
                    },
                )
                .detach();
            }
        })
        .register_action({
            move |workspace, _: &CloseProject, window, cx| {
                let Some(window_handle) = window.window_handle().downcast::<MultiWorkspace>() else {
                    return;
                };
                let old_group_key = workspace.project_group_key(cx);
                cx.spawn_in(window, async move |_, cx| {
                    let task = window_handle.update(cx, |multi_workspace, window, cx| {
                        multi_workspace.remove_project_group(&old_group_key, window, cx)
                    })?;
                    task.await?;
                    anyhow::Ok(())
                })
                .detach_and_log_err(cx);
            }
        })
        .register_action({
            let app_state = app_state.clone();
            move |_, _: &NewFile, _, cx| {
                open_new(
                    Default::default(),
                    app_state.clone(),
                    cx,
                    |workspace, window, cx| {
                        Editor::new_file(workspace, &Default::default(), window, cx)
                    },
                )
                .detach_and_log_err(cx);
            }
        });

    #[cfg(not(target_os = "windows"))]
    workspace.register_action(install_cli);

    if workspace.project().read(cx).is_via_remote_server() {
        workspace.register_action({
            move |workspace, _: &OpenServerSettings, window, cx| {
                let open_server_settings = workspace
                    .project()
                    .update(cx, |project, cx| project.open_server_settings(cx));

                cx.spawn_in(window, async move |workspace, cx| {
                    let buffer = open_server_settings.await?;

                    workspace
                        .update_in(cx, |workspace, window, cx| {
                            workspace.open_path(
                                buffer
                                    .read(cx)
                                    .project_path(cx)
                                    .expect("Settings file must have a location"),
                                None,
                                true,
                                window,
                                cx,
                            )
                        })?
                        .await?;

                    anyhow::Ok(())
                })
                .detach_and_log_err(cx);
            }
        });
    }

    #[cfg(feature = "agentic-tools")]
    workspace.register_action(sidebar::dump_workspace_info);

    #[cfg(debug_assertions)]
    workspace.register_action(|workspace, _: &ShowWorkspaceError, _, cx| {
        struct DebugError;
        struct SecondDebugError;

        impl WorkspaceError for DebugError {
            fn primary_message(&self) -> SharedString {
                SharedString::new_static(
                    "Error: Prepare rename via rust-analyzer failed: No references found at position",
                )
            }

            fn severity(&self) -> ErrorSeverity {
                ErrorSeverity::Warning
            }

            fn primary_action(&self) -> ErrorAction {
                ErrorAction::dismiss()
            }
        }

        impl WorkspaceError for SecondDebugError {
            fn primary_message(&self) -> SharedString {
                SharedString::new_static("This is some error to ignore.")
            }

            fn severity(&self) -> ErrorSeverity {
                ErrorSeverity::Error
            }

            fn primary_action(&self) -> ErrorAction {
                ErrorAction::dismiss()
            }
        }

        workspace.show_error(DebugError, cx);
        workspace.show_error(SecondDebugError, cx);
    });
}

fn initialize_pane(
    workspace: &Workspace,
    pane: &Entity<Pane>,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let workspace_handle = cx.weak_entity();
    pane.update(cx, |pane, cx| {
        pane.toolbar().update(cx, |toolbar, cx| {
            let multibuffer_hint = cx.new(|_| MultibufferHint::new());
            toolbar.add_item(multibuffer_hint, window, cx);
            let solo_diff_style_toolbar = cx.new(SoloDiffStyleToolbar::new);
            toolbar.add_item(solo_diff_style_toolbar, window, cx);
            let breadcrumbs = cx.new(|_| Breadcrumbs::new());
            toolbar.add_item(breadcrumbs, window, cx);
            let buffer_search_bar = cx.new(|cx| {
                search::BufferSearchBar::new(
                    Some(workspace.project().read(cx).languages().clone()),
                    window,
                    cx,
                )
            });
            toolbar.add_item(buffer_search_bar.clone(), window, cx);
            let quick_action_bar =
                cx.new(|cx| QuickActionBar::new(buffer_search_bar, workspace, cx));
            toolbar.add_item(quick_action_bar, window, cx);
            let diagnostic_editor_controls = cx.new(|_| diagnostics::ToolbarControls::new());
            toolbar.add_item(diagnostic_editor_controls, window, cx);
            let project_search_bar = cx.new(|_| ProjectSearchBar::new());
            toolbar.add_item(project_search_bar, window, cx);
            let lsp_log_item = cx.new(|_| LspLogToolbarItemView::new());
            toolbar.add_item(lsp_log_item, window, cx);
            let dap_log_item = cx.new(|_| debugger_tools::DapLogToolbarItemView::new());
            toolbar.add_item(dap_log_item, window, cx);
            #[cfg(feature = "agentic-tools")]
            {
                let acp_tools_item = cx.new(|_| acp_tools::AcpToolsToolbarItemView::new());
                toolbar.add_item(acp_tools_item, window, cx);
            }
            let telemetry_log_item =
                cx.new(|cx| telemetry_log::TelemetryLogToolbarItemView::new(window, cx));
            toolbar.add_item(telemetry_log_item, window, cx);
            let syntax_tree_item = cx.new(|_| language_tools::SyntaxTreeToolbarItemView::new());
            toolbar.add_item(syntax_tree_item, window, cx);
            let migration_banner =
                cx.new(|inner_cx| MigrationBanner::new(workspace_handle.clone(), inner_cx));
            toolbar.add_item(migration_banner, window, cx);
            let highlights_tree_item =
                cx.new(|_| language_tools::HighlightsTreeToolbarItemView::new());
            toolbar.add_item(highlights_tree_item, window, cx);
            let project_diff_toolbar = cx.new(|cx| ProjectDiffToolbar::new(workspace, cx));
            toolbar.add_item(project_diff_toolbar, window, cx);
            let staged_diff_toolbar = cx.new(|cx| StagedDiffToolbar::new(workspace, cx));
            toolbar.add_item(staged_diff_toolbar, window, cx);
            let unstaged_diff_toolbar = cx.new(|cx| UnstagedDiffToolbar::new(workspace, cx));
            toolbar.add_item(unstaged_diff_toolbar, window, cx);
            let branch_diff_toolbar = cx.new(BranchDiffToolbar::new);
            toolbar.add_item(branch_diff_toolbar, window, cx);
            let solo_diff_git_toolbar = cx.new(SoloDiffGitToolbar::new);
            toolbar.add_item(solo_diff_git_toolbar, window, cx);
            let commit_view_toolbar = cx.new(|_| CommitViewToolbar::new());
            toolbar.add_item(commit_view_toolbar, window, cx);
            #[cfg(feature = "agentic-tools")]
            {
                let agent_diff_toolbar = cx.new(AgentDiffToolbar::new);
                toolbar.add_item(agent_diff_toolbar, window, cx);
            }
            let basedpyright_banner = cx.new(|cx| BasedPyrightBanner::new(workspace, cx));
            toolbar.add_item(basedpyright_banner, window, cx);
            let image_view_toolbar = cx.new(|_| image_viewer::ImageViewToolbarControls::new());
            toolbar.add_item(image_view_toolbar, window, cx);
        })
    });
}

fn open_about_window(cx: &mut App) {
    struct AboutWindow {
        focus_handle: FocusHandle,
        ok_entry: NavigableEntry,
        copy_entry: NavigableEntry,
        message: SharedString,
        commit: Option<SharedString>,
        full_version: SharedString,
    }

    impl AboutWindow {
        fn new(cx: &mut Context<Self>) -> Self {
            let release_channel = ReleaseChannel::global(cx);
            let release_channel_name = release_channel.display_name();
            let mut version = AppVersion::global(cx);
            let full_version: SharedString = version.to_string().into();
            version.build = Default::default();

            let debug = if cfg!(debug_assertions) {
                "(debug)"
            } else {
                ""
            };
            let message: SharedString = format!("{release_channel_name} {version} {debug}").into();
            let commit = AppCommitSha::try_global(cx)
                .map(|sha| sha.full())
                .filter(|commit| !commit.is_empty())
                .map(SharedString::from);

            Self {
                focus_handle: cx.focus_handle(),
                ok_entry: NavigableEntry::focusable(cx),
                copy_entry: NavigableEntry::focusable(cx),
                message,
                commit,
                full_version,
            }
        }

        fn copy_details(&self, window: &mut Window, cx: &mut Context<Self>) {
            let content = match self.commit.as_ref() {
                Some(commit) => {
                    format!(
                        "{}\nCommit: {}\nVersion: {}",
                        self.message, commit, self.full_version
                    )
                }
                None => format!("{}\nVersion: {}", self.message, self.full_version),
            };
            cx.write_to_clipboard(ClipboardItem::new_string(content));
            window.remove_window();
        }
    }

    impl Render for AboutWindow {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let ok_is_focused = self.ok_entry.focus_handle.contains_focused(window, cx);
            let copy_is_focused = self.copy_entry.focus_handle.contains_focused(window, cx);

            Navigable::new(
                v_flex()
                    .id("about-window")
                    .track_focus(&self.focus_handle)
                    .on_action(cx.listener(|_, _: &menu::Cancel, window, _cx| {
                        window.remove_window();
                    }))
                    .min_w_0()
                    .size_full()
                    .bg(cx.theme().colors().editor_background)
                    .text_color(cx.theme().colors().text)
                    .p_4()
                    .when(cfg!(target_os = "macos"), |this| this.pt_10())
                    .gap_4()
                    .text_center()
                    .justify_between()
                    .child(
                        v_flex()
                            .w_full()
                            .gap_2()
                            .items_center()
                            .child(img(product_flavor::UI_LOGO_ASSET).size_16().flex_none())
                            .child(Headline::new(self.message.clone()))
                            .when_some(self.commit.clone(), |this, commit| {
                                this.child(
                                    Label::new("Commit")
                                        .color(Color::Muted)
                                        .size(LabelSize::XSmall),
                                )
                                .child(Label::new(commit).size(LabelSize::Small))
                            })
                            .child(
                                Label::new("Version")
                                    .color(Color::Muted)
                                    .size(LabelSize::XSmall),
                            )
                            .child(Label::new(self.full_version.clone()).size(LabelSize::Small)),
                    )
                    .child(
                        h_flex()
                            .w_full()
                            .gap_1()
                            .child(
                                div()
                                    .flex_1()
                                    .track_focus(&self.ok_entry.focus_handle)
                                    .on_action(cx.listener(|_, _: &menu::Confirm, window, _cx| {
                                        window.remove_window();
                                    }))
                                    .child(
                                        Button::new("ok", "OK")
                                            .full_width()
                                            .style(ButtonStyle::OutlinedGhost)
                                            .toggle_state(ok_is_focused)
                                            .selected_style(ButtonStyle::Tinted(TintColor::Accent))
                                            .on_click(cx.listener(|_, _, window, _cx| {
                                                window.remove_window();
                                            })),
                                    ),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .track_focus(&self.copy_entry.focus_handle)
                                    .on_action(cx.listener(
                                        |this, _: &menu::Confirm, window, cx| {
                                            this.copy_details(window, cx);
                                        },
                                    ))
                                    .child(
                                        Button::new("copy", "Copy")
                                            .full_width()
                                            .style(ButtonStyle::Tinted(TintColor::Accent))
                                            .toggle_state(copy_is_focused)
                                            .selected_style(ButtonStyle::Tinted(TintColor::Accent))
                                            .on_click(cx.listener(|this, _event, window, cx| {
                                                this.copy_details(window, cx);
                                            })),
                                    ),
                            ),
                    )
                    .into_any_element(),
            )
            .entry(self.ok_entry.clone())
            .entry(self.copy_entry.clone())
        }
    }

    impl Focusable for AboutWindow {
        fn focus_handle(&self, _cx: &App) -> FocusHandle {
            self.ok_entry.focus_handle.clone()
        }
    }

    // Don't open about window twice
    if let Some(existing) = cx
        .windows()
        .into_iter()
        .find_map(|w| w.downcast::<AboutWindow>())
    {
        existing
            .update(cx, |about_window, window, cx| {
                window.activate_window();
                about_window.ok_entry.focus_handle.focus(window, cx);
            })
            .log_err();
        return;
    }

    let window_size = Size {
        width: px(440.),
        height: px(300.),
    };

    cx.open_window(
        WindowOptions {
            titlebar: Some(TitlebarOptions {
                title: Some(format!("About {}", product_flavor::DISPLAY_NAME).into()),
                appears_transparent: true,
                traffic_light_position: Some(point(px(12.), px(12.))),
            }),
            window_bounds: Some(WindowBounds::centered(window_size, cx)),
            is_resizable: false,
            is_minimizable: false,
            kind: WindowKind::Floating,
            app_id: Some(ReleaseChannel::global(cx).app_id().to_owned()),
            ..Default::default()
        },
        |window, cx| {
            let about_window = cx.new(AboutWindow::new);
            let focus_handle = about_window.read(cx).ok_entry.focus_handle.clone();
            window.activate_window();
            focus_handle.focus(window, cx);
            about_window
        },
    )
    .log_err();
}

#[cfg(not(target_os = "windows"))]
fn install_cli(
    _: &mut Workspace,
    _: &install_cli::InstallCliBinary,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    install_cli::install_cli_binary(window, cx)
}

static WAITING_QUIT_CONFIRMATION: AtomicBool = AtomicBool::new(false);
fn quit(_: &Quit, cx: &mut App) {
    if WAITING_QUIT_CONFIRMATION.load(atomic::Ordering::Acquire) {
        return;
    }

    let should_confirm = WorkspaceSettings::get_global(cx).confirm_quit;
    cx.spawn(async move |cx| {
        let mut workspace_windows: Vec<WindowHandle<MultiWorkspace>> = cx.update(|cx| {
            cx.windows()
                .into_iter()
                .filter_map(|window| window.downcast::<MultiWorkspace>())
                .collect::<Vec<_>>()
        });

        // If multiple windows have unsaved changes, and need a save prompt,
        // prompt in the active window before switching to a different window.
        cx.update(|cx| {
            workspace_windows.sort_by_key(|window| window.is_active(cx) == Some(false));
        });

        if should_confirm && let Some(multi_workspace) = workspace_windows.first() {
            let answer = multi_workspace
                .update(cx, |_, window, cx| {
                    window.prompt(
                        PromptLevel::Info,
                        "Are you sure you want to quit?",
                        None,
                        &["Quit", "Cancel"],
                        cx,
                    )
                })
                .log_err();

            if let Some(answer) = answer {
                WAITING_QUIT_CONFIRMATION.store(true, atomic::Ordering::Release);
                let answer = answer.await.ok();
                WAITING_QUIT_CONFIRMATION.store(false, atomic::Ordering::Release);
                if answer != Some(0) {
                    return Ok(());
                }
            }
        }

        // If the user cancels any save prompt, then keep the app open.
        for window in &workspace_windows {
            let window = *window;
            let active_and_workspaces = window
                .update(cx, |multi_workspace, _, _cx| {
                    (
                        multi_workspace.workspace().clone(),
                        multi_workspace.workspaces().cloned().collect::<Vec<_>>(),
                    )
                })
                .log_err();

            let Some((originally_active, workspaces)) = active_and_workspaces else {
                continue;
            };

            for workspace in workspaces {
                if let Some(should_close) = window
                    .update(cx, |multi_workspace, window, cx| {
                        multi_workspace.activate(workspace.clone(), None, window, cx);
                        window.activate_window();
                        workspace.update(cx, |workspace, cx| {
                            workspace.prepare_to_close(CloseIntent::Quit, window, cx)
                        })
                    })
                    .log_err()
                {
                    if !should_close.await? {
                        // Activating each workspace above to surface its save
                        // prompts changed which workspace is active. Restore the
                        // user's focused workspace before bailing so the window
                        // is left as they had it.
                        window
                            .update(cx, |multi_workspace, window, cx| {
                                multi_workspace.activate(
                                    originally_active.clone(),
                                    None,
                                    window,
                                    cx,
                                );
                            })
                            .log_err();
                        return Ok(());
                    }
                }
            }

            // The loop above activated each workspace in turn, overwriting the
            // persisted active workspace. Re-activate the workspace the user
            // actually had focused so it is the one serialized (and restored on
            // next launch) as active, rather than whichever happened to be last.
            window
                .update(cx, |multi_workspace, window, cx| {
                    multi_workspace.activate(originally_active, None, window, cx);
                })
                .log_err();
        }
        // Flush all pending workspace serialization before quitting so that
        // session_id/window_id are up-to-date in the database.
        let mut flush_tasks = Vec::new();
        for window in &workspace_windows {
            window
                .update(cx, |multi_workspace, window, cx| {
                    for workspace in multi_workspace.workspaces() {
                        flush_tasks.push(workspace.update(cx, |workspace, cx| {
                            workspace.flush_serialization(window, cx)
                        }));
                    }
                    flush_tasks.append(&mut multi_workspace.take_pending_removal_tasks());
                    flush_tasks.push(multi_workspace.flush_serialization());
                })
                .log_err();
        }
        futures::future::join_all(flush_tasks).await;

        cx.update(|cx| cx.quit());
        anyhow::Ok(())
    })
    .detach_and_log_err(cx);
}

fn open_log_file(workspace: &mut Workspace, window: &mut Window, cx: &mut Context<Workspace>) {
    const MAX_LINES: usize = 1000;
    let app_state = workspace.app_state();
    let languages = app_state.languages.clone();
    let fs = app_state.fs.clone();
    cx.spawn_in(window, async move |workspace, cx| {
        let log = {
            let result = futures::join!(
                fs.load(&paths::old_log_file()),
                fs.load(&paths::log_file()),
                languages.language_for_name("log")
            );
            match result {
                (Err(_), Err(e), _) => Err(e),
                (old_log, new_log, lang) => {
                    let mut lines = VecDeque::with_capacity(MAX_LINES);
                    for line in old_log
                        .iter()
                        .flat_map(|log| log.lines())
                        .chain(new_log.iter().flat_map(|log| log.lines()))
                    {
                        if lines.len() == MAX_LINES {
                            lines.pop_front();
                        }
                        lines.push_back(line);
                    }
                    Ok((
                        lines
                            .into_iter()
                            .flat_map(|line| [line, "\n"])
                            .collect::<String>(),
                        lang.ok(),
                    ))
                }
            }
        };

        let (log, log_language) = match log {
            Ok((log, log_language)) => (log, log_language),
            Err(e) => {
                struct OpenLogError;

                workspace
                    .update(cx, |workspace, cx| {
                        workspace.show_notification(
                            NotificationId::unique::<OpenLogError>(),
                            cx,
                            |cx| {
                                cx.new(|cx| {
                                    MessageNotification::new(
                                        format!(
                                            "Unable to access/open log file at path \
                                                    {}: {e:#}",
                                            paths::log_file().display()
                                        ),
                                        cx,
                                    )
                                })
                            },
                        );
                    })
                    .ok();
                return;
            }
        };
        maybe!(async move {
            let project = workspace
                .read_with(cx, |workspace, _| workspace.project().clone())
                .ok()?;
            let buffer = project
                .update(cx, |project, cx| {
                    project.create_buffer(log_language, false, cx)
                })
                .await
                .ok()?;
            buffer.update(cx, |buffer, cx| {
                buffer.set_capability(Capability::ReadOnly, cx);
                buffer.set_text(log, cx);
            });

            let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx).with_title("Log".into()));

            let editor = cx
                .new_window_entity(|window, cx| {
                    let mut editor = Editor::for_multibuffer(buffer, Some(project), window, cx);
                    editor.set_read_only(true);
                    editor.set_breadcrumb_header(format!(
                        "Last {} lines in {}",
                        MAX_LINES,
                        paths::log_file().display()
                    ));
                    let last_multi_buffer_offset = editor.buffer().read(cx).len(cx);
                    editor.change_selections(Default::default(), window, cx, |s| {
                        s.select_ranges(Some(last_multi_buffer_offset..last_multi_buffer_offset));
                    });
                    editor
                })
                .ok()?;

            workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.add_item_to_active_pane(Box::new(editor), None, true, window, cx);
                })
                .ok()
        })
        .await;
    })
    .detach();
}

fn notify_settings_errors(result: settings::SettingsParseResult, is_user: bool, cx: &mut App) {
    if let settings::ParseStatus::Failed { error: err } = &result.parse_status {
        let settings_type = if is_user { "user" } else { "global" };
        log::error!("Failed to load {} settings: {err}", settings_type);
    }

    let error = match result.parse_status {
        settings::ParseStatus::Failed { error } => Some(anyhow::format_err!(error)),
        settings::ParseStatus::Success => None,
        settings::ParseStatus::Unchanged => return,
    };
    let id = NotificationId::Named(format!("failed-to-parse-settings-{is_user}").into());

    let showed_parse_error = match error {
        Some(error) => {
            if let Some(InvalidSettingsError::LocalSettings { .. }) =
                error.downcast_ref::<InvalidSettingsError>()
            {
                false
                // Local settings errors are displayed by the projects
            } else {
                show_app_notification(id, cx, move |cx| {
                    cx.new(|cx| {
                        MessageNotification::new(format!("Invalid user settings file\n{error}"), cx)
                            .primary_message("Open Settings File")
                            .primary_icon(IconName::Settings)
                            .primary_on_click(|window, cx| {
                                window.dispatch_action(
                                    zed_actions::OpenSettingsFile.boxed_clone(),
                                    cx,
                                );
                                cx.emit(DismissEvent);
                            })
                    })
                });
                true
            }
        }
        None => {
            dismiss_app_notification(&id, cx);
            false
        }
    };
    let id = NotificationId::Named(format!("failed-to-migrate-settings-{is_user}").into());

    match result.migration_status {
        settings::MigrationStatus::Succeeded | settings::MigrationStatus::NotNeeded => {
            dismiss_app_notification(&id, cx);
        }
        settings::MigrationStatus::Failed { error: err } => {
            if !showed_parse_error {
                show_app_notification(id, cx, move |cx| {
                    cx.new(|cx| {
                        MessageNotification::new(
                            format!(
                                "Failed to migrate settings\n\
                                {err}"
                            ),
                            cx,
                        )
                        .primary_message("Open Settings File")
                        .primary_icon(IconName::Settings)
                        .primary_on_click(|window, cx| {
                            window.dispatch_action(zed_actions::OpenSettingsFile.boxed_clone(), cx);
                            cx.emit(DismissEvent);
                        })
                    })
                });
            }
        }
    };
}

fn init_global_config_error_notifications(cx: &mut App) {
    cx.observe_new(|_: &mut SettingsObserver, _, cx| {
        cx.subscribe_self::<SettingsObserverEvent>(|_, event, cx| {
            let (result, file_kind, on_click): (_, _, fn(&mut Window, &mut App)) = match event {
                SettingsObserverEvent::GlobalTasksUpdated(result) => {
                    (result, "tasks", |window, cx| {
                        window.dispatch_action(OpenTasks.boxed_clone(), cx)
                    })
                }
                SettingsObserverEvent::GlobalDebugScenariosUpdated(result) => {
                    (result, "debug scenarios", |window, cx| {
                        window.dispatch_action(OpenDebugTasks.boxed_clone(), cx)
                    })
                }
                _ => return,
            };
            let id = NotificationId::Named(format!("invalid-global-{file_kind}-file").into());
            match result {
                Ok(_) => dismiss_app_notification(&id, cx),
                Err(error) => {
                    let message = format!("Invalid global {file_kind} file\n{error}");
                    show_app_notification(id, cx, move |cx| {
                        cx.new(|cx| {
                            MessageNotification::new(message.clone(), cx)
                                .primary_message("Open File")
                                .primary_icon(IconName::Settings)
                                .primary_on_click(move |window, cx| {
                                    on_click(window, cx);
                                    cx.emit(DismissEvent);
                                })
                        })
                    });
                }
            }
        })
        .detach();
    })
    .detach();
}

#[derive(Copy, Clone, Debug, settings::RegisterSetting)]
struct CursorHideModeSetting(gpui::CursorHideMode);

impl Settings for CursorHideModeSetting {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        Self(match content.hide_mouse.unwrap_or_default() {
            settings::HideMouseMode::Never => gpui::CursorHideMode::Never,
            settings::HideMouseMode::OnTyping => gpui::CursorHideMode::OnTyping,
            settings::HideMouseMode::OnTypingAndAction => gpui::CursorHideMode::OnTypingAndAction,
        })
    }
}

fn init_cursor_hide_mode(cx: &mut App) {
    let apply = |cx: &mut App| cx.set_cursor_hide_mode(CursorHideModeSetting::get_global(cx).0);
    apply(cx);
    cx.observe_global::<SettingsStore>(apply).detach();
}

fn init_app_appearance(cx: &mut App) {
    // Force the native window chrome (border + titlebar) to match the selected theme.
    // `System` follows the OS (no override); any other theme forces its appearance, so a
    // dark theme doesn't render a light window border when the system is in light mode.
    let apply = |cx: &mut App| {
        let appearance = match ThemeSettings::get_global(cx).theme.mode() {
            Some(theme_settings::ThemeAppearanceMode::System) => None,
            _ => Some(match cx.theme().appearance() {
                theme::Appearance::Light => gpui::WindowAppearance::Light,
                theme::Appearance::Dark => gpui::WindowAppearance::Dark,
            }),
        };
        cx.set_window_appearance(appearance);
    };
    apply(cx);
    cx.observe_global::<SettingsStore>(apply).detach();
}

#[derive(Copy, Clone, Debug, settings::RegisterSetting)]
struct ReduceMotionSetting(settings::ReduceMotionMode);

impl Settings for ReduceMotionSetting {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        Self(content.reduce_motion.unwrap_or_default())
    }
}

fn init_reduce_motion(cx: &mut App) {
    let apply = |cx: &mut App| {
        let reduce_motion = ReduceMotionSetting::get_global(cx).0 == settings::ReduceMotionMode::On;
        cx.set_reduce_motion(reduce_motion);
    };
    apply(cx);
    cx.observe_global::<SettingsStore>(apply).detach();
}

/// Starts watching `~/.config/zed/AGENTS.md` (or the platform equivalent) and
/// surfaces any read errors using the same notification UI as settings errors.
///
/// The file itself is loaded into [`agent_settings::UserAgentsMd`] for inclusion
/// in prompts.
#[cfg(feature = "agentic-tools")]
pub fn watch_user_agents_md(fs: Arc<dyn fs::Fs>, cx: &mut App) {
    struct UserAgentsMdParseError;
    let notification_id = NotificationId::unique::<UserAgentsMdParseError>();

    init_user_agents_md(fs, cx, move |state, cx| match state {
        UserAgentsMdState::Loaded(_) | UserAgentsMdState::Empty => {
            dismiss_app_notification(&notification_id, cx);
        }
        UserAgentsMdState::Error(message) => {
            let path = paths::agents_file().display().to_string();
            log::error!("Failed to load user AGENTS.md from {path}: {message}");
            let body = format!("Failed to load {path}\n{message}");
            let notification_id = notification_id.clone();
            show_app_notification(notification_id, cx, move |cx| {
                let body = body.clone();
                cx.new(|cx| MessageNotification::new(body, cx))
            });
        }
    });
}

pub fn watch_settings_files(fs: Arc<dyn fs::Fs>, cx: &mut App) {
    MigrationNotification::set_global(cx.new(|_| MigrationNotification), cx);

    SettingsStore::update_global(cx, move |store, cx| {
        store.watch_settings_files(fs, cx, |settings_file, result, cx| {
            let is_user = matches!(settings_file, SettingsFile::User);
            let migrating_in_memory =
                matches!(&result.migration_status, MigrationStatus::Succeeded);
            notify_settings_errors(result, is_user, cx);
            if let Some(notifier) = MigrationNotification::try_global(cx) {
                notifier.update(cx, |_, cx| {
                    cx.emit(MigrationEvent::ContentChanged {
                        migration_type: MigrationType::Settings,
                        migrating_in_memory,
                    });
                });
            }
        });
    });
}

pub fn handle_keymap_file_changes(
    mut user_keymap_file_rx: mpsc::UnboundedReceiver<String>,
    user_keymap_watcher: gpui::Task<()>,
    cx: &mut App,
) {
    let (base_keymap_tx, mut base_keymap_rx) = mpsc::unbounded();
    let (keyboard_layout_tx, mut keyboard_layout_rx) = mpsc::unbounded();
    let mut old_base_keymap = *BaseKeymap::get_global(cx);
    let mut old_vim_enabled = VimModeSetting::get_global(cx).0;
    let mut old_helix_enabled = vim_mode_setting::HelixModeSetting::get_global(cx).0;
    let mut old_disable_ai = DisableAiSettings::get_global(cx).disable_ai;

    cx.observe_global::<SettingsStore>(move |cx| {
        let new_base_keymap = *BaseKeymap::get_global(cx);
        let new_vim_enabled = VimModeSetting::get_global(cx).0;
        let new_helix_enabled = vim_mode_setting::HelixModeSetting::get_global(cx).0;
        let new_disable_ai = DisableAiSettings::get_global(cx).disable_ai;

        if new_base_keymap != old_base_keymap
            || new_vim_enabled != old_vim_enabled
            || new_helix_enabled != old_helix_enabled
            || new_disable_ai != old_disable_ai
        {
            old_base_keymap = new_base_keymap;
            old_vim_enabled = new_vim_enabled;
            old_helix_enabled = new_helix_enabled;
            old_disable_ai = new_disable_ai;

            base_keymap_tx.unbounded_send(()).log_err();
        }
    })
    .detach();

    #[cfg(target_os = "windows")]
    {
        let mut current_layout_id = cx.keyboard_layout().id().to_string();
        cx.on_keyboard_layout_change(move |cx| {
            let next_layout_id = cx.keyboard_layout().id();
            if next_layout_id != current_layout_id {
                current_layout_id = next_layout_id.to_string();
                keyboard_layout_tx.unbounded_send(()).ok();
            }
        })
        .detach();
    }

    #[cfg(not(target_os = "windows"))]
    {
        let mut current_mapping = cx.keyboard_mapper().get_key_equivalents().cloned();
        cx.on_keyboard_layout_change(move |cx| {
            let next_mapping = cx.keyboard_mapper().get_key_equivalents();
            if current_mapping.as_ref() != next_mapping {
                current_mapping = next_mapping.cloned();
                keyboard_layout_tx.unbounded_send(()).ok();
            }
        })
        .detach();
    }

    load_default_keymap(cx);

    struct KeymapParseErrorNotification;
    let notification_id = NotificationId::unique::<KeymapParseErrorNotification>();

    cx.spawn(async move |cx| {
        let _user_keymap_watcher = user_keymap_watcher;
        let mut user_keymap_content = String::new();
        let mut migrating_in_memory = false;
        loop {
            select_biased! {
                _ = base_keymap_rx.next() => {},
                _ = keyboard_layout_rx.next() => {},
                content = user_keymap_file_rx.next() => {
                    if let Some(content) = content {
                        if let Ok(Some(migrated_content)) = migrate_keymap(&content) {
                            user_keymap_content = migrated_content;
                            migrating_in_memory = true;
                        } else {
                            user_keymap_content = content;
                            migrating_in_memory = false;
                        }
                    }
                }
            };
            cx.update(|cx| {
                if let Some(notifier) = MigrationNotification::try_global(cx) {
                    notifier.update(cx, |_, cx| {
                        cx.emit(MigrationEvent::ContentChanged {
                            migration_type: MigrationType::Keymap,
                            migrating_in_memory,
                        });
                    });
                }
                let load_result = KeymapFile::load(&user_keymap_content, cx);
                match load_result {
                    KeymapFileLoadResult::Success { key_bindings } => {
                        reload_keymaps(cx, key_bindings);
                        dismiss_app_notification(&notification_id.clone(), cx);
                    }
                    KeymapFileLoadResult::SomeFailedToLoad {
                        key_bindings,
                        error_message,
                    } => {
                        if !key_bindings.is_empty() {
                            reload_keymaps(cx, key_bindings);
                        }
                        show_keymap_file_load_error(notification_id.clone(), error_message, cx);
                    }
                    KeymapFileLoadResult::JsonParseFailure { error } => {
                        show_keymap_file_json_error(notification_id.clone(), &error, cx)
                    }
                }
            });
        }
    })
    .detach();
}

fn show_keymap_file_json_error(
    notification_id: NotificationId,
    error: &anyhow::Error,
    cx: &mut App,
) {
    let message: SharedString =
        format!("JSON parse error in keymap file. Bindings not reloaded.\n\n{error}").into();
    show_app_notification(notification_id, cx, move |cx| {
        cx.new(|cx| {
            MessageNotification::new(message.clone(), cx)
                .primary_message("Open Keymap File")
                .primary_icon(IconName::Settings)
                .primary_on_click(|window, cx| {
                    window.dispatch_action(zed_actions::OpenKeymapFile.boxed_clone(), cx);
                    cx.emit(DismissEvent);
                })
        })
    });
}

fn show_keymap_file_load_error(
    notification_id: NotificationId,
    error_message: MarkdownString,
    cx: &mut App,
) {
    show_markdown_app_notification(
        notification_id,
        error_message,
        "Open Keymap File".into(),
        |window, cx| {
            window.dispatch_action(zed_actions::OpenKeymapFile.boxed_clone(), cx);
            cx.emit(DismissEvent);
        },
        cx,
    )
}

fn show_markdown_app_notification<F>(
    notification_id: NotificationId,
    message: MarkdownString,
    primary_button_message: SharedString,
    primary_button_on_click: F,
    cx: &mut App,
) where
    F: 'static + Send + Sync + Fn(&mut Window, &mut Context<MessageNotification>),
{
    let markdown = cx.new(|cx| Markdown::new(message.0.into(), None, None, cx));
    let primary_button_on_click = Arc::new(primary_button_on_click);

    show_app_notification(notification_id, cx, move |cx| {
        let markdown = markdown.clone();
        let primary_button_message = primary_button_message.clone();
        let primary_button_on_click = primary_button_on_click.clone();

        cx.new(move |cx| {
            MessageNotification::new_from_builder(cx, move |window, cx| {
                image_cache(retain_all("notification-cache"))
                    .child(div().text_ui(cx).child(MarkdownElement::new(
                        markdown.clone(),
                        MarkdownStyle::themed(MarkdownFont::Editor, window, cx),
                    )))
                    .into_any()
            })
            .primary_message(primary_button_message)
            .primary_icon(IconName::Settings)
            .primary_on_click_arc(primary_button_on_click)
        })
    })
}

fn reload_keymaps(cx: &mut App, mut user_key_bindings: Vec<KeyBinding>) {
    cx.clear_key_bindings();
    load_default_keymap(cx);
    #[cfg(feature = "sim")]
    graphics_ui::bind_keys(cx);
    #[cfg(feature = "agentic-tools")]
    agent_ui::bind_keys(cx);
    #[cfg(feature = "rust-tools")]
    cargo_ui::bind_keys(cx);
    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    tasks_ui::bind_test_panel_keys(cx);
    #[cfg(feature = "jvm-tools")]
    jvm_ui::bind_keys(cx);

    for key_binding in &mut user_key_bindings {
        key_binding.set_meta(KeybindSource::User.meta());
    }
    cx.bind_keys(filter_disabled_ai_bindings(user_key_bindings, cx));

    let menus = app_menus(cx);
    cx.set_menus(menus);
    // On Windows, this is set in the `update_jump_list` method of the `HistoryManager`.
    #[cfg(not(target_os = "windows"))]
    cx.set_dock_menu(vec![gpui::MenuItem::action(
        "New Window",
        workspace::NewWindow,
    )]);
    // todo: nicer api here?
    keymap_editor::KeymapEventChannel::trigger_keymap_changed(cx);
}

pub fn load_default_keymap(cx: &mut App) {
    let base_keymap = *BaseKeymap::get_global(cx);
    let vim_enabled =
        VimModeSetting::get_global(cx).0 || vim_mode_setting::HelixModeSetting::get_global(cx).0;
    for (asset_path, source) in builtin_keymap_assets(base_keymap, vim_enabled) {
        match load_builtin_keymap_asset(asset_path, source, cx) {
            Ok(key_bindings) => cx.bind_keys(filter_disabled_ai_bindings(key_bindings, cx)),
            Err(error) => {
                log::error!("Failed to load built-in keymap {asset_path:?}: {error:#}");
            }
        }
    }
}

fn load_builtin_keymap_asset(
    asset_path: &str,
    source: KeybindSource,
    cx: &App,
) -> anyhow::Result<Vec<KeyBinding>> {
    #[cfg(feature = "agentic-tools")]
    let key_bindings = KeymapFile::load_asset(asset_path, Some(source), cx)?;
    #[cfg(not(feature = "agentic-tools"))]
    let mut key_bindings = KeymapFile::load_asset_allow_partial_failure(asset_path, cx)?;
    #[cfg(not(feature = "agentic-tools"))]
    for key_binding in &mut key_bindings {
        key_binding.set_meta(source.meta());
    }
    Ok(key_bindings)
}

#[cfg(feature = "agentic-tools")]
const AI_ACTION_NAMESPACES: &[&str] = &[
    "acp::",
    "agent::",
    "assistant::",
    "edit_prediction::",
    "inline_assistant::",
    "zeta::",
];

#[cfg(not(feature = "agentic-tools"))]
const AI_ACTION_NAMESPACES: &[&str] = &["acp::", "agent::", "assistant::", "inline_assistant::"];

fn is_ai_keybinding(binding: &KeyBinding) -> bool {
    let name = binding.action().name();
    AI_ACTION_NAMESPACES
        .iter()
        .any(|namespace| name.starts_with(namespace))
}

fn filter_disabled_ai_bindings(bindings: Vec<KeyBinding>, cx: &App) -> Vec<KeyBinding> {
    if cfg!(feature = "agentic-tools") && !DisableAiSettings::get_global(cx).disable_ai {
        return bindings;
    }
    bindings
        .into_iter()
        .filter(|binding| !is_ai_keybinding(binding))
        .collect()
}

fn builtin_keymap_assets(
    base_keymap: BaseKeymap,
    vim_enabled: bool,
) -> Vec<(&'static str, KeybindSource)> {
    if base_keymap == BaseKeymap::None {
        return Vec::new();
    }

    let mut assets = vec![(DEFAULT_KEYMAP_PATH, KeybindSource::Default)];
    if let Some(asset_path) = base_keymap.asset_path() {
        assets.push((asset_path, KeybindSource::Base));
    }
    if vim_enabled {
        assets.push((VIM_KEYMAP_PATH, KeybindSource::Vim));
    }
    #[cfg(feature = "sim")]
    assets.push((sim_ui::DEFAULT_SIM_KEYMAP_PATH, KeybindSource::Default));
    assets.push((SPECIFIC_OVERRIDES_KEYMAP_PATH, KeybindSource::Default));
    assets
}

pub fn open_new_ssh_project_from_project(
    workspace: &mut Workspace,
    paths: Vec<PathBuf>,
    create_new_window: bool,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) -> Task<anyhow::Result<()>> {
    let app_state = workspace.app_state().clone();
    let Some(ssh_client) = workspace.project().read(cx).remote_client() else {
        return Task::ready(Err(anyhow::anyhow!("Not an ssh project")));
    };
    let connection_options = ssh_client.read(cx).connection_options();
    let requesting_window = if create_new_window {
        None
    } else {
        window.window_handle().downcast::<MultiWorkspace>()
    };
    cx.spawn_in(window, async move |_, cx| {
        open_remote_project(
            connection_options,
            paths,
            app_state,
            workspace::OpenOptions {
                workspace_matching: workspace::WorkspaceMatching::None,
                requesting_window,
                ..Default::default()
            },
            cx,
        )
        .await
        .map(|_| ())
    })
}

fn open_project_settings_file(
    workspace: &mut Workspace,
    _: &OpenProjectSettingsFile,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    if let Some(task) = open_local_file(
        workspace,
        local_settings_file_relative_path(),
        initial_project_settings_content(),
        window,
        cx,
    ) {
        task.detach_and_log_err(cx);
    }
}

fn open_project_tasks_file(
    workspace: &mut Workspace,
    _: &OpenProjectTasks,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    if let Some(task) = open_local_file(
        workspace,
        local_tasks_file_relative_path(),
        initial_tasks_content(),
        window,
        cx,
    ) {
        task.detach_and_log_err(cx);
    }
}

fn open_worktree_setup_tasks_file(
    workspace: &mut Workspace,
    _: &zed_actions::OpenWorktreeSetupTasks,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    // Kept harmless on purpose: tasks with the `create_worktree` hook run automatically
    // when a worktree is created, so the example must be safe to save unedited.
    const WORKTREE_SETUP_TASK_EXAMPLE: &str = r#"  {
    // Runs automatically after Zed creates a new git worktree.
    // $ZED_WORKTREE_ROOT is the new worktree's root directory, and
    // $ZED_MAIN_GIT_WORKTREE is the original repository's working directory.
    "label": "Set up new worktree",
    "command": "echo \"Setting up $ZED_WORKTREE_ROOT — edit this command\"",
    "cwd": "$ZED_WORKTREE_ROOT",
    "hooks": ["create_worktree"]
  }"#;

    let Some(open_task) = open_local_file(
        workspace,
        local_tasks_file_relative_path(),
        settings::initial_worktree_setup_tasks_content(),
        window,
        cx,
    ) else {
        return;
    };

    cx.spawn_in(window, async move |_, cx| {
        let editor = open_task.await?;
        editor.update_in(cx, |editor, window, cx| {
            // Skip insertion if the file already mentions the hook (even in a comment,
            // like the seeded template's example — uncommenting it beats duplicating it).
            // `create_git_worktree` is a serde alias for the same hook.
            let text = editor.text(cx);
            if text.contains("create_worktree") || text.contains("create_git_worktree") {
                return anyhow::Ok(());
            }
            tasks_ui::insert_task_json_into_editor(
                editor,
                WORKTREE_SETUP_TASK_EXAMPLE.to_string(),
                window,
                cx,
            )
        })?
    })
    .detach_and_log_err(cx);
}

fn open_project_debug_tasks_file(
    workspace: &mut Workspace,
    _: &zed_actions::OpenProjectDebugTasks,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    if let Some(task) = open_local_file(
        workspace,
        local_debug_file_relative_path(),
        initial_local_debug_tasks_content(),
        window,
        cx,
    ) {
        task.detach_and_log_err(cx);
    }
}

fn open_local_file(
    workspace: &mut Workspace,
    settings_relative_path: &'static RelPath,
    initial_contents: Cow<'static, str>,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) -> Option<gpui::Task<anyhow::Result<Entity<Editor>>>> {
    let project = workspace.project().clone();
    let worktree = project
        .read(cx)
        .visible_worktrees(cx)
        .find_map(|tree| tree.read(cx).root_entry()?.is_dir().then_some(tree));
    if let Some(worktree) = worktree {
        let tree_id = worktree.read(cx).id();
        Some(cx.spawn_in(window, async move |workspace, cx| {
            // Check if the file actually exists on disk (even if it's excluded from worktree)
            let file_exists = {
                let full_path = worktree.read_with(cx, |tree, _| {
                    tree.abs_path().join(settings_relative_path.as_std_path())
                });

                let fs = project.read_with(cx, |project, _| project.fs().clone());

                fs.metadata(&full_path)
                    .await
                    .ok()
                    .flatten()
                    .is_some_and(|metadata| !metadata.is_dir && !metadata.is_fifo)
            };

            if !file_exists {
                if let Some(dir_path) = settings_relative_path.parent()
                    && worktree.read_with(cx, |tree, _| tree.entry_for_path(dir_path).is_none())
                {
                    project
                        .update(cx, |project, cx| {
                            project.create_entry((tree_id, dir_path), true, cx)
                        })
                        .await
                        .context("worktree was removed")?;
                }

                if worktree.read_with(cx, |tree, _| {
                    tree.entry_for_path(settings_relative_path).is_none()
                }) {
                    project
                        .update(cx, |project, cx| {
                            project.create_entry((tree_id, settings_relative_path), false, cx)
                        })
                        .await
                        .context("worktree was removed")?;
                }
            }

            let editor = workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.open_path((tree_id, settings_relative_path), None, true, window, cx)
                })?
                .await?
                .downcast::<Editor>()
                .context("unexpected item type: expected editor item")?;

            editor.update(cx, |editor, cx| {
                if let Some(buffer) = editor.buffer().read(cx).as_singleton()
                    && buffer.read(cx).is_empty()
                {
                    buffer.update(cx, |buffer, cx| {
                        buffer.edit([(0..0, initial_contents)], None, cx)
                    });
                }
            });

            anyhow::Ok(editor)
        }))
    } else {
        struct NoOpenFolders;

        workspace.show_notification(NotificationId::unique::<NoOpenFolders>(), cx, |cx| {
            cx.new(|cx| MessageNotification::new("This project has no folders open.", cx))
        });
        None
    }
}

fn open_bundled_file(
    workspace: &mut Workspace,
    text: Cow<'static, str>,
    title: &'static str,
    language: &'static str,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let existing = workspace.items_of_type::<Editor>(cx).find(|editor| {
        editor.read_with(cx, |editor, cx| {
            editor.read_only(cx)
                && editor.title(cx).as_ref() == title
                && editor
                    .buffer()
                    .read(cx)
                    .as_singleton()
                    .is_some_and(|buffer| buffer.read(cx).file().is_none())
        })
    });
    if let Some(existing) = existing {
        workspace.activate_item(&existing, true, true, window, cx);
        return;
    }

    let language = workspace.app_state().languages.language_for_name(language);
    cx.spawn_in(window, async move |workspace, cx| {
        let language = language.await.log_err();
        workspace
            .update_in(cx, move |workspace, window, cx| {
                let project = workspace.project().clone();
                let buffer = project.update(cx, move |project, cx| {
                    project.create_buffer(language, false, cx)
                });
                cx.spawn_in(window, async move |workspace, cx| {
                    let buffer = buffer.await?;
                    buffer.update(cx, |buffer, cx| {
                        buffer.set_text(text.into_owned(), cx);
                        buffer.set_capability(Capability::ReadOnly, cx);
                    });
                    let buffer =
                        cx.new(|cx| MultiBuffer::singleton(buffer, cx).with_title(title.into()));
                    workspace.update_in(cx, |workspace, window, cx| {
                        workspace.add_item_to_active_pane(
                            Box::new(cx.new(|cx| {
                                let mut editor = Editor::for_multibuffer(
                                    buffer,
                                    Some(project.clone()),
                                    window,
                                    cx,
                                );
                                editor.set_read_only(true);
                                editor.set_should_serialize(false, cx);
                                editor.set_breadcrumb_header(title.into());
                                editor
                            })),
                            None,
                            true,
                            window,
                            cx,
                        )
                    })
                })
            })?
            .await
    })
    .detach_and_log_err(cx);
}

fn open_settings_file(
    abs_path: &'static Path,
    default_content: impl FnOnce() -> Rope + Send + 'static,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    cx.spawn_in(window, async move |workspace, cx| {
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.with_local_or_wsl_workspace(window, cx, move |workspace, window, cx| {
                    let project = workspace.project().clone();

                    cx.spawn_in(window, async move |workspace, cx| {
                        let config_dir = project
                            .update(cx, |project, cx| {
                                project.try_windows_path_to_wsl(paths::config_dir().as_path(), cx)
                            })
                            .await?;
                        // Set up a dedicated worktree for settings, since
                        // otherwise we're dropping and re-starting LSP servers
                        // for each file inside on every settings file
                        // close/open

                        // TODO: Do note that all other external files (e.g.
                        // drag and drop from OS) still have their worktrees
                        // released on file close, causing LSP servers'
                        // restarts.
                        let (_worktree, _) = project
                            .update(cx, |project, cx| {
                                project.find_or_create_worktree(&config_dir, false, cx)
                            })
                            .await?;

                        workspace
                            .update_in(cx, |_, window, cx| {
                                create_and_open_local_file(abs_path, window, cx, default_content)
                            })?
                            .await?;
                        anyhow::Ok(())
                    })
                })
            })?
            .await?
            .await?;
        anyhow::Ok(())
    })
    .detach_and_log_err(cx);
}

/// Eagerly loads the active theme and icon theme based on the selections in the
/// theme settings.
///
/// This fast path exists to load these themes as soon as possible so the user
/// doesn't see the default themes while waiting on extensions to load.
pub(crate) fn eager_load_active_theme_and_icon_theme(fs: Arc<dyn Fs>, cx: &mut App) {
    let extension_store = ExtensionStore::global(cx);
    let theme_registry = ThemeRegistry::global(cx);
    let theme_settings = ThemeSettings::get_global(cx);
    let appearance = SystemAppearance::global(cx).0;

    enum LoadTarget {
        Theme(PathBuf),
        IconTheme((PathBuf, PathBuf)),
    }

    let theme_name = theme_settings.theme.name(appearance);
    let icon_theme_name = theme_settings.icon_theme.name(appearance);
    let themes_to_load = [
        theme_registry
            .get(&theme_name.0)
            .is_err()
            .then(|| {
                extension_store
                    .read(cx)
                    .path_to_extension_theme(&theme_name.0)
            })
            .flatten()
            .map(LoadTarget::Theme),
        theme_registry
            .get_icon_theme(&icon_theme_name.0)
            .is_err()
            .then(|| {
                extension_store
                    .read(cx)
                    .path_to_extension_icon_theme(&icon_theme_name.0)
            })
            .flatten()
            .map(LoadTarget::IconTheme),
    ];

    enum ReloadTarget {
        Theme,
        IconTheme,
    }

    let executor = cx.background_executor();
    let reload_tasks = parking_lot::Mutex::new(Vec::with_capacity(themes_to_load.len()));

    let mut themes_to_load = themes_to_load.into_iter().flatten().peekable();

    if themes_to_load.peek().is_none() {
        return;
    }

    cx.foreground_executor().block_on(executor.scoped(|scope| {
        for load_target in themes_to_load {
            let theme_registry = &theme_registry;
            let reload_tasks = &reload_tasks;
            let fs = fs.clone();

            scope.spawn(async move {
                match load_target {
                    LoadTarget::Theme(theme_path) => {
                        if let Some(bytes) = fs.load_bytes(&theme_path).await.log_err()
                            && load_user_theme(theme_registry, &bytes).log_err().is_some()
                        {
                            reload_tasks.lock().push(ReloadTarget::Theme);
                        }
                    }
                    LoadTarget::IconTheme((icon_theme_path, icons_root_path)) => {
                        if let Some(bytes) = fs.load_bytes(&icon_theme_path).await.log_err()
                            && let Some(icon_theme_family) =
                                deserialize_icon_theme(&bytes).log_err()
                            && theme_registry
                                .load_icon_theme(icon_theme_family, &icons_root_path)
                                .log_err()
                                .is_some()
                        {
                            reload_tasks.lock().push(ReloadTarget::IconTheme);
                        }
                    }
                }
            });
        }
    }));

    for reload_target in reload_tasks.into_inner() {
        match reload_target {
            ReloadTarget::Theme => theme_settings::reload_theme(cx),
            ReloadTarget::IconTheme => theme_settings::reload_icon_theme(cx),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "multiplayer-tools")]
    use acp_thread::AgentConnection as _;
    use assets::Assets;
    use collections::HashSet;
    use editor::{
        DisplayPoint, Editor, MultiBufferOffset, SelectionEffects, display_map::DisplayRow,
    };
    use extension::ExtensionHostProxy;
    use fs::FakeFs;
    #[cfg(feature = "multiplayer-tools")]
    use gpui::Empty;
    use gpui::{
        Action, AnyWindowHandle, App, AssetSource, BorrowAppContext, MenuItem, Modifiers,
        TestAppContext, UpdateGlobal, VisualTestContext, WindowHandle, actions, point, px,
    };
    #[cfg(feature = "sim")]
    use gpui::{KeyContext, Keystroke, Menu};
    use http_client::BlockedHttpClient;
    use language::LanguageRegistry;
    use languages::{markdown_lang, rust_lang};
    use node_runtime::NodeRuntime;
    use pretty_assertions::{assert_eq, assert_ne};
    use project::{Project, ProjectPath};
    #[cfg(feature = "agentic-tools")]
    use prompt_store::PromptBuilder;
    use remote::RemoteClient;
    use remote_server::{HeadlessAppState, HeadlessProject};
    use semver::Version;
    use serde_json::json;
    use settings::{SaturatingBool, SettingsStore, watch_config_file};
    use std::{
        path::{Path, PathBuf},
        sync::Arc,
        time::Duration,
    };

    #[test]
    fn provider_worker_bridge_bootstrap_retains_active_executor_and_headless_stays_denied() {
        let desktop = include_str!("zed.rs");
        let headless = include_str!("sim_cli.rs");
        for required in [
            "private_worker_executor: Arc<sim_plugin_host::PrivateWorkerPluginExecutor>",
            "let private_worker_executor = plugin_services.private_worker_executor()",
            "component_host.private_worker_executor = private_worker_executor",
            "component_host.private_worker_executor.clone()",
            "private_worker_executor.attach_provider_worker_bridge(provider_worker_bridge)",
            "sim_ui::clear_native_execution_services(cx)",
        ] {
            assert!(
                desktop.contains(required),
                "desktop provider bridge bootstrap lacks {required}"
            );
        }
        let current_fast_path = desktop
            .find("if matches_current")
            .expect("same-generation component-host fast path");
        let services_construction = desktop[current_fast_path..]
            .find("sim_plugin_services::private_worker_services(")
            .map(|offset| current_fast_path + offset)
            .expect("private-worker services construction");
        assert!(current_fast_path < services_construction);

        let register = desktop
            .find("sim_ui::register_native_execution_services(config, registry_bundle, cx)")
            .expect("native controller registration");
        let transition = desktop[register..]
            .find("activate_desktop_component_deployment(")
            .map(|offset| register + offset)
            .expect("exact desktop deployment transition");
        let attach = desktop
            .find("private_worker_executor.attach_provider_worker_bridge(provider_worker_bridge)")
            .expect("exact retained executor attachment");
        assert!(attach < register && register < transition);

        assert!(headless.contains("deny_only_private_worker_services("));
        assert!(headless.contains("start_with_provider_worker_bridge("));
        assert!(headless.contains("attach_provider_worker_bridge(provider_worker_bridge)"));
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    async fn native_sim_first_start_failure_recovers_through_initialized_extension_store(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        native_sim_first_start_failure_recovers_through_initialized_extension_store_result(cx)
            .await
            .expect("fresh-start recovery must use the retained canonical adapter");
    }

    #[cfg(feature = "sim")]
    async fn native_sim_first_start_failure_recovers_through_initialized_extension_store_result(
        cx: &mut TestAppContext,
    ) -> anyhow::Result<()> {
        let app_state = init_test(cx);
        let temporary = tempfile::tempdir()?;
        let (profile, security, adapter) = cx.update(|cx| -> anyhow::Result<_> {
            let (profile, security) = native_sim_registry_test_configuration(
                sim_ui::LOCAL_EXECUTION_PROFILE_ID.0,
                "balanced",
                sim_runtime::DEFAULT_COMPONENT_REGISTRY_GENERATION,
                cx,
            );
            let adapter = ensure_desktop_component_lifecycle_registration(cx)?;
            let mut invalid_profile = profile.clone();
            invalid_profile
                .model_roots
                .push(temporary.path().join("missing-models"));
            let failure = ensure_native_sim_assets(&invalid_profile, false, temporary.path(), cx)
                .err()
                .ok_or_else(|| anyhow::anyhow!("a missing model root unexpectedly succeeded"))?;
            fail_native_sim_activation(format!("native asset bootstrap failed: {failure}"), cx);
            assert!(cx.try_global::<SimComponentHostGlobal>().is_none());
            assert!(sim_ui::native_asset_services(cx).is_none());
            assert!(cx.try_global::<NativeSimRuntimeBinding>().is_none());
            assert!(sim_ui::initialization_error(cx).is_some());
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(profile.id), cx);
            Ok((profile, security, adapter))
        })?;
        let filesystem = FakeFs::new(cx.executor());
        let extensions_root = paths::extensions_dir().clone();
        filesystem
            .insert_tree(extensions_root.join("installed"), json!({}))
            .await;
        filesystem
            .insert_file(
                extensions_root.join("index.json"),
                serde_json::to_vec(&extension_host::ExtensionIndex::default())?,
            )
            .await;
        cx.update(|cx| {
            extension_host::init(
                Arc::new(ExtensionHostProxy::new()),
                filesystem.clone(),
                app_state.client.clone(),
                NodeRuntime::unavailable(),
                cx,
            );
        });
        let store = cx.update(|cx| ExtensionStore::global(cx));
        let first_reload = store.update(cx, |store, cx| store.reload(None, cx));
        cx.executor().advance_clock(Duration::from_secs(1));
        cx.executor().run_until_parked();
        let first_failure = first_reload
            .await
            .expect_err("unavailable host must reject inventory");
        assert!(
            first_failure
                .to_string()
                .contains("native Sim component host is unavailable")
        );
        assert!(
            adapter
                .accepted_candidate_identity
                .lock()
                .map_err(|_| anyhow::anyhow!("candidate state unavailable"))?
                .is_none()
        );
        cx.update(|cx| -> anyhow::Result<()> {
            assert!(ExtensionStore::try_global(cx).is_some());
            assert!(extension_host::register_component_lifecycle_adapter(
                Arc::new(adapter.clone()), cx,
            ).is_err(), "the canonical late-registration restriction remains intact");
            assert!(
                ensure_desktop_component_lifecycle_registration(cx)?
                    .active_adapter()
                    .is_err()
            );
            ensure_native_sim_assets(&profile, true, temporary.path(), cx)?;
            init_sim_component_host(true, cx)?;
            Ok(())
        })?;
        let router = cx.update(|cx| cx.global::<SimComponentHostGlobal>().router.clone());
        let receiver = router.subscribe_execution_registry_bundles()?;
        receiver.try_recv()?;
        let registrations = std::cell::Cell::new(0);
        let binding = NativeSimRuntimeBinding::new(&profile, &security);
        let recovery_reload = cx.update(|cx| -> anyhow::Result<_> {
            sim_ui::clear_initialization_error(cx);
            let reload = bind_native_sim_host_or_wait_for_inventory(
                &profile,
                binding.clone(),
                cx,
                |_, _| {
                    registrations.set(registrations.get() + 1);
                    Ok(())
                },
            )?
            .ok_or_else(|| anyhow::anyhow!("recovery must request the canonical reload"))?;
            assert_eq!(registrations.get(), 0);
            assert!(cx.try_global::<NativeSimRuntimeBinding>() == Some(&binding));
            assert!(!cx.global::<SimComponentHostGlobal>().accepted_inventory);
            assert!(
                !sim_ui::execution_ui_model(cx)
                    .ok_or_else(|| anyhow::anyhow!("execution UI missing"))?
                    .read(cx)
                    .runtime_controller_available()
            );
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(profile.id), cx);
            Ok(reload)
        })?;
        cx.executor().advance_clock(Duration::from_secs(1));
        cx.executor().run_until_parked();
        recovery_reload.await?;
        assert!(store.read_with(cx, |store, _| store.component_adapter_errors().is_empty()));
        let candidate =
            ExtensionStore::canonical_component_inventory_candidate(filesystem, &extensions_root)
                .await?;
        assert_eq!(
            adapter
                .accepted_candidate_identity
                .lock()
                .map_err(|_| anyhow::anyhow!("candidate state unavailable"))?
                .as_ref(),
            Some(candidate.identity())
        );
        let current_bundle = router.active_execution_registry_bundle()?;
        let mut latest = None;
        while let Ok(bundle) = receiver.try_recv() {
            latest = Some(bundle);
        }
        let notification =
            latest.ok_or_else(|| anyhow::anyhow!("real reload emitted no bundle"))?;
        assert_eq!(
            notification.identity_sha256(),
            current_bundle.identity_sha256()
        );
        cx.update(|cx| -> anyhow::Result<()> {
            let selection = native_sim_registry_event_profile(
                cx.global::<SettingsStore>().merged_settings(),
                cx.try_global::<NativeSimRuntimeBinding>(),
                &profile.id.to_string(),
                &security,
                &notification,
                &current_bundle,
            );
            rebind_native_sim_registry_event(selection, cx, |_, _| {
                registrations.set(registrations.get() + 1);
                Ok(())
            })?;
            assert_eq!(registrations.get(), 1);
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(profile.id), cx);
            assert!(cx.global::<SimComponentHostGlobal>().accepted_inventory);
            assert!(cx.try_global::<NativeSimRuntimeBinding>() == Some(&binding));
            assert!(sim_ui::initialization_error(cx).is_none());
            Ok(())
        })?;
        Ok(())
    }

    #[cfg(feature = "sim")]
    #[gpui::test]
    async fn desktop_component_registry_waits_for_an_accepted_inventory_bundle(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        desktop_component_registry_waits_for_an_accepted_inventory_bundle_result(cx)
            .await
            .expect("desktop component registry lifecycle should converge");
    }

    #[cfg(feature = "sim")]
    async fn desktop_component_registry_waits_for_an_accepted_inventory_bundle_result(
        cx: &mut TestAppContext,
    ) -> anyhow::Result<()> {
        let profile_id = sim_ui::LOCAL_EXECUTION_PROFILE_ID.0.to_string();
        let host = sim_plugin_host::ComponentHost::new(
            extension_host::ComponentRuntime::no_wasi()?,
            sim_runtime::PluginTrustPolicy::default(),
            sim_runtime::PermissionPolicy::new(profile_id, std::iter::empty())?,
            sim_plugin_host::ComponentExecutionBoundary::conformance_in_process(Arc::new(
                sim_plugin_host::UnavailablePluginCapabilityServices,
            )),
            sim_plugin_host::ComponentLimits::default(),
            sim_runtime::generated_native_node_registry_projection(None)?,
        )?;
        let router = sim_plugin_host::ComponentHostRouter::with_initial_generation(
            host,
            sim_runtime::DEFAULT_COMPONENT_REGISTRY_GENERATION,
        )?;
        assert!(accepted_desktop_component_registry_bundle(false, None, &router).is_err());

        let filesystem = FakeFs::new(cx.executor());
        let extensions_dir = Path::new("/task433-desktop-inventory");
        filesystem
            .insert_tree(extensions_dir.join("installed"), json!({}))
            .await;
        filesystem
            .insert_file(
                extensions_dir.join("index.json"),
                serde_json::to_vec(&extension_host::ExtensionIndex::default())?,
            )
            .await;
        let candidate = extension_host::ExtensionStore::canonical_component_inventory_candidate(
            filesystem,
            extensions_dir,
        )
        .await?;
        let accepted_candidate_identity = Arc::new(std::sync::Mutex::new(None));
        let adapter = DesktopComponentLifecycleAdapter {
            router: router.clone(),
            accepted_candidate_identity: accepted_candidate_identity.clone(),
        };
        extension_host::ComponentLifecycleAdapter::synchronize_candidate(&adapter, candidate)
            .await
            .map_err(anyhow::Error::msg)?;
        let identity = accepted_candidate_identity
            .lock()
            .map_err(|_| anyhow::anyhow!("desktop candidate identity is unavailable"))?
            .clone();
        let accepted = accepted_desktop_component_registry_bundle(true, identity.clone(), &router)?;
        assert_eq!(
            accepted
                .registry_bundle
                .worker_deployment()
                .begin()
                .generation()
                .get(),
            sim_runtime::DEFAULT_COMPONENT_REGISTRY_GENERATION
        );
        let replay = accepted_desktop_component_registry_bundle(true, identity.clone(), &router)?;
        assert_eq!(
            replay.registry_bundle.identity_sha256(),
            accepted.registry_bundle.identity_sha256()
        );
        assert_eq!(replay.candidate_identity, accepted.candidate_identity);
        assert_eq!(
            replay
                .registry_bundle
                .provider_registry()
                .map(sim_runtime::NativeProviderRegistryPin::identity_sha256),
            accepted
                .registry_bundle
                .provider_registry()
                .map(sim_runtime::NativeProviderRegistryPin::identity_sha256)
        );
        let replacement_generation = sim_runtime::DEFAULT_COMPONENT_REGISTRY_GENERATION
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("component generation overflowed"))?;
        let replacement_host = sim_plugin_host::ComponentHost::new(
            extension_host::ComponentRuntime::no_wasi()?,
            sim_runtime::PluginTrustPolicy::default(),
            sim_runtime::PermissionPolicy::new(
                sim_ui::LOCAL_EXECUTION_PROFILE_ID.0.to_string(),
                std::iter::empty(),
            )?,
            sim_plugin_host::ComponentExecutionBoundary::conformance_in_process(Arc::new(
                sim_plugin_host::UnavailablePluginCapabilityServices,
            )),
            sim_plugin_host::ComponentLimits::default(),
            sim_runtime::generated_native_node_registry_projection(None)?,
        )?;
        router.replace_with_initial_generation(replacement_host, replacement_generation)?;
        let active = accepted_desktop_component_registry_bundle(true, identity, &router)?;
        assert_eq!(
            active.candidate_identity.as_sha256(),
            accepted.candidate_identity.as_sha256()
        );
        assert_eq!(
            active
                .registry_bundle
                .worker_deployment()
                .begin()
                .generation()
                .get(),
            replacement_generation
        );
        let source = include_str!("zed.rs");
        let register = source
            .find("sim_ui::register_native_execution_services(config, registry_bundle, cx)")
            .ok_or_else(|| anyhow::anyhow!("desktop controller registration is absent"))?;
        let transition = source[register..]
            .find("activate_desktop_component_deployment(")
            .map(|offset| register + offset)
            .ok_or_else(|| anyhow::anyhow!("desktop deployment transition is absent"))?;
        let attach = source
            .find("private_worker_executor.attach_provider_worker_bridge(provider_worker_bridge)")
            .ok_or_else(|| anyhow::anyhow!("concrete private executor attachment is absent"))?;
        assert!(register < transition);
        assert!(attach < register);
        assert!(
            source.contains(
                "provider_worker_bridge: sim_runtime::NativeProviderWorkerBridgeAttachment"
            )
        );
        assert!(source.contains(
            "private_worker_executor: Arc<sim_plugin_host::PrivateWorkerPluginExecutor>"
        ));
        Ok(())
    }
    use theme::ThemeRegistry;
    use util::{
        path,
        rel_path::{RelPath, rel_path},
    };
    use workspace::MultiWorkspace;
    #[cfg(feature = "multiplayer-tools")]
    use workspace::PathList;
    use workspace::{
        NewFile, OpenOptions, OpenVisible, SERIALIZATION_THROTTLE_TIME, SaveIntent, SplitDirection,
        WorkspaceHandle,
        item::SaveOptions,
        item::{Item, ItemHandle},
        open_new, open_paths, pane,
    };

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn collaborative_project_review_registers_without_an_open_diff(cx: &mut TestAppContext) {
        init_test(cx);
        let project = Project::test(fs::FakeFs::new(cx.executor()), [], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        assert!(workspace.read_with(cx, |workspace, cx| {
            workspace.item_of_type::<ProjectDiff>(cx).is_none()
        }));
        cx.run_until_parked();
        let provider_id = workspace.read_with(cx, |workspace, cx| {
            assert!(workspace.collaborative_review().provider_available(
                workspace::collaborative_review::CollaborativeReviewSlot::ProjectChanges
            ));
            assert!(
                workspace.item_of_type::<ProjectDiff>(cx).is_none(),
                "background review registration must not open an editor tab"
            );
            workspace
                .collaborative_review()
                .selected_view()
                .expect("fresh startup must register project review")
                .downcast::<ProjectDiff>()
                .expect("startup provider must be the canonical native diff")
                .entity_id()
        });
        workspace.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, _| {
            assert_eq!(
                workspace
                    .collaborative_review()
                    .selected_view()
                    .expect("retained review")
                    .entity_id(),
                provider_id
            );
        });
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    #[derive(Clone)]
    struct CompositionConnection {
        agent_id: project::AgentId,
        authenticated: Arc<std::sync::atomic::AtomicBool>,
        auth_method: agent_client_protocol::schema::v1::AuthMethod,
        session_gate: Arc<
            parking_lot::Mutex<Option<futures::channel::oneshot::Receiver<Result<(), String>>>>,
        >,
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    impl CompositionConnection {
        fn new(agent_id: &str, authenticated: bool) -> Self {
            use agent_client_protocol::schema::v1 as acp;
            Self {
                agent_id: project::AgentId::new(agent_id),
                authenticated: Arc::new(std::sync::atomic::AtomicBool::new(authenticated)),
                auth_method: acp::AuthMethod::Agent(acp::AuthMethodAgent::new(
                    "composition-login",
                    "Test Login",
                )),
                session_gate: Arc::new(parking_lot::Mutex::new(None)),
            }
        }

        fn gate_session(&self) -> futures::channel::oneshot::Sender<Result<(), String>> {
            let (sender, receiver) = futures::channel::oneshot::channel();
            assert!(self.session_gate.lock().replace(receiver).is_none());
            sender
        }
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    impl acp_thread::AgentConnection for CompositionConnection {
        fn agent_id(&self) -> project::AgentId {
            self.agent_id.clone()
        }
        fn telemetry_id(&self) -> gpui::SharedString {
            "composition-test".into()
        }
        fn new_session(
            self: Rc<Self>,
            project: gpui::Entity<Project>,
            work_dirs: workspace::PathList,
            cx: &mut App,
        ) -> gpui::Task<gpui::Result<gpui::Entity<acp_thread::AcpThread>>> {
            use agent_client_protocol::schema::v1 as acp;
            if !self.authenticated.load(std::sync::atomic::Ordering::SeqCst) {
                return gpui::Task::ready(Err(acp_thread::AuthRequired::new()
                    .with_description("Sign in to continue".to_string())
                    .into()));
            }
            let gate = self.session_gate.lock().take();
            let action_log = cx.new(|_| action_log::ActionLog::new(project.clone()));
            let session_id = acp::SessionId::new(self.agent_id.0.to_string());
            let thread = cx.new(|cx| {
                acp_thread::AcpThread::new(
                    None,
                    None,
                    Some(work_dirs),
                    self,
                    project,
                    action_log,
                    session_id,
                    watch::Receiver::constant(acp::PromptCapabilities::new()),
                    cx,
                )
            });
            cx.spawn(async move |_| {
                if let Some(gate) = gate {
                    gate.await
                        .map_err(|_| anyhow::anyhow!("session gate dropped"))?
                        .map_err(anyhow::Error::msg)?;
                }
                Ok(thread)
            })
        }
        fn auth_methods(&self) -> &[agent_client_protocol::schema::v1::AuthMethod] {
            std::slice::from_ref(&self.auth_method)
        }
        fn authenticate(
            &self,
            method: agent_client_protocol::schema::v1::AuthMethodId,
            _cx: &mut App,
        ) -> gpui::Task<gpui::Result<()>> {
            if &method != self.auth_method.id() {
                return gpui::Task::ready(Err(anyhow::anyhow!("unknown authentication method")));
            }
            self.authenticated
                .store(true, std::sync::atomic::Ordering::SeqCst);
            gpui::Task::ready(Ok(()))
        }
        fn prompt(
            &self,
            _params: agent_client_protocol::schema::v1::PromptRequest,
            _cx: &mut App,
        ) -> gpui::Task<gpui::Result<agent_client_protocol::schema::v1::PromptResponse>> {
            use agent_client_protocol::schema::v1 as acp;
            if !self.authenticated.load(std::sync::atomic::Ordering::SeqCst) {
                return gpui::Task::ready(Err(acp_thread::AuthRequired::new().into()));
            }
            gpui::Task::ready(Ok(acp::PromptResponse::new(acp::StopReason::EndTurn)))
        }
        fn cancel(
            &self,
            _session_id: &agent_client_protocol::schema::v1::SessionId,
            _cx: &mut App,
        ) {
        }
        fn into_any(self: Rc<Self>) -> Rc<dyn std::any::Any> {
            self
        }
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    fn open_composition_connection(
        panel: &gpui::Entity<AgentPanel>,
        connection: CompositionConnection,
        cx: &mut VisualTestContext,
    ) {
        panel.update_in(cx, |panel, window, cx| {
            panel.open_external_thread_with_server(
                Rc::new(
                    agent_ui::test_support::StubAgentServer::new(connection)
                        .with_connection_agent_id(),
                ),
                window,
                cx,
            );
        });
        cx.run_until_parked();
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    async fn mounted_composition_panel(
        cx: &mut TestAppContext,
    ) -> (
        gpui::Entity<Workspace>,
        gpui::Entity<AgentPanel>,
        &mut VisualTestContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            language_model::LanguageModelRegistry::test(cx);
            load_default_keymap(cx);
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/composition-states", json!({"main.rs": "fn main() {}"}))
            .await;
        let project = Project::test(fs, ["/composition-states".as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        cx.simulate_resize(gpui::size(px(1200.), px(800.)));
        let asynchronous = cx.update(|window, cx| window.to_async(cx));
        let panel = AgentPanel::load(workspace.downgrade(), asynchronous)
            .await
            .expect("production panel load");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx)
        });
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.close_all_docks(window, cx)
        });
        cx.run_until_parked();
        (workspace, panel, cx)
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    #[track_caller]
    fn composition_frame(
        workspace: &gpui::Entity<Workspace>,
        cx: &mut VisualTestContext,
        state: Option<&'static str>,
        input: bool,
    ) {
        let caller = std::panic::Location::caller();
        cx.run_until_parked();
        cx.update(|window, cx| window.draw(cx).clear(cx));
        // Scroll callbacks consume measurements from the preceding frame's prepaint.
        for _ in 0..2 {
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.draw(cx).clear(cx);
            });
            cx.run_until_parked();
        }
        workspace.update_in(cx, |workspace, _window, cx| {
            for position in [
                workspace::dock::DockPosition::Left,
                workspace::dock::DockPosition::Bottom,
                workspace::dock::DockPosition::Right,
            ] {
                assert!(!workspace.is_dock_at_position_open(position, cx));
            }
            assert!(
                workspace
                    .collaborative_timeline_for_tests()
                    .view()
                    .is_some()
            );
            assert_eq!(
                workspace.collaborative_composer().focus_handle().is_some(),
                input,
                "composer registration for state {state:?} at {caller}"
            );
        });
        let timeline = cx
            .debug_bounds("COLLABORATIVE-NATIVE-ACP-TIMELINE")
            .expect("mounted canonical conversation");
        assert!(timeline.size.width > px(0.) && timeline.size.height > px(0.));
        for selector in [
            "CONVERSATION-LOADING",
            "CONVERSATION-LOAD-ERROR",
            "CONVERSATION-AUTH-REQUIRED",
        ] {
            let bounds = cx.debug_bounds(selector);
            assert_eq!(bounds.is_some(), state == Some(selector), "{selector}");
            if let Some(bounds) = bounds {
                assert!(bounds.size.width > px(0.) && bounds.size.height > px(0.));
            }
        }
        let composer = cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER");
        assert_eq!(composer.is_some(), input);
        if let Some(bounds) = composer {
            assert!(bounds.size.width > px(0.) && bounds.size.height > px(0.));
        }
    }

    #[cfg(all(
        unix,
        feature = "multiplayer-tools",
        feature = "multiplayer-test-support"
    ))]
    #[gpui::test]
    async fn collaborative_inline_terminal_shortcut_survives_keymap_reload(
        cx: &mut TestAppContext,
    ) {
        // The real PTY reader wakes the executor outside the deterministic dispatcher.
        cx.executor().allow_parking();
        let directory = tempfile::tempdir().expect("disposable real terminal cwd");
        init_test(cx);
        cx.update(|cx| {
            let shortcut = gpui::Keystroke::parse("ctrl-`").expect("terminal shortcut");
            let bindings_after_init = cx
                .all_bindings_for_input(std::slice::from_ref(&shortcut))
                .into_iter()
                .map(|binding| binding.action().name().to_owned())
                .collect::<Vec<_>>();
            language_model::LanguageModelRegistry::test(cx);
            let bindings_after_registry = cx
                .all_bindings_for_input(std::slice::from_ref(&shortcut))
                .into_iter()
                .map(|binding| binding.action().name().to_owned())
                .collect::<Vec<_>>();
            load_default_keymap(cx);
            let bindings_after_defaults = cx
                .all_bindings_for_input(std::slice::from_ref(&shortcut))
                .into_iter()
                .map(|binding| binding.action().name().to_owned())
                .collect::<Vec<_>>();
            println!(
                "INLINE_KEYBOARD_KEYMAP_BOOTSTRAP init={bindings_after_init:?} registry={bindings_after_registry:?} defaults={bindings_after_defaults:?}"
            );
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree(directory.path(), json!({"main.rs": "fn main() {}"}))
            .await;
        let project = Project::test(fs, [directory.path()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |workspace, _| workspace.workspace().clone());
        cx.simulate_resize(gpui::size(px(1200.), px(800.)));
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            assert!(
                workspace.panel::<AgentPanel>(cx).is_none(),
                "test initialization must not load an automatic Agent panel"
            );
        });
        let asynchronous = cx.update(|window, cx| window.to_async(cx));
        let panel = AgentPanel::load(workspace.downgrade(), asynchronous)
            .await
            .expect("canonical Agent panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx)
        });
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.close_all_docks(window, cx)
        });
        open_composition_connection(
            &panel,
            CompositionConnection::new("inline-keyboard", true),
            cx,
        );
        composition_frame(&workspace, cx, None, true);
        let thread = panel.read_with(cx, |panel, cx| {
            panel.active_thread_view(cx).expect("real thread")
        });
        let editor = thread.read_with(cx, |thread, _| thread.message_editor.clone());
        let draft = "Aλ🦀B";
        editor.update_in(cx, |editor, window, cx| editor.set_text(draft, window, cx));
        let keyboard_observation = cx.update(|_, cx| {
            cx.observe_keystrokes(|event, _, _| {
                if event.keystroke.key == "`" {
                    println!(
                        "INLINE_KEYBOARD_DISPATCH action={:?} contexts={:?}",
                        event.action.as_ref().map(|action| action.name()),
                        event.context_stack
                    );
                }
            })
        });
        let mut terminal_id = None;
        let mut original_process = None;
        for reload in [false, true] {
            cx.update(|_, cx| {
                if reload {
                    reload_keymaps(cx, Vec::new());
                }
            });
            editor.update_in(cx, |editor, window, cx| {
                editor.set_cursor_offset("Aλ".len(), window, cx);
                window.focus(&editor.focus_handle(cx), cx);
            });
            composition_frame(&workspace, cx, None, true);
            editor.read_with(cx, |editor, cx| {
                assert_eq!(
                    editor.text(cx),
                    draft,
                    "keyboard fixture retains Unicode draft"
                );
            });
            editor.update_in(cx, |editor, window, cx| {
                let keyboard_contexts = window.context_stack();
                let editor_focused = editor.focus_handle(cx).is_focused(window);
                let action = agent_ui::NewTerminalThread;
                let action_available = window.is_action_available(&action, cx);
                let project_supports_terminal = project.read(cx).supports_terminal(cx);
                let panel_supports_terminal = panel.read(cx).supports_terminal(cx);
                let registered_panel = workspace.read(cx).panel::<AgentPanel>(cx);
                let canonical_panel_matches = registered_panel.as_ref() == Some(&panel);
                println!(
                    "INLINE_KEYBOARD_PANEL registered={:?} mounted={:?} matches={canonical_panel_matches}",
                    registered_panel.as_ref().map(|panel| panel.entity_id()),
                    panel.entity_id()
                );
                assert!(
                    canonical_panel_matches,
                    "the rendered conversation panel must remain registered for the production Workspace action"
                );
                let agent_enabled = agent_settings::AgentSettings::get_global(cx).enabled;
                let ai_disabled = project::DisableAiSettings::get_global(cx).disable_ai;
                let base_keymap = *BaseKeymap::get_global(cx);
                let shortcut = gpui::Keystroke::parse("ctrl-`").expect("terminal shortcut");
                let (bound_actions, pending) = {
                    let keymap = cx.key_bindings();
                    let keymap = keymap.borrow();
                    let (bindings, pending) =
                        keymap.bindings_for_input(std::slice::from_ref(&shortcut), &keyboard_contexts);
                    (
                        bindings
                            .iter()
                            .map(|binding| binding.action().name().to_owned())
                            .collect::<Vec<_>>(),
                        pending,
                    )
                };
                println!(
                    "INLINE_KEYBOARD_PREFLIGHT reload={reload} editor_focused={editor_focused} contexts={keyboard_contexts:?} action_available={action_available} project_terminal={project_supports_terminal} panel_terminal={panel_supports_terminal} agent_enabled={agent_enabled} ai_disabled={ai_disabled} base_keymap={base_keymap:?} actions={bound_actions:?} pending={pending}"
                );
                assert!(editor_focused, "actual composer must own keyboard focus");
                assert!(
                    keyboard_contexts.iter().any(|context| context.contains("AcpThread")
                        && context.contains("CollaborativeConversation")),
                    "composer must be in the rendered collaborative conversation dispatch path"
                );
                assert!(action_available, "canonical NewTerminalThread handler must be available");
                assert!(
                    project_supports_terminal && panel_supports_terminal,
                    "mounted local project and canonical panel must support a real terminal"
                );
                assert!(
                    agent_enabled && !ai_disabled && base_keymap != BaseKeymap::None,
                    "fixture must retain enabled Agent and default keyboard settings"
                );
                assert!(!pending, "terminal shortcut must resolve without a chord");
                assert_eq!(
                    bound_actions.first().map(String::as_str),
                    Some(action.name()),
                    "actual rendered composer shortcut must resolve to canonical inline terminal action"
                );
            });
            cx.update(|_, cx| {
                SettingsStore::update_global(cx, |store, cx| {
                    store.update_user_settings(cx, |settings| {
                        settings.project.terminal.get_or_insert_default().shell =
                            Some(settings::Shell::WithArguments {
                                program: "/bin/sh".into(),
                                args: vec![
                                    "-c".into(),
                                    r#"/bin/stty -echo; printf 'INLINE_KEYBOARD_READY\n'; exec /bin/cat"#.into(),
                                ],
                                title_override: Some("Keymap fixture PTY".into()),
                            });
                    });
                });
            });
            cx.simulate_keystrokes("ctrl-`");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let terminal = loop {
                cx.run_until_parked();
                let entries = panel.read_with(cx, |panel, cx| panel.terminals(cx));
                assert!(
                    entries.len() <= 1,
                    "shortcut reuses one canonical inline execution"
                );
                if let Some(entry) = entries.first() {
                    assert_eq!(*terminal_id.get_or_insert(entry.id), entry.id);
                    let handles = project.read_with(cx, |project, _| {
                        project
                            .local_terminal_handles()
                            .iter()
                            .filter_map(|terminal| terminal.upgrade())
                            .collect::<Vec<_>>()
                    });
                    assert_eq!(
                        handles.len(),
                        1,
                        "inline shortcut must not create a dock terminal"
                    );
                    break handles.into_iter().next().expect("canonical project PTY");
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "actual shortcut did not create inline terminal after reload={reload}"
                );
                cx.executor()
                    .timer(std::time::Duration::from_millis(50))
                    .await;
            };
            composition_frame(&workspace, cx, None, true);
            loop {
                cx.run_until_parked();
                if terminal.read_with(cx, |terminal, _| {
                    terminal.get_content().contains("INLINE_KEYBOARD_READY")
                }) {
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "controlled PTY must be ready within the original terminal startup deadline: {:?}",
                    terminal.read_with(cx, |terminal, _| terminal.get_content())
                );
                cx.executor()
                    .timer(std::time::Duration::from_millis(50))
                    .await;
            }
            composition_frame(&workspace, cx, None, true);
            let timeline = cx
                .debug_bounds("COLLABORATIVE-NATIVE-ACP-TIMELINE")
                .expect("real timeline");
            let control = cx
                .debug_bounds("inline-terminal-release-focus")
                .expect("shortcut focuses inline PTY");
            assert!(
                timeline.contains(&control.center()),
                "terminal must be mounted inside conversation: timeline={timeline:?}, control={control:?}, reload={reload}"
            );
            terminal.read_with(cx, |terminal, _| {
                assert!(terminal.is_pty(), "production entry creates real PTY");
                let process = terminal
                    .pid_getter()
                    .expect("canonical child")
                    .fallback_pid();
                assert_eq!(
                    *original_process.get_or_insert(process),
                    process,
                    "reload shortcut reuses same child"
                );
                println!("INLINE_KEYBOARD_OWNED_PTY process_id={}", process.as_u32());
            });
            terminal.update(cx, |terminal, _| terminal.take_input_log());
            cx.simulate_keystrokes("x");
            cx.run_until_parked();
            assert_eq!(
                terminal
                    .update(cx, |terminal, _| terminal.take_input_log())
                    .concat(),
                b"x",
                "real keyboard goes to canonical PTY without refocusing"
            );
            editor.read_with(cx, |editor, cx| {
                assert_eq!(editor.text(cx), draft, "PTY typing must not edit draft")
            });
            cx.simulate_keystrokes("backspace");
            cx.run_until_parked();
            assert_eq!(
                terminal
                    .update(cx, |terminal, _| terminal.take_input_log())
                    .concat(),
                b"\x7f",
                "Backspace must edit the PTY input rather than the surrounding draft"
            );
            editor.read_with(cx, |editor, cx| {
                assert_eq!(
                    editor.text(cx),
                    draft,
                    "PTY Backspace preserves the Unicode draft"
                );
            });
            for keystroke in [
                "left",
                "right",
                "home",
                "end",
                "tab",
                "shift-tab",
                "shift-left",
                "shift-right",
            ] {
                cx.simulate_keystrokes(keystroke);
                cx.run_until_parked();
                assert!(
                    !terminal
                        .update(cx, |terminal, _| terminal.take_input_log())
                        .concat()
                        .is_empty(),
                    "focused PTY must receive {keystroke} through the canonical terminal mapping"
                );
                editor.read_with(cx, |editor, cx| {
                    assert_eq!(
                        editor.text(cx),
                        draft,
                        "PTY {keystroke} must not edit the draft"
                    );
                });
            }
            let history = cx.update(|_, cx| {
                agent_ui::terminal_thread_metadata_store::TerminalThreadMetadataStore::global(cx)
            });
            history.read_with(cx, |history, _| {
                let context = history
                    .entry(terminal_id.expect("terminal identity"))
                    .expect("real inline history")
                    .inline_context
                    .as_ref()
                    .expect("conversation association");
                assert_eq!(
                    context
                        .draft_position
                        .as_ref()
                        .expect("Unicode insertion position")
                        .byte_offset_for_source(draft)
                        .expect("same draft"),
                    "Aλ".len()
                );
            });
        }
        editor.update_in(cx, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx)
        });
        cx.update(|_, cx| {
            reload_keymaps(
                cx,
                vec![KeyBinding::new(
                    "ctrl-`",
                    gpui::NoAction,
                    Some("AcpThread && CollaborativeConversation"),
                )],
            )
        });
        composition_frame(&workspace, cx, None, true);
        let history = cx.update(|_, cx| {
            agent_ui::terminal_thread_metadata_store::TerminalThreadMetadataStore::global(cx)
        });
        let retained = history.read_with(cx, |history, _| {
            history
                .entry(terminal_id.expect("terminal identity"))
                .cloned()
        });
        cx.simulate_keystrokes("ctrl-`");
        composition_frame(&workspace, cx, None, true);
        assert_eq!(
            history.read_with(cx, |history, _| history
                .entry(terminal_id.expect("terminal identity"))
                .cloned()),
            retained,
            "explicit user NoAction preserves inline occurrence metadata"
        );
        editor.read_with(cx, |editor, cx| assert_eq!(editor.text(cx), draft));
        panel
            .update(cx, |panel, cx| {
                panel.request_stop_inline_terminal(terminal_id.expect("terminal identity"), cx)
            })
            .expect("selected canonical Stop");
        let process = original_process.expect("owned child");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            cx.run_until_parked();
            let process_id = process.as_u32();
            let status = cx
                .executor()
                .spawn(async move {
                    smol::process::Command::new("ps")
                        .args(["-p", &process_id.to_string(), "-o", "pid="])
                        .output()
                        .await
                })
                .await
                .expect("verify only owned PTY child cleanup");
            if status.status.code() == Some(1) {
                assert!(status.stdout.is_empty());
                break;
            }
            assert_eq!(status.status.code(), Some(0));
            assert!(
                std::time::Instant::now() < deadline,
                "selected Stop left owned child alive"
            );
            cx.executor()
                .timer(std::time::Duration::from_millis(50))
                .await;
        }
        println!(
            "INLINE_KEYBOARD_OWNED_PTY_ABSENT process_id={}",
            process.as_u32()
        );
        drop(keyboard_observation);
        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.disable_ai = Some(SaturatingBool(true));
                });
            });
        });
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            assert!(
                workspace.panel::<AgentPanel>(cx).is_none(),
                "real disabled-AI settings must still remove an explicitly mounted panel"
            );
        });
    }

    #[cfg(feature = "agentic-tools")]
    #[gpui::test]
    fn inline_terminal_keymap_respects_profiles_and_disabled_defaults(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            let key = gpui::Keystroke::parse("ctrl-`").expect("terminal shortcut");
            let workspace = gpui::KeyContext::parse("Workspace").expect("workspace context");
            let conversation = gpui::KeyContext::parse("AcpThread CollaborativeConversation")
                .expect("conversation context");
            let context = [workspace.clone(), conversation];
            let inline_action = agent_ui::NewTerminalThread.name();
            reload_keymaps(cx, Vec::new());
            let keymap = cx.key_bindings();
            let (bindings, _) = keymap
                .borrow()
                .bindings_for_input(std::slice::from_ref(&key), &context);
            #[cfg(feature = "multiplayer-tools")]
            assert_eq!(
                bindings.first().map(|binding| binding.action().name()),
                Some(inline_action)
            );
            #[cfg(not(feature = "multiplayer-tools"))]
            assert!(
                bindings
                    .iter()
                    .all(|binding| binding.action().name() != inline_action)
            );
            let (bindings, _) = keymap
                .borrow()
                .bindings_for_input(std::slice::from_ref(&key), &[workspace]);
            assert_eq!(
                bindings.first().map(|binding| binding.action().name()),
                Some("terminal_panel::Toggle")
            );
            let original_agent_settings = agent_settings::AgentSettings::get_global(cx).clone();
            let mut disabled_agent_settings = original_agent_settings.clone();
            disabled_agent_settings.enabled = false;
            agent_settings::AgentSettings::override_global(disabled_agent_settings, cx);
            reload_keymaps(cx, Vec::new());
            let keymap = cx.key_bindings();
            let (bindings, _) = keymap
                .borrow()
                .bindings_for_input(std::slice::from_ref(&key), &context);
            assert!(
                bindings
                    .iter()
                    .all(|binding| binding.action().name() != inline_action)
            );
            agent_settings::AgentSettings::override_global(original_agent_settings, cx);
            project::DisableAiSettings::override_global(
                project::DisableAiSettings { disable_ai: true },
                cx,
            );
            reload_keymaps(cx, Vec::new());
            let keymap = cx.key_bindings();
            let (bindings, _) = keymap
                .borrow()
                .bindings_for_input(std::slice::from_ref(&key), &context);
            assert!(
                bindings
                    .iter()
                    .all(|binding| binding.action().name() != inline_action)
            );
            project::DisableAiSettings::override_global(
                project::DisableAiSettings { disable_ai: false },
                cx,
            );
            BaseKeymap::override_global(BaseKeymap::None, cx);
            reload_keymaps(cx, Vec::new());
            let keymap = cx.key_bindings();
            let (bindings, _) = keymap
                .borrow()
                .bindings_for_input(std::slice::from_ref(&key), &context);
            assert!(
                bindings
                    .iter()
                    .all(|binding| binding.action().name() != inline_action)
            );
        });
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    #[gpui::test]
    async fn collaborative_conversation_pending_ready_clear_and_project_fences(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, cx) = mounted_composition_panel(cx).await;
        let connection = CompositionConnection::new("composition-pending", true);
        let gate = connection.gate_session();
        open_composition_connection(&panel, connection, cx);
        composition_frame(&workspace, cx, Some("CONVERSATION-LOADING"), false);
        let conversation = panel.read_with(cx, |panel, _| {
            panel
                .active_conversation_view()
                .expect("pending canonical conversation")
                .clone()
        });
        let provider = workspace.read_with(cx, |workspace, _| {
            workspace
                .collaborative_timeline_for_tests()
                .view()
                .expect("provider")
                .entity_id()
        });
        assert!(conversation.read_with(cx, |conversation, _| {
            conversation.collaborative_active_thread().is_none()
        }));
        gate.send(Ok(())).expect("deliver session");
        composition_frame(&workspace, cx, None, true);
        assert_eq!(
            workspace.read_with(cx, |workspace, _| workspace
                .collaborative_timeline_for_tests()
                .view()
                .expect("same conversation provider")
                .entity_id()),
            provider
        );
        let thread = panel.read_with(cx, |panel, cx| {
            panel.active_thread_view(cx).expect("loaded actual thread")
        });
        workspace.update_in(cx, |workspace, window, cx| {
            AgentPanel::reveal_conversation(workspace, true, window, cx);
            let editor = thread.read(cx).message_editor.clone();
            editor.update(cx, |editor, cx| {
                editor.set_text("Retained integrated input", window, cx)
            });
        });
        cx.run_until_parked();
        workspace.update_in(cx, |workspace, window, _cx| {
            assert!(
                workspace
                    .collaborative_composer()
                    .focus_handle()
                    .expect("ready input")
                    .is_focused(window)
            )
        });
        let other_fs = fs::FakeFs::new(cx.executor());
        other_fs
            .insert_tree(
                "/composition-foreign",
                json!({"main.rs": "fn foreign() {}"}),
            )
            .await;
        let other_project =
            Project::test(other_fs, ["/composition-foreign".as_ref()], &mut cx.cx).await;
        // The production adapter must refuse a panel from another workspace project.
        let (other_window, _) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(other_project, window, cx));
        let other_workspace = other_window.read_with(cx, |multi, _| multi.workspace().clone());
        cx.update(|_, cx| assert!(matches!(agent_ui::collaborative_timeline::CollaborativeTimelineAdapter::from_agent_panel(&panel, &other_workspace, cx), Err(agent_ui::collaborative_timeline::CollaborativeTimelineAdapterError::ProjectMismatch))));
        panel.update_in(cx, |panel, window, cx| {
            let thread_id = panel
                .active_thread_id(cx)
                .expect("disposable selected draft");
            panel.remove_thread_without_activating_draft(thread_id, window, cx);
        });
        cx.run_until_parked();
        cx.update(|window, cx| window.simulate_next_frame(cx));
        workspace.read_with(cx, |workspace, _| {
            assert!(
                workspace
                    .collaborative_timeline_for_tests()
                    .view()
                    .is_none()
            );
            assert!(workspace.collaborative_composer().focus_handle().is_none());
        });
        assert!(
            cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-TIMELINE")
                .is_none()
        );
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    #[gpui::test]
    async fn collaborative_conversation_load_error_and_stale_switch_stay_mounted(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, cx) = mounted_composition_panel(cx).await;
        let connection = CompositionConnection::new("composition-error", true);
        let gate = connection.gate_session();
        open_composition_connection(&panel, connection, cx);
        composition_frame(&workspace, cx, Some("CONVERSATION-LOADING"), false);
        gate.send(Err("composition session failure".to_string()))
            .expect("deliver load error");
        composition_frame(&workspace, cx, Some("CONVERSATION-LOAD-ERROR"), false);
        let error_conversation = panel.read_with(cx, |panel, _| {
            panel
                .active_conversation_view()
                .expect("error conversation")
                .entity_id()
        });
        let pending = CompositionConnection::new("composition-stale", true);
        let stale_gate = pending.gate_session();
        open_composition_connection(&panel, pending, cx);
        composition_frame(&workspace, cx, Some("CONVERSATION-LOADING"), false);
        open_composition_connection(
            &panel,
            CompositionConnection::new("composition-current", true),
            cx,
        );
        composition_frame(&workspace, cx, None, true);
        let selected = panel.read_with(cx, |panel, _| {
            panel
                .active_conversation_view()
                .expect("current conversation")
                .entity_id()
        });
        assert_ne!(selected, error_conversation);
        let current_provider = workspace.read_with(cx, |workspace, _| {
            workspace
                .collaborative_timeline_for_tests()
                .view()
                .expect("current provider")
                .entity_id()
        });
        stale_gate
            .send(Err("late stale session failure".to_string()))
            .expect("retained old pending load");
        composition_frame(&workspace, cx, None, true);
        assert_eq!(
            panel.read_with(cx, |panel, _| panel
                .active_conversation_view()
                .expect("current conversation retained")
                .entity_id()),
            selected
        );
        assert_eq!(
            workspace.read_with(cx, |workspace, _| workspace
                .collaborative_timeline_for_tests()
                .view()
                .expect("same current provider")
                .entity_id()),
            current_provider
        );
    }

    #[cfg(all(feature = "multiplayer-tools", feature = "multiplayer-test-support"))]
    #[gpui::test]
    async fn collaborative_conversation_auth_required_and_same_thread_reauthentication_gate_input(
        cx: &mut TestAppContext,
    ) {
        let (workspace, panel, cx) = mounted_composition_panel(cx).await;
        let connection = CompositionConnection::new("composition-auth", false);
        open_composition_connection(&panel, connection.clone(), cx);
        composition_frame(&workspace, cx, Some("CONVERSATION-AUTH-REQUIRED"), false);
        let provider = workspace.read_with(cx, |workspace, _| {
            workspace
                .collaborative_timeline_for_tests()
                .view()
                .expect("auth provider")
                .entity_id()
        });
        let login = cx
            .debug_bounds("CONVERSATION-AUTH-METHOD-composition-login")
            .expect("canonical actual authentication button");
        cx.simulate_click(login.center(), Modifiers::none());
        composition_frame(&workspace, cx, None, true);
        let thread = panel.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("authenticated actual thread")
        });
        let thread_id = thread.entity_id();
        connection
            .authenticated
            .store(false, std::sync::atomic::Ordering::SeqCst);
        thread.update_in(cx, |thread, window, cx| {
            thread.message_editor.update(cx, |editor, cx| {
                editor.set_text("Expired session draft remains recoverable", window, cx)
            });
            thread.send(window, cx);
        });
        cx.run_until_parked();
        let acp_thread = thread.read_with(cx, |thread, _| thread.thread.clone());
        cx.condition(&acp_thread, |thread, _| {
            thread.status() == acp_thread::ThreadStatus::Idle
        })
        .await;
        acp_thread.read_with(cx, |thread, _| {
            assert_eq!(
                thread.execution_outcome(),
                acp_thread::ThreadExecutionOutcome::Failed
            )
        });
        composition_frame(&workspace, cx, None, true);
        let reauthenticate = cx
            .debug_bounds("CONVERSATION-THREAD-REAUTHENTICATE")
            .expect("real expired-prompt recovery action");
        cx.simulate_click(reauthenticate.center(), Modifiers::none());
        composition_frame(&workspace, cx, Some("CONVERSATION-AUTH-REQUIRED"), false);
        assert_eq!(
            panel.read_with(cx, |panel, cx| panel
                .active_thread_view(cx)
                .expect("retained thread while signed out")
                .entity_id()),
            thread_id
        );
        assert_eq!(
            workspace.read_with(cx, |workspace, _| workspace
                .collaborative_timeline_for_tests()
                .view()
                .expect("retained conversation provider")
                .entity_id()),
            provider
        );
        let conversation = panel.read_with(cx, |panel, _| {
            panel
                .active_conversation_view()
                .expect("same conversation")
                .clone()
        });
        assert!(conversation.read_with(cx, |conversation, _| {
            conversation.collaborative_active_thread().is_none()
        }));
        let login = cx
            .debug_bounds("CONVERSATION-AUTH-METHOD-composition-login")
            .expect("reauthentication button remains mounted with dock closed");
        cx.simulate_click(login.center(), Modifiers::none());
        composition_frame(&workspace, cx, None, true);
        panel.read_with(cx, |panel, _| {
            assert_eq!(
                panel
                    .active_conversation_view()
                    .expect("recovered conversation")
                    .entity_id(),
                conversation.entity_id()
            );
        });
        let recovered = panel.read_with(cx, |panel, cx| {
            panel.active_thread_view(cx).expect("recovered thread")
        });
        recovered.read_with(cx, |thread, cx| {
            let editor = thread.message_editor.read(cx);
            assert_eq!(editor.text(cx), "Expired session draft remains recoverable");
            workspace.read_with(cx, |workspace, cx| {
                assert_eq!(
                    workspace
                        .collaborative_composer()
                        .focus_handle()
                        .expect("recovered registered editor focus"),
                    editor.focus_handle(cx)
                );
                assert_eq!(
                    workspace
                        .collaborative_timeline_for_tests()
                        .view()
                        .expect("same canonical conversation provider")
                        .entity_id(),
                    provider
                );
            });
        });
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn collaborative_native_no_model_builtin_submit_preserves_production_composition(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            language_model::LanguageModelRegistry::test(cx);
            language_model::LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.set_default_model(None, cx);
            });
            load_default_keymap(cx);
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/no-model-production", json!({"main.rs": "fn main() {}"}))
            .await;
        let project = Project::test(fs, ["/no-model-production".as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        cx.simulate_resize(gpui::size(px(1200.), px(800.)));
        let asynchronous = cx.update(|window, cx| window.to_async(cx));
        let panel = AgentPanel::load(workspace.downgrade(), asynchronous)
            .await
            .expect("production agent panel load");
        let panel = workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            panel.update(cx, |panel, cx| {
                panel.activate_new_thread(
                    false,
                    agent_ui::AgentThreadSource::AgentPanel,
                    window,
                    cx,
                );
            });
            panel
        });
        cx.run_until_parked();
        let thread_view = panel.read_with(cx, |panel, cx| {
            panel
                .active_thread_view(cx)
                .expect("canonical native draft")
        });
        let message_editor = thread_view.read_with(cx, |view, cx| {
            assert!(
                view.thread
                    .read(cx)
                    .connection()
                    .clone()
                    .downcast::<agent::NativeAgentConnection>()
                    .is_some()
            );
            let connection = view
                .thread
                .read(cx)
                .connection()
                .clone()
                .downcast::<agent::NativeAgentConnection>()
                .expect("real native connection");
            let thread = connection
                .thread(view.thread.read(cx).session_id(), cx)
                .expect("real native session");
            assert!(thread.read(cx).model().is_none(), "no configured model");
            view.message_editor.clone()
        });
        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, cx| {
            let thread_id = panel
                .read(cx)
                .active_thread_id(cx)
                .expect("canonical native draft");
            assert_eq!(
                workspace.collaborative_navigation().current(),
                Some(
                    &workspace::collaborative_navigation::CollaborativeNavigationTarget::thread(
                        thread_id.to_key_string(),
                    )
                ),
                "entering Multiplayer must select its mounted canonical conversation",
            );
        });
        let focus_transitions = Rc::new(RefCell::new(Vec::new()));
        let _focus_observation = cx.update(|window, cx| {
            let thread_view = thread_view.clone();
            let focus_transitions = focus_transitions.clone();
            window.on_focus_out(&message_editor.focus_handle(cx), cx, move |_, window, cx| {
                let view = thread_view.read(cx);
                focus_transitions.borrow_mut().push(format!(
                    "actual={:?}, entries={}, rows={}, generating={}, loading={}, last_row={:?}, viewport={:?}",
                    window.focused(cx),
                    view.thread.read(cx).entries().len(),
                    view.list_state.item_count(),
                    view.generating_indicator_in_list,
                    view.is_loading_contents,
                    view.list_state.bounds_for_item(view.list_state.item_count().saturating_sub(1)),
                    view.list_state.viewport_bounds(),
                ));
            })
        });
        let submit_key = if cfg!(target_os = "macos") {
            "cmd-enter"
        } else {
            "ctrl-enter"
        };
        for expected_entries in [1, 2] {
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.close_all_docks(window, cx);
                AgentPanel::reveal_conversation(workspace, true, window, cx);
            });
            cx.run_until_parked();
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.simulate_next_frame(cx);
            });
            workspace.update_in(cx, |workspace, window, cx| {
                assert!(
                    workspace
                        .collaborative_timeline_for_tests()
                        .view()
                        .is_some(),
                    "production PanelAdded callbacks must register the actual conversation"
                );
                let focus = workspace
                    .collaborative_composer()
                    .focus_handle()
                    .expect("production callbacks register the canonical input");
                assert!(focus.is_focused(window));
                assert_eq!(focus, message_editor.focus_handle(cx));
            });
            message_editor.update_in(cx, |editor, window, cx| {
                editor.clear(window, cx);
                editor.insert_text(
                    "Disposable no-model production-composition submission",
                    window,
                    cx,
                );
            });
            let pane_focus = workspace.read_with(cx, |workspace, cx| {
                let pane_focus = workspace.active_pane().focus_handle(cx);
                assert_ne!(pane_focus, message_editor.focus_handle(cx));
                if expected_entries == 2 {
                    assert!(
                        workspace.is_being_followed(workspace::CollaboratorId::Agent),
                        "the first actual CmdEnter keeps canonical agent following active",
                    );
                }
                pane_focus
            });
            cx.simulate_keystrokes(submit_key);
            workspace.update_in(cx, |workspace, window, cx| {
                assert!(workspace.is_being_followed(workspace::CollaboratorId::Agent));
                assert!(
                    message_editor.focus_handle(cx).is_focused(window),
                    "actual submission must preserve canonical keyboard input",
                );
                assert!(
                    !pane_focus.is_focused(window),
                    "hidden native pane must not take focus"
                );
            });
            cx.run_until_parked();
            let thread = thread_view.read_with(cx, |view, _| view.thread.clone());
            cx.condition(&thread, |thread, _| {
                thread.status() == acp_thread::ThreadStatus::Idle
            })
            .await;
            cx.update(|window, cx| {
                window.simulate_next_frame(cx);
                window.simulate_next_frame(cx);
            });
            thread_view.read_with(cx, |view, cx| {
                assert_eq!(
                    view.thread.read(cx).status(),
                    acp_thread::ThreadStatus::Idle
                );
                assert_eq!(
                    view.thread.read(cx).execution_outcome(),
                    acp_thread::ThreadExecutionOutcome::Failed
                );
                assert_eq!(view.thread.read(cx).entries().len(), expected_entries);
                assert_eq!(view.message_editor.entity_id(), message_editor.entity_id());
            });
            panel.read_with(cx, |panel, cx| {
                assert_eq!(
                    panel
                        .active_thread_view(cx)
                        .expect("same parent view")
                        .entity_id(),
                    thread_view.entity_id()
                );
            });
            cx.simulate_keystrokes("x");
            message_editor.read_with(cx, |editor, cx| {
                assert_eq!(
                    editor.text(cx),
                    "x",
                    "typing after a failed no-model turn must reach the canonical input without refocusing",
                );
            });
            workspace.update_in(cx, |workspace, window, cx| {
                for position in [
                    workspace::dock::DockPosition::Left,
                    workspace::dock::DockPosition::Bottom,
                    workspace::dock::DockPosition::Right,
                ] {
                    assert!(!workspace.is_dock_at_position_open(position, cx));
                }
                assert!(
                    workspace
                        .collaborative_composer()
                        .focus_handle()
                        .expect("retained canonical input")
                        .is_focused(window),
                    "submission must retain input focus; actual={:?}, input={:?}, workspace={:?}, transitions={:?}",
                    window.focused(cx),
                    message_editor.focus_handle(cx),
                    workspace.focus_handle(cx),
                    focus_transitions.borrow(),
                );
            });
            assert!(
                cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-TIMELINE")
                    .is_some()
            );
            assert!(
                cx.debug_bounds("COLLABORATIVE-NATIVE-ACP-COMPOSER")
                    .is_some()
            );
        }
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn collaborative_review_registration(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = fs::FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let connection = Rc::new(acp_thread::StubAgentConnection::new());
        let thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new::<&Path>(&[]), cx)
            })
            .await
            .expect("stub agent thread should start");
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());

        workspace.update_in(cx, |workspace, window, cx| {
            ProjectDiff::deploy_at(workspace, None, window, cx);
        });
        cx.run_until_parked();

        let project_diff_id = workspace.read_with(cx, |workspace, cx| {
            workspace
                .item_of_type::<ProjectDiff>(cx)
                .expect("native project diff should be open")
                .entity_id()
        });
        let state = Rc::new(RefCell::new(CollaborativeReviewCompositionState::default()));
        let workspace_handle = workspace.clone();
        cx.update(|window, cx| {
            reconcile_collaborative_agent_review(
                &workspace_handle,
                Some(thread.clone()),
                &state,
                window,
                cx,
            );
        });

        workspace.read_with(cx, |workspace, _| {
            assert_eq!(
                workspace.collaborative_review().project().entity_id(),
                project.entity_id()
            );
            assert_eq!(
                workspace.collaborative_review().selected_slot(),
                Some(workspace::collaborative_review::CollaborativeReviewSlot::AgentChanges)
            );
            workspace
                .collaborative_review()
                .selected_view()
                .expect("agent review should be selected")
                .downcast::<agent_ui::AgentDiffPane>()
                .expect("selected agent review should remain the native pane");
        });

        cx.dispatch_action(workspace::SwitchToCollaborativeWorkspace);
        cx.run_until_parked();
        assert!(cx.debug_bounds("COLLABORATIVE-REVIEW-CONTENT").is_some());
        assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER").is_some());
        assert!(cx.debug_bounds("COLLABORATIVE-COMPOSER-EDITOR").is_none());

        workspace
            .update(cx, |workspace, cx| {
                workspace.select_collaborative_review_provider(
                    workspace::collaborative_review::CollaborativeReviewSlot::ProjectChanges,
                    cx,
                )
            })
            .expect("project review should be available");
        cx.run_until_parked();
        workspace.read_with(cx, |workspace, _| {
            let project_diff = workspace
                .collaborative_review()
                .selected_view()
                .expect("project review should be selected")
                .downcast::<ProjectDiff>()
                .expect("selected project review should remain the native diff");
            assert_eq!(project_diff.entity_id(), project_diff_id);
        });
        assert!(cx.debug_bounds("COLLABORATIVE-REVIEW-CONTENT").is_some());

        let layout_bounds = cx
            .debug_bounds("COLLABORATIVE-LAYOUT")
            .expect("collaborative layout should render");
        let review_toggle = cx
            .debug_bounds("COLLABORATIVE-TOP-BAR-REVIEW-LAYOUT")
            .expect("review toggle should render");
        cx.simulate_click(review_toggle.center(), gpui::Modifiers::default());
        cx.run_until_parked();
        assert!(cx.debug_bounds("COLLABORATIVE-REVIEW-CONTENT").is_none());
        assert_eq!(
            cx.debug_bounds("COLLABORATIVE-TIMELINE-REGION")
                .expect("collapsed timeline should render"),
            layout_bounds
        );
    }

    #[cfg(feature = "multiplayer-tools")]
    #[gpui::test]
    async fn collaborative_participant_provider_registration(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = fs::FakeFs::new(cx.executor());
        let project = Project::test(fs, [], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        let first_thread_view = cx.new(|_| Empty);
        let replacement_thread_view = cx.new(|_| Empty);
        let state = Rc::new(RefCell::new(CollaborativeReviewCompositionState::default()));

        cx.update(|_, cx| {
            apply_collaborative_participant_projection(&workspace, None, &state, cx);
        });
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| workspace
                .collaborative_participants()
                .state(cx)),
            CollaborativeParticipantProviderState::Unavailable
        );

        let unknown_view_data = CollaborativeParticipantViewData {
            participants: vec![
                workspace::collaborative_participants::CollaborativeParticipant::agent(
                    "agent:primary",
                    "Primary Agent",
                    None,
                    workspace::collaborative_participants::CollaborativeParticipantPresence::Online,
                ),
            ],
            execution: Some(
                workspace::collaborative_participants::CollaborativeExecutionStatus {
                    phase: workspace::collaborative_participants::CollaborativeExecutionPhase::Idle,
                    model: None,
                    runtime: Some("ACP".into()),
                    location:
                        workspace::collaborative_participants::CollaborativeExecutionLocation::Unknown,
                },
            ),
            task_title: Some("Primary task".into()),
            connection: CollaborativeConnectionState::Disconnected,
        };
        let current_view_data = Rc::new(RefCell::new(unknown_view_data.clone()));
        cx.update(|_, cx| {
            apply_collaborative_participant_projection(
                &workspace,
                Some(CollaborativeParticipantProjection {
                    thread_view_id: first_thread_view.entity_id(),
                    provider: CollaborativeParticipantProvider::from_reader(
                        project.clone(),
                        first_thread_view.entity_id(),
                        {
                            let current_view_data = current_view_data.clone();
                            move |_| {
                                CollaborativeParticipantProviderState::Ready(
                                    current_view_data.borrow().clone(),
                                )
                            }
                        },
                    ),
                }),
                &state,
                cx,
            );
        });
        let first_registration = state
            .borrow()
            .participant_registration
            .expect("active thread should register one participant provider");
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| workspace
                .collaborative_participants()
                .state(cx)),
            CollaborativeParticipantProviderState::Ready(unknown_view_data.clone())
        );
        let occupied = workspace.update(cx, |workspace, cx| {
            workspace.register_collaborative_participant_provider(
                CollaborativeParticipantProvider::new(
                    project.clone(),
                    replacement_thread_view.entity_id(),
                    CollaborativeParticipantProviderState::Unavailable,
                ),
                cx,
            )
        });
        assert_eq!(
            occupied,
            Err(
                workspace::collaborative_participants::CollaborativeParticipantProviderError::ProviderOccupied
            )
        );

        let updated_view_data = CollaborativeParticipantViewData {
            execution: Some(
                workspace::collaborative_participants::CollaborativeExecutionStatus {
                    phase: workspace::collaborative_participants::CollaborativeExecutionPhase::WaitingForUser,
                    model: Some("model-current".into()),
                    runtime: Some("ACP".into()),
                    location:
                        workspace::collaborative_participants::CollaborativeExecutionLocation::Local,
                },
            ),
            ..unknown_view_data
        };
        *current_view_data.borrow_mut() = updated_view_data.clone();
        cx.update(|_, cx| {
            apply_collaborative_participant_projection(
                &workspace,
                Some(CollaborativeParticipantProjection {
                    thread_view_id: first_thread_view.entity_id(),
                    provider: CollaborativeParticipantProvider::new(
                        project.clone(),
                        first_thread_view.entity_id(),
                        CollaborativeParticipantProviderState::Ready(updated_view_data.clone()),
                    ),
                }),
                &state,
                cx,
            );
        });
        assert_eq!(
            state.borrow().participant_registration,
            Some(first_registration),
            "same-thread metadata updates should retain the sole registration"
        );
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| workspace
                .collaborative_participants()
                .state(cx)),
            CollaborativeParticipantProviderState::Ready(updated_view_data)
        );

        let replacement_view_data = CollaborativeParticipantViewData {
            participants: vec![
                workspace::collaborative_participants::CollaborativeParticipant::agent(
                    "agent:replacement",
                    "Replacement Agent",
                    None,
                    workspace::collaborative_participants::CollaborativeParticipantPresence::Online,
                ),
            ],
            execution: None,
            task_title: Some("Replacement task".into()),
            connection: CollaborativeConnectionState::Disconnected,
        };
        cx.update(|_, cx| {
            apply_collaborative_participant_projection(
                &workspace,
                Some(CollaborativeParticipantProjection {
                    thread_view_id: replacement_thread_view.entity_id(),
                    provider: CollaborativeParticipantProvider::new(
                        project,
                        replacement_thread_view.entity_id(),
                        CollaborativeParticipantProviderState::Ready(replacement_view_data.clone()),
                    ),
                }),
                &state,
                cx,
            );
        });
        let replacement_registration = state
            .borrow()
            .participant_registration
            .expect("replacement thread should register a participant provider");
        assert_ne!(replacement_registration, first_registration);
        assert!(!workspace.update(cx, |workspace, cx| {
            workspace.unregister_collaborative_participant_provider(first_registration, cx)
        }));

        cx.update(|_, cx| {
            apply_collaborative_participant_projection(&workspace, None, &state, cx);
        });
        assert!(state.borrow().participant_registration.is_none());
        assert_eq!(
            workspace.read_with(cx, |workspace, cx| workspace
                .collaborative_participants()
                .state(cx)),
            CollaborativeParticipantProviderState::Unavailable
        );
    }

    #[cfg(feature = "sim")]
    #[test]
    fn sim_build_boundary_loads_split_native_defaults() {
        let content =
            <settings::SettingsContent as settings::RootUserSettings>::parse_json_with_comments(
                include_str!("../../../assets/settings/default.json"),
            )
            .expect("parse registered Zed defaults");
        assert!(content.sim_runtime.is_none());
        let profile = active_native_sim_profile_from_settings(&content)
            .expect("fall back to the Sim-only native runtime defaults");
        assert_eq!(profile.id, sim_runtime::DEFAULT_NATIVE_PROFILE_ID);
        assert_eq!(profile.device, sim_types::DeviceKind::Cpu);
        assert_eq!(profile.provider_scope, "local");
        assert!(!profile.api_host.enabled);
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 367)]
    fn native_sim_component_host_starts_from_the_comprehensive_generated_registry(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            init_sim_component_host(false, cx).expect("initialize native Sim component host");
            let actual = cx
                .global::<SimComponentHostGlobal>()
                .router
                .current()
                .expect("read current native Sim component generation")
                .registry_snapshot()
                .expect("read native Sim component registry");
            let expected = sim_runtime::generated_native_node_registry_projection(None)
                .expect("build comprehensive generated native registry");
            actual
                .validate_comprehensive_bindings()
                .expect("validate component-host native registry");
            assert_eq!(actual.descriptor_len(), expected.descriptor_len());
            assert!(
                expected
                    .descriptors()
                    .all(|(class_type, _)| actual.descriptor(class_type).is_some())
            );
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_asset_bootstrap_is_checked_and_precedes_host_services(cx: &mut TestAppContext) {
        let temporary = tempfile::tempdir().expect("create isolated native asset root");
        let content = default_native_sim_settings_content().expect("parse native defaults");
        let profile =
            active_native_sim_profile_from_settings(&content).expect("validate native profile");
        cx.update(|cx| -> anyhow::Result<()> {
            anyhow::ensure!(sim_ui::native_asset_services(cx).is_none());
            anyhow::ensure!(cx.try_global::<SimComponentHostGlobal>().is_none());
            let initial = ensure_native_sim_assets(&profile, false, temporary.path(), cx)?;
            let registered = sim_ui::native_asset_services(cx)
                .ok_or_else(|| anyhow::anyhow!("bootstrap did not register native assets"))?;
            assert_eq!(registered.profile_id(), profile.id.to_string());
            assert!(Arc::ptr_eq(&initial, &registered.assets()));
            for namespace in ["input", "output", "temporary", "model", "plugin"] {
                assert!(temporary.path().join(namespace).is_dir());
            }
            anyhow::ensure!(cx.try_global::<SimComponentHostGlobal>().is_none());
            let reused = ensure_native_sim_assets(&profile, false, temporary.path(), cx)?;
            assert!(Arc::ptr_eq(&initial, &reused));

            let mut invalid_profile = profile.clone();
            invalid_profile
                .model_roots
                .push(temporary.path().join("missing-models"));
            assert!(
                ensure_native_sim_assets(&invalid_profile, true, temporary.path(), cx).is_err()
            );
            let retained = sim_ui::native_asset_services(cx)
                .ok_or_else(|| anyhow::anyhow!("failed bootstrap removed native assets"))?;
            assert!(Arc::ptr_eq(&initial, &retained.assets()));
            anyhow::ensure!(cx.try_global::<SimComponentHostGlobal>().is_none());

            let model_root = temporary.path().join("configured-models");
            std::fs::create_dir(&model_root)?;
            let mut rebound_profile = profile.clone();
            rebound_profile.model_roots.push(model_root);
            let rebound = ensure_native_sim_assets(&rebound_profile, true, temporary.path(), cx)?;
            assert!(!Arc::ptr_eq(&initial, &rebound));
            let registered = sim_ui::native_asset_services(cx)
                .ok_or_else(|| anyhow::anyhow!("rebind did not register native assets"))?;
            assert!(Arc::ptr_eq(&rebound, &registered.assets()));
            let reused = ensure_native_sim_assets(&rebound_profile, false, temporary.path(), cx)?;
            assert!(Arc::ptr_eq(&rebound, &reused));
            anyhow::ensure!(cx.try_global::<SimComponentHostGlobal>().is_none());
            Ok(())
        })
        .expect("canonical asset bootstrap must finish before host services");
    }

    #[cfg(feature = "sim")]
    #[derive(Default)]
    struct NativeSimFailureControllerProbe {
        accepts: std::sync::atomic::AtomicUsize,
        shutdowns: std::sync::atomic::AtomicUsize,
    }

    #[cfg(feature = "sim")]
    impl sim_runtime::ExecutionController for NativeSimFailureControllerProbe {
        fn accept(
            &self,
            _command: &sim_runtime::ExecutionControlCommand,
            _attempt_id: Option<sim_types::AttemptId>,
        ) -> Result<(), sim_runtime::ExecutionFailure> {
            self.accepts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn shutdown(&self) -> Result<(), sim_runtime::ExecutionFailure> {
            self.shutdowns
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[cfg(feature = "sim")]
    struct NativeSimFailurePlanProbe(Arc<std::sync::atomic::AtomicUsize>);

    #[cfg(feature = "sim")]
    impl sim_ui::ExecutionPlanProvider for NativeSimFailurePlanProbe {
        fn compile(
            &self,
            _request: &sim_ui::ExecutionPlanRequest,
        ) -> Result<sim_runtime::CompiledPlan, sim_runtime::ExecutionFailure> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(sim_runtime::ExecutionFailure::new(
                "unexpected_stale_plan_compile",
                "A failed activation must remove the retained plan provider",
            ))
        }
    }

    #[cfg(feature = "sim")]
    fn install_native_sim_failure_probe(
        profile: &sim_runtime::NativeRuntimeProfile,
        security: &sim_runtime::NativePluginSecurityPolicy,
        cx: &mut App,
    ) -> (
        Arc<NativeSimFailureControllerProbe>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        cx.set_global(NativeSimRuntimeBinding::new(profile, security));
        init_sim_component_host(false, cx).expect("initialize canonical conformance host");
        let controller = Arc::new(NativeSimFailureControllerProbe::default());
        let compiled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let model = sim_ui::execution_ui_model(cx).expect("canonical execution UI model");
        model.update(cx, |model, cx| {
            model.register_runtime_controller(controller.clone(), cx);
            model.register_plan_provider(Arc::new(NativeSimFailurePlanProbe(compiled.clone())), cx);
            model
                .set_snapshot_status(
                    sim_types::ProfileId(profile.id),
                    sim_runtime::ExecutionDataSource::Live,
                    sim_runtime::ExecutionSnapshotStatus::Ready,
                    cx,
                )
                .expect("mark actual canonical presentation Ready");
            assert!(model.runtime_controller_available());
            assert!(model.plan_provider_available());
        });
        (controller, compiled)
    }

    #[cfg(feature = "sim")]
    fn assert_native_sim_failure_cannot_queue(profile_id: sim_types::ProfileId, cx: &mut App) {
        let model = sim_ui::execution_ui_model(cx).expect("retained execution UI model");
        model.update(cx, |model, cx| {
            assert!(!model.runtime_controller_available());
            assert!(!model.plan_provider_available());
            assert!(matches!(
                model.queue(
                    sim_ui::ExecutionPlanRequest {
                        profile_id,
                        document_identity: "failed-activation-workflow".to_owned(),
                        workflow_bytes: Vec::new(),
                        selected_output_nodes: std::collections::BTreeSet::new(),
                    },
                    0,
                    false,
                    cx,
                ),
                Err(sim_ui::ExecutionUiModelError::PlanProviderUnavailable)
            ));
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_same_profile_activation_failure_disconnects_stale_queue(cx: &mut TestAppContext) {
        let _app_state = init_test(cx);
        let temporary = tempfile::tempdir().expect("isolated native asset state");
        let content = default_native_sim_settings_content().expect("native defaults");
        let (profile, security) =
            active_native_sim_configuration_from_settings(&content).expect("checked configuration");
        cx.update(|cx| {
            let previous = ensure_native_sim_assets(&profile, false, temporary.path(), cx)
                .expect("checked original assets");
            let (controller, compiled) = install_native_sim_failure_probe(&profile, &security, cx);
            let (inventory, candidate, registry) = {
                let host = cx.global::<SimComponentHostGlobal>();
                (
                    host.accepted_inventory,
                    host.accepted_candidate_identity
                        .lock()
                        .expect("candidate identity")
                        .clone(),
                    host.router
                        .active_execution_registry_bundle()
                        .expect("canonical bundle")
                        .identity_sha256()
                        .to_owned(),
                )
            };
            let replacement = ensure_native_sim_assets(&profile, true, temporary.path(), cx)
                .expect("publish checked replacement before simulated host failure");
            assert!(!Arc::ptr_eq(&previous, &replacement));
            fail_native_sim_activation("injected host construction failure".to_owned(), cx);
            assert!(cx.try_global::<NativeSimRuntimeBinding>().is_none());
            assert!(sim_ui::initialization_error(cx).is_some());
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(profile.id), cx);
            assert_eq!(
                controller
                    .shutdowns
                    .load(std::sync::atomic::Ordering::SeqCst),
                1
            );
            assert_eq!(
                controller.accepts.load(std::sync::atomic::Ordering::SeqCst),
                0
            );
            assert_eq!(compiled.load(std::sync::atomic::Ordering::SeqCst), 0);
            let host = cx.global::<SimComponentHostGlobal>();
            assert_eq!(host.accepted_inventory, inventory);
            assert_eq!(
                *host
                    .accepted_candidate_identity
                    .lock()
                    .expect("retained candidate"),
                candidate
            );
            assert_eq!(
                host.router
                    .active_execution_registry_bundle()
                    .expect("retained bundle")
                    .identity_sha256(),
                registry
            );
            let current = sim_ui::native_asset_services(cx).expect("checked replacement retained");
            assert!(Arc::ptr_eq(&current.assets(), &replacement));
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_cross_profile_activation_failure_disconnects_both_queues(
        cx: &mut TestAppContext,
    ) {
        let _app_state = init_test(cx);
        let temporary = tempfile::tempdir().expect("isolated profile states");
        let content = default_native_sim_settings_content().expect("native defaults");
        let (previous_profile, security) =
            active_native_sim_configuration_from_settings(&content).expect("checked configuration");
        let mut next_profile = previous_profile.clone();
        next_profile.id = Uuid::from_u128(0x4_417);
        cx.update(|cx| {
            ensure_native_sim_assets(&previous_profile, false, &temporary.path().join("old"), cx)
                .expect("checked original assets");
            let (controller, compiled) =
                install_native_sim_failure_probe(&previous_profile, &security, cx);
            sim_ui::init_for_profile(sim_types::ProfileId(next_profile.id), cx);
            ensure_native_sim_assets(&next_profile, true, &temporary.path().join("new"), cx)
                .expect("checked next profile assets");
            fail_native_sim_activation("injected next-profile host failure".to_owned(), cx);
            assert!(cx.try_global::<NativeSimRuntimeBinding>().is_none());
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(previous_profile.id), cx);
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(next_profile.id), cx);
            let model = sim_ui::execution_ui_model(cx).expect("execution UI model");
            assert_eq!(
                model
                    .read(cx)
                    .snapshot(sim_types::ProfileId(previous_profile.id))
                    .expect("old Ready history retained")
                    .status,
                sim_runtime::ExecutionSnapshotStatus::Ready
            );
            assert_eq!(
                controller
                    .shutdowns
                    .load(std::sync::atomic::Ordering::SeqCst),
                1
            );
            assert_eq!(
                controller.accepts.load(std::sync::atomic::Ordering::SeqCst),
                0
            );
            assert_eq!(compiled.load(std::sync::atomic::Ordering::SeqCst), 0);
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_restore_previous_settings_forces_checked_asset_host_rebind(
        cx: &mut TestAppContext,
    ) {
        let _app_state = init_test(cx);
        let temporary = tempfile::tempdir().expect("isolated rebind state");
        let content = default_native_sim_settings_content().expect("native defaults");
        let (original_profile, security) =
            active_native_sim_configuration_from_settings(&content).expect("checked configuration");
        let mut changed_profile = original_profile.clone();
        let configured_root = temporary.path().join("configured-models");
        std::fs::create_dir(&configured_root).expect("create checked model root");
        changed_profile.model_roots.push(configured_root);
        cx.update(|cx| {
            let original_assets =
                ensure_native_sim_assets(&original_profile, false, temporary.path(), cx)
                    .expect("original checked assets");
            let (controller, _) =
                install_native_sim_failure_probe(&original_profile, &security, cx);
            let changed_assets =
                ensure_native_sim_assets(&changed_profile, true, temporary.path(), cx)
                    .expect("changed checked model roots");
            fail_native_sim_activation("injected changed-root host failure".to_owned(), cx);
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(original_profile.id), cx);
            let restored_binding = NativeSimRuntimeBinding::new(&original_profile, &security);
            let requires_rebind =
                cx.try_global::<NativeSimRuntimeBinding>() != Some(&restored_binding);
            assert!(
                requires_rebind,
                "restoring old settings must not reuse the old host fast path"
            );
            let recovered_assets =
                ensure_native_sim_assets(&original_profile, requires_rebind, temporary.path(), cx)
                    .expect("recreate original checked configuration");
            assert!(!Arc::ptr_eq(&recovered_assets, &changed_assets));
            assert!(!Arc::ptr_eq(&recovered_assets, &original_assets));
            init_sim_component_host(requires_rebind, cx)
                .expect("checked canonical host replacement");
            let registered = sim_ui::native_asset_services(cx).expect("restored checked assets");
            assert!(Arc::ptr_eq(&registered.assets(), &recovered_assets));
            cx.global::<SimComponentHostGlobal>()
                .router
                .active_execution_registry_bundle()
                .expect("verified replay after host replacement");
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(original_profile.id), cx);
            assert!(cx.try_global::<NativeSimRuntimeBinding>().is_none());
            assert_eq!(
                controller
                    .shutdowns
                    .load(std::sync::atomic::Ordering::SeqCst),
                1
            );
        });
    }

    #[cfg(feature = "sim")]
    fn native_sim_registry_test_configuration(
        profile_id: Uuid,
        memory_policy: &str,
        generation: u64,
        cx: &mut App,
    ) -> (
        sim_runtime::NativeRuntimeProfile,
        sim_runtime::NativePluginSecurityPolicy,
    ) {
        if !cx.has_global::<SettingsStore>() {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
        }
        let settings = serde_json::json!({
            "sim_runtime": {
                "active_profile": profile_id.to_string(),
                "profiles": [{
                    "id": profile_id.to_string(),
                    "name": "Registry subscriber regression",
                    "model_roots": [],
                    "device": "cpu",
                    "memory_policy": memory_policy,
                    "api_host_enabled": false,
                    "api_bind": "127.0.0.1:8188",
                    "plugin_policy": "approved_only",
                    "provider_scope": "local",
                    "plugin_security": { "component_registry_generation": generation }
                }]
            }
        });
        SettingsStore::update_global(cx, |store, cx| {
            store
                .set_user_settings(&settings.to_string(), cx)
                .result()
                .expect("install actual checked user settings");
        });
        active_native_sim_configuration_from_settings(
            cx.global::<SettingsStore>().merged_settings(),
        )
        .expect("parse current canonical runtime configuration")
    }

    #[cfg(feature = "sim")]
    fn native_sim_registry_test_host(
        security: &sim_runtime::NativePluginSecurityPolicy,
    ) -> sim_plugin_host::ComponentHost {
        sim_plugin_host::ComponentHost::new(
            extension_host::ComponentRuntime::no_wasi()
                .expect("create canonical component runtime"),
            security.trust_policy().clone(),
            security.permission_policy().clone(),
            sim_plugin_host::ComponentExecutionBoundary::conformance_in_process(Arc::new(
                sim_plugin_host::UnavailablePluginCapabilityServices,
            )),
            sim_plugin_host::ComponentLimits::default(),
            sim_runtime::generated_native_node_registry_projection(None)
                .expect("create canonical native registry"),
        )
        .expect("create checked component host")
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_registry_event_uses_current_same_profile_policy(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let profile_id = sim_ui::LOCAL_EXECUTION_PROFILE_ID.0;
            let (captured_profile, captured_security) =
                native_sim_registry_test_configuration(profile_id, "balanced", 1, cx);
            let router = sim_plugin_host::ComponentHostRouter::with_initial_generation(
                native_sim_registry_test_host(&captured_security),
                1,
            )
            .expect("create actual retained router");
            let receiver = router
                .subscribe_execution_registry_bundles()
                .expect("subscribe actual component lifecycle");
            let initial_bundle = receiver
                .try_recv()
                .expect("consume initial canonical bundle");
            let (current_profile, current_security) =
                native_sim_registry_test_configuration(profile_id, "conservative", 2, cx);
            let current_binding = NativeSimRuntimeBinding::new(&current_profile, &current_security);
            cx.set_global(current_binding);
            router
                .replace_with_initial_generation(
                    native_sim_registry_test_host(&current_security),
                    2,
                )
                .expect("replay and replace canonical component host");
            assert!(matches!(
                receiver.try_recv(),
                Err(async_channel::TryRecvError::Empty)
            ));
            smol::block_on(extension_host::ComponentLifecycleAdapter::synchronize(
                &router,
                Vec::new(),
            ))
            .expect("publish genuine lifecycle notification");
            let notification = receiver.try_recv().expect("receive actual registry event");
            let active_bundle = router
                .active_execution_registry_bundle()
                .expect("read active bundle");
            assert_ne!(
                initial_bundle.identity_sha256(),
                notification.identity_sha256()
            );
            let selected = native_sim_registry_event_profile(
                cx.global::<SettingsStore>().merged_settings(),
                cx.try_global::<NativeSimRuntimeBinding>(),
                &profile_id.to_string(),
                &current_security,
                &notification,
                &active_bundle,
            )
            .expect("select coherent current profile before inventory acceptance");
            assert_eq!(selected.memory_policy, current_profile.memory_policy);
            assert_ne!(selected.memory_policy, captured_profile.memory_policy);
            assert!(
                native_sim_registry_event_profile(
                    cx.global::<SettingsStore>().merged_settings(),
                    cx.try_global::<NativeSimRuntimeBinding>(),
                    &profile_id.to_string(),
                    &current_security,
                    &initial_bundle,
                    &active_bundle,
                )
                .is_err()
            );
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_registry_event_rejects_old_profile_after_replacement(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let original_id = sim_ui::LOCAL_EXECUTION_PROFILE_ID.0;
            let (original_profile, original_security) =
                native_sim_registry_test_configuration(original_id, "balanced", 1, cx);
            let original_binding =
                NativeSimRuntimeBinding::new(&original_profile, &original_security);
            let router = sim_plugin_host::ComponentHostRouter::with_initial_generation(
                native_sim_registry_test_host(&original_security),
                1,
            )
            .expect("create actual original profile router");
            let receiver = router
                .subscribe_execution_registry_bundles()
                .expect("subscribe lifecycle");
            let old_event = receiver
                .try_recv()
                .expect("retain original profile notification");
            let next_id = Uuid::from_u128(0x00000000000040008000000000000002);
            let (current_profile, current_security) =
                native_sim_registry_test_configuration(next_id, "conservative", 2, cx);
            router
                .replace_with_initial_generation(
                    native_sim_registry_test_host(&current_security),
                    2,
                )
                .expect("replace host with selected profile");
            smol::block_on(extension_host::ComponentLifecycleAdapter::synchronize(
                &router,
                Vec::new(),
            ))
            .expect("publish current profile registry");
            let notification = receiver.try_recv().expect("receive new profile event");
            let active_bundle = router
                .active_execution_registry_bundle()
                .expect("read checked bundle");
            let current_binding = NativeSimRuntimeBinding::new(&current_profile, &current_security);
            let settings = cx.global::<SettingsStore>().merged_settings();
            assert!(
                native_sim_registry_event_profile(
                    settings,
                    Some(&original_binding),
                    &next_id.to_string(),
                    &current_security,
                    &notification,
                    &active_bundle,
                )
                .is_err()
            );
            assert!(
                native_sim_registry_event_profile(
                    settings,
                    Some(&current_binding),
                    &next_id.to_string(),
                    &current_security,
                    &old_event,
                    &active_bundle,
                )
                .is_err()
            );
            assert_eq!(
                native_sim_registry_event_profile(
                    settings,
                    Some(&current_binding),
                    &next_id.to_string(),
                    &current_security,
                    &notification,
                    &active_bundle,
                )
                .expect("admit coherent new profile notification")
                .id,
                next_id
            );
            assert_ne!(notification.profile_id().0, original_profile.id);
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_registry_event_requires_retained_binding_and_exact_host_security(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let profile_id = sim_ui::LOCAL_EXECUTION_PROFILE_ID.0;
            let (old_profile, old_security) =
                native_sim_registry_test_configuration(profile_id, "balanced", 1, cx);
            let old_binding = NativeSimRuntimeBinding::new(&old_profile, &old_security);
            let (current_profile, current_security) =
                native_sim_registry_test_configuration(profile_id, "conservative", 2, cx);
            let current_binding = NativeSimRuntimeBinding::new(&current_profile, &current_security);
            let router = sim_plugin_host::ComponentHostRouter::with_initial_generation(
                native_sim_registry_test_host(&current_security),
                2,
            )
            .expect("create current checked host");
            let receiver = router
                .subscribe_execution_registry_bundles()
                .expect("subscribe lifecycle");
            let notification = receiver.try_recv().expect("receive checked initial event");
            let bundle = router
                .active_execution_registry_bundle()
                .expect("read checked identity");
            let before_identity = bundle.identity_sha256().to_owned();
            let settings = cx.global::<SettingsStore>().merged_settings();
            for binding in [None, Some(&old_binding)] {
                assert!(
                    native_sim_registry_event_profile(
                        settings,
                        binding,
                        &profile_id.to_string(),
                        &current_security,
                        &notification,
                        &bundle,
                    )
                    .is_err()
                );
            }
            assert!(
                native_sim_registry_event_profile(
                    settings,
                    Some(&current_binding),
                    &profile_id.to_string(),
                    &old_security,
                    &notification,
                    &bundle,
                )
                .is_err()
            );
            assert!(
                native_sim_registry_event_profile(
                    settings,
                    Some(&current_binding),
                    "00000000-0000-4000-8000-000000000002",
                    &current_security,
                    &notification,
                    &bundle,
                )
                .is_err()
            );
            assert_eq!(
                native_sim_registry_event_profile(
                    settings,
                    Some(&current_binding),
                    &profile_id.to_string(),
                    &current_security,
                    &notification,
                    &bundle,
                )
                .expect("coherent restored binding can admit current event")
                .memory_policy,
                current_profile.memory_policy
            );
            assert_eq!(
                router
                    .active_execution_registry_bundle()
                    .expect("retained identity")
                    .identity_sha256(),
                before_identity
            );
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_stale_registry_event_preserves_healthy_execution(cx: &mut TestAppContext) {
        let _app_state = init_test(cx);
        let content = default_native_sim_settings_content().expect("native defaults");
        let (profile, security) = active_native_sim_configuration_from_settings(&content)
            .expect("checked current profile");
        cx.update(|cx| {
            let (controller, compiled) = install_native_sim_failure_probe(&profile, &security, cx);
            let (router, accepted_inventory, candidate) = {
                let host = cx.global::<SimComponentHostGlobal>();
                (
                    host.router.clone(),
                    host.accepted_inventory,
                    host.accepted_candidate_identity
                        .lock()
                        .expect("candidate identity")
                        .clone(),
                )
            };
            let receiver = router
                .subscribe_execution_registry_bundles()
                .expect("real lifecycle receiver");
            let old_bundle = receiver.try_recv().expect("retain genuine old bundle");
            for _ in 0..2 {
                smol::block_on(extension_host::ComponentLifecycleAdapter::synchronize(
                    &router,
                    Vec::new(),
                ))
                .expect("advance actual router generation");
                receiver
                    .try_recv()
                    .expect("consume genuine new notification");
            }
            let current_bundle = router
                .active_execution_registry_bundle()
                .expect("current checked bundle");
            assert_ne!(
                old_bundle.identity_sha256(),
                current_bundle.identity_sha256()
            );
            let current_identity = current_bundle.identity_sha256().to_owned();
            let selection = native_sim_registry_event_profile(
                &content,
                cx.try_global::<NativeSimRuntimeBinding>(),
                &profile.id.to_string(),
                &security,
                &old_bundle,
                &current_bundle,
            );
            let mut activation_calls = 0;
            let result = rebind_native_sim_registry_event(selection, cx, |_, _| {
                activation_calls += 1;
                Err(anyhow::anyhow!("stale event must not enter activation"))
            });
            assert!(result.is_err());
            assert_eq!(activation_calls, 0);
            assert_eq!(
                cx.try_global::<NativeSimRuntimeBinding>(),
                Some(&NativeSimRuntimeBinding::new(&profile, &security))
            );
            assert!(sim_ui::initialization_error(cx).is_none());
            let model = sim_ui::execution_ui_model(cx).expect("canonical execution model");
            model.read_with(cx, |model, _| {
                assert!(model.runtime_controller_available());
                assert!(model.plan_provider_available());
                assert_eq!(
                    model
                        .snapshot(sim_types::ProfileId(profile.id))
                        .expect("ready snapshot")
                        .status,
                    sim_runtime::ExecutionSnapshotStatus::Ready
                );
            });
            assert_eq!(
                controller
                    .shutdowns
                    .load(std::sync::atomic::Ordering::SeqCst),
                0
            );
            assert_eq!(
                controller.accepts.load(std::sync::atomic::Ordering::SeqCst),
                0
            );
            assert_eq!(compiled.load(std::sync::atomic::Ordering::SeqCst), 0);
            let host = cx.global::<SimComponentHostGlobal>();
            assert_eq!(host.accepted_inventory, accepted_inventory);
            assert_eq!(
                *host
                    .accepted_candidate_identity
                    .lock()
                    .expect("retained candidate"),
                candidate
            );
            assert_eq!(
                host.router
                    .active_execution_registry_bundle()
                    .expect("retained checked registry")
                    .identity_sha256(),
                current_identity
            );
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 417)]
    fn native_sim_current_registry_activation_failure_disconnects_and_surfaces_recovery(
        cx: &mut TestAppContext,
    ) {
        let _app_state = init_test(cx);
        let content = default_native_sim_settings_content().expect("native defaults");
        let (profile, security) = active_native_sim_configuration_from_settings(&content)
            .expect("checked current profile");
        cx.update(|cx| {
            let (controller, compiled) = install_native_sim_failure_probe(&profile, &security, cx);
            let (router, candidate) = {
                let host = cx.global::<SimComponentHostGlobal>();
                (
                    host.router.clone(),
                    host.accepted_candidate_identity
                        .lock()
                        .expect("candidate identity")
                        .clone(),
                )
            };
            let receiver = router
                .subscribe_execution_registry_bundles()
                .expect("real lifecycle receiver");
            let old_bundle = receiver
                .try_recv()
                .expect("consume initial checked notification");
            let mut notification = None;
            for _ in 0..2 {
                smol::block_on(extension_host::ComponentLifecycleAdapter::synchronize(
                    &router,
                    Vec::new(),
                ))
                .expect("advance genuine router lifecycle");
                notification = Some(receiver.try_recv().expect("receive actual new bundle"));
            }
            let notification = notification.expect("two real lifecycle notifications");
            let current_bundle = router
                .active_execution_registry_bundle()
                .expect("current checked bundle");
            assert_ne!(
                old_bundle.identity_sha256(),
                current_bundle.identity_sha256()
            );
            let current_identity = current_bundle.identity_sha256().to_owned();
            let selection = native_sim_registry_event_profile(
                &content,
                cx.try_global::<NativeSimRuntimeBinding>(),
                &profile.id.to_string(),
                &security,
                &notification,
                &current_bundle,
            );
            assert!(selection.is_ok());
            let mut activation_calls = 0;
            let result = rebind_native_sim_registry_event(selection, cx, |selected_profile, _| {
                activation_calls += 1;
                assert_eq!(selected_profile.id, profile.id);
                Err(anyhow::anyhow!(
                    "injected current worker registry activation failure"
                ))
            });
            assert!(
                result
                    .expect_err("activation must fail")
                    .to_string()
                    .contains("injected current worker registry activation failure")
            );
            assert_eq!(activation_calls, 1);
            assert!(cx.try_global::<NativeSimRuntimeBinding>().is_none());
            assert!(
                sim_ui::initialization_error(cx)
                    .expect("UI recovery error")
                    .contains("injected current worker registry activation failure")
            );
            assert_native_sim_failure_cannot_queue(sim_types::ProfileId(profile.id), cx);
            assert_eq!(
                controller
                    .shutdowns
                    .load(std::sync::atomic::Ordering::SeqCst),
                1
            );
            assert_eq!(
                controller.accepts.load(std::sync::atomic::Ordering::SeqCst),
                0
            );
            assert_eq!(compiled.load(std::sync::atomic::Ordering::SeqCst), 0);
            assert_ne!(
                cx.try_global::<NativeSimRuntimeBinding>(),
                Some(&NativeSimRuntimeBinding::new(&profile, &security)),
                "recovery must force complete coherent asset/host/controller binding"
            );
            let host = cx.global::<SimComponentHostGlobal>();
            assert!(host.accepted_inventory);
            assert_eq!(
                *host
                    .accepted_candidate_identity
                    .lock()
                    .expect("retained candidate"),
                candidate
            );
            assert_eq!(
                host.router
                    .active_execution_registry_bundle()
                    .expect("retained replay/trust registry")
                    .identity_sha256(),
                current_identity
            );
        });
    }

    #[cfg(feature = "sim")]
    #[test]
    fn native_sim_runtime_binding_rebinds_same_profile_policy_changes() {
        fn binding(settings: &str) -> NativeSimRuntimeBinding {
            let content = <settings::SettingsContent as settings::RootUserSettings>::parse_json_with_comments(settings)
                .expect("parse native runtime settings");
            let (profile, plugin_security) =
                active_native_sim_configuration_from_settings(&content)
                    .expect("validate native runtime settings");
            NativeSimRuntimeBinding::new(&profile, &plugin_security)
        }

        let profile_id = sim_runtime::DEFAULT_NATIVE_PROFILE_ID;
        let baseline = binding(&format!(
            r#"{{
                "sim_runtime": {{
                    "active_profile": "{profile_id}",
                    "profiles": [{{
                        "id": "{profile_id}",
                        "name": "Native Local",
                        "model_roots": [],
                        "device": "cpu",
                        "memory_policy": "balanced",
                        "api_host_enabled": false,
                        "api_bind": "127.0.0.1:8188",
                        "plugin_policy": "approved_only",
                        "provider_scope": "local",
                        "plugin_security": {{
                            "component_registry_generation": 1
                        }}
                    }}]
                }}
            }}"#
        ));
        let changed_runtime_policy = binding(&format!(
            r#"{{
                "sim_runtime": {{
                    "active_profile": "{profile_id}",
                    "profiles": [{{
                        "id": "{profile_id}",
                        "name": "Native Local",
                        "model_roots": ["/tmp/native-models"],
                        "device": "cpu",
                        "memory_policy": "conservative",
                        "api_host_enabled": false,
                        "api_bind": "127.0.0.1:8188",
                        "plugin_policy": "approved_only",
                        "provider_scope": "local",
                        "plugin_security": {{
                            "component_registry_generation": 2
                        }}
                    }}]
                }}
            }}"#
        ));
        let cosmetic_only_change = binding(&format!(
            r#"{{
                "sim_runtime": {{
                    "active_profile": "{profile_id}",
                    "profiles": [{{
                        "id": "{profile_id}",
                        "name": "Renamed profile",
                        "model_roots": [],
                        "device": "cpu",
                        "memory_policy": "balanced",
                        "api_host_enabled": false,
                        "api_bind": "127.0.0.1:8188",
                        "plugin_policy": "approved_only",
                        "provider_scope": "local",
                        "plugin_security": {{
                            "component_registry_generation": 1
                        }},
                        "future_display_field": true
                    }}]
                }}
            }}"#
        ));

        assert_ne!(baseline, changed_runtime_policy);
        assert_eq!(baseline, cosmetic_only_change);

        let rocm_binding = |package_root: &str, public_key_hex: &str| {
            binding(&format!(
                r#"{{
                    "sim_runtime": {{
                        "active_profile": "{profile_id}",
                        "profiles": [{{
                            "id": "{profile_id}",
                            "name": "Native ROCm",
                            "model_roots": [],
                            "device": "rocm",
                            "memory_policy": "balanced",
                            "api_host_enabled": false,
                            "api_bind": "127.0.0.1:8188",
                            "plugin_policy": "approved_only",
                            "provider_scope": "local",
                            "rocm_package_root": "{package_root}",
                            "rocm_package_signer": "rocm.release",
                            "rocm_package_public_key_hex": "{public_key_hex}"
                        }}]
                    }}
                }}"#
            ))
        };
        let rocm_baseline = rocm_binding("/reviewed/rocm-a", &"11".repeat(32));
        let rotated_root = rocm_binding("/reviewed/rocm-b", &"11".repeat(32));
        let rotated_key = rocm_binding("/reviewed/rocm-a", &"22".repeat(32));
        assert_ne!(rocm_baseline, rotated_root);
        assert_ne!(rocm_baseline, rotated_key);

        let metal_binding = |package_root: &str, public_key_hex: &str| {
            binding(&format!(
                r#"{{
                    "sim_runtime": {{
                        "active_profile": "{profile_id}",
                        "profiles": [{{
                            "id": "{profile_id}",
                            "name": "Native Metal",
                            "model_roots": [],
                            "device": "metal",
                            "memory_policy": "balanced",
                            "api_host_enabled": false,
                            "api_bind": "127.0.0.1:8188",
                            "plugin_policy": "approved_only",
                            "provider_scope": "local",
                            "metal_package_root": "{package_root}",
                            "metal_package_signer": "metal.release",
                            "metal_package_public_key_hex": "{public_key_hex}"
                        }}]
                    }}
                }}"#
            ))
        };
        let metal_baseline = metal_binding("/reviewed/metal-a", &"33".repeat(32));
        let metal_rotated_root = metal_binding("/reviewed/metal-b", &"33".repeat(32));
        let metal_rotated_key = metal_binding("/reviewed/metal-a", &"44".repeat(32));
        assert_ne!(metal_baseline, metal_rotated_root);
        assert_ne!(metal_baseline, metal_rotated_key);

        let mlu_binding = |package_root: &str, public_key_hex: &str| {
            binding(&format!(
                r#"{{
                    "sim_runtime": {{
                        "active_profile": "{profile_id}",
                        "profiles": [{{
                            "id": "{profile_id}",
                            "name": "Native MLU",
                            "model_roots": [],
                            "device": "mlu",
                            "memory_policy": "balanced",
                            "api_host_enabled": false,
                            "api_bind": "127.0.0.1:8188",
                            "plugin_policy": "approved_only",
                            "provider_scope": "local",
                            "mlu_package_root": "{package_root}",
                            "mlu_package_signer": "mlu.release",
                            "mlu_package_public_key_hex": "{public_key_hex}"
                        }}]
                    }}
                }}"#
            ))
        };
        let mlu_baseline = mlu_binding("/reviewed/mlu-a", &"55".repeat(32));
        let mlu_rotated_root = mlu_binding("/reviewed/mlu-b", &"55".repeat(32));
        let mlu_rotated_key = mlu_binding("/reviewed/mlu-a", &"66".repeat(32));
        assert_ne!(mlu_baseline, mlu_rotated_root);
        assert_ne!(mlu_baseline, mlu_rotated_key);

        let npu_binding = |package_root: &str, public_key_hex: &str| {
            binding(&format!(
                r#"{{
                    "sim_runtime": {{
                        "active_profile": "{profile_id}",
                        "profiles": [{{
                            "id": "{profile_id}",
                            "name": "Native NPU",
                            "model_roots": [],
                            "device": "npu",
                            "memory_policy": "balanced",
                            "api_host_enabled": false,
                            "api_bind": "127.0.0.1:8188",
                            "plugin_policy": "approved_only",
                            "provider_scope": "local",
                            "npu_package_root": "{package_root}",
                            "npu_package_signer": "npu.release",
                            "npu_package_public_key_hex": "{public_key_hex}"
                        }}]
                    }}
                }}"#
            ))
        };
        let npu_baseline = npu_binding("/reviewed/npu-a", &"57".repeat(32));
        let npu_rotated_root = npu_binding("/reviewed/npu-b", &"57".repeat(32));
        let npu_rotated_key = npu_binding("/reviewed/npu-a", &"68".repeat(32));
        assert_ne!(npu_baseline, npu_rotated_root);
        assert_ne!(npu_baseline, npu_rotated_key);

        let cuda_binding = |package_root: &str, public_key_hex: &str| {
            binding(&format!(
                r#"{{
                    "sim_runtime": {{
                        "active_profile": "{profile_id}",
                        "profiles": [{{
                            "id": "{profile_id}",
                            "name": "Native CUDA",
                            "model_roots": [],
                            "device": "cuda",
                            "memory_policy": "balanced",
                            "api_host_enabled": false,
                            "api_bind": "127.0.0.1:8188",
                            "plugin_policy": "approved_only",
                            "provider_scope": "local",
                            "cuda_package_root": "{package_root}",
                            "cuda_package_signer": "cuda.release",
                            "cuda_package_public_key_hex": "{public_key_hex}"
                        }}]
                    }}
                }}"#
            ))
        };
        let cuda_baseline = cuda_binding("/reviewed/cuda-a", &"56".repeat(32));
        let cuda_rotated_root = cuda_binding("/reviewed/cuda-b", &"56".repeat(32));
        let cuda_rotated_key = cuda_binding("/reviewed/cuda-a", &"67".repeat(32));
        assert_ne!(cuda_baseline, cuda_rotated_root);
        assert_ne!(cuda_baseline, cuda_rotated_key);

        let xpu_binding = |package_root: &str, public_key_hex: &str| {
            binding(&format!(
                r#"{{
                    "sim_runtime": {{
                        "active_profile": "{profile_id}",
                        "profiles": [{{
                            "id": "{profile_id}",
                            "name": "Native XPU",
                            "model_roots": [],
                            "device": "xpu",
                            "memory_policy": "balanced",
                            "api_host_enabled": false,
                            "api_bind": "127.0.0.1:8188",
                            "plugin_policy": "approved_only",
                            "provider_scope": "local",
                            "xpu_package_root": "{package_root}",
                            "xpu_package_signer": "xpu.release",
                            "xpu_package_public_key_hex": "{public_key_hex}"
                        }}]
                    }}
                }}"#
            ))
        };
        let xpu_baseline = xpu_binding("/reviewed/xpu-a", &"69".repeat(32));
        let xpu_rotated_root = xpu_binding("/reviewed/xpu-b", &"69".repeat(32));
        let xpu_rotated_key = xpu_binding("/reviewed/xpu-a", &"7a".repeat(32));
        assert_ne!(xpu_baseline, xpu_rotated_root);
        assert_ne!(xpu_baseline, xpu_rotated_key);

        let directml_binding = |package_root: &str, public_key_hex: &str| {
            binding(&format!(
                r#"{{
                    "sim_runtime": {{
                        "active_profile": "{profile_id}",
                        "profiles": [{{
                            "id": "{profile_id}",
                            "name": "Native DirectML",
                            "model_roots": [],
                            "device": "directml",
                            "memory_policy": "balanced",
                            "api_host_enabled": false,
                            "api_bind": "127.0.0.1:8188",
                            "plugin_policy": "approved_only",
                            "provider_scope": "local",
                            "directml_package_root": "{package_root}",
                            "directml_package_signer": "directml.release",
                            "directml_package_public_key_hex": "{public_key_hex}"
                        }}]
                    }}
                }}"#
            ))
        };
        let directml_baseline = directml_binding("/reviewed/directml-a", &"77".repeat(32));
        let directml_rotated_root = directml_binding("/reviewed/directml-b", &"77".repeat(32));
        let directml_rotated_key = directml_binding("/reviewed/directml-a", &"88".repeat(32));
        assert_ne!(directml_baseline, directml_rotated_root);
        assert_ne!(directml_baseline, directml_rotated_key);
    }

    #[cfg(feature = "sim")]
    #[test]
    fn invalid_native_sim_settings_preserve_the_selected_error_scope() {
        let selected_profile_id = Uuid::from_u128(0x4_101);
        let invalid_settings = format!(
            r#"{{
                "sim_runtime": {{
                    "active_profile": "{selected_profile_id}",
                    "profiles": [{{
                        "id": "{selected_profile_id}",
                        "device": "future-device",
                        "api_bind": "0.0.0.0:8188"
                    }}]
                }}
            }}"#
        );
        let content =
            <settings::SettingsContent as settings::RootUserSettings>::parse_json_with_comments(
                &invalid_settings,
            )
            .expect("parse invalid native profile shape");

        assert_eq!(
            native_sim_profile_id_from_settings(&content),
            Some(sim_types::ProfileId(selected_profile_id))
        );
        assert!(active_native_sim_profile_from_settings(&content).is_err());
        assert_ne!(
            native_sim_profile_id_from_settings(&content),
            Some(sim_ui::LOCAL_EXECUTION_PROFILE_ID)
        );
    }

    async fn flush_workspace_serialization(
        window: &WindowHandle<MultiWorkspace>,
        cx: &mut TestAppContext,
    ) {
        let all_tasks = window
            .update(cx, |multi_workspace, window, cx| {
                let mut tasks = multi_workspace
                    .workspaces()
                    .map(|workspace| {
                        workspace.update(cx, |workspace, cx| {
                            workspace.flush_serialization(window, cx)
                        })
                    })
                    .collect::<Vec<_>>();
                tasks.push(multi_workspace.flush_serialization());
                tasks
            })
            .unwrap();

        futures::future::join_all(all_tasks).await;
    }

    #[cfg(feature = "sim")]
    fn validation_digest(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> String {
        const OFFSETS: [u64; 4] = [
            0xcbf29ce484222325,
            0x84222325cbf29ce4,
            0x9e3779b185ebca87,
            0x517cc1b727220a95,
        ];
        let mut lanes = OFFSETS;
        for part in parts {
            for byte in part.as_ref() {
                for (index, lane) in lanes.iter_mut().enumerate() {
                    *lane ^= u64::from(*byte).wrapping_add(index as u64);
                    *lane = lane.wrapping_mul(0x100000001b3);
                }
            }
        }
        lanes
            .into_iter()
            .map(|lane| format!("{lane:016x}"))
            .collect()
    }

    fn collect_menu_action_names(items: &[MenuItem], names: &mut Vec<String>) {
        for item in items {
            match item {
                MenuItem::Action { action, .. } => names.push(action.name().to_owned()),
                MenuItem::Submenu(menu) => collect_menu_action_names(&menu.items, names),
                MenuItem::Separator | MenuItem::SystemMenu(_) => {}
            }
        }
    }

    #[cfg(feature = "sim")]
    #[test]
    fn sim_keymap_loads_between_vim_and_specific_overrides() {
        let keymap_order = builtin_keymap_assets(BaseKeymap::JetBrains, true);
        let vim_position = keymap_order
            .iter()
            .position(|(path, _)| *path == VIM_KEYMAP_PATH)
            .expect("Vim keymap must be present when enabled");
        let sim_position = keymap_order
            .iter()
            .position(|(path, _)| *path == sim_ui::DEFAULT_SIM_KEYMAP_PATH)
            .expect("Sim keymap must be present");
        let override_position = keymap_order
            .iter()
            .position(|(path, _)| *path == SPECIFIC_OVERRIDES_KEYMAP_PATH)
            .expect("specific overrides must be present");
        assert!(vim_position < sim_position);
        assert_eq!(sim_position + 1, override_position);
        assert!(builtin_keymap_assets(BaseKeymap::None, true).is_empty());
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 16013)]
    fn sim_static_menu_has_registered_action_targets(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let menus = app_menus(cx);
            let sim_menu = menus
                .iter()
                .find(|menu| menu.name.as_ref() == "Sim")
                .expect("static application menus must include Sim");
            let mut menu_action_names = Vec::new();
            collect_menu_action_names(&sim_menu.items, &mut menu_action_names);
            assert!(!menu_action_names.is_empty());
            let mut expected_action_names = sim_ui::native_menu_action_names()
                .expect("canonical Sim menu registry must resolve")
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            menu_action_names.sort();
            expected_action_names.sort();
            assert_eq!(menu_action_names, expected_action_names);
            let registered_actions = cx
                .all_action_names()
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            for action_name in menu_action_names {
                assert_ne!(action_name, "NoAction");
                assert!(registered_actions.contains(action_name.as_str()));
            }
        });
    }

    #[cfg(feature = "sim")]
    #[gpui::test(seed = 16013)]
    fn val_gpui_013(cx: &mut TestAppContext) {
        const MAIN_SOURCE: &str = include_str!("main.rs");
        const ZED_SOURCE: &str = include_str!("zed.rs");
        const MENU_SOURCE: &str = include_str!("zed/app_menus.rs");

        assert!(!MAIN_SOURCE.contains("Application::new_inaccessible"));
        assert!(!MAIN_SOURCE.contains("ZED_EXPERIMENTAL_A11Y"));
        assert!(MAIN_SOURCE.contains("Application::with_platform(platform)"));
        assert!(
            MAIN_SOURCE.contains(
                "build_application_with_platform(gpui_platform::current_platform(false))"
            )
        );

        #[cfg(not(target_os = "macos"))]
        {
            let application =
                crate::build_application_with_platform(gpui_platform::current_platform(true));
            drop(application);
        }

        let (keymap_order, keymap_actions, user_override_evidence, menu_action_names) =
            cx.update(|cx| {
                cx.set_global(db::AppDatabase::test_new());
                workspace::AppState::test(cx);
                init_sim_ui(cx);
                init_sim_ui(cx);
                #[cfg(feature = "test-support")]
                assert_eq!(sim_ui::initialization_passes_for_test(cx), Some(1));

                let keymap_order = builtin_keymap_assets(BaseKeymap::JetBrains, true)
                    .into_iter()
                    .map(|(path, source)| format!("{}:{path}", source.name()))
                    .collect::<Vec<_>>();
                let sim_position = keymap_order
                    .iter()
                    .position(|entry| entry.ends_with(sim_ui::DEFAULT_SIM_KEYMAP_PATH))
                    .expect("Sim keymap must be present in the built-in order");
                let vim_position = keymap_order
                    .iter()
                    .position(|entry| entry.ends_with(VIM_KEYMAP_PATH))
                    .expect("Vim keymap must be present when enabled");
                assert!(vim_position < sim_position);
                let expected_override = format!("Default:{SPECIFIC_OVERRIDES_KEYMAP_PATH}");
                assert_eq!(
                    keymap_order.get(sim_position + 1).map(String::as_str),
                    Some(expected_override.as_str()),
                );
                assert!(builtin_keymap_assets(BaseKeymap::None, true).is_empty());

                load_default_keymap(cx);
                let keymap_actions = {
                    let keymap = cx.key_bindings();
                    let keymap = keymap.borrow();
                    keymap
                        .bindings()
                        .filter_map(|binding| {
                            binding
                                .predicate()
                                .filter(|predicate| predicate.to_string().contains("SimGraph"))
                                .map(|predicate| {
                                    assert!(predicate.to_string().contains("SimGraph"));
                                    binding.action().name().to_owned()
                                })
                        })
                        .collect::<Vec<_>>()
                };
                assert!(
                    !keymap_actions.is_empty(),
                    "the embedded Sim keymap must register scoped bindings"
                );

                reload_keymaps(
                    cx,
                    vec![KeyBinding::new(
                        "ctrl-enter",
                        zed_actions::About,
                        Some("SimGraph"),
                    )],
                );
                let keystroke = Keystroke::parse("ctrl-enter")
                    .expect("parse the known conflicting Sim keystroke");
                let mut graph_context = KeyContext::new_with_defaults();
                graph_context.add("SimGraph");
                let keymap = cx.key_bindings();
                let keymap = keymap.borrow();
                let (resolved_bindings, pending) =
                    keymap.bindings_for_input(&[keystroke], &[graph_context]);
                assert!(!pending);
                let winning_binding = resolved_bindings
                    .first()
                    .expect("the conflicting user binding must resolve");
                assert_eq!(winning_binding.action().name(), zed_actions::About.name());
                assert_eq!(
                    winning_binding.meta().map(KeybindSource::from_meta),
                    Some(KeybindSource::User)
                );
                assert!(
                    resolved_bindings
                        .iter()
                        .skip(1)
                        .any(|binding| binding.action().name() == "sim_shell::QueuePrompt")
                );
                let user_override_evidence = format!(
                    "ctrl-enter:{}:{}",
                    winning_binding.action().name(),
                    KeybindSource::User.name()
                );
                drop(keymap);

                let menus = app_menus(cx);
                let sim_menu_index = menus
                    .iter()
                    .position(|menu| menu.name.as_ref() == "Sim")
                    .expect("static application menus must include Sim");
                let window_menu_index = menus
                    .iter()
                    .position(|menu| menu.name.as_ref() == "Window")
                    .expect("static application menus must include Window");
                assert!(sim_menu_index < window_menu_index);
                let sim_menu: &Menu = menus
                    .get(sim_menu_index)
                    .expect("Sim menu index must remain valid");
                let mut menu_action_names = Vec::new();
                collect_menu_action_names(&sim_menu.items, &mut menu_action_names);
                assert!(
                    !menu_action_names.is_empty(),
                    "Sim menu must contain executable actions"
                );
                let mut expected_menu_action_names = sim_ui::native_menu_action_names()
                    .expect("canonical Sim menu registry must resolve")
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                menu_action_names.sort();
                expected_menu_action_names.sort();
                assert_eq!(
                    menu_action_names, expected_menu_action_names,
                    "production Sim menu must exactly project its authoritative registry"
                );
                let registered_actions = cx
                    .all_action_names()
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
                for action_name in keymap_actions.iter().chain(&menu_action_names) {
                    assert_ne!(action_name, "NoAction");
                    assert!(
                        registered_actions.contains(action_name.as_str()),
                        "{action_name} must be registered with GPUI"
                    );
                }

                (
                    keymap_order,
                    keymap_actions,
                    user_override_evidence,
                    menu_action_names,
                )
            });

        assert!(ZED_SOURCE.contains("init_sim_ui(cx);"));
        assert!(ZED_SOURCE.contains("sim_ui::init(cx);"));
        assert!(MENU_SOURCE.contains("sim_ui::sim_menu()"));

        let source_digest = validation_digest([MAIN_SOURCE, ZED_SOURCE, MENU_SOURCE]);
        let keymap_digest = validation_digest(
            keymap_order
                .iter()
                .chain(&keymap_actions)
                .map(String::as_bytes),
        );
        let user_override_digest = validation_digest([user_override_evidence.as_bytes()]);
        let menu_digest = validation_digest(menu_action_names.iter().map(String::as_bytes));
        let artifact = serde_json::json!({
            "validation_id": "VAL-GPUI-013",
            "environment": {
                "backend": "gpui-test",
                "platform": if cfg!(target_os = "macos") {
                    "test-app-and-compiled-production-bootstrap"
                } else {
                    "test-app-and-headless-bootstrap"
                },
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "feature": "test-support",
                "scheduler_seed": 16013,
                "iterations": "1"
            },
            "fixture_digests": {
                "production_sources": source_digest,
                "keymap_order_and_actions": keymap_digest,
                "user_keymap_precedence": user_override_digest,
                "menu_actions": menu_digest
            },
            "cases": [
                {
                    "name": "production-accessibility-bootstrap",
                    "passed": true,
                    "digest": source_digest
                },
                {
                    "name": "sim-initialization-is-idempotently-invoked",
                    "passed": true,
                    "digest": validation_digest([ZED_SOURCE])
                },
                {
                    "name": "sim-keymap-load-order-and-context-scope",
                    "passed": true,
                    "digest": keymap_digest
                },
                {
                    "name": "user-keymap-overrides-conflicting-sim-default",
                    "passed": true,
                    "digest": user_override_digest
                },
                {
                    "name": "sim-static-menu-actions-are-registered",
                    "passed": true,
                    "digest": menu_digest
                }
            ],
            "skipped": []
        });
        let artifact_directory =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/comfy-parity");
        std::fs::create_dir_all(&artifact_directory)
            .expect("create Sim parity validation artifact directory");
        std::fs::write(
            artifact_directory.join("val-gpui-013.json"),
            serde_json::to_vec_pretty(&artifact).expect("serialize VAL-GPUI-013 artifact"),
        )
        .expect("write VAL-GPUI-013 artifact");
    }

    #[gpui::test]
    async fn test_open_non_existing_file(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({
                    "a": {
                    },
                }),
            )
            .await;

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/a/new"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.read(|cx| cx.windows().len()), 1);

        let multi_workspace = cx.windows()[0].downcast::<MultiWorkspace>().unwrap();
        multi_workspace
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    assert!(workspace.active_item_as::<Editor>(cx).is_some())
                });
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_open_remote_from_existing_connection_reuses_window(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let app_state = init_test(cx);
        let executor = cx.executor();

        server_cx.update(|cx| {
            release_channel::init(Version::new(0, 0, 0), cx);
        });

        let (connection_options, server_session, connect_guard) =
            RemoteClient::fake_server(cx, server_cx);
        let remote_fs = FakeFs::new(server_cx.executor());
        remote_fs
            .insert_tree(
                path!("/"),
                json!({
                    "project": {},
                    "other-project": {},
                }),
            )
            .await;

        server_cx.update(HeadlessProject::init);
        let http_client = Arc::new(BlockedHttpClient);
        let node_runtime = NodeRuntime::unavailable();
        let languages = Arc::new(LanguageRegistry::new(server_cx.executor()));
        let extension_host_proxy = Arc::new(ExtensionHostProxy::new());
        let _headless = server_cx.new(|cx| {
            HeadlessProject::new(
                HeadlessAppState {
                    session: server_session,
                    fs: remote_fs,
                    http_client,
                    node_runtime,
                    languages,
                    extension_host_proxy,
                    startup_time: std::time::Instant::now(),
                },
                false,
                cx,
            )
        });
        drop(connect_guard);

        let mut async_cx = cx.to_async();
        open_remote_project(
            connection_options,
            vec![PathBuf::from(path!("/project"))],
            app_state,
            OpenOptions::default(),
            &mut async_cx,
        )
        .await
        .expect("opening the initial remote project should succeed");
        executor.run_until_parked();

        assert_eq!(cx.update(|cx| cx.windows().len()), 1);
        let window = cx.update(|cx| cx.windows()[0].downcast::<MultiWorkspace>().unwrap());

        window
            .update(cx, |multi_workspace, _, cx| {
                let workspace = multi_workspace.workspace().clone();
                workspace.update(cx, |workspace, cx| {
                    let remote_client = workspace
                        .project()
                        .read(cx)
                        .remote_client()
                        .expect("initial project should have a remote client");
                    remote_client.update(cx, |remote_client, cx| {
                        remote_client.force_server_not_running(cx);
                    });
                });
            })
            .unwrap();
        executor.run_until_parked();

        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, _cx| {
                    workspace.set_prompt_for_open_path(Box::new(|_, _, _, _| {
                        let (sender, receiver) = futures::channel::oneshot::channel();
                        sender
                            .send(Some(vec![PathBuf::from(path!("/other-project"))]))
                            .expect("path prompt receiver should be open");
                        receiver
                    }));
                });
                window.dispatch_action(
                    Box::new(zed_actions::OpenRemote {
                        from_existing_connection: true,
                        create_new_window: Some(false),
                    }),
                    cx,
                );
            })
            .unwrap();
        executor.run_until_parked();

        assert_eq!(
            cx.update(|cx| cx.windows().len()),
            1,
            "create_new_window: false should reuse the current window"
        );
        cx.simulate_prompt_answer("Cancel");
        executor.run_until_parked();
    }

    #[gpui::test]
    async fn test_open_paths_action(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({
                    "a": {
                        "aa": null,
                        "ab": null,
                    },
                    "b": {
                        "ba": null,
                        "bb": null,
                    },
                    "c": {
                        "ca": null,
                        "cb": null,
                    },
                    "d": {
                        "da": null,
                        "db": null,
                    },
                    "e": {
                        "ea": null,
                        "eb": null,
                    }
                }),
            )
            .await;

        cx.update(|cx| {
            open_paths(
                &[
                    PathBuf::from(path!("/root/a")),
                    PathBuf::from(path!("/root/b")),
                ],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.read(|cx| cx.windows().len()), 1);

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/a"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.read(|cx| cx.windows().len()), 1);
        let multi_workspace_1 = cx
            .read(|cx| cx.windows()[0].downcast::<MultiWorkspace>())
            .unwrap();
        cx.run_until_parked();
        multi_workspace_1
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    assert_eq!(workspace.worktrees(cx).count(), 2);
                    assert!(workspace.right_dock().read(cx).is_open());
                    assert!(
                        workspace
                            .active_pane()
                            .read(cx)
                            .focus_handle(cx)
                            .is_focused(window)
                    );
                });
            })
            .unwrap();

        cx.update(|cx| {
            open_paths(
                &[
                    PathBuf::from(path!("/root/c")),
                    PathBuf::from(path!("/root/d")),
                ],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.read(|cx| cx.windows().len()), 1);
        cx.run_until_parked();
        multi_workspace_1
            .update(cx, |multi_workspace, _window, cx| {
                assert_eq!(multi_workspace.workspaces().count(), 2);
                assert!(multi_workspace.sidebar_open());
                let workspace = multi_workspace.workspace().read(cx);
                assert_eq!(
                    workspace
                        .worktrees(cx)
                        .map(|w| w.read(cx).abs_path())
                        .collect::<Vec<_>>(),
                    &[
                        Path::new(path!("/root/c")).into(),
                        Path::new(path!("/root/d")).into(),
                    ]
                );
            })
            .unwrap();

        // Opening with -n (reuse_worktrees: false) still creates a new window.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/e"))],
                app_state,
                workspace::OpenOptions {
                    workspace_matching: workspace::WorkspaceMatching::None,
                    ..Default::default()
                },
                cx,
            )
        })
        .await
        .unwrap();
        cx.background_executor.run_until_parked();
        assert_eq!(cx.read(|cx| cx.windows().len()), 2);
    }

    #[gpui::test]
    async fn test_open_add_new(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({"a": "hey", "b": "", "dir": {"c": "f"}}),
            )
            .await;

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/dir"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/a"))],
                app_state.clone(),
                workspace::OpenOptions {
                    workspace_matching: workspace::WorkspaceMatching::MatchSubdirectory,
                    ..Default::default()
                },
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        // Opening a file inside the existing worktree with -n creates a new window.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/dir/c"))],
                app_state.clone(),
                workspace::OpenOptions {
                    workspace_matching: workspace::WorkspaceMatching::None,
                    ..Default::default()
                },
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 2);

        // Opening a path NOT in any existing worktree with -n creates a new window.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/b"))],
                app_state.clone(),
                workspace::OpenOptions {
                    workspace_matching: workspace::WorkspaceMatching::None,
                    ..Default::default()
                },
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 3);
    }

    #[gpui::test]
    async fn test_open_file_in_many_spaces(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({"dir1": {"a": "b"}, "dir2": {"c": "d"}}),
            )
            .await;

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/dir1/a"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/dir2/c"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        // Opening a directory with default options adds to the existing window
        // rather than creating a new one.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/dir2"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        // Opening a directory already in a worktree with -n creates a new window.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/dir2"))],
                app_state.clone(),
                workspace::OpenOptions {
                    workspace_matching: workspace::WorkspaceMatching::None,
                    ..Default::default()
                },
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 2);

        // Opening a directory NOT in any worktree with -n creates a new window.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root"))],
                app_state.clone(),
                workspace::OpenOptions {
                    workspace_matching: workspace::WorkspaceMatching::None,
                    ..Default::default()
                },
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 3);
    }

    #[gpui::test]
    async fn test_window_edit_state_restoring_disabled(cx: &mut TestAppContext) {
        let executor = cx.executor();
        let app_state = init_test(cx);

        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .session
                        .get_or_insert_default()
                        .restore_unsaved_buffers = Some(false)
                });
            });
        });

        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/root"), json!({"a": "hey"}))
            .await;

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/a"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        // When opening the workspace, the window is not in a edited state.
        let window = cx.update(|cx| cx.windows()[0].downcast::<MultiWorkspace>().unwrap());

        let window_is_edited = |window: WindowHandle<MultiWorkspace>, cx: &mut TestAppContext| {
            cx.update(|cx| window.read(cx).unwrap().workspace().read(cx).is_edited())
        };
        let pane = window
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace.workspace().read(cx).active_pane().clone()
            })
            .unwrap();
        let editor = window
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_item(cx)
                    .unwrap()
                    .downcast::<Editor>()
                    .unwrap()
            })
            .unwrap();

        assert!(!window_is_edited(window, cx));

        // Editing a buffer marks the window as edited.
        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| editor.insert("EDIT", window, cx));
            })
            .unwrap();

        assert!(window_is_edited(window, cx));

        // Undoing the edit restores the window's edited state.
        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| {
                    editor.undo(&Default::default(), window, cx)
                });
            })
            .unwrap();
        assert!(!window_is_edited(window, cx));

        // Redoing the edit marks the window as edited again.
        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| {
                    editor.redo(&Default::default(), window, cx)
                });
            })
            .unwrap();
        assert!(window_is_edited(window, cx));
        let weak = editor.downgrade();

        // Closing the item restores the window's edited state.
        let close = window
            .update(cx, |_, window, cx| {
                pane.update(cx, |pane, cx| {
                    drop(editor);
                    pane.close_active_item(&Default::default(), window, cx)
                })
            })
            .unwrap();
        executor.run_until_parked();

        cx.simulate_prompt_answer("Don't Save");
        close.await.unwrap();

        // Advance the clock to ensure that the item has been serialized and dropped from the queue
        cx.executor().advance_clock(Duration::from_secs(1));

        weak.assert_released();
        assert!(!window_is_edited(window, cx));
        // Opening the buffer again doesn't impact the window's edited state.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/a"))],
                app_state,
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        executor.run_until_parked();

        window
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let editor = workspace
                        .active_item(cx)
                        .unwrap()
                        .downcast::<Editor>()
                        .unwrap();

                    editor.update(cx, |editor, cx| {
                        assert_eq!(editor.text(cx), "hey");
                    });
                });
            })
            .unwrap();

        let editor = window
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_item(cx)
                    .unwrap()
                    .downcast::<Editor>()
                    .unwrap()
            })
            .unwrap();
        assert!(!window_is_edited(window, cx));

        // Editing the buffer marks the window as edited.
        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| editor.insert("EDIT", window, cx));
            })
            .unwrap();
        executor.run_until_parked();
        assert!(window_is_edited(window, cx));

        // Ensure closing the window via the mouse gets preempted due to the
        // buffer having unsaved changes.
        assert!(!VisualTestContext::from_window(window.into(), cx).simulate_close());
        executor.run_until_parked();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        // The window is successfully closed after the user dismisses the prompt.
        cx.simulate_prompt_answer("Don't Save");
        executor.run_until_parked();
        assert_eq!(cx.update(|cx| cx.windows().len()), 0);
    }

    #[ignore = "This test has timing issues across platforms."]
    #[gpui::test]
    async fn test_window_edit_state_restoring_enabled(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/root"), json!({"a": "hey"}))
            .await;

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/a"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();

        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        // When opening the workspace, the window is not in a edited state.
        let window = cx.update(|cx| cx.windows()[0].downcast::<MultiWorkspace>().unwrap());

        let window_is_edited = |window: WindowHandle<MultiWorkspace>, cx: &mut TestAppContext| {
            cx.update(|cx| window.read(cx).unwrap().workspace().read(cx).is_edited())
        };
        let workspace_database_id = |window: WindowHandle<MultiWorkspace>,
                                     cx: &mut TestAppContext| {
            cx.update(|cx| window.read(cx).unwrap().workspace().read(cx).database_id())
        };

        let editor = window
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace
                    .workspace()
                    .read(cx)
                    .active_item(cx)
                    .unwrap()
                    .downcast::<Editor>()
                    .unwrap()
            })
            .unwrap();

        assert!(!window_is_edited(window, cx));
        let initial_database_id = workspace_database_id(window, cx);
        assert!(
            initial_database_id.is_some(),
            "a restored workspace must have a stable database id"
        );

        // Editing a buffer marks the window as edited.
        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| editor.insert("EDIT", window, cx));
            })
            .unwrap();
        cx.run_until_parked();

        assert!(window_is_edited(window, cx));

        // Advance the clock to make sure the workspace is serialized
        cx.executor().advance_clock(Duration::from_secs(1));

        // When closing the window, no prompt shows up and the window is closed.
        // buffer having unsaved changes.
        assert!(!VisualTestContext::from_window(window.into(), cx).simulate_close());
        cx.run_until_parked();
        assert_eq!(cx.update(|cx| cx.windows().len()), 0);

        // When we now reopen the window, the edited state and the edited buffer are back
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/root/a"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();

        assert_eq!(cx.update(|cx| cx.windows().len()), 1);
        assert!(cx.update(|cx| cx.active_window().is_some()));

        cx.run_until_parked();

        // When opening the workspace, the window is not in a edited state.
        let window = cx.update(|cx| {
            cx.active_window()
                .unwrap()
                .downcast::<MultiWorkspace>()
                .unwrap()
        });
        assert!(window_is_edited(window, cx));
        assert_eq!(
            workspace_database_id(window, cx),
            initial_database_id,
            "the workspace must keep the same database id across a close/reopen cycle"
        );

        window
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let editor = workspace
                        .active_item(cx)
                        .unwrap()
                        .downcast::<editor::Editor>()
                        .unwrap();
                    editor.update(cx, |editor, cx| {
                        assert_eq!(editor.text(cx), "EDIThey");
                        assert!(editor.is_dirty(cx));
                    });
                });
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_new_empty_workspace(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        cx.update(|cx| {
            open_new(
                Default::default(),
                app_state.clone(),
                cx,
                |workspace, window, cx| {
                    Editor::new_file(workspace, &Default::default(), window, cx)
                },
            )
        })
        .await
        .unwrap();
        cx.run_until_parked();

        let multi_workspace = cx
            .update(|cx| cx.windows().first().unwrap().downcast::<MultiWorkspace>())
            .unwrap();

        let editor = multi_workspace
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    #[cfg(feature = "sim")]
                    {
                        assert!(
                            workspace.panel::<sim_ui::ExecutionPanel>(cx).is_some(),
                            "the production workspace must register the native execution dock panel"
                        );
                        assert!(
                            workspace
                                .panel::<sim_ui::GraphPropertiesPanel>(cx)
                                .is_some(),
                            "the production workspace must register the native graph properties dock panel"
                        );
                    }
                    let editor = workspace
                        .active_item(cx)
                        .unwrap()
                        .downcast::<editor::Editor>()
                        .unwrap();
                    editor.update(cx, |editor, cx| {
                        assert!(editor.text(cx).is_empty());
                        assert!(!editor.is_dirty(cx));
                    });

                    editor
                })
            })
            .unwrap();

        let save_task = multi_workspace
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    workspace.save_active_item(SaveIntent::Save, window, cx)
                })
            })
            .unwrap();
        app_state.fs.create_dir(Path::new("/root")).await.unwrap();
        cx.background_executor.run_until_parked();
        cx.simulate_new_path_selection(|_| Some(PathBuf::from("/root/the-new-name")));
        save_task.await.unwrap();
        multi_workspace
            .update(cx, |_, _, cx| {
                editor.update(cx, |editor, cx| {
                    assert!(!editor.is_dirty(cx));
                    assert_eq!(editor.title(cx), "the-new-name");
                });
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_open_entry(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({
                    "a": {
                        "file1": "contents 1",
                        "file2": "contents 2",
                        "file3": "contents 3",
                    },
                }),
            )
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        project.update(cx, |project, _cx| project.languages().add(markdown_lang()));
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let entries = cx.read(|cx| workspace.file_project_paths(cx));
        let file1 = entries[0].clone();
        let file2 = entries[1].clone();
        let file3 = entries[2].clone();

        // Open the first entry
        let entry_1 = window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |w, cx| {
                    w.open_path(file1.clone(), None, true, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap();
        cx.read(|cx| {
            let pane = workspace.read(cx).active_pane().read(cx);
            assert_eq!(
                pane.active_item().unwrap().project_path(cx),
                Some(file1.clone())
            );
            assert_eq!(pane.items_len(), 1);
        });

        // Open the second entry
        window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |w, cx| {
                    w.open_path(file2.clone(), None, true, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap();
        cx.read(|cx| {
            let pane = workspace.read(cx).active_pane().read(cx);
            assert_eq!(
                pane.active_item().unwrap().project_path(cx),
                Some(file2.clone())
            );
            assert_eq!(pane.items_len(), 2);
        });

        // Open the first entry again. The existing pane item is activated.
        let entry_1b = window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |w, cx| {
                    w.open_path(file1.clone(), None, true, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap();
        assert_eq!(entry_1.item_id(), entry_1b.item_id());

        cx.read(|cx| {
            let pane = workspace.read(cx).active_pane().read(cx);
            assert_eq!(
                pane.active_item().unwrap().project_path(cx),
                Some(file1.clone())
            );
            assert_eq!(pane.items_len(), 2);
        });

        // Split the pane with the first entry, then open the second entry again.
        window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |w, cx| {
                    w.split_and_clone(w.active_pane().clone(), SplitDirection::Right, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap();
        window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |w, cx| {
                    w.open_path(file2.clone(), None, true, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap();

        cx.read(|cx| {
            assert_eq!(
                workspace
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .unwrap()
                    .project_path(cx),
                Some(file2.clone())
            );
        });

        // Open the third entry twice concurrently. Only one pane item is added.
        let (t1, t2) = window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |w, cx| {
                    (
                        w.open_path(file3.clone(), None, true, window, cx),
                        w.open_path(file3.clone(), None, true, window, cx),
                    )
                })
            })
            .unwrap();
        t1.await.unwrap();
        t2.await.unwrap();
        cx.read(|cx| {
            let pane = workspace.read(cx).active_pane().read(cx);
            assert_eq!(
                pane.active_item().unwrap().project_path(cx),
                Some(file3.clone())
            );
            let pane_entries = pane
                .items()
                .map(|i| i.project_path(cx).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(pane_entries, &[file1, file2, file3]);
        });
    }

    #[gpui::test]
    async fn test_open_paths(cx: &mut TestAppContext) {
        let app_state = init_test(cx);

        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/"),
                json!({
                    "dir1": {
                        "a.txt": ""
                    },
                    "dir2": {
                        "b.txt": ""
                    },
                    "dir3": {
                        "c.txt": ""
                    },
                    "d.txt": ""
                }),
            )
            .await;

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/dir1/"))],
                app_state,
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        cx.run_until_parked();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);
        let window = cx.update(|cx| cx.windows()[0].downcast::<MultiWorkspace>().unwrap());
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        #[track_caller]
        fn assert_project_panel_selection(
            workspace: &Workspace,
            expected_worktree_path: &Path,
            expected_entry_path: &RelPath,
            cx: &App,
        ) {
            let project_panel = [
                workspace.left_dock().read(cx).panel::<ProjectPanel>(),
                workspace.right_dock().read(cx).panel::<ProjectPanel>(),
                workspace.bottom_dock().read(cx).panel::<ProjectPanel>(),
            ]
            .into_iter()
            .find_map(std::convert::identity)
            .expect("found no project panels")
            .read(cx);
            let (selected_worktree, selected_entry) = project_panel
                .selected_entry(cx)
                .expect("project panel should have a selected entry");
            assert_eq!(
                selected_worktree.abs_path().as_ref(),
                expected_worktree_path,
                "Unexpected project panel selected worktree path"
            );
            assert_eq!(
                selected_entry.path.as_ref(),
                expected_entry_path,
                "Unexpected project panel selected entry path"
            );
        }

        // Open a file within an existing worktree.
        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    workspace.open_paths(
                        vec![path!("/dir1/a.txt").into()],
                        OpenOptions {
                            visible: Some(OpenVisible::All),
                            ..Default::default()
                        },
                        None,
                        window,
                        cx,
                    )
                })
            })
            .unwrap()
            .await;
        cx.run_until_parked();
        cx.read(|cx| {
            let workspace = workspace.read(cx);
            assert_project_panel_selection(
                workspace,
                Path::new(path!("/dir1")),
                rel_path("a.txt"),
                cx,
            );
            assert_eq!(
                workspace
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .unwrap()
                    .act_as::<Editor>(cx)
                    .unwrap()
                    .read(cx)
                    .title(cx),
                "a.txt"
            );
        });

        // Open a file outside of any existing worktree.
        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    workspace.open_paths(
                        vec![path!("/dir2/b.txt").into()],
                        OpenOptions {
                            visible: Some(OpenVisible::All),
                            ..Default::default()
                        },
                        None,
                        window,
                        cx,
                    )
                })
            })
            .unwrap()
            .await;
        cx.run_until_parked();
        cx.read(|cx| {
            let workspace = workspace.read(cx);
            assert_project_panel_selection(
                workspace,
                Path::new(path!("/dir2/b.txt")),
                rel_path(""),
                cx,
            );
            let worktree_roots = workspace
                .worktrees(cx)
                .map(|w| w.read(cx).as_local().unwrap().abs_path().as_ref())
                .collect::<HashSet<_>>();
            assert_eq!(
                worktree_roots,
                vec![path!("/dir1"), path!("/dir2/b.txt")]
                    .into_iter()
                    .map(Path::new)
                    .collect(),
            );
            assert_eq!(
                workspace
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .unwrap()
                    .act_as::<Editor>(cx)
                    .unwrap()
                    .read(cx)
                    .title(cx),
                "b.txt"
            );
        });

        // Ensure opening a directory and one of its children only adds one worktree.
        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    workspace.open_paths(
                        vec![path!("/dir3").into(), path!("/dir3/c.txt").into()],
                        OpenOptions {
                            visible: Some(OpenVisible::All),
                            ..Default::default()
                        },
                        None,
                        window,
                        cx,
                    )
                })
            })
            .unwrap()
            .await;
        cx.run_until_parked();
        cx.read(|cx| {
            let workspace = workspace.read(cx);
            assert_project_panel_selection(
                workspace,
                Path::new(path!("/dir3")),
                rel_path("c.txt"),
                cx,
            );
            let worktree_roots = workspace
                .worktrees(cx)
                .map(|w| w.read(cx).as_local().unwrap().abs_path().as_ref())
                .collect::<HashSet<_>>();
            assert_eq!(
                worktree_roots,
                vec![path!("/dir1"), path!("/dir2/b.txt"), path!("/dir3")]
                    .into_iter()
                    .map(Path::new)
                    .collect(),
            );
            assert_eq!(
                workspace
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .unwrap()
                    .act_as::<Editor>(cx)
                    .unwrap()
                    .read(cx)
                    .title(cx),
                "c.txt"
            );
        });

        // Ensure opening invisibly a file outside an existing worktree adds a new, invisible worktree.
        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    workspace.open_paths(
                        vec![path!("/d.txt").into()],
                        OpenOptions {
                            visible: Some(OpenVisible::None),
                            ..Default::default()
                        },
                        None,
                        window,
                        cx,
                    )
                })
            })
            .unwrap()
            .await;
        cx.run_until_parked();
        cx.read(|cx| {
            let workspace = workspace.read(cx);
            assert_project_panel_selection(workspace, Path::new(path!("/d.txt")), rel_path(""), cx);
            let worktree_roots = workspace
                .worktrees(cx)
                .map(|w| w.read(cx).as_local().unwrap().abs_path().as_ref())
                .collect::<HashSet<_>>();
            assert_eq!(
                worktree_roots,
                vec![
                    path!("/dir1"),
                    path!("/dir2/b.txt"),
                    path!("/dir3"),
                    path!("/d.txt")
                ]
                .into_iter()
                .map(Path::new)
                .collect(),
            );

            let visible_worktree_roots = workspace
                .visible_worktrees(cx)
                .map(|w| w.read(cx).as_local().unwrap().abs_path().as_ref())
                .collect::<HashSet<_>>();
            assert_eq!(
                visible_worktree_roots,
                vec![path!("/dir1"), path!("/dir2/b.txt"), path!("/dir3")]
                    .into_iter()
                    .map(Path::new)
                    .collect(),
            );

            assert_eq!(
                workspace
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .unwrap()
                    .act_as::<Editor>(cx)
                    .unwrap()
                    .read(cx)
                    .title(cx),
                "d.txt"
            );
        });
    }

    #[gpui::test]
    async fn test_opening_excluded_paths(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |project_settings| {
                    project_settings.project.worktree.file_scan_exclusions =
                        Some(vec!["excluded_dir".to_string(), "**/.git".to_string()]);
                });
            });
        });
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({
                    ".gitignore": "ignored_dir\n",
                    ".git": {
                        "HEAD": "ref: refs/heads/main",
                    },
                    "regular_dir": {
                        "file": "regular file contents",
                    },
                    "ignored_dir": {
                        "ignored_subdir": {
                            "file": "ignored subfile contents",
                        },
                        "file": "ignored file contents",
                    },
                    "excluded_dir": {
                        "file": "excluded file contents",
                        "ignored_subdir": {
                            "file": "ignored subfile contents",
                        },
                    },
                }),
            )
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        project.update(cx, |project, _cx| project.languages().add(markdown_lang()));
        let window = cx.add_window({
            let project = project.clone();
            |window, cx| MultiWorkspace::test_new(project, window, cx)
        });
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        let initial_entries = cx.read(|cx| workspace.file_project_paths(cx));
        let paths_to_open = [
            PathBuf::from(path!("/root/excluded_dir/file")),
            PathBuf::from(path!("/root/.git/HEAD")),
            PathBuf::from(path!("/root/excluded_dir/ignored_subdir")),
        ];
        let workspace::OpenResult {
            window: opened_workspace,
            opened_items: new_items,
            ..
        } = cx
            .update(|cx| {
                workspace::open_paths(
                    &paths_to_open,
                    app_state,
                    workspace::OpenOptions::default(),
                    cx,
                )
            })
            .await
            .unwrap();

        assert_eq!(
            opened_workspace
                .read_with(cx, |mw, _| mw.workspace().entity_id())
                .unwrap(),
            workspace.entity_id(),
            "Excluded files in subfolders of a workspace root should be opened in the workspace"
        );
        let mut opened_paths = cx.read(|cx| {
            assert_eq!(
                new_items.len(),
                paths_to_open.len(),
                "Expect to get the same number of opened items as submitted paths to open"
            );
            new_items
                .iter()
                .zip(paths_to_open.iter())
                .map(|(i, path)| {
                    match i {
                        Some(Ok(i)) => Some(i.project_path(cx).map(|p| p.path)),
                        Some(Err(e)) => panic!("Excluded file {path:?} failed to open: {e:?}"),
                        None => None,
                    }
                    .flatten()
                })
                .collect::<Vec<_>>()
        });
        opened_paths.sort();
        assert_eq!(
            opened_paths,
            vec![
                None,
                Some(rel_path(".git/HEAD").into()),
                Some(rel_path("excluded_dir/file").into()),
            ],
            "Excluded files should get opened, excluded dir should not get opened"
        );

        let entries = cx.read(|cx| workspace.file_project_paths(cx));
        assert_eq!(
            initial_entries, entries,
            "Workspace entries should not change after opening excluded files and directories paths"
        );

        cx.read(|cx| {
                let pane = workspace.read(cx).active_pane().read(cx);
                let mut opened_buffer_paths = pane
                    .items()
                    .map(|i| {
                        i.project_path(cx)
                            .expect("all excluded files that got open should have a path")
                            .path
                    })
                    .collect::<Vec<_>>();
                opened_buffer_paths.sort();
                assert_eq!(
                    opened_buffer_paths,
                    vec![rel_path(".git/HEAD").into(), rel_path("excluded_dir/file").into()],
                    "Despite not being present in the worktrees, buffers for excluded files are opened and added to the pane"
                );
            });
    }

    #[gpui::test]
    async fn test_save_conflicting_item(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/root"), json!({ "a.txt": "" }))
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        project.update(cx, |project, _cx| project.languages().add(markdown_lang()));
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        // Open a file within an existing worktree.
        window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.open_paths(
                        vec![PathBuf::from(path!("/root/a.txt"))],
                        OpenOptions {
                            visible: Some(OpenVisible::All),
                            ..Default::default()
                        },
                        None,
                        window,
                        cx,
                    )
                })
            })
            .unwrap()
            .await;
        let editor = cx.read(|cx| {
            let pane = workspace.read(cx).active_pane().read(cx);
            let item = pane.active_item().unwrap();
            item.downcast::<Editor>().unwrap()
        });

        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| editor.handle_input("x", window, cx));
            })
            .unwrap();

        app_state
            .fs
            .as_fake()
            .insert_file(path!("/root/a.txt"), b"changed".to_vec())
            .await;

        cx.run_until_parked();
        cx.read(|cx| assert!(editor.is_dirty(cx)));
        cx.read(|cx| assert!(editor.has_conflict(cx)));

        let save_task = window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.save_active_item(SaveIntent::Save, window, cx)
                })
            })
            .unwrap();
        cx.background_executor.run_until_parked();
        cx.simulate_prompt_answer("Overwrite");
        save_task.await.unwrap();
        window
            .update(cx, |_, _, cx| {
                editor.update(cx, |editor, cx| {
                    assert!(!editor.is_dirty(cx));
                    assert!(!editor.has_conflict(cx));
                });
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_open_and_save_new_file(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .create_dir(Path::new(path!("/root")))
            .await
            .unwrap();

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        project.update(cx, |project, _| {
            project.languages().add(markdown_lang());
            project.languages().add(rust_lang());
        });
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();
        let worktree = cx.read(|cx| workspace.read(cx).worktrees(cx).next().unwrap());

        // Create a new untitled buffer
        cx.dispatch_action(window.into(), NewFile);
        let editor = cx.read(|cx| {
            workspace
                .read(cx)
                .active_item(cx)
                .unwrap()
                .downcast::<Editor>()
                .unwrap()
        });

        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| {
                    assert!(!editor.is_dirty(cx));
                    assert_eq!(editor.title(cx), "untitled");
                    assert!(Arc::ptr_eq(
                        &editor
                            .buffer()
                            .read(cx)
                            .language_at(MultiBufferOffset(0), cx)
                            .unwrap(),
                        &languages::PLAIN_TEXT
                    ));
                    editor.handle_input("hi", window, cx);
                    assert!(editor.is_dirty(cx));
                });
            })
            .unwrap();

        // Save the buffer. This prompts for a filename.
        let save_task = window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.save_active_item(SaveIntent::Save, window, cx)
                })
            })
            .unwrap();
        cx.background_executor.run_until_parked();
        cx.simulate_new_path_selection(|parent_dir| {
            assert_eq!(parent_dir, Path::new(path!("/root")));
            Some(parent_dir.join("the-new-name.rs"))
        });
        cx.read(|cx| {
            assert!(editor.is_dirty(cx));
            assert_eq!(editor.read(cx).title(cx), "hi");
        });

        // When the save completes, the buffer's title is updated and the language is assigned based
        // on the path.
        save_task.await.unwrap();
        window
            .update(cx, |_, _, cx| {
                editor.update(cx, |editor, cx| {
                    assert!(!editor.is_dirty(cx));
                    assert_eq!(editor.title(cx), "the-new-name.rs");
                    assert_eq!(
                        editor
                            .buffer()
                            .read(cx)
                            .language_at(MultiBufferOffset(0), cx)
                            .unwrap()
                            .name(),
                        "Rust"
                    );
                });
            })
            .unwrap();

        // Edit the file and save it again. This time, there is no filename prompt.
        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| {
                    editor.handle_input(" there", window, cx);
                    assert!(editor.is_dirty(cx));
                });
            })
            .unwrap();

        let save_task = window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.save_active_item(SaveIntent::Save, window, cx)
                })
            })
            .unwrap();
        save_task.await.unwrap();

        assert!(!cx.did_prompt_for_new_path());
        window
            .update(cx, |_, _, cx| {
                editor.update(cx, |editor, cx| {
                    assert!(!editor.is_dirty(cx));
                    assert_eq!(editor.title(cx), "the-new-name.rs")
                });
            })
            .unwrap();

        // Open the same newly-created file in another pane item. The new editor should reuse
        // the same buffer.
        cx.dispatch_action(window.into(), NewFile);
        window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.split_and_clone(
                        workspace.active_pane().clone(),
                        SplitDirection::Right,
                        window,
                        cx,
                    )
                })
            })
            .unwrap()
            .await
            .unwrap();
        window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.open_path(
                        (worktree.read(cx).id(), rel_path("the-new-name.rs")),
                        None,
                        true,
                        window,
                        cx,
                    )
                })
            })
            .unwrap()
            .await
            .unwrap();
        let editor2 = cx.read(|cx| {
            workspace
                .read(cx)
                .active_item(cx)
                .unwrap()
                .downcast::<Editor>()
                .unwrap()
        });
        cx.read(|cx| {
            assert_eq!(
                editor2.read(cx).buffer().read(cx).as_singleton().unwrap(),
                editor.read(cx).buffer().read(cx).as_singleton().unwrap()
            );
        })
    }

    #[gpui::test]
    async fn test_setting_language_when_saving_as_single_file_worktree(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state.fs.create_dir(Path::new("/root")).await.unwrap();

        let project = Project::test(app_state.fs.clone(), [], cx).await;
        project.update(cx, |project, _| {
            project.languages().add(language::rust_lang());
            project.languages().add(language::markdown_lang());
        });
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        // Create a new untitled buffer
        cx.dispatch_action(window.into(), NewFile);
        let editor = cx.read(|cx| {
            workspace
                .read(cx)
                .active_item(cx)
                .unwrap()
                .downcast::<Editor>()
                .unwrap()
        });
        window
            .update(cx, |_, window, cx| {
                editor.update(cx, |editor, cx| {
                    assert!(Arc::ptr_eq(
                        &editor
                            .buffer()
                            .read(cx)
                            .language_at(MultiBufferOffset(0), cx)
                            .unwrap(),
                        &languages::PLAIN_TEXT
                    ));
                    editor.handle_input("hi", window, cx);
                    assert!(editor.is_dirty(cx));
                });
            })
            .unwrap();

        // Save the buffer. This prompts for a filename.
        let save_task = window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.save_active_item(SaveIntent::Save, window, cx)
                })
            })
            .unwrap();
        cx.background_executor.run_until_parked();
        cx.simulate_new_path_selection(|_| Some(PathBuf::from("/root/the-new-name.rs")));
        save_task.await.unwrap();
        // The buffer is not dirty anymore and the language is assigned based on the path.
        window
            .update(cx, |_, _, cx| {
                editor.update(cx, |editor, cx| {
                    assert!(!editor.is_dirty(cx));
                    assert_eq!(
                        editor
                            .buffer()
                            .read(cx)
                            .language_at(MultiBufferOffset(0), cx)
                            .unwrap()
                            .name(),
                        "Rust"
                    )
                });
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_pane_actions(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({
                    "a": {
                        "file1": "contents 1",
                        "file2": "contents 2",
                        "file3": "contents 3",
                    },
                }),
            )
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        project.update(cx, |project, _cx| project.languages().add(markdown_lang()));
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();
        let cx = &mut VisualTestContext::from_window(*window, cx);

        let entries = cx.read(|cx| workspace.file_project_paths(cx));
        let file1 = entries[0].clone();

        let pane_1 = cx.read(|cx| workspace.read(cx).active_pane().clone());

        workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file1.clone(), None, true, window, cx)
            })
            .await
            .unwrap();

        let (editor_1, buffer) = workspace.update_in(cx, |_, window, cx| {
            pane_1.update(cx, |pane_1, cx| {
                let editor = pane_1.active_item().unwrap().downcast::<Editor>().unwrap();
                assert_eq!(editor.read(cx).active_project_path(cx), Some(file1.clone()));
                let buffer = editor.update(cx, |editor, cx| {
                    editor.insert("dirt", window, cx);
                    editor.buffer().downgrade()
                });
                (editor.downgrade(), buffer)
            })
        });

        cx.dispatch_action(pane::SplitRight::default());
        let editor_2 = cx.update(|_, cx| {
            let pane_2 = workspace.read(cx).active_pane().clone();
            assert_ne!(pane_1, pane_2);

            let pane2_item = pane_2.read(cx).active_item().unwrap();
            assert_eq!(pane2_item.project_path(cx), Some(file1.clone()));

            pane2_item.downcast::<Editor>().unwrap().downgrade()
        });
        cx.dispatch_action(workspace::CloseActiveItem {
            save_intent: None,
            close_pinned: false,
        });

        cx.background_executor.run_until_parked();
        workspace.read_with(cx, |workspace, _| {
            assert_eq!(workspace.panes().len(), 1);
            assert_eq!(workspace.active_pane(), &pane_1);
        });

        cx.dispatch_action(workspace::CloseActiveItem {
            save_intent: None,
            close_pinned: false,
        });
        cx.background_executor.run_until_parked();
        cx.simulate_prompt_answer("Don't Save");
        cx.background_executor.run_until_parked();

        workspace.read_with(cx, |workspace, cx| {
            assert_eq!(workspace.panes().len(), 1);
            assert!(workspace.active_item(cx).is_none());
        });

        cx.background_executor
            .advance_clock(SERIALIZATION_THROTTLE_TIME);
        cx.update(|_, _| {});
        editor_1.assert_released();
        editor_2.assert_released();
        buffer.assert_released();
    }

    #[gpui::test]
    async fn test_editor_zoom_with_scroll_wheel(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/root"), json!({ "file.txt": "hello\nworld\n" }))
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();
        let cx = &mut VisualTestContext::from_window(*window, cx);

        let mouse_position = point(px(250.), px(250.));

        let event_modifiers = {
            #[cfg(target_os = "macos")]
            {
                Modifiers {
                    platform: true,
                    ..Modifiers::default()
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                Modifiers {
                    control: true,
                    ..Modifiers::default()
                }
            }
        };

        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_abs_path(
                    PathBuf::from(path!("/root/file.txt")),
                    OpenOptions::default(),
                    window,
                    cx,
                )
            })
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();

        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });

        // mouse_wheel_zoom is disabled by default — zoom should not work.
        let initial_font_size =
            cx.update(|_, cx| ThemeSettings::get_global(cx).buffer_font_size(cx).as_f32());

        cx.simulate_event(gpui::ScrollWheelEvent {
            position: mouse_position,
            delta: gpui::ScrollDelta::Pixels(point(px(0.), px(1.))),
            modifiers: event_modifiers,
            ..Default::default()
        });

        let font_size_after_disabled_zoom =
            cx.update(|_, cx| ThemeSettings::get_global(cx).buffer_font_size(cx).as_f32());

        assert_eq!(
            initial_font_size, font_size_after_disabled_zoom,
            "Editor buffer font-size should not change when mouse_wheel_zoom is disabled"
        );

        // Enable mouse_wheel_zoom and verify zoom works.
        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.editor.mouse_wheel_zoom = Some(true);
                });
            });
        });

        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });

        cx.simulate_event(gpui::ScrollWheelEvent {
            position: mouse_position,
            delta: gpui::ScrollDelta::Pixels(point(px(0.), px(1.))),
            modifiers: event_modifiers,
            ..Default::default()
        });

        let increased_font_size =
            cx.update(|_, cx| ThemeSettings::get_global(cx).buffer_font_size(cx).as_f32());

        assert!(
            increased_font_size > initial_font_size,
            "Editor buffer font-size should have increased from scroll-zoom"
        );

        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });

        cx.simulate_event(gpui::ScrollWheelEvent {
            position: mouse_position,
            delta: gpui::ScrollDelta::Pixels(point(px(0.), px(-1.))),
            modifiers: event_modifiers,
            ..Default::default()
        });

        let decreased_font_size =
            cx.update(|_, cx| ThemeSettings::get_global(cx).buffer_font_size(cx).as_f32());

        assert!(
            decreased_font_size < increased_font_size,
            "Editor buffer font-size should have decreased from scroll-zoom"
        );

        // Disable mouse_wheel_zoom again and verify zoom stops working.
        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.editor.mouse_wheel_zoom = Some(false);
                });
            });
        });

        let font_size_before =
            cx.update(|_, cx| ThemeSettings::get_global(cx).buffer_font_size(cx).as_f32());

        cx.update(|window, cx| {
            window.draw(cx).clear(cx);
        });

        cx.simulate_event(gpui::ScrollWheelEvent {
            position: mouse_position,
            delta: gpui::ScrollDelta::Pixels(point(px(0.), px(1.))),
            modifiers: event_modifiers,
            ..Default::default()
        });

        let font_size_after =
            cx.update(|_, cx| ThemeSettings::get_global(cx).buffer_font_size(cx).as_f32());

        assert_eq!(
            font_size_before, font_size_after,
            "Editor buffer font-size should not change when mouse_wheel_zoom is re-disabled"
        );
    }

    #[gpui::test]
    async fn test_navigation(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({
                    "a": {
                        "file1": "contents 1\n".repeat(20),
                        "file2": "contents 2\n".repeat(20),
                        "file3": "contents 3\n".repeat(20),
                    },
                }),
            )
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        project.update(cx, |project, _cx| project.languages().add(markdown_lang()));
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();
        let cx = &mut VisualTestContext::from_window(*window, cx);
        let pane = workspace.read_with(cx, |workspace, _| workspace.active_pane().clone());

        let entries = cx.read(|cx| workspace.file_project_paths(cx));
        let file1 = entries[0].clone();
        let file2 = entries[1].clone();
        let file3 = entries[2].clone();

        let editor1 = workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file1.clone(), None, true, window, cx)
            })
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();
        workspace.update_in(cx, |_, window, cx| {
            editor1.update(cx, |editor, cx| {
                editor.change_selections(Default::default(), window, cx, |s| {
                    s.select_display_ranges([
                        DisplayPoint::new(DisplayRow(10), 0)..DisplayPoint::new(DisplayRow(10), 0)
                    ])
                });
            });
        });

        let editor2 = workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file2.clone(), None, true, window, cx)
            })
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();
        let editor3 = workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file3.clone(), None, true, window, cx)
            })
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();

        workspace
            .update_in(cx, |_, window, cx| {
                editor3.update(cx, |editor, cx| {
                    editor.change_selections(Default::default(), window, cx, |s| {
                        s.select_display_ranges([DisplayPoint::new(DisplayRow(12), 0)
                            ..DisplayPoint::new(DisplayRow(12), 0)])
                    });
                    editor.newline(&Default::default(), window, cx);
                    editor.newline(&Default::default(), window, cx);
                    editor.move_down(&Default::default(), window, cx);
                    editor.move_down(&Default::default(), window, cx);
                    editor.save(
                        SaveOptions {
                            format: true,
                            force_format: false,
                            autosave: false,
                        },
                        project.clone(),
                        window,
                        cx,
                    )
                })
            })
            .await
            .unwrap();
        workspace.update_in(cx, |_, window, cx| {
            editor3.update(cx, |editor, cx| {
                editor.set_scroll_position(point(0., 12.5), window, cx)
            });
        });
        assert_eq!(
            active_location(&workspace, cx),
            (file3.clone(), DisplayPoint::new(DisplayRow(16), 0), 12.5)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file3.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file2.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file1.clone(), DisplayPoint::new(DisplayRow(10), 0), 0.)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file1.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        // Go back one more time and ensure we don't navigate past the first item in the history.
        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file1.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_forward(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file1.clone(), DisplayPoint::new(DisplayRow(10), 0), 0.)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_forward(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file2.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        // Go forward to an item that has been closed, ensuring it gets re-opened at the same
        // location.
        workspace
            .update_in(cx, |_, window, cx| {
                pane.update(cx, |pane, cx| {
                    let editor3_id = editor3.entity_id();
                    drop(editor3);
                    pane.close_item_by_id(editor3_id, SaveIntent::Close, window, cx)
                })
            })
            .await
            .unwrap();
        workspace
            .update_in(cx, |w, window, cx| {
                w.go_forward(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file3.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_forward(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file3.clone(), DisplayPoint::new(DisplayRow(16), 0), 12.5)
        );

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file3.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        // Go back to an item that has been closed and removed from disk
        workspace
            .update_in(cx, |_, window, cx| {
                pane.update(cx, |pane, cx| {
                    let editor2_id = editor2.entity_id();
                    drop(editor2);
                    pane.close_item_by_id(editor2_id, SaveIntent::Close, window, cx)
                })
            })
            .await
            .unwrap();
        app_state
            .fs
            .remove_file(Path::new(path!("/root/a/file2")), Default::default())
            .await
            .unwrap();
        cx.background_executor.run_until_parked();

        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file2.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );
        workspace
            .update_in(cx, |w, window, cx| {
                w.go_forward(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file3.clone(), DisplayPoint::new(DisplayRow(0), 0), 0.)
        );

        // Modify file to collapse multiple nav history entries into the same location.
        // Ensure we don't visit the same location twice when navigating.
        workspace.update_in(cx, |_, window, cx| {
            editor1.update(cx, |editor, cx| {
                editor.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                    s.select_display_ranges([
                        DisplayPoint::new(DisplayRow(15), 0)..DisplayPoint::new(DisplayRow(15), 0)
                    ])
                })
            });
        });
        for _ in 0..5 {
            workspace.update_in(cx, |_, window, cx| {
                editor1.update(cx, |editor, cx| {
                    editor.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                        s.select_display_ranges([DisplayPoint::new(DisplayRow(3), 0)
                            ..DisplayPoint::new(DisplayRow(3), 0)])
                    });
                });
            });

            workspace.update_in(cx, |_, window, cx| {
                editor1.update(cx, |editor, cx| {
                    editor.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                        s.select_display_ranges([DisplayPoint::new(DisplayRow(13), 0)
                            ..DisplayPoint::new(DisplayRow(13), 0)])
                    });
                });
            });
        }
        workspace.update_in(cx, |_, window, cx| {
            editor1.update(cx, |editor, cx| {
                editor.transact(window, cx, |editor, window, cx| {
                    editor.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                        s.select_display_ranges([DisplayPoint::new(DisplayRow(2), 0)
                            ..DisplayPoint::new(DisplayRow(14), 0)])
                    });
                    editor.insert("", window, cx);
                })
            });
        });

        workspace.update_in(cx, |_, window, cx| {
            editor1.update(cx, |editor, cx| {
                editor.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                    s.select_display_ranges([
                        DisplayPoint::new(DisplayRow(1), 0)..DisplayPoint::new(DisplayRow(1), 0)
                    ])
                })
            });
        });
        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file1.clone(), DisplayPoint::new(DisplayRow(2), 0), 0.)
        );
        workspace
            .update_in(cx, |w, window, cx| {
                w.go_back(w.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();
        assert_eq!(
            active_location(&workspace, cx),
            (file1.clone(), DisplayPoint::new(DisplayRow(3), 0), 0.)
        );

        fn active_location(
            workspace: &Entity<Workspace>,
            cx: &mut VisualTestContext,
        ) -> (ProjectPath, DisplayPoint, f64) {
            workspace.update(cx, |workspace, cx| {
                let item = workspace.active_item(cx).unwrap();
                let editor = item.downcast::<Editor>().unwrap();

                editor.update(cx, |editor_ref, cx| {
                    let selections = editor_ref
                        .selections
                        .display_ranges(&editor_ref.display_snapshot(cx));
                    let scroll_position = editor_ref.scroll_position(cx);

                    (
                        editor_ref.active_project_path(cx).unwrap(),
                        selections[0].start,
                        scroll_position.y,
                    )
                })
            })
        }
    }

    #[gpui::test]
    async fn test_reopening_closed_items(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/root"),
                json!({
                    "a": {
                        "file1": "",
                        "file2": "",
                        "file3": "",
                        "file4": "",
                    },
                }),
            )
            .await;

        let project = Project::test(app_state.fs.clone(), [path!("/root").as_ref()], cx).await;
        project.update(cx, |project, _cx| project.languages().add(markdown_lang()));
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();
        let cx = &mut VisualTestContext::from_window(*window, cx);
        let pane = workspace.read_with(cx, |workspace, _| workspace.active_pane().clone());

        let entries = cx.read(|cx| workspace.file_project_paths(cx));
        let file1 = entries[0].clone();
        let file2 = entries[1].clone();
        let file3 = entries[2].clone();
        let file4 = entries[3].clone();

        let file1_item_id = workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file1.clone(), None, true, window, cx)
            })
            .await
            .unwrap()
            .item_id();
        let file2_item_id = workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file2.clone(), None, true, window, cx)
            })
            .await
            .unwrap()
            .item_id();
        let file3_item_id = workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file3.clone(), None, true, window, cx)
            })
            .await
            .unwrap()
            .item_id();
        let file4_item_id = workspace
            .update_in(cx, |w, window, cx| {
                w.open_path(file4.clone(), None, true, window, cx)
            })
            .await
            .unwrap()
            .item_id();
        assert_eq!(active_path(&workspace, cx), Some(file4.clone()));

        // Close all the pane items in some arbitrary order.
        workspace
            .update_in(cx, |_, window, cx| {
                pane.update(cx, |pane, cx| {
                    pane.close_item_by_id(file1_item_id, SaveIntent::Close, window, cx)
                })
            })
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file4.clone()));

        workspace
            .update_in(cx, |_, window, cx| {
                pane.update(cx, |pane, cx| {
                    pane.close_item_by_id(file4_item_id, SaveIntent::Close, window, cx)
                })
            })
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file3.clone()));

        workspace
            .update_in(cx, |_, window, cx| {
                pane.update(cx, |pane, cx| {
                    pane.close_item_by_id(file2_item_id, SaveIntent::Close, window, cx)
                })
            })
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file3.clone()));
        workspace
            .update_in(cx, |_, window, cx| {
                pane.update(cx, |pane, cx| {
                    pane.close_item_by_id(file3_item_id, SaveIntent::Close, window, cx)
                })
            })
            .await
            .unwrap();

        assert_eq!(active_path(&workspace, cx), None);

        // Reopen all the closed items, ensuring they are reopened in the same order
        // in which they were closed.
        workspace
            .update_in(cx, Workspace::reopen_closed_item)
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file3.clone()));

        workspace
            .update_in(cx, Workspace::reopen_closed_item)
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file2.clone()));

        workspace
            .update_in(cx, Workspace::reopen_closed_item)
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file4.clone()));

        workspace
            .update_in(cx, Workspace::reopen_closed_item)
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file1.clone()));

        // Reopening past the last closed item is a no-op.
        workspace
            .update_in(cx, Workspace::reopen_closed_item)
            .await
            .unwrap();
        assert_eq!(active_path(&workspace, cx), Some(file1.clone()));

        // Reopening closed items doesn't interfere with navigation history.
        // Verify we can navigate back through the history after reopening items.
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.go_back(workspace.active_pane().downgrade(), window, cx)
            })
            .await
            .unwrap();

        // After go_back, we should be at a different file than file1
        let after_go_back = active_path(&workspace, cx);
        assert!(
            after_go_back.is_some() && after_go_back != Some(file1.clone()),
            "After go_back from file1, should be at a different file"
        );

        pane.read_with(cx, |pane, _| {
            assert!(pane.can_navigate_forward(), "Should be able to go forward");
        });

        fn active_path(
            workspace: &Entity<Workspace>,
            cx: &VisualTestContext,
        ) -> Option<ProjectPath> {
            workspace.read_with(cx, |workspace, cx| {
                let item = workspace.active_item(cx)?;
                item.project_path(cx)
            })
        }
    }

    fn init_keymap_test(cx: &mut TestAppContext) -> Arc<AppState> {
        cx.update(|cx| {
            let app_state = AppState::test(cx);

            theme_settings::init(theme::LoadThemes::JustBase, cx);
            client::init(&app_state.client, cx);
            workspace::init(app_state.clone(), cx);
            onboarding::init(cx);
            app_state
        })
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    #[gpui::test]
    fn test_language_tool_panel_keys_survive_keymap_reload(cx: &mut TestAppContext) {
        init_keymap_test(cx);
        cx.update(|cx| {
            #[cfg(feature = "rust-tools")]
            cargo_ui::init(cx);
            tasks_ui::init(cx);
            #[cfg(feature = "jvm-tools")]
            jvm_ui::init(cx);

            reload_keymaps(cx, Vec::new());
            let down = gpui::Keystroke::parse("down").expect("Down keystroke");
            let keymap = cx.key_bindings();
            let keymap = keymap.borrow();
            for context in [
                #[cfg(feature = "rust-tools")]
                "CargoPanel",
                "TestsPanel",
                #[cfg(feature = "jvm-tools")]
                "GroovyDependenciesPanel",
            ] {
                let context = gpui::KeyContext::parse(context).expect("panel key context");
                let (bindings, _) =
                    keymap.bindings_for_input(std::slice::from_ref(&down), &[context]);
                assert_eq!(
                    bindings.first().map(|binding| binding.action().name()),
                    Some("language_tool_tree::SelectNext")
                );
            }
        });
    }

    actions!(test_only, [ActionA, ActionB]);

    /// The actions the emacs keymap resolves for `keystroke` in `context`.
    fn emacs_bindings_for(keystroke: &str, context: &str, cx: &mut TestAppContext) -> Vec<String> {
        cx.update(|cx| {
            let mut bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-linux.json",
                cx,
            )
            .unwrap();
            for binding in &mut bindings {
                binding.set_meta(settings::KeybindSource::Default.meta());
            }
            let mut emacs_bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/linux/emacs.json",
                cx,
            )
            .unwrap();
            for binding in &mut emacs_bindings {
                binding.set_meta(settings::KeybindSource::Base.meta());
            }
            bindings.extend(emacs_bindings);

            gpui::Keymap::new(bindings)
                .bindings_for_input(
                    &[gpui::Keystroke::parse(keystroke).unwrap()],
                    &[gpui::KeyContext::parse(context).unwrap()],
                )
                .0
                .iter()
                .map(|binding| binding.action().name().to_string())
                .collect()
        })
    }

    /// `editor::MoveDown` and `editor::MoveUp` propagate when the cursor doesn't move, which at the
    /// ends of a buffer let `ctrl-n` and `ctrl-p` fall through to the default bindings and open a
    /// new file / the file finder.
    #[gpui::test]
    fn test_emacs_cursor_keys_do_not_fall_back_to_default_bindings(cx: &mut TestAppContext) {
        init_keymap_test(cx);

        let ctrl_n = emacs_bindings_for("ctrl-n", "Workspace Editor", cx);
        assert!(
            ctrl_n.contains(&"editor::MoveDown".to_string()),
            "ctrl-n should still move down, got {ctrl_n:?}"
        );
        assert!(
            !ctrl_n.contains(&"workspace::NewFile".to_string()),
            "ctrl-n should not fall through to workspace::NewFile, got {ctrl_n:?}"
        );

        let ctrl_p = emacs_bindings_for("ctrl-p", "Workspace Editor", cx);
        assert!(
            ctrl_p.contains(&"editor::MoveUp".to_string()),
            "ctrl-p should still move up, got {ctrl_p:?}"
        );
        assert!(
            !ctrl_p.contains(&"file_finder::Toggle".to_string()),
            "ctrl-p should not fall through to file_finder::Toggle, got {ctrl_p:?}"
        );
    }

    /// The unbind above only targets `workspace::NewFile` / `file_finder::Toggle`, so the narrower
    /// `ctrl-n` and `ctrl-p` bindings still win where they apply.
    #[gpui::test]
    fn test_emacs_cursor_keys_keep_narrower_bindings(cx: &mut TestAppContext) {
        init_keymap_test(cx);

        let completions = "Workspace Editor showing_completions";
        assert_eq!(
            emacs_bindings_for("ctrl-n", completions, cx).first(),
            Some(&"editor::ContextMenuNext".to_string())
        );
        assert_eq!(
            emacs_bindings_for("ctrl-p", completions, cx).first(),
            Some(&"editor::ContextMenuPrevious".to_string())
        );

        let selection_mode = "Workspace Editor selection_mode";
        assert_eq!(
            emacs_bindings_for("ctrl-n", selection_mode, cx).first(),
            Some(&"editor::SelectDown".to_string())
        );
        assert_eq!(
            emacs_bindings_for("ctrl-p", selection_mode, cx).first(),
            Some(&"editor::SelectUp".to_string())
        );
    }

    #[gpui::test]
    async fn test_base_keymap(cx: &mut gpui::TestAppContext) {
        let executor = cx.executor();
        let app_state = init_keymap_test(cx);
        let project = Project::test(app_state.fs.clone(), [], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        // From the Atom keymap
        use workspace::ActivatePreviousPane;
        // From the JetBrains keymap
        use workspace::ActivatePreviousItem;
        // From the VSCode keymap
        use debugger_ui::Start;

        app_state
            .fs
            .save(
                paths::settings_file(),
                &r#"{"base_keymap": "Atom"}"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        app_state
            .fs
            .save(
                "/keymap.json".as_ref(),
                &r#"[{"bindings": {"backspace": "test_only::ActionA"}}]"#.into(),
                Default::default(),
            )
            .await
            .unwrap();
        executor.run_until_parked();
        cx.update(|cx| {
            let (keymap_rx, keymap_watcher) = watch_config_file(
                &executor,
                app_state.fs.clone(),
                PathBuf::from("/keymap.json"),
            );
            watch_settings_files(app_state.fs.clone(), cx);
            handle_keymap_file_changes(keymap_rx, keymap_watcher, cx);
        });
        window
            .update(cx, |_, _, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.register_action(|_, _: &ActionA, _window, _cx| {});
                    workspace.register_action(|_, _: &ActionB, _window, _cx| {});
                    workspace.register_action(|_, _: &ActivatePreviousPane, _window, _cx| {});
                    workspace.register_action(|_, _: &ActivatePreviousItem, _window, _cx| {});
                    cx.notify();
                });
            })
            .unwrap();
        executor.run_until_parked();
        // Test loading the keymap base at all
        assert_key_bindings_for(
            window.into(),
            cx,
            vec![("backspace", &ActionA), ("k", &ActivatePreviousPane)],
            line!(),
        );

        // Test modifying the users keymap, while retaining the base keymap
        app_state
            .fs
            .save(
                "/keymap.json".as_ref(),
                &r#"[{"bindings": {"backspace": "test_only::ActionB"}}]"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        executor.run_until_parked();

        assert_key_bindings_for(
            window.into(),
            cx,
            vec![("backspace", &ActionB), ("k", &ActivatePreviousPane)],
            line!(),
        );

        // Test modifying the base, while retaining the users keymap
        app_state
            .fs
            .save(
                paths::settings_file(),
                &r#"{"base_keymap": "JetBrains"}"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        executor.run_until_parked();

        assert_key_bindings_for(
            window.into(),
            cx,
            vec![
                ("backspace", &ActionB),
                ("{", &ActivatePreviousItem::default()),
            ],
            line!(),
        );

        // Test the VSCode keymap overlay
        app_state
            .fs
            .save(
                paths::settings_file(),
                &r#"{"base_keymap": "VSCode"}"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        executor.run_until_parked();

        window
            .update(cx, |_, _, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.register_action(|_, _: &Start, _window, _cx| {});
                    cx.notify();
                });
            })
            .unwrap();
        executor.run_until_parked();

        assert_key_bindings_for(
            window.into(),
            cx,
            vec![("backspace", &ActionB), ("f5", &Start)],
            line!(),
        );
    }

    #[gpui::test]
    async fn test_disabled_keymap_binding(cx: &mut gpui::TestAppContext) {
        let executor = cx.executor();
        let app_state = init_keymap_test(cx);
        let project = Project::test(app_state.fs.clone(), [], cx).await;
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();

        // From the Atom keymap
        use workspace::ActivatePreviousPane;
        // From the JetBrains keymap
        use diagnostics::Deploy;

        window
            .update(cx, |_, _, cx| {
                workspace.update(cx, |workspace, cx| {
                    workspace.register_action(|_, _: &ActionA, _window, _cx| {});
                    workspace.register_action(|_, _: &ActionB, _window, _cx| {});
                    workspace.register_action(|_, _: &Deploy, _window, _cx| {});
                    cx.notify();
                });
            })
            .unwrap();
        app_state
            .fs
            .save(
                paths::settings_file(),
                &r#"{"base_keymap": "Atom"}"#.into(),
                Default::default(),
            )
            .await
            .unwrap();
        app_state
            .fs
            .save(
                "/keymap.json".as_ref(),
                &r#"[{"bindings": {"backspace": "test_only::ActionA"}}]"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        cx.update(|cx| {
            let (keymap_rx, keymap_watcher) = watch_config_file(
                &executor,
                app_state.fs.clone(),
                PathBuf::from("/keymap.json"),
            );

            watch_settings_files(app_state.fs.clone(), cx);
            handle_keymap_file_changes(keymap_rx, keymap_watcher, cx);
        });

        cx.background_executor.run_until_parked();

        cx.background_executor.run_until_parked();
        // Test loading the keymap base at all
        assert_key_bindings_for(
            window.into(),
            cx,
            vec![("backspace", &ActionA), ("k", &ActivatePreviousPane)],
            line!(),
        );

        // Test disabling the key binding for the base keymap
        app_state
            .fs
            .save(
                "/keymap.json".as_ref(),
                &r#"[{"bindings": {"backspace": null}}]"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        cx.background_executor.run_until_parked();

        assert_key_bindings_for(
            window.into(),
            cx,
            vec![("k", &ActivatePreviousPane)],
            line!(),
        );

        // Test modifying the base, while retaining the users keymap
        app_state
            .fs
            .save(
                paths::settings_file(),
                &r#"{"base_keymap": "JetBrains"}"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        cx.background_executor.run_until_parked();

        assert_key_bindings_for(window.into(), cx, vec![("6", &Deploy)], line!());
    }

    #[gpui::test]
    fn graphics_surface_respects_sim_feature(cx: &mut TestAppContext) {
        init_keymap_test(cx);
        cx.update(|cx| {
            let is_graphics_action = |action: &str| {
                action.starts_with("graphite::")
                    || action.starts_with("sim::")
                    || action.starts_with("sim_")
                    || action.starts_with("comfy::")
                    || action.starts_with("comfy_")
            };
            let registered = cx.all_action_names();
            for action in [
                "graphite::OpenGraphite",
                "graphite::OpenPngEditor",
                "graphite::OpenSimDocument",
                "graphite::OpenPublishedPng",
                "graphite::FocusNextArtboardControl",
                "graphite::FocusPreviousArtboardControl",
                "sim_shell::QueuePrompt",
                "sim_graph::GraphZoomIn",
            ] {
                assert_eq!(
                    registered.contains(&action),
                    cfg!(feature = "sim"),
                    "{action}"
                );
            }
            if !cfg!(feature = "sim") {
                assert!(!registered.iter().any(|action| is_graphics_action(action)));
            }
            for _ in 0..2 {
                reload_keymaps(cx, Vec::new());
                let keymap = cx.key_bindings();
                let keymap = keymap.borrow();
                let context = gpui::KeyContext::parse("GraphiteArtboard")
                    .expect("Graphite Artboard key context");
                for (key, action) in [
                    ("tab", "graphite::FocusNextArtboardControl"),
                    ("shift-tab", "graphite::FocusPreviousArtboardControl"),
                ] {
                    let keystroke = gpui::Keystroke::parse(key).expect("Artboard navigation key");
                    let (bindings, _) =
                        keymap.bindings_for_input(&[keystroke], std::slice::from_ref(&context));
                    assert_eq!(
                        bindings
                            .iter()
                            .any(|binding| binding.action().name() == action),
                        cfg!(feature = "sim"),
                        "{action} after keymap reload",
                    );
                }
                let context = gpui::KeyContext::parse("SimGraph").expect("Sim graph key context");
                let keystroke = gpui::Keystroke::parse("ctrl-enter").expect("Sim queue key");
                let (bindings, _) =
                    keymap.bindings_for_input(&[keystroke], std::slice::from_ref(&context));
                assert_eq!(
                    bindings
                        .iter()
                        .any(|binding| binding.action().name() == "sim_shell::QueuePrompt"),
                    cfg!(feature = "sim"),
                    "Sim queue after keymap reload",
                );
                if !cfg!(feature = "sim") {
                    assert!(
                        !keymap
                            .bindings()
                            .any(|binding| is_graphics_action(binding.action().name()))
                    );
                }
            }
            let mut menu_actions = Vec::new();
            for menu in app_menus(cx) {
                collect_menu_action_names(&menu.items, &mut menu_actions);
            }
            if !cfg!(feature = "sim") {
                assert!(!menu_actions.iter().any(|action| is_graphics_action(action)));
            }
        });
    }

    #[gpui::test]
    async fn test_generate_keymap_json_schema_for_registered_actions(
        cx: &mut gpui::TestAppContext,
    ) {
        init_keymap_test(cx);
        cx.update(|cx| {
            // Make sure it doesn't panic.
            KeymapFile::generate_json_schema_for_registered_actions(cx);
        });
    }

    /// Checks that action namespaces are the expected set. The purpose of this is to prevent typos
    /// and let you know when introducing a new namespace.
    #[gpui::test]
    async fn test_action_namespaces(cx: &mut gpui::TestAppContext) {
        use itertools::Itertools;

        init_keymap_test(cx);
        cx.update(|cx| {
            let all_actions = cx.all_action_names();

            let mut actions_without_namespace = Vec::new();
            let all_namespaces = all_actions
                .iter()
                .filter_map(|action_name| {
                    let namespace = action_name
                        .split("::")
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .skip(1)
                        .rev()
                        .join("::");
                    if namespace.is_empty() {
                        actions_without_namespace.push(*action_name);
                    }
                    if &namespace == "test_only" || &namespace == "stories" {
                        None
                    } else {
                        Some(namespace)
                    }
                })
                .sorted()
                .dedup()
                .collect::<Vec<_>>();
            assert_eq!(actions_without_namespace, Vec::<&str>::new());

            #[cfg(not(feature = "agentic-tools"))]
            for action_name in [
                "editor::CancelEditReviewComment",
                "editor::ConfirmEditReviewComment",
                "editor::DeleteReviewComment",
                "editor::EditReviewComment",
                "editor::SendReviewToAgent",
                "editor::SubmitDiffReviewComment",
                "editor::ToggleReviewCommentsExpanded",
                "zed::AcpRegistry",
            ] {
                assert!(
                    !all_actions.contains(&action_name),
                    "agentic action registered in disabled build: {action_name}"
                );
            }

            let expected_namespaces = vec![
                "action",
                "activity_indicator",
                #[cfg(feature = "agentic-tools")]
                "agent",
                #[cfg(feature = "agentic-tools")]
                "agents_sidebar",
                "app_menu",
                #[cfg(feature = "agentic-tools")]
                "assistant",
                #[cfg(feature = "agentic-tools")]
                "assistant2",
                "auto_update",
                "branch_picker",
                #[cfg(feature = "agentic-tools")]
                "bedrock",
                "branches",
                "buffer_search",
                "call_hierarchy",
                #[cfg(feature = "rust-tools")]
                "cargo_actions",
                #[cfg(feature = "rust-tools")]
                "cargo_panel",
                #[cfg(feature = "rust-tools")]
                "cargo_preset_editor",
                "channel_modal",
                "cli",
                "client",
                "collab",
                "collab_panel",
                #[cfg(feature = "multiplayer-tools")]
                "collaborative_workspace",
                #[cfg(feature = "sim")]
                "sim_graph",
                #[cfg(feature = "sim")]
                "sim_shell",
                "command_palette",
                "console",
                "context_server",
                "copilot",
                "copilot_edit_predictions",
                "csv",
                "debug_panel",
                "debugger",
                "dev",
                "diagnostics",
                "edit_prediction",
                "editor",
                "encoding_selector",
                "feedback",
                "file_finder",
                "git",
                "git_graph",
                "git_onboarding",
                "git_panel",
                "git_picker",
                "go_to_line",
                #[cfg(feature = "sim")]
                "graphite",
                "highlights_tree_view",
                "icon_theme_selector",
                "image_viewer",
                #[cfg(feature = "agentic-tools")]
                "inline_assistant",
                "journal",
                "keymap_editor",
                "keystroke_input",
                "language_selector",
                "language_tool_tree",
                "welcome",
                "line_ending_selector",
                "lsp_tool",
                "markdown",
                "menu",
                "multi_workspace",
                "new_process_modal",
                "notebook",
                "onboarding",
                "outline",
                "outline_panel",
                "pane",
                "panel",
                "picker",
                "profiling",
                "project_panel",
                "project_search",
                "project_symbols",
                "projects",
                "recent_projects",
                "remote_debug",
                "repl",
                "search",
                "settings_editor",
                "settings_profile_selector",
                #[cfg(feature = "agentic-tools")]
                "skill_creator",
                "snippets",
                "stash_picker",
                "svg",
                "syntax_tree_view",
                "tab_switcher",
                "task",
                "terminal",
                "terminal_panel",
                #[cfg(feature = "rust-tools")]
                "test_explorer",
                "text_finder",
                "theme",
                "theme_selector",
                "toast",
                "toolchain",
                "variable_list",
                "vim",
                "window",
                "workspace",
                "worktree_picker",
                "zed",
                "zed_actions",
                "zed_predict_onboarding",
                "zeta",
            ];
            assert_eq!(
                all_namespaces,
                expected_namespaces
                    .into_iter()
                    .map(|namespace| namespace.to_string())
                    .sorted()
                    .collect::<Vec<_>>()
            );
        });
    }

    #[gpui::test]
    fn test_bundled_settings_and_themes(cx: &mut App) {
        cx.text_system()
            .add_fonts(vec![
                Assets
                    .load("fonts/lilex/Lilex-Regular.ttf")
                    .unwrap()
                    .unwrap(),
                Assets
                    .load("fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf")
                    .unwrap()
                    .unwrap(),
            ])
            .unwrap();
        let themes = ThemeRegistry::default();
        settings::init(cx);
        theme_settings::init(theme::LoadThemes::JustBase, cx);

        let mut has_default_theme = false;
        for theme_name in themes.list().into_iter().map(|meta| meta.name) {
            let theme = themes.get(&theme_name).unwrap();
            assert_eq!(theme.name, theme_name);
            if theme.name.as_ref() == "One Dark" {
                has_default_theme = true;
            }
        }
        assert!(has_default_theme);
    }

    #[gpui::test]
    async fn test_bundled_files_editor(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        cx.update(init);

        let project = Project::test(app_state.fs.clone(), [], cx).await;
        let _window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));

        cx.update(|cx| {
            cx.dispatch_action(&OpenDefaultSettings);
        });
        cx.run_until_parked();

        assert_eq!(cx.read(|cx| cx.windows().len()), 1);

        let multi_workspace = cx.windows()[0].downcast::<MultiWorkspace>().unwrap();
        let active_editor = multi_workspace
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace
                    .workspace()
                    .update(cx, |workspace, cx| workspace.active_item_as::<Editor>(cx))
            })
            .unwrap();
        assert!(
            active_editor.is_some(),
            "Settings action should have opened an editor with the default file contents"
        );

        let active_editor = active_editor.unwrap();
        assert!(
            active_editor.read_with(cx, |editor, cx| editor.read_only(cx)),
            "Default settings should be readonly"
        );
        assert!(
            active_editor.read_with(cx, |editor, cx| editor.buffer().read(cx).read_only()),
            "The underlying buffer should also be readonly for the shipped default settings"
        );
    }

    #[gpui::test]
    async fn test_bundled_files_reuse_existing_editor(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        cx.update(init);

        let project = Project::test(app_state.fs.clone(), [], cx).await;
        let _window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));

        cx.update(|cx| {
            cx.dispatch_action(&OpenDefaultSettings);
        });
        cx.run_until_parked();

        let multi_workspace = cx.windows()[0].downcast::<MultiWorkspace>().unwrap();
        let first_item_id = multi_workspace
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    workspace
                        .active_item(cx)
                        .expect("default settings should be open")
                        .item_id()
                })
            })
            .unwrap();

        cx.update(|cx| {
            cx.dispatch_action(&OpenDefaultSettings);
        });
        cx.run_until_parked();

        let (second_item_id, item_count) = multi_workspace
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.workspace().update(cx, |workspace, cx| {
                    let pane = workspace.active_pane().read(cx);
                    (
                        pane.active_item()
                            .expect("default settings should still be open")
                            .item_id(),
                        pane.items_len(),
                    )
                })
            })
            .unwrap();

        assert_eq!(first_item_id, second_item_id);
        assert_eq!(item_count, 1);
    }

    #[gpui::test]
    async fn test_bundled_languages(cx: &mut TestAppContext) {
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        env_logger::builder().is_test(true).try_init().ok();
        let settings = cx.update(SettingsStore::test);
        cx.set_global(settings);
        let languages = LanguageRegistry::test(cx.executor());
        let languages = Arc::new(languages);
        let node_runtime = node_runtime::NodeRuntime::unavailable();
        cx.update(|cx| {
            languages::init(languages.clone(), fs, node_runtime, cx);
        });
        for name in languages.language_names() {
            languages
                .language_for_name(name.as_ref())
                .await
                .with_context(|| format!("language name {name}"))
                .unwrap();
        }
        cx.run_until_parked();
    }

    #[gpui::test]
    fn editor_prediction_initializes_without_agent_tools(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        cx.update(|cx| {
            edit_prediction_registry::init(
                app_state.client.clone(),
                app_state.user_store.clone(),
                cx,
            );
            assert!(edit_prediction::EditPredictionStore::try_global(cx).is_some());
        });
    }

    pub(crate) fn init_test(cx: &mut TestAppContext) -> Arc<AppState> {
        let app_state = cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            AppState::test(cx)
        });
        init_test_with_state(cx, app_state)
    }

    fn init_test_with_state(
        cx: &mut TestAppContext,
        mut app_state: Arc<AppState>,
    ) -> Arc<AppState> {
        cx.update(move |cx| {
            env_logger::builder().is_test(true).try_init().ok();

            let state = Arc::get_mut(&mut app_state).unwrap();
            state.build_window_options = build_window_options;
            app_state.languages.add(markdown_lang());

            gpui_tokio::init(cx);
            AppState::set_global(app_state.clone(), cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            audio::init(cx);
            channel::init(&app_state.client, app_state.user_store.clone(), cx);
            call::init(app_state.client.clone(), app_state.user_store.clone(), cx);
            notifications::init(app_state.client.clone(), app_state.user_store.clone(), cx);
            workspace::init(app_state.clone(), cx);
            release_channel::init(Version::new(0, 0, 0), cx);
            command_palette::init(cx);
            editor::init(cx);
            #[cfg(feature = "sim")]
            init_sim_ui(cx);
            collab_ui::init(&app_state, cx);
            git_ui::init(cx);
            project_panel::init(cx);
            outline_panel::init(cx);
            #[cfg(feature = "rust-tools")]
            cargo_ui::init(cx);
            #[cfg(feature = "jvm-tools")]
            jvm_ui::init(cx);
            terminal_view::init(cx);
            let credentials_provider = zed_credentials_provider::global(cx);
            copilot_chat::init(
                app_state.client.http_client(),
                credentials_provider,
                copilot_chat::CopilotChatConfiguration::default(),
                cx,
            );
            image_viewer::init(cx);
            language_model::init(cx);
            client::RefreshLlmTokenListener::register(
                app_state.client.clone(),
                app_state.user_store.clone(),
                cx,
            );
            #[cfg(feature = "agentic-tools")]
            {
                language_models::init(app_state.user_store.clone(), app_state.client.clone(), cx);
                web_search::init(cx);
                web_search_providers::init(
                    app_state.client.clone(),
                    app_state.user_store.clone(),
                    cx,
                );
                let prompt_builder = PromptBuilder::load(app_state.fs.clone(), false, cx);
                project::AgentRegistryStore::init_global(
                    cx,
                    app_state.fs.clone(),
                    app_state.client.http_client(),
                );
                agent_ui::init(
                    app_state.fs.clone(),
                    prompt_builder,
                    app_state.languages.clone(),
                    true,
                    false,
                    false,
                    cx,
                );
            }

            repl::init(app_state.fs.clone(), cx);
            repl::notebook::init(cx);
            tasks_ui::init(cx);
            project::debugger::breakpoint_store::BreakpointStore::init(
                &app_state.client.clone().into(),
            );
            project::debugger::dap_store::DapStore::init(&app_state.client.clone().into(), cx);
            debugger_ui::init(cx);
            initialize_workspace(app_state.clone(), cx);
            search::init(cx);
            lsp_locations::init(cx);
            cx.set_global(workspace::PaneSearchBarCallbacks {
                setup_search_bar: |languages, toolbar, window, cx| {
                    let search_bar =
                        cx.new(|cx| search::BufferSearchBar::new(languages, window, cx));
                    toolbar.update(cx, |toolbar, cx| {
                        toolbar.add_item(search_bar, window, cx);
                    });
                },
                wrap_div_with_search_actions: search::buffer_search::register_pane_search_actions,
            });
            app_state
        })
    }

    #[track_caller]
    fn assert_key_bindings_for(
        window: AnyWindowHandle,
        cx: &TestAppContext,
        actions: Vec<(&'static str, &dyn Action)>,
        line: u32,
    ) {
        let available_actions = cx
            .update(|cx| window.update(cx, |_, window, cx| window.available_actions(cx)))
            .unwrap();
        for (key, action) in actions {
            let bindings = cx
                .update(|cx| window.update(cx, |_, window, _| window.bindings_for_action(action)))
                .unwrap();
            // assert that...
            assert!(
                available_actions.iter().any(|bound_action| {
                    // actions match...
                    bound_action.partial_eq(action)
                }),
                "On {} Failed to find {}",
                line,
                action.name(),
            );
            assert!(
                // and key strokes contain the given key
                bindings
                    .into_iter()
                    .any(|binding| binding.keystrokes().iter().any(|k| k.key() == key)),
                "On {} Failed to find {} with key binding {}",
                line,
                action.name(),
                key
            );
        }
    }

    #[gpui::test]
    async fn test_opening_project_settings_when_excluded(cx: &mut gpui::TestAppContext) {
        // Use the proper initialization for runtime state
        let app_state = init_keymap_test(cx);

        eprintln!("Running test_opening_project_settings_when_excluded");

        // 1. Set up a project with some project settings
        let settings_init =
            r#"{ "UNIQUEVALUE": true, "git": { "inline_blame": { "enabled": false } } }"#;
        app_state
            .fs
            .as_fake()
            .insert_tree(
                Path::new("/root"),
                json!({
                    ".zed": {
                        "settings.json": settings_init
                    }
                }),
            )
            .await;

        eprintln!("Created project with .zed/settings.json containing UNIQUEVALUE");

        // 2. Create a project with the file system and load it
        let project = Project::test(app_state.fs.clone(), [Path::new("/root")], cx).await;

        // Save original settings content for comparison
        let original_settings = app_state
            .fs
            .load(Path::new("/root/.zed/settings.json"))
            .await
            .unwrap();

        let original_settings_str = original_settings.clone();

        // Verify settings exist on disk and have expected content
        eprintln!("Original settings content: {}", original_settings_str);
        assert!(
            original_settings_str.contains("UNIQUEVALUE"),
            "Test setup failed - settings file doesn't contain our marker"
        );

        // 3. Add .zed to file scan exclusions in user settings
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |worktree_settings| {
                worktree_settings.project.worktree.file_scan_exclusions =
                    Some(vec![".zed".to_string()]);
            });
        });

        eprintln!("Added .zed to file_scan_exclusions in settings");

        // 4. Run tasks to apply settings
        cx.background_executor.run_until_parked();

        // 5. Critical: Verify .zed is actually excluded from worktree
        let worktree = cx.update(|cx| project.read(cx).worktrees(cx).next().unwrap());

        let has_zed_entry =
            cx.update(|cx| worktree.read(cx).entry_for_path(rel_path(".zed")).is_some());

        eprintln!(
            "Is .zed directory visible in worktree after exclusion: {}",
            has_zed_entry
        );

        // This assertion verifies the test is set up correctly to show the bug
        // If .zed is not excluded, the test will fail here
        assert!(
            !has_zed_entry,
            "Test precondition failed: .zed directory should be excluded but was found in worktree"
        );

        // 6. Create workspace and trigger the actual function that causes the bug
        let window =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = window
            .read_with(cx, |mw, _| mw.workspace().clone())
            .unwrap();
        window
            .update(cx, |_, window, cx| {
                workspace.update(cx, |workspace, cx| {
                    // Call the exact function that contains the bug
                    eprintln!("About to call open_project_settings_file");
                    open_project_settings_file(workspace, &OpenProjectSettingsFile, window, cx);
                });
            })
            .unwrap();

        // 7. Run background tasks until completion
        cx.background_executor.run_until_parked();

        // 8. Verify file contents after calling function
        let new_content = app_state
            .fs
            .load(Path::new("/root/.zed/settings.json"))
            .await
            .unwrap();

        let new_content_str = new_content;
        eprintln!("New settings content: {}", new_content_str);

        // The bug causes the settings to be overwritten with empty settings
        // So if the unique value is no longer present, the bug has been reproduced
        let bug_exists = !new_content_str.contains("UNIQUEVALUE");
        eprintln!("Bug reproduced: {}", bug_exists);

        // This assertion should fail if the bug exists - showing the bug is real
        assert!(
            new_content_str.contains("UNIQUEVALUE"),
            "BUG FOUND: Project settings were overwritten when opening via command - original custom content was lost"
        );
    }

    #[gpui::test]
    async fn test_disable_ai_crash(cx: &mut gpui::TestAppContext) {
        let app_state = init_test(cx);
        cx.update(init);
        let project = Project::test(app_state.fs.clone(), [], cx).await;
        let _window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));

        cx.run_until_parked();

        cx.update(|cx| {
            SettingsStore::update_global(cx, |settings_store, cx| {
                settings_store.update_user_settings(cx, |settings| {
                    settings.project.disable_ai = Some(SaturatingBool(true));
                });
            });
        });

        cx.run_until_parked();

        // If this panics, the test has failed
    }

    #[gpui::test]
    async fn test_invalid_global_tasks_file_shows_notification_on_startup(
        cx: &mut gpui::TestAppContext,
    ) {
        let app_state = init_test(cx);
        let tasks_file_path = paths::tasks_file().as_path();
        app_state
            .fs
            .create_dir(tasks_file_path.parent().unwrap())
            .await
            .unwrap();
        app_state
            .fs
            .save(
                tasks_file_path,
                &r#"[{ "label": "first" }] [{ "label": "trailing garbage" }]"#.into(),
                Default::default(),
            )
            .await
            .unwrap();

        let project = Project::test(app_state.fs.clone(), [], cx).await;
        let window = cx.add_window(|window, cx| MultiWorkspace::test_new(project, window, cx));
        cx.run_until_parked();

        let workspace = window
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .unwrap();
        let notification_id = NotificationId::Named("invalid-global-tasks-file".into());
        let shown_notifications = workspace.read_with(cx, |workspace, _| {
            workspace
                .notification_ids()
                .into_iter()
                .filter(|id| *id == notification_id)
                .count()
        });
        assert_eq!(
            shown_notifications, 1,
            "invalid global tasks file at startup should show an app notification"
        );

        app_state
            .fs
            .save(
                tasks_file_path,
                &r#"[{ "label": "first", "command": "echo" }]"#.into(),
                Default::default(),
            )
            .await
            .unwrap();
        cx.run_until_parked();

        let shown_notifications = workspace.read_with(cx, |workspace, _| {
            workspace
                .notification_ids()
                .into_iter()
                .filter(|id| *id == notification_id)
                .count()
        });
        assert_eq!(
            shown_notifications, 0,
            "fixing the global tasks file should dismiss the notification"
        );
    }

    #[cfg(feature = "agentic-tools")]
    #[gpui::test]
    async fn test_disable_ai_filters_keybindings(cx: &mut gpui::TestAppContext) {
        let _app_state = init_keymap_test(cx);

        // With AI enabled, the default keymap should include the assistant
        // bindings that intercept e.g. ctrl-enter in the editor.
        cx.update(load_default_keymap);
        cx.update(|cx| {
            let keymap = cx.key_bindings();
            let keymap = keymap.borrow();
            let has_ai_binding = keymap.bindings().any(|binding| is_ai_keybinding(binding));
            assert!(
                has_ai_binding,
                "expected AI-namespaced bindings in the default keymap before disabling AI"
            );
        });

        cx.update(|cx| {
            SettingsStore::update_global(cx, |settings_store, cx| {
                settings_store.update_user_settings(cx, |settings| {
                    settings.project.disable_ai = Some(SaturatingBool(true));
                });
            });
        });

        // The default keymap should drop every AI-namespaced binding so that
        // lower-precedence editor defaults can run instead.
        cx.update(|cx| {
            cx.clear_key_bindings();
            load_default_keymap(cx);
        });
        cx.update(|cx| {
            let keymap = cx.key_bindings();
            let keymap = keymap.borrow();
            if let Some(binding) = keymap.bindings().find(|b| is_ai_keybinding(b)) {
                panic!(
                    "expected no AI-namespaced bindings after disabling AI, but found `{}`",
                    binding.action().name()
                );
            }
        });

        // User-defined bindings to AI actions should also be filtered.
        let user_binding = KeyBinding::new(
            "ctrl-enter",
            zed_actions::assistant::InlineAssist { prompt: None },
            None,
        );
        cx.update(|cx| reload_keymaps(cx, vec![user_binding]));
        cx.update(|cx| {
            let keymap = cx.key_bindings();
            let keymap = keymap.borrow();
            if let Some(binding) = keymap.bindings().find(|b| is_ai_keybinding(b)) {
                panic!(
                    "expected user binding `{}` to be filtered when AI is disabled",
                    binding.action().name()
                );
            }
        });
    }

    #[gpui::test]
    async fn test_prefer_focused_window(cx: &mut gpui::TestAppContext) {
        let app_state = init_test(cx);
        let paths = [PathBuf::from(path!("/dir/document.txt"))];

        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/dir"),
                json!({
                    "document.txt": "Some of the documentation's content."
                }),
            )
            .await;

        let project_a = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window_a = cx.add_window({
            let project = project_a.clone();
            |window, cx| MultiWorkspace::test_new(project, window, cx)
        });

        let project_b = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window_b = cx.add_window({
            let project = project_b.clone();
            |window, cx| MultiWorkspace::test_new(project, window, cx)
        });

        let project_c = Project::test(app_state.fs.clone(), [path!("/dir").as_ref()], cx).await;
        let window_c = cx.add_window({
            let project = project_c.clone();
            |window, cx| MultiWorkspace::test_new(project, window, cx)
        });

        for window in [window_a, window_b, window_c] {
            let _ = cx.update_window(*window, |_, window, _| {
                window.activate_window();
            });

            cx.update(|cx| {
                let open_options = OpenOptions {
                    wait: true,
                    ..Default::default()
                };

                workspace::open_paths(&paths, app_state.clone(), open_options, cx)
            })
            .await
            .unwrap();

            cx.update_window(*window, |_, window, _| assert!(window.is_window_active()))
                .unwrap();

            let _ = window.read_with(cx, |multi_workspace, cx| {
                let pane = multi_workspace.workspace().read(cx).active_pane().read(cx);
                let project_path = pane.active_item().unwrap().project_path(cx).unwrap();

                assert_eq!(
                    project_path.path.as_ref().as_std_path().to_str().unwrap(),
                    path!("document.txt")
                )
            });
        }
    }

    #[gpui::test]
    async fn test_open_paths_switches_to_best_workspace(cx: &mut TestAppContext) {
        let app_state = init_test(cx);

        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/"),
                json!({
                    "dir1": {
                        "a.txt": "content a"
                    },
                    "dir2": {
                        "b.txt": "content b"
                    },
                    "dir3": {
                        "c.txt": "content c"
                    }
                }),
            )
            .await;

        // Create a window with workspace 0 containing /dir1
        let project1 = Project::test(app_state.fs.clone(), [path!("/dir1").as_ref()], cx).await;

        let window = cx.add_window({
            let project = project1.clone();
            |window, cx| MultiWorkspace::test_new(project, window, cx)
        });
        window
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.open_sidebar(cx);
            })
            .unwrap();

        cx.run_until_parked();
        assert_eq!(cx.windows().len(), 1, "Should start with 1 window");

        // Create workspace 2 with /dir2
        let project2 = Project::test(app_state.fs.clone(), [path!("/dir2").as_ref()], cx).await;
        let workspace2 = window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project2.clone(), window, cx)
            })
            .unwrap();

        // Create workspace 3 with /dir3
        let project3 = Project::test(app_state.fs.clone(), [path!("/dir3").as_ref()], cx).await;
        let workspace3 = window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project3.clone(), window, cx)
            })
            .unwrap();

        let workspace1 = window
            .read_with(cx, |multi_workspace, _| {
                multi_workspace.workspaces().next().unwrap().clone()
            })
            .unwrap();

        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.activate(workspace2.clone(), None, window, cx);
                multi_workspace.activate(workspace3.clone(), None, window, cx);
                // Switch back to workspace1 for test setup
                multi_workspace.activate(workspace1.clone(), None, window, cx);
                assert_eq!(multi_workspace.workspace(), &workspace1);
            })
            .unwrap();

        cx.run_until_parked();

        // Verify setup: 3 workspaces, workspace 0 active, still 1 window
        window
            .read_with(cx, |multi_workspace, _| {
                assert_eq!(multi_workspace.workspaces().count(), 3);
                assert_eq!(multi_workspace.workspace(), &workspace1);
            })
            .unwrap();
        assert_eq!(cx.windows().len(), 1);

        // Open a file in /dir3 - should switch to workspace 3 (not just "the other one")
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/dir3/c.txt"))],
                app_state.clone(),
                OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();

        cx.run_until_parked();

        // Verify workspace 2 is active and file opened there
        window
            .read_with(cx, |multi_workspace, cx| {
                assert_eq!(
                    multi_workspace.workspace(),
                    &workspace3,
                    "Should have switched to workspace 3 which contains /dir3"
                );
                let active_item = multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .expect("Should have an active item");
                assert_eq!(active_item.tab_content_text(0, cx), "c.txt");
            })
            .unwrap();
        assert_eq!(cx.windows().len(), 1, "Should reuse existing window");

        // Open a file in /dir2 - should switch to workspace 2
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/dir2/b.txt"))],
                app_state.clone(),
                OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();

        cx.run_until_parked();

        // Verify workspace 1 is active and file opened there
        window
            .read_with(cx, |multi_workspace, cx| {
                assert_eq!(
                    multi_workspace.workspace(),
                    &workspace2,
                    "Should have switched to workspace 2 which contains /dir2"
                );
                let active_item = multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .expect("Should have an active item");
                assert_eq!(active_item.tab_content_text(0, cx), "b.txt");
            })
            .unwrap();

        // Verify c.txt is still in workspace 3 (file opened in correct workspace, not active one)
        workspace3.read_with(cx, |workspace, cx| {
            let active_item = workspace
                .active_pane()
                .read(cx)
                .active_item()
                .expect("Workspace 2 should have an active item");
            assert_eq!(
                active_item.tab_content_text(0, cx),
                "c.txt",
                "c.txt should have been opened in workspace 3, not the active workspace"
            );
        });

        assert_eq!(cx.windows().len(), 1, "Should still have only 1 window");

        // Open a file in /dir1 - should switch back to workspace 0
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/dir1/a.txt"))],
                app_state.clone(),
                OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();

        cx.run_until_parked();

        // Verify workspace 0 is active and file opened there
        window
            .read_with(cx, |multi_workspace, cx| {
                assert_eq!(
                    multi_workspace.workspace(),
                    &workspace1,
                    "Should have switched back to workspace 0 which contains /dir1"
                );
                let active_item = multi_workspace
                    .workspace()
                    .read(cx)
                    .active_pane()
                    .read(cx)
                    .active_item()
                    .expect("Should have an active item");
                assert_eq!(active_item.tab_content_text(0, cx), "a.txt");
            })
            .unwrap();
        assert_eq!(cx.windows().len(), 1, "Should still have only 1 window");
    }

    #[gpui::test]
    async fn test_open_paths_in_gitignored_dir_opens_new_workspace(cx: &mut TestAppContext) {
        let app_state = init_test(cx);

        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/project"),
                json!({
                    ".git": {},
                    ".gitignore": ".checkouts/\n",
                    "src": {
                        "main.rs": "fn main() {}"
                    },
                    ".checkouts": {
                        "worktrees": {
                            "foo": {
                                "README.md": "hello"
                            }
                        }
                    }
                }),
            )
            .await;

        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/project"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        cx.run_until_parked();
        assert_eq!(cx.update(|cx| cx.windows().len()), 1);

        // Opening a directory inside a gitignored folder must not be treated
        // as contained by the open project: its contents were never scanned,
        // and it may be an independent checkout (e.g. a git worktree kept in
        // an ignored directory). It should become its own workspace root
        // instead.
        cx.update(|cx| {
            open_paths(
                &[PathBuf::from(path!("/project/.checkouts/worktrees/foo"))],
                app_state.clone(),
                workspace::OpenOptions::default(),
                cx,
            )
        })
        .await
        .unwrap();
        cx.run_until_parked();

        let workspace_roots = cx.update(|cx| {
            cx.windows()
                .into_iter()
                .filter_map(|window| window.downcast::<MultiWorkspace>())
                .flat_map(|window| {
                    let mut roots = Vec::new();
                    if let Ok(multi_workspace) = window.read(cx) {
                        for workspace in multi_workspace.workspaces() {
                            roots.push(
                                workspace
                                    .read(cx)
                                    .worktrees(cx)
                                    .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                                    .collect::<Vec<_>>(),
                            );
                        }
                    }
                    roots
                })
                .collect::<Vec<_>>()
        });
        assert!(
            workspace_roots.contains(&vec![PathBuf::from(path!(
                "/project/.checkouts/worktrees/foo"
            ))]),
            "the gitignored directory should be the root of its own workspace, got {workspace_roots:?}"
        );
        assert!(
            workspace_roots.contains(&vec![PathBuf::from(path!("/project"))]),
            "the original project workspace should be unchanged, got {workspace_roots:?}"
        );
    }

    #[gpui::test]
    async fn test_quit_checks_all_workspaces_for_dirty_items(cx: &mut TestAppContext) {
        let app_state = init_test(cx);
        cx.update(init);

        app_state
            .fs
            .as_fake()
            .insert_tree(
                path!("/"),
                json!({
                    "dir1": {
                        "a.txt": "content a"
                    },
                    "dir2": {
                        "b.txt": "content b"
                    },
                    "dir3": {
                        "c.txt": "content c"
                    }
                }),
            )
            .await;

        // === Setup Window 1 with two workspaces ===
        let project1 = Project::test(app_state.fs.clone(), [path!("/dir1").as_ref()], cx).await;
        let window1 = cx.add_window({
            let project = project1.clone();
            |window, cx| MultiWorkspace::test_new(project, window, cx)
        });
        window1
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.open_sidebar(cx);
            })
            .unwrap();

        cx.run_until_parked();

        let project2 = Project::test(app_state.fs.clone(), [path!("/dir2").as_ref()], cx).await;
        let workspace1_1 = window1
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .unwrap();
        let workspace1_2 = window1
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project2.clone(), window, cx)
            })
            .unwrap();

        window1
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.activate(workspace1_2.clone(), None, window, cx);
                multi_workspace.activate(workspace1_1.clone(), None, window, cx);
            })
            .unwrap();

        // === Setup Window 2 with one workspace ===
        let project3 = Project::test(app_state.fs.clone(), [path!("/dir3").as_ref()], cx).await;
        let window2 = cx.add_window({
            let project = project3.clone();
            |window, cx| MultiWorkspace::test_new(project, window, cx)
        });
        window2
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.open_sidebar(cx);
            })
            .unwrap();

        cx.run_until_parked();
        assert_eq!(cx.windows().len(), 2);

        // === Case 1: Active workspace has dirty item, quit can be cancelled ===
        let worktree1_id = project1.update(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        let editor1 = window1
            .update(cx, |_, window, cx| {
                workspace1_1.update(cx, |workspace, cx| {
                    workspace.open_path((worktree1_id, rel_path("a.txt")), None, true, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();

        window1
            .update(cx, |_, window, cx| {
                editor1.update(cx, |editor, cx| {
                    editor.insert("dirty in active workspace", window, cx);
                });
            })
            .unwrap();

        cx.run_until_parked();

        // Verify workspace1_1 is active
        window1
            .read_with(cx, |multi_workspace, _| {
                assert_eq!(multi_workspace.workspace(), &workspace1_1);
            })
            .unwrap();

        cx.dispatch_action(*window1, Quit);
        cx.run_until_parked();

        assert!(
            cx.has_pending_prompt(),
            "Case 1: Should prompt to save dirty item in active workspace"
        );

        cx.simulate_prompt_answer("Cancel");
        cx.run_until_parked();

        assert_eq!(
            cx.windows().len(),
            2,
            "Case 1: Windows should still exist after cancelling quit"
        );

        // Clean up Case 1: Close the dirty item without saving
        let close_task = window1
            .update(cx, |_, window, cx| {
                workspace1_1.update(cx, |workspace, cx| {
                    workspace.active_pane().update(cx, |pane, cx| {
                        pane.close_active_item(&Default::default(), window, cx)
                    })
                })
            })
            .unwrap();
        cx.run_until_parked();
        cx.simulate_prompt_answer("Don't Save");
        close_task.await.ok();
        cx.run_until_parked();

        // === Case 2: Non-active workspace (same window) has dirty item ===
        let worktree2_id = project2.update(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        let editor2 = window1
            .update(cx, |_, window, cx| {
                workspace1_2.update(cx, |workspace, cx| {
                    workspace.open_path((worktree2_id, rel_path("b.txt")), None, true, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();

        window1
            .update(cx, |_, window, cx| {
                editor2.update(cx, |editor, cx| {
                    editor.insert("dirty in non-active workspace", window, cx);
                });
            })
            .unwrap();

        cx.run_until_parked();

        // Verify workspace1_1 is still active (not workspace1_2 with dirty item)
        window1
            .read_with(cx, |multi_workspace, _| {
                assert_eq!(multi_workspace.workspace(), &workspace1_1);
            })
            .unwrap();

        cx.dispatch_action(*window1, Quit);
        cx.run_until_parked();

        // Verify the non-active workspace got activated to show the dirty item
        window1
            .read_with(cx, |multi_workspace, _| {
                assert_eq!(
                    multi_workspace.workspace(),
                    &workspace1_2,
                    "Case 2: Non-active workspace should be activated when it has dirty item"
                );
            })
            .unwrap();

        assert!(
            cx.has_pending_prompt(),
            "Case 2: Should prompt to save dirty item in non-active workspace"
        );

        cx.simulate_prompt_answer("Cancel");
        cx.run_until_parked();

        assert_eq!(
            cx.windows().len(),
            2,
            "Case 2: Windows should still exist after cancelling quit"
        );

        // Clean up Case 2: Close the dirty item without saving
        let close_task = window1
            .update(cx, |_, window, cx| {
                workspace1_2.update(cx, |workspace, cx| {
                    workspace.active_pane().update(cx, |pane, cx| {
                        pane.close_active_item(&Default::default(), window, cx)
                    })
                })
            })
            .unwrap();
        cx.run_until_parked();
        cx.simulate_prompt_answer("Don't Save");
        close_task.await.ok();
        cx.run_until_parked();

        // === Case 3: Non-active window has dirty item ===
        let workspace3 = window2
            .read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone())
            .unwrap();

        let worktree3_id = project3.update(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        let editor3 = window2
            .update(cx, |_, window, cx| {
                workspace3.update(cx, |workspace, cx| {
                    workspace.open_path((worktree3_id, rel_path("c.txt")), None, true, window, cx)
                })
            })
            .unwrap()
            .await
            .unwrap()
            .downcast::<Editor>()
            .unwrap();

        window2
            .update(cx, |_, window, cx| {
                editor3.update(cx, |editor, cx| {
                    editor.insert("dirty in other window", window, cx);
                });
            })
            .unwrap();

        cx.run_until_parked();

        // Activate window1 explicitly (editing in window2 may have activated it)
        window1
            .update(cx, |_, window, _| window.activate_window())
            .unwrap();
        cx.run_until_parked();

        // Verify window2 is not active (window1 should still be active)
        assert_eq!(
            cx.update(|cx| window2.is_active(cx)),
            Some(false),
            "Case 3: window2 should not be active before quit"
        );

        // Dispatch quit from window1 (window2 has the dirty item)
        cx.dispatch_action(*window1, Quit);
        cx.run_until_parked();

        // Verify window2 is now active (quit handler activated it to show dirty item)
        assert_eq!(
            cx.update(|cx| window2.is_active(cx)),
            Some(true),
            "Case 3: window2 should be activated when it has dirty item"
        );

        assert!(
            cx.has_pending_prompt(),
            "Case 3: Should prompt to save dirty item in non-active window"
        );

        cx.simulate_prompt_answer("Cancel");
        cx.run_until_parked();

        assert_eq!(
            cx.windows().len(),
            2,
            "Case 3: Windows should still exist after cancelling quit"
        );
    }

    #[gpui::test]
    async fn test_multi_workspace_session_restore(cx: &mut TestAppContext) {
        use collections::HashMap;
        use session::Session;
        use util::path_list::PathList;
        use workspace::{OpenMode, ProjectGroupKey, Workspace, WorkspaceId};

        let app_state = init_test(cx);

        let dir1 = path!("/dir1");
        let dir2 = path!("/dir2");
        let dir3 = path!("/dir3");

        let fs = app_state.fs.clone();
        let fake_fs = fs.as_fake();
        fake_fs.insert_tree(dir1, json!({})).await;
        fake_fs.insert_tree(dir2, json!({})).await;
        fake_fs.insert_tree(dir3, json!({})).await;

        let session_id = cx.read(|cx| app_state.session.read(cx).id().to_owned());

        // --- Create 3 workspaces in 2 windows ---
        //
        //   Window A: workspace for dir1, workspace for dir2
        //   Window B: workspace for dir3
        let workspace::OpenResult {
            window: window_a, ..
        } = cx
            .update(|cx| {
                Workspace::new_local(
                    vec![dir1.into()],
                    app_state.clone(),
                    None,
                    None,
                    None,
                    OpenMode::Activate,
                    cx,
                )
            })
            .await
            .expect("failed to open first workspace");

        window_a
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.open_sidebar(cx);
            })
            .unwrap();

        window_a
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.open_project(vec![dir2.into()], OpenMode::Activate, window, cx)
            })
            .unwrap()
            .await
            .expect("failed to open second workspace into window A");
        cx.run_until_parked();

        let workspace::OpenResult {
            window: window_b, ..
        } = cx
            .update(|cx| {
                Workspace::new_local(
                    vec![dir3.into()],
                    app_state.clone(),
                    None,
                    None,
                    None,
                    OpenMode::Activate,
                    cx,
                )
            })
            .await
            .expect("failed to open third workspace");

        window_b
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.open_sidebar(cx);
            })
            .unwrap();

        // Currently dir2 is active because it was added last.
        // So, switch window_a's active workspace to dir1 (index 0).
        // This sets up a non-trivial assertion: after restore, dir1 should
        // still be active rather than whichever workspace happened to restore last.
        window_a
            .update(cx, |multi_workspace, window, cx| {
                let workspace = multi_workspace.workspaces().next().unwrap().clone();
                multi_workspace.activate(workspace, None, window, cx);
            })
            .unwrap();

        cx.run_until_parked();
        flush_workspace_serialization(&window_a, cx).await;
        flush_workspace_serialization(&window_b, cx).await;
        cx.run_until_parked();

        // Verify all workspaces retained their session_ids.
        let db = cx.update(|cx| workspace::WorkspaceDb::global(cx));
        let locations =
            workspace::last_session_workspace_locations(&db, &session_id, None, fs.as_ref())
                .await
                .expect("expected session workspace locations");
        assert_eq!(
            locations.len(),
            3,
            "all 3 workspaces should have session_ids in the DB"
        );

        // Close the original windows.
        window_a
            .update(cx, |_, window, _| window.remove_window())
            .unwrap();
        window_b
            .update(cx, |_, window, _| window.remove_window())
            .unwrap();
        cx.run_until_parked();

        // Simulate a new session launch: replace the session so that
        // `last_session_id()` returns the ID used during workspace creation.
        // `restore_on_startup` defaults to `LastSession`, which is what we need.
        cx.update(|cx| {
            app_state.session.update(cx, |app_session, _cx| {
                app_session
                    .replace_session_for_test(Session::test_with_old_session(session_id.clone()));
            });
        });

        // --- Read back from DB and verify grouping ---
        let locations =
            workspace::last_session_workspace_locations(&db, &session_id, None, fs.as_ref())
                .await
                .expect("expected session workspace locations");

        assert_eq!(locations.len(), 3, "expected 3 session workspaces");

        let mut groups_by_window: HashMap<gpui::WindowId, Vec<WorkspaceId>> = HashMap::default();
        for session_workspace in &locations {
            if let Some(window_id) = session_workspace.window_id {
                groups_by_window
                    .entry(window_id)
                    .or_default()
                    .push(session_workspace.workspace_id);
            }
        }
        assert_eq!(
            groups_by_window.len(),
            2,
            "expected 2 window groups, got {groups_by_window:?}"
        );
        assert!(
            groups_by_window.values().any(|g| g.len() == 2),
            "expected one group with 2 workspaces"
        );
        assert!(
            groups_by_window.values().any(|g| g.len() == 1),
            "expected one group with 1 workspace"
        );

        let mut async_cx = cx.to_async();
        crate::restore_or_create_workspace(app_state.clone(), &mut async_cx)
            .await
            .expect("failed to restore workspaces");
        cx.run_until_parked();

        // --- Verify the restored windows ---
        let restored_windows: Vec<WindowHandle<MultiWorkspace>> = cx.read(|cx| {
            cx.windows()
                .into_iter()
                .filter_map(|window| window.downcast::<MultiWorkspace>())
                .collect()
        });
        assert_eq!(restored_windows.len(), 2,);

        // Identify restored windows by their active workspace root paths.
        let (restored_a, restored_b) = {
            let (mut with_dir1, mut with_dir3) = (None, None);
            for window in &restored_windows {
                let active_paths = window
                    .read_with(cx, |mw, cx| mw.workspace().read(cx).root_paths(cx))
                    .unwrap();
                if active_paths.iter().any(|p| p.as_ref() == Path::new(dir1)) {
                    with_dir1 = Some(window);
                } else {
                    with_dir3 = Some(window);
                }
            }
            (
                with_dir1.expect("expected a window with dir1 active"),
                with_dir3.expect("expected a window with dir3 active"),
            )
        };

        // Window A (dir1+dir2): 1 workspace restored, but 2 project group keys.
        restored_a
            .read_with(cx, |mw, _| {
                assert_eq!(
                    mw.project_group_keys(),
                    vec![
                        ProjectGroupKey::new(None, PathList::new(&[dir2])),
                        ProjectGroupKey::new(None, PathList::new(&[dir1])),
                    ]
                );
                assert_eq!(mw.workspaces().count(), 1);
            })
            .unwrap();

        // Window B (dir3): 1 workspace, 1 project group key.
        restored_b
            .read_with(cx, |mw, _| {
                assert_eq!(
                    mw.project_group_keys(),
                    vec![ProjectGroupKey::new(None, PathList::new(&[dir3]))]
                );
                assert_eq!(mw.workspaces().count(), 1);
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_quit_preserves_focused_workspace_for_restore(cx: &mut TestAppContext) {
        use session::Session;
        use workspace::{OpenMode, Workspace};

        let app_state = init_test(cx);
        cx.update(init);

        let dir1 = path!("/dir1");
        let dir2 = path!("/dir2");

        let fs = app_state.fs.clone();
        let fake_fs = fs.as_fake();
        fake_fs.insert_tree(dir1, json!({})).await;
        fake_fs.insert_tree(dir2, json!({})).await;

        let session_id = cx.read(|cx| app_state.session.read(cx).id().to_owned());

        // Window with two retained workspaces: dir1 added first, dir2 second.
        let workspace::OpenResult { window, .. } = cx
            .update(|cx| {
                Workspace::new_local(
                    vec![dir1.into()],
                    app_state.clone(),
                    None,
                    None,
                    None,
                    OpenMode::Activate,
                    cx,
                )
            })
            .await
            .expect("failed to open first workspace");

        window
            .update(cx, |multi_workspace, _, cx| {
                multi_workspace.open_sidebar(cx);
            })
            .unwrap();

        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.open_project(vec![dir2.into()], OpenMode::Activate, window, cx)
            })
            .unwrap()
            .await
            .expect("failed to open second workspace");
        cx.run_until_parked();

        // Focus dir1 (the first workspace). dir2 was activated last when it was
        // opened and is iterated last by the quit-time close-prompt loop, so
        // without the fix the persisted active workspace gets clobbered to dir2.
        window
            .update(cx, |multi_workspace, window, cx| {
                let workspace = multi_workspace.workspaces().next().unwrap().clone();
                multi_workspace.activate(workspace, None, window, cx);
            })
            .unwrap();
        cx.run_until_parked();

        window
            .read_with(cx, |mw, cx| {
                assert!(
                    mw.workspace()
                        .read(cx)
                        .root_paths(cx)
                        .iter()
                        .any(|p| p.as_ref() == Path::new(dir1)),
                    "dir1 should be the focused workspace before quitting"
                );
            })
            .unwrap();

        // Quit. With no dirty items there are no save prompts, so the quit flow
        // runs the prepare_to_close loop (which activates every workspace in
        // turn to surface prompts) and then flushes serialization. cx.quit() is
        // a no-op in tests, so the window stays around for inspection.
        cx.dispatch_action(*window, Quit);
        cx.run_until_parked();

        // The fix re-activates the originally-focused workspace after the loop,
        // so the window must still be focused on dir1, not dir2.
        window
            .read_with(cx, |mw, cx| {
                let active = mw.workspace().read(cx).root_paths(cx);
                assert!(
                    active.iter().any(|p| p.as_ref() == Path::new(dir1)),
                    "quitting must not change which workspace is focused"
                );
                assert!(
                    !active.iter().any(|p| p.as_ref() == Path::new(dir2)),
                    "dir2 must not become the focused workspace after quitting"
                );
            })
            .unwrap();

        // Simulate a fresh launch and verify dir1 is restored as the active
        // workspace rather than dir2 (or an empty window).
        window
            .update(cx, |_, window, _| window.remove_window())
            .unwrap();
        cx.run_until_parked();

        cx.update(|cx| {
            app_state.session.update(cx, |app_session, _cx| {
                app_session
                    .replace_session_for_test(Session::test_with_old_session(session_id.clone()));
            });
        });

        let mut async_cx = cx.to_async();
        crate::restore_or_create_workspace(app_state.clone(), &mut async_cx)
            .await
            .expect("failed to restore workspaces");
        cx.run_until_parked();

        let restored_windows: Vec<WindowHandle<MultiWorkspace>> = cx.read(|cx| {
            cx.windows()
                .into_iter()
                .filter_map(|window| window.downcast::<MultiWorkspace>())
                .collect()
        });
        assert_eq!(restored_windows.len(), 1);

        restored_windows[0]
            .read_with(cx, |mw, cx| {
                let active = mw.workspace().read(cx).root_paths(cx);
                assert!(
                    active.iter().any(|p| p.as_ref() == Path::new(dir1)),
                    "the focused workspace (dir1) must be restored as active"
                );
                assert!(
                    !active.iter().any(|p| p.as_ref() == Path::new(dir2)),
                    "dir2 must not be restored as the active workspace"
                );
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_restored_project_groups_survive_workspace_key_change(cx: &mut TestAppContext) {
        use session::Session;
        use util::path_list::PathList;
        use workspace::{OpenMode, ProjectGroupKey};

        let app_state = init_test(cx);

        let fs = app_state.fs.clone();
        let fake_fs = fs.as_fake();
        fake_fs
            .insert_tree(path!("/root_a"), json!({ "file.txt": "" }))
            .await;
        fake_fs
            .insert_tree(path!("/root_b"), json!({ "file.txt": "" }))
            .await;
        fake_fs
            .insert_tree(path!("/root_c"), json!({ "file.txt": "" }))
            .await;
        fake_fs
            .insert_tree(path!("/root_d"), json!({ "other.txt": "" }))
            .await;

        let session_id = cx.read(|cx| app_state.session.read(cx).id().to_owned());

        // --- Phase 1: Build a multi-workspace with 3 project groups ---

        let workspace::OpenResult { window, .. } = cx
            .update(|cx| {
                workspace::Workspace::new_local(
                    vec![path!("/root_a").into()],
                    app_state.clone(),
                    None,
                    None,
                    None,
                    OpenMode::Activate,
                    cx,
                )
            })
            .await
            .expect("failed to open workspace");

        window.update(cx, |mw, _, cx| mw.open_sidebar(cx)).unwrap();

        window
            .update(cx, |mw, window, cx| {
                mw.open_project(vec![path!("/root_b").into()], OpenMode::Add, window, cx)
            })
            .unwrap()
            .await
            .expect("failed to add root_b");

        window
            .update(cx, |mw, window, cx| {
                mw.open_project(vec![path!("/root_c").into()], OpenMode::Add, window, cx)
            })
            .unwrap()
            .await
            .expect("failed to add root_c");
        cx.run_until_parked();

        let key_b = ProjectGroupKey::new(None, PathList::new(&[path!("/root_b")]));
        let key_c = ProjectGroupKey::new(None, PathList::new(&[path!("/root_c")]));

        // Make root_a the active workspace so it's the one eagerly restored.
        window
            .update(cx, |mw, window, cx| {
                let workspace_a = mw
                    .workspaces()
                    .find(|ws| {
                        ws.read(cx)
                            .root_paths(cx)
                            .iter()
                            .any(|p| p.as_ref() == Path::new(path!("/root_a")))
                    })
                    .expect("workspace_a should exist")
                    .clone();
                mw.activate(workspace_a, None, window, cx);
            })
            .unwrap();
        cx.run_until_parked();

        // --- Phase 2: Serialize, close, and restore ---

        flush_workspace_serialization(&window, cx).await;
        cx.run_until_parked();

        window
            .update(cx, |_, window, _| window.remove_window())
            .unwrap();
        cx.run_until_parked();

        cx.update(|cx| {
            app_state.session.update(cx, |app_session, _cx| {
                app_session
                    .replace_session_for_test(Session::test_with_old_session(session_id.clone()));
            });
        });

        let mut async_cx = cx.to_async();
        crate::restore_or_create_workspace(app_state.clone(), &mut async_cx)
            .await
            .expect("failed to restore workspace");
        cx.run_until_parked();

        let restored_windows: Vec<WindowHandle<MultiWorkspace>> = cx.read(|cx| {
            cx.windows()
                .into_iter()
                .filter_map(|w| w.downcast::<MultiWorkspace>())
                .collect()
        });
        assert_eq!(restored_windows.len(), 1);
        let restored = &restored_windows[0];

        // Verify the restored window has all 3 project groups.
        restored
            .read_with(cx, |mw, _cx| {
                let keys = mw.project_group_keys();
                assert_eq!(
                    keys.len(),
                    3,
                    "restored window should have 3 groups; got {keys:?}"
                );
                assert!(keys.contains(&key_b), "should contain key_b");
                assert!(keys.contains(&key_c), "should contain key_c");
            })
            .unwrap();

        // --- Phase 3: Trigger a workspace key change and verify survival ---

        let active_project = restored
            .read_with(cx, |mw, cx| mw.workspace().read(cx).project().clone())
            .unwrap();

        active_project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/root_d"), true, cx)
            })
            .await
            .expect("adding worktree should succeed");
        cx.run_until_parked();

        restored
            .read_with(cx, |mw, _cx| {
                let keys = mw.project_group_keys();
                assert!(
                    keys.contains(&key_b),
                    "restored group key_b should survive a workspace key change; got {keys:?}"
                );
                assert!(
                    keys.contains(&key_c),
                    "restored group key_c should survive a workspace key change; got {keys:?}"
                );
            })
            .unwrap();
    }

    #[gpui::test]
    async fn test_close_project_removes_project_group(cx: &mut TestAppContext) {
        use util::path_list::PathList;
        use workspace::{OpenMode, ProjectGroupKey};

        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/my-project"), json!({}))
            .await;

        let workspace::OpenResult { window, .. } = cx
            .update(|cx| {
                workspace::Workspace::new_local(
                    vec![path!("/my-project").into()],
                    app_state.clone(),
                    None,
                    None,
                    None,
                    OpenMode::Activate,
                    cx,
                )
            })
            .await
            .unwrap();

        window.update(cx, |mw, _, cx| mw.open_sidebar(cx)).unwrap();
        cx.background_executor.run_until_parked();

        let project_key = ProjectGroupKey::new(None, PathList::new(&[path!("/my-project")]));
        let keys = window
            .read_with(cx, |mw, _| mw.project_group_keys())
            .unwrap();
        assert_eq!(
            keys,
            vec![project_key],
            "project group should exist before CloseProject: {keys:?}"
        );

        cx.dispatch_action(window.into(), CloseProject);

        let keys = window
            .read_with(cx, |mw, _| mw.project_group_keys())
            .unwrap();
        assert!(
            keys.is_empty(),
            "project group should be removed after CloseProject: {keys:?}"
        );
    }

    #[gpui::test]
    async fn test_close_project_switches_to_neighbor_in_multi_project(cx: &mut TestAppContext) {
        use workspace::OpenMode;

        let app_state = init_test(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/project-a"), json!({}))
            .await;
        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/project-b"), json!({}))
            .await;

        let workspace::OpenResult {
            window,
            workspace: workspace_a,
            ..
        } = cx
            .update(|cx| {
                workspace::Workspace::new_local(
                    vec![path!("/project-a").into()],
                    app_state.clone(),
                    None,
                    None,
                    None,
                    OpenMode::Activate,
                    cx,
                )
            })
            .await
            .unwrap();

        let project_b = Project::test(app_state.fs.clone(), [Path::new("/project-b")], cx).await;

        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.test_add_workspace(project_b, window, cx);
            })
            .unwrap();
        cx.background_executor.run_until_parked();

        // Reactivate workspace A so we close it via CloseProject.
        window
            .update(cx, |multi_workspace, window, cx| {
                multi_workspace.activate(workspace_a, None, window, cx);
            })
            .unwrap();
        cx.background_executor.run_until_parked();

        let keys_before = window
            .read_with(cx, |multi_workspace, _| {
                multi_workspace.project_group_keys()
            })
            .unwrap();
        assert_eq!(
            keys_before.len(),
            2,
            "should have 2 project groups before CloseProject: {keys_before:?}"
        );

        cx.dispatch_action(window.into(), CloseProject);
        cx.background_executor.run_until_parked();

        let keys_after = window
            .read_with(cx, |multi_workspace, _| {
                multi_workspace.project_group_keys()
            })
            .unwrap();
        assert_eq!(
            keys_after.len(),
            1,
            "one project group should remain after CloseProject: {keys_after:?}"
        );

        let active_paths = window
            .read_with(cx, |multi_workspace, cx| {
                multi_workspace.workspace().read(cx).root_paths(cx)
            })
            .unwrap();
        assert!(
            !active_paths.is_empty(),
            "active workspace should contain the remaining project, not be empty: {active_paths:?}"
        );
    }
}
