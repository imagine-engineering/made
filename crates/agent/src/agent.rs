pub mod buzz_tool_compat;
#[cfg(feature = "multiplayer-tools")]
pub mod collaboration_mention;
pub mod collaboration_session;
mod db;
#[cfg(feature = "multiplayer-tools")]
pub mod job_delegation;
#[cfg(feature = "multiplayer-tools")]
pub mod jobs;
mod legacy_thread;
#[cfg(feature = "multiplayer-tools")]
pub mod managed_agents;
#[cfg(feature = "multiplayer-tools")]
pub mod memory;
mod native_agent_server;
pub mod outline;
mod pattern_extraction;
pub mod recipe_import;
pub mod recipe_scan;
pub mod recipe_schedule;
pub mod recipe_workflow;
pub mod recipes;
#[cfg(feature = "multiplayer-tools")]
pub mod remote_execution;
pub mod remote_provider_config;
mod sandboxing;
#[cfg(feature = "multiplayer-tools")]
pub mod snapshot;
mod templates;
#[cfg(test)]
mod tests;
mod thread;
mod thread_store;
mod tool_permissions;
mod tools;
#[cfg(feature = "multiplayer-tools")]
pub mod usage;

use context_server::ContextServerId;
pub use db::*;
use itertools::Itertools;
#[cfg(feature = "multiplayer-tools")]
pub use managed_agents::*;
#[cfg(feature = "multiplayer-tools")]
pub use memory::*;
pub use native_agent_server::NativeAgentServer;
pub use pattern_extraction::*;
pub use sandboxing::{
    ThreadSandbox, sandbox_worktree_writable_paths, settings_sandbox_policy,
    settings_thread_sandbox,
};
pub use shell_command_parser::extract_commands;
#[cfg(feature = "multiplayer-tools")]
pub use snapshot::*;
pub use templates::*;
pub use thread::*;
pub use thread_store::*;
pub use tool_permissions::*;
pub use tools::*;
#[cfg(feature = "multiplayer-tools")]
pub use usage::*;

use acp_thread::{
    AcpThread, AcpThreadEvent, AgentModelId, AgentModelSelector, AgentSessionInfo,
    AgentSessionList, AgentSessionListRequest, AgentSessionListResponse, ClientUserMessageId,
    TokenUsageRatio,
};
use agent_client_protocol::schema::v1 as acp;
use agent_skills::{
    AGENTS_DIR_NAME, MAX_SKILL_DESCRIPTIONS_SIZE, MAX_SKILL_FILE_SIZE, ProjectSkillGroup,
    SKILL_FILE_NAME, Skill, SkillIndex, SkillLoadError, SkillLoadWarning, SkillScopeId,
    SkillSource, SkillSummary, builtin_skills, global_skills_dir, load_skills_from_directory,
    parse_skill_frontmatter, project_skills_relative_path, read_skill_body_from_content,
};
use anyhow::{Context as _, Result, anyhow};
use chrono::{DateTime, Utc};
use collections::{HashMap, HashSet, IndexMap};

use fs::Fs;
use futures::channel::{mpsc, oneshot};
use futures::future::Shared;
use futures::{FutureExt as _, StreamExt as _, future};
use gpui::{
    App, AppContext, AsyncApp, Context, Entity, EntityId, SharedString, Subscription, Task,
    TaskExt, WeakEntity,
};
use language_model::{
    IconOrSvg, LanguageModel, LanguageModelId, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelRegistry,
};
use project::{
    AgentId, Project, ProjectItem, ProjectPath, Worktree, WorktreeId,
    trusted_worktrees::TrustedWorktrees,
};
use prompt_store::{ProjectContext, RULES_FILE_NAMES, RulesFileContext, WorktreeContext};
use serde::{Deserialize, Serialize};
use settings::{LanguageModelSelection, Settings as _, update_settings_file};
use std::any::Any;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, LazyLock};
use util::ResultExt;
use util::path_list::PathList;
use util::rel_path::RelPath;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectSnapshot {
    pub worktree_snapshots: Vec<project::telemetry_snapshot::TelemetryWorktreeSnapshot>,
    pub timestamp: DateTime<Utc>,
}

pub struct RulesLoadingError {
    pub path: PathBuf,
    pub worktree: SharedString,
    pub relative_path: PathBuf,
    pub message: SharedString,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkillLoadingIssueKind {
    LoadFailed,
    DescriptionTooLong,
    CatalogBudgetExceeded,
    ProjectInstructionLoadFailed,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SkillLoadingIssue {
    pub project_id: EntityId,
    pub path: PathBuf,
    pub source_label: Option<SharedString>,
    pub relative_path: Option<PathBuf>,
    pub message: SharedString,
    pub kind: SkillLoadingIssueKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SkillLoadingIssueData {
    path: PathBuf,
    source_label: Option<SharedString>,
    relative_path: Option<PathBuf>,
    message: String,
    kind: SkillLoadingIssueKind,
}

impl SkillLoadingIssueData {
    fn from_load_error(error: SkillLoadError) -> Self {
        Self {
            path: error.path,
            source_label: None,
            relative_path: None,
            message: error.message,
            kind: SkillLoadingIssueKind::LoadFailed,
        }
    }

    fn from_load_warning(skill: &Skill, warning: &SkillLoadWarning) -> Self {
        let kind = match warning {
            SkillLoadWarning::DescriptionTooLong { .. } => {
                SkillLoadingIssueKind::DescriptionTooLong
            }
        };
        Self {
            path: skill.skill_file_path.clone(),
            source_label: None,
            relative_path: None,
            message: warning.message(),
            kind,
        }
    }

    fn catalog_budget_exceeded(path: PathBuf, message: String) -> Self {
        Self {
            path,
            source_label: None,
            relative_path: None,
            message,
            kind: SkillLoadingIssueKind::CatalogBudgetExceeded,
        }
    }

    fn from_rules_error(error: RulesLoadingError) -> Self {
        Self {
            path: error.path,
            source_label: Some(error.worktree),
            relative_path: Some(error.relative_path),
            message: error.message.to_string(),
            kind: SkillLoadingIssueKind::ProjectInstructionLoadFailed,
        }
    }
}

/// Emitted whenever the set of skill loading issues for a project changes.
/// The `issues` field is the full replacement list; subscribers should treat
/// it as a snapshot rather than appending. An empty `issues` list means all
/// previously-reported issues have been resolved.
#[derive(Clone, Debug)]
pub struct SkillLoadingIssuesUpdated {
    pub project_id: EntityId,
    pub issues: Vec<SkillLoadingIssue>,
}

#[derive(Clone)]
pub struct DoctorOpenRequest {
    pub project: Entity<Project>,
    pub session_id: acp::SessionId,
    completion: Rc<std::cell::RefCell<Option<futures::channel::oneshot::Sender<Result<()>>>>>,
}

impl DoctorOpenRequest {
    pub fn complete(&self, result: Result<()>) {
        if let Some(completion) = self.completion.borrow_mut().take()
            && completion.send(result).is_err()
        {
            log::debug!("Doctor command caller closed before desktop acknowledgment");
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeAvailableSkill {
    pub name: String,
    pub description: String,
    pub source: SharedString,
    pub skill_file_path: PathBuf,
    pub warning: Option<SharedString>,
}

impl From<&Skill> for NativeAvailableSkill {
    fn from(skill: &Skill) -> Self {
        Self {
            name: skill.name.clone(),
            description: skill.description.clone(),
            source: skill.source.display_label().to_string().into(),
            skill_file_path: skill.skill_file_path.clone(),
            warning: skill
                .load_warnings
                .first()
                .map(|warning| warning.message().into()),
        }
    }
}

pub const COMPACT_COMMAND_NAME: &str = "compact";
pub const CLEAR_COMMAND_NAME: &str = "clear";
pub const SKILLS_COMMAND_NAME: &str = "skills";
pub const STATUS_COMMAND_NAME: &str = "status";
pub const GOAL_COMMAND_NAME: &str = "goal";
pub const GRIND_COMMAND_NAME: &str = "grind";

pub const DEFAULT_GRIND_MAX_TURNS: u8 = 5;
pub const HARD_MAX_GRIND_TURNS: u8 = 20;

struct NativeCommandDefinition {
    name: &'static str,
    description: &'static str,
    input_hint: Option<&'static str>,
}

const NATIVE_COMMANDS: &[NativeCommandDefinition] = &[
    NativeCommandDefinition {
        name: "doctor",
        description: "Open Made Doctor for scoped diagnosis and reviewed repairs",
        input_hint: None,
    },
    NativeCommandDefinition {
        name: "schedules",
        description: "Manage Made scheduled recipes",
        input_hint: Some(
            "list | create | edit/pause/resume/run/cancel/delete/history/open <id> [--consent]",
        ),
    },
    NativeCommandDefinition {
        name: "recipes",
        description: "List available Made recipes and their commands",
        input_hint: Some("[list]"),
    },
    NativeCommandDefinition {
        name: COMPACT_COMMAND_NAME,
        description: "Summarize the conversation so far to free up context",
        input_hint: None,
    },
    NativeCommandDefinition {
        name: CLEAR_COMMAND_NAME,
        description: "Clear the current conversation",
        input_hint: None,
    },
    NativeCommandDefinition {
        name: SKILLS_COMMAND_NAME,
        description: "List the skills available in this project",
        input_hint: None,
    },
    NativeCommandDefinition {
        name: STATUS_COMMAND_NAME,
        description: "Show the current model and developer context status",
        input_hint: None,
    },
    NativeCommandDefinition {
        name: GOAL_COMMAND_NAME,
        description: "Set, show, or clear the current session goal",
        input_hint: Some("[<goal text> | clear | off | none]"),
    },
    NativeCommandDefinition {
        name: GRIND_COMMAND_NAME,
        description: "Continue automatically toward the current goal within a turn limit",
        input_hint: Some("[max_turns=<1-20>]"),
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GoalCommandArgument<'a> {
    Show,
    Clear,
    Set(&'a str),
}

pub(crate) fn parse_goal_command_argument(argument: &str) -> GoalCommandArgument<'_> {
    let argument = argument.trim();
    if argument.is_empty() {
        GoalCommandArgument::Show
    } else if matches!(argument, "clear" | "off" | "none") {
        GoalCommandArgument::Clear
    } else {
        GoalCommandArgument::Set(argument)
    }
}

pub(crate) fn parse_grind_max_turns(argument: &str) -> Result<u8> {
    let argument = argument.trim();
    if argument.is_empty() {
        return Ok(DEFAULT_GRIND_MAX_TURNS);
    }

    let Some(value) = argument.strip_prefix("max_turns=") else {
        anyhow::bail!("Usage: /grind [max_turns=<1-20>]");
    };
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        anyhow::bail!("Usage: /grind [max_turns=<1-20>]");
    }
    let max_turns = value
        .parse::<u8>()
        .map_err(|_| anyhow!("`max_turns` must be an integer from 1 through 20"))?;
    if !(1..=HARD_MAX_GRIND_TURNS).contains(&max_turns) {
        anyhow::bail!("`max_turns` must be an integer from 1 through 20");
    }
    Ok(max_turns)
}

/// Returns the set of MCP prompt names that must be server-qualified
/// (`/<server>.<name>`) to stay unambiguous in the slash-command popup: names
/// shared by more than one MCP prompt, or names colliding with a reserved
/// built-in command (e.g. `/compact`). A built-in always wins an unqualified
/// invocation, so colliding MCP prompts are only reachable when prefixed.
fn ambiguous_mcp_prompt_names<'a>(
    reserved: impl IntoIterator<Item = &'a str>,
    prompt_names: impl IntoIterator<Item = &'a str>,
) -> HashSet<&'a str> {
    let mut counts: HashMap<&str, usize> = HashMap::default();
    for name in reserved.into_iter().chain(prompt_names) {
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect()
}

fn mcp_prompt_input_hint(
    arguments: Option<&[context_server::types::PromptArgument]>,
) -> Option<String> {
    let arguments = arguments.filter(|arguments| !arguments.is_empty())?;
    let syntax = arguments
        .iter()
        .map(|argument| {
            if argument.required == Some(true) {
                format!("<{}=value>", argument.name)
            } else {
                format!("[{}=value]", argument.name)
            }
        })
        .join(" ");
    let details = arguments
        .iter()
        .map(|argument| {
            let requirement = if argument.required == Some(true) {
                "required"
            } else {
                "optional"
            };
            match argument.description.as_deref() {
                Some(description) if !description.is_empty() => {
                    format!("{} ({requirement}): {description}", argument.name)
                }
                Some(_) | None => format!("{} ({requirement})", argument.name),
            }
        })
        .join("; ");
    Some(format!("{syntax} — {details}"))
}

fn split_shell_words(input: &str) -> Result<Vec<String>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Quote {
        Single,
        Double,
    }

    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut word_started = false;

    for character in input.chars() {
        if escaped {
            word.push(character);
            escaped = false;
            word_started = true;
            continue;
        }

        match (quote, character) {
            (Some(Quote::Single), '\'') => quote = None,
            (Some(Quote::Single), _) => word.push(character),
            (Some(Quote::Double), '"') => quote = None,
            (Some(Quote::Double), '\\') => escaped = true,
            (Some(Quote::Double), _) => word.push(character),
            (None, '\'') => {
                quote = Some(Quote::Single);
                word_started = true;
            }
            (None, '"') => {
                quote = Some(Quote::Double);
                word_started = true;
            }
            (None, '\\') => {
                escaped = true;
                word_started = true;
            }
            (None, character) if character.is_whitespace() => {
                if word_started {
                    words.push(std::mem::take(&mut word));
                    word_started = false;
                }
            }
            (None, _) => {
                word.push(character);
                word_started = true;
            }
        }
    }

    if escaped {
        anyhow::bail!("MCP prompt arguments end with an incomplete escape");
    }
    if quote.is_some() {
        anyhow::bail!("MCP prompt arguments contain an unterminated quote");
    }
    if word_started {
        words.push(word);
    }
    Ok(words)
}

fn parse_mcp_prompt_arguments(
    arguments: Option<&[context_server::types::PromptArgument]>,
    input: &str,
) -> Result<HashMap<String, String>> {
    let arguments = arguments.unwrap_or_default();
    let input = input.trim();
    if arguments.is_empty() {
        if input.is_empty() {
            return Ok(HashMap::default());
        }
        anyhow::bail!("This MCP prompt does not accept arguments");
    }

    if arguments.len() == 1 && !input.is_empty() {
        let words = split_shell_words(input)?;
        let is_named_input = words.first().is_some_and(|word| word.contains('='));
        if !is_named_input {
            return Ok(HashMap::from_iter([(
                arguments[0].name.clone(),
                input.to_string(),
            )]));
        }
    }

    let mut parsed = HashMap::default();
    for word in split_shell_words(input)? {
        let Some((name, value)) = word.split_once('=') else {
            anyhow::bail!("Malformed MCP prompt argument `{word}`; expected name=value");
        };
        if name.is_empty() {
            anyhow::bail!("MCP prompt argument names cannot be empty");
        }
        if !arguments.iter().any(|argument| argument.name == name) {
            anyhow::bail!("Unknown MCP prompt argument `{name}`");
        }
        if parsed.insert(name.to_string(), value.to_string()).is_some() {
            anyhow::bail!("Duplicate MCP prompt argument `{name}`");
        }
    }

    let missing_required = arguments
        .iter()
        .filter(|argument| argument.required == Some(true))
        .filter(|argument| !parsed.contains_key(&argument.name))
        .map(|argument| argument.name.as_str())
        .collect::<Vec<_>>();
    if !missing_required.is_empty() {
        anyhow::bail!(
            "Missing required MCP prompt argument{}: {}",
            if missing_required.len() == 1 { "" } else { "s" },
            missing_required.join(", ")
        );
    }

    Ok(parsed)
}

struct ProjectState {
    project: Entity<Project>,
    project_context: Entity<ProjectContext>,
    skills: Arc<Vec<Skill>>,
    recipes: Vec<recipes::RecipeEntry>,
    skill_loading_issues: Vec<SkillLoadingIssue>,
    project_context_needs_refresh: watch::Sender<()>,
    _maintain_project_context: Task<Result<()>>,
    context_server_registry: Entity<ContextServerRegistry>,
    _subscriptions: Vec<Subscription>,
}

/// Holds both the internal Thread and the AcpThread for a session
struct Session {
    /// The internal thread that processes messages
    thread: Entity<Thread>,
    /// The ACP thread that handles protocol communication
    acp_thread: Entity<acp_thread::AcpThread>,
    project_id: EntityId,
    pending_save: Task<Result<()>>,
    _subscriptions: Vec<Subscription>,
    ref_count: usize,
    active_grind: Option<ActiveGrind>,
}

struct ActiveGrind {
    invocation_id: Uuid,
    max_turns: u8,
    completed_turns: u8,
    stop_requested: Option<GrindStopReason>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrindStopReason {
    Cancelled,
    ToolFailure,
    UserAttention,
    SessionClosed,
}

fn grind_stop_output(
    stop_reason: GrindStopReason,
    completed_turns: u8,
    max_turns: u8,
) -> Option<String> {
    match stop_reason {
        GrindStopReason::Cancelled => Some(format!(
            "Grind cancelled after {completed_turns}/{max_turns} turns."
        )),
        GrindStopReason::ToolFailure => Some(format!(
            "Grind stopped after {completed_turns}/{max_turns} turns because a tool failed."
        )),
        GrindStopReason::UserAttention => Some(format!(
            "Grind stopped after {completed_turns}/{max_turns} turns because user attention is required."
        )),
        GrindStopReason::SessionClosed => None,
    }
}

struct PendingSession {
    task: Shared<Task<Result<Entity<AcpThread>, Arc<anyhow::Error>>>>,
    ref_count: usize,
}

pub struct LanguageModels {
    /// Access language model by ID
    models: HashMap<AgentModelId, Arc<dyn LanguageModel>>,
    /// Cached list for returning language model information
    model_list: acp_thread::AgentModelList,
    refresh_models_rx: watch::Receiver<()>,
    refresh_models_tx: watch::Sender<()>,
    _authenticate_all_providers_task: Task<()>,
}

impl LanguageModels {
    fn new(cx: &mut App) -> Self {
        let (refresh_models_tx, refresh_models_rx) = watch::channel(());

        let mut this = Self {
            models: HashMap::default(),
            model_list: acp_thread::AgentModelList::Grouped(IndexMap::default()),
            refresh_models_rx,
            refresh_models_tx,
            _authenticate_all_providers_task: Self::authenticate_all_language_model_providers(cx),
        };
        this.refresh_list(cx);
        this
    }

    fn refresh_list(&mut self, cx: &App) {
        let providers = LanguageModelRegistry::global(cx)
            .read(cx)
            .visible_providers()
            .into_iter()
            .filter(|provider| provider.is_authenticated(cx))
            .collect::<Vec<_>>();

        let mut language_model_list = IndexMap::default();
        let mut recommended_models = HashSet::default();

        let mut recommended = Vec::new();
        for provider in &providers {
            for model in provider.recommended_models(cx) {
                recommended_models.insert((model.provider_id(), model.id()));
                recommended.push(Self::map_language_model_to_info(&model, provider));
            }
        }
        if !recommended.is_empty() {
            language_model_list.insert(
                acp_thread::AgentModelGroupName("Recommended".into()),
                recommended,
            );
        }

        let mut models = HashMap::default();
        for provider in providers {
            let mut provider_models = Vec::new();
            for model in provider.provided_models(cx) {
                let model_info = Self::map_language_model_to_info(&model, &provider);
                let model_id = model_info.id.clone();
                provider_models.push(model_info);
                models.insert(model_id, model);
            }
            if !provider_models.is_empty() {
                language_model_list.insert(
                    acp_thread::AgentModelGroupName(provider.name().0.clone()),
                    provider_models,
                );
            }
        }

        self.models = models;
        self.model_list = acp_thread::AgentModelList::Grouped(language_model_list);
        self.refresh_models_tx.send(()).ok();
    }

    fn watch(&self) -> watch::Receiver<()> {
        self.refresh_models_rx.clone()
    }

    pub fn notify_model_selection_changed(&mut self) {
        self.refresh_models_tx.send(()).ok();
    }

    pub fn model_from_id(&self, model_id: &AgentModelId) -> Option<Arc<dyn LanguageModel>> {
        self.models.get(model_id).cloned()
    }

    fn map_language_model_to_info(
        model: &Arc<dyn LanguageModel>,
        provider: &Arc<dyn LanguageModelProvider>,
    ) -> acp_thread::AgentModelInfo {
        acp_thread::AgentModelInfo {
            id: Self::model_id(model),
            name: model.name().0,
            description: None,
            icon: Some(match provider.icon() {
                IconOrSvg::Svg(path) => acp_thread::AgentModelIcon::Path(path),
                IconOrSvg::Icon(name) => acp_thread::AgentModelIcon::Named(name),
            }),
            is_latest: model.is_latest(),
            cost: model.model_cost_info().map(|cost| cost.to_shared_string()),
            disabled: model.is_disabled(),
        }
    }

    fn model_id(model: &Arc<dyn LanguageModel>) -> AgentModelId {
        AgentModelId::new(format!("{}/{}", model.provider_id().0, model.id().0))
    }

    fn authenticate_all_language_model_providers(cx: &mut App) -> Task<()> {
        let authenticate_all_providers = LanguageModelRegistry::global(cx)
            .read(cx)
            .visible_providers()
            .iter()
            .map(|provider| (provider.id(), provider.name(), provider.authenticate(cx)))
            .collect::<Vec<_>>();

        cx.spawn(async move |cx| {
            for (provider_id, provider_name, authenticate_task) in authenticate_all_providers {
                if let Err(err) = authenticate_task.await {
                    match err {
                        language_model::AuthenticateError::CredentialsNotFound => {
                            // Since we're authenticating these providers in the
                            // background for the purposes of populating the
                            // language selector, we don't care about providers
                            // where the credentials are not found.
                        }
                        language_model::AuthenticateError::ConnectionRefused => {
                            // Not logging connection refused errors as they are mostly from LM Studio's noisy auth failures.
                            // LM Studio only has one auth method (endpoint call) which fails for users who haven't enabled it.
                            // TODO: Better manage LM Studio auth logic to avoid these noisy failures.
                        }
                        _ => {
                            // Some providers have noisy failure states that we
                            // don't want to spam the logs with every time the
                            // language model selector is initialized.
                            //
                            // Ideally these should have more clear failure modes
                            // that we know are safe to ignore here, like what we do
                            // with `CredentialsNotFound` above.
                            match provider_id.0.as_ref() {
                                "lmstudio" | "ollama" => {
                                    // LM Studio and Ollama both make fetch requests to the local APIs to determine if they are "authenticated".
                                    //
                                    // These fail noisily, so we don't log them.
                                }
                                "copilot_chat" => {
                                    // Copilot Chat returns an error if Copilot is not enabled, so we don't log those errors.
                                }
                                _ => {
                                    log::error!(
                                        "Failed to authenticate provider: {}: {err:#}",
                                        provider_name.0
                                    );
                                }
                            }
                        }
                    }
                }
            }

            cx.update(|cx| {
                LanguageModelRegistry::global(cx)
                    .update(cx, |registry, cx| registry.refresh_fallback_model(cx))
            });
        })
    }
}

/// Implemented by the UI layer to provide the ability for agent tools to create
/// sibling threads that appear in the agent panel.
///
/// `agent_ui::AgentPanel` installs an implementation of this trait on the
/// `NativeAgent` when it sets up a connection. Tools in a native-agent thread
/// then discover and use the host via `NativeThreadEnvironment`. The UI side
/// is responsible for keeping the installed host current; a host whose
/// backing UI has been torn down will fail its first request with a clear
/// error rather than being detected up front.
pub trait SiblingThreadHost {
    fn create_sibling_thread(
        &self,
        request: SiblingThreadRequest,
        cx: &mut AsyncApp,
    ) -> Task<Result<SiblingThreadInfo>>;

    fn list_available_agents(&self, cx: &mut App) -> Result<AvailableAgents>;
}

pub struct NativeAgent {
    /// Session ID -> Session mapping
    sessions: HashMap<acp::SessionId, Session>,
    pending_sessions: HashMap<acp::SessionId, PendingSession>,
    thread_store: Entity<ThreadStore>,
    /// Project-specific state keyed by project EntityId
    projects: HashMap<EntityId, ProjectState>,
    /// Shared templates for all threads
    templates: Arc<Templates>,
    /// Cached model information
    models: LanguageModels,
    /// Handler installed by the UI for `create_thread` / `list_agents_and_models` tools.
    sibling_thread_host: Option<Rc<dyn SiblingThreadHost>>,
    fs: Arc<dyn Fs>,
    _subscriptions: Vec<Subscription>,
    /// Tracks the lifecycle of global skills directory observation. We
    /// don't eagerly watch (or even check for) `~/.agents/skills/` at
    /// startup; users who never engage with the agent panel pay zero
    /// filesystem cost. The watch is kicked off lazily by
    /// [`Self::ensure_skills_scan_started`], which is called from the
    /// three agent-panel interaction points: input box focus, slash
    /// autocomplete, and conversation submit.
    skills_state: SkillsState,
}

#[derive(Default)]
enum SkillsState {
    /// No scan or watch is active. A user-interaction trigger will kick
    /// off a fresh scan.
    #[default]
    Idle,
    /// A one-shot scan task is in flight. It checks whether
    /// `~/.agents/skills/` exists; if so, transitions to `Watching`,
    /// otherwise back to `Idle`.
    Scanning,
    /// A watch task is observing `~/.agents/skills/`. It transitions
    /// back to `Idle` if the watched directory itself is removed.
    Watching,
}

impl gpui::EventEmitter<SkillLoadingIssuesUpdated> for NativeAgent {}
impl gpui::EventEmitter<DoctorOpenRequest> for NativeAgent {}

static RULES_FILE_REL_PATHS: LazyLock<Vec<Arc<RelPath>>> = LazyLock::new(|| {
    RULES_FILE_NAMES
        .iter()
        .filter_map(|name| {
            RelPath::from_unix_str(name)
                .ok()
                .map(|path| path.into_arc())
        })
        .collect()
});

static AGENTS_PREFIX: LazyLock<Option<Arc<RelPath>>> = LazyLock::new(|| {
    RelPath::from_unix_str(AGENTS_DIR_NAME)
        .ok()
        .map(|path| path.into_arc())
});

static SKILLS_PREFIX: LazyLock<Option<Arc<RelPath>>> = LazyLock::new(|| {
    RelPath::from_unix_str(project_skills_relative_path())
        .ok()
        .map(|path| path.into_arc())
});

struct ProjectSkillFile {
    relative_path: Arc<RelPath>,
    display_path: PathBuf,
    size: u64,
}

async fn expand_worktree_directory(
    worktree: &Entity<Worktree>,
    path: &RelPath,
    cx: &mut AsyncApp,
) -> Result<()> {
    let expand_task = worktree.update(cx, |worktree, cx| {
        let entry_id = worktree
            .entry_for_path(path)
            .filter(|entry| entry.is_dir())
            .map(|entry| entry.id);
        entry_id.and_then(|entry_id| worktree.expand_entry(entry_id, cx))
    });

    if let Some(expand_task) = expand_task {
        expand_task.await?;
    }

    Ok(())
}

async fn expand_project_skills_directories(
    worktree: &Entity<Worktree>,
    cx: &mut AsyncApp,
) -> Result<()> {
    let agents_dir = RelPath::from_unix_str(AGENTS_DIR_NAME)?;
    let Some(skills_prefix) = SKILLS_PREFIX.as_ref() else {
        return Ok(());
    };

    expand_worktree_directory(worktree, agents_dir, cx).await?;
    expand_worktree_directory(worktree, skills_prefix, cx).await?;

    let skill_dirs = worktree.update(cx, |worktree, _cx| {
        worktree
            .child_entries(skills_prefix)
            .filter(|entry| entry.is_dir())
            .map(|entry| entry.path.clone())
            .collect::<Vec<_>>()
    });
    for skill_dir in skill_dirs {
        expand_worktree_directory(worktree, &skill_dir, cx).await?;
    }

    Ok(())
}

fn project_skill_files_from_worktree(worktree: &Worktree) -> Vec<ProjectSkillFile> {
    let Some(skills_prefix) = SKILLS_PREFIX.as_ref() else {
        return Vec::new();
    };
    let Ok(skill_file_name) = RelPath::from_unix_str(SKILL_FILE_NAME) else {
        return Vec::new();
    };

    let mut skill_files = Vec::new();
    for skill_dir in worktree.child_entries(skills_prefix) {
        if !skill_dir.is_dir() {
            continue;
        }

        let relative_path = skill_dir.path.join(skill_file_name);
        let Some(skill_file) = worktree.entry_for_path(&relative_path) else {
            continue;
        };
        if !skill_file.is_file() {
            continue;
        }

        skill_files.push(ProjectSkillFile {
            display_path: worktree.absolutize(&relative_path),
            relative_path: relative_path.into(),
            size: skill_file.size,
        });
    }

    skill_files.sort_by(|a, b| {
        a.relative_path
            .as_unix_str()
            .cmp(b.relative_path.as_unix_str())
    });
    skill_files
}

impl NativeAgent {
    pub fn has_active_work(&self, cx: &App) -> bool {
        !self.pending_sessions.is_empty()
            || self.sessions.values().any(|session| {
                session.active_grind.is_some()
                    || session.acp_thread.read(cx).status() == acp_thread::ThreadStatus::Generating
                    || session.thread.read(cx).is_generating_summary()
                    || session.thread.read(cx).is_generating_title()
            })
    }

    pub fn recipe_entries_for_project(
        &self,
        project: &Entity<Project>,
    ) -> Option<&[recipes::RecipeEntry]> {
        self.projects
            .get(&project.entity_id())
            .map(|state| state.recipes.as_slice())
    }

    pub fn new(
        thread_store: Entity<ThreadStore>,
        templates: Arc<Templates>,
        fs: Arc<dyn Fs>,
        cx: &mut App,
    ) -> Entity<NativeAgent> {
        log::debug!("Creating new NativeAgent");

        cx.new(|cx| {
            let mut package_paths = cx
                .try_global::<SkillIndex>()
                .map(|index| {
                    index
                        .package_skills
                        .iter()
                        .map(|skill| skill.skill_file_path.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let subscriptions = vec![
                cx.observe_global::<SkillIndex>(move |this: &mut Self, cx| {
                    let current_paths = cx
                        .global::<SkillIndex>()
                        .package_skills
                        .iter()
                        .map(|skill| skill.skill_file_path.clone())
                        .collect::<Vec<_>>();
                    if current_paths != package_paths {
                        package_paths = current_paths;
                        for state in this.projects.values_mut() {
                            state.project_context_needs_refresh.send(()).log_err();
                        }
                    }
                }),
                cx.subscribe(
                    &LanguageModelRegistry::global(cx),
                    Self::handle_models_updated_event,
                ),
                // Flush thread content on quit so an in-flight async save
                // can't leave a thread orphaned ("no thread found with ID").
                cx.on_app_quit(Self::flush_threads_on_quit),
            ];

            if !cx.has_global::<SkillIndex>() {
                cx.set_global(SkillIndex::default());
            }

            Self {
                sessions: HashMap::default(),
                pending_sessions: HashMap::default(),
                thread_store,
                projects: HashMap::default(),
                templates,
                models: LanguageModels::new(cx),
                sibling_thread_host: None,
                fs,
                _subscriptions: subscriptions,
                skills_state: SkillsState::default(),
            }
        })
    }

    /// Kicks off a one-time scan of the global skills directory if one
    /// isn't already in progress and a watch isn't already active.
    ///
    /// Idempotent and cheap: returns immediately if a scan or watch is
    /// already running. The expected callers are user-interaction events
    /// from the agent panel (input focus, slash autocomplete, conversation
    /// submit); firing this from any of them is equivalent and safe to
    /// repeat.
    ///
    /// The scan itself runs detached on the foreground executor. If
    /// `~/.agents/skills/` exists it transitions state to
    /// [`SkillsState::Watching`] and starts a recursive watch;
    /// otherwise it transitions back to [`SkillsState::Idle`] so the
    /// next trigger retries (covering the case where the user creates
    /// the directory after the first scan).
    pub fn ensure_skills_scan_started(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.skills_state, SkillsState::Idle) {
            return;
        }
        self.skills_state = SkillsState::Scanning;
        let fs = self.fs.clone();
        cx.spawn(async move |this, cx| Self::run_skills_scan(this, fs, cx).await)
            .detach();
    }

    async fn run_skills_scan(this: WeakEntity<Self>, fs: Arc<dyn Fs>, cx: &mut AsyncApp) {
        let skills_dir = global_skills_dir();
        if !fs.is_dir(&skills_dir).await {
            // Skills directory doesn't exist; revert state so the next
            // user trigger retries.
            let _ = this.update(cx, |this, _cx| {
                this.skills_state = SkillsState::Idle;
            });
            return;
        }

        // Skills directory exists. Start a watch and trigger a refresh
        // of every project's context so the freshly-discovered skills
        // get loaded.
        let _ = this.update(cx, |this, cx| {
            cx.spawn({
                let fs = fs.clone();
                let skills_dir = skills_dir.clone();
                async move |this, cx| Self::run_skills_watch(this, fs, skills_dir, cx).await
            })
            .detach();
            this.skills_state = SkillsState::Watching;
            for state in this.projects.values_mut() {
                state.project_context_needs_refresh.send(()).ok();
            }
        });
    }

    async fn run_skills_watch(
        this: WeakEntity<Self>,
        fs: Arc<dyn Fs>,
        skills_dir: PathBuf,
        cx: &mut AsyncApp,
    ) {
        let (mut events, watcher) = fs
            .watch(&skills_dir, std::time::Duration::from_millis(500))
            .await;

        // Linux's inotify backend is non-recursive, so a watch on
        // `skills_dir` only fires for direct children. Skill discovery
        // is intentionally one level deep (`<skills_dir>/<skill>/SKILL.md`),
        // so we only register watches on each immediate child directory
        // and deliberately do NOT recurse: a stray `node_modules`,
        // `target`, or `.git` inside a skill folder would otherwise
        // register watches for tens of thousands of subdirectories.
        // These per-child adds are cheap no-ops on macOS/Windows where
        // the OS-level watch is already recursive.
        if let Ok(mut entries) = fs.read_dir(&skills_dir).await {
            while let Some(entry) = entries.next().await {
                let Ok(path) = entry else { continue };
                if let Ok(Some(metadata)) = fs.metadata(&path).await
                    && metadata.is_dir
                {
                    watcher.add(&path).ok();
                }
            }
        }

        while let Some(events) = events.next().await {
            // When a new immediate child directory of `skills_dir` is
            // created, add a single watch for it so changes to its
            // `SKILL.md` are observed on Linux. We intentionally do not
            // recurse into the new directory — skill discovery is only
            // one level deep.
            for event in &events {
                if event.kind == Some(fs::PathEventKind::Created)
                    && event.path.parent() == Some(skills_dir.as_path())
                    && fs.is_dir(&event.path).await
                {
                    watcher.add(&event.path).ok();
                }
            }

            let watched_root_removed = events.iter().any(|event| {
                event.path == skills_dir && event.kind == Some(fs::PathEventKind::Removed)
            });

            let updated = this.update(cx, |this, _cx| {
                for state in this.projects.values_mut() {
                    state.project_context_needs_refresh.send(()).ok();
                }
                if watched_root_removed {
                    // Drop back to Idle so the next user trigger
                    // retries the scan; the next trigger will rediscover
                    // the directory if the user has recreated it.
                    this.skills_state = SkillsState::Idle;
                }
            });
            if updated.is_err() || watched_root_removed {
                return;
            }
        }
    }

    pub fn set_sibling_thread_host(&mut self, host: Rc<dyn SiblingThreadHost>) {
        self.sibling_thread_host = Some(host);
    }

    pub fn sibling_thread_host(&self) -> Option<Rc<dyn SiblingThreadHost>> {
        self.sibling_thread_host.clone()
    }

    fn new_session(
        &mut self,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Entity<AcpThread> {
        let project_id = self.get_or_create_project_state(&project, cx);
        let project_state = &self.projects[&project_id];

        let registry = LanguageModelRegistry::read_global(cx);
        let available_count = registry.available_models(cx).count();
        log::debug!("Total available models: {}", available_count);

        let default_model = registry.default_model().and_then(|default_model| {
            self.models
                .model_from_id(&LanguageModels::model_id(&default_model.model))
        });
        let thread = cx.new(|cx| {
            Thread::new(
                project,
                project_state.project_context.clone(),
                project_state.context_server_registry.clone(),
                self.templates.clone(),
                default_model,
                cx,
            )
        });

        self.register_session(thread, project_id, 1, cx)
    }

    fn register_session(
        &mut self,
        thread_handle: Entity<Thread>,
        project_id: EntityId,
        ref_count: usize,
        cx: &mut Context<Self>,
    ) -> Entity<AcpThread> {
        let connection = Rc::new(NativeAgentConnection(cx.entity()));

        let thread = thread_handle.read(cx);
        let session_id = thread.id().clone();
        let parent_session_id = thread.parent_thread_id();
        let title = thread.title();
        let draft_prompt = thread.draft_prompt().map(Vec::from);
        let scroll_position = thread.ui_scroll_position();
        let token_usage = thread.latest_token_usage();
        let project = thread.project.clone();
        let action_log = thread.action_log.clone();
        let prompt_capabilities_rx = thread.prompt_capabilities_rx.clone();
        let acp_thread = cx.new(|cx| {
            let mut acp_thread = acp_thread::AcpThread::new(
                parent_session_id,
                title,
                None,
                connection,
                project.clone(),
                action_log.clone(),
                session_id.clone(),
                prompt_capabilities_rx,
                cx,
            );
            acp_thread.set_draft_prompt(draft_prompt, cx);
            acp_thread.set_ui_scroll_position(scroll_position);
            acp_thread.update_token_usage(token_usage, cx);
            acp_thread
        });

        let registry = LanguageModelRegistry::read_global(cx);
        let summarization_model = registry.thread_summary_model(cx).map(|c| c.model);

        let weak = cx.weak_entity();
        let weak_thread = thread_handle.downgrade();
        thread_handle.update(cx, |thread, cx| {
            thread.set_summarization_model(summarization_model, cx);
            thread.add_default_tools(
                Rc::new(NativeThreadEnvironment {
                    acp_thread: acp_thread.downgrade(),
                    thread: weak_thread,
                    agent: weak.clone(),
                }) as _,
                cx,
            );
            // The resolver closure reads `state.skills` at invocation
            // time, so skills added or removed by the SKILL.md watcher
            // after the thread is constructed are still visible to the
            // model — without this, the catalog and tool would drift out
            // of sync until the session was reopened.
            thread.add_tool(SkillTool::with_body_resolver(
                skills_resolver_for_project(weak.clone(), project_id),
                skill_body_resolver_for_project(project.clone(), self.fs.clone()),
            ));
        });

        let subscriptions = vec![
            cx.subscribe(&thread_handle, Self::handle_thread_title_updated),
            cx.subscribe(&thread_handle, Self::handle_thread_token_usage_updated),
            cx.observe(&thread_handle, move |this, thread, cx| {
                this.save_thread(thread, cx)
            }),
            cx.subscribe(&acp_thread, {
                let session_id = session_id.clone();
                move |this, _, event, cx| {
                    if matches!(event, AcpThreadEvent::ElicitationRequested(_)) {
                        this.request_grind_stop(&session_id, GrindStopReason::UserAttention, cx);
                    }
                }
            }),
        ];

        self.sessions.insert(
            session_id,
            Session {
                thread: thread_handle,
                acp_thread: acp_thread.clone(),
                project_id,
                _subscriptions: subscriptions,
                pending_save: Task::ready(Ok(())),
                ref_count,
                active_grind: None,
            },
        );

        self.update_available_commands_for_project(project_id, cx);

        acp_thread
    }

    pub fn models(&self) -> &LanguageModels {
        &self.models
    }

    fn get_or_create_project_state(
        &mut self,
        project: &Entity<Project>,
        cx: &mut Context<Self>,
    ) -> EntityId {
        let project_id = project.entity_id();
        if self.projects.contains_key(&project_id) {
            return project_id;
        }

        let project_context = cx.new(|_| ProjectContext::new(vec![]));
        self.register_project_with_initial_context(project.clone(), project_context, cx);
        if let Some(state) = self.projects.get_mut(&project_id) {
            state.project_context_needs_refresh.send(()).ok();
        }
        project_id
    }

    fn register_project_with_initial_context(
        &mut self,
        project: Entity<Project>,
        project_context: Entity<ProjectContext>,
        cx: &mut Context<Self>,
    ) {
        let project_id = project.entity_id();

        let context_server_store = project.read(cx).context_server_store();
        let context_server_registry =
            cx.new(|cx| ContextServerRegistry::new(context_server_store.clone(), cx));

        let mut subscriptions = vec![
            cx.subscribe(&project, Self::handle_project_event),
            cx.subscribe(
                &context_server_store,
                Self::handle_context_server_store_updated,
            ),
            cx.subscribe(
                &context_server_registry,
                Self::handle_context_server_registry_event,
            ),
        ];
        // When the user trusts a worktree (or revokes trust), project-local
        // skills become eligible (or ineligible) for loading. Trigger a
        // refresh so the catalog and slash-command list update without a
        // restart. This is unconditional — a `Trusted` event for any
        // worktree under any project is cheap to handle and keeps the
        // logic straightforward.
        if let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) {
            subscriptions.push(
                cx.subscribe(&trusted_worktrees, move |this, _, _event, _cx| {
                    if let Some(state) = this.projects.get_mut(&project_id) {
                        state.project_context_needs_refresh.send(()).ok();
                    }
                }),
            );
        }

        let (project_context_needs_refresh_tx, project_context_needs_refresh_rx) =
            watch::channel(());

        self.projects.insert(
            project_id,
            ProjectState {
                project,
                project_context,
                skills: Arc::new(Vec::new()),
                recipes: Vec::new(),
                skill_loading_issues: Vec::new(),
                project_context_needs_refresh: project_context_needs_refresh_tx,
                _maintain_project_context: cx.spawn(async move |this, cx| {
                    Self::maintain_project_context(
                        this,
                        project_id,
                        project_context_needs_refresh_rx,
                        cx,
                    )
                    .await
                }),
                context_server_registry,
                _subscriptions: subscriptions,
            },
        );
    }

    fn session_project_state(&self, session_id: &acp::SessionId) -> Option<&ProjectState> {
        self.sessions
            .get(session_id)
            .and_then(|session| self.projects.get(&session.project_id))
    }

    async fn maintain_project_context(
        this: WeakEntity<Self>,
        project_id: EntityId,
        mut needs_refresh: watch::Receiver<()>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        while needs_refresh.changed().await.is_ok() {
            let task = this.update(cx, |this, cx| {
                let state = this
                    .projects
                    .get(&project_id)
                    .context("project state not found")?;
                anyhow::Ok(Self::build_project_context(
                    &state.project,
                    this.fs.clone(),
                    cx,
                ))
            })??;
            let (project_context, skills, skill_issue_data) = task.await;
            let recipes = this
                .update(cx, |this, cx| {
                    let project = this
                        .projects
                        .get(&project_id)
                        .context("project state not found")?
                        .project
                        .clone();
                    anyhow::Ok(recipes::discover_recipes(project, this.fs.clone(), cx))
                })??
                .await;
            let skills = Arc::new(skills);
            let skill_loading_issues: Vec<SkillLoadingIssue> = skill_issue_data
                .into_iter()
                .map(|issue| SkillLoadingIssue {
                    project_id,
                    path: issue.path,
                    source_label: issue.source_label,
                    relative_path: issue.relative_path,
                    message: issue.message.into(),
                    kind: issue.kind,
                })
                .collect();
            this.update(cx, |this, cx| {
                // Only emit SkillLoadingIssuesUpdated when the issue list
                // actually changed. Refreshes happen frequently (prompt-store
                // updates, rules-file edits, worktree events, trust-state
                // changes), and re-emitting an unchanged list causes the UI
                // to redisplay issues the user has already dismissed.
                // Transitions from non-empty to empty still count as a change,
                // so subscribers continue to receive an empty list to clear
                // previously-displayed issues when they get resolved.
                let issues_changed = this
                    .projects
                    .get(&project_id)
                    .map(|state| state.skill_loading_issues != skill_loading_issues)
                    .unwrap_or(true);

                if let Some(state) = this.projects.get_mut(&project_id) {
                    state.skills = skills;
                    state.recipes = recipes;
                    state.skill_loading_issues = skill_loading_issues.clone();
                    // Only push the new `ProjectContext` through if it
                    // differs from the current one. The system prompt is
                    // re-rendered from this on every turn, so an unchanged
                    // `ProjectContext` means a byte-identical system prompt
                    // and a continued hit on the model API's prompt cache.
                    // Refreshes fire on many events that don't actually
                    // change what the model sees (e.g. a SKILL.md body edit
                    // that leaves the catalog — name, description, location
                    // — untouched), so this check matters in practice.
                    state
                        .project_context
                        .update(cx, |current_project_context, cx| {
                            if *current_project_context != project_context {
                                *current_project_context = project_context;
                                cx.notify();
                            }
                        });
                }
                if issues_changed {
                    cx.emit(SkillLoadingIssuesUpdated {
                        project_id,
                        issues: skill_loading_issues,
                    });
                }
                // Skills appear in the slash-command list, so a change in
                // the loaded skills needs to be pushed out to active sessions.
                // This runs unconditionally because MCP prompts (also part of
                // the available commands) can change without affecting the
                // skill error list.
                this.update_available_commands_for_project(project_id, cx);
                this.publish_skill_index(cx);
            })?;
        }

        Ok(())
    }

    fn build_project_context(
        project: &Entity<Project>,
        fs: Arc<dyn Fs>,
        cx: &mut App,
    ) -> Task<(ProjectContext, Vec<Skill>, Vec<SkillLoadingIssueData>)> {
        let package_skills = cx
            .try_global::<SkillIndex>()
            .map(|index| index.package_skills.clone())
            .unwrap_or_default();
        let worktrees = project.read(cx).visible_worktrees(cx).collect::<Vec<_>>();
        let worktree_tasks = worktrees
            .iter()
            .map(|worktree| {
                Self::load_worktree_info_for_system_prompt(worktree.clone(), project.clone(), cx)
            })
            .collect::<Vec<_>>();

        // Load global skills
        let global_skills_task = {
            let global_skills_dir = global_skills_dir();
            let global_skills_fs = fs.clone();
            cx.background_spawn(async move {
                load_skills_from_directory(
                    &global_skills_fs,
                    &global_skills_dir,
                    SkillSource::Global,
                )
                .await
            })
        };

        // Load project-local skills, but only from worktrees the user has
        // trusted. Skills in `.agents/skills/` ship with the project; a
        // freshly cloned untrusted repo can carry hostile descriptions or
        // bodies, so we keep them out of the catalog and the slash-command
        // list until trust is granted. The subscription in
        // `register_project_with_initial_context` triggers a context
        // refresh when a worktree's trust state changes, so newly trusted
        // worktrees pick up their skills without restarting.
        let trusted_worktrees = TrustedWorktrees::try_get_global(cx);
        let worktree_store = project.read(cx).worktree_store();
        let project_skills_task = {
            let project = project.clone();
            let trusted_worktrees = worktrees
                .iter()
                .filter_map(|worktree| {
                    let worktree_id = worktree.read(cx).id();
                    let is_trusted = trusted_worktrees.as_ref().is_none_or(|trusted_worktrees| {
                        trusted_worktrees.update(cx, |trusted_worktrees, cx| {
                            trusted_worktrees.can_trust(&worktree_store, worktree_id, cx)
                        })
                    });
                    if !is_trusted {
                        return None;
                    }

                    let worktree_snapshot = worktree.read(cx);
                    let worktree_root_name: Arc<str> = worktree_snapshot.root_name_str().into();
                    let scan_complete = worktree_snapshot
                        .as_local()
                        .map(|local| local.scan_complete());
                    Some((
                        worktree.clone(),
                        worktree_id,
                        worktree_root_name,
                        scan_complete,
                    ))
                })
                .collect::<Vec<_>>();

            cx.spawn(async move |cx| {
                let mut project_skills_results = Vec::new();
                for (worktree, worktree_id, worktree_root_name, scan_complete) in trusted_worktrees
                {
                    if let Some(scan_complete) = scan_complete {
                        scan_complete.await;
                    }
                    if let Err(error) = expand_project_skills_directories(&worktree, cx).await {
                        project_skills_results.push(vec![Err(SkillLoadError {
                            path: PathBuf::from(project_skills_relative_path()),
                            message: format!("Failed to scan project skills: {}", error),
                        })]);
                        continue;
                    }

                    let skill_files = worktree.update(cx, |worktree, _cx| {
                        project_skill_files_from_worktree(worktree)
                    });
                    let source = SkillSource::ProjectLocal {
                        worktree_id: SkillScopeId(worktree_id.to_usize()),
                        worktree_root_name,
                    };

                    let mut worktree_results = Vec::new();
                    for skill_file in skill_files {
                        if skill_file.size > MAX_SKILL_FILE_SIZE as u64 {
                            worktree_results.push(Err(SkillLoadError {
                                path: skill_file.display_path.clone(),
                                message: format!(
                                    "SKILL.md file exceeds maximum size of {}KB",
                                    MAX_SKILL_FILE_SIZE / 1024
                                ),
                            }));
                            continue;
                        }

                        let buffer = match project
                            .update(cx, |project, cx| {
                                project.open_buffer(
                                    (worktree_id, skill_file.relative_path.clone()),
                                    cx,
                                )
                            })
                            .await
                        {
                            Ok(buffer) => buffer,
                            Err(error) => {
                                worktree_results.push(Err(SkillLoadError {
                                    path: skill_file.display_path.clone(),
                                    message: format!("Failed to read file: {}", error),
                                }));
                                continue;
                            }
                        };

                        let content = cx
                            .update(|cx| buffer.read(cx).as_text_snapshot().as_rope().to_string());

                        worktree_results.push(
                            parse_skill_frontmatter(
                                &skill_file.display_path,
                                &content,
                                source.clone(),
                            )
                            .map_err(|error| SkillLoadError {
                                path: skill_file.display_path,
                                message: error.to_string(),
                            }),
                        );
                    }
                    project_skills_results.push(worktree_results);
                }
                project_skills_results
            })
        };
        cx.spawn(async move |_cx| {
            let worktrees = future::join_all(worktree_tasks).await;

            let mut rules_issues = Vec::new();
            let worktrees = worktrees
                .into_iter()
                .map(|(worktree, rules_error)| {
                    if let Some(rules_error) = rules_error {
                        rules_issues.push(SkillLoadingIssueData::from_rules_error(rules_error));
                    }
                    worktree
                })
                .collect::<Vec<_>>();

            // Load and combine skills. `combine_skills` deliberately
            // does NOT deduplicate — the autocomplete popup needs to
            // see every entry so users can disambiguate same-named
            // global vs. project-local skills via the source label.
            // Project-overrides-global is applied below, only for the
            // model-facing catalog.
            let global_skills = global_skills_task.await;
            let project_skills_results = project_skills_task.await;
            let (mut skills, skill_errors) =
                combine_skills(global_skills, project_skills_results.into_iter().flatten());
            skills.extend(package_skills);
            let mut skill_issues = skill_errors
                .into_iter()
                .map(SkillLoadingIssueData::from_load_error)
                .collect::<Vec<_>>();
            skill_issues.extend(rules_issues);
            for skill in &skills {
                skill_issues.extend(
                    skill
                        .load_warnings
                        .iter()
                        .map(|warning| SkillLoadingIssueData::from_load_warning(skill, warning)),
                );
            }

            // Apply project-overrides-global before catalog selection
            // so the model sees at most one entry per name. The full
            // `skills` list is still stored on `ProjectState` and used
            // by the autocomplete popup.
            let overridden = apply_skill_overrides(&skills);

            // Enforce the catalog size budget here so that skills which
            // don't fit produce an issue in the UI rather than being
            // silently swallowed by ProjectContext.
            let (catalog_skills, budget_issues) = select_catalog_skills(&overridden);
            skill_issues.extend(budget_issues);

            let project_context = ProjectContext::new(worktrees).with_skills(catalog_skills);
            (project_context, skills, skill_issues)
        })
    }

    fn load_worktree_info_for_system_prompt(
        worktree: Entity<Worktree>,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Task<(WorktreeContext, Option<RulesLoadingError>)> {
        let tree = worktree.read(cx);
        let root_name = tree.root_name_str().to_string();
        let abs_path = tree.abs_path();
        let scan_complete = tree.as_local().map(|local| local.scan_complete());

        let mut context = WorktreeContext {
            root_name: root_name.clone(),
            abs_path,
            rules_file: None,
        };

        cx.spawn(async move |cx| {
            if let Some(scan_complete) = scan_complete {
                scan_complete.await;
            }

            let rules_task = cx.update(|cx| Self::load_worktree_rules_file(worktree, project, cx));

            let (rules_file, rules_file_error) = match rules_task {
                Some((relative_path, path, rules_task)) => match rules_task.await {
                    Ok(rules_file) => (Some(rules_file), None),
                    Err(err) => {
                        log::warn!(
                            "Failed to load project instructions for {} at {}: {err:#}",
                            root_name,
                            relative_path.display()
                        );
                        (
                            None,
                            Some(RulesLoadingError {
                                path,
                                worktree: root_name.clone().into(),
                                relative_path,
                                message: "The project instruction file could not be read.".into(),
                            }),
                        )
                    }
                },
                None => (None, None),
            };
            context.rules_file = rules_file;
            (context, rules_file_error)
        })
    }

    fn load_worktree_rules_file(
        worktree: Entity<Worktree>,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Option<(PathBuf, PathBuf, Task<Result<RulesFileContext>>)> {
        let worktree = worktree.read(cx);
        let worktree_id = worktree.id();
        let selected_rules_file = RULES_FILE_REL_PATHS
            .iter()
            .filter_map(|name| {
                worktree
                    .entry_for_path(name)
                    .filter(|entry| entry.is_file())
                    .map(|entry| {
                        let path = entry.path.clone();
                        let absolute_path = worktree.absolutize(&path);
                        (path, absolute_path)
                    })
            })
            .next();
        // Note that Cline supports `.clinerules` being a directory, but that is not currently
        // supported. This doesn't seem to occur often in GitHub repositories.
        selected_rules_file.map(|(path_in_worktree, absolute_path)| {
            let relative_path = PathBuf::from(path_in_worktree.as_unix_str().to_string());
            let project_path = ProjectPath {
                worktree_id,
                path: path_in_worktree.clone(),
            };
            let buffer_task =
                project.update(cx, |project, cx| project.open_buffer(project_path, cx));
            let rope_task = cx.spawn(async move |cx| {
                let buffer = buffer_task.await?;
                let (project_entry_id, rope) = buffer.read_with(cx, |buffer, cx| {
                    let project_entry_id = buffer.entry_id(cx).context("buffer has no file")?;
                    anyhow::Ok((project_entry_id, buffer.as_rope().clone()))
                })?;
                anyhow::Ok((project_entry_id, rope))
            });
            // Build a string from the rope on a background thread.
            let rules_task = cx.background_spawn(async move {
                let (project_entry_id, rope) = rope_task.await?;
                anyhow::Ok(RulesFileContext {
                    path_in_worktree,
                    text: rope.to_string().trim().to_string(),
                    project_entry_id: project_entry_id.to_usize(),
                })
            });
            (relative_path, absolute_path, rules_task)
        })
    }

    fn handle_thread_title_updated(
        &mut self,
        thread: Entity<Thread>,
        _: &TitleUpdated,
        cx: &mut Context<Self>,
    ) {
        let session_id = thread.read(cx).id();
        let Some(session) = self.sessions.get(session_id) else {
            return;
        };

        let thread = thread.downgrade();
        let acp_thread = session.acp_thread.downgrade();
        cx.spawn(async move |_, cx| {
            let title = thread.read_with(cx, |thread, _| thread.title())?;
            if let Some(title) = title {
                let task =
                    acp_thread.update(cx, |acp_thread, cx| acp_thread.set_title(title, cx))?;
                task.await?;
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn handle_thread_token_usage_updated(
        &mut self,
        thread: Entity<Thread>,
        usage: &TokenUsageUpdated,
        cx: &mut Context<Self>,
    ) {
        let Some(session) = self.sessions.get(thread.read(cx).id()) else {
            return;
        };
        session.acp_thread.update(cx, |acp_thread, cx| {
            acp_thread.update_token_usage(usage.0.clone(), cx);
        });
    }

    fn handle_project_event(
        &mut self,
        project: Entity<Project>,
        event: &project::Event,
        _cx: &mut Context<Self>,
    ) {
        let project_id = project.entity_id();
        let Some(state) = self.projects.get_mut(&project_id) else {
            return;
        };
        match event {
            project::Event::WorktreeAdded(_)
            | project::Event::WorktreeRemoved(_)
            | project::Event::WorktreeOrderChanged
            | project::Event::WorktreePathsChanged { .. } => {
                state.project_context_needs_refresh.send(()).ok();
            }
            project::Event::WorktreeUpdatedEntries(_, items) => {
                if items.iter().any(|(path, _, _)| {
                    let path_ref = path.as_ref();
                    RULES_FILE_REL_PATHS
                        .iter()
                        .any(|rules_path| path_ref == rules_path.as_ref())
                        || AGENTS_PREFIX
                            .as_ref()
                            .is_some_and(|prefix| path_ref.starts_with(prefix))
                }) {
                    state.project_context_needs_refresh.send(()).ok();
                }
            }
            _ => {}
        }
    }

    fn handle_models_updated_event(
        &mut self,
        _registry: Entity<LanguageModelRegistry>,
        event: &language_model::Event,
        cx: &mut Context<Self>,
    ) {
        self.models.refresh_list(cx);

        let registry = LanguageModelRegistry::read_global(cx);
        let default_model = registry.default_model().map(|m| m.model);
        let summarization_model = registry.thread_summary_model(cx).map(|m| m.model);

        for session in self.sessions.values_mut() {
            session.thread.update(cx, |thread, cx| {
                thread.ensure_model(default_model.as_ref(), cx);

                if let Some(model) = summarization_model.clone() {
                    if thread.summarization_model().is_none()
                        || matches!(event, language_model::Event::ThreadSummaryModelChanged)
                    {
                        thread.set_summarization_model(Some(model), cx);
                    }
                }
            });
        }
    }

    fn handle_context_server_store_updated(
        &mut self,
        store: Entity<project::context_server_store::ContextServerStore>,
        _event: &project::context_server_store::ServerStatusChangedEvent,
        cx: &mut Context<Self>,
    ) {
        let project_id = self.projects.iter().find_map(|(id, state)| {
            if *state.context_server_registry.read(cx).server_store() == store {
                Some(*id)
            } else {
                None
            }
        });
        if let Some(project_id) = project_id {
            self.update_available_commands_for_project(project_id, cx);
        }
    }

    fn handle_context_server_registry_event(
        &mut self,
        registry: Entity<ContextServerRegistry>,
        event: &ContextServerRegistryEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            ContextServerRegistryEvent::ToolsChanged => {}
            ContextServerRegistryEvent::PromptsChanged => {
                let project_id = self.projects.iter().find_map(|(id, state)| {
                    if state.context_server_registry == registry {
                        Some(*id)
                    } else {
                        None
                    }
                });
                if let Some(project_id) = project_id {
                    self.update_available_commands_for_project(project_id, cx);
                }
            }
        }
    }

    fn publish_skill_index(&self, cx: &mut Context<Self>) {
        let mut global_skills = Vec::new();
        let mut project_groups: Vec<ProjectSkillGroup> = Vec::new();
        let mut seen_global = false;

        for state in self.projects.values() {
            for skill in state.skills.iter() {
                match &skill.source {
                    SkillSource::BuiltIn | SkillSource::Package { .. } => {}
                    SkillSource::Global => {
                        if !seen_global {
                            global_skills.push(skill.clone());
                        }
                    }
                    SkillSource::ProjectLocal {
                        worktree_id,
                        worktree_root_name,
                    } => {
                        if let Some(group) = project_groups
                            .iter_mut()
                            .find(|g| g.worktree_id == *worktree_id)
                        {
                            group.skills.push(skill.clone());
                        } else {
                            project_groups.push(ProjectSkillGroup {
                                worktree_id: *worktree_id,
                                worktree_root_name: SharedString::from(worktree_root_name.clone()),
                                skills: vec![skill.clone()],
                            });
                        }
                    }
                }
            }
            if !global_skills.is_empty() {
                seen_global = true;
            }
        }

        let package_skills = cx.global::<SkillIndex>().package_skills.clone();
        cx.set_global(SkillIndex {
            package_skills,
            global_skills,
            project_skills: project_groups,
        });
    }

    fn update_available_commands_for_project(&self, project_id: EntityId, cx: &mut Context<Self>) {
        let available_commands =
            Self::build_available_commands_for_project(self.projects.get(&project_id), cx);
        for session in self.sessions.values() {
            if session.project_id != project_id {
                continue;
            }
            let session_commands = if session.thread.read(cx).is_doctor_repair_proposal() {
                Vec::new()
            } else {
                available_commands.clone()
            };
            session.acp_thread.update(cx, |thread, cx| {
                thread
                    .handle_session_update(
                        acp::SessionUpdate::AvailableCommandsUpdate(
                            acp::AvailableCommandsUpdate::new(session_commands),
                        ),
                        cx,
                    )
                    .log_err();
            });
        }
    }

    fn build_available_commands_for_project(
        project_state: Option<&ProjectState>,
        cx: &App,
    ) -> Vec<acp::AvailableCommand> {
        let Some(state) = project_state else {
            return Vec::new();
        };
        let native_commands = NATIVE_COMMANDS.iter().map(|definition| {
            let command = acp::AvailableCommand::new(definition.name, definition.description).meta(
                acp_thread::meta_with_command_category(acp_thread::CommandCategory::Native),
            );
            if let Some(input_hint) = definition.input_hint {
                command.input(acp::AvailableCommandInput::Unstructured(
                    acp::UnstructuredCommandInput::new(input_hint),
                ))
            } else {
                command
            }
        });

        let registry = state.context_server_registry.read(cx);

        // Reserve the built-in command names so same-named MCP prompts are
        // force-prefixed and stay reachable: an unqualified invocation always
        // routes to the native command.
        let ambiguous_prompt_names = ambiguous_mcp_prompt_names(
            NATIVE_COMMANDS.iter().map(|definition| definition.name),
            registry.prompts().map(|p| p.prompt.name.as_str()),
        );

        let mcp_commands = registry.prompts().flat_map(|context_server_prompt| {
            let prompt = &context_server_prompt.prompt;

            let should_prefix = ambiguous_prompt_names.contains(prompt.name.as_str())
                || prompt.name.starts_with("recipe/");

            let name = if should_prefix {
                format!("{}.{}", context_server_prompt.server_id, prompt.name)
            } else {
                prompt.name.clone()
            };

            let mut command =
                acp::AvailableCommand::new(name, prompt.description.clone().unwrap_or_default())
                    .meta(acp_thread::meta_with_command_category(
                        acp_thread::CommandCategory::Mcp,
                    ));

            if let Some(hint) = mcp_prompt_input_hint(prompt.arguments.as_deref()) {
                command = command.input(acp::AvailableCommandInput::Unstructured(
                    acp::UnstructuredCommandInput::new(hint),
                ));
            }

            Some(command)
        });

        native_commands
            .chain(mcp_commands)
            .chain(
                recipes::commands_for_project(state, cx)
                    .into_iter()
                    .map(|(_, command)| command),
            )
            .collect()
    }

    fn send_local_command_output(
        &mut self,
        session_id: acp::SessionId,
        output: String,
        cx: &mut Context<Self>,
    ) -> Task<Result<acp::PromptResponse>> {
        let Some(session) = self.sessions.get(&session_id) else {
            return Task::ready(Err(anyhow!("Session not found")));
        };
        session.acp_thread.update(cx, |thread, cx| {
            thread.push_local_command_output(output, cx);
        });
        Task::ready(Ok(acp::PromptResponse::new(acp::StopReason::EndTurn)))
    }

    fn send_goal_command(
        &mut self,
        session_id: acp::SessionId,
        argument: &str,
        cx: &mut Context<Self>,
    ) -> Task<Result<acp::PromptResponse>> {
        let command = match parse_goal_command_argument(argument) {
            GoalCommandArgument::Show => {
                let Some(session) = self.sessions.get(&session_id) else {
                    return Task::ready(Err(anyhow!("Session not found")));
                };
                let output = session.thread.read(cx).goal().map_or_else(
                    || "No goal set. Use `/goal <text>` to set one.".to_string(),
                    |goal| format!("Current goal: {goal}"),
                );
                return self.send_local_command_output(session_id, output, cx);
            }
            command => command,
        };

        if self
            .sessions
            .get(&session_id)
            .is_some_and(|session| session.active_grind.is_some())
        {
            return self.send_local_command_output(
                session_id,
                "The goal cannot be changed while a grind is active.".to_string(),
                cx,
            );
        }
        let Some(session) = self.sessions.get_mut(&session_id) else {
            return Task::ready(Err(anyhow!("Session not found")));
        };
        if !session.thread.read(cx).can_clear_conversation()
            || session.acp_thread.read(cx).status() != acp_thread::ThreadStatus::Idle
        {
            return Task::ready(Err(anyhow!(
                "The goal cannot be changed while a turn is running"
            )));
        }

        let Some(state) = self.projects.get(&session.project_id) else {
            return Task::ready(Err(anyhow!("Project state not found for session")));
        };
        let folder_paths = PathList::new(
            &state
                .project
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                .collect::<Vec<_>>(),
        );
        let goal = match command {
            GoalCommandArgument::Clear => None,
            GoalCommandArgument::Set(goal) => Some(SharedString::from(goal.to_string())),
            GoalCommandArgument::Show => return Task::ready(Err(anyhow!("Invalid goal command"))),
        };
        let output = goal.as_ref().map_or_else(
            || "Goal cleared.".to_string(),
            |goal| format!("Goal set: {goal}"),
        );
        let draft_prompt = session.acp_thread.read(cx).draft_prompt().map(Vec::from);
        let db_thread = session.thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(draft_prompt);
            thread.to_db(cx)
        });
        let prior_save = std::mem::replace(&mut session.pending_save, Task::ready(Ok(())));
        let thread_store = self.thread_store.clone();

        cx.spawn(async move |this, cx| {
            prior_save.await?;
            let mut db_thread = db_thread.await;
            db_thread.goal = goal.clone();
            db_thread.updated_at = Utc::now();
            thread_store
                .update(cx, |store, cx| {
                    store.save_thread(session_id.clone(), db_thread, folder_paths, cx)
                })
                .await?;

            this.update(cx, |this, cx| {
                let session = this
                    .sessions
                    .get(&session_id)
                    .context("Session disappeared while changing the goal")?;
                if !session.thread.read(cx).can_clear_conversation()
                    || session.acp_thread.read(cx).status() != acp_thread::ThreadStatus::Idle
                {
                    return Err(anyhow!(
                        "The conversation became active while the goal was being saved"
                    ));
                }
                session
                    .thread
                    .update(cx, |thread, cx| thread.set_goal(goal, cx));
                session.acp_thread.update(cx, |thread, cx| {
                    thread.push_local_command_output(output, cx)
                });
                anyhow::Ok(())
            })??;

            Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
        })
    }

    fn start_grind(
        &mut self,
        session_id: &acp::SessionId,
        max_turns: u8,
        cx: &mut Context<Self>,
    ) -> Result<(Uuid, SharedString)> {
        let session = self
            .sessions
            .get_mut(session_id)
            .context("Session not found")?;
        if session.active_grind.is_some() {
            anyhow::bail!("A grind is already active for this session");
        }
        if !session.thread.read(cx).can_clear_conversation()
            || session.acp_thread.read(cx).status() != acp_thread::ThreadStatus::Idle
        {
            anyhow::bail!("A grind cannot start while a turn is running");
        }
        let goal = session
            .thread
            .read(cx)
            .goal()
            .cloned()
            .context("No goal is set. Use `/goal <text>` before `/grind`.")?;
        let invocation_id = Uuid::new_v4();
        session.active_grind = Some(ActiveGrind {
            invocation_id,
            max_turns,
            completed_turns: 0,
            stop_requested: None,
        });
        session.acp_thread.update(cx, |thread, cx| {
            thread.push_local_command_output(
                format!(
                    "Grind started. This invocation authorizes up to {max_turns} turn{}.",
                    if max_turns == 1 { "" } else { "s" }
                ),
                cx,
            );
        });
        Ok((invocation_id, goal))
    }

    fn request_grind_stop(
        &mut self,
        session_id: &acp::SessionId,
        stop_reason: GrindStopReason,
        cx: &mut Context<Self>,
    ) {
        let Some(session) = self.sessions.get_mut(session_id) else {
            return;
        };
        let Some(grind) = session.active_grind.as_mut() else {
            return;
        };
        if grind.stop_requested.is_none() {
            grind.stop_requested = Some(stop_reason);
        }
        session
            .thread
            .update(cx, |thread, cx| thread.cancel(cx))
            .detach();
    }

    fn send_clear_command(
        &mut self,
        session_id: acp::SessionId,
        cx: &mut Context<Self>,
    ) -> Task<Result<acp::PromptResponse>> {
        let Some(session) = self.sessions.get_mut(&session_id) else {
            return Task::ready(Err(anyhow!("Session not found")));
        };
        if session.active_grind.is_some()
            || !session.thread.read(cx).can_clear_conversation()
            || session.acp_thread.read(cx).status() != acp_thread::ThreadStatus::Idle
        {
            return Task::ready(Err(anyhow!(
                "The conversation cannot be cleared while a turn is running"
            )));
        }

        let Some(state) = self.projects.get(&session.project_id) else {
            return Task::ready(Err(anyhow!("Project state not found for session")));
        };
        let folder_paths = PathList::new(
            &state
                .project
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                .collect::<Vec<_>>(),
        );
        let draft_prompt = session.acp_thread.read(cx).draft_prompt().map(Vec::from);
        let db_thread = session.thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(draft_prompt);
            thread.to_db(cx)
        });
        let prior_save = std::mem::replace(&mut session.pending_save, Task::ready(Ok(())));
        let thread_store = self.thread_store.clone();

        cx.spawn(async move |this, cx| {
            prior_save.await?;
            let mut db_thread = db_thread.await;
            db_thread.clear_conversation();
            thread_store
                .update(cx, |store, cx| {
                    store.save_thread(session_id.clone(), db_thread, folder_paths, cx)
                })
                .await?;

            this.update(cx, |this, cx| {
                let session = this
                    .sessions
                    .get(&session_id)
                    .context("Session disappeared while clearing the conversation")?;
                if !session.thread.read(cx).can_clear_conversation()
                    || session.acp_thread.read(cx).status() != acp_thread::ThreadStatus::Idle
                {
                    return Err(anyhow!(
                        "The conversation became active while it was being cleared"
                    ));
                }

                session
                    .thread
                    .update(cx, |thread, cx| thread.clear_conversation(cx));
                session.acp_thread.update(cx, |thread, cx| {
                    thread.clear_entries(cx);
                    thread.push_local_command_output("Conversation cleared.", cx);
                });
                anyhow::Ok(())
            })??;

            Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
        })
    }

    fn skills_command_output(&self, session_id: &acp::SessionId) -> Result<String> {
        let project_state = self
            .session_project_state(session_id)
            .context("Session not found")?;
        let mut name_counts: HashMap<&str, usize> = HashMap::default();
        for skill in project_state.skills.iter() {
            *name_counts.entry(skill.name.as_str()).or_insert(0) += 1;
        }

        let mut output = String::from("## Available skills\n");
        if project_state.skills.is_empty() {
            output.push_str("\nNo skills are currently available.");
            return Ok(output);
        }

        for skill in project_state.skills.iter() {
            let needs_qualification = matches!(skill.source, SkillSource::Package { .. })
                || name_counts.get(skill.name.as_str()).copied().unwrap_or(0) > 1
                || NATIVE_COMMANDS
                    .iter()
                    .any(|definition| definition.name == skill.name);
            let invocation = if needs_qualification {
                format!("/{}:{}", skill.source.scope_prefix(), skill.name)
            } else {
                format!("/{}", skill.name)
            };
            output.push_str(&format!(
                "\n- `{invocation}` — {} (source: {})",
                skill.description,
                skill.source.display_label(),
            ));
        }
        Ok(output)
    }

    fn status_command_output(&self, session_id: &acp::SessionId, cx: &App) -> Result<String> {
        let session = self.sessions.get(session_id).context("Session not found")?;
        let project_state = self
            .projects
            .get(&session.project_id)
            .context("Project state not found")?;
        let thread = session.thread.read(cx);
        let grind_status = session.active_grind.as_ref().map_or_else(
            || "inactive".to_string(),
            |grind| {
                format!(
                    "active ({}/{} turns completed)",
                    grind.completed_turns, grind.max_turns
                )
            },
        );

        let (model, provider) = match thread.thread_model() {
            ThreadModel::Ready(model) => (
                format!("{} ({})", model.name().0, model.id().0),
                format!("{} ({})", model.provider_name(), model.provider_id()),
            ),
            ThreadModel::Unresolved(selection) => (
                format!("{} (unavailable)", selection.model.0),
                format!("{} (unavailable)", selection.provider),
            ),
            ThreadModel::Unset => ("unavailable".to_string(), "unavailable".to_string()),
        };
        let context_tokens = thread.latest_token_usage().map_or_else(
            || "unavailable".to_string(),
            |usage| format!("{} / {} tokens", usage.used_tokens, usage.max_tokens),
        );
        let personal_instructions =
            match agent_settings::UserAgentsMd::global(cx).map(|agents_md| agents_md.state()) {
                Some(agent_settings::UserAgentsMdState::Loaded(_)) => "loaded (AGENTS.md)",
                Some(agent_settings::UserAgentsMdState::Error(_)) => "unavailable (load error)",
                Some(agent_settings::UserAgentsMdState::Empty) | None => "not loaded",
            };
        let project_context = project_state.project_context.read(cx);

        let mut output = format!(
            "## Status\n\n- Model: {model}\n- Provider: {provider}\n- Context: {context_tokens}\n- Goal: {}\n- Grind: {grind_status}\n- Personal instructions: {personal_instructions}\n- Skills: {}\n- Developer context issues: {}\n- Visible worktrees:",
            thread.goal().map_or("none", SharedString::as_ref),
            project_state.skills.len(),
            project_state.skill_loading_issues.len(),
        );
        if project_context.worktrees.is_empty() {
            output.push_str(" unavailable");
        } else {
            for worktree in &project_context.worktrees {
                let instruction = worktree.rules_file.as_ref().map_or_else(
                    || "no project instructions".to_string(),
                    |rules| rules.path_in_worktree.as_unix_str().to_string(),
                );
                output.push_str(&format!(
                    "\n  - {} — instructions: {}",
                    worktree.root_name, instruction
                ));
            }
        }
        Ok(output)
    }

    pub fn load_thread(
        &mut self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Thread>>> {
        let database_future = ThreadsDatabase::connect(cx);
        cx.spawn(async move |this, cx| {
            let database = database_future.await.map_err(|err| anyhow!(err))?;
            let db_thread = database
                .load_thread(id.clone())
                .await?
                .with_context(|| format!("no thread found with ID: {id:?}"))?;

            this.update(cx, |this, cx| {
                let project_id = this.get_or_create_project_state(&project, cx);
                let project_state = this
                    .projects
                    .get(&project_id)
                    .context("project state not found")?;
                let summarization_model = LanguageModelRegistry::read_global(cx)
                    .thread_summary_model(cx)
                    .map(|c| c.model);

                Ok(cx.new(|cx| {
                    let mut thread = Thread::from_db(
                        id.clone(),
                        db_thread,
                        project_state.project.clone(),
                        project_state.project_context.clone(),
                        project_state.context_server_registry.clone(),
                        this.templates.clone(),
                        cx,
                    );
                    thread.set_summarization_model(summarization_model, cx);
                    thread
                }))
            })?
        })
    }

    pub fn open_thread(
        &mut self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<AcpThread>>> {
        if let Some(session) = self.sessions.get_mut(&id) {
            session.ref_count += 1;
            return Task::ready(Ok(session.acp_thread.clone()));
        }

        if let Some(pending) = self.pending_sessions.get_mut(&id) {
            pending.ref_count += 1;
            let task = pending.task.clone();
            return cx.background_spawn(async move { task.await.map_err(|err| anyhow!(err)) });
        }

        let task = self.load_thread(id.clone(), project.clone(), cx);
        let shared_task = cx
            .spawn({
                let id = id.clone();
                async move |this, cx| {
                    let thread = match task.await {
                        Ok(thread) => thread,
                        Err(err) => {
                            this.update(cx, |this, _cx| {
                                this.pending_sessions.remove(&id);
                            })
                            .ok();
                            return Err(Arc::new(err));
                        }
                    };
                    let acp_thread = this
                        .update(cx, |this, cx| {
                            let project_id = this.get_or_create_project_state(&project, cx);
                            let ref_count = this
                                .pending_sessions
                                .remove(&id)
                                .map_or(1, |pending| pending.ref_count);
                            this.register_session(thread.clone(), project_id, ref_count, cx)
                        })
                        .map_err(Arc::new)?;
                    let events = thread.update(cx, |thread, cx| thread.replay(cx));
                    cx.update(|cx| {
                        NativeAgentConnection::handle_thread_events(
                            events,
                            acp_thread.downgrade(),
                            None,
                            None,
                            cx,
                        )
                    })
                    .await
                    .map_err(Arc::new)?;
                    acp_thread.update(cx, |thread, cx| {
                        thread.snapshot_completed_plan(cx);
                    });
                    Ok(acp_thread)
                }
            })
            .shared();
        self.pending_sessions.insert(
            id,
            PendingSession {
                task: shared_task.clone(),
                ref_count: 1,
            },
        );

        cx.background_spawn(async move { shared_task.await.map_err(|err| anyhow!(err)) })
    }

    pub fn thread_summary(
        &mut self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Result<SharedString>> {
        let thread = self.open_thread(id.clone(), project, cx);
        cx.spawn(async move |this, cx| {
            let acp_thread = thread.await?;
            let result = this
                .update(cx, |this, cx| {
                    this.sessions
                        .get(&id)
                        .unwrap()
                        .thread
                        .update(cx, |thread, cx| thread.summary(cx))
                })?
                .await
                .context("Failed to generate summary")?;

            this.update(cx, |this, cx| this.close_session(&id, cx))?
                .await?;
            drop(acp_thread);
            Ok(result)
        })
    }

    fn close_session(
        &mut self,
        session_id: &acp::SessionId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let Some(session) = self.sessions.get_mut(session_id) else {
            return Task::ready(Ok(()));
        };

        if session.ref_count == 0 {
            return Task::ready(Ok(()));
        }
        session.ref_count -= 1;
        if session.ref_count > 0 {
            return Task::ready(Ok(()));
        }

        if let Some(grind) = session.active_grind.as_mut() {
            grind.stop_requested = Some(GrindStopReason::SessionClosed);
            let cancellation = session.thread.update(cx, |thread, cx| thread.cancel(cx));
            let session_id = session_id.clone();
            return cx.spawn(async move |this, cx| {
                cancellation.await;
                this.update(cx, |this, cx| this.finish_close_session(&session_id, cx))?
                    .await
            });
        }

        self.finish_close_session(session_id, cx)
    }

    fn finish_close_session(
        &mut self,
        session_id: &acp::SessionId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let Some(session) = self.sessions.get(session_id) else {
            return Task::ready(Ok(()));
        };
        let thread = session.thread.clone();
        self.save_thread(thread, cx);
        let Some(session) = self.sessions.remove(session_id) else {
            return Task::ready(Ok(()));
        };
        let project_id = session.project_id;

        let has_remaining = self.sessions.values().any(|s| s.project_id == project_id);
        if !has_remaining {
            self.projects.remove(&project_id);
            self.publish_skill_index(cx);
        }

        session.pending_save
    }

    fn save_thread(&mut self, thread: Entity<Thread>, cx: &mut Context<Self>) {
        let id = thread.read(cx).id().clone();
        let Some(session) = self.sessions.get(&id) else {
            return;
        };
        let Some((id, folder_paths, db_thread)) = self.thread_save_payload(session, cx) else {
            return;
        };

        let database_future = ThreadsDatabase::connect(cx);
        let thread_store = self.thread_store.clone();
        let Some(session) = self.sessions.get_mut(&id) else {
            return;
        };
        session.pending_save = cx.spawn(async move |_, cx| {
            let Some(database) = database_future.await.map_err(|err| anyhow!(err)).log_err() else {
                return Ok(());
            };
            let db_thread = db_thread.await;
            database
                .save_thread(id, db_thread, folder_paths)
                .await
                .log_err();
            thread_store.update(cx, |store, cx| store.reload(cx));
            Ok(())
        });
    }

    /// Builds everything needed to persist a session's thread content,
    /// capturing the current draft prompt from the ACP thread. Returns `None`
    /// if the thread is empty or its project state is gone.
    fn thread_save_payload(
        &self,
        session: &Session,
        cx: &mut App,
    ) -> Option<(acp::SessionId, PathList, Task<DbThread>)> {
        if session.thread.read(cx).is_empty() {
            return None;
        }
        let state = self.projects.get(&session.project_id)?;
        let folder_paths = PathList::new(
            &state
                .project
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                .collect::<Vec<_>>(),
        );
        let folder_paths = session
            .thread
            .read(cx)
            .recipe_scan_policy
            .as_ref()
            .and_then(|policy| policy.project_paths.as_ref())
            .map(PathList::deserialize)
            .unwrap_or(folder_paths);
        let draft_prompt = session.acp_thread.read(cx).draft_prompt().map(Vec::from);
        let id = session.thread.read(cx).id().clone();
        let db_thread = session.thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(draft_prompt);
            thread.to_db(cx)
        });
        Some((id, folder_paths, db_thread))
    }

    /// Commits every non-empty thread's content on shutdown so the async
    /// `save_thread` losing the race can't leave metadata without content.
    fn flush_threads_on_quit(
        &mut self,
        cx: &mut Context<Self>,
    ) -> impl Future<Output = ()> + use<> {
        let database_future = ThreadsDatabase::connect(cx);

        for session in self.sessions.values_mut() {
            if let Some(grind) = session.active_grind.as_mut() {
                grind.stop_requested = Some(GrindStopReason::SessionClosed);
            }
            session
                .thread
                .update(cx, |thread, cx| thread.prepare_for_quit(cx));
        }
        let saves = self
            .sessions
            .values()
            .filter_map(|session| self.thread_save_payload(session, cx))
            .collect::<Vec<_>>();

        async move {
            let Some(database) = database_future.await.map_err(|err| anyhow!(err)).log_err() else {
                return;
            };
            // All quit observers share `gpui::SHUTDOWN_TIMEOUT`, so run the
            // saves concurrently instead of one at a time.
            future::join_all(saves.into_iter().map(|(id, folder_paths, db_thread)| {
                let database = database.clone();
                async move {
                    let db_thread = db_thread.await;
                    database
                        .save_thread(id, db_thread, folder_paths)
                        .await
                        .log_err();
                }
            }))
            .await;
        }
    }

    fn send_mcp_prompt(
        &self,
        client_user_message_id: ClientUserMessageId,
        session_id: acp::SessionId,
        prompt_name: String,
        server_id: ContextServerId,
        arguments: HashMap<String, String>,
        original_content: Vec<acp::ContentBlock>,
        cx: &mut Context<Self>,
    ) -> Task<Result<acp::PromptResponse>> {
        let Some(state) = self.session_project_state(&session_id) else {
            return Task::ready(Err(anyhow!("Project state not found for session")));
        };
        let server_store = state
            .context_server_registry
            .read(cx)
            .server_store()
            .clone();
        let path_style = state.project.read(cx).path_style(cx);

        cx.spawn(async move |this, cx| {
            let prompt =
                crate::get_prompt(&server_store, &server_id, &prompt_name, arguments, cx).await?;

            let (acp_thread, thread) = this.update(cx, |this, _cx| {
                let session = this
                    .sessions
                    .get(&session_id)
                    .context("Failed to get session")?;
                anyhow::Ok((session.acp_thread.clone(), session.thread.clone()))
            })??;

            let mut last_is_user = true;

            thread.update(cx, |thread, cx| {
                thread.push_acp_user_block(
                    client_user_message_id,
                    original_content.into_iter().skip(1),
                    path_style,
                    cx,
                );
            });

            for message in prompt.messages {
                let context_server::types::PromptMessage { role, content } = message;
                let block = mcp_message_content_to_acp_content_block(content);

                match role {
                    context_server::types::Role::User => {
                        let id = acp_thread::ClientUserMessageId::new();

                        acp_thread.update(cx, |acp_thread, cx| {
                            acp_thread.push_user_content_block_with_indent(
                                Some(id.clone()),
                                block.clone(),
                                true,
                                cx,
                            );
                        });

                        thread.update(cx, |thread, cx| {
                            thread.push_acp_user_block(id, [block], path_style, cx);
                        });
                    }
                    context_server::types::Role::Assistant => {
                        acp_thread.update(cx, |acp_thread, cx| {
                            acp_thread.push_assistant_content_block_with_indent(
                                block.clone(),
                                false,
                                true,
                                cx,
                            );
                        });

                        thread.update(cx, |thread, cx| {
                            thread.push_acp_agent_block(block, cx);
                        });
                    }
                }

                last_is_user = role == context_server::types::Role::User;
            }

            let response_stream = thread.update(cx, |thread, cx| {
                if last_is_user {
                    thread.send_existing(cx)
                } else {
                    // Resume if MCP prompt did not end with a user message
                    thread.resume(cx)
                }
            })?;

            let connection = this.upgrade().map(NativeAgentConnection);
            cx.update(|cx| {
                NativeAgentConnection::handle_thread_events(
                    response_stream,
                    acp_thread.downgrade(),
                    connection,
                    None,
                    cx,
                )
            })
            .await
        })
    }

    /// Run a summary-based context compaction in response to the built-in
    /// `/compact` slash command.
    fn send_compact_command(
        &self,
        client_user_message_id: ClientUserMessageId,
        session_id: acp::SessionId,
        cx: &mut Context<Self>,
    ) -> Task<Result<acp::PromptResponse>> {
        cx.spawn(async move |this, cx| {
            let (acp_thread, thread) = this.update(cx, |this, _cx| {
                let session = this
                    .sessions
                    .get(&session_id)
                    .context("Failed to get session")?;
                anyhow::Ok((session.acp_thread.clone(), session.thread.clone()))
            })??;

            let response_stream =
                thread.update(cx, |thread, cx| thread.compact(client_user_message_id, cx))?;
            acp_thread.update(cx, |acp_thread, cx| {
                acp_thread.update_token_usage(None, cx);
            });

            let connection = this.upgrade().map(NativeAgentConnection);
            cx.update(|cx| {
                NativeAgentConnection::handle_thread_events(
                    response_stream,
                    acp_thread.downgrade(),
                    connection,
                    None,
                    cx,
                )
            })
            .await
        })
    }

    /// Activate a skill in response to a `/skill-name` slash command. The
    /// skill body is wrapped in the same `<skill_content>` envelope the
    /// model-driven `skill` tool uses, so the conversation looks the same
    /// regardless of who initiated the load. Any text the user typed after
    /// the command on the same line — plus any additional content blocks
    /// they attached (file mentions, etc.) — is appended to the same user
    /// message after the skill envelope, so the model sees the skill
    /// instructions followed by the user's request.
    fn send_skill_invocation(
        &self,
        client_user_message_id: ClientUserMessageId,
        session_id: acp::SessionId,
        skill: Skill,
        original_content: Vec<acp::ContentBlock>,
        cx: &mut Context<Self>,
    ) -> Task<Result<acp::PromptResponse>> {
        let Some(state) = self.session_project_state(&session_id) else {
            return Task::ready(Err(anyhow!("Project state not found for session")));
        };
        let path_style = state.project.read(cx).path_style(cx);
        let read_skill_body =
            skill_body_resolver_for_project(state.project.clone(), self.fs.clone());

        cx.spawn(async move |this, cx| {
            let (acp_thread, thread) = this.update(cx, |this, _cx| {
                let session = this
                    .sessions
                    .get(&session_id)
                    .context("Failed to get session")?;
                anyhow::Ok((session.acp_thread.clone(), session.thread.clone()))
            })??;

            // Build the model-context message: skill envelope first, then
            // anything the user wrote after the slash command. The first
            // text block has its leading `/cmd` stripped so the literal
            // command name isn't echoed into the model's context, but any
            // text the user typed after it on the same line is preserved
            // verbatim and appended after the envelope.
            //
            // Read the body on demand here — bodies live on disk between
            // materializations to keep memory cost O(total frontmatter)
            // rather than O(total file size).
            let body = if let Some(embedded) = skill.embedded_body {
                embedded.to_string()
            } else {
                read_skill_body(skill.clone(), cx).await.with_context(|| {
                    format!(
                        "Failed to read skill body from {}",
                        skill.skill_file_path.display()
                    )
                })?
            };
            anyhow::ensure!(
                cx.update(|cx| agent_skills::package_skill_is_current(&skill, cx)),
                "Package skill is no longer enabled; refresh the skill list"
            );
            let envelope = crate::tools::render_skill_envelope(&skill, &body);
            let envelope_block = acp::ContentBlock::Text(acp::TextContent::new(envelope));

            let mut user_blocks = original_content;
            if let Some(acp::ContentBlock::Text(text_content)) = user_blocks.first_mut() {
                let stripped = strip_slash_command_prefix(&text_content.text);
                if stripped.trim().is_empty() {
                    user_blocks.remove(0);
                } else {
                    text_content.text = stripped;
                }
            }

            // UI: show the rendered envelope as a sibling user message so
            // the user can see what context was loaded for the skill. The
            // user's own typed message is already rendered by the normal
            // prompt flow, so we don't push it to the UI again here.
            let injected_id = acp_thread::ClientUserMessageId::new();
            acp_thread.update(cx, |acp_thread, cx| {
                acp_thread.push_user_content_block_with_indent(
                    Some(injected_id),
                    envelope_block.clone(),
                    true,
                    cx,
                );
            });

            // Model context: a single user message containing the skill
            // envelope followed by the user's appended content.
            let mut combined = Vec::with_capacity(user_blocks.len() + 1);
            combined.push(envelope_block);
            combined.extend(user_blocks);

            thread.update(cx, |thread, cx| {
                thread.push_acp_user_block(client_user_message_id, combined, path_style, cx);
            });

            let response_stream = thread.update(cx, |thread, cx| thread.send_existing(cx))?;

            let connection = this.upgrade().map(NativeAgentConnection);
            cx.update(|cx| {
                NativeAgentConnection::handle_thread_events(
                    response_stream,
                    acp_thread.downgrade(),
                    connection,
                    None,
                    cx,
                )
            })
            .await
        })
    }
}

/// Wrapper struct that implements the AgentConnection trait
#[derive(Clone)]
pub struct NativeAgentConnection(pub Entity<NativeAgent>);

impl NativeAgentConnection {
    pub fn record_doctor_context(
        &self,
        session_id: &acp::SessionId,
        project: &Entity<Project>,
        summary: &str,
        cx: &mut App,
    ) -> Result<()> {
        anyhow::ensure!(
            summary.len() <= 8192,
            "Doctor conversation summary exceeds its bound"
        );
        let (thread, acp_thread) = {
            let agent = self.0.read(cx);
            let session = agent
                .sessions
                .get(session_id)
                .context("Doctor's native conversation is no longer available")?;
            anyhow::ensure!(
                session.project_id == project.entity_id(),
                "Doctor conversation belongs to another project"
            );
            (session.thread.clone(), session.acp_thread.clone())
        };
        anyhow::ensure!(
            !thread.read(cx).is_doctor_repair_proposal(),
            "A source-proposal session cannot receive unrelated Doctor collection"
        );
        let changed = thread.update(cx, |thread, cx| {
            thread.record_local_command(
                ClientUserMessageId::new(),
                "/doctor collection",
                summary,
                cx,
            )
        });
        if changed {
            acp_thread.update(cx, |thread, cx| {
                thread.push_local_command_output(summary.to_owned(), cx)
            });
        }
        Ok(())
    }

    pub fn thread(&self, session_id: &acp::SessionId, cx: &App) -> Option<Entity<Thread>> {
        self.0
            .read(cx)
            .sessions
            .get(session_id)
            .map(|session| session.thread.clone())
    }

    pub fn restrict_doctor_repair_proposal(
        &self,
        session_id: &acp::SessionId,
        cx: &mut App,
    ) -> Task<Result<()>> {
        let prepared = self.0.update(cx, |agent, cx| {
            let session = agent
                .sessions
                .get(session_id)
                .context("Native session is unavailable")?;
            let project_id = session.project_id;
            let thread = session.thread.clone();
            let state = agent
                .projects
                .get(&project_id)
                .context("Native project is unavailable")?;
            let folder_paths = PathList::new(
                &state
                    .project
                    .read(cx)
                    .visible_worktrees(cx)
                    .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
                    .collect::<Vec<_>>(),
            );
            thread.update(cx, |thread, cx| thread.restrict_doctor_repair_proposal(cx))?;
            let snapshot = thread.read(cx).to_db(cx);
            let session = agent
                .sessions
                .get_mut(session_id)
                .context("Native session disappeared")?;
            let prior_save = std::mem::replace(&mut session.pending_save, Task::ready(Ok(())));
            let store = agent.thread_store.clone();
            agent.update_available_commands_for_project(project_id, cx);
            Ok::<_, anyhow::Error>((snapshot, prior_save, store, folder_paths))
        });
        let (snapshot, prior_save, store, folder_paths) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Task::ready(Err(error)),
        };
        let session_id = session_id.clone();
        cx.spawn(async move |cx| {
            // Normal autosave skips empty sessions; recovery must retain this purpose before any proposal request.
            prior_save.await?;
            let snapshot = snapshot.await;
            store
                .update(cx, |store, cx| {
                    store.save_thread(session_id, snapshot, folder_paths, cx)
                })
                .await?;
            Ok(())
        })
    }

    /// Forwards to [`NativeAgent::ensure_skills_scan_started`]. The
    /// agent panel calls this from its three user-interaction trigger
    /// points (input box focus, slash-autocomplete invocation, and
    /// conversation submit) so that the skills directory is observed
    /// only when the user is actually engaging with the panel.
    pub fn ensure_skills_scan_started(&self, cx: &mut App) {
        self.0
            .update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
    }

    pub fn refresh_skills_for_project(&self, project: Entity<Project>, cx: &mut App) {
        self.0.update(cx, |agent, cx| {
            let project_id = agent.get_or_create_project_state(&project, cx);
            agent.ensure_skills_scan_started(cx);
            if let Some(state) = agent.projects.get_mut(&project_id) {
                state.project_context_needs_refresh.send(()).ok();
            }
        });
    }

    pub fn available_skills(
        &self,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Vec<NativeAvailableSkill> {
        if self
            .thread(session_id, cx)
            .is_some_and(|thread| thread.read(cx).is_doctor_repair_proposal())
        {
            return Vec::new();
        }
        self.0
            .read(cx)
            .session_project_state(session_id)
            .map(|state| {
                state
                    .skills
                    .iter()
                    .filter(|skill| agent_skills::package_skill_is_current(skill, cx))
                    .map(NativeAvailableSkill::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn load_thread(
        &self,
        id: acp::SessionId,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Task<Result<Entity<Thread>>> {
        self.0
            .update(cx, |this, cx| this.load_thread(id, project, cx))
    }

    fn run_turn(
        &self,
        session_id: acp::SessionId,
        cx: &mut App,
        f: impl 'static
        + FnOnce(Entity<Thread>, &mut App) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>>,
    ) -> Task<Result<acp::PromptResponse>> {
        self.run_turn_with_policy(session_id, None, cx, f)
    }

    fn run_grind_turn(
        &self,
        session_id: acp::SessionId,
        cx: &mut App,
        f: impl 'static
        + FnOnce(Entity<Thread>, &mut App) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>>,
    ) -> Task<Result<acp::PromptResponse>> {
        self.run_turn_with_policy(session_id.clone(), Some(session_id), cx, f)
    }

    fn run_turn_with_policy(
        &self,
        session_id: acp::SessionId,
        grind_session_id: Option<acp::SessionId>,
        cx: &mut App,
        f: impl 'static
        + FnOnce(Entity<Thread>, &mut App) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>>,
    ) -> Task<Result<acp::PromptResponse>> {
        let Some((thread, acp_thread)) = self.0.update(cx, |agent, _cx| {
            agent
                .sessions
                .get_mut(&session_id)
                .map(|s| (s.thread.clone(), s.acp_thread.clone()))
        }) else {
            log::error!("Session not found in run_turn: {}", session_id);
            return Task::ready(Err(anyhow!("Session not found")));
        };
        log::debug!("Found session for: {}", session_id);

        let response_stream = match f(thread, cx) {
            Ok(stream) => stream,
            Err(err) => return Task::ready(Err(err)),
        };
        Self::handle_thread_events(
            response_stream,
            acp_thread.downgrade(),
            Some(self.clone()),
            grind_session_id,
            cx,
        )
    }

    fn send_grind_command(
        &self,
        session_id: acp::SessionId,
        argument: &str,
        cx: &mut App,
    ) -> Task<Result<acp::PromptResponse>> {
        let max_turns = match parse_grind_max_turns(argument) {
            Ok(max_turns) => max_turns,
            Err(error) => {
                return self.0.update(cx, |agent, cx| {
                    agent.send_local_command_output(
                        session_id,
                        format!("Cannot start grind: {error}"),
                        cx,
                    )
                });
            }
        };
        let (invocation_id, goal) = match self.0.update(cx, |agent, cx| {
            agent.start_grind(&session_id, max_turns, cx)
        }) {
            Ok(started) => started,
            Err(error) => {
                return self.0.update(cx, |agent, cx| {
                    agent.send_local_command_output(
                        session_id,
                        format!("Cannot start grind: {error}"),
                        cx,
                    )
                });
            }
        };

        let connection = self.clone();
        cx.spawn(async move |cx| {
            loop {
                let Some((turn_number, authorized_turns, completed_turns, stop_requested)) =
                    cx.update(|cx| {
                    connection.0.read_with(cx, |agent, _cx| {
                        let session = agent.sessions.get(&session_id)?;
                        let grind = session.active_grind.as_ref()?;
                        if grind.invocation_id != invocation_id
                            || grind.completed_turns >= grind.max_turns
                        {
                            return None;
                        }
                        Some((
                            grind.completed_turns + 1,
                            grind.max_turns,
                            grind.completed_turns,
                            grind.stop_requested,
                        ))
                    })
                })
                else {
                    return Ok(acp::PromptResponse::new(acp::StopReason::Cancelled));
                };

                if let Some(stop_reason) = stop_requested {
                    cx.update(|cx| {
                        connection.0.update(cx, |agent, cx| {
                            let Some(session) = agent.sessions.get_mut(&session_id) else {
                                return;
                            };
                            if session
                                .active_grind
                                .as_ref()
                                .is_some_and(|grind| grind.invocation_id == invocation_id)
                            {
                                session.active_grind.take();
                                if let Some(output) = grind_stop_output(
                                    stop_reason,
                                    completed_turns,
                                    authorized_turns,
                                ) {
                                    session.acp_thread.update(cx, |thread, cx| {
                                        thread.push_local_command_output(output, cx);
                                    });
                                }
                            }
                        });
                    });
                    return Ok(acp::PromptResponse::new(acp::StopReason::Cancelled));
                }

                let turn = cx.update(|cx| {
                    connection.run_grind_turn(session_id.clone(), cx, {
                        let goal = goal.clone();
                        move |thread, cx| {
                            thread.update(cx, |thread, _cx| {
                                thread.begin_grind_turn(
                                    goal,
                                    turn_number,
                                    authorized_turns,
                                )
                            })?;
                            thread.update(cx, |thread, cx| thread.resume(cx))
                        }
                    })
                });
                let response = turn.await;

                let turn_result = cx.update(|cx| {
                    connection.0.update(cx, |agent, cx| {
                        let session = agent.sessions.get_mut(&session_id)?;
                        let satisfied = session
                            .thread
                            .update(cx, |thread, _cx| thread.finish_grind_turn());
                        let grind = session
                            .active_grind
                            .as_mut()
                            .filter(|grind| grind.invocation_id == invocation_id)?;
                        grind.completed_turns = turn_number;
                        Some((satisfied, grind.stop_requested))
                    })
                });
                let Some((satisfaction, stop_requested)) = turn_result else {
                    return Ok(acp::PromptResponse::new(acp::StopReason::Cancelled));
                };

                if let Some(stop_reason) = stop_requested {
                    cx.update(|cx| {
                        connection.0.update(cx, |agent, cx| {
                            let Some(session) = agent.sessions.get_mut(&session_id) else {
                                return;
                            };
                            if session
                                .active_grind
                                .as_ref()
                                .is_some_and(|grind| grind.invocation_id == invocation_id)
                            {
                                session.active_grind.take();
                                if let Some(output) = grind_stop_output(
                                    stop_reason,
                                    turn_number,
                                    authorized_turns,
                                ) {
                                    session.acp_thread.update(cx, |thread, cx| {
                                        thread.push_local_command_output(output, cx);
                                    });
                                }
                            }
                        });
                    });
                    return Ok(acp::PromptResponse::new(acp::StopReason::Cancelled));
                }

                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        cx.update(|cx| {
                            connection.0.update(cx, |agent, cx| {
                                let Some(session) = agent.sessions.get_mut(&session_id) else {
                                    return;
                                };
                                if session
                                    .active_grind
                                    .as_ref()
                                    .is_some_and(|grind| grind.invocation_id == invocation_id)
                                {
                                    session.active_grind.take();
                                    session.acp_thread.update(cx, |thread, cx| {
                                        thread.push_local_command_output(
                                            format!(
                                                "Grind stopped after {turn_number}/{authorized_turns} turns: {error}"
                                            ),
                                            cx,
                                        );
                                    });
                                }
                            });
                        });
                        return Err(error);
                    }
                };

                let termination = if satisfaction
                    && response.stop_reason == acp::StopReason::EndTurn
                {
                    Some(format!(
                        "Grind complete: goal satisfied after {turn_number}/{authorized_turns} turns."
                    ))
                } else if turn_number >= authorized_turns {
                    Some(format!(
                        "Grind stopped at the turn limit ({turn_number}/{authorized_turns})."
                    ))
                } else if response.stop_reason != acp::StopReason::EndTurn {
                    Some(format!(
                        "Grind stopped after {turn_number}/{authorized_turns} turns ({:?}).",
                        response.stop_reason
                    ))
                } else {
                    None
                };

                if let Some(output) = termination {
                    cx.update(|cx| {
                        connection.0.update(cx, |agent, cx| {
                            let Some(session) = agent.sessions.get_mut(&session_id) else {
                                return;
                            };
                            if session
                                .active_grind
                                .as_ref()
                                .is_some_and(|grind| grind.invocation_id == invocation_id)
                            {
                                session.active_grind.take();
                                session.acp_thread.update(cx, |thread, cx| {
                                    thread.push_local_command_output(output, cx);
                                });
                            }
                        });
                    });
                    return Ok(response);
                }
            }
        })
    }

    fn handle_thread_events(
        mut events: mpsc::UnboundedReceiver<Result<ThreadEvent>>,
        acp_thread: WeakEntity<AcpThread>,
        connection: Option<NativeAgentConnection>,
        grind_session_id: Option<acp::SessionId>,
        cx: &App,
    ) -> Task<Result<acp::PromptResponse>> {
        cx.spawn(async move |cx| {
            // Handle response stream and forward to session.acp_thread
            while let Some(result) = events.next().await {
                match result {
                    Ok(event) => {
                        log::trace!("Received completion event: {:?}", event);

                        match event {
                            ThreadEvent::UserMessage(message) => {
                                acp_thread.update(cx, |thread, cx| {
                                    for content in &*message.content {
                                        thread.push_user_content_block(
                                            Some(message.id.clone()),
                                            content.clone().into(),
                                            cx,
                                        );
                                    }
                                })?;
                            }
                            ThreadEvent::AgentText(text) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.push_assistant_content_block(text.into(), false, cx)
                                })?;
                            }
                            ThreadEvent::AgentThinking(text) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.push_assistant_content_block(text.into(), true, cx)
                                })?;
                            }
                            ThreadEvent::ToolCallAuthorization(ToolCallAuthorization {
                                tool_call,
                                options,
                                response,
                                context: _,
                                kind,
                            }) => {
                                let outcome_task = acp_thread.update(cx, |thread, cx| {
                                    thread.request_tool_call_authorization(
                                        tool_call, options, kind, cx,
                                    )
                                })??;
                                cx.background_spawn(async move {
                                    let outcome = match outcome_task.await {
                                        acp_thread::RequestPermissionOutcome::Selected(outcome) => outcome,
                                        acp_thread::RequestPermissionOutcome::InterruptedByFollowUp => {
                                            acp_thread::SelectedPermissionOutcome::new(
                                                acp::PermissionOptionId::new(
                                                    FOLLOW_UP_PERMISSION_DENIED_OPTION_ID,
                                                ),
                                                acp::PermissionOptionKind::RejectOnce,
                                            )
                                        }
                                        acp_thread::RequestPermissionOutcome::Cancelled => return,
                                    };
                                    response
                                        .send(outcome)
                                        .map_err(|_| anyhow!("authorization receiver was dropped"))
                                        .log_err();
                                })
                                .detach();
                                if let (Some(connection), Some(session_id)) =
                                    (&connection, &grind_session_id)
                                {
                                    cx.update(|cx| {
                                        connection.0.update(cx, |agent, cx| {
                                            agent.request_grind_stop(
                                                session_id,
                                                GrindStopReason::UserAttention,
                                                cx,
                                            );
                                        });
                                    });
                                }
                            }
                            ThreadEvent::ToolCallAuthorizationResolved {
                                tool_call_id,
                                outcome,
                            } => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.authorize_tool_call(tool_call_id, outcome, cx);
                                })?;
                            }
                            ThreadEvent::ToolCall(tool_call) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.upsert_tool_call(tool_call, cx)
                                })??;
                            }
                            ThreadEvent::ToolCallUpdate(update) => {
                                let tool_failed = matches!(
                                    &update,
                                    acp_thread::ToolCallUpdate::UpdateFields(update)
                                        if update.fields.status
                                            == Some(acp::ToolCallStatus::Failed)
                                );
                                acp_thread.update(cx, |thread, cx| {
                                    thread.update_tool_call(update, cx)
                                })??;
                                if tool_failed
                                    && let (Some(connection), Some(session_id)) =
                                        (&connection, &grind_session_id)
                                {
                                    cx.update(|cx| {
                                        connection.0.update(cx, |agent, cx| {
                                            agent.request_grind_stop(
                                                session_id,
                                                GrindStopReason::ToolFailure,
                                                cx,
                                            );
                                        });
                                    });
                                }
                            }
                            ThreadEvent::SubagentSpawned(session_id) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.subagent_spawned(session_id, cx);
                                })?;
                            }
                            ThreadEvent::Retry(status) => {
                                if acp_thread::refusal_fallback_model_from_meta(&status.meta)
                                    .is_some()
                                {
                                    if let Some(connection) = &connection {
                                        cx.update(|cx| {
                                            connection.0.update(cx, |agent, _| {
                                                agent.models.notify_model_selection_changed();
                                            });
                                        });
                                    }
                                }
                                acp_thread.update(cx, |thread, cx| {
                                    thread.update_retry_status(status, cx)
                                })?;
                            }
                            ThreadEvent::ContextCompaction(compaction) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.push_context_compaction(compaction, cx);
                                })?;
                            }
                            ThreadEvent::ContextCompactionUpdate(update) => {
                                acp_thread.update(cx, |thread, cx| {
                                    thread.update_context_compaction(update, cx);
                                })?;
                            }
                            ThreadEvent::Stop(stop_reason) => {
                                log::debug!("Assistant message complete: {:?}", stop_reason);
                                return Ok(acp::PromptResponse::new(stop_reason));
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Error in model response stream: {:?}", e);
                        return Err(e);
                    }
                }
            }

            log::debug!("Response stream completed");
            anyhow::Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
        })
    }
}

struct Command<'a> {
    prompt_name: &'a str,
    arg_value: &'a str,
    /// MCP server prefix from `/<server>.<prompt>` syntax. Mutually
    /// exclusive with `skill_scope` — the two grammars use different
    /// delimiters (`.` for MCP, `:` for skill scopes) so they can't
    /// collide.
    explicit_server_id: Option<&'a str>,
    /// Skill scope qualifier from `/<scope>:<name>` syntax, where
    /// `<scope>` is either the literal `global` or a worktree root
    /// name. The `:` separator namespaces these against MCP server
    /// prefixes (which use `.`) so an MCP server literally named
    /// `global` or named after a worktree still parses unambiguously.
    skill_scope: Option<&'a str>,
}

impl<'a> Command<'a> {
    fn is_unqualified(&self, prompt_name: &str) -> bool {
        self.prompt_name == prompt_name
            && self.explicit_server_id.is_none()
            && self.skill_scope.is_none()
    }

    fn parse(prompt: &'a [acp::ContentBlock]) -> Option<Self> {
        let acp::ContentBlock::Text(text_content) = prompt.first()? else {
            return None;
        };
        let text = text_content.text.trim();
        let command = text.strip_prefix('/')?;
        let (command, arg_value) = command
            .split_once(char::is_whitespace)
            .unwrap_or((command, ""));

        if command.starts_with("recipe/") {
            return Some(Self {
                prompt_name: command,
                arg_value,
                explicit_server_id: None,
                skill_scope: None,
            });
        }

        // Skill scope qualifier: `/<scope>:<name>`. Checked before the
        // MCP `.` grammar because `:` and `.` are different delimiters
        // — the two namespaces can't collide. Skill names are
        // restricted to `[a-z0-9-]+` (no colons), so the LAST `:` is
        // always the scope/name boundary; using `rsplit_once` lets
        // scope labels (e.g. a worktree root name) themselves contain
        // colons without breaking the parse.
        //
        // An empty scope (`/:<name>`) is the qualified form for a
        // global skill — see `SkillSource::scope_prefix`. The name
        // must be non-empty for the colon to be meaningful.
        if let Some((scope, prompt_name)) = command.rsplit_once(':')
            && !prompt_name.is_empty()
        {
            return Some(Self {
                prompt_name,
                arg_value,
                explicit_server_id: None,
                skill_scope: Some(scope),
            });
        }

        if let Some((server_id, prompt_name)) = command.split_once('.') {
            Some(Self {
                prompt_name,
                arg_value,
                explicit_server_id: Some(server_id),
                skill_scope: None,
            })
        } else {
            Some(Self {
                prompt_name: command,
                arg_value,
                explicit_server_id: None,
                skill_scope: None,
            })
        }
    }
}

/// Strip a leading `/cmd` slash command from the start of a text block,
/// returning whatever text comes after it. Mirrors the parsing in
/// [`Command::parse`]: leading whitespace is ignored when locating the `/`,
/// then everything up to (and including) the first whitespace inside the
/// stripped text is dropped. The remainder is preserved verbatim — including
/// any embedded newlines — because users may format their continuation
/// intentionally.
///
/// If the input doesn't begin with `/`, it is returned unchanged so callers
/// degrade gracefully rather than silently mangling unrelated text.
fn strip_slash_command_prefix(text: &str) -> String {
    let trimmed_start = text.trim_start();
    let Some(rest) = trimmed_start.strip_prefix('/') else {
        return text.to_string();
    };
    rest.split_once(char::is_whitespace)
        .map(|(_, after)| after.to_string())
        .unwrap_or_default()
}

struct NativeAgentModelSelector {
    session_id: acp::SessionId,
    connection: NativeAgentConnection,
}

impl acp_thread::AgentModelSelector for NativeAgentModelSelector {
    fn list_models(&self, cx: &mut App) -> Task<Result<acp_thread::AgentModelList>> {
        log::debug!("NativeAgentConnection::list_models called");
        let list = self.connection.0.read(cx).models.model_list.clone();
        Task::ready(if list.is_empty() {
            Err(anyhow::anyhow!("No models available"))
        } else {
            Ok(list)
        })
    }

    fn select_model(&self, model_id: AgentModelId, cx: &mut App) -> Task<Result<()>> {
        log::debug!(
            "Setting model for session {}: {}",
            self.session_id,
            model_id
        );
        let Some(thread) = self
            .connection
            .0
            .read(cx)
            .sessions
            .get(&self.session_id)
            .map(|session| session.thread.clone())
        else {
            return Task::ready(Err(anyhow!("Session not found")));
        };

        let Some(model) = self.connection.0.read(cx).models.model_from_id(&model_id) else {
            return Task::ready(Err(anyhow!("Invalid model ID {}", model_id)));
        };

        let favorite = agent_settings::AgentSettings::get_global(cx)
            .favorite_models
            .iter()
            .find(|favorite| {
                favorite.provider.0 == model.provider_id().0.as_ref()
                    && favorite.model == model.id().0.as_ref()
            })
            .cloned();

        let LanguageModelSelection {
            enable_thinking,
            effort,
            speed,
            ..
        } = agent_settings::language_model_to_selection(&model, favorite.as_ref());

        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.set_thinking_effort(effort.clone(), cx);
            thread.set_thinking_enabled(enable_thinking, cx);
            if let Some(speed) = speed {
                thread.set_speed(speed, cx);
            }
        });

        update_settings_file(
            self.connection.0.read(cx).fs.clone(),
            cx,
            move |settings, cx| {
                let provider = model.provider_id().0.to_string();
                let model = model.id().0.to_string();
                let enable_thinking = thread.read(cx).thinking_enabled();
                let speed = thread.read(cx).speed();
                settings
                    .agent
                    .get_or_insert_default()
                    .set_model(LanguageModelSelection {
                        provider: provider.into(),
                        model,
                        enable_thinking,
                        effort,
                        speed,
                    });
            },
        );

        Task::ready(Ok(()))
    }

    fn selected_model(&self, cx: &mut App) -> Task<Result<acp_thread::AgentModelInfo>> {
        let Some(thread) = self
            .connection
            .0
            .read(cx)
            .sessions
            .get(&self.session_id)
            .map(|session| session.thread.clone())
        else {
            return Task::ready(Err(anyhow!("Session not found")));
        };
        let Some(model) = thread.read(cx).model() else {
            return Task::ready(Err(anyhow!("Model not found")));
        };
        let Some(provider) = LanguageModelRegistry::read_global(cx).provider(&model.provider_id())
        else {
            return Task::ready(Err(anyhow!("Provider not found")));
        };
        Task::ready(Ok(LanguageModels::map_language_model_to_info(
            model, &provider,
        )))
    }

    fn favorite_model_ids(&self, cx: &mut App) -> HashSet<AgentModelId> {
        agent_settings::AgentSettings::get_global(cx)
            .favorite_model_ids()
            .into_iter()
            .map(AgentModelId::from)
            .collect()
    }

    fn toggle_favorite_model(&self, model_id: AgentModelId, should_be_favorite: bool, cx: &App) {
        let selection = model_id_to_selection(&model_id, cx);
        let fs = self.connection.0.read(cx).fs.clone();
        update_settings_file(fs, cx, move |settings, _| {
            let agent = settings.agent.get_or_insert_default();
            if should_be_favorite {
                agent.add_favorite_model(selection.clone());
            } else {
                agent.remove_favorite_model(&selection);
            }
        });
    }

    fn watch(&self, cx: &mut App) -> Option<watch::Receiver<()>> {
        Some(self.connection.0.read(cx).models.watch())
    }

    fn should_render_footer(&self) -> bool {
        true
    }
}

fn model_id_to_selection(model_id: &AgentModelId, cx: &App) -> LanguageModelSelection {
    let id = model_id.as_ref();
    let (provider, model) = id.split_once('/').unwrap_or(("", id));

    let provider_id = LanguageModelProviderId(provider.to_string().into());
    let model_id = LanguageModelId(model.to_string().into());
    let resolved = LanguageModelRegistry::global(cx)
        .read(cx)
        .provider(&provider_id)
        .and_then(|provider| {
            provider
                .provided_models(cx)
                .into_iter()
                .find(|model| model.id() == model_id)
        });

    let Some(resolved) = resolved else {
        return LanguageModelSelection {
            provider: provider.to_owned().into(),
            model: model.to_owned(),
            enable_thinking: false,
            effort: None,
            speed: None,
        };
    };

    let current_user_selection = agent_settings::AgentSettings::get_global(cx)
        .default_model
        .as_ref()
        .filter(|selection| {
            selection.provider.0 == resolved.provider_id().0.as_ref()
                && selection.model == resolved.id().0.as_ref()
        })
        .cloned();

    agent_settings::language_model_to_selection(&resolved, current_user_selection.as_ref())
}

pub fn native_agent_name() -> SharedString {
    format!("{} Agent", product_flavor::DISPLAY_NAME).into()
}

pub static ZED_AGENT_ID: LazyLock<AgentId> = LazyLock::new(|| AgentId::new("Zed Agent"));

impl acp_thread::AgentConnection for NativeAgentConnection {
    fn agent_id(&self) -> AgentId {
        ZED_AGENT_ID.clone()
    }

    fn telemetry_id(&self) -> SharedString {
        "zed".into()
    }

    fn new_session(
        self: Rc<Self>,
        project: Entity<Project>,
        work_dirs: PathList,
        cx: &mut App,
    ) -> Task<Result<Entity<acp_thread::AcpThread>>> {
        log::debug!("Creating new thread for project at: {work_dirs:?}");
        Task::ready(Ok(self
            .0
            .update(cx, |agent, cx| agent.new_session(project, cx))))
    }

    fn supports_load_session(&self) -> bool {
        true
    }

    fn load_session(
        self: Rc<Self>,
        session_id: acp::SessionId,
        project: Entity<Project>,
        _work_dirs: PathList,
        _title: Option<SharedString>,
        cx: &mut App,
    ) -> Task<Result<Entity<acp_thread::AcpThread>>> {
        self.0
            .update(cx, |agent, cx| agent.open_thread(session_id, project, cx))
    }

    fn supports_close_session(&self) -> bool {
        true
    }

    fn close_session(
        self: Rc<Self>,
        session_id: &acp::SessionId,
        cx: &mut App,
    ) -> Task<Result<()>> {
        self.0
            .update(cx, |agent, cx| agent.close_session(session_id, cx))
    }

    fn auth_methods(&self) -> &[acp::AuthMethod] {
        &[] // No auth for in-process
    }

    fn authenticate(&self, _method: acp::AuthMethodId, _cx: &mut App) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn model_selector(&self, session_id: &acp::SessionId) -> Option<Rc<dyn AgentModelSelector>> {
        Some(Rc::new(NativeAgentModelSelector {
            session_id: session_id.clone(),
            connection: self.clone(),
        }) as Rc<dyn AgentModelSelector>)
    }

    fn client_user_message_ids(
        &self,
        _cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionClientUserMessageIds>> {
        let prompt: Rc<dyn acp_thread::AgentSessionClientUserMessageIds> = Rc::new(self.clone());
        Some(prompt)
    }

    fn prompt(
        &self,
        params: acp::PromptRequest,
        cx: &mut App,
    ) -> Task<Result<acp::PromptResponse>> {
        acp_thread::AgentSessionClientUserMessageIds::prompt(
            self,
            acp_thread::AgentSessionClientUserMessageIds::new_id(self),
            params,
            cx,
        )
    }

    fn retry(
        &self,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionRetry>> {
        if self
            .0
            .read(cx)
            .sessions
            .get(session_id)
            .is_some_and(|session| session.thread.read(cx).is_recipe_scan())
        {
            return None;
        }
        Some(Rc::new(NativeAgentSessionRetry {
            connection: self.clone(),
            session_id: session_id.clone(),
        }) as _)
    }

    fn cancel(&self, session_id: &acp::SessionId, cx: &mut App) {
        log::info!("Cancelling on session: {}", session_id);
        self.0.update(cx, |agent, cx| {
            if agent
                .sessions
                .get(session_id)
                .is_some_and(|session| session.active_grind.is_some())
            {
                agent.request_grind_stop(session_id, GrindStopReason::Cancelled, cx);
                return;
            }
            if let Some(session) = agent.sessions.get(session_id) {
                session
                    .thread
                    .update(cx, |thread, cx| {
                        if let Some(cancellation) = thread.recipe_cancellation.take() {
                            if cancellation.send(()).is_err() {
                                log::debug!("Recipe workflow already stopped before cancellation");
                            }
                        }
                        thread.cancel(cx)
                    })
                    .detach();
            }
        });
    }

    fn truncate(
        &self,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionTruncate>> {
        self.0.read_with(cx, |agent, _cx| {
            agent
                .sessions
                .get(session_id)
                .filter(|session| !session.thread.read(cx).is_recipe_scan())
                .map(|session| {
                    Rc::new(NativeAgentSessionTruncate {
                        thread: session.thread.clone(),
                        acp_thread: session.acp_thread.downgrade(),
                    }) as _
                })
        })
    }

    fn set_title(
        &self,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionSetTitle>> {
        self.0.read_with(cx, |agent, _cx| {
            agent
                .sessions
                .get(session_id)
                .filter(|s| !s.thread.read(cx).is_subagent())
                .map(|session| {
                    Rc::new(NativeAgentSessionSetTitle {
                        thread: session.thread.clone(),
                    }) as _
                })
        })
    }

    fn session_list(&self, cx: &mut App) -> Option<Rc<dyn AgentSessionList>> {
        let thread_store = self.0.read(cx).thread_store.clone();
        Some(Rc::new(NativeAgentSessionList::new(thread_store, cx)) as _)
    }

    fn telemetry(&self) -> Option<Rc<dyn acp_thread::AgentTelemetry>> {
        Some(Rc::new(self.clone()) as Rc<dyn acp_thread::AgentTelemetry>)
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

impl acp_thread::AgentSessionClientUserMessageIds for NativeAgentConnection {
    fn prompt(
        &self,
        client_user_message_id: acp_thread::ClientUserMessageId,
        params: acp::PromptRequest,
        cx: &mut App,
    ) -> Task<Result<acp::PromptResponse>> {
        let session_id = params.session_id.clone();
        log::info!("Received prompt request for session: {}", session_id);
        log::debug!("Prompt blocks count: {}", params.prompt.len());

        let Some(project_state) = self.0.read(cx).session_project_state(&session_id) else {
            log::error!("Session not found in prompt: {}", session_id);
            if self.0.read(cx).sessions.contains_key(&session_id) {
                log::error!(
                    "Session found in sessions map, but not in project state: {}",
                    session_id
                );
            }
            return Task::ready(Err(anyhow::anyhow!("Session not found")));
        };

        if let Some(session) = self.0.read(cx).sessions.get(&session_id) {
            if let Some(policy) = &session.thread.read(cx).recipe_scan_policy {
                if let Err(error) = policy.check() {
                    return Task::ready(Err(error));
                }
                return Task::ready(Err(anyhow!(
                    "This isolated test is controlled by its reviewed recipe. Launch a new reviewed test to change the instructions."
                )));
            }
        }

        if let Some(parsed_command) = Command::parse(&params.prompt) {
            if self
                .thread(&session_id, cx)
                .is_some_and(|thread| thread.read(cx).is_doctor_repair_proposal())
            {
                return Task::ready(Err(anyhow!(
                    "Slash commands are unavailable in a Doctor source-proposal session. Use a separately reviewed Doctor operation or an ordinary native session; no command was executed."
                )));
            }
            if parsed_command.is_unqualified("doctor") {
                if !parsed_command.arg_value.trim().is_empty() {
                    return Task::ready(Err(anyhow!("Usage: /doctor")));
                }
                let (completion, acknowledged) = futures::channel::oneshot::channel();
                let request = DoctorOpenRequest {
                    project: project_state.project.clone(),
                    session_id: session_id.clone(),
                    completion: Rc::new(std::cell::RefCell::new(Some(completion))),
                };
                self.0.update(cx, |_, cx| cx.emit(request));
                let agent = self.0.clone();
                return cx.spawn(async move |cx| {
                    let result = acknowledged.await.map_err(|_| anyhow!("Made Doctor's desktop surface is unavailable; no diagnosis or repair was performed")).and_then(|result| result);
                    let output = match &result {
                        Ok(()) => "/doctor — Made Doctor is open in the project editor. Select a worktree, execution host and checks, then choose Collect checks. No checks, repairs or permission grants have run as part of this invocation.".to_owned(),
                        Err(error) => format!("/doctor could not open Made Doctor: {error:#}. No checks, repairs or permission grants have run as part of this invocation."),
                    };
                    agent.update(cx, |agent, cx| {
                        let session = agent.sessions.get(&session_id).ok_or_else(|| anyhow!("Session not found"))?;
                        let changed = session.thread.update(cx, |thread, cx| thread.record_local_command(client_user_message_id, "/doctor", &output, cx));
                        if result.is_ok() && changed {
                            session.acp_thread.update(cx, |thread, cx| thread.push_local_command_output(output, cx));
                        }
                        anyhow::Ok(())
                    })?;
                    result?;
                    Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
                });
            }
            if parsed_command.is_unqualified("schedules") {
                let Some(store) = recipes::RecipeStore::try_global(cx) else {
                    return Task::ready(Err(anyhow!("Schedule management is unavailable")));
                };
                let project = project_state.project.clone();
                let operation = store.update(cx, |store, cx| {
                    store.command(project, &parsed_command.arg_value, cx)
                });
                let agent = self.0.clone();
                return cx.spawn(async move |cx| {
                    let output = operation.await?;
                    agent
                        .update(cx, |agent, cx| {
                            agent.send_local_command_output(session_id, output, cx)
                        })
                        .await
                });
            }
            if parsed_command.is_unqualified("recipes") {
                if !matches!(parsed_command.arg_value.trim(), "" | "list") {
                    return Task::ready(Err(anyhow!("Usage: /recipes [list]")));
                }
                let output = if project_state.recipes.is_empty() {
                    "No recipes are available yet. Add a YAML recipe in a trusted worktree's .agents/recipes directory.".to_owned()
                } else {
                    project_state
                        .recipes
                        .iter()
                        .map(|entry| match &entry.recipe {
                            Ok(recipe) => format!(
                                "{} — {}\n/{}\n{}",
                                recipe.title,
                                entry.origin,
                                entry.qualified_command(),
                                recipe.description
                            ),
                            Err(error) => format!("{} — {error}", entry.origin),
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n")
                };
                return self.0.update(cx, |agent, cx| {
                    agent.send_local_command_output(session_id, output, cx)
                });
            }
            if parsed_command.explicit_server_id.is_none() && parsed_command.skill_scope.is_none() {
                let entry = recipes::commands_for_project(project_state, cx)
                    .into_iter()
                    .find(|(_, command)| command.name == parsed_command.prompt_name)
                    .map(|(entry, _)| entry.clone());
                if let Some(entry) = entry {
                    let arguments = parsed_command.arg_value.to_owned();
                    return recipes::run_command(
                        self,
                        session_id,
                        client_user_message_id,
                        entry,
                        &arguments,
                        params.prompt.into_iter().skip(1).collect(),
                        cx,
                    );
                }
                if parsed_command.prompt_name.starts_with("recipe/") {
                    return Task::ready(Err(anyhow!(
                        "Recipe command is no longer available; refresh recipes"
                    )));
                }
            }
            if parsed_command.is_unqualified(COMPACT_COMMAND_NAME) {
                return self.0.update(cx, |agent, cx| {
                    agent.send_compact_command(client_user_message_id, session_id, cx)
                });
            }

            if parsed_command.is_unqualified(CLEAR_COMMAND_NAME) {
                return self
                    .0
                    .update(cx, |agent, cx| agent.send_clear_command(session_id, cx));
            }

            if parsed_command.is_unqualified(GOAL_COMMAND_NAME) {
                return self.0.update(cx, |agent, cx| {
                    agent.send_goal_command(session_id, parsed_command.arg_value, cx)
                });
            }

            if parsed_command.is_unqualified(GRIND_COMMAND_NAME) {
                return self.send_grind_command(session_id, parsed_command.arg_value, cx);
            }

            if parsed_command.is_unqualified(SKILLS_COMMAND_NAME) {
                let output = match self.0.read(cx).skills_command_output(&session_id) {
                    Ok(output) => output,
                    Err(error) => return Task::ready(Err(error)),
                };
                return self.0.update(cx, |agent, cx| {
                    agent.send_local_command_output(session_id, output, cx)
                });
            }

            if parsed_command.is_unqualified(STATUS_COMMAND_NAME) {
                let output = match self.0.read(cx).status_command_output(&session_id, cx) {
                    Ok(output) => output,
                    Err(error) => return Task::ready(Err(error)),
                };
                return self.0.update(cx, |agent, cx| {
                    agent.send_local_command_output(session_id, output, cx)
                });
            }

            // Skill scope qualifiers (`/:<name>` and
            // `/<worktree>:<name>`) use a colon separator that can't
            // collide with MCP's `/<server>.<name>` grammar. The popup
            // inserts a qualified form for every skill so picking the
            // global row unambiguously runs the global skill even when
            // a same-named project-local one exists.
            if let Some(scope) = parsed_command.skill_scope
                && let Some(skill) = project_state.skills.iter().find(|skill| {
                    skill.name == parsed_command.prompt_name && skill.source.matches_scope(scope)
                })
            {
                let skill = skill.clone();
                return self.0.update(cx, |agent, cx| {
                    agent.send_skill_invocation(
                        client_user_message_id,
                        session_id.clone(),
                        skill,
                        params.prompt,
                        cx,
                    )
                });
            }

            // MCP prompts and skills both register slash commands. MCP
            // prompts are checked first — if a user has both an MCP prompt
            // and a skill with the same name, the MCP prompt wins (matching
            // the order they appear in the catalog).
            let registry = project_state.context_server_registry.read(cx);

            let explicit_server_id = parsed_command
                .explicit_server_id
                .map(|server_id| ContextServerId(server_id.into()));

            if let Some(prompt) =
                registry.find_prompt(explicit_server_id.as_ref(), parsed_command.prompt_name)
            {
                let arguments = match parse_mcp_prompt_arguments(
                    prompt.prompt.arguments.as_deref(),
                    parsed_command.arg_value,
                ) {
                    Ok(arguments) => arguments,
                    Err(error) => return Task::ready(Err(error)),
                };

                let prompt_name = prompt.prompt.name.clone();
                let server_id = prompt.server_id.clone();

                return self.0.update(cx, |agent, cx| {
                    agent.send_mcp_prompt(
                        client_user_message_id,
                        session_id.clone(),
                        prompt_name,
                        server_id,
                        arguments,
                        params.prompt,
                        cx,
                    )
                });
            }

            // Unqualified skill match (`/skill-name` with no scope
            // prefix and no MCP server prefix). Slash commands work
            // for *all* skills regardless of `disable_model_invocation`
            // — that flag only hides the skill from the model's catalog.
            // The user explicitly typed the name, so they get to invoke
            // it.
            //
            // Inlined rather than calling `apply_skill_overrides` so
            // we don't clone the entire skill list on every prompt
            // (including prompts like `/help` that aren't skills at
            // all). The resolution rule matches the override-applied
            // view: among skills with the matching name, pick the one
            // with the highest source precedence, so the slash command
            // picks the same entry the model sees in its catalog.
            // Ties (e.g. two project-local skills from different
            // worktrees) resolve to the first in iteration order to
            // match `apply_skill_overrides`.
            if parsed_command.explicit_server_id.is_none()
                && parsed_command.skill_scope.is_none()
                && !project_state.skills.is_empty()
            {
                let prompt_name = parsed_command.prompt_name;
                let resolved = project_state
                    .skills
                    .iter()
                    .filter(|skill| skill.lookup_name() == prompt_name)
                    .reduce(|best, candidate| {
                        if candidate.source.precedence() > best.source.precedence() {
                            candidate
                        } else {
                            best
                        }
                    });
                if let Some(skill) = resolved {
                    let skill = skill.clone();
                    return self.0.update(cx, |agent, cx| {
                        agent.send_skill_invocation(
                            client_user_message_id,
                            session_id.clone(),
                            skill,
                            params.prompt,
                            cx,
                        )
                    });
                }
            }
        };

        let path_style = project_state.project.read(cx).path_style(cx);

        self.run_turn(session_id, cx, move |thread, cx| {
            let content: Vec<UserMessageContent> = params
                .prompt
                .into_iter()
                .map(|block| UserMessageContent::from_content_block(block, path_style))
                .collect::<Vec<_>>();
            log::debug!("Converted prompt to message: {} chars", content.len());
            log::debug!("Client user message id: {:?}", client_user_message_id);
            log::debug!("Message content: {:?}", content);

            thread.update(cx, |thread, cx| {
                thread.send(client_user_message_id, content, cx)
            })
        })
    }
}

impl acp_thread::AgentTelemetry for NativeAgentConnection {
    fn thread_data(
        &self,
        session_id: &acp::SessionId,
        cx: &mut App,
    ) -> Task<Result<serde_json::Value>> {
        let Some(session) = self.0.read(cx).sessions.get(session_id) else {
            return Task::ready(Err(anyhow!("Session not found")));
        };

        let task = session.thread.read(cx).to_db(cx);
        cx.background_spawn(async move {
            serde_json::to_value(task.await).context("Failed to serialize thread")
        })
    }
}

pub struct NativeAgentSessionList {
    thread_store: Entity<ThreadStore>,
    updates_tx: async_channel::Sender<acp_thread::SessionListUpdate>,
    updates_rx: async_channel::Receiver<acp_thread::SessionListUpdate>,
    _subscription: Subscription,
}

impl NativeAgentSessionList {
    fn new(thread_store: Entity<ThreadStore>, cx: &mut App) -> Self {
        let (tx, rx) = async_channel::unbounded();
        let this_tx = tx.clone();
        let subscription = cx.observe(&thread_store, move |_, _| {
            this_tx
                .try_send(acp_thread::SessionListUpdate::Refresh)
                .ok();
        });
        Self {
            thread_store,
            updates_tx: tx,
            updates_rx: rx,
            _subscription: subscription,
        }
    }

    pub fn thread_store(&self) -> &Entity<ThreadStore> {
        &self.thread_store
    }
}

impl AgentSessionList for NativeAgentSessionList {
    fn list_sessions(
        &self,
        _request: AgentSessionListRequest,
        cx: &mut App,
    ) -> Task<Result<AgentSessionListResponse>> {
        let sessions = self
            .thread_store
            .read(cx)
            .entries()
            .map(|entry| AgentSessionInfo::from(&entry))
            .collect();
        Task::ready(Ok(AgentSessionListResponse::new(sessions)))
    }

    fn supports_delete(&self) -> bool {
        true
    }

    fn delete_session(&self, session_id: &acp::SessionId, cx: &mut App) -> Task<Result<()>> {
        self.thread_store
            .update(cx, |store, cx| store.delete_thread(session_id.clone(), cx))
    }

    fn delete_sessions(&self, cx: &mut App) -> Task<Result<()>> {
        self.thread_store
            .update(cx, |store, cx| store.delete_threads(cx))
    }

    fn watch(
        &self,
        _cx: &mut App,
    ) -> Option<async_channel::Receiver<acp_thread::SessionListUpdate>> {
        Some(self.updates_rx.clone())
    }

    fn notify_refresh(&self) {
        self.updates_tx
            .try_send(acp_thread::SessionListUpdate::Refresh)
            .ok();
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

struct NativeAgentSessionTruncate {
    thread: Entity<Thread>,
    acp_thread: WeakEntity<AcpThread>,
}

impl acp_thread::AgentSessionTruncate for NativeAgentSessionTruncate {
    fn run(
        &self,
        client_user_message_id: acp_thread::ClientUserMessageId,
        cx: &mut App,
    ) -> Task<Result<()>> {
        match self.thread.update(cx, |thread, cx| {
            thread.truncate(client_user_message_id.clone(), cx)?;
            Ok(thread.latest_token_usage())
        }) {
            Ok(usage) => {
                self.acp_thread
                    .update(cx, |thread, cx| {
                        thread.update_token_usage(usage, cx);
                    })
                    .ok();
                Task::ready(Ok(()))
            }
            Err(error) => Task::ready(Err(error)),
        }
    }
}

struct NativeAgentSessionRetry {
    connection: NativeAgentConnection,
    session_id: acp::SessionId,
}

impl acp_thread::AgentSessionRetry for NativeAgentSessionRetry {
    fn run(&self, cx: &mut App) -> Task<Result<acp::PromptResponse>> {
        self.connection
            .run_turn(self.session_id.clone(), cx, |thread, cx| {
                thread.update(cx, |thread, cx| thread.resume(cx))
            })
    }
}

struct NativeAgentSessionSetTitle {
    thread: Entity<Thread>,
}

impl acp_thread::AgentSessionSetTitle for NativeAgentSessionSetTitle {
    fn run(&self, title: SharedString, cx: &mut App) -> Task<Result<()>> {
        self.thread
            .update(cx, |thread, cx| thread.set_title(title, cx));
        Task::ready(Ok(()))
    }
}

pub struct NativeThreadEnvironment {
    agent: WeakEntity<NativeAgent>,
    thread: WeakEntity<Thread>,
    acp_thread: WeakEntity<AcpThread>,
}

impl NativeThreadEnvironment {
    pub(crate) fn create_subagent_thread(
        &self,
        label: String,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let Some(parent_thread_entity) = self.thread.upgrade() else {
            anyhow::bail!("Parent thread no longer exists".to_string());
        };
        let parent_thread = parent_thread_entity.read(cx);
        let current_depth = parent_thread.depth();
        let parent_session_id = parent_thread.id().clone();

        if current_depth >= MAX_SUBAGENT_DEPTH {
            return Err(anyhow!(
                "Maximum subagent depth ({}) reached",
                MAX_SUBAGENT_DEPTH
            ));
        }

        let subagent_thread: Entity<Thread> = cx.new(|cx| {
            let mut thread = Thread::new_subagent(&parent_thread_entity, cx);
            thread.set_title(label.into(), cx);
            thread
        });

        let session_id = subagent_thread.read(cx).id().clone();

        let acp_thread = self
            .agent
            .update(cx, |agent, cx| -> Result<Entity<AcpThread>> {
                let project_id = agent
                    .sessions
                    .get(&parent_session_id)
                    .map(|s| s.project_id)
                    .context("parent session not found")?;
                Ok(agent.register_session(subagent_thread.clone(), project_id, 1, cx))
            })??;

        let depth = current_depth + 1;

        telemetry::event!(
            "Subagent Started",
            session = parent_thread_entity.read(cx).id().to_string(),
            subagent_session = session_id.to_string(),
            depth,
            is_resumed = false,
        );

        self.prompt_subagent(session_id, subagent_thread, acp_thread)
    }

    pub(crate) fn resume_subagent_thread(
        &self,
        session_id: acp::SessionId,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let (subagent_thread, acp_thread) = self.agent.update(cx, |agent, _cx| {
            let session = agent
                .sessions
                .get(&session_id)
                .ok_or_else(|| anyhow!("No subagent session found with id {session_id}"))?;
            anyhow::Ok((session.thread.clone(), session.acp_thread.clone()))
        })??;

        let depth = subagent_thread.read(cx).depth();

        if let Some(parent_thread_entity) = self.thread.upgrade() {
            telemetry::event!(
                "Subagent Started",
                session = parent_thread_entity.read(cx).id().to_string(),
                subagent_session = session_id.to_string(),
                depth,
                is_resumed = true,
            );
        }

        self.prompt_subagent(session_id, subagent_thread, acp_thread)
    }

    fn prompt_subagent(
        &self,
        session_id: acp::SessionId,
        subagent_thread: Entity<Thread>,
        acp_thread: Entity<acp_thread::AcpThread>,
    ) -> Result<Rc<dyn SubagentHandle>> {
        let Some(parent_thread_entity) = self.thread.upgrade() else {
            anyhow::bail!("Parent thread no longer exists".to_string());
        };
        Ok(Rc::new(NativeSubagentHandle::new(
            session_id,
            subagent_thread,
            acp_thread,
            parent_thread_entity,
        )) as _)
    }
}

impl ThreadEnvironment for NativeThreadEnvironment {
    fn create_terminal(
        &self,
        command: String,
        extra_env: Vec<acp::EnvVariable>,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        sandbox_wrap: Option<acp_thread::SandboxWrap>,
        cx: &mut AsyncApp,
    ) -> Task<Result<Rc<dyn TerminalHandle>>> {
        // On Seatbelt-style sandboxes (macOS) there's no tmpfs overlay, so to
        // give the command a writable temp area we point `$TMPDIR`/`$TMP`/
        // `$TEMP` at a per-thread directory inside the sandbox's writable
        // scope. Doing this even when sandboxing is disabled keeps `$TMPDIR`
        // stable so the model can't infer sandbox state from it.
        //
        // Only do this for local projects. For remote projects the temp
        // directory would be created on the client, but the terminal runs on
        // the remote host, so pointing `$TMPDIR` (and the sandbox writable
        // scope) at a client-side path would leak client environment into the
        // remote terminal and reference a directory that doesn't exist there.
        //
        // Linux and Windows are excluded: the bwrap sandbox (run directly on
        // Linux, and via WSL on Windows) already mounts a fresh, writable
        // `tmpfs` over `/tmp`, so the environment looks like a normal
        // filesystem with no special `$TMPDIR` (which would only make the
        // sandbox more obviously Zed-specific). On Windows a per-thread
        // `$TMPDIR` would also be a Windows path that's meaningless inside
        // WSL, and adding it to the writable scope would bind a stray
        // `/mnt/<drive>/...` path.
        #[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(unused_mut))]
        let mut extra_env = extra_env;
        #[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(unused_mut))]
        let mut sandbox_wrap = sandbox_wrap;
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            let temp_dir = self.thread.update(cx, |thread, cx| {
                thread
                    .project()
                    .read(cx)
                    .is_local()
                    .then(|| thread.sandboxed_terminal_temp_dir(cx))
            });
            match temp_dir {
                Ok(Some(Ok(temp_dir))) => {
                    // Canonicalize so the path matches what the sandbox
                    // resolves symlinks to (e.g. `/var` -> `/private/var` on
                    // macOS). `$TMPDIR` and the writable-scope entry below must
                    // agree, and they must agree with the path the kernel
                    // actually checks.
                    let temp_dir = temp_dir.canonicalize().unwrap_or(temp_dir);
                    let temp_dir_string = temp_dir.to_string_lossy().into_owned();
                    extra_env.extend([
                        acp::EnvVariable::new("TMPDIR", &temp_dir_string),
                        acp::EnvVariable::new("TMP", &temp_dir_string),
                        acp::EnvVariable::new("TEMP", &temp_dir_string),
                    ]);
                    // The command's `$TMPDIR` must live inside the sandbox's
                    // writable scope. The per-thread temp directory is owned
                    // here (not in the terminal tool that assembles the rest
                    // of the writable set), so add it whenever the command is
                    // sandboxed.
                    if let Some(sandbox_wrap) = &mut sandbox_wrap {
                        sandbox_wrap.writable_paths.push(temp_dir);
                    }
                }
                Ok(None) => {}
                Ok(Some(Err(error))) => return Task::ready(Err(error)),
                Err(error) => return Task::ready(Err(error)),
            };
        }
        let task = self.acp_thread.update(cx, |thread, cx| {
            thread.create_terminal(
                command,
                vec![],
                extra_env,
                cwd,
                output_byte_limit,
                sandbox_wrap,
                cx,
            )
        });

        let acp_thread = self.acp_thread.clone();
        cx.spawn(async move |cx| {
            let terminal = task?.await?;

            let (drop_tx, drop_rx) = oneshot::channel();
            let terminal_id = terminal.read_with(cx, |terminal, _cx| terminal.id().clone());

            cx.spawn(async move |cx| {
                drop_rx.await.ok();
                acp_thread.update(cx, |thread, cx| thread.release_terminal(terminal_id, cx))
            })
            .detach();

            let handle = AcpTerminalHandle {
                terminal,
                _drop_tx: Some(drop_tx),
            };

            Ok(Rc::new(handle) as _)
        })
    }

    fn create_subagent(&self, label: String, cx: &mut App) -> Result<Rc<dyn SubagentHandle>> {
        self.create_subagent_thread(label, cx)
    }

    fn resume_subagent(
        &self,
        session_id: acp::SessionId,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        self.resume_subagent_thread(session_id, cx)
    }

    fn create_sibling_thread(
        &self,
        request: SiblingThreadRequest,
        cx: &mut AsyncApp,
    ) -> Task<Result<SiblingThreadInfo>> {
        let host = match self
            .agent
            .read_with(cx, |agent, _| agent.sibling_thread_host())
        {
            Ok(Some(host)) => host,
            Ok(None) => {
                return Task::ready(Err(anyhow!(
                    "No sibling-thread host is registered. This usually means the \
                     agent panel hasn't been initialized in this workspace."
                )));
            }
            Err(err) => return Task::ready(Err(err)),
        };
        host.create_sibling_thread(request, cx)
    }

    fn list_available_agents(&self, cx: &mut App) -> Result<AvailableAgents> {
        let host = self
            .agent
            .read_with(cx, |agent, _| agent.sibling_thread_host())?
            .ok_or_else(|| {
                anyhow!(
                    "No sibling-thread host is registered. This usually means the \
                     agent panel hasn't been initialized in this workspace."
                )
            })?;
        host.list_available_agents(cx)
    }
}

#[derive(Debug, Clone)]
enum SubagentPromptResult {
    Completed,
    Cancelled,
    ContextWindowWarning,
    Error(String),
}

pub struct NativeSubagentHandle {
    session_id: acp::SessionId,
    parent_thread: WeakEntity<Thread>,
    subagent_thread: Entity<Thread>,
    acp_thread: Entity<acp_thread::AcpThread>,
}

impl NativeSubagentHandle {
    fn new(
        session_id: acp::SessionId,
        subagent_thread: Entity<Thread>,
        acp_thread: Entity<acp_thread::AcpThread>,
        parent_thread_entity: Entity<Thread>,
    ) -> Self {
        NativeSubagentHandle {
            session_id,
            subagent_thread,
            parent_thread: parent_thread_entity.downgrade(),
            acp_thread,
        }
    }
}

impl SubagentHandle for NativeSubagentHandle {
    fn id(&self) -> acp::SessionId {
        self.session_id.clone()
    }

    fn num_entries(&self, cx: &App) -> usize {
        self.acp_thread.read(cx).entries().len()
    }

    fn send(&self, message: String, cx: &AsyncApp) -> Task<Result<String>> {
        let thread = self.subagent_thread.clone();
        let acp_thread = self.acp_thread.clone();
        let subagent_session_id = self.session_id.clone();
        let parent_thread = self.parent_thread.clone();

        cx.spawn(async move |cx| {
            let (task, _subscription) = cx.update(|cx| {
                let ratio_before_prompt = thread
                    .read(cx)
                    .latest_token_usage()
                    .map(|usage| usage.ratio());

                parent_thread
                    .update(cx, |parent_thread, _cx| {
                        parent_thread.register_running_subagent(thread.downgrade())
                    })
                    .ok();

                let task = acp_thread.update(cx, |acp_thread, cx| {
                    acp_thread.send(vec![message.into()], cx)
                });

                let (token_limit_tx, token_limit_rx) = oneshot::channel::<()>();
                let mut token_limit_tx = Some(token_limit_tx);

                let subscription = cx.subscribe(
                    &thread,
                    move |_thread, event: &TokenUsageUpdated, _cx| {
                        if let Some(usage) = &event.0 {
                            let old_ratio = ratio_before_prompt
                                .clone()
                                .unwrap_or(TokenUsageRatio::Normal);
                            let new_ratio = usage.ratio();
                            if old_ratio == TokenUsageRatio::Normal
                                && new_ratio == TokenUsageRatio::Warning
                            {
                                if let Some(tx) = token_limit_tx.take() {
                                    tx.send(()).ok();
                                }
                            }
                        }
                    },
                );

                let wait_for_prompt = cx
                    .background_spawn(async move {
                        futures::select! {
                            response = task.fuse() => match response {
                                Ok(Some(response)) => {
                                    match response.stop_reason {
                                        acp::StopReason::Cancelled => SubagentPromptResult::Cancelled,
                                        acp::StopReason::MaxTokens => SubagentPromptResult::Error("The agent reached the maximum number of tokens.".into()),
                                        acp::StopReason::MaxTurnRequests => SubagentPromptResult::Error("The agent reached the maximum number of allowed requests between user turns. Try prompting again.".into()),
                                        acp::StopReason::Refusal => SubagentPromptResult::Error("The agent refused to process that prompt. Try again.".into()),
                                        acp::StopReason::EndTurn | _ => SubagentPromptResult::Completed,
                                    }
                                }
                                Ok(None) => SubagentPromptResult::Error("No response from the agent. You can try messaging again.".into()),
                                Err(error) => SubagentPromptResult::Error(error.to_string()),
                            },
                            _ = token_limit_rx.fuse() => SubagentPromptResult::ContextWindowWarning,
                        }
                    });

                (wait_for_prompt, subscription)
            });

            let result = match task.await {
                SubagentPromptResult::Completed => thread.read_with(cx, |thread, _cx| {
                    thread
                        .last_message()
                        .and_then(|message| {
                            let content = message.as_agent_message()?
                                .content
                                .iter()
                                .filter_map(|c| match c {
                                    AgentMessageContent::Text(text) => Some(text.as_str()),
                                    _ => None,
                                })
                                .join("\n\n");
                            if content.is_empty() {
                                None
                            } else {
                                Some( content)
                            }
                        })
                        .context("No response from subagent")
                }),
                SubagentPromptResult::Cancelled => Err(anyhow!("User canceled")),
                SubagentPromptResult::Error(message) => Err(anyhow!("{message}")),
                SubagentPromptResult::ContextWindowWarning => {
                    thread.update(cx, |thread, cx| thread.cancel(cx)).await;
                    Err(anyhow!(
                        "The agent is nearing the end of its context window and has been \
                         stopped. You can prompt the thread again to have the agent wrap up \
                         or hand off its work."
                    ))
                }
            };

            parent_thread
                .update(cx, |parent_thread, cx| {
                    parent_thread.unregister_running_subagent(&subagent_session_id, cx)
                })
                .ok();

            result
        })
    }
}

pub struct AcpTerminalHandle {
    terminal: Entity<acp_thread::Terminal>,
    _drop_tx: Option<oneshot::Sender<()>>,
}

impl TerminalHandle for AcpTerminalHandle {
    fn id(&self, cx: &AsyncApp) -> Result<acp::TerminalId> {
        Ok(self.terminal.read_with(cx, |term, _cx| term.id().clone()))
    }

    fn wait_for_exit(&self, cx: &AsyncApp) -> Result<Shared<Task<acp::TerminalExitStatus>>> {
        Ok(self
            .terminal
            .read_with(cx, |term, _cx| term.wait_for_exit()))
    }

    fn current_output(&self, cx: &AsyncApp) -> Result<acp::TerminalOutputResponse> {
        Ok(self
            .terminal
            .read_with(cx, |term, cx| term.current_output(cx)))
    }

    fn kill(&self, cx: &AsyncApp) -> Result<()> {
        cx.update(|cx| {
            self.terminal.update(cx, |terminal, cx| {
                terminal.kill(cx);
            });
        });
        Ok(())
    }

    fn was_stopped_by_user(&self, cx: &AsyncApp) -> Result<bool> {
        Ok(self
            .terminal
            .read_with(cx, |term, _cx| term.was_stopped_by_user()))
    }
}

/// Build the catalog the model sees in its system prompt: filter out hidden
/// (`disable_model_invocation`) skills, then drop the rest if they would push
/// the catalog past the description budget.
///
/// Returns `SkillSummary` values rather than full `Skill`s so that the
/// (potentially ~100KB) skill bodies aren't cloned just to be discarded by
/// `ProjectContext::new`, which only needs the summary fields.
fn select_catalog_skills(skills: &[Skill]) -> (Vec<SkillSummary>, Vec<SkillLoadingIssueData>) {
    let mut kept = Vec::new();
    let mut issues = Vec::new();
    let mut dropped: Vec<&Skill> = Vec::new();
    let mut total_size = 0usize;
    let mut budget_exceeded = false;

    for skill in skills {
        if skill.disable_model_invocation {
            continue;
        }

        let entry_size = skill.lookup_name().len() + skill.description.len();
        if !budget_exceeded && total_size.saturating_add(entry_size) <= MAX_SKILL_DESCRIPTIONS_SIZE
        {
            total_size += entry_size;
            kept.push(SkillSummary::from(skill));
        } else {
            // Once any model-invocable skill overflows the budget, stop
            // packing entirely so the cutoff is deterministic by sort order
            // rather than dependent on which skills happen to be small
            // enough to fit in the remaining space.
            budget_exceeded = true;
            dropped.push(skill);
        }
    }

    if !dropped.is_empty() {
        let budget_kb = MAX_SKILL_DESCRIPTIONS_SIZE / 1024;
        let first = dropped[0];
        let message = if dropped.len() == 1 {
            let entry_size = first.name.len() + first.description.len();
            format!(
                "Skill '{}' ({:.1}KB description) was dropped from the catalog because the previous skills already used the entire {}KB description budget.",
                first.name,
                entry_size as f64 / 1024.0,
                budget_kb,
            )
        } else {
            let mut message = format!(
                "{} skills were dropped from the catalog because they exceeded the {}KB description budget:",
                dropped.len(),
                budget_kb,
            );
            for skill in &dropped {
                let entry_size = skill.lookup_name().len() + skill.description.len();
                message.push('\n');
                message.push_str(&format!(
                    "- {} ({:.1}KB description)",
                    skill.name,
                    entry_size as f64 / 1024.0,
                ));
            }
            message
        };
        issues.push(SkillLoadingIssueData::catalog_budget_exceeded(
            first.skill_file_path.clone(),
            message,
        ));
    }

    (kept, issues)
}

/// Build a closure that, when called, reads the latest `state.skills`
/// for the given project from the `NativeAgent` and applies
/// project-overrides-global so the `SkillTool` resolves a name to the
/// same entry the model sees in its catalog. Run at invocation time
/// (not thread-build time) so skill changes after thread construction
/// become visible without re-registering the tool.
pub fn skills_resolver_for_project(
    weak_agent: WeakEntity<NativeAgent>,
    project_id: EntityId,
) -> impl Fn(&App) -> Arc<Vec<Skill>> + Send + Sync + 'static {
    move |cx: &App| {
        weak_agent
            .upgrade()
            .and_then(|agent| {
                agent.read(cx).projects.get(&project_id).map(|state| {
                    let skills = state
                        .skills
                        .iter()
                        .filter(|skill| agent_skills::package_skill_is_current(skill, cx))
                        .cloned()
                        .collect::<Vec<_>>();
                    Arc::new(apply_skill_overrides(&skills))
                })
            })
            .unwrap_or_else(|| Arc::new(Vec::new()))
    }
}

pub fn skill_body_resolver_for_project(
    project: Entity<Project>,
    fs: Arc<dyn Fs>,
) -> impl Fn(Skill, &mut AsyncApp) -> Task<Result<String>> + Send + Sync + 'static {
    move |skill, cx| match skill.source.clone() {
        SkillSource::Package { root, .. } => {
            let fs = fs.clone();
            cx.spawn(async move |cx| {
                anyhow::ensure!(
                    cx.update(|cx| agent_skills::package_skill_is_current(&skill, cx)),
                    "Package skill is no longer enabled"
                );
                let content =
                    agent_skills::read_package_skill(fs.as_ref(), &root, &skill.skill_file_path)
                        .await?;
                anyhow::ensure!(
                    cx.update(|cx| agent_skills::package_skill_is_current(&skill, cx)),
                    "Package skill is no longer enabled"
                );
                agent_skills::parse_package_skill_content(&content)
                    .map(|(_, body)| body.trim().to_string())
            })
        }
        SkillSource::ProjectLocal { worktree_id, .. } => {
            let project = project.clone();
            cx.spawn(async move |cx| {
                let worktree_id = WorktreeId::from_usize(worktree_id.0);
                let worktree = project
                    .update(cx, |project, cx| project.worktree_for_id(worktree_id, cx))
                    .context("no such worktree")?;
                expand_project_skills_directories(&worktree, cx).await?;
                let relative_path = worktree.update(cx, |worktree, _cx| {
                    let worktree_root = worktree.abs_path();
                    worktree
                        .path_style()
                        .strip_prefix(&skill.skill_file_path, &worktree_root)
                        .map(|relative_path| relative_path.into_arc())
                        .context("skill file is not inside its worktree")
                })?;

                let buffer = project
                    .update(cx, |project, cx| {
                        project.open_buffer((worktree_id, relative_path), cx)
                    })
                    .await?;
                let content =
                    cx.update(|cx| buffer.read(cx).as_text_snapshot().as_rope().to_string());

                read_skill_body_from_content(&skill.skill_file_path, &content).map_err(Into::into)
            })
        }
        SkillSource::BuiltIn | SkillSource::Global => {
            let fs = fs.clone();
            cx.background_spawn(async move {
                agent_skills::read_skill_body(fs.as_ref(), &skill.skill_file_path)
                    .await
                    .map_err(Into::into)
            })
        }
    }
}

/// Collect successfully-loaded global and project-local skills into a
/// single list, preserving every entry — even when two skills share a
/// name. The autocomplete popup shows the full list with origin labels
/// so users can tell same-named skills apart; override resolution
/// (project-local wins over global) happens later via
/// [`apply_skill_overrides`] at the boundaries where the model
/// interacts with skills (system-prompt catalog, `SkillTool` lookup,
/// slash-command invocation).
///
/// Global versions of skills will be before the local versions
fn combine_skills(
    global: Vec<Result<Skill, SkillLoadError>>,
    project: impl Iterator<Item = Result<Skill, SkillLoadError>>,
) -> (Vec<Skill>, Vec<SkillLoadError>) {
    // Built-in skills go first (lowest priority) so that global and
    // project-local skills with the same name shadow them.
    let mut skills = builtin_skills();
    let mut errors = Vec::new();
    for result in global.into_iter().chain(project) {
        match result {
            Ok(skill) => skills.push(skill),
            Err(e) => errors.push(e),
        }
    }
    log_skill_conflicts(&skills);
    (skills, errors)
}

/// Emit a warning for each name collision between skills. Called once
/// per skill load (not per query), so the log isn't spammed by repeated
/// catalog rebuilds.
fn log_skill_conflicts(skills: &[Skill]) {
    let mut by_name: HashMap<std::borrow::Cow<'_, str>, &Skill> = HashMap::default();
    for skill in skills {
        match by_name.get(&skill.lookup_name()) {
            Some(existing) => {
                if skill.source.precedence() > existing.source.precedence() {
                    log::warn!(
                        "Skill '{}' at '{}' overrides skill at '{}' for the model; both appear in the slash-command popup with their source",
                        skill.name,
                        skill.skill_file_path.display(),
                        existing.skill_file_path.display(),
                    );
                    by_name.insert(skill.lookup_name(), skill);
                } else {
                    log::warn!(
                        "Skill '{}' at '{}' conflicts with skill at '{}'; the model will see the first one, but both appear in the slash-command popup with their source",
                        skill.name,
                        skill.skill_file_path.display(),
                        existing.skill_file_path.display(),
                    );
                }
            }
            None => {
                by_name.insert(skill.lookup_name(), skill);
            }
        }
    }
}

/// Project-local skills override same-named global skills. Returns a
/// new list with at most one entry per name. Two skills of the same
/// source colliding (e.g. two globals or two project-locals) keep the
/// first one to match the historical behavior.
///
/// This is the projection of `state.skills` used by everything the
/// model interacts with: the system-prompt catalog, the `SkillTool`'s
/// name resolver, and slash-command invocation. The autocomplete popup
/// deliberately does *not* go through this — it shows the full list so
/// users can see what's shadowed.
fn apply_skill_overrides(skills: &[Skill]) -> Vec<Skill> {
    let mut result: Vec<Skill> = Vec::new();
    let mut indices: HashMap<std::borrow::Cow<'_, str>, usize> = HashMap::default();
    for skill in skills {
        match indices.get(&skill.lookup_name()).copied() {
            Some(idx) => {
                if skill.source.precedence() > result[idx].source.precedence() {
                    result[idx] = skill.clone();
                }
            }
            None => {
                indices.insert(skill.lookup_name(), result.len());
                result.push(skill.clone());
            }
        }
    }
    result
}

#[cfg(test)]
mod internal_tests {
    use std::path::Path;

    use super::*;
    use acp_thread::{AgentConnection, AgentModelGroupName, AgentModelInfo, MentionUri};
    use agent_settings::COMPACTION_PROMPT;
    use context_server::test::FakeTransport;
    use context_server::{ContextServer, types as mcp};
    use feature_flags::{AcpBetaFeatureFlag, FeatureFlag as _, FeatureFlagAppExt as _};
    use fs::FakeFs;
    use gpui::{BorrowAppContext as _, TestAppContext};
    use indoc::formatdoc;
    use language_model::fake_provider::{FakeLanguageModel, FakeLanguageModelProvider};
    use language_model::{
        CompletionIntent, LanguageModelCompletionError, LanguageModelCompletionEvent,
        LanguageModelProviderId, LanguageModelProviderName, LanguageModelToolUse,
        StopReason as LanguageModelStopReason,
    };
    use schemars::JsonSchema;
    use serde_json::json;
    use settings::SettingsStore;
    use std::sync::Mutex;
    use util::{path, rel_path::rel_path};

    #[derive(JsonSchema, Serialize, Deserialize)]
    struct GrindPermissionToolInput {}

    struct GrindPermissionTool;

    impl AgentTool for GrindPermissionTool {
        type Input = GrindPermissionToolInput;
        type Output = String;

        const NAME: &'static str = "grind_permission_tool";

        fn kind() -> acp::ToolKind {
            acp::ToolKind::Other
        }

        fn initial_title(
            &self,
            _input: Result<Self::Input, serde_json::Value>,
            _cx: &mut App,
        ) -> SharedString {
            "Grind permission tool".into()
        }

        fn run(
            self: Arc<Self>,
            input: ToolInput<Self::Input>,
            event_stream: ToolCallEventStream,
            cx: &mut App,
        ) -> Task<Result<String, String>> {
            cx.spawn(async move |cx| {
                input.recv().await.map_err(|error| error.to_string())?;
                let authorization = cx.update(|cx| {
                    event_stream.authorize(
                        "Authorize grind test tool?",
                        ToolPermissionContext::new(Self::NAME, vec![String::new()]),
                        cx,
                    )
                });
                authorization.await.map_err(|error| error.to_string())?;
                Ok("authorized".to_string())
            })
        }
    }

    #[derive(JsonSchema, Serialize, Deserialize)]
    struct GrindPendingToolInput {}

    struct GrindPendingTool;

    impl AgentTool for GrindPendingTool {
        type Input = GrindPendingToolInput;
        type Output = String;

        const NAME: &'static str = "grind_pending_tool";

        fn kind() -> acp::ToolKind {
            acp::ToolKind::Other
        }

        fn initial_title(
            &self,
            _input: Result<Self::Input, serde_json::Value>,
            _cx: &mut App,
        ) -> SharedString {
            "Pending grind tool".into()
        }

        fn run(
            self: Arc<Self>,
            input: ToolInput<Self::Input>,
            _event_stream: ToolCallEventStream,
            cx: &mut App,
        ) -> Task<Result<String, String>> {
            cx.spawn(async move |_cx| {
                input.recv().await.map_err(|error| error.to_string())?;
                future::pending().await
            })
        }
    }

    fn make_global_skill(name: &str, description: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: description.to_string(),
            source: SkillSource::Global,
            directory_path: PathBuf::from(format!("/home/user/.agents/skills/{name}")),
            skill_file_path: PathBuf::from(format!("/home/user/.agents/skills/{name}/SKILL.md")),
            load_warnings: Vec::new(),
            disable_model_invocation: false,
            embedded_body: None,
        }
    }

    fn mcp_argument(name: &str, description: Option<&str>, required: bool) -> mcp::PromptArgument {
        mcp::PromptArgument {
            name: name.to_string(),
            title: None,
            description: description.map(str::to_string),
            required: Some(required),
        }
    }

    fn mcp_prompt(
        name: &str,
        description: &str,
        arguments: Option<Vec<mcp::PromptArgument>>,
    ) -> mcp::Prompt {
        mcp::Prompt {
            name: name.to_string(),
            title: None,
            description: Some(description.to_string()),
            arguments,
        }
    }

    async fn start_test_mcp_prompt_server(
        project: &Entity<Project>,
        server_id: &str,
        prompts: Vec<mcp::Prompt>,
        cx: &mut TestAppContext,
    ) -> (
        Arc<ContextServer>,
        Arc<Mutex<Vec<(String, Option<HashMap<String, String>>)>>>,
    ) {
        let prompts = Arc::new(serde_json::to_value(prompts).unwrap());
        let prompt_calls = Arc::new(Mutex::new(Vec::new()));
        let transport = FakeTransport::new(cx.executor())
            .on_request::<mcp::requests::Initialize, _>(|_params| async move {
                mcp::InitializeResponse {
                    protocol_version: mcp::ProtocolVersion(
                        mcp::LATEST_PROTOCOL_VERSION.to_string(),
                    ),
                    capabilities: mcp::ServerCapabilities {
                        prompts: Some(mcp::PromptsCapabilities {
                            list_changed: Some(true),
                        }),
                        ..Default::default()
                    },
                    server_info: mcp::Implementation {
                        name: "test prompts".to_string(),
                        title: None,
                        version: "1.0.0".to_string(),
                        description: None,
                    },
                    meta: None,
                }
            })
            .on_request::<mcp::requests::PromptsList, _>(move |_params| {
                let prompts = serde_json::from_value((*prompts).clone()).unwrap();
                async move {
                    mcp::PromptsListResponse {
                        prompts,
                        next_cursor: None,
                        meta: None,
                    }
                }
            })
            .on_request::<mcp::requests::PromptsGet, _>({
                let prompt_calls = prompt_calls.clone();
                move |params| {
                    prompt_calls
                        .lock()
                        .unwrap()
                        .push((params.name.clone(), params.arguments.clone()));
                    async move {
                        let arguments = params.arguments.unwrap_or_default();
                        mcp::PromptsGetResponse {
                            description: None,
                            messages: vec![
                                mcp::PromptMessage {
                                    role: mcp::Role::User,
                                    content: mcp::MessageContent::Text {
                                        text: format!("generated prompt: {arguments:?}"),
                                        annotations: None,
                                    },
                                },
                                mcp::PromptMessage {
                                    role: mcp::Role::Assistant,
                                    content: mcp::MessageContent::Text {
                                        text: "prepared response".to_string(),
                                        annotations: None,
                                    },
                                },
                            ],
                            meta: None,
                        }
                    }
                }
            });
        let server = Arc::new(ContextServer::new(
            ContextServerId(server_id.into()),
            Arc::new(transport),
        ));
        let store = project.read_with(cx, |project, _cx| project.context_server_store());
        store.update(cx, |store, cx| store.test_start_server(server.clone(), cx));
        cx.run_until_parked();
        (server, prompt_calls)
    }

    async fn setup_native_agent_session(
        cx: &mut TestAppContext,
    ) -> (
        Rc<NativeAgentConnection>,
        Entity<NativeAgent>,
        Entity<Project>,
        Entity<AcpThread>,
    ) {
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [Path::new("/a")], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs, cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));
        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();

        (connection, agent, project, acp_thread)
    }

    fn native_thread_for_session(
        agent: &Entity<NativeAgent>,
        session_id: &acp::SessionId,
        cx: &App,
    ) -> Entity<Thread> {
        agent.read_with(cx, |agent, _cx| {
            agent.sessions.get(session_id).unwrap().thread.clone()
        })
    }

    fn enable_test_tools_for_thread(
        thread: &Entity<Thread>,
        tool_names: &[&str],
        cx: &mut TestAppContext,
    ) {
        let profile_id = thread.read_with(cx, |thread, _| thread.profile().clone());
        cx.update(|cx| {
            let mut settings = agent_settings::AgentSettings::get_global(cx).clone();
            let profile = settings
                .profiles
                .get_mut(&profile_id)
                .expect("native test profile should exist");
            for tool_name in tool_names {
                profile.tools.insert((*tool_name).into(), true);
            }
            agent_settings::AgentSettings::override_global(settings, cx);
        });
    }

    async fn send_native_command_for_test(
        connection: &Rc<NativeAgentConnection>,
        session_id: &acp::SessionId,
        command: &str,
        cx: &mut TestAppContext,
    ) -> Result<acp::PromptResponse> {
        cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec![command.into()]),
                cx,
            )
        })
        .await
    }

    async fn send_native_command_through_acp_thread_for_test(
        acp_thread: &Entity<AcpThread>,
        command: &str,
        cx: &mut TestAppContext,
    ) -> Result<Option<acp::PromptResponse>> {
        let send = acp_thread.update(cx, |thread, cx| {
            thread.send_command(vec![command.into()], cx)
        });
        cx.foreground_executor().spawn(send).await
    }

    fn start_grind_for_test(
        connection: &Rc<NativeAgentConnection>,
        session_id: &acp::SessionId,
        command: &str,
        cx: &mut TestAppContext,
    ) -> Task<Result<acp::PromptResponse>> {
        cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec![command.into()]),
                cx,
            )
        })
    }

    fn request_texts_after_system(
        messages: &[language_model::LanguageModelRequestMessage],
    ) -> Vec<String> {
        messages
            .iter()
            .skip(1)
            .map(language_model::LanguageModelRequestMessage::string_contents)
            .collect()
    }

    #[gpui::test]
    async fn doctor_source_command_catalog_and_dispatch_preserve_normal_sessions(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, restricted) = setup_native_agent_session(cx).await;
        let session_id = restricted.read_with(cx, |thread, _| thread.session_id().clone());
        let ordinary = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .expect("Ordinary sibling");
        let ordinary_id = ordinary.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        model.forbid_requests();
        let opened = Rc::new(std::cell::Cell::new(0));
        let observed = opened.clone();
        let _subscription = cx.update(|cx| {
            cx.subscribe(&agent, move |_, request: &DoctorOpenRequest, _| {
                observed.set(observed.get() + 1);
                request.complete(Ok(()));
            })
        });
        cx.update(|cx| {
            agent.update(cx, |agent, _| {
                Arc::make_mut(
                    &mut agent
                        .projects
                        .get_mut(&project.entity_id())
                        .expect("Project catalog")
                        .skills,
                )
                .push(make_builtin_skill(
                    "doctor-catalog-sentinel",
                    "Source boundary fixture",
                ));
            });
            assert!(!connection.available_skills(&session_id, cx).is_empty());
            native_thread_for_session(&agent, &session_id, cx)
                .update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        });
        cx.update(|cx| connection.restrict_doctor_repair_proposal(&session_id, cx))
            .await
            .expect("Durable fresh Doctor purpose");
        assert!(
            cx.update(|cx| connection.restrict_doctor_repair_proposal(&session_id, cx))
                .await
                .is_err(),
            "Duplicate purpose setup is rejected"
        );
        assert!(
            cx.update(|cx| connection
                .restrict_doctor_repair_proposal(&"missing-doctor-session".into(), cx))
                .await
                .is_err()
        );
        cx.run_until_parked();
        assert!(restricted.read_with(cx, |thread, _| thread.available_commands().is_empty()));
        cx.update(|cx| {
            assert!(connection.available_skills(&session_id, cx).is_empty());
            assert!(!connection.available_skills(&ordinary_id, cx).is_empty());
            agent.update(cx, |agent, cx| {
                agent.update_available_commands_for_project(project.entity_id(), cx)
            });
        });
        cx.run_until_parked();
        assert!(
            restricted.read_with(cx, |thread, _| thread.available_commands().is_empty()),
            "Late project catalogs remain filtered"
        );
        assert!(ordinary.read_with(cx, |thread, _| {
            thread
                .available_commands()
                .iter()
                .any(|command| command.name == "doctor")
        }));
        let before = restricted.read_with(cx, |thread, _| thread.entries().len());
        for command in [
            "/doctor",
            "/recipes",
            "/schedules",
            "/recipe/builtin/release-risk",
            "/release-risk",
            "/skills",
            "/:doctor-catalog-sentinel",
            "/fixture-server.review",
            "/goal new work",
            "/grind 10",
            "/compact",
            "/clear",
            "/status",
        ] {
            let error = send_native_command_for_test(&connection, &session_id, command, cx)
                .await
                .expect_err("Doctor dispatch rejects before effects");
            assert!(
                error.to_string().contains("Doctor source-proposal session"),
                "{command}: {error}"
            );
        }
        assert_eq!(
            opened.get(),
            0,
            "Rejected commands never open another surface"
        );
        assert_eq!(model.completion_count(), 0);
        assert_eq!(
            restricted.read_with(cx, |thread, _| thread.entries().len()),
            before
        );
        let response = send_native_command_for_test(&connection, &ordinary_id, "/doctor", cx)
            .await
            .expect("Normal dispatch remains available");
        assert_eq!(response.stop_reason, acp::StopReason::EndTurn);
        assert_eq!(opened.get(), 1);
    }

    #[gpui::test]
    async fn doctor_source_command_persistence_failure_and_interruption_never_submit(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        for fail_prior_save in [false, true] {
            let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
            let session_id = session.read_with(cx, |thread, _| thread.session_id().clone());
            let store = agent.read_with(cx, |agent, _| agent.thread_store.clone());
            let model = Arc::new(FakeLanguageModel::default());
            model.forbid_requests();
            let (release, blocked) = futures::channel::oneshot::channel::<()>();
            cx.update(|cx| {
                native_thread_for_session(&agent, &session_id, cx)
                    .update(cx, |thread, cx| thread.set_model(model.clone(), cx));
                if fail_prior_save {
                    agent.update(cx, |agent, _| {
                        agent
                            .sessions
                            .get_mut(&session_id)
                            .expect("Session")
                            .pending_save = Task::ready(Err(anyhow!("Fixture failed prior save")));
                    });
                } else {
                    store.update(cx, |store, _| {
                        store.save_barrier = Some(
                            async move {
                                blocked.await.expect("Release persistence barrier");
                            }
                            .boxed()
                            .shared(),
                        );
                    });
                }
            });
            let restriction =
                cx.update(|cx| connection.restrict_doctor_repair_proposal(&session_id, cx));
            cx.run_until_parked();
            assert!(session.read_with(cx, |thread, _| thread.available_commands().is_empty()));
            if fail_prior_save {
                assert!(
                    restriction
                        .await
                        .expect_err("Failed durable admission")
                        .to_string()
                        .contains("failed prior save")
                );
            } else {
                drop(restriction);
                release.send(()).expect("Release cancelled admission");
                store.update(cx, |store, _| store.save_barrier = None);
                cx.run_until_parked();
            }
            assert_eq!(model.completion_count(), 0);
            let saved = store
                .update(cx, |store, cx| store.load_thread(session_id.clone(), cx))
                .await
                .expect("Read existing store");
            assert!(
                saved.is_none(),
                "Uncommitted admission never becomes a durable proposal"
            );
            assert!(
                send_native_command_for_test(&connection, &session_id, "/recipes", cx)
                    .await
                    .is_err(),
                "In-memory purpose still cannot enlarge authority"
            );
        }
    }

    #[gpui::test]
    async fn doctor_source_command_persisted_purpose_reloads_without_commands_or_replay(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |thread, _| thread.session_id().clone());
        cx.update(|cx| connection.restrict_doctor_repair_proposal(&session_id, cx))
            .await
            .expect("Durable fresh source proposal");
        let native = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        let saved = native.read_with(cx, |thread, cx| thread.to_db(cx)).await;
        assert!(saved.doctor_repair_proposal);
        let store = agent.read_with(cx, |agent, _| agent.thread_store.clone());
        let committed = store
            .update(cx, |store, cx| store.load_thread(session_id.clone(), cx))
            .await
            .expect("Existing durable session owner")
            .expect("Purpose saved before any model request");
        assert!(committed.doctor_repair_proposal);
        assert!(committed.messages.is_empty());
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .expect("Close source proposal");
        let reopened = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project.clone(),
                    PathList::new(&["/a"]),
                    None,
                    cx,
                )
            })
            .await
            .expect("Existing session owner reload");
        cx.run_until_parked();
        let model = Arc::new(FakeLanguageModel::default());
        model.forbid_requests();
        cx.update(|cx| {
            let native = native_thread_for_session(&agent, &session_id, cx);
            assert!(native.read(cx).is_doctor_repair_proposal());
            native.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
            agent.update(cx, |agent, cx| {
                agent.update_available_commands_for_project(project.entity_id(), cx)
            });
            assert!(connection.available_skills(&session_id, cx).is_empty());
        });
        cx.run_until_parked();
        assert!(reopened.read_with(cx, |thread, _| thread.available_commands().is_empty()));
        for command in ["/recipes", "/fixture-server.review", "/grind 10"] {
            assert!(
                send_native_command_for_test(&connection, &session_id, command, cx)
                    .await
                    .expect_err("Loaded restriction")
                    .to_string()
                    .contains("Doctor source-proposal session")
            );
        }
        assert_eq!(
            model.completion_count(),
            0,
            "Reload does not execute or replay work"
        );
    }

    #[gpui::test]
    async fn doctor_command_acknowledged_desktop_opens_without_model_or_permissions(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, thread) = setup_native_agent_session(cx).await;
        let session = thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        model.forbid_requests();
        cx.update(|cx| {
            native_thread_for_session(&agent, &session, cx)
                .update(cx, |thread, cx| thread.set_model(model.clone(), cx))
        });
        let opened = Rc::new(std::cell::Cell::new(0));
        let observed = opened.clone();
        let desktop_project = project.clone();
        let _subscription = cx.update(|cx| {
            cx.subscribe(&agent, move |_, request: &DoctorOpenRequest, _| {
                assert_eq!(request.project, desktop_project);
                observed.set(observed.get() + 1);
                request.complete(Ok(()));
            })
        });
        let response = send_native_command_through_acp_thread_for_test(&thread, "/doctor", cx)
            .await
            .expect("Native dispatch")
            .expect("Local response");
        assert_eq!(response.stop_reason, acp::StopReason::EndTurn);
        assert_eq!(opened.get(), 1);
        assert_eq!(model.completion_count(), 0);
        send_native_command_through_acp_thread_for_test(&thread, "/doctor", cx)
            .await
            .expect("Repeated dispatch")
            .expect("Local response");
        assert_eq!(
            opened.get(),
            2,
            "Every invocation must reveal the actual desktop view"
        );
        cx.update(|cx| {
            let native = native_thread_for_session(&agent, &session, cx);
            let history = native.read(cx).to_markdown();
            assert!(history.contains("/doctor"));
            assert!(history.contains("Collect checks"));
            assert!(history.contains("No checks, repairs or permission grants have run"));
            assert_eq!(
                native.read(cx).recipe_message_count(),
                2,
                "One command/status pair is retained without duplicate adjacent acknowledgements"
            );
        });
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.entries().len()),
            1,
            "Repeated open does not duplicate transient acknowledgements"
        );
        thread.read_with(cx, |thread, cx| {
            assert!(
                thread
                    .entries()
                    .last()
                    .expect("Acknowledged output")
                    .to_markdown(cx)
                    .contains("Made Doctor is open")
            );
        });
        let collection_context = "/doctor collection status: Collected. Project: a. Cargo failure observed; application checks Not run; no repair was performed.";
        cx.update(|cx| {
            connection
                .record_doctor_context(&session, &project, collection_context, cx)
                .expect("Existing native conversation receives actual collection status");
            connection
                .record_doctor_context(&session, &project, collection_context, cx)
                .expect("Repeated status is deduplicated");
            let native = native_thread_for_session(&agent, &session, cx);
            assert_eq!(native.read(cx).recipe_message_count(), 4);
        });
        cx.update(|cx| connection.clone().close_session(&session, cx))
            .await
            .expect("Save and close local-command session");
        cx.update(|cx| {
            connection.clone().load_session(
                session.clone(),
                project.clone(),
                PathList::new(&["/a"]),
                None,
                cx,
            )
        })
        .await
        .expect("Reload through the existing session owner");
        cx.run_until_parked();
        let restored = cx.update(|cx| native_thread_for_session(&agent, &session, cx));
        let history = restored.read_with(cx, |thread, _| thread.to_markdown());
        assert!(history.contains("/doctor") && history.contains("Collect checks"));
        assert!(history.contains(collection_context));
        assert_eq!(opened.get(), 2, "Restoration never replays the command");
        assert_eq!(model.completion_count(), 0);
        restored.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let request = restored
            .read_with(cx, |thread, cx| {
                thread.build_completion_request(CompletionIntent::UserPrompt, cx)
            })
            .expect("Follow-up model context");
        let context = request
            .messages
            .iter()
            .map(language_model::LanguageModelRequestMessage::string_contents)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            context.contains("/doctor")
                && context.contains("Collect checks")
                && context.contains("No checks, repairs or permission grants have run")
        );
        assert!(
            context.contains(collection_context),
            "Follow-up context must include collection status, not only the opening acknowledgement"
        );
    }

    #[gpui::test]
    async fn doctor_command_missing_rejected_and_closed_desktop_never_claim_success(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _, thread) = setup_native_agent_session(cx).await;
        let session = thread.read_with(cx, |thread, _| thread.session_id().clone());
        assert!(
            send_native_command_for_test(&connection, &session, "/doctor unexpected", cx)
                .await
                .expect_err("No arguments accepted")
                .to_string()
                .contains("Usage")
        );
        assert!(
            send_native_command_for_test(&connection, &session, "/doctor", cx)
                .await
                .expect_err("No desktop host")
                .to_string()
                .contains("unavailable")
        );
        let _subscription = cx.update(|cx| {
            cx.subscribe(&agent, |_, request: &DoctorOpenRequest, _| {
                request.complete(Err(anyhow!("Desktop project changed")))
            })
        });
        assert!(
            send_native_command_for_test(&connection, &session, "/doctor", cx)
                .await
                .expect_err("Rejected host")
                .to_string()
                .contains("project changed")
        );
        assert!(thread.read_with(cx, |thread, _| thread.entries().is_empty()));
    }

    #[gpui::test]
    async fn doctor_command_is_reserved_advertised_and_isolated_scans_cannot_open_ui(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _, thread) = setup_native_agent_session(cx).await;
        let session = thread.read_with(cx, |thread, _| thread.session_id().clone());
        cx.run_until_parked();
        assert!(
            thread.read_with(cx, |thread, _| thread.available_commands().iter().any(
                |command| command.name == "doctor"
                    && acp_thread::command_category_from_meta(&command.meta)
                        == Some(acp_thread::CommandCategory::Native)
            ))
        );
        assert!(
            ambiguous_mcp_prompt_names(
                NATIVE_COMMANDS.iter().map(|command| command.name),
                ["doctor"]
            )
            .contains("doctor")
        );
        let opened = Rc::new(std::cell::Cell::new(false));
        let observed = opened.clone();
        let _subscription = cx.update(|cx| {
            cx.subscribe(&agent, move |_, request: &DoctorOpenRequest, _| {
                observed.set(true);
                request.complete(Ok(()));
            })
        });
        cx.update(|cx| {
            native_thread_for_session(&agent, &session, cx).update(cx, |thread, _| {
                thread.recipe_scan_policy = Some(Default::default())
            })
        });
        assert!(
            send_native_command_for_test(&connection, &session, "/doctor", cx)
                .await
                .is_err()
        );
        assert!(
            !opened.get(),
            "Isolation is checked before command dispatch"
        );
    }

    #[gpui::test]
    async fn test_native_commands_are_available(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        let connection = NativeAgentConnection(agent.clone());
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        cx.update(|cx| {
            let commands = acp_thread.read(cx).available_commands();
            let native_commands = commands
                .iter()
                .filter(|command| {
                    acp_thread::command_category_from_meta(&command.meta)
                        == Some(acp_thread::CommandCategory::Native)
                })
                .map(|command| command.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                native_commands,
                vec![
                    "doctor",
                    "schedules",
                    "recipes",
                    "compact",
                    "clear",
                    "skills",
                    "status",
                    "goal",
                    "grind",
                    "recipe/builtin/release-risk",
                    "release-risk"
                ]
            );
            for command_name in [GOAL_COMMAND_NAME, GRIND_COMMAND_NAME] {
                let command = commands
                    .iter()
                    .find(|command| command.name == command_name)
                    .expect("argument-bearing native command should be advertised");
                assert!(
                    matches!(
                        command.input,
                        Some(acp::AvailableCommandInput::Unstructured(_))
                    ),
                    "{command_name} should advertise unstructured input"
                );
            }
            assert!(!commands.iter().any(|command| command.name == "help"));
        });
    }

    #[test]
    fn test_goal_command_parse() {
        assert_eq!(parse_goal_command_argument(""), GoalCommandArgument::Show);
        assert_eq!(
            parse_goal_command_argument("  \t"),
            GoalCommandArgument::Show
        );
        for clear_argument in ["clear", "off", "none"] {
            assert_eq!(
                parse_goal_command_argument(clear_argument),
                GoalCommandArgument::Clear
            );
        }
        assert_eq!(
            parse_goal_command_argument("  ship the native commands  "),
            GoalCommandArgument::Set("ship the native commands")
        );
        assert_eq!(
            parse_goal_command_argument("clear after validation"),
            GoalCommandArgument::Set("clear after validation")
        );
    }

    #[test]
    fn test_grind_command_parse() {
        assert_eq!(parse_grind_max_turns("").unwrap(), DEFAULT_GRIND_MAX_TURNS);
        assert_eq!(
            parse_grind_max_turns(" \t").unwrap(),
            DEFAULT_GRIND_MAX_TURNS
        );
        assert_eq!(parse_grind_max_turns("max_turns=1").unwrap(), 1);
        assert_eq!(
            parse_grind_max_turns("max_turns=20").unwrap(),
            HARD_MAX_GRIND_TURNS
        );

        for invalid in [
            "max_turns=0",
            "max_turns=21",
            "max_turns=-1",
            "max_turns=abc",
            "max_turns=",
            "max_turns=1 extra",
            "other=5",
            "5",
            "max_turns=999999999999999999999999",
        ] {
            assert!(
                parse_grind_max_turns(invalid).is_err(),
                "{invalid:?} should be rejected"
            );
        }
    }

    #[gpui::test]
    async fn test_goal_command_set_show_replace_and_clear(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));

        send_native_command_for_test(&connection, &session_id, "/goal ship goal support", cx)
            .await
            .unwrap();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.goal().cloned()),
            Some("ship goal support".into())
        );

        send_native_command_for_test(&connection, &session_id, "/goal", cx)
            .await
            .unwrap();
        acp_thread.read_with(cx, |thread, cx| {
            assert!(
                thread
                    .entries()
                    .last()
                    .unwrap()
                    .to_markdown(cx)
                    .contains("Current goal: ship goal support")
            );
        });

        send_native_command_for_test(&connection, &session_id, "/goal clear after validation", cx)
            .await
            .unwrap();
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.goal().cloned()),
            Some("clear after validation".into())
        );

        for clear_alias in ["clear", "off", "none"] {
            send_native_command_for_test(
                &connection,
                &session_id,
                &format!("/goal before {clear_alias}"),
                cx,
            )
            .await
            .unwrap();
            send_native_command_for_test(
                &connection,
                &session_id,
                &format!("/goal {clear_alias}"),
                cx,
            )
            .await
            .unwrap();
            assert!(thread.read_with(cx, |thread, _| thread.goal().is_none()));
        }

        assert_eq!(model.completion_count(), 0);
        let snapshot = thread.update(cx, |thread, cx| thread.to_db(cx)).await;
        assert!(snapshot.messages.is_empty());
        assert!(snapshot.goal.is_none());
    }

    #[gpui::test]
    async fn test_native_local_command_output_does_not_reenter_acp_thread(cx: &mut TestAppContext) {
        init_test(cx);
        let (_connection, _agent, _project, acp_thread) = setup_native_agent_session(cx).await;

        for (command, expected_output) in [
            ("/status", "## Status"),
            ("/skills", "## Available skills"),
            ("/goal", "No goal set"),
            ("/grind", "Cannot start grind"),
        ] {
            let response =
                send_native_command_through_acp_thread_for_test(&acp_thread, command, cx)
                    .await
                    .unwrap()
                    .expect("native command should produce a response");

            assert_eq!(response.stop_reason, acp::StopReason::EndTurn);
            acp_thread.read_with(cx, |thread, cx| {
                assert!(
                    thread
                        .entries()
                        .last()
                        .expect("local command output should be appended")
                        .to_markdown(cx)
                        .contains(expected_output),
                    "{command} should append its local output"
                );
            });
        }
    }

    #[gpui::test]
    async fn test_goal_persistence_across_reload(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        send_native_command_for_test(&connection, &session_id, "/goal persist through reload", cx)
            .await
            .unwrap();
        let database = cx.update(|cx| ThreadsDatabase::connect(cx)).await.unwrap();
        let stored = database
            .load_thread(session_id.clone())
            .await
            .unwrap()
            .expect("goal-only session should be persisted");
        assert_eq!(stored.goal.as_deref(), Some("persist through reload"));

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(acp_thread);
        let restored_acp_thread = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    None,
                    cx,
                )
            })
            .await
            .unwrap();
        let restored_thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        assert_eq!(
            restored_thread.read_with(cx, |thread, _| thread.goal().cloned()),
            Some("persist through reload".into())
        );

        send_native_command_for_test(&connection, &session_id, "/goal clear", cx)
            .await
            .unwrap();
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(restored_acp_thread);
        let cleared = database
            .load_thread(session_id)
            .await
            .unwrap()
            .expect("cleared goal-only session should retain an explicit database row");
        assert!(cleared.goal.is_none());
    }

    #[gpui::test]
    async fn test_goal_command_storage_failure_preserves_live_goal(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| ThreadsDatabase::fail_connections_for_test(cx, "save unavailable"));
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));

        let error = send_native_command_for_test(
            &connection,
            &session_id,
            "/goal must not be acknowledged",
            cx,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("save unavailable"));
        assert!(thread.read_with(cx, |thread, _| thread.goal().is_none()));
        assert!(acp_thread.read_with(cx, |thread, _| thread.entries().is_empty()));
    }

    #[gpui::test]
    async fn test_grind_success_stops_after_satisfaction(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        send_native_command_for_test(
            &connection,
            &session_id,
            "/goal finish the focused tests",
            cx,
        )
        .await
        .unwrap();

        let grind = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec!["/grind".into()]),
                cx,
            )
        });
        cx.run_until_parked();
        let first_request = model.pending_completions().pop().unwrap();
        assert!(
            first_request
                .messages
                .last()
                .unwrap()
                .string_contents()
                .contains("turn 1/5")
        );
        model.send_completion_stream_event(
            &first_request,
            LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                id: "goal-satisfied".into(),
                name: GRIND_SATISFACTION_TOOL_NAME.into(),
                raw_input: json!({"summary": "The focused tests pass"}).to_string(),
                input: language_model::LanguageModelToolUseInput::Json(
                    json!({"summary": "The focused tests pass"}),
                ),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        model.end_completion_stream(&first_request);
        cx.run_until_parked();

        let final_request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_text_chunk(&final_request, "All requested work is done.");
        model.end_completion_stream(&final_request);
        cx.run_until_parked();
        let response = grind.await.unwrap();
        assert_eq!(response.stop_reason, acp::StopReason::EndTurn);
        assert!(model.pending_completions().is_empty());
        agent.read_with(cx, |agent, _| {
            assert!(
                agent.sessions[&session_id].active_grind.is_none(),
                "satisfaction should clean up active grind state"
            );
        });
        acp_thread.read_with(cx, |thread, cx| {
            assert!(
                thread
                    .entries()
                    .last()
                    .unwrap()
                    .to_markdown(cx)
                    .contains("goal satisfied after 1/5 turns")
            );
        });
    }

    #[gpui::test]
    async fn test_grind_command_rejects_missing_goal_and_invalid_arguments(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));

        send_native_command_for_test(&connection, &session_id, "/grind", cx)
            .await
            .unwrap();
        acp_thread.read_with(cx, |thread, cx| {
            assert!(
                thread
                    .entries()
                    .last()
                    .unwrap()
                    .to_markdown(cx)
                    .contains("No goal is set")
            );
        });

        send_native_command_for_test(&connection, &session_id, "/goal bounded work", cx)
            .await
            .unwrap();
        for command in [
            "/grind max_turns=0",
            "/grind max_turns=21",
            "/grind max_turns=abc",
            "/grind max_turns=1 extra",
            "/grind unexpected=5",
        ] {
            send_native_command_for_test(&connection, &session_id, command, cx)
                .await
                .unwrap();
            assert!(acp_thread.read_with(cx, |thread, cx| {
                thread
                    .entries()
                    .last()
                    .unwrap()
                    .to_markdown(cx)
                    .contains("Cannot start grind")
            }));
        }
        assert!(model.pending_completions().is_empty());
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
    }

    #[gpui::test]
    async fn test_grind_turn_bound_enforces_default_and_hard_maximum(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        send_native_command_for_test(&connection, &session_id, "/goal keep going", cx)
            .await
            .unwrap();

        for (command, expected_turns) in [
            ("/grind", DEFAULT_GRIND_MAX_TURNS),
            ("/grind max_turns=20", HARD_MAX_GRIND_TURNS),
        ] {
            let grind = cx.update(|cx| {
                acp_thread::AgentSessionClientUserMessageIds::prompt(
                    connection.as_ref(),
                    ClientUserMessageId::new(),
                    acp::PromptRequest::new(session_id.clone(), vec![command.into()]),
                    cx,
                )
            });
            for turn_number in 1..=expected_turns {
                cx.run_until_parked();
                let pending = model.pending_completions();
                assert_eq!(
                    pending.len(),
                    1,
                    "expected one request for turn {turn_number}"
                );
                let request = pending.last().unwrap();
                assert!(
                    request
                        .messages
                        .last()
                        .unwrap()
                        .string_contents()
                        .contains(&format!("turn {turn_number}/{expected_turns}"))
                );
                model.end_completion_stream(request);
            }
            cx.run_until_parked();
            grind.await.unwrap();
            assert!(model.pending_completions().is_empty());
            agent.read_with(cx, |agent, _| {
                assert!(agent.sessions[&session_id].active_grind.is_none());
            });
            acp_thread.read_with(cx, |thread, cx| {
                assert!(
                    thread
                        .entries()
                        .last()
                        .unwrap()
                        .to_markdown(cx)
                        .contains(&format!("turn limit ({expected_turns}/{expected_turns})"))
                );
            });
        }
    }

    #[gpui::test]
    async fn test_concurrent_grind_and_goal_mutation_are_rejected(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        send_native_command_for_test(&connection, &session_id, "/goal original goal", cx)
            .await
            .unwrap();

        let first_grind = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec!["/grind max_turns=1".into()]),
                cx,
            )
        });
        cx.run_until_parked();
        assert_eq!(model.pending_completions().len(), 1);

        send_native_command_for_test(&connection, &session_id, "/grind", cx)
            .await
            .unwrap();
        send_native_command_for_test(&connection, &session_id, "/goal replacement", cx)
            .await
            .unwrap();
        assert_eq!(model.pending_completions().len(), 1);
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.goal().cloned()),
            Some("original goal".into())
        );
        let status = agent
            .read_with(cx, |agent, cx| agent.status_command_output(&session_id, cx))
            .unwrap();
        assert!(status.contains("- Goal: original goal"));
        assert!(status.contains("- Grind: active (0/1 turns completed)"));
        acp_thread.read_with(cx, |thread, cx| {
            let outputs = thread
                .entries()
                .iter()
                .map(|entry| entry.to_markdown(cx))
                .collect::<Vec<_>>();
            assert!(
                outputs
                    .iter()
                    .any(|output| output.contains("A grind is already active"))
            );
            assert!(outputs.iter().any(|output| {
                output.contains("goal cannot be changed while a grind is active")
            }));
        });

        let request = model.pending_completions().pop().unwrap();
        model.end_completion_stream(&request);
        cx.run_until_parked();
        first_grind.await.unwrap();
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
    }

    #[gpui::test]
    async fn test_grind_cancel_at_provider_tool_and_continuation_boundaries(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.add_tool(GrindPendingTool);
        });
        enable_test_tools_for_thread(&thread, &[GrindPendingTool::NAME], cx);
        send_native_command_for_test(&connection, &session_id, "/goal cancel safely", cx)
            .await
            .unwrap();

        let provider_grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let provider_request = model.pending_completions().pop().unwrap();
        cx.update(|cx| connection.cancel(&session_id, cx));
        model.end_completion_stream(&provider_request);
        cx.run_until_parked();
        assert_eq!(
            provider_grind.await.unwrap().stop_reason,
            acp::StopReason::Cancelled
        );
        assert!(
            model.pending_completions().is_empty(),
            "provider cancellation left a pending request"
        );

        let tool_grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let tool_request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_event(
            &tool_request,
            LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                id: "pending-tool".into(),
                name: GrindPendingTool::NAME.into(),
                raw_input: "{}".to_string(),
                input: language_model::LanguageModelToolUseInput::Json(json!({})),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        model.end_completion_stream(&tool_request);
        cx.run_until_parked();
        assert!(
            model.pending_completions().is_empty(),
            "pending tool unexpectedly completed before cancellation"
        );
        cx.update(|cx| connection.cancel(&session_id, cx));
        cx.run_until_parked();
        assert_eq!(
            tool_grind.await.unwrap().stop_reason,
            acp::StopReason::Cancelled
        );
        assert!(
            model.pending_completions().is_empty(),
            "tool cancellation left a pending request"
        );

        let boundary_grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let completed_request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_event(
            &completed_request,
            LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
        );
        model.end_completion_stream(&completed_request);
        cx.update(|cx| connection.cancel(&session_id, cx));
        cx.run_until_parked();
        assert_eq!(
            boundary_grind.await.unwrap().stop_reason,
            acp::StopReason::Cancelled
        );
        assert!(
            model.pending_completions().is_empty(),
            "boundary cancellation started another request"
        );
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
    }

    #[gpui::test]
    async fn test_grind_provider_failure_stops_without_continuation(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        send_native_command_for_test(
            &connection,
            &session_id,
            "/goal stop on provider failure",
            cx,
        )
        .await
        .unwrap();

        for (model_stop_reason, acp_stop_reason) in [
            (LanguageModelStopReason::Refusal, acp::StopReason::Refusal),
            (
                LanguageModelStopReason::MaxTokens,
                acp::StopReason::MaxTokens,
            ),
        ] {
            let grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
            cx.run_until_parked();
            let request = model.pending_completions().pop().unwrap();
            model.send_completion_stream_event(
                &request,
                LanguageModelCompletionEvent::Stop(model_stop_reason),
            );
            model.end_completion_stream(&request);
            cx.run_until_parked();
            assert_eq!(grind.await.unwrap().stop_reason, acp_stop_reason);
            assert!(model.pending_completions().is_empty());
        }

        let failed_grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let failed_request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_error(
            &failed_request,
            LanguageModelCompletionError::PaymentRequired,
        );
        model.end_completion_stream(&failed_request);
        cx.run_until_parked();
        assert!(failed_grind.await.is_err());
        assert!(model.pending_completions().is_empty());
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
    }

    #[gpui::test]
    async fn test_grind_tool_failure_stops_and_persists_action_record(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        send_native_command_for_test(&connection, &session_id, "/goal preserve failed tools", cx)
            .await
            .unwrap();

        let grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                id: "failed-tool".into(),
                name: "missing_grind_tool".into(),
                raw_input: "{}".to_string(),
                input: language_model::LanguageModelToolUseInput::Json(json!({})),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        assert_eq!(grind.await.unwrap().stop_reason, acp::StopReason::Cancelled);
        assert!(model.pending_completions().is_empty());

        let snapshot = thread.update(cx, |thread, cx| thread.to_db(cx)).await;
        let serialized = serde_json::to_string(&snapshot.messages).unwrap();
        assert!(serialized.contains("missing_grind_tool"));
        assert!(serialized.contains("failed-tool"));

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        let database = cx.update(|cx| ThreadsDatabase::connect(cx)).await.unwrap();
        let stored = database
            .load_thread(session_id)
            .await
            .unwrap()
            .expect("failed tool turn should be persisted on final close");
        let stored_serialized = serde_json::to_string(&stored.messages).unwrap();
        assert!(stored_serialized.contains("missing_grind_tool"));
        assert!(stored_serialized.contains("failed-tool"));
    }

    #[gpui::test]
    async fn test_grind_permission_and_elicitation_stop_without_restart(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            cx.update_flags(false, vec![AcpBetaFeatureFlag::NAME.to_string()]);
        });
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.add_tool(GrindPermissionTool);
        });
        enable_test_tools_for_thread(&thread, &[GrindPermissionTool::NAME], cx);
        send_native_command_for_test(&connection, &session_id, "/goal stop for attention", cx)
            .await
            .unwrap();

        let permission_grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let permission_request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_event(
            &permission_request,
            LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                id: "permission-tool".into(),
                name: GrindPermissionTool::NAME.into(),
                raw_input: "{}".to_string(),
                input: language_model::LanguageModelToolUseInput::Json(json!({})),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        model.end_completion_stream(&permission_request);
        cx.run_until_parked();
        assert_eq!(
            permission_grind.await.unwrap().stop_reason,
            acp::StopReason::Cancelled
        );
        acp_thread.update(cx, |thread, cx| {
            thread.authorize_tool_call(
                "permission-tool".into(),
                acp_thread::SelectedPermissionOutcome::new(
                    "allow".into(),
                    acp::PermissionOptionKind::AllowOnce,
                ),
                cx,
            );
        });
        cx.run_until_parked();
        assert!(model.pending_completions().is_empty());

        let elicitation_grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let elicitation_request = model.pending_completions().pop().unwrap();
        let (elicitation_id, elicitation_response) = acp_thread
            .update(cx, |thread, cx| {
                thread.request_elicitation_with_id(
                    acp::CreateElicitationRequest::new(
                        acp::ElicitationFormMode::new(
                            acp::ElicitationSessionScope::new(session_id.clone()),
                            acp::ElicitationSchema::new().string("answer", true),
                        ),
                        "User attention is required",
                    ),
                    cx,
                )
            })
            .unwrap();
        model.end_completion_stream(&elicitation_request);
        cx.run_until_parked();
        assert_eq!(
            elicitation_grind.await.unwrap().stop_reason,
            acp::StopReason::Cancelled
        );
        acp_thread.update(cx, |thread, cx| {
            thread.respond_to_elicitation(
                &elicitation_id,
                acp::CreateElicitationResponse::new(acp::ElicitationAction::Cancel),
                cx,
            );
        });
        elicitation_response.await;
        cx.run_until_parked();
        assert!(model.pending_completions().is_empty());
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
    }

    #[gpui::test]
    async fn test_close_session_saves_thread_grind_session_close_and_reloads_inactive(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        send_native_command_for_test(&connection, &session_id, "/goal survive reload", cx)
            .await
            .unwrap();

        let grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_text_chunk(&request, "Persist this partial response.");
        cx.run_until_parked();
        let close = cx.update(|cx| connection.clone().close_session(&session_id, cx));
        model.end_completion_stream(&request);
        close.await.unwrap();
        assert_eq!(grind.await.unwrap().stop_reason, acp::StopReason::Cancelled);
        assert!(!agent.read_with(cx, |agent, _| agent.sessions.contains_key(&session_id)));

        let database = cx.update(|cx| ThreadsDatabase::connect(cx)).await.unwrap();
        let stored = database
            .load_thread(session_id.clone())
            .await
            .unwrap()
            .expect("closed grind should persist a final snapshot");
        assert_eq!(stored.goal.as_deref(), Some("survive reload"));
        assert!(
            serde_json::to_string(&stored.messages)
                .unwrap()
                .contains("Persist this partial response")
        );

        drop(acp_thread);
        let reopened = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[Path::new("/a")]),
                    None,
                    cx,
                )
            })
            .await
            .unwrap();
        let reopened_thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        assert_eq!(
            reopened_thread.read_with(cx, |thread, _| thread.goal().cloned()),
            Some("survive reload".into())
        );
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
        assert!(model.pending_completions().is_empty());
        drop(reopened);
    }

    #[gpui::test]
    async fn test_goal_grind_reload_excludes_transient_commands_and_stays_inactive(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));

        send_native_command_for_test(&connection, &session_id, "/goal persistent objective", cx)
            .await
            .unwrap();
        send_native_command_for_test(&connection, &session_id, "/goal", cx)
            .await
            .unwrap();
        send_native_command_for_test(&connection, &session_id, "/status", cx)
            .await
            .unwrap();
        assert!(
            thread
                .update(cx, |thread, cx| thread.to_db(cx))
                .await
                .messages
                .is_empty()
        );

        let grind = start_grind_for_test(&connection, &session_id, "/grind max_turns=1", cx);
        cx.run_until_parked();
        let grind_request = model.pending_completions().pop().unwrap();
        assert!(
            grind_request
                .messages
                .last()
                .unwrap()
                .string_contents()
                .contains("<grind_control>")
        );
        model.end_completion_stream(&grind_request);
        cx.run_until_parked();
        grind.await.unwrap();

        let ordinary_turn = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(
                    session_id.clone(),
                    vec!["inspect persisted context".into()],
                ),
                cx,
            )
        });
        cx.run_until_parked();
        let ordinary_request = model.pending_completions().pop().unwrap();
        let model_context = ordinary_request
            .messages
            .iter()
            .map(language_model::LanguageModelRequestMessage::string_contents)
            .join("\n");
        for transient_text in [
            "Goal set: persistent objective",
            "Current goal: persistent objective",
            "## Status",
            "Grind started.",
            "Grind stopped at the turn limit",
            "<grind_control>",
        ] {
            assert!(
                !model_context.contains(transient_text),
                "transient command output leaked into model context: {transient_text}"
            );
        }
        model.end_completion_stream(&ordinary_request);
        cx.run_until_parked();
        ordinary_turn.await.unwrap();

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(acp_thread);
        let reopened = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[Path::new("/a")]),
                    None,
                    cx,
                )
            })
            .await
            .unwrap();
        let reopened_thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        assert_eq!(
            reopened_thread.read_with(cx, |thread, _| thread.goal().cloned()),
            Some("persistent objective".into())
        );
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
        let status = agent
            .read_with(cx, |agent, cx| agent.status_command_output(&session_id, cx))
            .unwrap();
        assert!(status.contains("- Goal: persistent objective"));
        assert!(status.contains("- Grind: inactive"));
        assert!(model.pending_completions().is_empty());
        drop(reopened);
    }

    #[gpui::test]
    async fn test_skills_command_uses_catalog_invocations_without_model_turn(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());

        cx.update(|cx| {
            let thread = native_thread_for_session(&agent, &session_id, cx);
            thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
            agent.update(cx, |agent, _cx| {
                let project_state = agent.session_project_state(&session_id).unwrap();
                let project_id = project_state.project.entity_id();
                agent.projects.get_mut(&project_id).unwrap().skills = Arc::new(vec![
                    make_global_skill("deploy", "Deploy globally"),
                    make_project_skill("deploy", "Deploy this worktree", "a"),
                    make_global_skill("status", "Inspect service status"),
                    make_global_skill("lint", "Run lint checks"),
                ]);
            });
        });

        let response = cx
            .update(|cx| {
                acp_thread::AgentSessionClientUserMessageIds::prompt(
                    connection.as_ref(),
                    ClientUserMessageId::new(),
                    acp::PromptRequest::new(session_id.clone(), vec!["/skills".into()]),
                    cx,
                )
            })
            .await
            .unwrap();
        assert_eq!(response.stop_reason, acp::StopReason::EndTurn);
        assert_eq!(model.completion_count(), 0);

        acp_thread.read_with(cx, |thread, cx| {
            let [entry] = thread.entries() else {
                panic!("expected one local command entry")
            };
            let output = entry.to_markdown(cx);
            assert!(output.contains("`/:deploy` — Deploy globally (source: global)"));
            assert!(output.contains("`/a:deploy` — Deploy this worktree (source: a)"));
            assert!(output.contains("`/:status` — Inspect service status (source: global)"));
            assert!(output.contains("`/lint` — Run lint checks (source: global)"));
        });
    }

    #[gpui::test]
    async fn test_status_command_formats_session_snapshot_without_model_turn(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        cx.run_until_parked();
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());

        cx.update(|cx| {
            let thread = native_thread_for_session(&agent, &session_id, cx);
            thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
            let project_id = project.entity_id();
            agent.update(cx, |agent, _cx| {
                agent
                    .projects
                    .get_mut(&project_id)
                    .unwrap()
                    .skill_loading_issues = vec![SkillLoadingIssue {
                    project_id,
                    path: PathBuf::from("broken/SKILL.md"),
                    source_label: None,
                    relative_path: None,
                    message: "invalid frontmatter".into(),
                    kind: SkillLoadingIssueKind::LoadFailed,
                }];
            });
        });

        let previous_turn = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec!["previous message".into()]),
                cx,
            )
        });
        cx.run_until_parked();
        model.send_last_completion_stream_text_chunk("previous response");
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::UsageUpdate(
            language_model::TokenUsage {
                input_tokens: 7,
                output_tokens: 5,
                ..Default::default()
            },
        ));
        model.end_last_completion_stream();
        cx.run_until_parked();
        previous_turn.await.unwrap();

        cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id, vec!["/status".into()]),
                cx,
            )
        })
        .await
        .unwrap();
        assert_eq!(model.completion_count(), 0);

        acp_thread.read_with(cx, |thread, cx| {
            let entry = thread
                .entries()
                .last()
                .expect("missing local command entry");
            let output = entry.to_markdown(cx);
            assert!(output.contains("- Model: Fake (fake)"));
            assert!(output.contains("- Provider: Fake (fake)"));
            assert!(output.contains("- Context: 12 / 1000000 tokens"));
            assert!(output.contains("- Personal instructions: not loaded"));
            assert!(output.contains("- Developer context issues: 1"));
            assert!(output.contains("- a — instructions: no project instructions"));
        });
    }

    #[gpui::test]
    async fn test_status_command_labels_unavailable_fields(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            LanguageModelRegistry::global(cx)
                .update(cx, |registry, cx| registry.set_default_model(None, cx));
        });
        let (connection, _agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());

        cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id, vec!["/status".into()]),
                cx,
            )
        })
        .await
        .unwrap();

        acp_thread.read_with(cx, |thread, cx| {
            let output = thread.entries().last().unwrap().to_markdown(cx);
            assert!(output.contains("- Model: unavailable"));
            assert!(output.contains("- Provider: unavailable"));
            assert!(output.contains("- Context: unavailable"));
        });
    }

    #[gpui::test]
    async fn test_compact_prompt_routes_to_manual_compaction(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        let model = Arc::new(FakeLanguageModel::default());
        let old_message_id = ClientUserMessageId::new();

        cx.update(|cx| {
            let path_style = project.read(cx).path_style(cx);
            thread.update(cx, |thread, cx| {
                thread.set_model(model.clone(), cx);
                thread.push_acp_user_block(
                    old_message_id,
                    [acp::ContentBlock::from("old user")],
                    path_style,
                    cx,
                );
                thread.push_acp_agent_block("old assistant".into(), cx);
            });
        });

        let compact_message_id = ClientUserMessageId::new();
        let prompt_task = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                compact_message_id,
                acp::PromptRequest::new(session_id.clone(), vec!["/compact".into()]),
                cx,
            )
        });
        cx.run_until_parked();

        let request = model.pending_completions().pop().unwrap();
        assert_eq!(
            request.intent,
            Some(CompletionIntent::ThreadContextSummarization)
        );
        assert_eq!(
            request_texts_after_system(&request.messages),
            vec![
                "old user".to_string(),
                "old assistant".to_string(),
                COMPACTION_PROMPT.to_string(),
            ]
        );

        model.send_completion_stream_text_chunk(&request, "summary");
        model.end_completion_stream(&request);
        cx.run_until_parked();
        prompt_task.await.unwrap();
    }

    #[gpui::test]
    async fn test_clear_command_clear_conversation_preserves_goal_and_metadata(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        let model = Arc::new(FakeLanguageModel::default());
        let action_log_id = thread.read_with(cx, |thread, _| thread.action_log().entity_id());

        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.set_title("Keep this title".into(), cx);
        });
        send_native_command_for_test(&connection, &session_id, "/goal keep this objective", cx)
            .await
            .unwrap();
        let first_turn = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec!["old user".into()]),
                cx,
            )
        });
        cx.run_until_parked();
        model.send_last_completion_stream_text_chunk("old assistant");
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::UsageUpdate(
            language_model::TokenUsage {
                input_tokens: 9,
                output_tokens: 4,
                ..Default::default()
            },
        ));
        model.end_last_completion_stream();
        cx.run_until_parked();
        first_turn.await.unwrap();

        let response = cx
            .update(|cx| {
                acp_thread::AgentSessionClientUserMessageIds::prompt(
                    connection.as_ref(),
                    ClientUserMessageId::new(),
                    acp::PromptRequest::new(session_id.clone(), vec!["/clear".into()]),
                    cx,
                )
            })
            .await
            .unwrap();
        assert_eq!(response.stop_reason, acp::StopReason::EndTurn);
        assert_eq!(model.completion_count(), 0);

        let database = cx.update(|cx| ThreadsDatabase::connect(cx)).await.unwrap();
        let restored = database
            .load_thread(session_id.clone())
            .await
            .unwrap()
            .expect("cleared thread should remain persisted");
        assert!(restored.messages.is_empty());
        assert!(restored.detailed_summary.is_none());
        assert_eq!(restored.cumulative_token_usage, Default::default());
        assert!(restored.request_token_usage.is_empty());
        assert_eq!(restored.goal.as_deref(), Some("keep this objective"));
        assert_eq!(restored.title.as_ref(), "Keep this title");
        assert!(restored.model.is_some());
        assert!(restored.profile.is_some());

        let live_snapshot = thread.update(cx, |thread, cx| thread.to_db(cx)).await;
        assert!(live_snapshot.messages.is_empty());
        assert_eq!(live_snapshot.title.as_ref(), "Keep this title");
        assert_eq!(live_snapshot.goal.as_deref(), Some("keep this objective"));
        assert_eq!(live_snapshot.cumulative_token_usage, Default::default());
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.action_log().entity_id()),
            action_log_id
        );
        acp_thread.read_with(cx, |thread, cx| {
            let [confirmation] = thread.entries() else {
                panic!("expected only the transient clear confirmation")
            };
            assert!(
                confirmation
                    .to_markdown(cx)
                    .contains("Conversation cleared.")
            );
            assert!(thread.token_usage().is_none());
            assert_eq!(thread.action_log().entity_id(), action_log_id);
        });

        let next_turn = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id, vec!["new user".into()]),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().unwrap();
        assert_eq!(
            request_texts_after_system(&request.messages),
            vec!["new user"]
        );
        model.send_completion_stream_text_chunk(&request, "new assistant");
        model.end_completion_stream(&request);
        cx.run_until_parked();
        next_turn.await.unwrap();
    }

    #[gpui::test]
    async fn test_clear_command_storage_failure_preserves_live_conversation(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| ThreadsDatabase::fail_connections_for_test(cx, "save unavailable"));
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));

        cx.update(|cx| {
            let path_style = project.read(cx).path_style(cx);
            thread.update(cx, |thread, cx| {
                thread.push_acp_user_block(
                    ClientUserMessageId::new(),
                    [acp::ContentBlock::from("keep me")],
                    path_style,
                    cx,
                );
            });
            acp_thread.update(cx, |thread, cx| {
                thread.push_user_content_block(None, "keep me".into(), cx);
            });
        });

        let error = cx
            .update(|cx| {
                acp_thread::AgentSessionClientUserMessageIds::prompt(
                    connection.as_ref(),
                    ClientUserMessageId::new(),
                    acp::PromptRequest::new(session_id, vec!["/clear".into()]),
                    cx,
                )
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("save unavailable"));
        assert_eq!(
            thread
                .update(cx, |thread, cx| thread.to_db(cx))
                .await
                .messages
                .len(),
            1
        );
        acp_thread.read_with(cx, |thread, _| assert_eq!(thread.entries().len(), 1));
    }

    #[gpui::test]
    async fn test_clear_command_reports_session_disappearance_after_persistence(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        cx.update(|cx| {
            let path_style = project.read(cx).path_style(cx);
            thread.update(cx, |thread, cx| {
                thread.push_acp_user_block(
                    ClientUserMessageId::new(),
                    [acp::ContentBlock::from("persisted before clear")],
                    path_style,
                    cx,
                );
            });
        });

        let clear_task = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec!["/clear".into()]),
                cx,
            )
        });
        agent.update(cx, |agent, _cx| {
            agent.sessions.remove(&session_id);
        });

        let error = clear_task.await.unwrap_err();
        assert!(error.to_string().contains("Session disappeared"));
        let database = cx.update(|cx| ThreadsDatabase::connect(cx)).await.unwrap();
        let restored = database
            .load_thread(session_id)
            .await
            .unwrap()
            .expect("the cleared row should have been saved before the session disappeared");
        assert!(restored.messages.is_empty());
        assert_eq!(
            thread
                .update(cx, |thread, cx| thread.to_db(cx))
                .await
                .messages
                .len(),
            1
        );
    }

    #[gpui::test]
    async fn test_threads_flushed_to_database_on_app_quit(cx: &mut TestAppContext) {
        init_test(cx);

        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));

        // A second session whose thread stays empty must be skipped by the
        // quit flush rather than persisted as an empty row.
        let empty_acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let empty_session_id = cx.update(|cx| empty_acp_thread.read(cx).session_id().clone());

        // Give the first thread content so it's no longer an empty draft, plus
        // an in-progress draft prompt that the flush must capture.
        cx.update(|cx| {
            let path_style = project.read(cx).path_style(cx);
            thread.update(cx, |thread, cx| {
                thread.push_acp_user_block(
                    ClientUserMessageId::new(),
                    [acp::ContentBlock::from("hello from the user")],
                    path_style,
                    cx,
                );
            });
            acp_thread.update(cx, |acp_thread, cx| {
                acp_thread
                    .set_draft_prompt(Some(vec![acp::ContentBlock::from("draft in progress")]), cx);
            });
        });
        cx.run_until_parked();

        // Reproduce the orphaned state from the bug: the sidebar metadata and
        // serialized panel still reference the session, but the per-session
        // async content save never landed, so the content row is absent.
        let database = cx.update(|cx| ThreadsDatabase::connect(cx)).await.unwrap();
        database.delete_thread(session_id.clone()).await.unwrap();
        assert!(
            database
                .load_thread(session_id.clone())
                .await
                .unwrap()
                .is_none(),
            "precondition: content row should be missing before the quit flush"
        );

        // Quit through the real shutdown path so the `on_app_quit`
        // registration is exercised, not just the flush itself.
        cx.update(|cx| cx.shutdown());

        let restored = database
            .load_thread(session_id.clone())
            .await
            .unwrap()
            .expect("thread content should be persisted to the database on quit");
        assert_eq!(
            restored.messages.len(),
            1,
            "the user message should survive the quit flush"
        );
        assert_eq!(
            restored.draft_prompt,
            Some(vec![acp::ContentBlock::from("draft in progress")]),
            "the current draft prompt should be captured by the quit flush"
        );
        assert!(
            database
                .load_thread(empty_session_id)
                .await
                .unwrap()
                .is_none(),
            "empty threads should not be persisted by the quit flush"
        );
    }

    #[test]
    fn test_ambiguous_mcp_prompt_names() {
        let ambiguous = ambiguous_mcp_prompt_names(
            NATIVE_COMMANDS.iter().map(|definition| definition.name),
            [
                "compact", "clear", "skills", "status", "goal", "grind", "deploy",
            ],
        );
        for name in ["compact", "clear", "skills", "status", "goal", "grind"] {
            assert!(ambiguous.contains(name));
        }
        assert!(!ambiguous.contains("deploy"));

        // Without the reservation, a unique MCP prompt is left bare.
        let ambiguous = ambiguous_mcp_prompt_names([], ["compact", "deploy"]);
        assert!(ambiguous.is_empty());

        // Two MCP prompts sharing a name are both qualified regardless of
        // reservation.
        let ambiguous = ambiguous_mcp_prompt_names([], ["dup", "dup", "unique"]);
        assert!(ambiguous.contains("dup"));
        assert!(!ambiguous.contains("unique"));
    }

    #[test]
    fn test_mcp_prompt_command_input_metadata() {
        assert_eq!(mcp_prompt_input_hint(None), None);
        assert_eq!(mcp_prompt_input_hint(Some(&[])), None);

        let arguments = [
            mcp_argument("query", Some("Search query"), true),
            mcp_argument("format", Some("Output format"), false),
            mcp_argument("empty", None, false),
        ];
        assert_eq!(
            mcp_prompt_input_hint(Some(&arguments)).as_deref(),
            Some(
                "<query=value> [format=value] [empty=value] — query (required): Search query; format (optional): Output format; empty (optional)"
            )
        );
    }

    #[test]
    fn test_mcp_prompt_command_argument_validation() {
        let required = mcp_argument("query", None, true);
        let optional = mcp_argument("format", None, false);

        assert!(parse_mcp_prompt_arguments(None, "").unwrap().is_empty());
        assert!(parse_mcp_prompt_arguments(None, "unexpected").is_err());
        assert_eq!(
            parse_mcp_prompt_arguments(Some(std::slice::from_ref(&required)), "plain remainder")
                .unwrap(),
            HashMap::from_iter([("query".to_string(), "plain remainder".to_string())])
        );
        assert_eq!(
            parse_mcp_prompt_arguments(
                Some(std::slice::from_ref(&required)),
                "query=\"quoted value\""
            )
            .unwrap(),
            HashMap::from_iter([("query".to_string(), "quoted value".to_string())])
        );
        assert_eq!(
            parse_mcp_prompt_arguments(Some(&[required, optional]), "query='two words' format=")
                .unwrap(),
            HashMap::from_iter([
                ("query".to_string(), "two words".to_string()),
                ("format".to_string(), String::new()),
            ])
        );

        let arguments = [
            mcp_argument("query", None, true),
            mcp_argument("format", None, false),
        ];
        for invalid in [
            "",
            "format=json",
            "query=one query=two",
            "query=ok unknown=value",
            "query=ok malformed",
            "query=\"unterminated",
        ] {
            assert!(
                parse_mcp_prompt_arguments(Some(&arguments), invalid).is_err(),
                "expected `{invalid}` to fail"
            );
        }
    }

    #[gpui::test]
    async fn test_available_commands_include_all_mcp_prompt_argument_shapes(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (_connection, _agent, project, acp_thread) = setup_native_agent_session(cx).await;
        start_test_mcp_prompt_server(
            &project,
            "test",
            vec![
                mcp_prompt("zero", "No arguments", None),
                mcp_prompt(
                    "single",
                    "One argument",
                    Some(vec![mcp_argument("query", Some("Search query"), true)]),
                ),
                mcp_prompt(
                    "optional",
                    "Optional argument",
                    Some(vec![mcp_argument("format", Some("Output format"), false)]),
                ),
                mcp_prompt(
                    "multi",
                    "Multiple arguments",
                    Some(vec![
                        mcp_argument("query", Some("Search query"), true),
                        mcp_argument("format", Some("Output format"), false),
                    ]),
                ),
                mcp_prompt("status", "Collides with native status", None),
            ],
            cx,
        )
        .await;

        acp_thread.read_with(cx, |thread, _cx| {
            let commands = thread.available_commands();
            for name in ["zero", "single", "optional", "multi", "test.status"] {
                assert!(
                    commands.iter().any(|command| command.name == name),
                    "missing command {name}: {commands:?}"
                );
            }
            assert!(!commands.iter().any(|command| command.name == "prompt"));
            assert!(!commands.iter().any(|command| command.name == "prompts"));
            assert!(
                commands
                    .iter()
                    .find(|command| command.name == "zero")
                    .unwrap()
                    .input
                    .is_none()
            );
            let multi = commands
                .iter()
                .find(|command| command.name == "multi")
                .unwrap();
            let Some(acp::AvailableCommandInput::Unstructured(input)) = &multi.input else {
                panic!("multi command should expose unstructured input metadata")
            };
            assert!(input.hint.contains("<query=value>"));
            assert!(input.hint.contains("[format=value]"));
            assert!(input.hint.contains("query (required): Search query"));
            assert!(input.hint.contains("format (optional): Output format"));
        });
    }

    #[gpui::test]
    async fn test_mcp_prompt_command_validates_before_server_and_model_calls(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        cx.update(|cx| {
            native_thread_for_session(&agent, &session_id, cx)
                .update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        });
        let (server, prompt_calls) = start_test_mcp_prompt_server(
            &project,
            "test",
            vec![
                mcp_prompt("zero", "No arguments", None),
                mcp_prompt(
                    "compose",
                    "Compose content",
                    Some(vec![
                        mcp_argument("title", Some("Title"), true),
                        mcp_argument("body", Some("Body"), true),
                        mcp_argument("empty", Some("Optional empty value"), false),
                    ]),
                ),
            ],
            cx,
        )
        .await;

        for invalid in [
            "/compose title=ok",
            "/compose title=one title=two body=ok",
            "/compose title=ok body=ok unknown=value",
            "/compose title=ok malformed body=ok",
            "/compose title=\"unterminated body=ok",
            "/zero unexpected",
        ] {
            let result = cx
                .update(|cx| {
                    acp_thread::AgentSessionClientUserMessageIds::prompt(
                        connection.as_ref(),
                        ClientUserMessageId::new(),
                        acp::PromptRequest::new(session_id.clone(), vec![invalid.into()]),
                        cx,
                    )
                })
                .await;
            assert!(result.is_err(), "expected `{invalid}` to fail");
        }
        assert!(prompt_calls.lock().unwrap().is_empty());
        assert_eq!(model.completion_count(), 0);

        let valid_prompt = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(
                    session_id.clone(),
                    vec!["/compose title=\"Hello world\" body='Line one two' empty=".into()],
                ),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().unwrap();
        let request_text = request_texts_after_system(&request.messages).join("\n");
        assert!(request_text.contains("generated prompt:"));
        assert!(request_text.contains("prepared response"));
        model.end_completion_stream(&request);
        cx.run_until_parked();
        valid_prompt.await.unwrap();

        {
            let calls = prompt_calls.lock().unwrap();
            let [(name, Some(arguments))] = calls.as_slice() else {
                panic!("unexpected prompt calls: {calls:?}")
            };
            assert_eq!(name, "compose");
            assert_eq!(
                arguments.get("title").map(String::as_str),
                Some("Hello world")
            );
            assert_eq!(
                arguments.get("body").map(String::as_str),
                Some("Line one two")
            );
            assert_eq!(arguments.get("empty").map(String::as_str), Some(""));
        }

        server.stop().unwrap();
        let result = cx
            .update(|cx| {
                acp_thread::AgentSessionClientUserMessageIds::prompt(
                    connection.as_ref(),
                    ClientUserMessageId::new(),
                    acp::PromptRequest::new(session_id, vec!["/compose title=ok body=ok".into()]),
                    cx,
                )
            })
            .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Context server not initialized")
        );
        assert_eq!(prompt_calls.lock().unwrap().len(), 1);
        assert_eq!(model.completion_count(), 0);
    }

    #[test]
    fn test_qualified_compact_commands_are_not_native_compact() {
        let unqualified_blocks = [acp::ContentBlock::from("/compact")];
        let unqualified = Command::parse(&unqualified_blocks).unwrap();
        assert!(unqualified.is_unqualified("compact"));

        let mcp_blocks = [acp::ContentBlock::from("/server.compact")];
        let mcp_qualified = Command::parse(&mcp_blocks).unwrap();
        assert_eq!(mcp_qualified.prompt_name, "compact");
        assert_eq!(mcp_qualified.explicit_server_id, Some("server"));
        assert!(!mcp_qualified.is_unqualified("compact"));

        let skill_blocks = [acp::ContentBlock::from("/:compact")];
        let skill_qualified = Command::parse(&skill_blocks).unwrap();
        assert_eq!(skill_qualified.prompt_name, "compact");
        assert_eq!(skill_qualified.skill_scope, Some(""));
        assert!(!skill_qualified.is_unqualified("compact"));
    }

    fn make_project_skill(name: &str, description: &str, worktree: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: description.to_string(),
            source: SkillSource::ProjectLocal {
                worktree_id: SkillScopeId(1),
                worktree_root_name: worktree.into(),
            },
            directory_path: PathBuf::from(format!("/{worktree}/.agents/skills/{name}")),
            skill_file_path: PathBuf::from(format!("/{worktree}/.agents/skills/{name}/SKILL.md")),
            load_warnings: Vec::new(),
            disable_model_invocation: false,
            embedded_body: None,
        }
    }

    fn make_builtin_skill(name: &str, description: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: description.to_string(),
            source: SkillSource::BuiltIn,
            directory_path: PathBuf::from(format!("/builtin/{name}")),
            skill_file_path: PathBuf::from(format!("/builtin/{name}/SKILL.md")),
            load_warnings: Vec::new(),
            disable_model_invocation: false,
            embedded_body: Some("built-in body"),
        }
    }

    /// Filter to only user-defined (non-built-in) skills for test assertions.
    fn user_skills(skills: &[Skill]) -> Vec<&Skill> {
        skills
            .iter()
            .filter(|s| !matches!(s.source, SkillSource::BuiltIn))
            .collect()
    }

    #[test]
    fn test_combine_skills_keeps_every_entry_for_autocomplete() {
        // The autocomplete popup needs both same-named entries so the
        // source label can disambiguate them. `combine_skills` must not
        // drop the global when a project-local shares its name.
        let global = make_global_skill("review", "Global review");
        let project = make_project_skill("review", "Project review", "project");

        let (skills, errors) = combine_skills(vec![Ok(global)], vec![Ok(project)].into_iter());

        assert!(errors.is_empty());
        let user = user_skills(&skills);
        assert_eq!(user.len(), 2);
        assert!(matches!(user[0].source, SkillSource::Global));
        assert!(matches!(user[1].source, SkillSource::ProjectLocal { .. }));
    }

    #[test]
    fn plugin_commands_use_existing_scope_parser() {
        let prompt = vec![acp::ContentBlock::Text(acp::TextContent::new(
            "/plugin/acme.tools:review inspect this",
        ))];
        let command = Command::parse(&prompt).expect("qualified skill command");
        assert_eq!(command.skill_scope, Some("plugin/acme.tools"));
        assert_eq!(command.prompt_name, "review");
        assert_eq!(command.arg_value, "inspect this");
        assert!(command.explicit_server_id.is_none());
    }

    #[test]
    fn plugin_skill_qualification_preserves_native_sources() {
        let content = "---\nname: review\ndescription: Review changes\n---\nInstructions";
        let make = |source| {
            agent_skills::parse_skill_frontmatter(
                Path::new("/skills/review/SKILL.md"),
                content,
                source,
            )
            .expect("valid skill")
        };
        let skills = vec![
            make(SkillSource::Global),
            make(SkillSource::Package {
                scope: "plugin/one".into(),
                root: Path::new("/one").into(),
            }),
            make(SkillSource::Package {
                scope: "plugin/two".into(),
                root: Path::new("/two").into(),
            }),
        ];
        let resolved = apply_skill_overrides(&skills);
        assert_eq!(
            resolved
                .iter()
                .map(|skill| skill.lookup_name().into_owned())
                .collect::<Vec<_>>(),
            ["review", "plugin/one:review", "plugin/two:review"]
        );
        assert!(
            skills
                .get(1)
                .expect("first package")
                .source
                .matches_scope("plugin/one")
        );
    }

    #[test]
    fn test_apply_skill_overrides_project_wins_over_global() {
        // The model-facing projection collapses the same name to a
        // single entry, with the project-local winning. This is what
        // `select_catalog_skills`, `SkillTool`, and the slash-command
        // resolver all see.
        let global = make_global_skill("review", "Global review");
        let project = make_project_skill("review", "Project review", "project");

        let resolved = apply_skill_overrides(&[global, project]);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].description, "Project review");
        assert!(matches!(
            resolved[0].source,
            SkillSource::ProjectLocal { .. }
        ));
    }

    #[test]
    fn test_apply_skill_overrides_same_source_collision_keeps_first() {
        // Two globals (or two project-locals from different worktrees)
        // colliding don't have a clear winner; preserve the historical
        // "first one wins" behavior.
        let first = make_global_skill("review", "First");
        let second = make_global_skill("review", "Second");

        let resolved = apply_skill_overrides(&[first, second]);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].description, "First");
    }

    #[test]
    fn test_apply_skill_overrides_global_wins_over_builtin() {
        // A global skill with the same name as a built-in must shadow
        // the built-in in the model-facing projection, regardless of
        // iteration order.
        let built_in = make_builtin_skill("create-skill", "Built-in version");
        let global = make_global_skill("create-skill", "User override");

        let resolved = apply_skill_overrides(&[built_in, global]);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].description, "User override");
        assert!(matches!(resolved[0].source, SkillSource::Global));
    }

    #[test]
    fn test_apply_skill_overrides_project_wins_over_builtin() {
        let built_in = make_builtin_skill("create-skill", "Built-in version");
        let project = make_project_skill("create-skill", "Project override", "my-project");

        let resolved = apply_skill_overrides(&[built_in, project]);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].description, "Project override");
        assert!(matches!(
            resolved[0].source,
            SkillSource::ProjectLocal { .. }
        ));
    }

    #[test]
    fn test_apply_skill_overrides_project_wins_over_builtin_and_global() {
        // All three sources present — the project-local must win and
        // both lower-precedence entries must be dropped from the
        // model-facing projection.
        let built_in = make_builtin_skill("create-skill", "Built-in");
        let global = make_global_skill("create-skill", "Global");
        let project = make_project_skill("create-skill", "Project", "my-project");

        let resolved = apply_skill_overrides(&[built_in, global, project]);

        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].description, "Project");
    }

    #[test]
    fn test_apply_skill_overrides_preserves_unique_skills() {
        let global_a = make_global_skill("alpha", "a");
        let global_b = make_global_skill("beta", "b");
        let project_c = make_project_skill("gamma", "c", "project");

        let resolved = apply_skill_overrides(&[global_a, global_b, project_c]);

        assert_eq!(resolved.len(), 3);
        let names: Vec<&str> = resolved.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_skill_source_scope_prefix_and_matches_scope() {
        // The popup inserts `/<prefix>:<name>` using `scope_prefix`,
        // and the resolver routes via `matches_scope`. This test pins
        // the contract that the two stay in sync.
        let global = SkillSource::Global;
        // Globals use an empty prefix, so the popup inserts `/:<name>`.
        assert_eq!(global.scope_prefix(), "");
        assert!(global.matches_scope(""));
        // Hand-typed `/global:<name>` is not aliased to the global
        // source; it looks for a worktree literally named `global`.
        assert!(!global.matches_scope("global"));
        assert!(!global.matches_scope("zed"));

        let project = SkillSource::ProjectLocal {
            worktree_id: SkillScopeId(1),
            worktree_root_name: "zed".into(),
        };
        // Project-local skills are scoped by their worktree root name
        // so multiple open worktrees with same-named skills can each
        // be addressed unambiguously.
        assert_eq!(project.scope_prefix(), "zed");
        assert!(project.matches_scope("zed"));
        // The empty scope is reserved for globals.
        assert!(!project.matches_scope(""));
        // An unrelated worktree name (or MCP server name) must not
        // match a project skill from a different worktree.
        assert!(!project.matches_scope("extensions"));

        // A worktree literally named `global` is no longer ambiguous
        // with the global source: its skills are invoked as
        // `/global:<name>` while globals are invoked as `/:<name>`.
        let project_named_global = SkillSource::ProjectLocal {
            worktree_id: SkillScopeId(2),
            worktree_root_name: "global".into(),
        };
        assert_eq!(project_named_global.scope_prefix(), "global");
        assert!(project_named_global.matches_scope("global"));
        assert!(!project_named_global.matches_scope(""));
    }

    #[test]
    fn test_select_catalog_skills_emits_issue_for_dropped_skills() {
        // Each skill's name + description occupies ~10KB. With a 50KB
        // budget, only the first ~5 visible skills fit; the rest must
        // appear as loading issues so the UI can surface them.
        let description = "x".repeat(10 * 1024);
        let mut skills = Vec::new();
        let total = 10;
        for i in 0..total {
            let name = format!("skill-{i:02}");
            skills.push(Skill {
                name: name.clone(),
                description: description.clone(),
                source: SkillSource::Global,
                directory_path: PathBuf::from(format!("/skills/{name}")),
                skill_file_path: PathBuf::from(format!("/skills/{name}/SKILL.md")),
                load_warnings: Vec::new(),
                disable_model_invocation: false,
                embedded_body: None,
            });
        }

        let (kept, issues) = select_catalog_skills(&skills);

        assert!(
            kept.len() < skills.len(),
            "some skills should be dropped due to the budget (kept {} of {})",
            kept.len(),
            skills.len(),
        );
        assert_eq!(
            issues.len(),
            1,
            "all dropped skills should be consolidated into a single issue, got {issues:?}",
        );

        let kept_size: usize = kept
            .iter()
            .map(|s| s.name.len() + s.description.len())
            .sum();
        assert!(
            kept_size <= MAX_SKILL_DESCRIPTIONS_SIZE,
            "kept skills must fit in the budget (got {kept_size} bytes)",
        );

        let issue = &issues[0];
        assert_eq!(issue.kind, SkillLoadingIssueKind::CatalogBudgetExceeded);
        assert!(
            issue.message.contains("50KB") && issue.message.contains("budget"),
            "issue message {:?} should describe the budget",
            issue.message,
        );
        assert_eq!(
            issue.path,
            skills[kept.len()].skill_file_path,
            "issue path should match the first dropped skill",
        );

        for dropped_skill in &skills[kept.len()..total] {
            let name = &dropped_skill.name;
            assert!(
                issue.message.contains(name.as_str()),
                "issue message {:?} should mention the dropped skill name {name:?}",
                issue.message,
            );
            let bullet_line = format!("- {name}");
            assert!(
                issue
                    .message
                    .lines()
                    .any(|line| line.starts_with(&bullet_line)),
                "issue message {:?} should contain a bullet line starting with {bullet_line:?}",
                issue.message,
            );
        }
    }

    #[test]
    fn test_select_catalog_skills_stops_packing_after_first_overflow() {
        // Once a model-invocable skill overflows the budget, no later
        // skills should be admitted, even if they're small enough to fit
        // in the remaining sliver. This keeps the cutoff deterministic by
        // sort order rather than dependent on individual skill sizes.
        let half_description = "a".repeat(MAX_SKILL_DESCRIPTIONS_SIZE / 2);
        let big_description = "b".repeat(MAX_SKILL_DESCRIPTIONS_SIZE);
        let small_description = "c".repeat(100);

        let first = Skill {
            name: "skill-01-first".to_string(),
            description: half_description,
            source: SkillSource::Global,
            directory_path: PathBuf::from("/skills/skill-01-first"),
            skill_file_path: PathBuf::from("/skills/skill-01-first/SKILL.md"),
            load_warnings: Vec::new(),
            disable_model_invocation: false,
            embedded_body: None,
        };
        let second = Skill {
            name: "skill-02-overflows".to_string(),
            description: big_description,
            source: SkillSource::Global,
            directory_path: PathBuf::from("/skills/skill-02-overflows"),
            skill_file_path: PathBuf::from("/skills/skill-02-overflows/SKILL.md"),
            load_warnings: Vec::new(),
            disable_model_invocation: false,
            embedded_body: None,
        };
        let third = Skill {
            name: "skill-03-would-fit".to_string(),
            description: small_description,
            source: SkillSource::Global,
            directory_path: PathBuf::from("/skills/skill-03-would-fit"),
            skill_file_path: PathBuf::from("/skills/skill-03-would-fit/SKILL.md"),
            load_warnings: Vec::new(),
            disable_model_invocation: false,
            embedded_body: None,
        };

        // Sanity-check the test setup: the third skill is small enough
        // that a greedy packer would have squeezed it in alongside the
        // first one.
        let leftover_after_first =
            MAX_SKILL_DESCRIPTIONS_SIZE - (first.name.len() + first.description.len());
        assert!(
            third.name.len() + third.description.len() <= leftover_after_first,
            "third skill must fit in the leftover sliver for this test to be meaningful",
        );

        let skills = vec![first.clone(), second.clone(), third.clone()];
        let (kept, issues) = select_catalog_skills(&skills);

        let kept_names: Vec<&str> = kept.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(kept_names, vec![first.name.as_str()]);

        assert_eq!(issues.len(), 1, "expected a single consolidated issue");
        assert_eq!(issues[0].kind, SkillLoadingIssueKind::CatalogBudgetExceeded);
        assert_eq!(issues[0].path, second.skill_file_path);
        assert!(
            issues[0].message.contains(second.name.as_str()),
            "issue message {:?} should mention {:?}",
            issues[0].message,
            second.name,
        );
        assert!(
            issues[0].message.contains(third.name.as_str()),
            "issue message {:?} should mention {:?}",
            issues[0].message,
            third.name,
        );
        assert!(
            issues[0].message.contains("- "),
            "issue message {:?} should use bullet form when multiple skills are dropped",
            issues[0].message,
        );
    }

    #[test]
    fn test_select_catalog_skills_excludes_hidden_skills_from_catalog() {
        // Hidden skills (`disable_model_invocation: true`) are slash-only and
        // must not appear in the catalog returned by `select_catalog_skills`,
        // even when they would otherwise fit in the budget. They also don't
        // count against the budget, so a hidden skill larger than the entire
        // budget shouldn't generate a loading issue or prevent later visible
        // skills from fitting.
        let huge_description = "y".repeat(MAX_SKILL_DESCRIPTIONS_SIZE * 2);
        let hidden = Skill {
            name: "hidden-huge".to_string(),
            description: huge_description,
            source: SkillSource::Global,
            directory_path: PathBuf::from("/skills/hidden-huge"),
            skill_file_path: PathBuf::from("/skills/hidden-huge/SKILL.md"),
            load_warnings: Vec::new(),
            disable_model_invocation: true,
            embedded_body: None,
        };
        let visible = Skill {
            name: "visible".to_string(),
            description: "short".to_string(),
            source: SkillSource::Global,
            directory_path: PathBuf::from("/skills/visible"),
            skill_file_path: PathBuf::from("/skills/visible/SKILL.md"),
            load_warnings: Vec::new(),
            disable_model_invocation: false,
            embedded_body: None,
        };

        let (kept, issues) = select_catalog_skills(&[hidden, visible]);

        assert!(issues.is_empty(), "expected no issues, got: {issues:?}");
        let kept_names: Vec<&str> = kept.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(kept_names, vec!["visible"]);
    }

    #[gpui::test]
    async fn test_maintaining_project_context(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {
                    ".agents": {
                        "skills": {
                            "review": {
                                "SKILL.md": "---\nname: review\ndescription: Review from a\n---\n\nbody from a"
                            }
                        }
                    }
                },
                "b": {
                    "AGENTS.md": "instructions for b",
                    "CLAUDE.md": "lower-priority instructions must not load",
                    ".goosehints": "must not load",
                    ".simhints": "must not load",
                    ".agents": {
                        "skills": {
                            "review": {
                                "SKILL.md": "---\nname: review\ndescription: Review from b\n---\n\nbody from b"
                            }
                        }
                    }
                }
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        // Creating a session registers the project and triggers context building.
        let connection = Rc::new(NativeAgentConnection(agent.clone()));
        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let thread = agent.read_with(cx, |agent, _cx| {
            agent.sessions.values().next().unwrap().thread.clone()
        });
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());

        agent.read_with(cx, |agent, cx| {
            let project_id = project.entity_id();
            let state = agent.projects.get(&project_id).unwrap();
            assert_eq!(state.project_context.read(cx).worktrees, vec![]);
            assert_eq!(thread.read(cx).project_context().read(cx).worktrees, vec![]);
        });

        let worktree = project
            .update(cx, |project, cx| project.create_worktree("/a", true, cx))
            .await
            .unwrap();
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let project_id = project.entity_id();
            let state = agent.projects.get(&project_id).unwrap();
            let expected_worktrees = vec![WorktreeContext {
                root_name: "a".into(),
                abs_path: Path::new("/a").into(),
                rules_file: None,
            }];
            assert_eq!(state.project_context.read(cx).worktrees, expected_worktrees);
            assert_eq!(
                thread.read(cx).project_context().read(cx).worktrees,
                expected_worktrees
            );
        });

        // Creating `/a/.rules` updates the project context.
        fs.insert_file("/a/.rules", b"instructions for a".to_vec())
            .await;
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let project_id = project.entity_id();
            let state = agent.projects.get(&project_id).unwrap();
            let rules_entry = worktree
                .read(cx)
                .entry_for_path(rel_path(".rules"))
                .unwrap();
            let expected_worktrees = vec![WorktreeContext {
                root_name: "a".into(),
                abs_path: Path::new("/a").into(),
                rules_file: Some(RulesFileContext {
                    path_in_worktree: rel_path(".rules").into(),
                    text: "instructions for a".into(),
                    project_entry_id: rules_entry.id.to_usize(),
                }),
            }];
            assert_eq!(state.project_context.read(cx).worktrees, expected_worktrees);
            assert_eq!(
                thread.read(cx).project_context().read(cx).worktrees,
                expected_worktrees
            );
        });

        fs.insert_file("/a/.rules", b"updated instructions for a".to_vec())
            .await;
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let state = agent.projects.get(&project.entity_id()).unwrap();
            assert_eq!(
                state.project_context.read(cx).worktrees[0]
                    .rules_file
                    .as_ref()
                    .unwrap()
                    .text,
                "updated instructions for a"
            );
        });

        let second_worktree = project
            .update(cx, |project, cx| project.create_worktree("/b", true, cx))
            .await
            .unwrap();
        cx.run_until_parked();
        let first_worktree_id = worktree.read_with(cx, |worktree, _cx| worktree.id());
        let second_worktree_id = second_worktree.read_with(cx, |worktree, _cx| worktree.id());
        project
            .update(cx, |project, cx| {
                project.move_worktree(first_worktree_id, second_worktree_id, cx)
            })
            .unwrap();
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let context = agent
                .projects
                .get(&project.entity_id())
                .unwrap()
                .project_context
                .read(cx);
            assert_eq!(
                context
                    .worktrees
                    .iter()
                    .map(|worktree| worktree.root_name.as_str())
                    .collect::<Vec<_>>(),
                vec!["b", "a"]
            );
            assert_eq!(thread.read(cx).project_context().read(cx), context);
            let review_skills = user_skills(&agent.projects[&project.entity_id()].skills)
                .into_iter()
                .filter(|skill| skill.name == "review")
                .collect::<Vec<_>>();
            assert_eq!(review_skills.len(), 2);
            assert_eq!(
                review_skills
                    .iter()
                    .map(|skill| skill.source.display_label().to_string())
                    .collect::<Vec<_>>(),
                vec!["b", "a"]
            );
            assert_eq!(
                context
                    .skills()
                    .iter()
                    .find(|skill| skill.name == "review")
                    .unwrap()
                    .description,
                "Review from b"
            );
            let rendered = SystemPromptTemplate {
                project: context,
                available_tools: Vec::new(),
                model_name: None,
                date: "2026-08-11".to_string(),
                user_agents_md: Some("personal instructions".into()),
                sandboxing: false,
                is_linux: false,
                is_windows: false,
            }
            .render(&Templates::new())
            .unwrap();
            assert!(rendered.contains("personal instructions"));
            assert!(rendered.contains("`b/AGENTS.md`"));
            assert!(rendered.contains("Review from b"));
            assert!(!rendered.contains("body from b"));
            assert!(!rendered.contains("lower-priority instructions must not load"));
            assert!(!rendered.contains("must not load"));
        });
        cx.update(|cx| {
            assert_eq!(
                connection
                    .available_skills(&session_id, cx)
                    .iter()
                    .filter(|skill| skill.name == "review")
                    .count(),
                2
            );
        });

        fs.insert_file(
            "/b/.agents/skills/review/SKILL.md",
            b"---\nname: review\ndescription: Updated review from b\n---\n\nupdated body from b"
                .to_vec(),
        )
        .await;
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let state = &agent.projects[&project.entity_id()];
            assert!(state.skills.iter().any(|skill| {
                skill.name == "review"
                    && skill.source.display_label() == "b"
                    && skill.description == "Updated review from b"
            }));
            let status = agent.status_command_output(&session_id, cx).unwrap();
            assert!(status.contains(&format!("- Skills: {}", state.skills.len())));
            assert!(status.contains("b — instructions: AGENTS.md"));
            assert!(status.contains("- Developer context issues: 0"));
        });

        fs.rename(
            Path::new("/b"),
            Path::new("/renamed"),
            fs::RenameOptions::default(),
        )
        .await
        .unwrap();
        second_worktree.update(cx, |worktree, cx| {
            worktree
                .as_local_mut()
                .unwrap()
                .update_abs_path_and_refresh(
                    util::paths::SanitizedPath::new_arc(Path::new("/renamed")),
                    cx,
                );
        });
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let context = agent
                .projects
                .get(&project.entity_id())
                .unwrap()
                .project_context
                .read(cx);
            assert_eq!(context.worktrees[0].root_name, "renamed");
            assert_eq!(
                context.worktrees[0].abs_path.as_ref(),
                Path::new("/renamed")
            );
            assert_eq!(
                context.worktrees[0].rules_file.as_ref().unwrap().text,
                "instructions for b"
            );
        });

        project.update(cx, |project, cx| {
            project.remove_worktree(first_worktree_id, cx)
        });
        cx.run_until_parked();
        agent.read_with(cx, |agent, cx| {
            let context = agent
                .projects
                .get(&project.entity_id())
                .unwrap()
                .project_context
                .read(cx);
            assert_eq!(context.worktrees.len(), 1);
            assert_eq!(context.worktrees[0].root_name, "renamed");
            assert_eq!(
                context.worktrees[0].rules_file.as_ref().unwrap().text,
                "instructions for b"
            );
            assert_eq!(thread.read(cx).project_context().read(cx), context);

            let status = agent.status_command_output(&session_id, cx).unwrap();
            assert!(status.contains("renamed"));
            assert!(status.contains("renamed — instructions: AGENTS.md"));
        });
    }

    #[gpui::test]
    async fn test_system_prompt_refreshes_personal_instructions_for_open_session(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "project": {} })).await;
        let agents_file = paths::agents_file();
        fs.create_dir(agents_file.parent().unwrap()).await.unwrap();
        fs.insert_file(agents_file, b"personal instructions v1".to_vec())
            .await;
        cx.update(|cx| agent_settings::init_user_agents_md(fs.clone(), cx, |_, _| {}));
        cx.run_until_parked();

        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));
        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project, PathList::new(&[Path::new("/project")]), cx)
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        cx.update(|cx| {
            native_thread_for_session(&agent, &session_id, cx)
                .update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        });

        let first_turn = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id.clone(), vec!["first turn".into()]),
                cx,
            )
        });
        cx.run_until_parked();
        let first_request = model.pending_completions().pop().unwrap();
        let first_system_prompt = first_request.messages[0].string_contents();
        assert!(first_system_prompt.contains("personal instructions v1"));
        model.end_completion_stream(&first_request);
        cx.run_until_parked();
        first_turn.await.unwrap();

        fs.insert_file(agents_file, b"personal instructions v2".to_vec())
            .await;
        cx.run_until_parked();
        let second_turn = cx.update(|cx| {
            acp_thread::AgentSessionClientUserMessageIds::prompt(
                connection.as_ref(),
                ClientUserMessageId::new(),
                acp::PromptRequest::new(session_id, vec!["second turn".into()]),
                cx,
            )
        });
        cx.run_until_parked();
        let second_request = model.pending_completions().pop().unwrap();
        let second_system_prompt = second_request.messages[0].string_contents();
        assert!(second_system_prompt.contains("personal instructions v2"));
        assert!(!second_system_prompt.contains("personal instructions v1"));
        model.end_completion_stream(&second_request);
        cx.run_until_parked();
        second_turn.await.unwrap();
    }

    #[gpui::test]
    async fn test_project_context_refreshes_instruction_deletion(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                "AGENTS.md": "instructions to delete"
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().unwrap()
        });
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));
        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/project")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());
        agent.read_with(cx, |agent, cx| {
            assert!(
                agent.projects[&project.entity_id()]
                    .project_context
                    .read(cx)
                    .worktrees[0]
                    .rules_file
                    .is_some()
            );
        });

        fs.remove_file(
            Path::new("/project/AGENTS.md"),
            fs::RemoveOptions::default(),
        )
        .await
        .unwrap();
        let mut refresh = worktree.read_with(cx, |worktree, _cx| {
            worktree
                .as_local()
                .unwrap()
                .manually_refresh_entries_for_paths(vec![rel_path("AGENTS.md").into()])
        });
        refresh.next().await;
        agent.update(cx, |agent, _cx| {
            agent
                .projects
                .get_mut(&project.entity_id())
                .unwrap()
                .project_context_needs_refresh
                .send(())
                .unwrap();
        });
        cx.run_until_parked();

        agent.read_with(cx, |agent, cx| {
            assert!(
                agent.projects[&project.entity_id()]
                    .project_context
                    .read(cx)
                    .worktrees[0]
                    .rules_file
                    .is_none()
            );
            let status = agent.status_command_output(&session_id, cx).unwrap();
            assert!(status.contains("project — instructions: no project instructions"));
        });
    }

    #[gpui::test]
    async fn test_global_skills_load_and_reload(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let skills_dir = global_skills_dir();
        let initial_skill_dir = skills_dir.join("my-skill");
        let initial_skill_path = initial_skill_dir.join("SKILL.md");
        fs.create_dir(&initial_skill_dir).await.unwrap();
        fs.insert_file(
            &initial_skill_path,
            b"---\nname: my-skill\ndescription: First version\n---\n\nbody-v1".to_vec(),
        )
        .await;

        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        // Simulate the user-interaction trigger that the agent panel
        // fires (input focus, slash autocomplete, or submit). In tests
        // we call it directly because there's no panel.
        cx.update(|cx| {
            agent.update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
        });

        let connection = NativeAgentConnection(agent.clone());
        let _acp_thread = cx
            .update(|cx| {
                Rc::new(connection).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        // The pre-existing skill should be loaded into the project state.
        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project.entity_id()).unwrap();
            let user = user_skills(&state.skills);
            assert_eq!(user.len(), 1);
            assert_eq!(user[0].name, "my-skill");
            assert_eq!(user[0].description, "First version");
        });

        // Modify the SKILL.md and verify the project context refreshes.
        fs.write(
            &initial_skill_path,
            b"---\nname: my-skill\ndescription: Second version\n---\n\nbody-v2",
        )
        .await
        .unwrap();
        cx.run_until_parked();

        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project.entity_id()).unwrap();
            let user = user_skills(&state.skills);
            assert_eq!(user.len(), 1);
            assert_eq!(user[0].description, "Second version");
        });
    }

    #[gpui::test]
    async fn test_global_skill_with_long_description_loads_with_warning(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let skills_dir = global_skills_dir();
        let skill_dir = skills_dir.join("long-description");
        let skill_path = skill_dir.join("SKILL.md");
        let long_description = "a".repeat(agent_skills::MAX_SKILL_DESCRIPTION_LEN + 1);
        fs.create_dir(&skill_dir).await.unwrap();
        fs.insert_file(
            &skill_path,
            format!("---\nname: long-description\ndescription: {long_description}\n---\n\nbody")
                .into_bytes(),
        )
        .await;

        let project = Project::test(fs.clone(), [], cx).await;
        let project_id = project.entity_id();
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        cx.update(|cx| {
            agent.update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
        });

        let connection = NativeAgentConnection(agent.clone());
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let loaded_skill = agent.read_with(cx, |agent, cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let user = user_skills(&state.skills);
            assert_eq!(user.len(), 1);
            assert_eq!(user[0].name, "long-description");
            assert_eq!(user[0].description, long_description);

            let catalog_names: Vec<&str> = state
                .project_context
                .read(cx)
                .skills()
                .iter()
                .map(|skill| skill.name.as_str())
                .collect();
            assert!(
                catalog_names.contains(&"long-description"),
                "long-description skill should remain in the model catalog: {catalog_names:?}"
            );

            assert!(
                state.skill_loading_issues.iter().any(|issue| {
                    issue.kind == SkillLoadingIssueKind::DescriptionTooLong
                        && issue.path == skill_path
                        && issue.message.to_string().contains("1024-byte limit")
                }),
                "expected a description-length warning issue, got {:?}",
                state.skill_loading_issues
            );

            (*user[0]).clone()
        });

        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());
        cx.update(|cx| {
            let available_skills = connection.available_skills(&session_id, cx);
            let available_skill = available_skills
                .iter()
                .find(|skill| skill.name == "long-description")
                .expect("long-description should appear in available skills");
            assert_eq!(available_skill.description, long_description);
            assert!(
                available_skill
                    .warning
                    .as_ref()
                    .is_some_and(|warning| warning.contains("1024-byte limit")),
                "available skill should expose warning text, got {:?}",
                available_skill.warning
            );
        });

        let body = agent_skills::read_skill_body(fs.as_ref(), &loaded_skill.skill_file_path)
            .await
            .expect("body should load despite description-length warning");
        assert_eq!(body, "body");
    }

    #[gpui::test]
    async fn test_symlinked_global_skills_load_and_reload(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let skills_dir = global_skills_dir();
        let external_skill_dir = PathBuf::from(path!("/external/my-skill"));
        let skill_link_dir = skills_dir.join("my-skill");
        let skill_link_path = skill_link_dir.join("SKILL.md");

        fs.insert_tree(
            &external_skill_dir,
            json!({
                "SKILL.md": "---\nname: my-skill\ndescription: First symlinked version\n---\n\nbody-v1"
            }),
        )
        .await;
        fs.create_dir(&skills_dir).await.unwrap();
        fs.create_symlink(&skill_link_dir, external_skill_dir)
            .await
            .unwrap();

        let project = Project::test(fs.clone(), [], cx).await;
        let project_id = project.entity_id();
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        cx.update(|cx| {
            agent.update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
        });

        let connection = NativeAgentConnection(agent.clone());
        let _acp_thread = cx
            .update(|cx| {
                Rc::new(connection).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let loaded_skill = agent.read_with(cx, |agent, cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let user = user_skills(&state.skills);
            assert_eq!(user.len(), 1);
            assert_eq!(user[0].name, "my-skill");
            assert_eq!(user[0].description, "First symlinked version");
            assert_eq!(user[0].source, SkillSource::Global);
            assert_eq!(user[0].skill_file_path, skill_link_path);

            let catalog_skills = state.project_context.read(cx).skills();
            let catalog_skill = catalog_skills
                .iter()
                .find(|skill| skill.name == "my-skill")
                .expect("symlinked skill should be included in the model-facing catalog");
            assert_eq!(catalog_skill.description, "First symlinked version");
            assert_eq!(
                catalog_skill.location,
                skill_link_path.to_string_lossy().as_ref()
            );

            (*user[0]).clone()
        });
        let body = agent_skills::read_skill_body(fs.as_ref(), &loaded_skill.skill_file_path)
            .await
            .unwrap();
        assert_eq!(body, "body-v1");

        fs.write(
            &skill_link_path,
            b"---\nname: my-skill\ndescription: Second symlinked version\n---\n\nbody-v2",
        )
        .await
        .unwrap();
        cx.run_until_parked();

        let reloaded_skill = agent.read_with(cx, |agent, cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let user = user_skills(&state.skills);
            assert_eq!(user.len(), 1);
            assert_eq!(user[0].name, "my-skill");
            assert_eq!(user[0].description, "Second symlinked version");
            assert_eq!(user[0].source, SkillSource::Global);
            assert_eq!(user[0].skill_file_path, skill_link_path);

            let catalog_skills = state.project_context.read(cx).skills();
            let catalog_skill = catalog_skills
                .iter()
                .find(|skill| skill.name == "my-skill")
                .expect("reloaded symlinked skill should be included in the model-facing catalog");
            assert_eq!(catalog_skill.description, "Second symlinked version");
            assert_eq!(
                catalog_skill.location,
                skill_link_path.to_string_lossy().as_ref()
            );

            (*user[0]).clone()
        });
        let body = agent_skills::read_skill_body(fs.as_ref(), &reloaded_skill.skill_file_path)
            .await
            .unwrap();
        assert_eq!(body, "body-v2");
    }

    #[gpui::test]
    async fn test_global_skills_dir_created_after_startup(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let skills_dir = global_skills_dir();

        // Intentionally do NOT pre-create `skills_dir`. The first scan
        // trigger should find no directory and leave the watch state
        // idle; a later trigger after the directory is created should
        // attach to the deepest existing ancestor and react when the
        // directory is created later.

        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        // First scan trigger: nothing on disk yet, state stays idle.
        cx.update(|cx| {
            agent.update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
        });

        let connection = NativeAgentConnection(agent.clone());
        let _acp_thread = cx
            .update(|cx| {
                Rc::new(connection).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        // No skills directory exists yet, so no skills should be loaded.
        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project.entity_id()).unwrap();
            assert!(
                user_skills(&state.skills).is_empty(),
                "expected no user skills before the global skills dir exists, got {:?}",
                state.skills
            );
        });

        // Create the global skills directory and a skill within it.
        let new_skill_dir = skills_dir.join("late-skill");
        fs.create_dir(&new_skill_dir).await.unwrap();
        fs.insert_file(
            &new_skill_dir.join("SKILL.md"),
            b"---\nname: late-skill\ndescription: Created after startup\n---\n\nbody".to_vec(),
        )
        .await;

        // Fire the trigger again, simulating the user interacting with
        // the agent panel after creating the skills directory. The
        // second scan should find the directory and start the watch,
        // which refreshes project context.
        cx.update(|cx| {
            agent.update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
        });
        cx.run_until_parked();

        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project.entity_id()).unwrap();
            let user = user_skills(&state.skills);
            assert_eq!(user.len(), 1);
            assert_eq!(user[0].name, "late-skill");
            assert_eq!(user[0].description, "Created after startup");
        });
    }

    /// Regression test for the case where a skill is added (e.g. by the
    /// SKILL.md file watcher) AFTER a session is registered. The system
    /// prompt and slash-command list both read live state, so they pick
    /// up the new skill automatically. The `SkillTool` registered on the
    /// thread used to hold a stale snapshot of `state.skills` taken at
    /// thread-construction time, which meant the model would see the new
    /// skill in `<available_skills>` but get "not found" when it tried to
    /// invoke it. The fix wires the tool to a dynamic resolver closure
    /// that re-reads `state.skills` for the project on every invocation.
    #[gpui::test]
    async fn test_skills_added_after_session_visible_to_skill_tool(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let skills_dir = global_skills_dir();

        // No skills directory exists at startup; the watcher should
        // create one and pick up SKILL.md when it's added later.
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        // First scan trigger: nothing on disk yet.
        cx.update(|cx| {
            agent.update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
        });

        let connection = NativeAgentConnection(agent.clone());
        let _acp_thread = cx
            .update(|cx| {
                Rc::new(connection).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let project_id = project.entity_id();
        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            assert!(
                user_skills(&state.skills).is_empty(),
                "expected no user skills before the global skills dir exists, got {:?}",
                state.skills
            );
        });

        // Build the same resolver closure that `register_session` uses.
        // This is the production resolver factored into a helper so the
        // test can verify resolution behavior directly without setting
        // up the full tool-call plumbing (`ToolInput`,
        // `ToolCallEventStream`, authorization channel, ...).
        let resolve =
            cx.update(|_cx| super::skills_resolver_for_project(agent.downgrade(), project_id));

        // Sanity check: before any skills exist, the resolver returns an
        // empty list — NOT the snapshot that `Thread::new` would have
        // captured.
        cx.update(|cx| {
            let all = resolve(cx);
            let user: Vec<_> = all
                .iter()
                .filter(|s| !matches!(s.source, SkillSource::BuiltIn))
                .collect();
            assert!(user.is_empty());
        });

        // Now create a SKILL.md AFTER the session was registered. With
        // the old code this would be invisible to the `SkillTool`
        // because the tool held an `Arc<Vec<Skill>>` snapshot taken at
        // thread construction time.
        let new_skill_dir = skills_dir.join("my-skill");
        fs.create_dir(&new_skill_dir).await.unwrap();
        fs.insert_file(
            &new_skill_dir.join("SKILL.md"),
            b"---\nname: my-skill\ndescription: Created after session\n---\n\nbody".to_vec(),
        )
        .await;

        // Second scan trigger: now the directory exists, so the scan
        // starts the watch and refreshes project context.
        cx.update(|cx| {
            agent.update(cx, |agent, cx| agent.ensure_skills_scan_started(cx));
        });
        cx.run_until_parked();

        // `state.skills` reflects the new skill (the watcher ran).
        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let user = user_skills(&state.skills);
            assert_eq!(user.len(), 1);
            assert_eq!(user[0].name, "my-skill");
        });

        // The resolver the `SkillTool` uses must see it too. This is the
        // crux of the regression test: the tool's view of skills is
        // resolved at invocation time, not at thread-construction time.
        cx.update(|cx| {
            let all = resolve(cx);
            let snapshot: Vec<_> = all
                .iter()
                .filter(|s| !matches!(s.source, SkillSource::BuiltIn))
                .collect();
            assert_eq!(
                snapshot.len(),
                1,
                "dynamic resolver should see the new skill"
            );
            assert_eq!(snapshot[0].name, "my-skill");
            assert_eq!(snapshot[0].description, "Created after session");
        });

        // And rendering the envelope through the same path the tool uses
        // produces a `<skill_content name="my-skill">` block, confirming
        // the model would see the new skill if it invoked the tool.
        let skill_for_render = cx.update(|cx| {
            let snapshot = resolve(cx);
            snapshot
                .iter()
                .find(|s| s.name == "my-skill" && !s.disable_model_invocation)
                .cloned()
                .expect("my-skill should be model-invocable")
        });
        let body = agent_skills::read_skill_body(fs.as_ref(), &skill_for_render.skill_file_path)
            .await
            .expect("skill body should load");
        let rendered = render_skill_envelope(&skill_for_render, &body);
        assert!(
            rendered.contains("<skill_content name=\"my-skill\">"),
            "rendered envelope missing skill_content tag: {rendered}"
        );
    }

    /// Subagents must inherit access to the same skills as their parent.
    /// Production wires this up in `NativeThreadEnvironment::create_subagent_thread`,
    /// which calls `agent.register_session(subagent, project_id, ...)` —
    /// `register_session` is what installs the `SkillTool` on the thread
    /// using a resolver closure keyed on `project_id`. Because the
    /// subagent shares its parent's `project_id`, both threads end up
    /// resolving skills against the same `state.skills`.
    ///
    /// This test exercises that production path directly: it creates a
    /// parent session via the agent connection, builds a subagent thread
    /// the same way `create_subagent_thread` does, and runs it through
    /// `register_session`. It then asserts that the `SkillTool` is
    /// registered on the subagent thread and that resolving against the
    /// same `project_id` produces the same skill set the parent sees.
    #[gpui::test]
    async fn test_subagent_skills_lookup_matches_parent(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let skills_dir = global_skills_dir();
        let skill_dir = skills_dir.join("shared-skill");
        fs.create_dir(&skill_dir).await.unwrap();
        fs.insert_file(
            &skill_dir.join("SKILL.md"),
            b"---\nname: shared-skill\ndescription: A shared skill\n---\n\nbody".to_vec(),
        )
        .await;

        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        // Open a parent session through the connection, the same way
        // production does. This triggers project-context refresh which
        // populates `state.skills` for the project.
        let connection = NativeAgentConnection(agent.clone());
        let _parent_acp = cx
            .update(|cx| {
                Rc::new(connection).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let project_id = project.entity_id();

        // Sanity check: resolving against the parent's project sees the skill.
        let parent_resolve =
            cx.update(|_cx| super::skills_resolver_for_project(agent.downgrade(), project_id));
        cx.update(|cx| {
            let all = parent_resolve(cx);
            let parent_skills: Vec<_> = all
                .iter()
                .filter(|s| !matches!(s.source, SkillSource::BuiltIn))
                .collect();
            assert_eq!(parent_skills.len(), 1);
            assert_eq!(parent_skills[0].name, "shared-skill");
        });

        // Grab the parent thread out of the agent's session map. This
        // mirrors what `create_subagent_thread` does internally — it
        // looks up the parent session by `parent_session_id` and reads
        // its `project_id` to forward to `register_session`.
        let (parent_thread, parent_project_id) = agent.read_with(cx, |agent, _cx| {
            let session = agent
                .sessions
                .values()
                .next()
                .expect("parent session should exist");
            (session.thread.clone(), session.project_id)
        });
        assert_eq!(parent_project_id, project_id);

        // Build the subagent thread the same way
        // `NativeThreadEnvironment::create_subagent_thread` does.
        let subagent_thread = cx.update(|cx| cx.new(|cx| Thread::new_subagent(&parent_thread, cx)));

        // Run the subagent through the production registration path.
        // This is what installs the `SkillTool` on the thread.
        let _subagent_acp = agent.update(cx, |agent, cx| {
            agent.register_session(subagent_thread.clone(), parent_project_id, 1, cx)
        });

        // Verify the subagent thread has the `SkillTool` installed —
        // without `register_session`, it would not.
        subagent_thread.read_with(cx, |thread, _cx| {
            assert!(thread.is_subagent());
            assert!(
                thread.has_registered_tool(SkillTool::NAME),
                "subagent should have SkillTool registered after register_session"
            );
        });

        // The subagent's `SkillTool` is wired to a resolver closure keyed
        // on the same `project_id` the parent used, so it sees the same
        // skill set. We check this by constructing an equivalent resolver
        // against the same project_id and asserting it matches.
        let subagent_resolve = cx
            .update(|_cx| super::skills_resolver_for_project(agent.downgrade(), parent_project_id));
        cx.update(|cx| {
            let all = subagent_resolve(cx);
            let subagent_skills: Vec<_> = all
                .iter()
                .filter(|s| !matches!(s.source, SkillSource::BuiltIn))
                .collect();
            assert_eq!(subagent_skills.len(), 1);
            assert_eq!(subagent_skills[0].name, "shared-skill");
        });
    }

    #[gpui::test]
    async fn test_skills_appear_as_available_skills(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let skills_dir = global_skills_dir();

        // Two skills: one model-invocable (default), one slash-only via
        // `disable-model-invocation: true`. Both should still appear in
        // the slash menu as first-class skills.
        let visible_dir = skills_dir.join("visible-skill");
        fs.create_dir(&visible_dir).await.unwrap();
        fs.insert_file(
            &visible_dir.join("SKILL.md"),
            b"---\nname: visible-skill\ndescription: Visible skill\n---\n\nbody".to_vec(),
        )
        .await;

        let hidden_dir = skills_dir.join("deploy");
        fs.create_dir(&hidden_dir).await.unwrap();
        fs.insert_file(
            &hidden_dir.join("SKILL.md"),
            b"---\nname: deploy\ndescription: Deploy to prod\ndisable-model-invocation: true\n---\n\nbody"
                .to_vec(),
        )
        .await;

        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        let connection = NativeAgentConnection(agent.clone());
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let project_id = project.entity_id();
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());

        agent.read_with(cx, |agent, cx| {
            let commands = NativeAgent::build_available_commands_for_project(
                agent.projects.get(&project_id),
                cx,
            );
            let names: Vec<&str> = commands.iter().map(|c| c.name.as_str()).collect();
            assert!(
                !names.contains(&"visible-skill"),
                "skills should not be exposed as ACP slash commands: {names:?}"
            );
            assert!(
                !names.contains(&"deploy"),
                "slash-only skills should not be exposed as ACP slash commands: {names:?}"
            );
        });

        cx.update(|cx| {
            let skills = connection.available_skills(&session_id, cx);
            let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
            assert!(
                names.contains(&"visible-skill"),
                "visible skill missing from available skills: {names:?}"
            );
            assert!(
                names.contains(&"deploy"),
                "slash-only skill missing from available skills: {names:?}"
            );
        });

        // The model's catalog (ProjectContext.skills) should NOT include
        // `deploy` since it has disable_model_invocation set.
        agent.read_with(cx, |agent, cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let catalog: Vec<&str> = state
                .project_context
                .read(cx)
                .skills()
                .iter()
                .map(|s| s.name.as_str())
                .collect();
            assert!(
                catalog.contains(&"visible-skill"),
                "visible skill missing from catalog: {catalog:?}"
            );
            assert!(
                !catalog.contains(&"deploy"),
                "deploy should be excluded from catalog: {catalog:?}"
            );
        });
    }

    #[gpui::test]
    async fn test_project_skills_require_worktree_trust(cx: &mut TestAppContext) {
        use collections::{HashMap, HashSet};
        use project::trusted_worktrees::{self, PathTrust, TrustedWorktrees};

        init_test(cx);
        cx.update(|cx| {
            // The trust global isn't created by `init_test`. We need it
            // for `Project::test_with_worktree_trust` to actually wire up
            // trust tracking and for our subscription in
            // `register_project_with_initial_context` to fire.
            trusted_worktrees::init(HashMap::default(), cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                "AGENTS.md": "project instructions remain available while restricted",
                ".agents": {
                    "skills": {
                        "my-skill": {
                            "SKILL.md": "---\nname: my-skill\ndescription: A project skill\n---\n\nbody"
                        }
                    }
                }
            }),
        )
        .await;

        // `test_with_worktree_trust` initializes the trust system and
        // starts every worktree as restricted, mirroring production
        // behavior on a freshly opened folder.
        let project =
            Project::test_with_worktree_trust(fs.clone(), [Path::new("/project")], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        let connection = NativeAgentConnection(agent.clone());
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/project")]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let project_id = project.entity_id();
        let session_id = acp_thread.read_with(cx, |thread, _cx| thread.session_id().clone());
        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        // Untrusted: project skills are excluded from the loaded list and
        // never make it into the catalog or slash commands.
        agent.read_with(cx, |agent, cx| {
            let state = agent.projects.get(&project_id).unwrap();
            assert!(
                user_skills(&state.skills).is_empty(),
                "untrusted worktree skills should not load: {:?}",
                state
                    .skills
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
            );
            let commands = NativeAgent::build_available_commands_for_project(Some(state), cx);
            let names: Vec<&str> = commands.iter().map(|c| c.name.as_str()).collect();
            assert!(
                !names.contains(&"my-skill"),
                "untrusted skill leaked into slash commands: {names:?}"
            );
            let context = state.project_context.read(cx);
            assert_eq!(
                context.worktrees[0].rules_file.as_ref().unwrap().text,
                "project instructions remain available while restricted"
            );
        });

        // Granting trust should trigger a context refresh; the skill then
        // appears in both the catalog and the slash-command list.
        cx.update(|cx| {
            let trusted_worktrees = TrustedWorktrees::try_get_global(cx)
                .expect("trusted worktrees global initialized by test_with_worktree_trust");
            trusted_worktrees.update(cx, |trusted_worktrees, cx| {
                trusted_worktrees.trust(
                    &project.read(cx).worktree_store(),
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                );
            });
        });
        cx.run_until_parked();

        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let user = user_skills(&state.skills);
            let names: Vec<&str> = user.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(names, vec!["my-skill"]);
        });

        cx.update(|cx| {
            let skills = connection.available_skills(&session_id, cx);
            let skill_names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
            assert!(
                skill_names.contains(&"my-skill"),
                "trusted skill should appear in available skills: {skill_names:?}"
            );
        });

        cx.update(|cx| {
            let trusted_worktrees = TrustedWorktrees::try_get_global(cx).unwrap();
            let worktree_store = project.read(cx).worktree_store();
            trusted_worktrees.update(cx, |trusted_worktrees, cx| {
                trusted_worktrees.restrict(
                    worktree_store.downgrade(),
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                );
            });
        });
        cx.run_until_parked();

        agent.read_with(cx, |agent, cx| {
            let state = agent.projects.get(&project_id).unwrap();
            assert!(user_skills(&state.skills).is_empty());
            assert_eq!(
                state.project_context.read(cx).worktrees[0]
                    .rules_file
                    .as_ref()
                    .unwrap()
                    .text,
                "project instructions remain available while restricted"
            );
            let commands = NativeAgent::build_available_commands_for_project(Some(state), cx);
            assert!(!commands.iter().any(|command| command.name == "my-skill"));
        });
        cx.update(|cx| {
            assert!(
                connection
                    .available_skills(&session_id, cx)
                    .iter()
                    .all(|skill| skill.name != "my-skill")
            );
        });
    }

    /// Open a session against a freshly created project and trust its only
    /// worktree, so project-local skills load. Returns the agent, the
    /// project, and the worktree id of the project root.
    async fn open_trusted_project_skills(
        cx: &mut TestAppContext,
        fs: Arc<FakeFs>,
        root: &str,
    ) -> (Entity<NativeAgent>, Entity<Project>, WorktreeId) {
        use collections::{HashMap, HashSet};
        use project::trusted_worktrees::{self, PathTrust, TrustedWorktrees};

        cx.update(|cx| {
            trusted_worktrees::init(HashMap::default(), cx);
        });

        let project = Project::test_with_worktree_trust(fs.clone(), [Path::new(root)], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));

        let connection = NativeAgentConnection(agent.clone());
        let _acp_thread = cx
            .update(|cx| {
                Rc::new(connection).new_session(
                    project.clone(),
                    PathList::new(&[Path::new(root)]),
                    cx,
                )
            })
            .await
            .unwrap();
        cx.run_until_parked();

        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });
        cx.update(|cx| {
            let trusted_worktrees = TrustedWorktrees::try_get_global(cx)
                .expect("trusted worktrees global initialized by test_with_worktree_trust");
            trusted_worktrees.update(cx, |trusted_worktrees, cx| {
                trusted_worktrees.trust(
                    &project.read(cx).worktree_store(),
                    HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                    cx,
                );
            });
        });
        cx.run_until_parked();

        (agent, project, worktree_id)
    }

    /// The body resolver for a project-local skill must read the file
    /// through a project buffer rather than the local filesystem. This is
    /// what makes project skills resolvable in remote workspaces, where
    /// the `fs` the agent holds is the client's filesystem and not where
    /// the project files actually live. We prove the buffer path is used
    /// by editing the buffer in memory (without saving) and asserting the
    /// resolver returns the edited body, not the on-disk body.
    #[gpui::test]
    async fn test_project_skill_body_resolves_through_buffer(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".agents": {
                    "skills": {
                        "my-skill": {
                            "SKILL.md": "---\nname: my-skill\ndescription: A project skill\n---\n\ndisk body"
                        }
                    }
                }
            }),
        )
        .await;

        let (agent, project, worktree_id) =
            open_trusted_project_skills(cx, fs.clone(), "/project").await;
        let project_id = project.entity_id();

        let skill = agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            user_skills(&state.skills)
                .into_iter()
                .find(|s| s.name == "my-skill")
                .cloned()
                .expect("project skill should be loaded")
        });
        assert!(matches!(skill.source, SkillSource::ProjectLocal { .. }));

        let resolver =
            cx.update(|_cx| super::skill_body_resolver_for_project(project.clone(), fs.clone()));

        let body = cx
            .update(|cx| resolver(skill.clone(), &mut cx.to_async()))
            .await
            .unwrap();
        assert_eq!(body, "disk body");

        // Edit the buffer in memory without writing to disk.
        let relative_path: Arc<RelPath> = rel_path(".agents/skills/my-skill/SKILL.md").into();
        let buffer = project
            .update(cx, |project, cx| {
                project.open_buffer((worktree_id, relative_path), cx)
            })
            .await
            .unwrap();
        buffer.update(cx, |buffer, cx| {
            buffer.set_text(
                "---\nname: my-skill\ndescription: A project skill\n---\n\nedited body",
                cx,
            );
        });

        let body = cx
            .update(|cx| resolver(skill.clone(), &mut cx.to_async()))
            .await
            .unwrap();
        assert_eq!(
            body, "edited body",
            "resolver must read the in-memory buffer, not the on-disk file"
        );
    }

    #[gpui::test]
    async fn plugin_skill_changes_refresh_native_sessions(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({})).await;
        fs.insert_tree("/managed", json!({"skills": {"review": {"SKILL.md": "---\nname: review\ndescription: Review changes\n---\nInspect changes"}}})).await;
        let (agent, project, _) = open_trusted_project_skills(cx, fs.clone(), "/project").await;
        let filesystem: Arc<dyn fs::Fs> = fs;
        let skills = agent_skills::load_package_skills(&filesystem, Path::new("/managed"), "acme")
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("load package skill");
        cx.update(|cx| cx.update_global::<SkillIndex, _>(|index, _| index.package_skills = skills));
        cx.run_until_parked();
        agent.read_with(cx, |agent, _| {
            assert!(
                agent
                    .projects
                    .get(&project.entity_id())
                    .expect("project state")
                    .skills
                    .iter()
                    .any(|skill| skill.lookup_name() == "plugin/acme:review")
            );
        });
        cx.update(|cx| cx.update_global::<SkillIndex, _>(|index, _| index.package_skills.clear()));
        cx.run_until_parked();
        agent.read_with(cx, |agent, _| {
            assert!(
                !agent
                    .projects
                    .get(&project.entity_id())
                    .expect("project state")
                    .skills
                    .iter()
                    .any(|skill| skill.lookup_name() == "plugin/acme:review")
            );
        });
    }

    #[gpui::test]
    async fn test_project_instruction_issue_omits_only_broken_source(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "broken": { "AGENTS.md": "broken instructions" },
                "healthy": { "AGENTS.md": "healthy instructions" }
            }),
        )
        .await;
        let project = Project::test(
            fs.clone(),
            [Path::new("/broken"), Path::new("/healthy")],
            cx,
        )
        .await;
        fs.remove_file(Path::new("/broken/AGENTS.md"), fs::RemoveOptions::default())
            .await
            .unwrap();

        let (context, _skills, issues) = cx
            .update(|cx| NativeAgent::build_project_context(&project, fs, cx))
            .await;

        let broken = context
            .worktrees
            .iter()
            .find(|worktree| worktree.root_name == "broken")
            .unwrap();
        let healthy = context
            .worktrees
            .iter()
            .find(|worktree| worktree.root_name == "healthy")
            .unwrap();
        assert!(broken.rules_file.is_none());
        assert_eq!(
            healthy.rules_file.as_ref().unwrap().text.as_str(),
            "healthy instructions"
        );
        let [issue] = issues.as_slice() else {
            panic!("expected one project instruction issue, got {issues:?}")
        };
        assert_eq!(
            issue.kind,
            SkillLoadingIssueKind::ProjectInstructionLoadFailed
        );
        assert_eq!(issue.source_label.as_deref(), Some("broken"));
        assert_eq!(issue.relative_path.as_deref(), Some(Path::new("AGENTS.md")));
        assert!(!issue.message.contains("broken instructions"));
    }

    /// A project SKILL.md whose on-disk size exceeds the cap must be
    /// rejected with a size-limit error and excluded from the loaded
    /// skills, exercising the size guard in `load_project_skills`.
    #[gpui::test]
    async fn test_oversized_project_skill_reports_error(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let oversized = format!(
            "---\nname: huge-skill\ndescription: Too big\n---\n\n{}",
            "a".repeat(MAX_SKILL_FILE_SIZE + 1)
        );
        fs.insert_tree(
            "/project",
            json!({
                ".agents": { "skills": { "huge-skill": { "SKILL.md": oversized } } }
            }),
        )
        .await;

        let (agent, project, _worktree_id) =
            open_trusted_project_skills(cx, fs.clone(), "/project").await;
        let project_id = project.entity_id();

        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            assert!(
                user_skills(&state.skills).is_empty(),
                "oversized skill must not load: {:?}",
                user_skills(&state.skills)
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
            );
            assert!(
                state
                    .skill_loading_issues
                    .iter()
                    .any(|issue| issue.kind == SkillLoadingIssueKind::LoadFailed
                        && issue.message.to_string().contains("maximum size")),
                "expected a size-limit error, got {:?}",
                state.skill_loading_issues
            );
        });
    }

    /// A malformed project SKILL.md must surface a per-skill load error
    /// without preventing sibling skills in the same worktree from
    /// loading.
    #[gpui::test]
    async fn test_malformed_project_skill_reports_error(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".agents": {
                    "skills": {
                        "good": {
                            "SKILL.md": "---\nname: good\ndescription: Fine\n---\n\nbody"
                        },
                        "bad": {
                            "SKILL.md": "this file has no frontmatter"
                        }
                    }
                }
            }),
        )
        .await;

        let (agent, project, _worktree_id) =
            open_trusted_project_skills(cx, fs.clone(), "/project").await;
        let project_id = project.entity_id();

        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let names: Vec<&str> = user_skills(&state.skills)
                .iter()
                .map(|s| s.name.as_str())
                .collect();
            assert_eq!(names, vec!["good"], "only the valid skill should load");
            assert!(
                state
                    .skill_loading_issues
                    .iter()
                    .any(|issue| issue.kind == SkillLoadingIssueKind::LoadFailed
                        && issue.path.ends_with("bad/SKILL.md")),
                "expected an error for the malformed skill, got {:?}",
                state.skill_loading_issues
            );
        });
    }

    /// The skill catalog (metadata) is also loaded through project
    /// buffers, and the broadened `.agents` refresh trigger must rebuild
    /// it when files under `.agents` change. We edit the SKILL.md buffer
    /// in memory, then touch an unrelated file directly under `.agents`
    /// (not under `.agents/skills`) and assert the catalog reflects the
    /// in-memory edit. Under the previous `.agents/skills`-only trigger
    /// this refresh would not have fired.
    #[gpui::test]
    async fn test_project_skill_metadata_refreshes_from_buffer(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".agents": {
                    "skills": {
                        "my-skill": {
                            "SKILL.md": "---\nname: my-skill\ndescription: Original\n---\n\nbody"
                        }
                    }
                }
            }),
        )
        .await;

        let (agent, project, worktree_id) =
            open_trusted_project_skills(cx, fs.clone(), "/project").await;
        let project_id = project.entity_id();

        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let skill = user_skills(&state.skills)
                .into_iter()
                .find(|s| s.name == "my-skill")
                .expect("skill should be loaded");
            assert_eq!(skill.description, "Original");
        });

        let relative_path: Arc<RelPath> = rel_path(".agents/skills/my-skill/SKILL.md").into();
        let buffer = project
            .update(cx, |project, cx| {
                project.open_buffer((worktree_id, relative_path), cx)
            })
            .await
            .unwrap();
        buffer.update(cx, |buffer, cx| {
            buffer.set_text(
                "---\nname: my-skill\ndescription: Edited in buffer\n---\n\nbody",
                cx,
            );
        });

        // Touch a file directly under `.agents` (not under
        // `.agents/skills`) to trigger the broadened refresh path.
        fs.insert_file("/project/.agents/marker.txt", b"hello".to_vec())
            .await;
        cx.run_until_parked();

        agent.read_with(cx, |agent, _cx| {
            let state = agent.projects.get(&project_id).unwrap();
            let skill = user_skills(&state.skills)
                .into_iter()
                .find(|s| s.name == "my-skill")
                .expect("skill should still be loaded");
            assert_eq!(
                skill.description, "Edited in buffer",
                "catalog must reflect the in-memory buffer after a refresh"
            );
        });
    }

    #[gpui::test]
    async fn test_listing_models(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {}  })).await;
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let connection = NativeAgentConnection(
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx)),
        );

        // Create a thread/session
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();

        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());

        let models = cx
            .update(|cx| {
                connection
                    .model_selector(&session_id)
                    .unwrap()
                    .list_models(cx)
            })
            .await
            .unwrap();

        let acp_thread::AgentModelList::Grouped(models) = models else {
            panic!("Unexpected model group");
        };
        assert_eq!(
            models,
            IndexMap::from_iter([(
                AgentModelGroupName("Fake".into()),
                vec![AgentModelInfo {
                    id: AgentModelId::new("fake/fake"),
                    name: "Fake".into(),
                    description: None,
                    icon: Some(acp_thread::AgentModelIcon::Named(
                        ui::IconName::ZedAssistant
                    )),
                    is_latest: false,
                    disabled: None,
                    cost: None,
                }]
            )])
        );
    }

    #[gpui::test]
    async fn test_model_selection_persists_to_settings(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.create_dir(paths::settings_file().parent().unwrap())
            .await
            .unwrap();
        fs.insert_file(
            paths::settings_file(),
            json!({
                "agent": {
                    "default_model": {
                        "provider": "foo",
                        "model": "bar"
                    }
                }
            })
            .to_string()
            .into_bytes(),
        )
        .await;
        let project = Project::test(fs.clone(), [], cx).await;

        let thread_store = cx.new(|cx| ThreadStore::new(cx));

        // Create the agent and connection
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));
        let connection = NativeAgentConnection(agent.clone());

        // Create a thread/session
        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();

        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());

        // Select a model
        let selector = connection.model_selector(&session_id).unwrap();
        let model_id = AgentModelId::new("fake/fake");
        cx.update(|cx| selector.select_model(model_id.clone(), cx))
            .await
            .unwrap();

        // Verify the thread has the selected model
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert_eq!(thread.model().unwrap().id().0, "fake");
            });
        });

        cx.run_until_parked();

        // Verify settings file was updated
        let settings_content = fs.load(paths::settings_file()).await.unwrap();
        let settings_json: serde_json::Value = serde_json::from_str(&settings_content).unwrap();

        // Check that the agent settings contain the selected model
        assert_eq!(
            settings_json["agent"]["default_model"]["model"],
            json!("fake")
        );
        assert_eq!(
            settings_json["agent"]["default_model"]["provider"],
            json!("fake")
        );

        // Register a thinking model and select it.
        cx.update(|cx| {
            let thinking_model = Arc::new(FakeLanguageModel::with_id_and_thinking(
                "fake-corp",
                "fake-thinking",
                "Fake Thinking",
                true,
            ));
            let thinking_provider = Arc::new(
                FakeLanguageModelProvider::new(
                    LanguageModelProviderId::from("fake-corp".to_string()),
                    LanguageModelProviderName::from("Fake Corp".to_string()),
                )
                .with_models(vec![thinking_model]),
            );
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(thinking_provider, cx);
            });
        });
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(AgentModelId::new("fake-corp/fake-thinking"), cx))
            .await
            .unwrap();
        cx.run_until_parked();

        // Verify enable_thinking was written to settings as true.
        let settings_content = fs.load(paths::settings_file()).await.unwrap();
        let settings_json: serde_json::Value = serde_json::from_str(&settings_content).unwrap();
        assert_eq!(
            settings_json["agent"]["default_model"]["enable_thinking"],
            json!(true),
            "selecting a thinking model should persist enable_thinking: true to settings"
        );
    }

    #[gpui::test]
    async fn test_select_model_updates_thinking_enabled(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.create_dir(paths::settings_file().parent().unwrap())
            .await
            .unwrap();
        fs.insert_file(paths::settings_file(), b"{}".to_vec()).await;
        let project = Project::test(fs.clone(), [], cx).await;

        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));
        let connection = NativeAgentConnection(agent.clone());

        let acp_thread = cx
            .update(|cx| {
                Rc::new(connection.clone()).new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = cx.update(|cx| acp_thread.read(cx).session_id().clone());

        // Register a second provider with a thinking model.
        cx.update(|cx| {
            let thinking_model = Arc::new(FakeLanguageModel::with_id_and_thinking(
                "fake-corp",
                "fake-thinking",
                "Fake Thinking",
                true,
            ));
            let thinking_provider = Arc::new(
                FakeLanguageModelProvider::new(
                    LanguageModelProviderId::from("fake-corp".to_string()),
                    LanguageModelProviderName::from("Fake Corp".to_string()),
                )
                .with_models(vec![thinking_model]),
            );
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(thinking_provider, cx);
            });
        });
        // Refresh the agent's model list so it picks up the new provider.
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        // Thread starts with thinking_enabled = false (the default).
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert!(!thread.thinking_enabled(), "thinking defaults to false");
            });
        });

        // Select the thinking model via select_model.
        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(AgentModelId::new("fake-corp/fake-thinking"), cx))
            .await
            .unwrap();

        // select_model should have enabled thinking based on the model's supports_thinking().
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert!(
                    thread.thinking_enabled(),
                    "select_model should enable thinking when model supports it"
                );
            });
        });

        // Switch back to the non-thinking model.
        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(AgentModelId::new("fake/fake"), cx))
            .await
            .unwrap();

        // select_model should have disabled thinking.
        agent.read_with(cx, |agent, _| {
            let session = agent.sessions.get(&session_id).unwrap();
            session.thread.read_with(cx, |thread, _| {
                assert!(
                    !thread.thinking_enabled(),
                    "select_model should disable thinking when model does not support it"
                );
            });
        });
    }

    #[gpui::test]
    async fn test_summarization_model_survives_transient_registry_clearing(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [], cx).await;

        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        thread.read_with(cx, |thread, _| {
            assert!(
                thread.summarization_model().is_some(),
                "session should have a summarization model from the test registry"
            );
        });

        // Simulate what happens during a provider blip:
        // update_active_language_model_from_settings calls set_default_model(None)
        // when it can't resolve the model, clearing all fallbacks.
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.set_default_model(None, cx);
            });
        });
        cx.run_until_parked();

        thread.read_with(cx, |thread, _| {
            assert!(
                thread.summarization_model().is_some(),
                "summarization model should survive a transient default model clearing"
            );
        });
    }

    #[gpui::test]
    async fn test_loaded_thread_preserves_thinking_enabled(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        // Register a thinking model.
        let thinking_model = Arc::new(FakeLanguageModel::with_id_and_thinking(
            "fake-corp",
            "fake-thinking",
            "Fake Thinking",
            true,
        ));
        let thinking_provider = Arc::new(
            FakeLanguageModelProvider::new(
                LanguageModelProviderId::from("fake-corp".to_string()),
                LanguageModelProviderName::from("Fake Corp".to_string()),
            )
            .with_models(vec![thinking_model.clone()]),
        );
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(thinking_provider, cx);
            });
        });
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        // Create a thread and select the thinking model.
        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(AgentModelId::new("fake-corp/fake-thinking"), cx))
            .await
            .unwrap();

        // Verify thinking is enabled after selecting the thinking model.
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        thread.read_with(cx, |thread, _| {
            assert!(
                thread.thinking_enabled(),
                "thinking should be enabled after selecting thinking model"
            );
        });

        // Send a message so the thread gets persisted.
        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["Hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        thinking_model.send_last_completion_stream_text_chunk("Response.");
        thinking_model.end_last_completion_stream();

        send.await.unwrap();
        cx.run_until_parked();

        // Close the session so it can be reloaded from disk.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(thread);
        drop(acp_thread);
        agent.read_with(cx, |agent, _| {
            assert!(agent.sessions.is_empty());
        });

        // Reload the thread and verify thinking_enabled is still true.
        let reloaded_acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        let reloaded_thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        reloaded_thread.read_with(cx, |thread, _| {
            assert!(
                thread.thinking_enabled(),
                "thinking_enabled should be preserved when reloading a thread with a thinking model"
            );
        });

        drop(reloaded_acp_thread);
    }

    #[gpui::test]
    async fn test_loaded_thread_preserves_model(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        // Register a model where id() != name(), like real Anthropic models
        // (e.g. id="claude-sonnet-4-5-thinking-latest", name="Claude Sonnet 4.5 Thinking").
        let model = Arc::new(FakeLanguageModel::with_id_and_thinking(
            "fake-corp",
            "custom-model-id",
            "Custom Model Display Name",
            false,
        ));
        let provider = Arc::new(
            FakeLanguageModelProvider::new(
                LanguageModelProviderId::from("fake-corp".to_string()),
                LanguageModelProviderName::from("Fake Corp".to_string()),
            )
            .with_models(vec![model.clone()]),
        );
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(provider, cx);
            });
        });
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        // Create a thread and select the model.
        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(AgentModelId::new("fake-corp/custom-model-id"), cx))
            .await
            .unwrap();

        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        thread.read_with(cx, |thread, _| {
            assert_eq!(
                thread.model().unwrap().id().0.as_ref(),
                "custom-model-id",
                "model should be set before persisting"
            );
        });

        // Send a message so the thread gets persisted.
        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["Hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        model.send_last_completion_stream_text_chunk("Response.");
        model.end_last_completion_stream();

        send.await.unwrap();
        cx.run_until_parked();

        // Close the session so it can be reloaded from disk.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(thread);
        drop(acp_thread);
        agent.read_with(cx, |agent, _| {
            assert!(agent.sessions.is_empty());
        });

        // Reload the thread and verify the model was preserved.
        let reloaded_acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        let reloaded_thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        reloaded_thread.read_with(cx, |thread, _| {
            let reloaded_model = thread
                .model()
                .expect("model should be present after reload");
            assert_eq!(
                reloaded_model.id().0.as_ref(),
                "custom-model-id",
                "reloaded thread should have the same model, not fall back to the default"
            );
        });

        drop(reloaded_acp_thread);
    }

    async fn persist_thread_with_fake_corp_model(
        cx: &mut TestAppContext,
    ) -> (
        Entity<NativeAgent>,
        Rc<NativeAgentConnection>,
        Entity<Project>,
        acp::SessionId,
        Arc<FakeLanguageModelProvider>,
    ) {
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let model = Arc::new(FakeLanguageModel::with_id_and_thinking(
            "fake-corp",
            "custom-model-id",
            "Custom Model Display Name",
            false,
        ));
        let provider = Arc::new(
            FakeLanguageModelProvider::new(
                LanguageModelProviderId::from("fake-corp".to_string()),
                LanguageModelProviderName::from("Fake Corp".to_string()),
            )
            .with_models(vec![model.clone()]),
        );
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(provider.clone(), cx);
            });
        });
        agent.update(cx, |agent, cx| agent.models.refresh_list(cx));

        let acp_thread = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new("/a")]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(AgentModelId::new("fake-corp/custom-model-id"), cx))
            .await
            .unwrap();

        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["Hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();
        model.send_last_completion_stream_text_chunk("Response.");
        model.end_last_completion_stream();
        send.await.unwrap();
        cx.run_until_parked();

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(acp_thread);

        (agent, connection, project, session_id, provider)
    }

    fn unregister_fake_corp(cx: &mut TestAppContext) {
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.unregister_provider(
                    LanguageModelProviderId::from("fake-corp".to_string()),
                    cx,
                );
            });
        });
    }

    #[gpui::test]
    async fn test_loaded_thread_resolves_model_when_provider_loads_late(cx: &mut TestAppContext) {
        init_test(cx);
        let (agent, _connection, project, session_id, provider) =
            persist_thread_with_fake_corp_model(cx).await;

        // Simulate a restart where the provider hasn't fetched its model list
        // yet, so the saved selection can't be resolved at load time.
        unregister_fake_corp(cx);

        let reloaded_acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        thread.read_with(cx, |thread, _| {
            assert!(
                thread.model().is_none(),
                "should not fall back to an unrelated model"
            );
        });

        // The original selection is persisted even while unresolved, so a save
        // during the window can't overwrite the user's choice with a fallback.
        let db_thread = thread.read_with(cx, |thread, cx| thread.to_db(cx)).await;
        let saved = db_thread.model.expect("selection should be persisted");
        assert_eq!(saved.provider, "fake-corp");
        assert_eq!(saved.model, "custom-model-id");

        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(provider.clone(), cx);
            });
        });
        cx.run_until_parked();

        thread.read_with(cx, |thread, _| {
            assert_eq!(
                thread
                    .model()
                    .expect("model should resolve once provider loads")
                    .id()
                    .0
                    .as_ref(),
                "custom-model-id"
            );
        });

        drop(reloaded_acp_thread);
    }

    #[gpui::test]
    async fn test_explicit_model_selection_cancels_pending(cx: &mut TestAppContext) {
        init_test(cx);
        let (agent, connection, project, session_id, provider) =
            persist_thread_with_fake_corp_model(cx).await;

        unregister_fake_corp(cx);

        let reloaded_acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });
        thread.read_with(cx, |thread, _| {
            assert!(thread.model().is_none());
        });

        // The user explicitly picks a different, available model.
        let other_model = Arc::new(FakeLanguageModel::with_id_and_thinking(
            "other-corp",
            "other-model-id",
            "Other Model",
            false,
        ));
        let other_provider = Arc::new(
            FakeLanguageModelProvider::new(
                LanguageModelProviderId::from("other-corp".to_string()),
                LanguageModelProviderName::from("Other Corp".to_string()),
            )
            .with_models(vec![other_model.clone()]),
        );
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(other_provider, cx);
            });
        });
        cx.run_until_parked();

        let selector = connection.model_selector(&session_id).unwrap();
        cx.update(|cx| selector.select_model(AgentModelId::new("other-corp/other-model-id"), cx))
            .await
            .unwrap();

        thread.read_with(cx, |thread, _| {
            assert_eq!(thread.model().unwrap().id().0.as_ref(), "other-model-id");
        });

        // The original provider returning must not clobber the explicit choice.
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(provider.clone(), cx);
            });
        });
        cx.run_until_parked();

        thread.read_with(cx, |thread, _| {
            assert_eq!(
                thread.model().unwrap().id().0.as_ref(),
                "other-model-id",
                "a late provider load must not override the explicit selection"
            );
        });

        drop(reloaded_acp_thread);
    }

    #[gpui::test]
    async fn test_save_load_thread(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {
                    "b.md": "Lorem"
                }
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        // Ensure empty threads are not saved, even if they get mutated.
        let model = Arc::new(FakeLanguageModel::default());
        let summary_model = Arc::new(FakeLanguageModel::default());
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.set_summarization_model(Some(summary_model.clone()), cx);
        });
        cx.run_until_parked();
        assert_eq!(thread_entries(&thread_store, cx), vec![]);

        let send = acp_thread.update(cx, |thread, cx| {
            thread.send(
                vec![
                    "What does ".into(),
                    acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                        "b.md",
                        MentionUri::File {
                            abs_path: path!("/a/b.md").into(),
                        }
                        .to_uri()
                        .to_string(),
                    )),
                    " mean?".into(),
                ],
                cx,
            )
        });
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        model.send_last_completion_stream_text_chunk("Lorem.");
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::UsageUpdate(
            language_model::TokenUsage {
                input_tokens: 150,
                output_tokens: 75,
                ..Default::default()
            },
        ));
        model.end_last_completion_stream();
        cx.run_until_parked();
        summary_model
            .send_last_completion_stream_text_chunk(&format!("Explaining {}", path!("/a/b.md")));
        summary_model.end_last_completion_stream();

        send.await.unwrap();
        let uri = MentionUri::File {
            abs_path: path!("/a/b.md").into(),
        }
        .to_uri();
        acp_thread.read_with(cx, |thread, cx| {
            assert_eq!(
                thread.to_markdown(cx),
                formatdoc! {"
                    ## User

                    What does [@b.md]({uri}) mean?

                    ## Assistant

                    Lorem.

                "}
            )
        });

        cx.run_until_parked();

        // Set a draft prompt with rich content blocks and scroll position
        // AFTER run_until_parked, so the only save that captures these
        // changes is the one performed by close_session itself.
        let draft_blocks = vec![
            acp::ContentBlock::Text(acp::TextContent::new("Check out ")),
            acp::ContentBlock::ResourceLink(acp::ResourceLink::new("b.md", uri.to_string())),
            acp::ContentBlock::Text(acp::TextContent::new(" please")),
        ];
        acp_thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(Some(draft_blocks.clone()), cx);
        });
        thread.update(cx, |thread, _cx| {
            thread.set_ui_scroll_position(Some(gpui::ListOffset {
                item_ix: 5,
                offset_in_item: gpui::px(12.5),
            }));
        });

        // Close the session so it can be reloaded from disk.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(thread);
        drop(acp_thread);
        agent.read_with(cx, |agent, _| {
            assert_eq!(agent.sessions.keys().cloned().collect::<Vec<_>>(), []);
        });

        // Ensure the thread can be reloaded from disk.
        assert_eq!(
            thread_entries(&thread_store, cx),
            vec![(
                session_id.clone(),
                format!("Explaining {}", path!("/a/b.md"))
            )]
        );
        let acp_thread = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        acp_thread.read_with(cx, |thread, cx| {
            assert_eq!(
                thread.to_markdown(cx),
                formatdoc! {"
                    ## User

                    What does [@b.md]({uri}) mean?

                    ## Assistant

                    Lorem.

                "}
            )
        });

        // Ensure the draft prompt with rich content blocks survived the round-trip.
        acp_thread.read_with(cx, |thread, _| {
            assert_eq!(thread.draft_prompt(), Some(draft_blocks.as_slice()));
        });

        // Ensure token usage survived the round-trip.
        acp_thread.read_with(cx, |thread, _| {
            let usage = thread
                .token_usage()
                .expect("token usage should be restored after reload");
            assert_eq!(usage.input_tokens, 150);
            assert_eq!(usage.output_tokens, 75);
        });

        // Ensure scroll position survived the round-trip.
        acp_thread.read_with(cx, |thread, _| {
            let scroll = thread
                .ui_scroll_position()
                .expect("scroll position should be restored after reload");
            assert_eq!(scroll.item_ix, 5);
            assert_eq!(scroll.offset_in_item, gpui::px(12.5));
        });
    }

    #[gpui::test]
    async fn test_close_session_saves_thread(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {
                    "file.txt": "hello"
                }
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        let model = Arc::new(FakeLanguageModel::default());
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
        });

        // Send a message so the thread is non-empty (empty threads aren't saved).
        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        model.send_last_completion_stream_text_chunk("world");
        model.end_last_completion_stream();
        send.await.unwrap();
        cx.run_until_parked();

        // Set a draft prompt WITHOUT calling run_until_parked afterwards.
        // This means no observe-triggered save has run for this change.
        // The only way this data gets persisted is if close_session
        // itself performs the save.
        let draft_blocks = vec![acp::ContentBlock::Text(acp::TextContent::new(
            "unsaved draft",
        ))];
        acp_thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(Some(draft_blocks.clone()), cx);
        });

        // Close the session immediately — no run_until_parked in between.
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        cx.run_until_parked();

        // Reopen and verify the draft prompt was saved.
        let reloaded = agent
            .update(cx, |agent, cx| {
                agent.open_thread(session_id.clone(), project.clone(), cx)
            })
            .await
            .unwrap();
        reloaded.read_with(cx, |thread, _| {
            assert_eq!(
                thread.draft_prompt(),
                Some(draft_blocks.as_slice()),
                "close_session must save the thread; draft prompt was lost"
            );
        });
    }

    #[gpui::test]
    async fn test_flush_threads_on_quit_preserves_normal_stream_partial(cx: &mut TestAppContext) {
        let result: Result<()> = async {
            init_test(cx);
            let (_connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
            let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
            let model = Arc::new(FakeLanguageModel::default());
            let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
            thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
            let send = acp_thread.update(cx, |thread, cx| {
                thread.send(vec!["ordinary streaming response".into()], cx)
            });
            let send = cx.foreground_executor().spawn(send);
            cx.run_until_parked();
            let request = model
                .pending_completions()
                .pop()
                .context("normal turn must request a completion")?;
            model.send_completion_stream_text_chunk(&request, "partial response before quit");
            cx.run_until_parked();
            assert!(agent.read_with(cx, |agent, _| {
                agent
                    .sessions
                    .get(&session_id)
                    .is_some_and(|session| session.active_grind.is_none())
            }));
            let pending = thread.read_with(cx, |thread, _| thread.last_received_or_pending_message());
            assert!(serde_json::to_string(&pending)?.contains("partial response before quit"));

            cx.update(|cx| cx.shutdown());
            let database = cx
                .update(|cx| ThreadsDatabase::connect(cx))
                .await
                .map_err(|error| anyhow!("{error:#}"))?;
            let stored = database
                .load_thread(session_id)
                .await?
                .context("quit flush must persist the normal session")?;
            model.end_completion_stream(&request);
            let response = send.await?.context("normal prompt must return a response")?;
            assert!(
                serde_json::to_string(&stored.messages)?.contains("partial response before quit"),
                "quit must flush the displayed normal-turn partial before taking the database snapshot"
            );
            assert_eq!(response.stop_reason, acp::StopReason::Cancelled);
            Ok(())
        }
        .await;
        assert!(
            result.is_ok(),
            "normal stream quit fixture failed: {result:?}"
        );
    }

    #[gpui::test]
    async fn test_flush_threads_on_quit_cancels_grind_and_saves_final_snapshot(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        send_native_command_for_test(&connection, &session_id, "/goal save on quit", cx)
            .await
            .unwrap();
        acp_thread.update(cx, |thread, cx| {
            thread.set_draft_prompt(Some(vec!["quit draft".into()]), cx);
        });

        let grind = start_grind_for_test(&connection, &session_id, "/grind", cx);
        cx.run_until_parked();
        let request = model.pending_completions().pop().unwrap();
        model.send_completion_stream_text_chunk(&request, "final partial output");
        cx.run_until_parked();

        cx.update(|cx| cx.shutdown());
        model.end_completion_stream(&request);
        assert_eq!(grind.await.unwrap().stop_reason, acp::StopReason::Cancelled);

        let database = cx.update(|cx| ThreadsDatabase::connect(cx)).await.unwrap();
        let stored = database
            .load_thread(session_id.clone())
            .await
            .unwrap()
            .expect("quit flush should persist the active grind snapshot");
        assert_eq!(stored.goal.as_deref(), Some("save on quit"));
        assert_eq!(
            stored.draft_prompt,
            Some(vec![acp::ContentBlock::from("quit draft")])
        );
        assert!(
            serde_json::to_string(&stored.messages)
                .unwrap()
                .contains("final partial output")
        );
        assert!(agent.read_with(cx, |agent, _| {
            agent.sessions[&session_id].active_grind.is_none()
        }));
    }

    #[gpui::test]
    async fn test_thread_summary_releases_loaded_session(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {
                    "file.txt": "hello"
                }
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        let model = Arc::new(FakeLanguageModel::default());
        let summary_model = Arc::new(FakeLanguageModel::default());
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.set_summarization_model(Some(summary_model.clone()), cx);
        });

        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        model.send_last_completion_stream_text_chunk("world");
        model.end_last_completion_stream();
        send.await.unwrap();
        cx.run_until_parked();

        let summary = agent.update(cx, |agent, cx| {
            agent.thread_summary(session_id.clone(), project.clone(), cx)
        });
        cx.run_until_parked();

        summary_model.send_last_completion_stream_text_chunk("summary");
        summary_model.end_last_completion_stream();

        assert_eq!(summary.await.unwrap(), "summary");
        cx.run_until_parked();

        agent.read_with(cx, |agent, _| {
            let session = agent
                .sessions
                .get(&session_id)
                .expect("thread_summary should not close the active session");
            assert_eq!(
                session.ref_count, 1,
                "thread_summary should release its temporary session reference"
            );
        });

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        cx.run_until_parked();

        agent.read_with(cx, |agent, _| {
            assert!(
                agent.sessions.is_empty(),
                "closing the active session after thread_summary should unload it"
            );
        });
    }

    #[gpui::test]
    async fn test_loaded_sessions_keep_state_until_last_close(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/",
            json!({
                "a": {
                    "file.txt": "hello"
                }
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        let model = cx.update(|cx| {
            LanguageModelRegistry::read_global(cx)
                .default_model()
                .map(|default_model| default_model.model)
                .expect("default test model should be available")
        });
        let fake_model = model.as_fake();
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
        });

        let send = acp_thread.update(cx, |thread, cx| thread.send(vec!["hello".into()], cx));
        let send = cx.foreground_executor().spawn(send);
        cx.run_until_parked();

        fake_model.send_last_completion_stream_text_chunk("world");
        fake_model.end_last_completion_stream();
        send.await.unwrap();
        cx.run_until_parked();

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();
        drop(thread);
        drop(acp_thread);
        agent.read_with(cx, |agent, _| {
            assert!(agent.sessions.is_empty());
        });

        let first_loaded_thread = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project.clone(),
                PathList::new(&[Path::new("")]),
                None,
                cx,
            )
        });
        let second_loaded_thread = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project.clone(),
                PathList::new(&[Path::new("")]),
                None,
                cx,
            )
        });

        let first_loaded_thread = first_loaded_thread.await.unwrap();
        let second_loaded_thread = second_loaded_thread.await.unwrap();

        cx.run_until_parked();

        assert_eq!(
            first_loaded_thread.entity_id(),
            second_loaded_thread.entity_id(),
            "concurrent loads for the same session should share one AcpThread"
        );

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();

        agent.read_with(cx, |agent, _| {
            assert!(
                agent.sessions.contains_key(&session_id),
                "closing one loaded session should not drop shared session state"
            );
        });

        let follow_up = second_loaded_thread.update(cx, |thread, cx| {
            thread.send(vec!["still there?".into()], cx)
        });
        let follow_up = cx.foreground_executor().spawn(follow_up);
        cx.run_until_parked();

        fake_model.send_last_completion_stream_text_chunk("yes");
        fake_model.end_last_completion_stream();
        follow_up.await.unwrap();
        cx.run_until_parked();

        second_loaded_thread.read_with(cx, |thread, cx| {
            assert_eq!(
                thread.to_markdown(cx),
                formatdoc! {"
                    ## User

                    hello

                    ## Assistant

                    world

                    ## User

                    still there?

                    ## Assistant

                    yes

                "}
            );
        });

        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .unwrap();

        cx.run_until_parked();

        drop(first_loaded_thread);
        drop(second_loaded_thread);
        agent.read_with(cx, |agent, _| {
            assert!(agent.sessions.is_empty());
        });
    }

    #[gpui::test]
    async fn test_pending_session_failure_clears_and_retries(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| ThreadsDatabase::fail_connections_for_test(cx, "load unavailable"));
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [path!("/a").as_ref()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent =
            cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));
        let session_id = acp::SessionId::new(Arc::<str>::from("failed-session"));

        let first = agent.update(cx, |agent, cx| {
            agent.open_thread(session_id.clone(), project.clone(), cx)
        });
        agent.read_with(cx, |agent, _cx| {
            let pending = agent.pending_sessions.get(&session_id).unwrap();
            assert_eq!(agent.pending_sessions.len(), 1);
            assert_eq!(pending.ref_count, 1);
        });
        let second = agent.update(cx, |agent, cx| {
            agent.open_thread(session_id.clone(), project.clone(), cx)
        });
        agent.read_with(cx, |agent, _cx| {
            let pending = agent.pending_sessions.get(&session_id).unwrap();
            assert_eq!(agent.pending_sessions.len(), 1);
            assert_eq!(pending.ref_count, 2);
        });

        let first_error = first.await.unwrap_err().to_string();
        let second_error = second.await.unwrap_err().to_string();
        assert!(first_error.contains("load unavailable"));
        assert_eq!(second_error, first_error);
        agent.read_with(cx, |agent, _cx| {
            assert!(!agent.pending_sessions.contains_key(&session_id));
            assert!(!agent.sessions.contains_key(&session_id));
        });

        let retry = agent.update(cx, |agent, cx| {
            agent.open_thread(session_id.clone(), project, cx)
        });
        agent.read_with(cx, |agent, _cx| {
            let pending = agent.pending_sessions.get(&session_id).unwrap();
            assert_eq!(agent.pending_sessions.len(), 1);
            assert_eq!(pending.ref_count, 1);
        });
        assert!(
            retry
                .await
                .unwrap_err()
                .to_string()
                .contains("load unavailable")
        );
        agent.read_with(cx, |agent, _cx| {
            assert!(!agent.pending_sessions.contains_key(&session_id));
            assert!(!agent.sessions.contains_key(&session_id));
        });
    }

    #[gpui::test]
    async fn test_rapid_title_changes_do_not_loop(cx: &mut TestAppContext) {
        // Regression test: rapid title changes must not cause a propagation loop
        // between Thread and AcpThread via handle_thread_title_updated.
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/", json!({ "a": {} })).await;
        let project = Project::test(fs.clone(), [], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx
            .update(|cx| NativeAgent::new(thread_store.clone(), Templates::new(), fs.clone(), cx));
        let connection = Rc::new(NativeAgentConnection(agent.clone()));

        let acp_thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), PathList::new(&[Path::new("")]), cx)
            })
            .await
            .unwrap();

        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let thread = agent.read_with(cx, |agent, _| {
            agent.sessions.get(&session_id).unwrap().thread.clone()
        });

        let title_updated_count = Rc::new(std::cell::RefCell::new(0usize));
        cx.update(|cx| {
            let count = title_updated_count.clone();
            cx.subscribe(
                &thread,
                move |_entity: Entity<Thread>, _event: &TitleUpdated, _cx: &mut App| {
                    let new_count = {
                        let mut count = count.borrow_mut();
                        *count += 1;
                        *count
                    };
                    assert!(
                        new_count <= 2,
                        "TitleUpdated fired {new_count} times; \
                         title updates are looping"
                    );
                },
            )
            .detach();
        });

        thread.update(cx, |thread, cx| thread.set_title("first".into(), cx));
        thread.update(cx, |thread, cx| thread.set_title("second".into(), cx));

        cx.run_until_parked();

        thread.read_with(cx, |thread, _| {
            assert_eq!(thread.title(), Some("second".into()));
        });
        acp_thread.read_with(cx, |acp_thread, _| {
            assert_eq!(acp_thread.title(), Some("second".into()));
        });

        assert_eq!(*title_updated_count.borrow(), 2);
    }

    fn thread_entries(
        thread_store: &Entity<ThreadStore>,
        cx: &mut TestAppContext,
    ) -> Vec<(acp::SessionId, String)> {
        thread_store.read_with(cx, |store, _| {
            store
                .entries()
                .map(|entry| (entry.id.clone(), entry.title.to_string()))
                .collect::<Vec<_>>()
        })
    }

    #[gpui::test]
    async fn recipe_discovery_validates_files_and_respects_trust(cx: &mut TestAppContext) {
        use project::trusted_worktrees::{self, PathTrust};

        init_test(cx);
        cx.update(|cx| trusted_worktrees::init(HashMap::default(), cx));
        let fs = FakeFs::new(cx.executor());
        let yaml = "version: 1\ntitle: Review\ndescription: Review changes\nprompt: Review the current changes\n";
        fs.insert_tree(
            "/project",
            json!({".agents": {"recipes": {
                "review.yaml": yaml,
                "broken.yml": "version: invalid",
                "ignored.txt": yaml
            }}}),
        )
        .await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), [Path::new("/project")], cx).await;
        let entries = cx
            .update(|cx| recipes::discover_recipes(project.clone(), fs.clone(), cx))
            .await;
        assert!(entries.iter().all(|entry| entry.path.is_none()));
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("worktree")
                .read(cx)
                .id()
        });
        cx.update(|cx| {
            TrustedWorktrees::try_get_global(cx)
                .expect("trust store")
                .update(cx, |trusted, cx| {
                    trusted.trust(
                        &project.read(cx).worktree_store(),
                        HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                        cx,
                    );
                });
        });
        let entries = cx
            .update(|cx| recipes::discover_recipes(project.clone(), fs.clone(), cx))
            .await;
        let local = entries
            .iter()
            .filter(|entry| entry.path.is_some())
            .collect::<Vec<_>>();
        assert_eq!(local.len(), 2);
        assert_eq!(local.iter().filter(|entry| entry.recipe.is_ok()).count(), 1);
        assert!(
            local
                .iter()
                .any(|entry| entry.recipe.is_err() && entry.origin.ends_with("broken.yml"))
        );
        fs.insert_file(
            "/project/.agents/recipes/review.yaml",
            "version: invalid".into(),
        )
        .await;
        cx.run_until_parked();
        let entries = cx
            .update(|cx| recipes::discover_recipes(project.clone(), fs.clone(), cx))
            .await;
        assert!(
            entries
                .iter()
                .filter(|entry| entry.path.is_some())
                .all(|entry| entry.recipe.is_err())
        );
        cx.update(|cx| {
            TrustedWorktrees::try_get_global(cx)
                .expect("trust store")
                .update(cx, |trusted, cx| {
                    trusted.restrict(
                        project.read(cx).worktree_store().downgrade(),
                        HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                        cx,
                    );
                });
        });
        let entries = cx
            .update(|cx| recipes::discover_recipes(project, fs, cx))
            .await;
        assert!(entries.iter().all(|entry| entry.path.is_none()));
    }

    #[gpui::test]
    async fn recipe_discovery_rejects_symlinks_outside_worktree(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({".agents": {"recipes": {}}}))
            .await;
        fs.insert_tree("/outside", json!({"private.yaml": "version: 1\ntitle: Private\ndescription: Private\nprompt: MUST_NOT_LOAD\n"})).await;
        fs.insert_symlink(
            "/project/.agents/recipes/escape.yaml",
            PathBuf::from("/outside/private.yaml"),
        )
        .await;
        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;
        let entries = cx
            .update(|cx| recipes::discover_recipes(project, fs, cx))
            .await;
        let escaped = entries
            .iter()
            .find(|entry| entry.origin.ends_with("escape.yaml"))
            .expect("source-labelled error");
        assert!(escaped.recipe.is_err());
        assert!(
            !escaped
                .recipe
                .as_ref()
                .expect_err("symlink must not load")
                .contains("MUST_NOT_LOAD")
        );
    }

    #[gpui::test]
    async fn recipe_execution_uses_native_turns_and_propagates_failures(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let recipe = recipes::Recipe::from_yaml("version: 1\ntitle: Review\ndescription: Review\nprompt: '/clear is literal recipe content {{focus}}'\nparameters:\n  - name: focus\n    description: Focus\n    required: true\n").expect("recipe");
        let invalid = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        assert!(invalid.await.is_err());
        assert!(model.pending_completions().is_empty());
        let parameters = HashMap::from_iter([("focus".to_string(), "current changes".to_string())]);
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &parameters,
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("provider request");
        assert!(
            request
                .messages
                .last()
                .expect("recipe prompt")
                .string_contents()
                .contains("/clear is literal recipe content current changes")
        );
        model.send_completion_stream_text_chunk(&request, "Review complete");
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        assert_eq!(
            run.await.expect("recipe completion").stop_reason,
            acp::StopReason::EndTurn
        );
        let failed = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &parameters,
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("provider request");
        model.send_completion_stream_error(&request, LanguageModelCompletionError::PaymentRequired);
        model.end_completion_stream(&request);
        cx.run_until_parked();
        assert!(failed.await.is_err());
        let cancelled = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &parameters,
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        cx.update(|cx| connection.cancel(&session_id, cx));
        cx.run_until_parked();
        assert_eq!(
            cancelled.await.expect("cancellation response").stop_reason,
            acp::StopReason::Cancelled
        );
        assert_eq!(agent.read_with(cx, |agent, _| agent.sessions.len()), 1);
    }

    #[gpui::test]
    async fn recipe_scan_saved_transcript_cannot_resume_or_dispatch_commands(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, _| {
            thread.recipe_scan_policy = Some(Default::default())
        });
        assert!(cx.update(|cx| connection.retry(&session_id, cx).is_none()));
        assert!(cx.update(|cx| connection.truncate(&session_id, cx).is_none()));
        assert!(!thread.read_with(cx, |thread, _| thread.can_clear_conversation()));
        let active_prompt = cx
            .update(|cx| {
                acp_thread::AgentConnection::prompt(
                    connection.as_ref(),
                    acp::PromptRequest::new(session_id.clone(), vec!["/clear".into()]),
                    cx,
                )
            })
            .await;
        assert!(
            active_prompt
                .expect_err("workflow owns active test prompts")
                .to_string()
                .contains("reviewed recipe")
        );
        let mut saved = thread.read_with(cx, |thread, cx| thread.to_db(cx)).await;
        saved.recipe_scan = true;
        let store = agent.read_with(cx, |agent, _| agent.thread_store.clone());
        store
            .update(cx, |store, cx| {
                store.save_thread(session_id.clone(), saved, PathList::new(&["/a"]), cx)
            })
            .await
            .expect("save marker");
        cx.update(|cx| connection.clone().close_session(&session_id, cx))
            .await
            .expect("close session");
        let reopened = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project,
                    PathList::new(&["/a"]),
                    None,
                    cx,
                )
            })
            .await
            .expect("load transcript");
        let before = reopened.read_with(cx, |thread, _| thread.entries().len());
        for prompt in ["Continue working", "/clear", "/recipes", "/schedules"] {
            let result = cx
                .update(|cx| {
                    acp_thread::AgentConnection::prompt(
                        connection.as_ref(),
                        acp::PromptRequest::new(session_id.clone(), vec![prompt.into()]),
                        cx,
                    )
                })
                .await;
            assert!(
                result
                    .expect_err("read-only history")
                    .to_string()
                    .contains("read-only")
            );
        }
        assert_eq!(
            reopened.read_with(cx, |thread, _| thread.entries().len()),
            before
        );
        let native = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        let before = native.read_with(cx, |thread, _| thread.recipe_message_count());
        assert!(
            native
                .update(cx, |thread, cx| thread.send(
                    ClientUserMessageId::new(),
                    vec!["Direct replay".to_owned()],
                    cx
                ))
                .is_err()
        );
        assert_eq!(
            native.read_with(cx, |thread, _| thread.recipe_message_count()),
            before
        );
        assert!(native.update(cx, |thread, cx| thread.resume(cx)).is_err());
        assert!(
            native
                .update(cx, |thread, cx| thread.send_existing(cx))
                .is_err()
        );
        assert!(cx.update(|cx| connection.retry(&session_id, cx).is_none()));
        assert!(cx.update(|cx| connection.truncate(&session_id, cx).is_none()));
        assert_eq!(
            native.read_with(cx, |thread, _| thread.recipe_message_count()),
            before
        );
        assert!(
            native
                .read_with(cx, |thread, cx| thread.to_db(cx))
                .await
                .recipe_scan
        );
    }

    #[gpui::test]
    async fn recipe_scan_automatic_save_keeps_original_project_identity(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| {
            let mut policy = crate::recipe_scan::RecipeScanSessionPolicy::default();
            policy.project_paths = Some(PathList::new(&["/original-project"]).serialize());
            policy.run_id = Some(Uuid::new_v4());
            thread.recipe_scan_policy = Some(policy);
            thread.push_acp_agent_block("Observed test transcript".into(), cx);
        });
        cx.update(|cx| connection.close_session(&session_id, cx))
            .await
            .expect("save and close native session");
        let store = agent.read_with(cx, |agent, _| agent.thread_store.clone());
        store.read_with(cx, |store, _| store.reload_task()).await;
        store.read_with(cx, |store, _| {
            let entry = store
                .thread_from_session_id(&session_id)
                .expect("saved metadata");
            assert_eq!(entry.folder_paths, PathList::new(&["/original-project"]));
        });
        assert!(
            store
                .update(cx, |store, cx| store.load_thread(session_id, cx))
                .await
                .expect("load")
                .expect("transcript")
                .recipe_scan
        );
    }

    #[gpui::test]
    async fn recipe_scan_dropped_persistence_waiter_does_not_abandon_commit(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(recipes::RecipeStore::init_global);
        cx.run_until_parked();
        let owner = cx.update(|cx| recipes::RecipeStore::try_global(cx).expect("recipe owner"));
        let id = Uuid::new_v4();
        let save = owner.update(cx, |owner, cx| {
            owner.transact(
                move |state, now| {
                    state.runs.push(recipe_schedule::RecipeRun {
                        id,
                        schedule_id: Uuid::new_v4(),
                        scheduled_for: now,
                        started_at: None,
                        finished_at: Some(now),
                        session_id: None,
                        status: recipe_schedule::RecipeRunStatus::Missed,
                        detail: Some("Cancelled waiter fixture".into()),
                    });
                    Ok(())
                },
                cx,
            )
        });
        drop(save);
        cx.run_until_parked();
        assert!(owner.read_with(cx, |owner, _| !owner.busy
            && owner.state.runs.iter().any(|run| run.id == id)));
        let database = cx
            .update(crate::ThreadsDatabase::connect)
            .await
            .expect("database");
        assert!(
            database
                .load_recipe_state()
                .await
                .expect("durable state")
                .1
                .runs
                .iter()
                .any(|run| run.id == id)
        );
        assert_eq!(
            owner
                .update(cx, |owner, cx| owner.transact(|_, _| Ok(42), cx))
                .await
                .expect("subsequent commit"),
            42
        );
    }

    #[gpui::test]
    #[ignore = "requires a local Docker engine, immutable image, and matching Linux Made server"]
    async fn recipe_scan_real_native_workflow(cx: &mut TestAppContext) {
        use crate::recipe_scan::{RecipeScanOutcome, RecipeScanRequest};
        init_test(cx);
        let case = std::env::var("MADE_RECIPE_TEST_CASE").unwrap_or_else(|_| "success".into());
        let expected = match case.as_str() {
            "success" => RecipeScanOutcome::Succeeded,
            "failure" => RecipeScanOutcome::Failed,
            "blocked" => RecipeScanOutcome::Incomplete,
            "cancel" => RecipeScanOutcome::Cancelled,
            "timeout" => RecipeScanOutcome::TimedOut,
            _ => panic!("unknown test scenario"),
        };
        cx.background_executor.allow_parking();
        let root = tempfile::tempdir().expect("disposable source fixture");
        std::fs::write(root.path().join("sentinel"), "original").expect("host sentinel");
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(root.path(), json!({"sentinel": "original"}))
            .await;
        let project = Project::test(fs.clone(), [root.path()], cx).await;
        let thread_store = cx.new(|cx| ThreadStore::new(cx));
        let agent = cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs, cx));
        cx.update(|cx| {
            release_channel::init(
                remote_server::VERSION
                    .parse()
                    .unwrap_or_else(|_| semver::Version::new(0, 0, 0)),
                cx,
            );
            release_channel::AppCommitSha::set_global(
                release_channel::AppCommitSha::new(
                    std::env::var("MADE_RECIPE_TEST_COMMIT").expect("matching build commit"),
                ),
                cx,
            );
            recipes::RecipeStore::init_global(cx);
        });
        cx.run_until_parked();
        let owner = cx.update(|cx| recipes::RecipeStore::try_global(cx).expect("recipe owner"));
        let model = Arc::new(FakeLanguageModel::default());
        let mut recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Container branch\ndescription: Real native workflow fixture\nsteps:\n- id: review\n  prompt: Review the isolated copy\n  next: report\n- id: unvisited\n  prompt: Must not execute\n- id: report\n  prompt: 'Report {{steps.review.output}}'\n  when:\n    operator: equals\n    left: {kind: status, step: review}\n    right: {kind: literal, value: succeeded}\n").expect("workflow");
        if matches!(case.as_str(), "blocked" | "cancel" | "timeout") {
            recipe.steps.first_mut().expect("first step").on_failure =
                crate::recipe_workflow::FailurePolicy::Continue;
        }
        let request = RecipeScanRequest {
            project: project.clone(),
            agent: agent.clone(),
            worktree: root.path().into(),
            recipe,
            parameters: HashMap::default(),
            selected_files: vec!["sentinel".into()],
            model: model.clone(),
            image: std::env::var("MADE_RECIPE_TEST_IMAGE").expect("immutable local image"),
            endpoint: std::env::var("MADE_RECIPE_TEST_DOCKER_HOST").expect("local endpoint"),
            server_binary: std::env::var("MADE_RECIPE_TEST_SERVER")
                .expect("local Linux server")
                .into(),
        };
        let prepared = owner
            .update(cx, |owner, cx| owner.prepare_scan(request, cx))
            .await
            .expect("review preparation");
        owner
            .update(cx, |owner, cx| owner.run_scan(prepared, true, cx))
            .expect("explicit test launch");
        let run = owner.update(cx, |owner, _| {
            owner.scan_task.take().expect("owned scan task")
        });
        let response_model = model.clone();
        let responses = std::rc::Rc::new(std::cell::Cell::new(0));
        let response_count = responses.clone();
        let response_owner = owner.clone();
        let observed = Rc::new(std::cell::RefCell::new(String::new()));
        let observed_output = observed.clone();
        let permissions = Rc::new(std::cell::Cell::new(0));
        let permission_count = permissions.clone();
        let response_case = case.clone();
        let responder = cx.spawn(async move |mut cx| {
            loop {
                if let Some(request) = response_model.pending_completions().pop() {
                    let index = response_count.get();
                    assert!(index < 3, "unvisited branch or unexpected auxiliary request");
                    if index == 2 {
                        assert_eq!(request.messages.last().expect("step prompt").string_contents(), "Report Observed review");
                    }
                    if index == 0 {
                        let input = if response_case == "blocked" {
                            json!({"url": "https://example.com/required-dependency"})
                        } else {
                            json!({"command": match response_case.as_str() { "failure" => "exit 7", "cancel" => "sleep 120 & echo MADE_SCAN_CHILD_RUNNING; wait", "timeout" => "sleep 120", _ => "id -u; cat sentinel; echo changed > sentinel; echo MADE_CONTAINER_TOOL_OK" }, "cd": "/workspace", "timeout_ms": if response_case == "timeout" { 100 } else { 10000 }})
                        };
                        response_model.send_completion_stream_event(&request, LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                            id: "isolated-terminal".into(), name: if response_case == "blocked" { "fetch" } else { "terminal" }.into(), raw_input: input.to_string(),
                            input: language_model::LanguageModelToolUseInput::Json(input), is_input_complete: true, thought_signature: None,
                        }));
                    } else {
                        if index == 1 {
                            *observed_output.borrow_mut() = request.messages.iter().map(|message| message.string_contents()).collect::<Vec<_>>().join("\n");

                        }
                        response_model.send_completion_stream_text_chunk(&request, if index == 1 { "Observed review" } else { "Reported" });
                        response_model.send_completion_stream_event(&request, LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn));
                    }
                    response_model.end_completion_stream(&request);
                    response_count.set(index + 1);
                }
                if let Some(session) = response_owner.read_with(&cx, |owner, _| owner.scan.as_ref().and_then(|scan| scan.session.clone())) {
                    let child_started = session.update(&mut cx, |session, cx| {
                        let pending = session.entries().iter().filter_map(|entry| match entry {
                            acp_thread::AgentThreadEntry::ToolCall(call) if matches!(call.status, acp_thread::ToolCallStatus::WaitingForConfirmation { .. }) => Some(call.id.clone()),
                            _ => None,
                        }).collect::<Vec<_>>();
                        for id in pending {
                            permission_count.set(permission_count.get() + 1);
                            session.authorize_tool_call(id, acp_thread::SelectedPermissionOutcome::new("allow".into(), acp::PermissionOptionKind::AllowOnce), cx);
                        }
                        session.entries().iter().any(|entry| match entry {
                            acp_thread::AgentThreadEntry::ToolCall(call) => call.terminals().any(|terminal| terminal.read(cx).to_markdown(cx).contains("MADE_SCAN_CHILD_RUNNING")),
                            _ => false,
                        })
                    });
                    if response_case == "cancel" && child_started {
                        response_owner.update(&mut cx, |owner, cx| owner.cancel_scan(cx));
                        break;
                    }
                }
                cx.background_executor().timer(std::time::Duration::from_millis(20)).await;
            }
        });
        run.await;
        drop(responder);
        owner.read_with(cx, |owner, _| {
            let scan = owner.scan.as_ref().expect("scan result");
            assert_eq!(scan.outcome, Some(expected), "{:?}", scan.detail);
            assert!(scan.cleanup_error.is_none(), "{:?}", scan.cleanup_error);
            assert!(scan.session.is_some());
            assert!(!scan.observations.borrow().is_empty());
            assert!(
                owner.state.schedules.is_empty(),
                "test must never create or enable schedules"
            );
        });
        assert_eq!(
            responses.get(),
            match case.as_str() {
                "success" => 3,
                "failure" => 2,
                _ => 1,
            }
        );
        assert_eq!(
            permissions.get(),
            usize::from(case != "blocked"),
            "terminal requires explicit one-time permission"
        );
        if case == "success" {
            let observed = observed.borrow();
            assert!(
                observed.contains("1000\noriginalMADE_CONTAINER_TOOL_OK"),
                "Missing actual isolated tool output: {observed}"
            );
        }
        assert_eq!(
            std::fs::read_to_string(root.path().join("sentinel")).expect("original"),
            "original"
        );
        assert!(agent.read_with(cx, |agent, _| agent.sessions.is_empty()));
        let record = owner.read_with(cx, |owner, _| {
            owner
                .state
                .scans
                .last()
                .expect("durable scan history")
                .clone()
        });
        assert_eq!(record.outcome, Some(expected));
        assert!(!record.cleanup_pending);
        assert!(record.finished_at.is_some());
        let session_id =
            acp::SessionId::new(record.session_id.clone().expect("durable transcript link"));
        let database = cx
            .update(crate::ThreadsDatabase::connect)
            .await
            .expect("existing database");
        let (_, restored) = database.load_recipe_state().await.expect("reload history");
        assert_eq!(
            restored.scans.last().expect("saved test").outcome,
            Some(expected)
        );
        let saved = database
            .load_thread(session_id.clone())
            .await
            .expect("load native transcript")
            .expect("native transcript");
        assert!(saved.recipe_scan, "persist the no-replay marker");
        let connection = Rc::new(NativeAgentConnection(agent.clone()));
        let history = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[root.path()]),
                    None,
                    cx,
                )
            })
            .await
            .expect("read test history");
        let before = history.read_with(cx, |history, _| history.entries().len());
        let continuation = cx
            .update(|cx| {
                acp_thread::AgentConnection::prompt(
                    connection.as_ref(),
                    acp::PromptRequest::new(session_id, vec!["/clear".into()]),
                    cx,
                )
            })
            .await;
        assert!(
            continuation
                .expect_err("history cannot dispatch commands")
                .to_string()
                .contains("read-only")
        );
        assert_eq!(
            history.read_with(cx, |history, _| history.entries().len()),
            before
        );
        if case == "success" {
            use crate::recipe_scan::{PreparedScanCopy, ScanContainerIdentity, ScanDockerClient};
            let client = ScanDockerClient::from_reviewed_endpoint(record.endpoint.clone())
                .expect("reviewed local Docker");
            let identity =
                ScanContainerIdentity::new(Uuid::new_v4(), record.identity.image.clone())
                    .expect("recovery identity");
            let copy = PreparedScanCopy::capture(root.path(), &["sentinel".into()])
                .expect("disposable recovery copy");
            let preflight = record.preflight.clone();
            let mut interrupted = record.clone();
            interrupted.identity = identity.clone();
            interrupted.outcome = None;
            interrupted.finished_at = None;
            interrupted.cleanup_pending = true;
            owner
                .update(cx, |owner, cx| {
                    owner.transact(
                        move |state, _| {
                            state.scans.push(interrupted);
                            Ok(())
                        },
                        cx,
                    )
                })
                .await
                .expect("durable admission before resource creation");
            let completions = model.completion_count();
            let recovery: anyhow::Result<()> = async {
                client
                    .create_verified(&identity, &preflight, &cx.background_executor)
                    .await?;
                let container = client
                    .start_with_copy(&identity, &preflight, &copy, &cx.background_executor)
                    .await?;
                client
                    .command(
                        &[
                            "exec".into(),
                            "--detach".into(),
                            "--user=1000:1000".into(),
                            container.clone(),
                            "/bin/sh".into(),
                            "-c".into(),
                            "sleep 120 & wait".into(),
                        ],
                        None,
                        &cx.background_executor,
                    )
                    .await?;
                let processes = client
                    .command(
                        &["top".into(), container, "-eo".into(), "pid,args".into()],
                        None,
                        &cx.background_executor,
                    )
                    .await?;
                anyhow::ensure!(
                    String::from_utf8_lossy(&processes).contains("sleep 120"),
                    "Recovery fixture must have a live child process"
                );
                let mut wrong_engine = preflight.clone();
                wrong_engine.engine_id = "unrelated-engine".into();
                anyhow::ensure!(
                    client
                        .remove_owned(&identity, &wrong_engine, &cx.background_executor)
                        .await
                        .is_err(),
                    "Never clean a different engine"
                );
                client
                    .inspect_owned(&identity, &preflight, &cx.background_executor)
                    .await?;
                owner
                    .update(cx, |owner, cx| {
                        owner.transact(|state, now| state.recover(now), cx)
                    })
                    .await?;
                owner.update(cx, |owner, cx| owner.recover_scans(cx));
                let cleanup = owner.update(cx, |owner, _| {
                    owner.scan_task.take().expect("owned restart cleanup")
                });
                cleanup.await;
                owner.read_with(cx, |owner, _| {
                    let saved = owner
                        .state
                        .scans
                        .iter()
                        .find(|saved| saved.identity.run_id == identity.run_id)
                        .expect("recovered record");
                    assert_eq!(saved.outcome, Some(RecipeScanOutcome::Interrupted));
                    assert!(!saved.cleanup_pending, "{:?}", saved.cleanup_error);
                });
                anyhow::ensure!(
                    model.completion_count() == completions,
                    "Recovery must never replay provider work"
                );
                let remaining = client
                    .command(
                        &[
                            "container".into(),
                            "ls".into(),
                            "--all".into(),
                            "--quiet".into(),
                            "--filter".into(),
                            format!("label=dev.made.recipe-test={}", identity.run_id),
                        ],
                        None,
                        &cx.background_executor,
                    )
                    .await?;
                anyhow::ensure!(
                    remaining.is_empty(),
                    "The process tree's container must be removed"
                );
                Ok(())
            }
            .await;
            client
                .remove_owned(&identity, &preflight, &cx.background_executor)
                .await
                .expect("fixture cleanup");
            recovery.expect("durable recovery removes only owned resources and child processes");
        }
    }

    #[gpui::test]
    async fn recipe_workflow_native_branches_and_literal_outputs(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Branch\ndescription: Native branch test\nsteps:\n- id: review\n  prompt: Review\n  next: report\n- id: unvisited\n  prompt: Must not execute\n- id: report\n  prompt: 'Report {{steps.review.output}}'\n  when:\n    operator: equals\n    left: {kind: status, step: review}\n    right: {kind: literal, value: succeeded}\n").expect("workflow");
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("first step");
        model.send_completion_stream_text_chunk(&request, "{{literal}} $(not-executed)");
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("report step");
        assert_eq!(
            request.messages.last().expect("prompt").string_contents(),
            "Report {{literal}} $(not-executed)"
        );
        model.send_completion_stream_text_chunk(&request, "Reported");
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        assert_eq!(
            run.await.expect("workflow completed").stop_reason,
            acp::StopReason::EndTurn
        );
        assert!(thread.read_with(cx, |thread, _| thread.recipe_permission_denied.is_none()));
        assert_eq!(agent.read_with(cx, |agent, _| agent.sessions.len()), 1);
    }

    #[gpui::test]
    async fn recipe_workflow_timeout_and_cancellation_stop_later_steps(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Bound\ndescription: Bound test\nsteps:\n- id: slow\n  prompt: Wait\n  on_failure: continue\n- id: later\n  prompt: Must not execute\n").expect("workflow");
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        assert_eq!(model.pending_completions().len(), 1);
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(301));
        cx.run_until_parked();
        assert!(
            run.await
                .expect_err("timeout")
                .to_string()
                .contains("timed out")
        );
        assert!(thread.read_with(cx, |thread, _| thread.recipe_permission_denied.is_none()));
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        cx.update(|cx| connection.cancel(&session_id, cx));
        cx.run_until_parked();
        assert_eq!(
            run.await.expect("cancelled").stop_reason,
            acp::StopReason::Cancelled
        );
    }

    #[gpui::test]
    async fn recipe_workflow_interrupts_stalled_persistence(cx: &mut TestAppContext) {
        init_test(cx);
        for after_first_step in [false, true] {
            for time_out in [false, true] {
                let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
                let session_id = session.read_with(cx, |session, _| session.session_id().clone());
                let model = Arc::new(FakeLanguageModel::default());
                let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
                thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
                let store = agent.read_with(cx, |agent, _| agent.thread_store.clone());
                let (release, blocked_save) = futures::channel::oneshot::channel::<()>();
                let barrier = async move { blocked_save.await.expect("release save barrier") }
                    .boxed()
                    .shared();
                if !after_first_step {
                    store.update(cx, |store, _| store.save_barrier = Some(barrier.clone()));
                }
                let recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Persistence\ndescription: Bounded saves\nsteps:\n- id: first\n  prompt: First\n  on_failure: continue\n- id: later\n  prompt: Must not execute\n").expect("workflow");
                let run = cx.update(|cx| {
                    recipes::execute_recipe(
                        &connection,
                        session_id.clone(),
                        ClientUserMessageId::new(),
                        &recipe,
                        &HashMap::default(),
                        Vec::new(),
                        cx,
                    )
                });
                cx.run_until_parked();
                if after_first_step {
                    let request = model
                        .pending_completions()
                        .pop()
                        .expect("first step started");
                    store.update(cx, |store, _| store.save_barrier = Some(barrier));
                    model.send_completion_stream_text_chunk(&request, "First finished");
                    model.send_completion_stream_event(
                        &request,
                        LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
                    );
                    model.end_completion_stream(&request);
                    cx.run_until_parked();
                }
                assert!(model.pending_completions().is_empty());
                if time_out {
                    cx.executor()
                        .advance_clock(std::time::Duration::from_secs(301));
                } else {
                    cx.update(|cx| connection.cancel(&session_id, cx));
                }
                cx.run_until_parked();
                let result = run
                    .now_or_never()
                    .expect("stalled save must not prevent workflow termination");
                if time_out {
                    assert!(
                        result
                            .expect_err("deadline")
                            .to_string()
                            .contains("timed out")
                    );
                } else {
                    assert_eq!(
                        result.expect("cancelled").stop_reason,
                        acp::StopReason::Cancelled
                    );
                }
                thread.read_with(cx, |thread, _| {
                    assert!(thread.recipe_permission_denied.is_none());
                    assert!(thread.recipe_cancellation.is_none());
                });
                release.send(()).expect("release persistence");
                store.update(cx, |store, _| store.save_barrier = None);
                cx.run_until_parked();
                assert!(
                    model.pending_completions().is_empty(),
                    "releasing persistence must not start later steps"
                );
            }
        }
    }

    #[gpui::test]
    async fn recipe_workflow_overall_deadline_includes_between_step_saves(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Overall bound\ndescription: Saves count toward the deadline\nsteps:\n- id: first\n  prompt: First\n- id: second\n  prompt: Second\n- id: third\n  prompt: Third\n- id: fourth\n  prompt: Fourth\n- id: later\n  prompt: Must not execute\n").expect("workflow");
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        let store = agent.read_with(cx, |agent, _| agent.thread_store.clone());
        let (release, blocked_save) = futures::channel::oneshot::channel::<()>();
        let barrier = async move { blocked_save.await.expect("release save barrier") }
            .boxed()
            .shared();
        for index in 0..4 {
            cx.run_until_parked();
            let request = model.pending_completions().pop().expect("step started");
            cx.executor()
                .advance_clock(std::time::Duration::from_secs(if index == 3 {
                    60
                } else {
                    240
                }));
            cx.run_until_parked();
            if index == 3 {
                store.update(cx, |store, _| store.save_barrier = Some(barrier.clone()));
            }
            model.send_completion_stream_text_chunk(&request, "Finished");
            model.send_completion_stream_event(
                &request,
                LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
            );
            model.end_completion_stream(&request);
        }
        cx.run_until_parked();
        assert!(model.pending_completions().is_empty());
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(121));
        cx.run_until_parked();
        let result = run
            .now_or_never()
            .expect("deadline must interrupt persistence");
        assert_eq!(
            result.expect_err("timeout").to_string(),
            "Recipe workflow timed out"
        );
        release.send(()).expect("release persistence");
        store.update(cx, |store, _| store.save_barrier = None);
        cx.run_until_parked();
        assert!(model.pending_completions().is_empty());
    }

    #[gpui::test]
    async fn recipe_workflow_cancellation_before_poll_and_dropped_admission(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Cancel\ndescription: Immediate cancellation\nsteps:\n- id: first\n  prompt: Must not execute\n").expect("workflow");
        let run = cx.update(|cx| {
            let run = recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            );
            connection.cancel(&session_id, cx);
            run
        });
        cx.run_until_parked();
        assert_eq!(
            run.await.expect("cancelled before polling").stop_reason,
            acp::StopReason::Cancelled
        );
        assert!(model.pending_completions().is_empty());
        cx.update(|cx| {
            let run = recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            );
            drop(run);
        });
        cx.run_until_parked();
        thread.read_with(cx, |thread, _| {
            assert!(thread.recipe_permission_denied.is_none());
            assert!(thread.recipe_cancellation.is_none());
            assert!(thread.automatic_retries_enabled());
        });
        assert!(model.pending_completions().is_empty());
    }

    #[gpui::test]
    async fn recipe_workflow_denial_overrides_continue_policy(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.add_tool(GrindPermissionTool);
        });
        enable_test_tools_for_thread(&thread, &[GrindPermissionTool::NAME], cx);
        let recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Denial\ndescription: No bypass\nsteps:\n- id: protected\n  prompt: Use the tool\n  on_failure: continue\n- id: forbidden\n  prompt: Must not run\n").expect("workflow");
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("tool request");
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                id: "deny-workflow".into(),
                name: GrindPermissionTool::NAME.into(),
                raw_input: "{}".into(),
                input: language_model::LanguageModelToolUseInput::Json(json!({})),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        let tool_id = session
            .read_with(cx, |session, _| {
                session.entries().iter().find_map(|entry| {
                    if let acp_thread::AgentThreadEntry::ToolCall(call) = entry {
                        matches!(
                            call.status,
                            acp_thread::ToolCallStatus::WaitingForConfirmation { .. }
                        )
                        .then(|| call.id.clone())
                    } else {
                        None
                    }
                })
            })
            .expect("permission prompt");
        session.update(cx, |session, cx| {
            session.authorize_tool_call(
                tool_id,
                acp_thread::SelectedPermissionOutcome::new(
                    "deny".into(),
                    acp::PermissionOptionKind::RejectOnce,
                ),
                cx,
            )
        });
        cx.run_until_parked();
        assert!(run.await.is_err());
        assert!(
            model.pending_completions().is_empty(),
            "Unexpected completions: {:?}",
            model.pending_completions()
        );
        assert!(thread.read_with(cx, |thread, _| thread.recipe_permission_denied.is_none()));
    }

    #[gpui::test]
    async fn recipe_scan_blocked_dependency_stops_workflow_failure_branches(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| {
            thread.set_model(model.clone(), cx);
            thread.recipe_scan_policy = Some(Default::default());
        });
        let recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Isolated dependency\ndescription: No bypass\nsteps:\n- id: protected\n  prompt: Fetch the required dependency\n  on_failure: continue\n- id: forbidden\n  prompt: Must not run\n").expect("workflow");
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id,
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("workflow request");
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                id: "blocked-dependency".into(),
                name: "fetch".into(),
                raw_input: "{}".into(),
                input: language_model::LanguageModelToolUseInput::Json(json!({})),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        let error = run
            .await
            .expect_err("blocked network dependency must stop the test");
        assert!(error.to_string().contains("incomplete"), "{error}");
        assert!(model.pending_completions().is_empty());
        let transcript = session.read_with(cx, |session, cx| {
            session
                .entries()
                .iter()
                .map(|entry| entry.to_markdown(cx))
                .collect::<String>()
        });
        assert!(transcript.contains("test incomplete"));
        assert!(!transcript.contains("Step forbidden: running"));
    }

    #[gpui::test]
    async fn recipe_workflow_failure_branch_uses_observed_status(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, _project, session) = setup_native_agent_session(cx).await;
        let session_id = session.read_with(cx, |session, _| session.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let mut recipe = recipes::Recipe::from_yaml("version: 2\ntitle: Recover\ndescription: Explicit failure branch\nsteps:\n- id: first\n  prompt: First step\n- id: skip\n  prompt: Must not run\n- id: recover\n  prompt: Recover the failed step\n  when:\n    operator: equals\n    left: {kind: status, step: first}\n    right: {kind: literal, value: failed}\n").expect("workflow");
        recipe.steps.first_mut().expect("first").on_failure =
            recipe_workflow::FailurePolicy::GoTo("recover".into());
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("first step");
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::Stop(LanguageModelStopReason::MaxTokens),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        let request = model.pending_completions().pop().expect("recovery branch");
        assert_eq!(
            request.messages.last().expect("prompt").string_contents(),
            "Recover the failed step"
        );
        model.send_completion_stream_text_chunk(&request, "Recovered");
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        assert_eq!(
            run.await.expect("handled failure").stop_reason,
            acp::StopReason::EndTurn
        );
    }

    #[gpui::test]
    async fn recipe_command_catalog_dispatch_and_local_listing(cx: &mut TestAppContext) {
        init_test(cx);
        let (connection, agent, project, acp_thread) = setup_native_agent_session(cx).await;
        cx.run_until_parked();
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let model = Arc::new(FakeLanguageModel::default());
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        thread.update(cx, |thread, cx| thread.set_model(model.clone(), cx));
        let mut entry = recipes::builtin_recipes().pop().expect("built-in recipe");
        entry.id = "builtin/compact-review".into();
        entry.recipe.as_mut().expect("valid recipe").command = Some("compact".into());
        agent.update(cx, |agent, cx| {
            agent
                .projects
                .get_mut(&project.entity_id())
                .expect("project state")
                .recipes
                .push(entry);
            agent.update_available_commands_for_project(project.entity_id(), cx);
        });
        acp_thread.read_with(cx, |thread, _| {
            let commands = thread.available_commands();
            assert_eq!(
                commands
                    .iter()
                    .filter(|command| command.name == "compact")
                    .count(),
                1
            );
            assert!(
                commands
                    .iter()
                    .any(|command| command.name == "recipe/builtin/compact-review")
            );
        });
        send_native_command_for_test(&connection, &session_id, "/recipes", cx)
            .await
            .expect("recipe list");
        assert!(model.pending_completions().is_empty());
        assert!(
            send_native_command_for_test(&connection, &session_id, "/recipe/missing", cx)
                .await
                .is_err()
        );
        assert!(
            send_native_command_for_test(
                &connection,
                &session_id,
                "/release-risk unknown=value",
                cx
            )
            .await
            .is_err()
        );
        assert!(model.pending_completions().is_empty());
        let run = cx.update(|cx| {
            acp_thread::AgentConnection::prompt(
                connection.as_ref(),
                acp::PromptRequest::new(
                    session_id.clone(),
                    vec!["/recipe/builtin/compact-review focus='the parser'".into()],
                ),
                cx,
            )
        });
        cx.run_until_parked();
        let request = model
            .pending_completions()
            .pop()
            .expect("recipe provider request");
        assert!(
            request
                .messages
                .last()
                .expect("prompt")
                .string_contents()
                .contains("Review the parser for release risks")
        );
        model.send_completion_stream_event(
            &request,
            LanguageModelCompletionEvent::Stop(LanguageModelStopReason::EndTurn),
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        assert_eq!(
            run.await.expect("recipe result").stop_reason,
            acp::StopReason::EndTurn
        );
    }

    #[gpui::test]
    async fn recipe_execution_missing_model_preserves_session(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.select_default_model(None, cx);
            });
        });
        let (connection, agent, _project, acp_thread) = setup_native_agent_session(cx).await;
        let session_id = acp_thread.read_with(cx, |thread, _| thread.session_id().clone());
        let recipe = recipes::builtin_recipes()
            .pop()
            .expect("built-in")
            .recipe
            .expect("valid recipe");
        let run = cx.update(|cx| {
            recipes::execute_recipe(
                &connection,
                session_id.clone(),
                ClientUserMessageId::new(),
                &recipe,
                &HashMap::default(),
                Vec::new(),
                cx,
            )
        });
        assert!(
            run.await
                .expect_err("missing model")
                .downcast_ref::<NoModelConfiguredError>()
                .is_some()
        );
        let thread = cx.update(|cx| native_thread_for_session(&agent, &session_id, cx));
        assert!(thread.read_with(cx, |thread, _| thread.last_message().is_none()));
    }

    #[gpui::test]
    async fn recipe_persistence_reloads_history_and_rejects_stale_writes(cx: &mut TestAppContext) {
        init_test(cx);
        let database = cx.update(ThreadsDatabase::connect).await.expect("database");
        let (revision, mut state) = database
            .load_recipe_state()
            .await
            .expect("schedule storage");
        let now = chrono::DateTime::parse_from_rfc3339("2026-09-12T12:00:00Z")
            .expect("instant")
            .to_utc();
        let run_id = Uuid::new_v4();
        state.runs.push(recipe_schedule::RecipeRun {
            id: run_id,
            schedule_id: Uuid::new_v4(),
            scheduled_for: now,
            started_at: Some(now),
            finished_at: Some(now),
            session_id: Some("saved-session".into()),
            status: recipe_schedule::RecipeRunStatus::Cancelled,
            detail: Some("Cancelled by user".into()),
        });
        let new_revision = database
            .save_recipe_state(revision, state.clone())
            .await
            .expect("save history");
        assert!(
            database
                .save_recipe_state(revision, recipe_schedule::RecipeScheduleState::default())
                .await
                .is_err()
        );
        let (loaded_revision, loaded) = database.load_recipe_state().await.expect("reload");
        assert_eq!(loaded_revision, new_revision);
        assert_eq!(loaded.runs.first().expect("saved run").id, run_id);
        assert_eq!(
            loaded
                .runs
                .first()
                .expect("saved run")
                .session_id
                .as_deref(),
            Some("saved-session")
        );
        assert_eq!(
            loaded.runs.first().expect("saved run").status,
            recipe_schedule::RecipeRunStatus::Cancelled
        );
        state.version = 999;
        assert!(
            database
                .save_recipe_state(new_revision, state)
                .await
                .is_err()
        );
        assert_eq!(
            database
                .load_recipe_state()
                .await
                .expect("unchanged history")
                .0,
            new_revision
        );
    }

    #[gpui::test]
    async fn recipe_proposal_validation_idempotence_and_permission(cx: &mut TestAppContext) {
        use crate::{AgentTool as _, ProposeRecipeTool, ToolCallEventStream, ToolInput};
        use recipes::{
            RecipeProposalInput, RecipeRecurrenceProposal, RecipeReviewStatus, RecipeStore,
            RecipeStoreEvent,
        };
        init_test(cx);
        let (_connection, agent, project, session) = setup_native_agent_session(cx).await;
        let thread =
            cx.update(|cx| native_thread_for_session(&agent, session.read(cx).session_id(), cx));
        cx.update(|cx| {
            ThreadStore::init_global(cx);
            RecipeStore::init_global(cx);
        });
        cx.run_until_parked();
        let store = cx.update(|cx| RecipeStore::try_global(cx).expect("store"));
        let model = cx.update(|cx| {
            LanguageModelRegistry::read_global(cx)
                .default_model()
                .expect("model")
        });
        thread.update(cx, |thread, cx| thread.set_model(model.model.clone(), cx));
        let input = RecipeProposalInput {
            proposal_id: "risk-review".into(),
            recipe_yaml: Some("version: 1\ntitle: Risk review\ndescription: Review risks\nprompt: Review {{focus}}\nparameters:\n  - name: focus\n    description: Scope\n    required: true\n".into()),
            recipe_id: None, filename: Some("risk.yaml".into()),
            parameters: HashMap::from_iter([("focus".into(), "risky changes".into())]),
            worktree: None, provider: None, model: None,
            schedule: Some(RecipeRecurrenceProposal { title: "Weekday review".into(), cron: "0 9 * * 1-5".into(), timezone: "Europe/London".into() }),
        };
        for invalid in [
            {
                let mut input = input.clone();
                input.parameters.clear();
                input
            },
            {
                let mut input = input.clone();
                input.worktree = Some("/outside".into());
                input
            },
            {
                let mut input = input.clone();
                input.filename = Some("../outside.yaml".into());
                input
            },
            {
                let mut input = input.clone();
                input.schedule.as_mut().expect("schedule").timezone = "not-a-zone".into();
                input
            },
            {
                let mut input = input.clone();
                input.schedule.as_mut().expect("schedule").cron = "every day".into();
                input
            },
            {
                let mut input = input.clone();
                input.provider = Some("missing".into());
                input.model = Some("missing".into());
                input
            },
        ] {
            assert!(
                store
                    .update(cx, |store, cx| store.propose(
                        thread.entity_id(),
                        project.clone(),
                        invalid,
                        Some(model.clone()),
                        Rc::new(std::cell::Cell::new(RecipeReviewStatus::Pending)),
                        cx
                    ))
                    .await
                    .is_err()
            );
        }
        let mut consent_injection = serde_json::to_value(&input).expect("input");
        consent_injection["consented"] = true.into();
        assert!(serde_json::from_value::<RecipeProposalInput>(consent_injection).is_err());
        let subscription = cx.update(|cx| {
            cx.subscribe(&store, |_, event, _| {
                if let RecipeStoreEvent::Review { proposal, .. } = event {
                    proposal.status.set(RecipeReviewStatus::Opened);
                }
            })
        });
        let call = |input, cx: &mut TestAppContext| {
            cx.update(|cx| {
                Arc::new(ProposeRecipeTool::new(thread.downgrade())).run(
                    ToolInput::resolved(input),
                    ToolCallEventStream::test().0,
                    cx,
                )
            })
        };
        assert!(
            call(input.clone(), cx)
                .await
                .expect("review")
                .contains("not saved")
        );
        let proposal = store
            .update(cx, |store, cx| {
                store.propose(
                    thread.entity_id(),
                    project.clone(),
                    input.clone(),
                    Some(model.clone()),
                    Rc::new(std::cell::Cell::new(RecipeReviewStatus::Pending)),
                    cx,
                )
            })
            .await
            .expect("repeat");
        let mut duplicate = input.clone();
        duplicate.proposal_id = "another-call".into();
        let repeated = store
            .update(cx, |store, cx| {
                store.propose(
                    thread.entity_id(),
                    project.clone(),
                    duplicate,
                    Some(model.clone()),
                    Rc::new(std::cell::Cell::new(RecipeReviewStatus::Pending)),
                    cx,
                )
            })
            .await
            .expect("deduplicate contents");
        assert!(Rc::ptr_eq(&proposal, &repeated));
        proposal.status.set(RecipeReviewStatus::Dismissed);
        assert!(
            call(input.clone(), cx)
                .await
                .expect("dismissed")
                .contains("closed")
        );
        let mut changed = input.clone();
        changed.filename = Some("different.yaml".into());
        assert!(call(changed, cx).await.is_err());
        thread.update(cx, |thread, _| thread.set_automatic_retries(false));
        assert!(
            call(input.clone(), cx)
                .await
                .expect_err("unattended denied")
                .contains("interactive")
        );
        thread.update(cx, |thread, _| thread.set_automatic_retries(true));
        let subagent = cx.new(|cx| Thread::new_subagent(&thread, cx));
        assert!(
            cx.update(
                |cx| Arc::new(ProposeRecipeTool::new(subagent.downgrade())).run(
                    ToolInput::resolved(input.clone()),
                    ToolCallEventStream::test().0,
                    cx
                )
            )
            .await
            .expect_err("subagent denied")
            .contains("interactive")
        );
        drop(subscription);
        let mut unavailable = input.clone();
        unavailable.proposal_id = "no-ui".into();
        unavailable.filename = Some("no-ui.yaml".into());
        let task = call(unavailable, cx);
        cx.run_until_parked();
        cx.background_executor
            .advance_clock(std::time::Duration::from_secs(11));
        assert!(
            task.await
                .expect_err("UI missing")
                .contains("could not open")
        );
        let (events, _, mut cancellation) = ToolCallEventStream::test_with_cancellation();
        ToolCallEventStream::signal_cancellation_with_sender(&mut cancellation);
        assert!(
            cx.update(
                |cx| Arc::new(ProposeRecipeTool::new(thread.downgrade())).run(
                    ToolInput::resolved(input.clone()),
                    events,
                    cx
                )
            )
            .await
            .expect_err("cancelled")
            .contains("cancelled")
        );
        assert!(
            store.read_with(cx, |store, _| store.state.schedules.is_empty()
                && store.state.runs.is_empty())
        );
        let database = cx.update(ThreadsDatabase::connect).await.expect("database");
        assert!(
            database
                .load_recipe_state()
                .await
                .expect("durable state")
                .1
                .schedules
                .is_empty()
        );
        use project::trusted_worktrees::{self, PathTrust};
        cx.update(|cx| trusted_worktrees::init(HashMap::default(), cx));
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/restricted", json!({})).await;
        fs.insert_tree("/second", json!({})).await;
        let restricted = Project::test_with_worktree_trust(
            fs.clone(),
            [Path::new("/restricted"), Path::new("/second")],
            cx,
        )
        .await;
        let mut restricted_input = input.clone();
        restricted_input.proposal_id = "restricted".into();
        assert!(
            store
                .update(cx, |store, cx| store.propose(
                    thread.entity_id(),
                    restricted.clone(),
                    restricted_input.clone(),
                    Some(model.clone()),
                    Rc::new(std::cell::Cell::new(RecipeReviewStatus::Pending)),
                    cx
                ))
                .await
                .err()
                .expect("ambiguous roots")
                .to_string()
                .contains("which worktree")
        );
        restricted_input.worktree = Some("/restricted".into());
        assert!(
            store
                .update(cx, |store, cx| store.propose(
                    thread.entity_id(),
                    restricted.clone(),
                    restricted_input.clone(),
                    Some(model.clone()),
                    Rc::new(std::cell::Cell::new(RecipeReviewStatus::Pending)),
                    cx
                ))
                .await
                .err()
                .expect("untrusted")
                .to_string()
                .contains("trusted")
        );
        let worktree_id = restricted.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .find(|worktree| worktree.read(cx).abs_path().as_ref() == Path::new("/restricted"))
                .expect("root")
                .read(cx)
                .id()
        });
        cx.update(|cx| {
            TrustedWorktrees::try_get_global(cx)
                .expect("trust")
                .update(cx, |trusted, cx| {
                    trusted.trust(
                        &restricted.read(cx).worktree_store(),
                        HashSet::from_iter([PathTrust::Worktree(worktree_id)]),
                        cx,
                    )
                })
        });
        let (events, _, mut cancellation) = ToolCallEventStream::test_with_cancellation();
        let original_root = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("original root")
                .read(cx)
                .id()
        });
        cx.update(|cx| {
            TrustedWorktrees::try_get_global(cx)
                .expect("trust")
                .update(cx, |trusted, cx| {
                    trusted.trust(
                        &project.read(cx).worktree_store(),
                        HashSet::from_iter([PathTrust::Worktree(original_root)]),
                        cx,
                    )
                })
        });
        let captured = Rc::new(std::cell::RefCell::new(None));
        let observed = captured.clone();
        let _pending_subscription = cx.update(|cx| {
            cx.subscribe(&store, move |_, event, _| {
                if let RecipeStoreEvent::Review { proposal, .. } = event {
                    *observed.borrow_mut() = Some(proposal.clone());
                }
            })
        });
        let mut pending_input = input.clone();
        pending_input.proposal_id = "cancel-pending".into();
        pending_input.filename = Some("pending.yaml".into());
        let pending = cx.update(|cx| {
            Arc::new(ProposeRecipeTool::new(thread.downgrade())).run(
                ToolInput::resolved(pending_input),
                events,
                cx,
            )
        });
        cx.run_until_parked();
        let waiting = captured.borrow().clone().expect("pending review");
        assert_eq!(waiting.status.get(), RecipeReviewStatus::Pending);
        ToolCallEventStream::signal_cancellation_with_sender(&mut cancellation);
        assert!(pending.await.is_err());
        assert_eq!(waiting.status.get(), RecipeReviewStatus::Cancelled);
        let mut dropped_input = input.clone();
        dropped_input.proposal_id = "dropped".into();
        dropped_input.filename = Some("dropped.yaml".into());
        let dropped = call(dropped_input, cx);
        cx.run_until_parked();
        let waiting = captured.borrow().clone().expect("drop review");
        assert_eq!(waiting.status.get(), RecipeReviewStatus::Pending);
        drop(dropped);
        cx.run_until_parked();
        assert_eq!(waiting.status.get(), RecipeReviewStatus::Cancelled);
    }

    #[gpui::test]
    async fn recipe_schedule_workflow_keeps_slot_and_cancels_on_delete(cx: &mut TestAppContext) {
        init_test(cx);
        let model = Arc::new(FakeLanguageModel::default());
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(
                    Arc::new(FakeLanguageModelProvider::default().with_models(vec![model.clone()])),
                    cx,
                );
            });
        });
        let (_connection, agent, project, _command_session) = setup_native_agent_session(cx).await;
        cx.update(|cx| {
            ThreadStore::init_global(cx);
            recipes::RecipeStore::init_global(cx);
        });
        cx.run_until_parked();
        let store = cx.update(|cx| recipes::RecipeStore::try_global(cx).expect("store"));
        store.update(cx, |store, _| store.register(&project, &agent));
        let now = cx
            .background_executor
            .scheduler_executor()
            .scheduler()
            .clock()
            .utc_now();
        let schedule = recipe_schedule::RecipeSchedule {
            id: Uuid::new_v4(),
            title: "Scheduled review".into(),
            project_paths: PathList::new(&["/a"]).serialize(),
            worktree: "/a".into(),
            cron: "* * * * *".into(),
            timezone: "UTC".into(),
            recipe: recipes::Recipe::from_yaml(
                "version: 2\ntitle: Scheduled workflow\ndescription: Verify lifecycle\nsteps:\n  - id: inspect\n    prompt: Inspect saved snapshot\n  - id: report\n    prompt: 'Report {{steps.inspect.output}}'\n",
            ).expect("workflow"),
            parameters: HashMap::default(),
            provider: model.provider_id().to_string(),
            model: model.id().0.to_string(),
            paused: true,
            consented: false,
            deleted: false,
            next_run: now,
            last_run: None,
        };
        let id = schedule.id;
        store
            .update(cx, |store, cx| store.save_schedule(schedule, cx))
            .await
            .expect("save");
        store
            .update(cx, |store, cx| store.set_enabled(id, true, true, cx))
            .await
            .expect("consent");
        store
            .update(cx, |store, cx| store.run_schedule(id, cx))
            .await
            .expect("start");
        cx.run_until_parked();
        let session_id = store.read_with(cx, |store, _| {
            store
                .state
                .active_run()
                .expect("first step active")
                .session_id
                .clone()
                .expect("session link")
        });
        assert!(
            store
                .update(cx, |store, cx| store.run_schedule(id, cx))
                .await
                .is_err()
        );
        model.send_last_completion_stream_text_chunk("observed changes");
        model.end_last_completion_stream();
        cx.run_until_parked();
        assert!(!model.pending_completions().is_empty());
        store.read_with(cx, |store, _| {
            assert_eq!(
                store
                    .state
                    .active_run()
                    .expect("second step holds slot")
                    .session_id
                    .as_deref(),
                Some(session_id.as_str())
            );
        });
        store
            .update(cx, |store, cx| store.set_enabled(id, false, false, cx))
            .await
            .expect("pause future runs");
        assert!(store.read_with(cx, |store, _| store.state.active_run().is_some()));
        store
            .update(cx, |store, cx| store.delete_schedule(id, cx))
            .await
            .expect("delete and cancel");
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert!(store.state.active_run().is_none());
            let run = store.state.runs.last().expect("history");
            assert_eq!(run.status, recipe_schedule::RecipeRunStatus::Cancelled);
            assert_eq!(run.session_id.as_deref(), Some(session_id.as_str()));
        });
        for request in model.pending_completions() {
            model.end_completion_stream(&request);
        }
        cx.run_until_parked();
        assert!(model.pending_completions().is_empty());
        let persisted = cx
            .update(|cx| {
                ThreadStore::global(cx).update(cx, |store, cx| {
                    store.load_thread(acp::SessionId::new(session_id), cx)
                })
            })
            .await
            .expect("load history session");
        assert!(persisted.is_some());
    }

    #[gpui::test]
    async fn recipe_schedule_native_session_history_and_cancellation(cx: &mut TestAppContext) {
        init_test(cx);
        let model = Arc::new(FakeLanguageModel::default());
        cx.update(|cx| {
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.register_provider(
                    Arc::new(FakeLanguageModelProvider::default().with_models(vec![model.clone()])),
                    cx,
                );
            });
        });
        let (connection, agent, project, command_session) = setup_native_agent_session(cx).await;
        cx.update(|cx| {
            ThreadStore::init_global(cx);
            recipes::RecipeStore::init_global(cx);
        });
        cx.run_until_parked();
        let store = cx.update(|cx| recipes::RecipeStore::try_global(cx).expect("store"));
        store.update(cx, |store, _| store.register(&project, &agent));
        let now = cx
            .background_executor
            .scheduler_executor()
            .scheduler()
            .clock()
            .utc_now();
        let schedule = recipe_schedule::RecipeSchedule {
            id: Uuid::new_v4(),
            title: "Scheduled review".into(),
            project_paths: PathList::new(&["/a"]).serialize(),
            worktree: "/a".into(),
            cron: "* * * * *".into(),
            timezone: "UTC".into(),
            recipe: recipes::builtin_recipes()
                .pop()
                .expect("recipe")
                .recipe
                .expect("valid"),
            parameters: HashMap::default(),
            provider: model.provider_id().to_string(),
            model: model.id().0.to_string(),
            paused: true,
            consented: false,
            deleted: false,
            next_run: now,
            last_run: None,
        };
        let id = schedule.id;
        store
            .update(cx, |store, cx| store.save_schedule(schedule, cx))
            .await
            .expect("save");
        assert!(
            store
                .update(cx, |store, cx| store.run_schedule(id, cx))
                .await
                .is_err()
        );
        store
            .update(cx, |store, cx| store.set_enabled(id, true, true, cx))
            .await
            .expect("consent");
        let due = store.read_with(cx, |store, _| {
            store.state.schedules.first().expect("schedule").next_run
        });
        cx.background_executor
            .advance_clock((due - now).to_std().expect("future occurrence"));
        cx.run_until_parked();
        let session_id = store.read_with(cx, |store, _| {
            let run = store.state.active_run().expect("active run");
            assert_eq!(run.status, recipe_schedule::RecipeRunStatus::Running);
            run.session_id.clone().expect("durable session link")
        });
        let original_schedule = store.read_with(cx, |store, _| {
            store.state.schedules.first().expect("schedule").clone()
        });
        let mut edited_schedule = original_schedule.clone();
        edited_schedule.provider = "not-configured".into();
        store
            .update(cx, |store, cx| store.save_schedule(edited_schedule, cx))
            .await
            .expect("edit future runs");
        cx.background_executor
            .advance_clock(std::time::Duration::from_secs(1));
        cx.run_until_parked();
        assert!(store.read_with(cx, |store, _| store.state.active_run().is_some()));
        store
            .update(cx, |store, cx| store.save_schedule(original_schedule, cx))
            .await
            .expect("restore future provider");
        store
            .update(cx, |store, cx| store.set_enabled(id, true, true, cx))
            .await
            .expect("approve restored inputs");
        assert!(!model.pending_completions().is_empty());
        model.send_last_completion_stream_text_chunk("Review finished");
        model.end_last_completion_stream();
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert!(store.state.active_run().is_none());
            assert_eq!(
                store.state.runs.last().expect("history").status,
                recipe_schedule::RecipeRunStatus::Succeeded
            );
        });
        let persisted = cx
            .update(|cx| {
                ThreadStore::global(cx).update(cx, |store, cx| {
                    store.load_thread(acp::SessionId::new(session_id), cx)
                })
            })
            .await
            .expect("load");
        assert!(persisted.is_some());
        cx.background_executor
            .dispatcher()
            .as_test()
            .expect("test dispatcher")
            .scheduler()
            .clock()
            .advance(std::time::Duration::from_secs(7200));
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert_eq!(
                store.state.runs.last().expect("missed run").status,
                recipe_schedule::RecipeRunStatus::Missed
            )
        });
        assert!(model.pending_completions().is_empty());
        store
            .update(cx, |store, cx| store.run_schedule(id, cx))
            .await
            .expect("permission run");
        cx.run_until_parked();
        let permission_session_id = store.read_with(cx, |store, _| {
            acp::SessionId::new(
                store
                    .state
                    .active_run()
                    .expect("active run")
                    .session_id
                    .clone()
                    .expect("session"),
            )
        });
        let permission_session = agent.read_with(cx, |agent, _| {
            agent
                .sessions
                .get(&permission_session_id)
                .expect("native session")
                .acp_thread
                .clone()
        });
        let permission_request = model
            .pending_completions()
            .pop()
            .expect("permission completion");
        let permission_response = permission_session
            .update(cx, |session, cx| {
                session.request_tool_call_authorization(
                    acp::ToolCall::new("scheduled-permission", "Approve scheduled tool")
                        .kind(acp::ToolKind::Execute)
                        .status(acp::ToolCallStatus::Pending)
                        .into(),
                    acp_thread::PermissionOptions::Flat(vec![acp::PermissionOption::new(
                        "allow",
                        "Allow",
                        acp::PermissionOptionKind::AllowOnce,
                    )]),
                    acp_thread::AuthorizationKind::PermissionGrant,
                    cx,
                )
            })
            .expect("permission request");
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert_eq!(
                store.state.runs.last().expect("attention run").status,
                recipe_schedule::RecipeRunStatus::NeedsAttention
            );
            assert!(
                store
                    .state
                    .schedules
                    .first()
                    .expect("paused schedule")
                    .paused
            );
        });
        model.end_completion_stream(&permission_request);
        assert!(model.pending_completions().is_empty());
        permission_session.update(cx, |session, cx| {
            session.authorize_tool_call(
                "scheduled-permission".into(),
                acp_thread::SelectedPermissionOutcome::new(
                    "allow".into(),
                    acp::PermissionOptionKind::AllowOnce,
                ),
                cx,
            )
        });
        cx.run_until_parked();
        permission_response.await;
        assert!(model.pending_completions().is_empty());
        store
            .update(cx, |store, cx| store.set_enabled(id, true, true, cx))
            .await
            .expect("renewed consent");
        store
            .update(cx, |store, cx| store.run_schedule(id, cx))
            .await
            .expect("credential failure run");
        cx.run_until_parked();
        let request = model
            .pending_completions()
            .pop()
            .expect("credential request");
        model.send_completion_stream_error(
            &request,
            LanguageModelCompletionError::NoApiKey {
                provider: "Fake".to_owned().into(),
            },
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert_eq!(
                store.state.runs.last().expect("credential run").status,
                recipe_schedule::RecipeRunStatus::NeedsAttention
            );
            assert!(
                store
                    .state
                    .schedules
                    .first()
                    .expect("paused schedule")
                    .paused
            );
        });
        store
            .update(cx, |store, cx| store.set_enabled(id, true, true, cx))
            .await
            .expect("credential consent");
        store
            .update(cx, |store, cx| store.run_schedule(id, cx))
            .await
            .expect("transient failure run");
        cx.run_until_parked();
        let request = model
            .pending_completions()
            .pop()
            .expect("transient request");
        model.send_completion_stream_error(
            &request,
            LanguageModelCompletionError::RateLimitExceeded {
                provider: "Fake".to_owned().into(),
                retry_after: Some(std::time::Duration::ZERO),
            },
        );
        model.end_completion_stream(&request);
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert_eq!(
                store.state.runs.last().expect("failed run").status,
                recipe_schedule::RecipeRunStatus::Failed
            )
        });
        assert!(model.pending_completions().is_empty());
        store
            .update(cx, |store, cx| store.run_schedule(id, cx))
            .await
            .expect("second run");
        cx.run_until_parked();
        let cancelled_request = model
            .pending_completions()
            .pop()
            .expect("cancelled request");
        store
            .update(cx, |store, cx| store.delete_schedule(id, cx))
            .await
            .expect("delete active schedule");
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert!(store.state.active_run().is_none());
            assert!(
                store
                    .state
                    .schedules
                    .first()
                    .expect("deleted schedule")
                    .deleted
            );
            assert_eq!(
                store.state.runs.last().expect("cancelled history").status,
                recipe_schedule::RecipeRunStatus::Cancelled
            );
        });
        let command_session_id =
            command_session.read_with(cx, |session, _| session.session_id().clone());
        model.end_completion_stream(&cancelled_request);
        send_native_command_for_test(
            &connection,
            &command_session_id,
            &format!("/schedules history {id}"),
            cx,
        )
        .await
        .expect("history command");
        assert!(command_session.read_with(cx, |session, cx| {
            session.to_markdown(cx).contains("Cancelled")
        }));
        assert!(model.pending_completions().is_empty());
    }

    #[gpui::test]
    async fn recipe_management_project_save_conflict_and_delete(cx: &mut TestAppContext) {
        init_test(cx);
        let (_connection, _agent, project, _session) = setup_native_agent_session(cx).await;
        let worktree = project.read_with(cx, |project, cx| {
            project.visible_worktrees(cx).next().expect("worktree")
        });
        let path = ProjectPath {
            worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
            path: util::rel_path::RelPath::from_unix_str(".agents/recipes/review.yaml")
                .expect("path")
                .into(),
        };
        let yaml = recipes::builtin_recipes()
            .pop()
            .expect("built-in")
            .recipe
            .expect("valid")
            .to_yaml()
            .expect("yaml");
        cx.update(|cx| {
            recipes::save_local_recipe(project.clone(), path.clone(), yaml.clone(), None, cx)
        })
        .await
        .expect("create recipe");
        cx.run_until_parked();
        assert!(
            cx.update(|cx| recipes::save_local_recipe(
                project.clone(),
                path.clone(),
                yaml.clone(),
                None,
                cx
            ))
            .await
            .is_err()
        );
        assert!(
            cx.update(|cx| recipes::save_local_recipe(
                project.clone(),
                path.clone(),
                yaml.clone(),
                Some("stale".into()),
                cx
            ))
            .await
            .is_err()
        );
        let edited = yaml.replace("release-risk", "review-work");
        cx.update(|cx| {
            recipes::save_local_recipe(
                project.clone(),
                path.clone(),
                edited.clone(),
                Some(yaml),
                cx,
            )
        })
        .await
        .expect("edit recipe");
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
            .await
            .expect("saved buffer");
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer
                .as_text_snapshot()
                .as_rope()
                .to_string()),
            edited
        );
        cx.update(|cx| recipes::delete_local_recipe(project.clone(), path, cx))
            .await
            .expect("delete recipe");
    }

    fn init_test(cx: &mut TestAppContext) {
        env_logger::try_init().ok();
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);

            LanguageModelRegistry::test(cx);
        });
    }

    #[test]
    fn test_strip_slash_command_prefix_keeps_inline_args() {
        // The bug being guarded against: skill slash invocation used to
        // discard the entire first text block, which threw away anything
        // the user typed on the same line as the command.
        assert_eq!(
            strip_slash_command_prefix("/fix-review #1, #2, #3"),
            "#1, #2, #3",
        );
    }

    #[test]
    fn test_strip_slash_command_prefix_preserves_newlines() {
        // Continuations across newlines are common when users compose
        // structured prompts; the first newline is the command terminator,
        // but everything after it must reach the model verbatim.
        assert_eq!(
            strip_slash_command_prefix("/fix-review\nline 1\nline 2"),
            "line 1\nline 2",
        );
    }

    #[test]
    fn test_strip_slash_command_prefix_command_only_is_empty() {
        assert_eq!(strip_slash_command_prefix("/fix-review"), "");
        assert_eq!(strip_slash_command_prefix("/fix-review "), "");
    }

    #[test]
    fn test_strip_slash_command_prefix_ignores_leading_whitespace() {
        assert_eq!(strip_slash_command_prefix("   /fix-review hello"), "hello",);
    }

    #[test]
    fn test_strip_slash_command_prefix_passes_through_non_command_text() {
        // Defense in depth: if somehow we're called with a non-slash-prefixed
        // block, the safe behavior is to return it unchanged rather than
        // silently mangling unrelated user text.
        assert_eq!(strip_slash_command_prefix("hello world"), "hello world",);
    }
}

fn mcp_message_content_to_acp_content_block(
    content: context_server::types::MessageContent,
) -> acp::ContentBlock {
    match content {
        context_server::types::MessageContent::Text {
            text,
            annotations: _,
        } => text.into(),
        context_server::types::MessageContent::Image {
            data,
            mime_type,
            annotations: _,
        } => acp::ContentBlock::Image(acp::ImageContent::new(data, mime_type)),
        context_server::types::MessageContent::Audio {
            data,
            mime_type,
            annotations: _,
        } => acp::ContentBlock::Audio(acp::AudioContent::new(data, mime_type)),
        context_server::types::MessageContent::Resource {
            resource,
            annotations: _,
        } => {
            let mut link =
                acp::ResourceLink::new(resource.uri.to_string(), resource.uri.to_string());
            if let Some(mime_type) = resource.mime_type {
                link = link.mime_type(mime_type);
            }
            acp::ContentBlock::ResourceLink(link)
        }
    }
}
