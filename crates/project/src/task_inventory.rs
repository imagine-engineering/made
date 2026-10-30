//! Project-wide storage of the tasks available, capable of updating itself from the sources set.

use std::{
    borrow::Cow,
    cmp::{self, Reverse},
    collections::hash_map,
    path::PathBuf,
    sync::Arc,
};

use anyhow::Result;
use collections::{HashMap, HashSet, VecDeque};
use dap::DapRegistry;
#[cfg(feature = "jvm-tools")]
use gpui::ReadGlobal as _;
use gpui::{App, AppContext as _, Context, Entity, SharedString, Task, WeakEntity};
use itertools::Itertools;
use language::{
    Buffer, ContextLocation, ContextProvider, File, Language, LanguageToolchainStore, Location,
    language_settings::LanguageSettings,
};
use lsp::{LanguageServerId, LanguageServerName};
#[cfg(feature = "jvm-tools")]
use path::PathStyle;
use paths::{debug_task_file_name, task_file_name};
#[cfg(feature = "jvm-tools")]
use settings::Settings as _;
use settings::{InvalidSettingsError, parse_json_with_comments};
use task::{
    DebugScenario, ResolvedTask, SharedTaskContext, TaskContext, TaskHook, TaskId, TaskTemplate,
    TaskTemplates, TaskVariables, VariableName,
};
use text::{BufferId, Point, ToPoint};
use util::{NumericPrefixWithSuffix, ResultExt as _, post_inc, rel_path::RelPath};
use worktree::WorktreeId;

use crate::{git_store::GitStore, task_store::TaskSettingsLocation, worktree_store::WorktreeStore};

pub const GIT_COMMAND_TASK_TAG: &str = "git-command";

#[cfg(feature = "jvm-tools")]
pub(crate) fn jvm_build_entry_is_available(entry: &worktree::Entry) -> bool {
    entry.is_file()
        && !entry.is_ignored
        && !entry.is_private
        && !entry.is_external
        && !entry.is_fifo
}

#[derive(Clone, Debug, Default)]
pub struct DebugScenarioContext {
    pub task_context: SharedTaskContext,
    pub worktree_id: Option<WorktreeId>,
    pub active_buffer: Option<WeakEntity<Buffer>>,
    pub experimental_policy_version: u32,
}

impl DebugScenarioContext {
    pub const CURRENT_EXPERIMENTAL_POLICY_VERSION: u32 = 1;

    pub fn validate_replay_policy(&self) -> Result<()> {
        anyhow::ensure!(
            self.experimental_policy_version == Self::CURRENT_EXPERIMENTAL_POLICY_VERSION,
            "Refresh and reselect this debug scenario before replaying legacy history"
        );
        Ok(())
    }
}

/// Inventory tracks available tasks for a given project.
pub struct Inventory {
    last_scheduled_tasks: VecDeque<(TaskSourceKind, ResolvedTask)>,
    last_scheduled_scenarios: VecDeque<(DebugScenario, DebugScenarioContext)>,
    templates_from_settings: InventoryFor<TaskTemplate>,
    scenarios_from_settings: InventoryFor<DebugScenario>,
    #[cfg(feature = "jvm-tools")]
    worktree_store: Option<WeakEntity<WorktreeStore>>,
    #[cfg(feature = "jvm-tools")]
    jvm_operations: HashMap<TaskId, AdmittedJvmOperation>,
}

#[cfg(feature = "jvm-tools")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct AdmittedJvmOperation {
    receipt: crate::jvm_project::JvmBuildOperationReceipt,
    plan: crate::jvm_project::JvmBuildOperationPlan,
    resolved: ResolvedTask,
    worktree_entity: gpui::EntityId,
    root: Arc<std::path::Path>,
    root_entry: worktree::Entry,
    wrapper: worktree::Entry,
    manifests: Vec<Option<worktree::Entry>>,
    settings: serde_json::Value,
}

impl std::fmt::Debug for Inventory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inventory")
            .field("last_scheduled_tasks", &self.last_scheduled_tasks)
            .field("last_scheduled_scenarios", &self.last_scheduled_scenarios)
            .field("templates_from_settings", &self.templates_from_settings)
            .field("scenarios_from_settings", &self.scenarios_from_settings)
            .finish()
    }
}

// Helper trait for better error messages in [InventoryFor]
trait InventoryContents: Clone {
    const GLOBAL_SOURCE_FILE: &'static str;
    const LABEL: &'static str;
}

impl InventoryContents for TaskTemplate {
    const GLOBAL_SOURCE_FILE: &'static str = "tasks.json";
    const LABEL: &'static str = "tasks";
}

impl InventoryContents for DebugScenario {
    const GLOBAL_SOURCE_FILE: &'static str = "debug.json";

    const LABEL: &'static str = "debug scenarios";
}

#[derive(Debug)]
struct InventoryFor<T> {
    global: HashMap<PathBuf, Vec<T>>,
    worktree: HashMap<WorktreeId, HashMap<Arc<RelPath>, Vec<T>>>,
}

impl<T: InventoryContents> InventoryFor<T> {
    fn worktree_scenarios(
        &self,
        worktree: WorktreeId,
    ) -> impl '_ + Iterator<Item = (TaskSourceKind, T)> {
        let worktree_dirs = self.worktree.get(&worktree);
        let has_zed_dir = worktree_dirs
            .map(|dirs| {
                dirs.keys()
                    .any(|dir| dir.file_name().is_some_and(|name| name == ".zed"))
            })
            .unwrap_or(false);

        worktree_dirs
            .into_iter()
            .flatten()
            .filter(move |(directory, _)| {
                !(has_zed_dir && directory.file_name().is_some_and(|name| name == ".vscode"))
            })
            .flat_map(|(directory, templates)| {
                templates.iter().map(move |template| (directory, template))
            })
            .map(move |(directory, template)| {
                (
                    TaskSourceKind::Worktree {
                        id: worktree,
                        directory_in_worktree: directory.clone(),
                        id_base: Cow::Owned(format!(
                            "local worktree {} from directory {directory:?}",
                            T::LABEL
                        )),
                    },
                    template.clone(),
                )
            })
    }

    fn global_scenarios(&self) -> impl '_ + Iterator<Item = (TaskSourceKind, T)> {
        self.global.iter().flat_map(|(file_path, templates)| {
            templates.iter().map(|template| {
                (
                    TaskSourceKind::AbsPath {
                        id_base: Cow::Owned(format!("global {}", T::GLOBAL_SOURCE_FILE)),
                        abs_path: file_path.clone(),
                    },
                    template.clone(),
                )
            })
        })
    }
}

impl<T> Default for InventoryFor<T> {
    fn default() -> Self {
        Self {
            global: HashMap::default(),
            worktree: HashMap::default(),
        }
    }
}

/// Kind of a source the tasks are fetched from, used to display more source information in the UI.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TaskSourceKind {
    /// bash-like commands spawned by users, not associated with any path
    UserInput,
    /// Tasks from the worktree's .zed/task.json
    Worktree {
        id: WorktreeId,
        directory_in_worktree: Arc<RelPath>,
        id_base: Cow<'static, str>,
    },
    /// ~/.config/zed/task.json - like global files with task definitions, applicable to any path
    AbsPath {
        id_base: Cow<'static, str>,
        abs_path: PathBuf,
    },
    /// Languages-specific tasks coming from extensions.
    Language { name: SharedString },
    /// Language-specific tasks coming from LSP servers.
    Lsp {
        language_name: SharedString,
        server: LanguageServerId,
    },
}

/// A collection of task contexts, derived from the current state of the workspace.
/// Only contains worktrees that are visible and with their root being a directory.
#[derive(Debug, Default)]
pub struct TaskContexts {
    /// A context, related to the currently opened item.
    /// Item can be opened from an invisible worktree, or any other, not necessarily active worktree.
    pub active_item_context: Option<(Option<WorktreeId>, Option<Location>, TaskContext)>,
    /// A worktree that corresponds to the active item, or the only worktree in the workspace.
    pub active_worktree_context: Option<(WorktreeId, TaskContext)>,
    /// If there are multiple worktrees in the workspace, all non-active ones are included here.
    pub other_worktree_contexts: Vec<(WorktreeId, TaskContext)>,
    pub lsp_task_sources: HashMap<LanguageServerName, Vec<BufferId>>,
    pub latest_selection: Option<text::Anchor>,
}

impl TaskContexts {
    pub fn active_context(&self) -> Option<&TaskContext> {
        self.active_item_context
            .as_ref()
            .map(|(_, _, context)| context)
            .or_else(|| {
                self.active_worktree_context
                    .as_ref()
                    .map(|(_, context)| context)
            })
    }

    pub fn location(&self) -> Option<&Location> {
        self.active_item_context
            .as_ref()
            .and_then(|(_, location, _)| location.as_ref())
    }

    pub fn file(&self, cx: &App) -> Option<Arc<dyn File>> {
        self.active_item_context
            .as_ref()
            .and_then(|(_, location, _)| location.as_ref())
            .and_then(|location| location.buffer.read(cx).file().cloned())
    }

    pub fn worktree(&self) -> Option<WorktreeId> {
        self.active_item_context
            .as_ref()
            .and_then(|(worktree_id, _, _)| worktree_id.as_ref())
            .or_else(|| {
                self.active_worktree_context
                    .as_ref()
                    .map(|(worktree_id, _)| worktree_id)
            })
            .copied()
    }

    pub fn task_context_for_worktree_id(&self, worktree_id: WorktreeId) -> Option<&TaskContext> {
        self.active_worktree_context
            .iter()
            .chain(self.other_worktree_contexts.iter())
            .find(|(id, _)| *id == worktree_id)
            .map(|(_, context)| context)
    }

    /// Re-resolves a previously resolved task against the current contexts, so that
    /// any context-dependent variables (e.g. the active file or selection) reflect the
    /// editor's current state rather than the state from when the task was first run.
    ///
    /// For worktree tasks the search is scoped to the task's own worktree; other tasks
    /// fall back through the active item, active worktree, and finally an empty context.
    /// If none of the contexts resolve, the original `resolved_task` is returned unchanged.
    fn reresolve_task(&self, kind: &TaskSourceKind, resolved_task: &ResolvedTask) -> ResolvedTask {
        #[cfg(feature = "jvm-tools")]
        if kind.groovy_jvm_operation_worktree().is_some() {
            return resolved_task.clone();
        }
        let id_base = kind.to_id_base();
        let task = resolved_task.original_task();

        let candidate_contexts: Vec<Option<&TaskContext>> = match kind {
            TaskSourceKind::Worktree { id, .. } => vec![
                self.active_item_context
                    .as_ref()
                    .filter(|(worktree_id, _, _)| worktree_id.as_ref() == Some(id))
                    .map(|(_, _, context)| context),
                self.active_worktree_context
                    .as_ref()
                    .filter(|(worktree_id, _)| worktree_id == id)
                    .map(|(_, context)| context),
                self.other_worktree_contexts
                    .iter()
                    .find(|(worktree_id, _)| worktree_id == id)
                    .map(|(_, context)| context),
            ],
            _ => vec![
                self.active_item_context
                    .as_ref()
                    .map(|(_, _, context)| context),
                self.active_worktree_context
                    .as_ref()
                    .map(|(_, context)| context),
            ],
        };

        candidate_contexts
            .into_iter()
            .flatten()
            .find_map(|context| task.resolve_task(&id_base, context))
            .or_else(|| task.resolve_task(&id_base, &TaskContext::default()))
            .unwrap_or_else(|| resolved_task.clone())
    }
}

impl TaskSourceKind {
    #[cfg(feature = "jvm-tools")]
    pub fn groovy_jvm_build(worktree_id: WorktreeId) -> Self {
        Self::Worktree {
            id: worktree_id,
            directory_in_worktree: RelPath::empty_arc(),
            id_base: Cow::Borrowed(GROOVY_JVM_BUILD_TASK_SOURCE),
        }
    }

    #[cfg(feature = "jvm-tools")]
    pub fn groovy_jvm_operation(worktree_id: WorktreeId) -> Self {
        Self::Worktree {
            id: worktree_id,
            directory_in_worktree: RelPath::empty_arc(),
            id_base: Cow::Borrowed(GROOVY_JVM_OPERATION_TASK_SOURCE),
        }
    }

    #[cfg(feature = "jvm-tools")]
    pub fn groovy_jvm_operation_worktree(&self) -> Option<WorktreeId> {
        match self {
            Self::Worktree { id, id_base, .. } if id_base == GROOVY_JVM_OPERATION_TASK_SOURCE => {
                Some(*id)
            }
            _ => None,
        }
    }

    #[cfg(feature = "jvm-tools")]
    pub fn groovy_jvm_test(worktree_id: WorktreeId) -> Self {
        Self::Worktree {
            id: worktree_id,
            directory_in_worktree: RelPath::empty_arc(),
            id_base: Cow::Borrowed(GROOVY_JVM_TEST_TASK_SOURCE),
        }
    }

    pub fn groovy_jvm_task_worktree(&self) -> Option<WorktreeId> {
        #[cfg(feature = "jvm-tools")]
        {
            match self {
                Self::Worktree { id, id_base, .. }
                    if id_base == GROOVY_JVM_BUILD_TASK_SOURCE
                        || id_base == GROOVY_JVM_TEST_TASK_SOURCE
                        || id_base == GROOVY_JVM_OPERATION_TASK_SOURCE =>
                {
                    Some(*id)
                }
                _ => None,
            }
        }
        #[cfg(not(feature = "jvm-tools"))]
        {
            None
        }
    }

    pub fn groovy_jvm_build_worktree(&self) -> Option<WorktreeId> {
        #[cfg(feature = "jvm-tools")]
        {
            match self {
                Self::Worktree { id, id_base, .. } if id_base == GROOVY_JVM_BUILD_TASK_SOURCE => {
                    Some(*id)
                }
                _ => None,
            }
        }
        #[cfg(not(feature = "jvm-tools"))]
        {
            None
        }
    }

    pub fn to_id_base(&self) -> String {
        match self {
            Self::UserInput => "oneshot".to_string(),
            Self::AbsPath { id_base, abs_path } => {
                format!("{id_base}_{}", abs_path.display())
            }
            Self::Worktree {
                id,
                id_base,
                directory_in_worktree,
            } => {
                format!("{id_base}_{id}_{}", directory_in_worktree.as_unix_str())
            }
            Self::Language { name } => format!("language_{name}"),
            Self::Lsp {
                server,
                language_name,
            } => format!("lsp_{language_name}_{server}"),
        }
    }
}

#[cfg(feature = "jvm-tools")]
const GROOVY_JVM_BUILD_TASK_SOURCE: &str = "Groovy JVM build";
#[cfg(feature = "jvm-tools")]
const GROOVY_JVM_TEST_TASK_SOURCE: &str = "Groovy JVM selected test";
#[cfg(feature = "jvm-tools")]
const GROOVY_JVM_OPERATION_TASK_SOURCE: &str = "Groovy JVM operation";

impl Inventory {
    pub fn new(cx: &mut App) -> Entity<Self> {
        cx.new(|_| Self {
            last_scheduled_tasks: VecDeque::default(),
            last_scheduled_scenarios: VecDeque::default(),
            templates_from_settings: InventoryFor::default(),
            scenarios_from_settings: InventoryFor::default(),
            #[cfg(feature = "jvm-tools")]
            worktree_store: None,
            #[cfg(feature = "jvm-tools")]
            jvm_operations: HashMap::default(),
        })
    }

    #[cfg(feature = "jvm-tools")]
    pub fn bind_worktree_store(&mut self, worktree_store: WeakEntity<WorktreeStore>) {
        self.worktree_store = Some(worktree_store);
    }

    pub fn scenario_scheduled(
        &mut self,
        scenario: DebugScenario,
        task_context: SharedTaskContext,
        worktree_id: Option<WorktreeId>,
        active_buffer: Option<WeakEntity<Buffer>>,
    ) {
        self.last_scheduled_scenarios
            .retain(|(s, _)| s.label != scenario.label);
        self.last_scheduled_scenarios.push_front((
            scenario,
            DebugScenarioContext {
                task_context,
                worktree_id,
                active_buffer,
                experimental_policy_version: 0,
            },
        ));
        if self.last_scheduled_scenarios.len() > 5_000 {
            self.last_scheduled_scenarios.pop_back();
        }
    }

    pub fn admitted_scenario_scheduled(
        &mut self,
        scenario: DebugScenario,
        task_context: SharedTaskContext,
        worktree_id: Option<WorktreeId>,
        active_buffer: Option<WeakEntity<Buffer>>,
    ) {
        self.scenario_scheduled(scenario, task_context, worktree_id, active_buffer);
        if let Some((_, context)) = self.last_scheduled_scenarios.front_mut() {
            context.experimental_policy_version =
                DebugScenarioContext::CURRENT_EXPERIMENTAL_POLICY_VERSION;
        }
    }

    pub fn last_scheduled_scenario(&self) -> Option<&(DebugScenario, DebugScenarioContext)> {
        self.last_scheduled_scenarios.front()
    }

    pub fn list_debug_scenarios(
        &self,
        task_contexts: &TaskContexts,
        lsp_tasks: Vec<(TaskSourceKind, task::ResolvedTask)>,
        current_resolved_tasks: Vec<(TaskSourceKind, task::ResolvedTask)>,
        add_current_language_tasks: bool,
        cx: &mut App,
    ) -> Task<(
        Vec<(DebugScenario, DebugScenarioContext)>,
        Vec<(TaskSourceKind, DebugScenario)>,
    )> {
        let mut scenarios = Vec::new();

        if let Some(worktree_id) = task_contexts
            .active_worktree_context
            .iter()
            .chain(task_contexts.other_worktree_contexts.iter())
            .map(|context| context.0)
            .next()
        {
            scenarios.extend(self.worktree_scenarios_from_settings(worktree_id));
        }
        scenarios.extend(self.global_debug_scenarios_from_settings());

        let last_scheduled_scenarios = self.last_scheduled_scenarios.iter().cloned().collect();

        let adapter = task_contexts.location().and_then(|location| {
            let buffer = location.buffer.read(cx);
            let adapter = LanguageSettings::for_buffer(&buffer, cx)
                .debuggers
                .first()
                .map(SharedString::from)
                .or_else(|| {
                    buffer
                        .language()
                        .and_then(|l| l.config().debuggers.first().map(SharedString::from))
                });
            adapter.map(|adapter| (adapter, DapRegistry::global(cx).locators()))
        });
        cx.background_spawn(async move {
            if let Some((adapter, locators)) = adapter {
                for (kind, task) in
                    lsp_tasks
                        .into_iter()
                        .chain(current_resolved_tasks.into_iter().filter(|(kind, _)| {
                            add_current_language_tasks
                                || !matches!(kind, TaskSourceKind::Language { .. })
                        }))
                {
                    let adapter = adapter.clone().into();

                    for locator in locators.values() {
                        if let Some(scenario) = locator
                            .create_scenario(task.original_task(), task.display_label(), &adapter)
                            .await
                        {
                            scenarios.push((kind, scenario));
                            break;
                        }
                    }
                }
            }
            (last_scheduled_scenarios, scenarios)
        })
    }

    pub fn task_template_by_label(
        &self,
        buffer: Option<Entity<Buffer>>,
        worktree_id: Option<WorktreeId>,
        label: &str,
        cx: &App,
    ) -> Task<Option<TaskTemplate>> {
        let (buffer_worktree_id, language) = buffer
            .as_ref()
            .map(|buffer| {
                let buffer = buffer.read(cx);
                (
                    buffer.file().as_ref().map(|file| file.worktree_id(cx)),
                    buffer.language().cloned(),
                )
            })
            .unwrap_or((None, None));

        let tasks = self.list_tasks(buffer, language, worktree_id.or(buffer_worktree_id), cx);
        let label = label.to_owned();
        cx.background_spawn(async move {
            tasks
                .await
                .into_iter()
                .find(|(_, template)| template.label == label)
                .map(|val| val.1)
        })
    }

    /// Pulls its task sources relevant to the worktree and the language given,
    /// returns all task templates with their source kinds, worktree tasks first, language tasks second
    /// and global tasks last. No specific order inside source kinds groups.
    pub fn list_tasks(
        &self,
        buffer: Option<Entity<Buffer>>,
        language: Option<Arc<Language>>,
        worktree: Option<WorktreeId>,
        cx: &App,
    ) -> Task<Vec<(TaskSourceKind, TaskTemplate)>> {
        let global_tasks = self.global_templates_from_settings().collect::<Vec<_>>();
        let mut worktree_tasks = worktree
            .into_iter()
            .flat_map(|worktree| self.worktree_templates_from_settings(worktree))
            .collect::<Vec<_>>();
        #[cfg(feature = "jvm-tools")]
        if let Some(worktree_id) = worktree {
            worktree_tasks.extend(self.jvm_build_tasks(worktree_id, cx));
        }

        let task_source_kind = language.as_ref().map(|language| TaskSourceKind::Language {
            name: language.name().into(),
        });
        let language_tasks = language
            .filter(|language| {
                LanguageSettings::resolve(
                    buffer.as_ref().map(|b| b.read(cx)),
                    Some(&language.name()),
                    cx,
                )
                .tasks
                .enabled
            })
            .and_then(|language| {
                language
                    .context_provider()
                    .map(|provider| provider.associated_tasks(buffer, cx))
            });
        cx.background_spawn(async move {
            if let Some(t) = language_tasks {
                worktree_tasks.extend(t.await.into_iter().flat_map(|tasks| {
                    tasks
                        .0
                        .into_iter()
                        .filter_map(|task| Some((task_source_kind.clone()?, task)))
                }));
            }
            worktree_tasks.extend(global_tasks);
            worktree_tasks
        })
    }

    /// Pulls its task sources relevant to the worktree and the language given and resolves them with the [`TaskContexts`] given.
    /// Joins the new resolutions with the resolved tasks that were used (spawned) before,
    /// orders them so that the most recently used come first, all equally used ones are ordered so that the most specific tasks come first.
    /// Deduplicates the tasks by their labels and context and splits the ordered list into two: used tasks and the rest, newly resolved tasks.
    pub fn used_and_current_resolved_tasks(
        &self,
        task_contexts: Arc<TaskContexts>,
        cx: &mut Context<Self>,
    ) -> Task<(
        Vec<(TaskSourceKind, ResolvedTask)>,
        Vec<(TaskSourceKind, ResolvedTask)>,
    )> {
        let worktree = task_contexts.worktree();
        let location = task_contexts.location();
        let language = location.and_then(|location| location.buffer.read(cx).language());
        let task_source_kind = language.as_ref().map(|language| TaskSourceKind::Language {
            name: language.name().into(),
        });
        let buffer = location.map(|location| location.buffer.clone());

        let worktrees_with_zed_tasks: HashSet<WorktreeId> = self
            .templates_from_settings
            .worktree
            .iter()
            .filter(|(_, dirs)| {
                dirs.keys()
                    .any(|dir| dir.file_name().is_some_and(|name| name == ".zed"))
            })
            .map(|(id, _)| *id)
            .collect();

        let mut task_labels_to_ids = HashMap::<String, HashSet<TaskId>>::default();
        let mut lru_score = 0_u32;
        let previously_spawned_tasks = self
            .last_scheduled_tasks
            .iter()
            .rev()
            .filter(|(task_kind, _)| {
                if matches!(task_kind, TaskSourceKind::Language { .. }) {
                    Some(task_kind) == task_source_kind.as_ref()
                } else if let TaskSourceKind::Worktree {
                    id,
                    directory_in_worktree: dir,
                    ..
                } = task_kind
                {
                    !(worktrees_with_zed_tasks.contains(id)
                        && dir.file_name().is_some_and(|name| name == ".vscode"))
                } else {
                    true
                }
            })
            .map(|(task_source_kind, resolved_task)| {
                let reresolved = task_contexts.reresolve_task(task_source_kind, resolved_task);
                let task = if reresolved.resolved_label == resolved_task.resolved_label {
                    reresolved
                } else {
                    resolved_task.clone()
                };
                (task_source_kind.clone(), task)
            })
            .filter(|(_, resolved_task)| {
                match task_labels_to_ids.entry(resolved_task.resolved_label.clone()) {
                    hash_map::Entry::Occupied(mut o) => {
                        // Allow distinct re-resolved tasks that share a label but differ by context.
                        o.get_mut().insert(resolved_task.id.clone())
                    }
                    hash_map::Entry::Vacant(v) => {
                        v.insert(HashSet::from_iter(Some(resolved_task.id.clone())));
                        true
                    }
                }
            })
            .map(|(kind, task)| (kind, task, post_inc(&mut lru_score)))
            .sorted_unstable_by(task_lru_comparator)
            .map(|(kind, task, _)| (kind, task))
            .collect::<Vec<_>>();

        let not_used_score = post_inc(&mut lru_score);
        let global_tasks = self.global_templates_from_settings().collect::<Vec<_>>();
        let associated_tasks = language
            .filter(|language| {
                LanguageSettings::resolve(
                    buffer.as_ref().map(|b| b.read(cx)),
                    Some(&language.name()),
                    cx,
                )
                .tasks
                .enabled
            })
            .and_then(|language| {
                language
                    .context_provider()
                    .map(|provider| provider.associated_tasks(buffer, cx))
            });
        let worktree_tasks = worktree
            .into_iter()
            .flat_map(|worktree| self.worktree_templates_from_settings(worktree))
            .collect::<Vec<_>>();
        #[cfg(feature = "jvm-tools")]
        let worktree_tasks = {
            let mut worktree_tasks = worktree_tasks;
            if let Some(worktree_id) = worktree {
                worktree_tasks.extend(self.jvm_build_tasks(worktree_id, cx));
            }
            worktree_tasks
        };
        let task_contexts = task_contexts.clone();
        cx.background_spawn(async move {
            let language_tasks = if let Some(task) = associated_tasks {
                task.await.map(|templates| {
                    templates
                        .0
                        .into_iter()
                        .flat_map(|task| Some((task_source_kind.clone()?, task)))
                })
            } else {
                None
            };

            let worktree_tasks = worktree_tasks
                .into_iter()
                .chain(language_tasks.into_iter().flatten())
                .chain(global_tasks);

            let new_resolved_tasks = worktree_tasks
                .flat_map(|(kind, task)| {
                    let id_base = kind.to_id_base();

                    if let TaskSourceKind::Worktree { id, .. } = &kind {
                        None.or_else(|| {
                            let (_, _, item_context) =
                                task_contexts.active_item_context.as_ref().filter(
                                    |(worktree_id, _, _)| Some(id) == worktree_id.as_ref(),
                                )?;
                            task.resolve_task(&id_base, item_context)
                        })
                        .or_else(|| {
                            let (_, worktree_context) = task_contexts
                                .active_worktree_context
                                .as_ref()
                                .filter(|(worktree_id, _)| id == worktree_id)?;
                            task.resolve_task(&id_base, worktree_context)
                        })
                        .or_else(|| {
                            if let TaskSourceKind::Worktree { id, .. } = &kind {
                                let worktree_context = task_contexts
                                    .other_worktree_contexts
                                    .iter()
                                    .find(|(worktree_id, _)| worktree_id == id)
                                    .map(|(_, context)| context)?;
                                task.resolve_task(&id_base, worktree_context)
                            } else {
                                None
                            }
                        })
                    } else {
                        None.or_else(|| {
                            let (_, _, item_context) =
                                task_contexts.active_item_context.as_ref()?;
                            task.resolve_task(&id_base, item_context)
                        })
                        .or_else(|| {
                            let (_, worktree_context) =
                                task_contexts.active_worktree_context.as_ref()?;
                            task.resolve_task(&id_base, worktree_context)
                        })
                    }
                    .or_else(|| task.resolve_task(&id_base, &TaskContext::default()))
                    .map(move |resolved_task| (kind.clone(), resolved_task, not_used_score))
                })
                .filter(|(_, resolved_task, _)| {
                    match task_labels_to_ids.entry(resolved_task.resolved_label.clone()) {
                        hash_map::Entry::Occupied(mut o) => {
                            // Allow new tasks with the same label, if their context is different
                            o.get_mut().insert(resolved_task.id.clone())
                        }
                        hash_map::Entry::Vacant(v) => {
                            v.insert(HashSet::from_iter(Some(resolved_task.id.clone())));
                            true
                        }
                    }
                })
                .sorted_unstable_by(task_lru_comparator)
                .map(|(kind, task, _)| (kind, task))
                .collect::<Vec<_>>();

            (previously_spawned_tasks, new_resolved_tasks)
        })
    }

    /// Returns the last scheduled task by task_id if provided.
    /// Otherwise, returns the last scheduled task.
    pub fn last_scheduled_task(
        &self,
        task_id: Option<&TaskId>,
    ) -> Option<(TaskSourceKind, ResolvedTask)> {
        if let Some(task_id) = task_id {
            self.last_scheduled_tasks
                .iter()
                .find(|(_, task)| &task.id == task_id)
                .cloned()
        } else {
            self.last_scheduled_tasks.back().cloned()
        }
    }

    /// Registers task "usage" as being scheduled – to be used for LRU sorting when listing all tasks.
    pub fn task_scheduled(
        &mut self,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
    ) {
        self.last_scheduled_tasks
            .push_back((task_source_kind, resolved_task));
        if self.last_scheduled_tasks.len() > 5_000 {
            self.last_scheduled_tasks.pop_front();
        }
        #[cfg(feature = "jvm-tools")]
        self.prune_jvm_operations(None);
    }

    /// Deletes a resolved task from history, using its id.
    /// A similar may still resurface in `used_and_current_resolved_tasks` when its [`TaskTemplate`] is resolved again.
    pub fn delete_previously_used(&mut self, id: &TaskId) {
        self.last_scheduled_tasks.retain(|(_, task)| &task.id != id);
        #[cfg(feature = "jvm-tools")]
        self.prune_jvm_operations(None);
    }

    /// Returns global task templates with the provided tag.
    pub fn global_templates_with_tag(&self, tag: &str) -> Vec<(TaskSourceKind, TaskTemplate)> {
        self.global_templates_from_settings()
            .filter(|(_, template)| template.tags.iter().any(|template_tag| template_tag == tag))
            .collect()
    }

    /// Resolves global task templates with the provided tag against the provided task context.
    pub fn resolve_global_tasks_with_tag(
        &self,
        tag: &str,
        task_context: &TaskContext,
    ) -> Vec<(TaskSourceKind, ResolvedTask)> {
        self.global_templates_with_tag(tag)
            .into_iter()
            .filter_map(|(task_source_kind, task_template)| {
                let id_base = task_source_kind.to_id_base();
                task_template
                    .resolve_task(&id_base, task_context)
                    .map(|resolved_task| (task_source_kind, resolved_task))
            })
            .collect()
    }

    /// Returns all task templates (worktree and global) that have at least one
    /// hook in the provided set.
    pub fn templates_with_hooks(
        &self,
        hooks: &HashSet<TaskHook>,
        worktree: WorktreeId,
    ) -> Vec<(TaskSourceKind, TaskTemplate)> {
        self.worktree_templates_from_settings(worktree)
            .chain(self.global_templates_from_settings())
            .filter(|(_, template)| !template.hooks.is_disjoint(hooks))
            .collect()
    }

    fn global_templates_from_settings(
        &self,
    ) -> impl '_ + Iterator<Item = (TaskSourceKind, TaskTemplate)> {
        self.templates_from_settings.global_scenarios()
    }

    fn global_debug_scenarios_from_settings(
        &self,
    ) -> impl '_ + Iterator<Item = (TaskSourceKind, DebugScenario)> {
        self.scenarios_from_settings.global_scenarios()
    }

    fn worktree_scenarios_from_settings(
        &self,
        worktree: WorktreeId,
    ) -> impl '_ + Iterator<Item = (TaskSourceKind, DebugScenario)> {
        self.scenarios_from_settings.worktree_scenarios(worktree)
    }

    fn worktree_templates_from_settings(
        &self,
        worktree: WorktreeId,
    ) -> impl '_ + Iterator<Item = (TaskSourceKind, TaskTemplate)> {
        self.templates_from_settings.worktree_scenarios(worktree)
    }

    #[cfg(feature = "jvm-tools")]
    fn jvm_build_tasks(
        &self,
        worktree_id: WorktreeId,
        cx: &App,
    ) -> Vec<(TaskSourceKind, TaskTemplate)> {
        let Some(worktree_store) = self.worktree_store.as_ref().and_then(WeakEntity::upgrade)
        else {
            return Vec::new();
        };
        let Some(worktree) = worktree_store.read(cx).worktree_for_id(worktree_id, cx) else {
            return Vec::new();
        };
        let worktree = worktree.read(cx);
        let has_file = |name: &str| {
            RelPath::from_unix_str(name)
                .ok()
                .and_then(|path| worktree.entry_for_path(path))
                .is_some_and(jvm_build_entry_is_available)
        };
        let path_style = worktree.path_style();
        let mut tasks = Vec::new();

        let maven_wrapper = match path_style {
            PathStyle::Unix => ("mvnw", "./mvnw"),
            PathStyle::Windows => ("mvnw.cmd", ".\\mvnw.cmd"),
        };
        if has_file("pom.xml") && has_file(maven_wrapper.0) {
            tasks.push(("Build Maven project", maven_wrapper.1, "package"));
        }

        let gradle_wrapper = match path_style {
            PathStyle::Unix => ("gradlew", "./gradlew"),
            PathStyle::Windows => ("gradlew.bat", ".\\gradlew.bat"),
        };
        if [
            "settings.gradle",
            "settings.gradle.kts",
            "build.gradle",
            "build.gradle.kts",
        ]
        .iter()
        .any(|name| has_file(name))
            && has_file(gradle_wrapper.0)
        {
            tasks.push(("Build Gradle project", gradle_wrapper.1, "build"));
        }

        tasks
            .into_iter()
            .map(|(label, command, argument)| {
                (
                    TaskSourceKind::groovy_jvm_build(worktree_id),
                    TaskTemplate {
                        label: label.to_string(),
                        command: command.to_string(),
                        args: vec![argument.to_string()],
                        cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
                        ..TaskTemplate::default()
                    },
                )
            })
            .collect()
    }

    #[cfg(feature = "jvm-tools")]
    pub fn groovy_jvm_build_task_is_current(
        &self,
        worktree_id: WorktreeId,
        resolved_task: &ResolvedTask,
        cx: &App,
    ) -> bool {
        let Some(worktree_store) = self.worktree_store.as_ref().and_then(WeakEntity::upgrade)
        else {
            return false;
        };
        let Some(worktree) = worktree_store.read(cx).worktree_for_id(worktree_id, cx) else {
            return false;
        };
        let worktree_root = worktree.read(cx).abs_path();
        self.jvm_build_tasks(worktree_id, cx)
            .iter()
            .any(|(_, task)| {
                task == resolved_task.original_task()
                    && resolved_task.resolved.command.as_deref() == Some(task.command.as_str())
                    && resolved_task.resolved.args == task.args
                    && resolved_task.resolved.cwd.as_deref() == Some(worktree_root.as_ref())
                    && resolved_task.resolved.shell == task.shell
            })
    }

    #[cfg(feature = "jvm-tools")]
    fn prune_jvm_operations(&mut self, pending: Option<&TaskId>) {
        let history: HashSet<_> = self
            .last_scheduled_tasks
            .iter()
            .map(|(_, task)| &task.id)
            .collect();
        self.jvm_operations
            .retain(|id, _| pending == Some(id) || history.contains(id));
    }

    #[cfg(feature = "jvm-tools")]
    pub fn register_jvm_operation(
        &mut self,
        receipt: crate::jvm_project::JvmBuildOperationReceipt,
        snapshot: &crate::jvm_project::JdtlsProjectSnapshot,
        current: bool,
        cx: &App,
    ) -> Result<(TaskSourceKind, ResolvedTask)> {
        use anyhow::{Context as _, ensure};
        use sha2::{Digest as _, Sha256};
        let store = self
            .worktree_store
            .as_ref()
            .and_then(WeakEntity::upgrade)
            .context("The selected build project is unavailable; reopen it and refresh")?;
        let worktree_entity = store
            .read(cx)
            .worktree_for_id(receipt.worktree_id, cx)
            .context("The selected build project is unavailable; reopen it and refresh")?;
        let worktree = worktree_entity.read(cx);
        ensure!(
            worktree.is_local() && worktree.is_visible(),
            "Discovered build operations currently require a visible local project; remote execution is not qualified"
        );
        let plan = receipt.plan(snapshot, current, worktree.path_style())?;
        ensure!(
            worktree.abs_path().to_str().is_some(),
            "The build root needs a valid Unicode path; reopen it under a supported path"
        );
        let root_entry = worktree
            .snapshot()
            .root_entry()
            .filter(|entry| entry.is_dir())
            .context("The selected build root is unavailable; reopen the project and refresh")?
            .clone();
        let wrapper = worktree.snapshot().entry_for_path(&plan.wrapper.path)
            .filter(|entry| jvm_build_entry_is_available(entry))
            .context("The project wrapper is missing or unavailable; restore the trusted local wrapper and refresh")?.clone();
        let manifests = plan
            .build_files
            .iter()
            .map(|path| worktree.snapshot().entry_for_path(&path.path).cloned())
            .collect::<Vec<_>>();
        ensure!(
            manifests
                .iter()
                .any(|entry| entry.as_ref().is_some_and(jvm_build_entry_is_available))
                && manifests.iter().flatten().all(jvm_build_entry_is_available),
            "Build files are missing or unavailable; restore the trusted project files and refresh"
        );
        let location = settings::SettingsLocation {
            worktree_id: receipt.worktree_id,
            path: RelPath::empty(),
        };
        let project_settings = crate::project_settings::ProjectSettings::get(Some(location), cx);
        project_settings
            .jvm_build_selection_intent(settings::SettingsStore::global(cx), location)?;
        let settings = serde_json::to_value(&project_settings.jvm_builds)?;
        let source = TaskSourceKind::groovy_jvm_operation(receipt.worktree_id);
        let binding = serde_json::to_vec(&(
            receipt.server_id.0,
            receipt.generation,
            receipt.module_root.path.as_unix_str(),
            &receipt.operations,
            worktree.abs_path().as_ref(),
            format!("{:?}", worktree_entity.entity_id()),
            format!(
                "{:?}",
                (&root_entry.id, root_entry.inode, &root_entry.canonical_path)
            ),
            format!("{:?}", (&wrapper, &manifests)),
            &settings,
        ))?;
        let id_base = format!("{}_{:x}", source.to_id_base(), Sha256::digest(binding));
        let context = TaskContext {
            cwd: Some(worktree.abs_path().as_ref().to_owned()),
            task_variables: {
                let mut variables = TaskVariables::default();
                variables.insert(
                    VariableName::WorktreeRoot,
                    worktree.abs_path().to_string_lossy().into_owned(),
                );
                variables
            },
            ..Default::default()
        };
        let resolved = plan.template.resolve_task_with_literal_arguments(&id_base,&context)
            .context("Build operation arguments cannot be resolved safely; repair the project path and refresh")?;
        let admitted = AdmittedJvmOperation {
            receipt,
            plan,
            resolved: resolved.clone(),
            worktree_entity: worktree_entity.entity_id(),
            root: worktree.abs_path(),
            root_entry,
            wrapper,
            manifests,
            settings,
        };
        ensure!(
            self.jvm_operations
                .get(&resolved.id)
                .is_none_or(|existing| existing == &admitted),
            "Build operation identity conflicts with a held action; reopen the project and refresh"
        );
        self.prune_jvm_operations(Some(&resolved.id));
        self.jvm_operations.insert(resolved.id.clone(), admitted);
        Ok((source, resolved))
    }

    #[cfg(feature = "jvm-tools")]
    pub fn jvm_operation_is_current(
        &self,
        worktree_id: WorktreeId,
        resolved: &ResolvedTask,
        snapshot: &crate::jvm_project::JdtlsProjectSnapshot,
        current: bool,
        cx: &App,
    ) -> bool {
        let Some(admitted) = self.jvm_operations.get(&resolved.id) else {
            return false;
        };
        if admitted.receipt.worktree_id != worktree_id || admitted.resolved != *resolved {
            return false;
        }
        let Some(store) = self.worktree_store.as_ref().and_then(WeakEntity::upgrade) else {
            return false;
        };
        let Some(entity) = store.read(cx).worktree_for_id(worktree_id, cx) else {
            return false;
        };
        let worktree = entity.read(cx);
        if !worktree.is_local()
            || !worktree.is_visible()
            || entity.entity_id() != admitted.worktree_entity
            || worktree.abs_path() != admitted.root
        {
            return false;
        }
        let worktree_snapshot = worktree.snapshot();
        let Some(root) = worktree_snapshot.root_entry() else {
            return false;
        };
        if !root.is_dir()
            || root.id != admitted.root_entry.id
            || root.inode != admitted.root_entry.inode
            || root.canonical_path != admitted.root_entry.canonical_path
        {
            return false;
        }
        if !admitted
            .receipt
            .plan(snapshot, current, worktree.path_style())
            .is_ok_and(|plan| plan == admitted.plan)
        {
            return false;
        }
        let location = settings::SettingsLocation {
            worktree_id,
            path: RelPath::empty(),
        };
        let settings = crate::project_settings::ProjectSettings::get(Some(location), cx);
        if settings
            .jvm_build_selection_intent(settings::SettingsStore::global(cx), location)
            .is_err()
            || !serde_json::to_value(&settings.jvm_builds)
                .is_ok_and(|value| value == admitted.settings)
        {
            return false;
        }
        worktree
            .snapshot()
            .entry_for_path(&admitted.plan.wrapper.path)
            == Some(&admitted.wrapper)
            && admitted
                .plan
                .build_files
                .iter()
                .zip(&admitted.manifests)
                .all(|(path, entry)| {
                    worktree.snapshot().entry_for_path(&path.path) == entry.as_ref()
                })
    }

    /// Updates in-memory task metadata from the JSON string given.
    /// Will fail if the JSON is not a valid array of objects, but will continue if any object will not parse into a [`TaskTemplate`].
    ///
    /// Global tasks are updated for no worktree provided, otherwise the worktree metadata for a given path will be updated.
    pub fn update_file_based_tasks(
        &mut self,
        location: TaskSettingsLocation<'_>,
        raw_tasks_json: Option<&str>,
    ) -> Result<(), InvalidSettingsError> {
        let raw_tasks = match parse_json_with_comments::<Vec<serde_json::Value>>(
            raw_tasks_json.unwrap_or("[]"),
        ) {
            Ok(tasks) => tasks,
            Err(e) => {
                return Err(InvalidSettingsError::Tasks {
                    path: match location {
                        TaskSettingsLocation::Global(path) => path.to_owned(),
                        TaskSettingsLocation::Worktree(settings_location) => {
                            settings_location.path.as_std_path().join(task_file_name())
                        }
                    },
                    message: format!("Failed to parse tasks file content as a JSON array: {e}"),
                });
            }
        };

        let mut validation_errors = Vec::new();
        let new_templates = raw_tasks.into_iter().filter_map(|raw_template| {
            let template = serde_json::from_value::<TaskTemplate>(raw_template).log_err()?;

            // Validate the variable names used in the `TaskTemplate`.
            let unknown_variables = template.unknown_variables();
            if !unknown_variables.is_empty() {
                let variables_list = unknown_variables
                    .iter()
                    .map(|variable| format!("${variable}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                validation_errors.push(format!(
                    "Task '{}' uses unknown variables: {}",
                    template.label, variables_list
                ));

                // Skip this template, since it uses unknown variable names, but
                // continue processing others.
                return None;
            }

            Some(template)
        });

        let parsed_templates = &mut self.templates_from_settings;
        match location {
            TaskSettingsLocation::Global(path) => {
                parsed_templates
                    .global
                    .entry(path.to_owned())
                    .insert_entry(new_templates.collect());
                self.last_scheduled_tasks.retain(|(kind, _)| {
                    if let TaskSourceKind::AbsPath { abs_path, .. } = kind {
                        abs_path != path
                    } else {
                        true
                    }
                });
            }
            TaskSettingsLocation::Worktree(location) => {
                let new_templates = new_templates.collect::<Vec<_>>();
                if new_templates.is_empty() {
                    if let Some(worktree_tasks) =
                        parsed_templates.worktree.get_mut(&location.worktree_id)
                    {
                        worktree_tasks.remove(location.path);
                    }
                } else {
                    parsed_templates
                        .worktree
                        .entry(location.worktree_id)
                        .or_default()
                        .insert(Arc::from(location.path), new_templates);
                }
                self.last_scheduled_tasks.retain(|(kind, _)| {
                    if let TaskSourceKind::Worktree {
                        directory_in_worktree,
                        id,
                        ..
                    } = kind
                    {
                        *id != location.worktree_id
                            || directory_in_worktree.as_ref() != location.path
                    } else {
                        true
                    }
                });
            }
        }

        if !validation_errors.is_empty() {
            return Err(InvalidSettingsError::Tasks {
                path: match &location {
                    TaskSettingsLocation::Global(path) => path.to_path_buf(),
                    TaskSettingsLocation::Worktree(location) => {
                        location.path.as_std_path().join(task_file_name())
                    }
                },
                message: validation_errors.join("\n"),
            });
        }

        Ok(())
    }

    /// Updates in-memory task metadata from the JSON string given.
    /// Will fail if the JSON is not a valid array of objects, but will continue if any object will not parse into a [`TaskTemplate`].
    ///
    /// Global tasks are updated for no worktree provided, otherwise the worktree metadata for a given path will be updated.
    pub fn update_file_based_scenarios(
        &mut self,
        location: TaskSettingsLocation<'_>,
        raw_tasks_json: Option<&str>,
    ) -> Result<(), InvalidSettingsError> {
        let raw_tasks = match parse_json_with_comments::<Vec<serde_json::Value>>(
            raw_tasks_json.unwrap_or("[]"),
        ) {
            Ok(tasks) => tasks,
            Err(e) => {
                return Err(InvalidSettingsError::Debug {
                    path: match location {
                        TaskSettingsLocation::Global(path) => path.to_owned(),
                        TaskSettingsLocation::Worktree(settings_location) => settings_location
                            .path
                            .as_std_path()
                            .join(debug_task_file_name()),
                    },
                    message: format!("Failed to parse tasks file content as a JSON array: {e}"),
                });
            }
        };

        let new_templates = raw_tasks
            .into_iter()
            .filter_map(|raw_template| {
                serde_json::from_value::<DebugScenario>(raw_template).log_err()
            })
            .collect::<Vec<_>>();

        let parsed_scenarios = &mut self.scenarios_from_settings;
        let mut new_definitions: HashMap<_, _> = new_templates
            .iter()
            .map(|template| (template.label.clone(), template.clone()))
            .collect();
        let previously_existing_scenarios;

        match location {
            TaskSettingsLocation::Global(path) => {
                previously_existing_scenarios = parsed_scenarios
                    .global_scenarios()
                    .map(|(_, scenario)| scenario.label)
                    .collect::<HashSet<_>>();
                parsed_scenarios
                    .global
                    .entry(path.to_owned())
                    .insert_entry(new_templates);
            }
            TaskSettingsLocation::Worktree(location) => {
                previously_existing_scenarios = parsed_scenarios
                    .worktree_scenarios(location.worktree_id)
                    .map(|(_, scenario)| scenario.label)
                    .collect::<HashSet<_>>();

                if new_templates.is_empty() {
                    if let Some(worktree_tasks) =
                        parsed_scenarios.worktree.get_mut(&location.worktree_id)
                    {
                        worktree_tasks.remove(location.path);
                    }
                } else {
                    parsed_scenarios
                        .worktree
                        .entry(location.worktree_id)
                        .or_default()
                        .insert(Arc::from(location.path), new_templates);
                }
            }
        }
        self.last_scheduled_scenarios.retain_mut(|(scenario, _)| {
            if !previously_existing_scenarios.contains(&scenario.label) {
                return true;
            }
            if let Some(new_definition) = new_definitions.remove(&scenario.label) {
                *scenario = new_definition;
                true
            } else {
                false
            }
        });

        Ok(())
    }
}

fn task_lru_comparator(
    (kind_a, task_a, lru_score_a): &(TaskSourceKind, ResolvedTask, u32),
    (kind_b, task_b, lru_score_b): &(TaskSourceKind, ResolvedTask, u32),
) -> cmp::Ordering {
    lru_score_a
        // First, display recently used templates above all.
        .cmp(lru_score_b)
        // Then, ensure more specific sources are displayed first.
        .then(task_source_kind_preference(kind_a).cmp(&task_source_kind_preference(kind_b)))
        // After that, display first more specific tasks, using more template variables.
        // Bonus points for tasks with symbol variables.
        .then(task_variables_preference(task_a).cmp(&task_variables_preference(task_b)))
        // Finally, sort by the resolved label, but a bit more specifically, to avoid mixing letters and digits.
        .then({
            NumericPrefixWithSuffix::from_numeric_prefixed_str(&task_a.resolved_label)
                .cmp(&NumericPrefixWithSuffix::from_numeric_prefixed_str(
                    &task_b.resolved_label,
                ))
                .then(task_a.resolved_label.cmp(&task_b.resolved_label))
                .then(kind_a.cmp(kind_b))
        })
}

pub fn task_source_kind_preference(kind: &TaskSourceKind) -> u32 {
    match kind {
        TaskSourceKind::Lsp { .. } => 0,
        TaskSourceKind::Language { .. } => 1,
        TaskSourceKind::UserInput => 2,
        TaskSourceKind::Worktree { .. } => 3,
        TaskSourceKind::AbsPath { .. } => 4,
    }
}

fn task_variables_preference(task: &ResolvedTask) -> Reverse<usize> {
    let task_variables = task.substituted_variables();
    Reverse(if task_variables.contains(&VariableName::Symbol) {
        task_variables.len() + 1
    } else {
        task_variables.len()
    })
}

/// A context provided that tries to provide values for all non-custom [`VariableName`] variants for a currently opened file.
/// Applied as a base for every custom [`ContextProvider`] unless explicitly oped out.
pub struct BasicContextProvider {
    worktree_store: Entity<WorktreeStore>,
    git_store: Entity<GitStore>,
}

impl BasicContextProvider {
    pub fn new(worktree_store: Entity<WorktreeStore>, git_store: Entity<GitStore>) -> Self {
        Self {
            worktree_store,
            git_store,
        }
    }
}

impl ContextProvider for BasicContextProvider {
    fn build_context(
        &self,
        _: &TaskVariables,
        location: ContextLocation<'_>,
        _: Option<HashMap<String, String>>,
        _: Arc<dyn LanguageToolchainStore>,
        cx: &mut App,
    ) -> Task<Result<TaskVariables>> {
        let location = location.file_location;
        let buffer = location.buffer.read(cx);
        let buffer_snapshot = buffer.snapshot();
        let symbols = buffer_snapshot.symbols_containing(location.range.start, None);
        let symbol = symbols.last().map(|symbol| {
            let range = symbol
                .name_ranges
                .last()
                .cloned()
                .unwrap_or(0..symbol.text.len());
            symbol.text[range].to_string()
        });

        let current_file = buffer.file().and_then(|file| file.as_local());
        let Point { row, column } = location.range.start.to_point(&buffer_snapshot);
        let row = row + 1;
        let column = column + 1;
        let selected_text = buffer
            .chars_for_range(location.range.clone())
            .collect::<String>();

        let mut task_variables = TaskVariables::from_iter([
            (VariableName::Row, row.to_string()),
            (VariableName::Column, column.to_string()),
        ]);

        if let Some(symbol) = symbol {
            task_variables.insert(VariableName::Symbol, symbol);
        }
        if !selected_text.trim().is_empty() {
            task_variables.insert(VariableName::SelectedText, selected_text);
        }
        let worktree = buffer
            .file()
            .map(|file| file.worktree_id(cx))
            .and_then(|worktree_id| {
                self.worktree_store
                    .read(cx)
                    .worktree_for_id(worktree_id, cx)
            });

        if let Some(worktree) = worktree {
            let worktree = worktree.read(cx);
            let path_style = worktree.path_style();
            task_variables.insert(
                VariableName::WorktreeRoot,
                worktree.abs_path().to_string_lossy().into_owned(),
            );
            if let Some(current_file) = current_file.as_ref() {
                let relative_path = current_file.path();
                task_variables.insert(
                    VariableName::RelativeFile,
                    relative_path.display(path_style).to_string(),
                );
                if let Some(relative_dir) = relative_path.parent() {
                    task_variables.insert(
                        VariableName::RelativeDir,
                        if relative_dir.is_empty() {
                            String::from(".")
                        } else {
                            relative_dir.display(path_style).to_string()
                        },
                    );
                }
            }
        }

        if let Some(worktree_id) = location.buffer.read(cx).file().map(|f| f.worktree_id(cx)) {
            if let Some(path) = self
                .git_store
                .read(cx)
                .original_repo_path_for_worktree(worktree_id, cx)
            {
                task_variables.insert(
                    VariableName::MainGitWorktree,
                    path.to_string_lossy().into_owned(),
                );
            }
        }

        if let Some(current_file) = current_file {
            let path = current_file.abs_path(cx);
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                task_variables.insert(VariableName::Filename, String::from(filename));
            }

            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                task_variables.insert(VariableName::Stem, stem.into());
            }

            if let Some(dirname) = path.parent().and_then(|s| s.to_str()) {
                task_variables.insert(VariableName::Dirname, dirname.into());
            }

            task_variables.insert(VariableName::File, path.to_string_lossy().into_owned());
        }

        if let Some(language) = buffer.language() {
            task_variables.insert(VariableName::Language, language.name().to_string());
        }

        Task::ready(Ok(task_variables))
    }
}

/// A ContextProvider that doesn't provide any task variables on it's own, though it has some associated tasks.
pub struct ContextProviderWithTasks {
    templates: TaskTemplates,
}

impl ContextProviderWithTasks {
    pub fn new(definitions: TaskTemplates) -> Self {
        Self {
            templates: definitions,
        }
    }
}

impl ContextProvider for ContextProviderWithTasks {
    fn associated_tasks(&self, _: Option<Entity<Buffer>>, _: &App) -> Task<Option<TaskTemplates>> {
        Task::ready(Some(self.templates.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn jvm_operation_admission_binds_current_wrapper_history_and_literal_data(
        cx: &mut gpui::TestAppContext,
    ) {
        use crate::jvm_project::*;
        use fs::Fs as _;
        use gpui::BorrowAppContext as _;
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/operation project",serde_json::json!({"pom.xml":"<project/>","mvnw":"#!/bin/sh","application":{"pom.xml":"<project/>"}})).await;
        let project =
            crate::Project::test(fs.clone(), [std::path::Path::new("/operation project")], cx)
                .await;
        let (inventory, worktree_id) = project.read_with(cx, |project, cx| {
            (
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .expect("inventory")
                    .clone(),
                project
                    .worktrees(cx)
                    .next()
                    .expect("worktree")
                    .read(cx)
                    .id(),
            )
        });
        let path = |text: &str| crate::ProjectPath {
            worktree_id,
            path: RelPath::from_unix_str(text).expect("fixture path").into(),
        };
        let module = path("application");
        let snapshot = JdtlsProjectSnapshot {
            metadata: Default::default(),
            server_id: LanguageServerId(2),
            worktree_id,
            generation: 3,
            partial: true,
            projects: vec![JdtlsProject {
                module_root: module.clone(),
                membership: JdtlsProjectMembership::ImportedJava,
                imported_build_kind: JdtlsImportedBuildKind::Maven,
                source_roots: vec![],
                project_references: vec![],
                library_entries: 0,
                library_files: vec![],
                library_artifacts: vec![],
                project_vm_known: false,
                project_vm_java_major: None,
                server_launch_java_major: None,
                java_source_level: None,
                java_target_level: None,
                gradle: None,
                maven: Some(MavenProject {
                    pom: path("application/pom.xml"),
                    modules: vec![],
                    group_id: "fixture".into(),
                    artifact_id: "application".into(),
                    version: "1".into(),
                    packaging: "jar".into(),
                    profiles: Default::default(),
                    profile_definitions_known: false,
                    profile_definitions: vec![],
                    reactor_position: None,
                    reactor_order_problem: None,
                    plugins: vec![],
                    lifecycles: vec![MavenLifecycle {
                        id: "default".into(),
                        phases: vec!["verify".into(), "compile".into()],
                    }],
                    dependency_graph: None,
                }),
            }],
        };
        let receipt = JvmBuildOperationReceipt::new(
            &snapshot,
            true,
            module,
            vec![JvmBuildOperation::MavenPhase {
                lifecycle: "default".into(),
                phase: "verify".into(),
            }],
            path::PathStyle::Unix,
        )
        .expect("receipt");
        let (source, resolved) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(receipt.clone(), &snapshot, true, cx)
            })
            .expect("admission");
        assert_eq!(
            resolved.resolved.args,
            ["--offline", "--file", "application/pom.xml", "verify"]
        );
        assert_eq!(resolved.resolved.literal_prefix_args, 4);
        assert_eq!(
            resolved.resolved.command.as_deref(),
            Some("/operation project/mvnw")
        );
        inventory.read_with(cx, |inventory, cx| {
            assert!(inventory.jvm_operation_is_current(worktree_id, &resolved, &snapshot, true, cx))
        });
        assert_eq!(
            TaskContexts::default().reresolve_task(&source, &resolved),
            resolved,
            "held receipt must survive ordinary context resolution"
        );
        for field in 0..5 {
            let mut changed = resolved.clone();
            match field {
                0 => changed.resolved.args.push("--online".into()),
                1 => changed.resolved.command = Some("evil".into()),
                2 => changed.resolved.cwd = Some("/foreign".into()),
                3 => changed.resolved.literal_prefix_args = 0,
                _ => changed.resolved.shell = task::Shell::Program("foreign".into()),
            }
            inventory.read_with(cx, |inventory, cx| {
                assert!(!inventory.jvm_operation_is_current(
                    worktree_id,
                    &changed,
                    &snapshot,
                    true,
                    cx
                ))
            });
        }
        for contents in [
            r#"{"jvm_builds":{"maven":{"selected_poms":[]}}}"#,
            r#"{"jvm_builds":{"maven":{"selected_poms":"private-canary"}}}"#,
        ] {
            cx.update(|cx| {
                let result = cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                    store.set_local_settings(
                        worktree_id,
                        settings::LocalSettingsPath::InWorktree(RelPath::empty_arc()),
                        settings::LocalSettingsKind::Settings,
                        Some(contents),
                        cx,
                    )
                });
                assert_eq!(result.is_err(), contents.contains("private-canary"));
            });
            inventory.read_with(cx, |inventory, cx| {
                assert!(!inventory.jvm_operation_is_current(
                    worktree_id,
                    &resolved,
                    &snapshot,
                    true,
                    cx
                ));
            });
        }
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.set_local_settings(
                    worktree_id,
                    settings::LocalSettingsPath::InWorktree(RelPath::empty_arc()),
                    settings::LocalSettingsKind::Settings,
                    None,
                    cx,
                )
            })
            .expect("restore fixture settings");
        });
        inventory.read_with(cx, |inventory, cx| {
            assert!(inventory.jvm_operation_is_current(
                worktree_id,
                &resolved,
                &snapshot,
                true,
                cx
            ));
        });
        fs.insert_file(
            "/operation project/application/pom.xml",
            b"changed manifest".to_vec(),
        )
        .await;
        cx.run_until_parked();
        inventory.read_with(cx, |inventory, cx| {
            assert!(!inventory.jvm_operation_is_current(
                worktree_id,
                &resolved,
                &snapshot,
                true,
                cx
            ));
        });
        let (source, resolved) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(receipt.clone(), &snapshot, true, cx)
            })
            .expect("new manifest receipt");
        inventory.update(cx, |inventory, _| {
            inventory.task_scheduled(source.clone(), resolved.clone())
        });
        let mut next = snapshot.clone();
        next.generation += 1;
        let mut next_receipt = receipt.clone();
        next_receipt.generation = next.generation;
        let (_, next_task) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(next_receipt, &next, true, cx)
            })
            .expect("replacement generation");
        assert_ne!(next_task.id, resolved.id);
        inventory.read_with(cx, |inventory, cx| {
            assert_eq!(inventory.jvm_operations.len(), 2);
            assert!(!inventory.jvm_operation_is_current(worktree_id, &resolved, &next, true, cx));
            assert!(inventory.jvm_operation_is_current(worktree_id, &next_task, &next, true, cx));
        });
        fs.insert_file("/operation project/mvnw", b"changed wrapper".to_vec())
            .await;
        cx.run_until_parked();
        inventory.read_with(cx, |inventory, cx| {
            assert!(!inventory.jvm_operation_is_current(worktree_id, &next_task, &next, true, cx))
        });
        let (_, repaired) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(receipt.clone(), &snapshot, true, cx)
            })
            .expect("new wrapper receipt");
        assert_ne!(repaired.id, resolved.id);
        inventory.update(cx, |inventory, _| {
            inventory.delete_previously_used(&resolved.id);
            assert!(inventory.jvm_operations.is_empty());
        });
        assert!(
            inventory
                .update(cx, |inventory, cx| inventory.register_jvm_operation(
                    receipt.clone(),
                    &snapshot,
                    false,
                    cx
                ))
                .is_err()
        );
        fs.remove_file(
            std::path::Path::new("/operation project/mvnw"),
            Default::default(),
        )
        .await
        .expect("remove owned fixture wrapper");
        cx.run_until_parked();
        assert!(
            inventory
                .update(cx, |inventory, cx| inventory.register_jvm_operation(
                    receipt.clone(),
                    &snapshot,
                    true,
                    cx
                ))
                .is_err()
        );
        fs.insert_file("/operation project/mvnw", b"restored wrapper".to_vec())
            .await;
        cx.run_until_parked();
        let (_, before_root_change) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(receipt.clone(), &snapshot, true, cx)
            })
            .expect("restored local files");
        let old_root = inventory.read_with(cx, |inventory, _| {
            inventory.jvm_operations[&before_root_change.id]
                .root_entry
                .inode
        });
        fs.remove_dir(
            std::path::Path::new("/operation project"),
            fs::RemoveOptions {
                recursive: true,
                ignore_if_not_exists: false,
            },
        )
        .await
        .expect("remove only fake fixture root");
        fs.insert_tree("/operation project", serde_json::json!({"pom.xml":"<project/>","mvnw":"restored wrapper","application":{"pom.xml":"changed manifest"}})).await;
        cx.run_until_parked();
        inventory.read_with(cx, |inventory, cx| {
            assert!(
                !inventory.jvm_operation_is_current(
                    worktree_id,
                    &before_root_change,
                    &snapshot,
                    true,
                    cx
                ),
                "held history cannot revive a replaced root"
            );
        });
        let (_, after_root_change) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(receipt.clone(), &snapshot, true, cx)
            })
            .expect("replacement root admission");
        inventory.read_with(cx, |inventory, cx| {
            assert_ne!(
                inventory.jvm_operations[&after_root_change.id]
                    .root_entry
                    .inode,
                old_root
            );
            assert_ne!(before_root_change.id, after_root_change.id);
            assert!(inventory.jvm_operation_is_current(
                worktree_id,
                &after_root_change,
                &snapshot,
                true,
                cx
            ));
        });
        for generation in 100..120 {
            let mut next = snapshot.clone();
            next.generation = generation;
            let mut next_receipt = receipt.clone();
            next_receipt.generation = generation;
            inventory.update(cx, |inventory, cx| {
                inventory
                    .register_jvm_operation(next_receipt, &next, true, cx)
                    .expect("bounded pending receipt");
                assert!(inventory.jvm_operations.len() <= inventory.last_scheduled_tasks.len() + 1);
            });
        }
    }

    fn context_with_greeting(value: &str) -> TaskContext {
        TaskContext {
            task_variables: TaskVariables::from_iter([(
                VariableName::Custom(Cow::Owned("GREETING".to_string())),
                value.to_string(),
            )]),
            ..TaskContext::default()
        }
    }

    fn greeting_template() -> TaskTemplate {
        TaskTemplate {
            label: "echo $ZED_CUSTOM_GREETING".to_string(),
            command: "echo".to_string(),
            args: vec!["$ZED_CUSTOM_GREETING".to_string()],
            ..TaskTemplate::default()
        }
    }

    #[gpui::test]
    fn debug_history_replays_latest_admitted_scenario_and_evicts_oldest(cx: &mut App) {
        let inventory = Inventory::new(cx);
        let scenario = |label: String| DebugScenario {
            adapter: "fake-adapter".into(),
            label: label.into(),
            build: None,
            config: serde_json::json!({"request": "launch"}),
            tcp_connection: None,
        };
        inventory.update(cx, |inventory, _| {
            let legacy = scenario("legacy history".into());
            inventory.scenario_scheduled(legacy.clone(), SharedTaskContext::default(), None, None);
            assert!(
                inventory
                    .last_scheduled_scenario()
                    .expect("legacy recorded")
                    .1
                    .validate_replay_policy()
                    .is_err()
            );
            let current = scenario("current ordinary manual".into());
            inventory.admitted_scenario_scheduled(
                current.clone(),
                SharedTaskContext::default(),
                None,
                None,
            );
            let (replay, context) = inventory.last_scheduled_scenario().expect("current replay");
            assert_eq!(replay, &current);
            assert!(context.validate_replay_policy().is_ok());
            assert_eq!(inventory.last_scheduled_scenarios.len(), 2);
            assert!(
                inventory
                    .last_scheduled_scenarios
                    .iter()
                    .any(|(entry, context)| entry == &legacy
                        && context.experimental_policy_version == 0)
            );
            inventory.admitted_scenario_scheduled(
                current.clone(),
                SharedTaskContext::default(),
                None,
                None,
            );
            assert_eq!(inventory.last_scheduled_scenarios.len(), 2);
            assert_eq!(
                &inventory
                    .last_scheduled_scenario()
                    .expect("current deduplicated replay")
                    .0,
                &current
            );
            for index in 0..4_999 {
                inventory.scenario_scheduled(
                    scenario(format!("bounded history {index}")),
                    SharedTaskContext::default(),
                    None,
                    None,
                );
            }
            assert_eq!(inventory.last_scheduled_scenarios.len(), 5_000);
            assert!(
                !inventory
                    .last_scheduled_scenarios
                    .iter()
                    .any(|(entry, _)| entry == &legacy)
            );
            let newest = scenario("fresh after history capacity".into());
            inventory.admitted_scenario_scheduled(
                newest.clone(),
                SharedTaskContext::default(),
                None,
                None,
            );
            assert_eq!(inventory.last_scheduled_scenarios.len(), 5_000);
            let (replay, context) = inventory
                .last_scheduled_scenario()
                .expect("fresh bounded replay");
            assert_eq!(replay, &newest);
            assert!(context.validate_replay_policy().is_ok());
            assert!(
                !inventory
                    .last_scheduled_scenarios
                    .iter()
                    .any(|(entry, _)| entry == &current)
            );
            assert!(
                inventory
                    .last_scheduled_scenarios
                    .iter()
                    .filter(|(entry, _)| entry.label.starts_with("bounded history "))
                    .all(|(_, context)| context.experimental_policy_version == 0)
            );
        });
    }

    #[test]
    fn debug_replay_policy_refuses_unknown_and_future_versions() {
        let mut context = DebugScenarioContext::default();
        assert!(context.validate_replay_policy().is_err());
        context.experimental_policy_version =
            DebugScenarioContext::CURRENT_EXPERIMENTAL_POLICY_VERSION;
        assert!(context.validate_replay_policy().is_ok());
        context.experimental_policy_version += 1;
        assert!(context.validate_replay_policy().is_err());
    }

    #[test]
    fn reresolve_task_uses_current_active_item_context() {
        let template = greeting_template();
        let stale = template
            .resolve_task("oneshot", &context_with_greeting("stale"))
            .expect("template should resolve against the stale context");
        assert_eq!(stale.resolved_label, "echo stale");

        let task_contexts = TaskContexts {
            active_item_context: Some((None, None, context_with_greeting("fresh"))),
            ..TaskContexts::default()
        };

        let reresolved = task_contexts.reresolve_task(&TaskSourceKind::UserInput, &stale);
        assert_eq!(
            reresolved.resolved_label, "echo fresh",
            "re-resolution should pick up the current active item context"
        );
    }

    #[test]
    fn reresolve_task_scopes_worktree_tasks_to_their_worktree() {
        let template = greeting_template();
        let stale = template
            .resolve_task("worktree", &context_with_greeting("stale"))
            .expect("template should resolve against the stale context");

        let task_worktree = WorktreeId::from_usize(1);
        let other_worktree = WorktreeId::from_usize(2);
        let kind = TaskSourceKind::Worktree {
            id: task_worktree,
            directory_in_worktree: RelPath::empty_arc(),
            id_base: Cow::Borrowed("worktree"),
        };

        // The active item belongs to a different worktree, so it must not be used; the
        // task's own worktree context should win instead.
        let task_contexts = TaskContexts {
            active_item_context: Some((Some(other_worktree), None, context_with_greeting("other"))),
            active_worktree_context: Some((task_worktree, context_with_greeting("fresh"))),
            ..TaskContexts::default()
        };

        let reresolved = task_contexts.reresolve_task(&kind, &stale);
        assert_eq!(
            reresolved.resolved_label, "echo fresh",
            "worktree tasks should only re-resolve against their own worktree's context"
        );
    }

    #[test]
    fn reresolve_task_prefers_active_item_over_worktree_context() {
        let template = greeting_template();
        let stale = template
            .resolve_task("oneshot", &context_with_greeting("stale"))
            .expect("template should resolve against the stale context");

        // Both contexts can resolve the task; the active item context is more specific
        // and must take precedence over the active worktree context.
        let task_contexts = TaskContexts {
            active_item_context: Some((None, None, context_with_greeting("item"))),
            active_worktree_context: Some((
                WorktreeId::from_usize(1),
                context_with_greeting("worktree"),
            )),
            ..TaskContexts::default()
        };

        let reresolved = task_contexts.reresolve_task(&TaskSourceKind::UserInput, &stale);
        assert_eq!(
            reresolved.resolved_label, "echo item",
            "the active item context should take precedence over the worktree context"
        );
    }

    #[test]
    fn reresolve_task_falls_back_to_original_when_no_context_resolves() {
        let template = greeting_template();
        let stale = template
            .resolve_task("oneshot", &context_with_greeting("stale"))
            .expect("template should resolve against the stale context");

        // No contexts provide the variable, and the empty default context cannot resolve
        // a template that requires it, so the original resolved task is returned unchanged.
        let task_contexts = TaskContexts::default();
        let reresolved = task_contexts.reresolve_task(&TaskSourceKind::UserInput, &stale);
        assert_eq!(reresolved, stale);
    }
}
