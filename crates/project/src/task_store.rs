use feature_flags::FeatureFlagAppExt as _;
#[cfg(feature = "rust-coverage")]
use feature_flags::RustCoverageExperimentalFeatureFlag;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Context as _;
use collections::HashMap;
use fs::Fs;
use gpui::{App, AsyncApp, Context, Entity, EventEmitter, Task, WeakEntity};
use language::{
    ContextLocation, ContextProvider as _, LanguageToolchainStore, Location,
    proto::{deserialize_anchor, serialize_anchor},
};
use rpc::{AnyProtoClient, TypedEnvelope, proto};
use settings::{InvalidSettingsError, SettingsLocation};
use task::{ResolvedTask, TaskContext, TaskTemplate, TaskVariables, VariableName};
use text::{BufferId, OffsetRangeExt};
use util::ResultExt;
use worktree::WorktreeId;

use crate::{
    BasicContextProvider, Inventory, ProjectEnvironment, buffer_store::BufferStore,
    git_store::GitStore, task_inventory::TaskSourceKind, worktree_store::WorktreeStore,
};

// platform-dependent warning
pub enum TaskStore {
    Functional(StoreState),
    Noop,
}

pub fn resolve_task_reference(
    tasks: impl IntoIterator<Item = (TaskSourceKind, TaskTemplate)>,
    label: &str,
    context: &TaskContext,
) -> anyhow::Result<(TaskSourceKind, ResolvedTask)> {
    let mut matches = tasks
        .into_iter()
        .filter(|(_, template)| template.label == label);
    let (source, template) = matches
        .next()
        .with_context(|| format!("pre-launch task `{label}` was not found"))?;
    anyhow::ensure!(
        matches.next().is_none(),
        "pre-launch task reference `{label}` is ambiguous"
    );
    let resolved = template
        .resolve_task(&source.to_id_base(), context)
        .with_context(|| format!("pre-launch task `{label}` could not be resolved"))?;
    Ok((source, resolved))
}

pub struct StoreState {
    mode: StoreMode,
    task_inventory: Entity<Inventory>,
    buffer_store: WeakEntity<BufferStore>,
    worktree_store: Entity<WorktreeStore>,
    git_store: Entity<GitStore>,
    toolchain_store: Arc<dyn LanguageToolchainStore>,
}

enum StoreMode {
    Local {
        downstream_client: Option<(AnyProtoClient, u64)>,
        environment: Entity<ProjectEnvironment>,
    },
    Remote {
        upstream_client: AnyProtoClient,
        project_id: u64,
    },
}

impl EventEmitter<crate::Event> for TaskStore {}

#[derive(Debug)]
pub enum TaskSettingsLocation<'a> {
    Global(&'a Path),
    Worktree(SettingsLocation<'a>),
}

impl TaskStore {
    pub fn init(client: Option<&AnyProtoClient>) {
        if let Some(client) = client {
            client.add_entity_request_handler(Self::handle_task_context_for_location);
        }
    }

    async fn handle_task_context_for_location(
        store: Entity<Self>,
        envelope: TypedEnvelope<proto::TaskContextForLocation>,
        mut cx: AsyncApp,
    ) -> anyhow::Result<proto::TaskContext> {
        let location = envelope
            .payload
            .location
            .context("no location given for task context handling")?;
        let (buffer_store, is_remote) = store.read_with(&cx, |store, _| {
            Ok(match store {
                TaskStore::Functional(state) => (
                    state.buffer_store.clone(),
                    match &state.mode {
                        StoreMode::Local { .. } => false,
                        StoreMode::Remote { .. } => true,
                    },
                ),
                TaskStore::Noop => {
                    anyhow::bail!("empty task store cannot handle task context requests")
                }
            })
        })?;
        let buffer_store = buffer_store
            .upgrade()
            .context("no buffer store when handling task context request")?;

        let buffer_id = BufferId::new(location.buffer_id).with_context(|| {
            format!(
                "cannot handle task context request for invalid buffer id: {}",
                location.buffer_id
            )
        })?;

        let start = location
            .start
            .and_then(deserialize_anchor)
            .context("missing task context location start")?;
        let end = location
            .end
            .and_then(deserialize_anchor)
            .context("missing task context location end")?;
        let buffer = buffer_store
            .update(&mut cx, |buffer_store, cx| {
                if is_remote {
                    buffer_store.wait_for_remote_buffer(buffer_id, cx)
                } else {
                    Task::ready(
                        buffer_store
                            .get(buffer_id)
                            .with_context(|| format!("no local buffer with id {buffer_id}")),
                    )
                }
            })
            .await?;

        let location = Location {
            buffer,
            range: start..end,
        };
        let context_task = store.update(&mut cx, |store, cx| {
            let captured_variables = {
                let mut variables = TaskVariables::from_iter(
                    envelope
                        .payload
                        .task_variables
                        .into_iter()
                        .filter_map(|(k, v)| Some((k.parse().ok()?, v))),
                );

                let snapshot = location.buffer.read(cx).snapshot();
                let range = location.range.to_offset(&snapshot);

                for range in snapshot.runnable_ranges(range) {
                    for (capture_name, value) in range.extra_captures {
                        variables.insert(VariableName::Custom(capture_name.into()), value);
                    }
                }
                variables
            };
            store.task_context_for_location(captured_variables, location, cx)
        });
        let task_context = context_task.await?.unwrap_or_default();
        Ok(proto::TaskContext {
            project_env: task_context.project_env.into_iter().collect(),
            cwd: task_context
                .cwd
                .map(|cwd| cwd.to_string_lossy().into_owned()),
            task_variables: task_context
                .task_variables
                .into_iter()
                .map(|(variable_name, variable_value)| (variable_name.to_string(), variable_value))
                .collect(),
        })
    }

    pub fn local(
        buffer_store: WeakEntity<BufferStore>,
        worktree_store: Entity<WorktreeStore>,
        toolchain_store: Arc<dyn LanguageToolchainStore>,
        environment: Entity<ProjectEnvironment>,
        git_store: Entity<GitStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let task_inventory = Inventory::new(cx);
        #[cfg(feature = "jvm-tools")]
        task_inventory.update(cx, |inventory, _| {
            inventory.bind_worktree_store(worktree_store.downgrade())
        });
        Self::Functional(StoreState {
            mode: StoreMode::Local {
                downstream_client: None,
                environment,
            },
            task_inventory,
            buffer_store,
            git_store,
            toolchain_store,
            worktree_store,
        })
    }

    pub fn remote(
        buffer_store: WeakEntity<BufferStore>,
        worktree_store: Entity<WorktreeStore>,
        toolchain_store: Arc<dyn LanguageToolchainStore>,
        upstream_client: AnyProtoClient,
        project_id: u64,
        git_store: Entity<GitStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let task_inventory = Inventory::new(cx);
        #[cfg(feature = "jvm-tools")]
        task_inventory.update(cx, |inventory, _| {
            inventory.bind_worktree_store(worktree_store.downgrade())
        });
        Self::Functional(StoreState {
            mode: StoreMode::Remote {
                upstream_client,
                project_id,
            },
            task_inventory,
            buffer_store,
            git_store,
            toolchain_store,
            worktree_store,
        })
    }

    pub fn task_context_for_location(
        &self,
        captured_variables: TaskVariables,
        location: Location,
        cx: &mut App,
    ) -> Task<anyhow::Result<Option<TaskContext>>> {
        match self {
            TaskStore::Functional(state) => match &state.mode {
                StoreMode::Local { environment, .. } => local_task_context_for_location(
                    state.worktree_store.clone(),
                    state.git_store.clone(),
                    state.toolchain_store.clone(),
                    environment.clone(),
                    captured_variables,
                    location,
                    cx,
                ),
                StoreMode::Remote {
                    upstream_client,
                    project_id,
                } => remote_task_context_for_location(
                    *project_id,
                    upstream_client.clone(),
                    state.worktree_store.clone(),
                    state.git_store.clone(),
                    captured_variables,
                    location,
                    state.toolchain_store.clone(),
                    cx,
                ),
            },
            TaskStore::Noop => Task::ready(Ok(None)),
        }
    }

    pub fn task_inventory(&self) -> Option<&Entity<Inventory>> {
        match self {
            TaskStore::Functional(state) => Some(&state.task_inventory),
            TaskStore::Noop => None,
        }
    }

    pub fn shared(&mut self, remote_id: u64, new_downstream_client: AnyProtoClient, _cx: &mut App) {
        if let Self::Functional(StoreState {
            mode: StoreMode::Local {
                downstream_client, ..
            },
            ..
        }) = self
        {
            *downstream_client = Some((new_downstream_client, remote_id));
        }
    }

    pub fn unshared(&mut self, _: &mut Context<Self>) {
        if let Self::Functional(StoreState {
            mode: StoreMode::Local {
                downstream_client, ..
            },
            ..
        }) = self
        {
            *downstream_client = None;
        }
    }

    pub(super) fn update_user_tasks(
        &self,
        location: TaskSettingsLocation<'_>,
        raw_tasks_json: Option<&str>,
        cx: &mut Context<Self>,
    ) -> Result<(), InvalidSettingsError> {
        let task_inventory = match self {
            TaskStore::Functional(state) => &state.task_inventory,
            TaskStore::Noop => return Ok(()),
        };
        let raw_tasks_json = raw_tasks_json
            .map(|json| json.trim())
            .filter(|json| !json.is_empty());

        task_inventory.update(cx, |inventory, _| {
            inventory.update_file_based_tasks(location, raw_tasks_json)
        })
    }

    pub(super) fn update_user_debug_scenarios(
        &self,
        location: TaskSettingsLocation<'_>,
        raw_tasks_json: Option<&str>,
        cx: &mut Context<Self>,
    ) -> Result<(), InvalidSettingsError> {
        let task_inventory = match self {
            TaskStore::Functional(state) => &state.task_inventory,
            TaskStore::Noop => return Ok(()),
        };
        let raw_tasks_json = raw_tasks_json
            .map(|json| json.trim())
            .filter(|json| !json.is_empty());

        task_inventory.update(cx, |inventory, _| {
            inventory.update_file_based_scenarios(location, raw_tasks_json)
        })
    }
}

/// The bounded number of ordered before-launch dependencies one named
/// configuration may declare.
pub const MAX_BEFORE_LAUNCH_DEPENDENCIES: usize = 16;

/// Resolves the ordered before-launch dependencies of one named configuration
/// into the same saved-task identities the existing Tasks owner dispatches.
/// Dependencies resolve in order through the existing saved task inventory;
/// duplicates, self-references and missing saved tasks are rejected before any
/// scheduling. The existing execution plumbing keeps ownership of running the
/// returned plans, their cancellation and failure short-circuit policy.
pub fn plan_before_launch_dependencies(
    inventory: &Entity<Inventory>,
    buffer: Option<Entity<language::Buffer>>,
    worktree_id: Option<WorktreeId>,
    launching_label: &str,
    dependencies: &[String],
    context: &TaskContext,
    cx: &mut App,
) -> anyhow::Result<Task<anyhow::Result<Vec<task::ResolvedTask>>>> {
    let labels: Vec<_> = dependencies.iter().map(String::as_str).collect();
    validate_before_launch_labels(launching_label, &labels)?;
    let templates = inventory.read(cx).list_tasks(buffer, None, worktree_id, cx);
    let launching_label = launching_label.to_owned();
    let dependencies = dependencies.to_vec();
    let context = context.clone();
    Ok(cx.spawn(async move |_| {
        let entries: Vec<_> = templates.await.into_iter()
            .filter(|(source, _)| matches!(source, TaskSourceKind::Worktree { .. } | TaskSourceKind::AbsPath { .. })
                && source.groovy_jvm_task_worktree().is_none())
            .collect();
        let labels: Vec<_> = dependencies.iter().map(String::as_str).collect();
        ordered_before_launch_entries(&launching_label, &labels, &entries)?
            .into_iter()
            .map(|(source, template)| {
                let id_base = source.to_id_base();
                let resolved = if template.literal_arguments {
                    template.resolve_task_with_literal_arguments(&id_base, &context)
                } else {
                    template.resolve_task(&id_base, &context)
                };
                resolved.with_context(|| {
                    format!(
                        "Before-launch dependency '{}' could not resolve in the launching task's context",
                        template.label
                    )
                })
            })
            .collect()
    }))
}

fn before_launch_labels(template: &TaskTemplate) -> Vec<&str> {
    template
        .tags
        .iter()
        .filter_map(|tag| tag.strip_prefix("before-launch:"))
        .collect()
}

fn validate_before_launch_labels(
    launching_label: &str,
    dependencies: &[&str],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        dependencies.len() <= MAX_BEFORE_LAUNCH_DEPENDENCIES,
        "Before-launch dependencies exceed the {MAX_BEFORE_LAUNCH_DEPENDENCIES}-dependency limit"
    );
    let mut listed_labels = std::collections::HashSet::new();
    for &label in dependencies {
        anyhow::ensure!(
            !label.trim().is_empty() && !label.chars().any(char::is_control),
            "Before-launch dependency labels must be nonempty and have no control characters"
        );
        anyhow::ensure!(
            listed_labels.insert(label),
            "Before-launch dependency '{label}' is listed more than once"
        );
        anyhow::ensure!(
            label != launching_label,
            "A before-launch dependency cannot reference its own configuration"
        );
    }
    Ok(())
}

fn ordered_before_launch_entries<'a>(
    launching_label: &'a str,
    dependencies: &[&str],
    entries: &'a [(TaskSourceKind, TaskTemplate)],
) -> anyhow::Result<Vec<&'a (TaskSourceKind, TaskTemplate)>> {
    fn collect<'a>(
        owner: &str,
        dependencies: &[&str],
        entries: &'a [(TaskSourceKind, TaskTemplate)],
        ancestors: &mut Vec<&'a str>,
        ordered: &mut Vec<&'a (TaskSourceKind, TaskTemplate)>,
    ) -> anyhow::Result<()> {
        validate_before_launch_labels(owner, dependencies)?;
        for &label in dependencies {
            if ordered.iter().any(|(_, template)| template.label == label) {
                continue;
            }
            anyhow::ensure!(
                !ancestors.contains(&label),
                "Before-launch dependencies contain a cycle at '{label}'"
            );
            anyhow::ensure!(
                ordered.len() + ancestors.len() <= MAX_BEFORE_LAUNCH_DEPENDENCIES,
                "Before-launch dependencies exceed the {MAX_BEFORE_LAUNCH_DEPENDENCIES}-dependency limit"
            );
            let mut matches = entries
                .iter()
                .filter(|(_, template)| template.label == label);
            let entry = matches.next().with_context(|| {
                format!("Before-launch dependency '{label}' is not a saved task")
            })?;
            anyhow::ensure!(
                matches.next().is_none(),
                "Before-launch dependency '{label}' is ambiguous across saved tasks"
            );
            ancestors.push(&entry.1.label);
            collect(
                &entry.1.label,
                &before_launch_labels(&entry.1),
                entries,
                ancestors,
                ordered,
            )?;
            ancestors.pop();
            ordered.push(entry);
        }
        Ok(())
    }

    let mut ordered = Vec::new();
    collect(
        launching_label,
        dependencies,
        entries,
        &mut vec![launching_label],
        &mut ordered,
    )?;
    Ok(ordered)
}

/// The bounded number of ordered saved configurations one activation trigger
/// may declare.
pub const MAX_TRIGGER_SEQUENCE_DEPENDENCIES: usize = MAX_BEFORE_LAUNCH_DEPENDENCIES;

/// The bounded before/after sync/build/rebuild/run/debug activation labels.
/// Labels are stored on saved configurations through the existing `tags` owner
/// field; unrelated tags never authorize a trigger.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum JvmTaskTrigger {
    BeforeSync,
    AfterSync,
    BeforeBuild,
    AfterBuild,
    BeforeRebuild,
    AfterRebuild,
    BeforeRun,
    AfterRun,
    BeforeDebug,
    AfterDebug,
}

impl JvmTaskTrigger {
    pub const ALL: [Self; 10] = [
        Self::BeforeSync,
        Self::AfterSync,
        Self::BeforeBuild,
        Self::AfterBuild,
        Self::BeforeRebuild,
        Self::AfterRebuild,
        Self::BeforeRun,
        Self::AfterRun,
        Self::BeforeDebug,
        Self::AfterDebug,
    ];

    /// The visible activation label shown when the trigger's sequence runs.
    pub fn label(self) -> &'static str {
        match self {
            Self::BeforeSync => "before-sync",
            Self::AfterSync => "after-sync",
            Self::BeforeBuild => "before-build",
            Self::AfterBuild => "after-build",
            Self::BeforeRebuild => "before-rebuild",
            Self::AfterRebuild => "after-rebuild",
            Self::BeforeRun => "before-run",
            Self::AfterRun => "after-run",
            Self::BeforeDebug => "before-debug",
            Self::AfterDebug => "after-debug",
        }
    }

    /// Parses one bounded activation label from a saved configuration's tags.
    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|trigger| trigger.label() == tag)
    }
}

/// Resolves the bounded ordered sequence of saved configurations tagged for
/// one activation trigger, in saved file order, through the existing saved
/// task inventory. The returned entries are resolved and dispatched by the
/// existing execution plumbing; the planner never schedules anything, so sync
/// cannot implicitly authorize untagged or arbitrary unrelated tasks.
pub fn plan_trigger_sequence(
    inventory: &Entity<Inventory>,
    worktree_id: Option<WorktreeId>,
    trigger: JvmTaskTrigger,
    cx: &mut App,
) -> Task<anyhow::Result<Vec<(TaskSourceKind, TaskTemplate)>>> {
    let entries = inventory.read(cx).list_tasks(None, None, worktree_id, cx);
    cx.spawn(async move |_| select_jvm_trigger_tasks(entries.await, trigger))
}

pub fn select_jvm_trigger_tasks(
    entries: impl IntoIterator<Item = (TaskSourceKind, TaskTemplate)>,
    trigger: JvmTaskTrigger,
) -> anyhow::Result<Vec<(TaskSourceKind, TaskTemplate)>> {
    let mut sequence = Vec::new();
    for (source, template) in entries {
        if !template.tags.iter().any(|tag| tag == trigger.label()) {
            continue;
        }
        anyhow::ensure!(
            sequence.len() < MAX_TRIGGER_SEQUENCE_DEPENDENCIES,
            "The {} trigger sequence exceeds the {MAX_TRIGGER_SEQUENCE_DEPENDENCIES}-configuration limit",
            trigger.label(),
        );
        anyhow::ensure!(
            sequence
                .iter()
                .all(|(_, saved): &(TaskSourceKind, TaskTemplate)| saved.label != template.label),
            "The {} trigger sequence lists configuration '{}' more than once",
            trigger.label(),
            template.label,
        );
        sequence.push((source, template));
    }
    sequence.sort_by(|(left, _), (right, _)| left.cmp(right));
    Ok(sequence)
}

#[cfg(feature = "jvm-tools")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SavedJvmTask {
    pub worktree_id: WorktreeId,
    pub source: TaskSourceKind,
    pub label: String,
}

#[cfg(feature = "jvm-tools")]
#[derive(Clone, PartialEq, Eq)]
pub struct PreparedSavedJvmTask {
    pub worktree: gpui::EntityId,
    pub root: Arc<Path>,
    pub task: task::ResolvedTask,
}

#[cfg(feature = "jvm-tools")]
impl SavedJvmTask {
    pub fn prepare(
        &self,
        project: &crate::Project,
        cx: &App,
    ) -> anyhow::Result<PreparedSavedJvmTask> {
        let worktree_id = self.worktree_id;
        anyhow::ensure!(
            !project.is_remote()
                && !project.is_disconnected(cx)
                && project.remote_connection_state(cx).is_none(),
            "The project host is unavailable; reconnect and refresh before running configurations"
        );
        let worktree_store = project.worktree_store();
        let Some(trusted_worktrees) =
            crate::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
        else {
            anyhow::bail!("Trust the project before running a named configuration");
        };
        anyhow::ensure!(
            !trusted_worktrees
                .read(cx)
                .restricted_worktrees(&worktree_store, cx)
                .into_iter()
                .any(|(restricted_id, _)| restricted_id == worktree_id),
            "Trust the project before running a named configuration"
        );
        let inventory = project
            .task_store()
            .read(cx)
            .task_inventory()
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!("The build Task inventory is unavailable; reopen the project")
            })?;
        let worktree = project.worktree_for_id(worktree_id, cx).context(
            "The saved configuration's project folder is unavailable; refresh the panel",
        )?;
        let worktree_entity = worktree.entity_id();
        let worktree = worktree.read(cx);
        anyhow::ensure!(
            worktree.is_local() && worktree.is_visible(),
            "The saved configuration's project folder is unavailable; refresh the panel"
        );
        let root = worktree.abs_path();
        let mut context = TaskContext {
            cwd: Some(root.to_path_buf()),
            ..TaskContext::default()
        };
        context.task_variables.insert(
            VariableName::WorktreeRoot,
            root.to_string_lossy().into_owned(),
        );

        let mut matching = inventory
            .read(cx)
            .saved_task_templates(worktree_id)
            .filter(|(source, template)| *source == self.source && template.label == self.label);
        let (_, template) = matching
            .next()
            .with_context(|| format!("Named configuration '{}' is no longer saved", self.label))?;
        anyhow::ensure!(
            matching.next().is_none(),
            "Named configuration '{}' is ambiguous in its saved task file; use a unique label",
            self.label
        );
        let id_base = self.source.to_id_base();
        let resolved = if template.literal_arguments {
            template.resolve_task_with_literal_arguments(&id_base, &context)
        } else {
            template.resolve_task(&id_base, &context)
        }
        .with_context(|| format!(
            "Named configuration '{}' did not resolve in the current project context; run it from the task list",
            self.label
        ))?;
        Ok(PreparedSavedJvmTask {
            worktree: worktree_entity,
            root,
            task: resolved,
        })
    }
}

#[cfg(feature = "jvm-tools")]
pub fn prepare_jvm_before_launch_tasks(
    project: &crate::Project,
    worktree_id: WorktreeId,
    main: &PreparedSavedJvmTask,
    cx: &App,
) -> anyhow::Result<Vec<(SavedJvmTask, PreparedSavedJvmTask)>> {
    let template = main.task.original_task();
    let labels = before_launch_labels(template);
    if labels.is_empty() {
        return Ok(Vec::new());
    }
    let inventory = project
        .task_store()
        .read(cx)
        .task_inventory()
        .context("The build Task inventory is unavailable; reopen the project")?;
    let entries: Vec<_> = inventory
        .read(cx)
        .saved_task_templates(worktree_id)
        .collect();
    ordered_before_launch_entries(&template.label, &labels, &entries)?
        .into_iter()
        .map(|(source, template)| {
            let target = SavedJvmTask {
                worktree_id,
                source: source.clone(),
                label: template.label.clone(),
            };
            let prepared = target.prepare(project, cx)?;
            Ok((target, prepared))
        })
        .collect()
}

#[cfg(feature = "jvm-tools")]
pub fn prepare_jvm_hook_tasks(
    project: &crate::Project,
    worktree_id: WorktreeId,
    trigger: JvmTaskTrigger,
    cx: &App,
) -> anyhow::Result<Vec<(SavedJvmTask, PreparedSavedJvmTask)>> {
    let Some(inventory) = project.task_store().read(cx).task_inventory() else {
        return Ok(Vec::new());
    };
    select_jvm_trigger_tasks(
        inventory.read(cx).saved_task_templates(worktree_id),
        trigger,
    )?
    .into_iter()
    .map(|(source, template)| {
        let target = SavedJvmTask {
            worktree_id,
            source,
            label: template.label,
        };
        let prepared = target.prepare(project, cx)?;
        Ok((target, prepared))
    })
    .collect()
}

fn local_task_context_for_location(
    worktree_store: Entity<WorktreeStore>,
    git_store: Entity<GitStore>,
    toolchain_store: Arc<dyn LanguageToolchainStore>,
    environment: Entity<ProjectEnvironment>,
    captured_variables: TaskVariables,
    location: Location,
    cx: &App,
) -> Task<anyhow::Result<Option<TaskContext>>> {
    let worktree_id = location.buffer.read(cx).file().map(|f| f.worktree_id(cx));
    let worktree_abs_path = worktree_id
        .and_then(|worktree_id| worktree_store.read(cx).worktree_for_id(worktree_id, cx))
        .and_then(|worktree| worktree.read(cx).root_dir());
    let fs = worktree_store.read(cx).fs();

    cx.spawn(async move |cx| {
        let project_env = environment
            .update(cx, |environment, cx| {
                environment.buffer_environment(&location.buffer, &worktree_store, cx)
            })
            .await;

        let mut task_variables = cx
            .update(|cx| {
                combine_task_variables(
                    captured_variables,
                    fs,
                    worktree_store.clone(),
                    location,
                    project_env.clone(),
                    BasicContextProvider::new(worktree_store, git_store),
                    toolchain_store,
                    cx,
                )
            })
            .await?;

        // Remove all custom entries starting with _, as they're not intended for use by the end user.
        task_variables.sweep();

        Ok(Some(TaskContext {
            project_env: project_env.unwrap_or_default(),
            cwd: worktree_abs_path.map(|p| p.to_path_buf()),
            task_variables,
        }))
    })
}

fn remote_task_context_for_location(
    project_id: u64,
    upstream_client: AnyProtoClient,
    worktree_store: Entity<WorktreeStore>,
    git_store: Entity<GitStore>,
    captured_variables: TaskVariables,
    location: Location,
    toolchain_store: Arc<dyn LanguageToolchainStore>,
    cx: &mut App,
) -> Task<anyhow::Result<Option<TaskContext>>> {
    cx.spawn(async move |cx| {
        // We need to gather a client context, as the headless one may lack certain information (e.g. tree-sitter parsing is disabled there, so symbols are not available).
        let mut remote_context = cx
            .update(|cx| {
                let worktree_root = worktree_root(&worktree_store, &location, cx);

                BasicContextProvider::new(worktree_store, git_store).build_context(
                    &TaskVariables::default(),
                    ContextLocation {
                        fs: None,
                        worktree_root,
                        file_location: &location,
                    },
                    None,
                    toolchain_store,
                    cx,
                )
            })
            .await
            .log_err()
            .unwrap_or_default();
        remote_context.extend(captured_variables);

        let buffer_id = cx.update(|cx| location.buffer.read(cx).remote_id().to_proto());
        let context_task = upstream_client.request(proto::TaskContextForLocation {
            project_id,
            location: Some(proto::Location {
                buffer_id,
                start: Some(serialize_anchor(&location.range.start)),
                end: Some(serialize_anchor(&location.range.end)),
            }),
            task_variables: remote_context
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        });
        let task_context = context_task.await?;
        Ok(Some(TaskContext {
            cwd: task_context.cwd.map(PathBuf::from),
            task_variables: task_context
                .task_variables
                .into_iter()
                .filter_map(
                    |(variable_name, variable_value)| match variable_name.parse() {
                        Ok(variable_name) => Some((variable_name, variable_value)),
                        Err(()) => {
                            log::error!("Unknown variable name: {variable_name}");
                            None
                        }
                    },
                )
                .collect(),
            project_env: task_context.project_env.into_iter().collect(),
        }))
    })
}

fn worktree_root(
    worktree_store: &Entity<WorktreeStore>,
    location: &Location,
    cx: &mut App,
) -> Option<PathBuf> {
    location
        .buffer
        .read(cx)
        .file()
        .map(|f| f.worktree_id(cx))
        .and_then(|worktree_id| worktree_store.read(cx).worktree_for_id(worktree_id, cx))
        .and_then(|worktree| {
            let worktree = worktree.read(cx);
            if !worktree.is_visible() {
                return None;
            }
            let root_entry = worktree.root_entry()?;
            if !root_entry.is_dir() {
                return None;
            }
            Some(worktree.absolutize(&root_entry.path))
        })
}

fn combine_task_variables(
    mut captured_variables: TaskVariables,
    fs: Option<Arc<dyn Fs>>,
    worktree_store: Entity<WorktreeStore>,
    location: Location,
    project_env: Option<HashMap<String, String>>,
    baseline: BasicContextProvider,
    toolchain_store: Arc<dyn LanguageToolchainStore>,
    cx: &mut App,
) -> Task<anyhow::Result<TaskVariables>> {
    let language_context_provider = location
        .buffer
        .read(cx)
        .language()
        .and_then(|language| language.context_provider());
    cx.spawn(async move |cx| {
        let baseline = cx
            .update(|cx| {
                let worktree_root = worktree_root(&worktree_store, &location, cx);
                baseline.build_context(
                    &captured_variables,
                    ContextLocation {
                        fs: fs.clone(),
                        worktree_root,
                        file_location: &location,
                    },
                    project_env.clone(),
                    toolchain_store.clone(),
                    cx,
                )
            })
            .await
            .context("building basic default context")?;
        captured_variables.extend(baseline);
        if let Some(provider) = language_context_provider {
            captured_variables.extend(
                cx.update(|cx| {
                    let worktree_root = worktree_root(&worktree_store, &location, cx);
                    provider.build_context(
                        &captured_variables,
                        ContextLocation {
                            fs,
                            worktree_root,
                            file_location: &location,
                        },
                        project_env,
                        toolchain_store,
                        cx,
                    )
                })
                .await?,
            );
        }
        Ok(captured_variables)
    })
}

pub struct ProjectTaskArtifactCapture {
    inner: ProjectTaskArtifactCaptureKind,
}

enum ProjectTaskArtifactCaptureKind {
    #[cfg(feature = "rust-coverage")]
    RustCoverage(crate::rust_coverage_provider::RustCoverageTaskCapture),
}

impl ProjectTaskArtifactCapture {
    pub fn run_id(&self) -> &str {
        #[cfg(not(feature = "rust-coverage"))]
        {
            let _ = &self.inner;
            ""
        }
        #[cfg(feature = "rust-coverage")]
        match &self.inner {
            #[cfg(feature = "rust-coverage")]
            ProjectTaskArtifactCaptureKind::RustCoverage(capture) => &capture.run_id,
        }
    }

    pub fn is_current(&self, cx: &App) -> bool {
        #[cfg(not(feature = "rust-coverage"))]
        {
            let _ = (&self.inner, cx);
            false
        }
        #[cfg(feature = "rust-coverage")]
        match &self.inner {
            #[cfg(feature = "rust-coverage")]
            ProjectTaskArtifactCaptureKind::RustCoverage(capture) => capture.is_current(cx),
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
pub fn set_experimental_feature_for_test(feature: &str, enabled: bool, cx: &mut gpui::App) {
    use gpui::UpdateGlobal as _;
    task::validate_experimental_feature_requirement(feature)
        .expect("known experimental fixture flag");
    if !cx.has_global::<settings::SettingsStore>() {
        let store = settings::SettingsStore::test(cx);
        cx.set_global(store);
    }
    settings::SettingsStore::update_global(cx, |store, _| {
        store.register_setting::<feature_flags::FeatureFlagsSettings>()
    });
    feature_flags::FeatureFlagStore::init(cx);
    settings::SettingsStore::update_global(cx, |store, cx| {
        store.update_user_settings(cx, |content| {
            content.feature_flags.get_or_insert_default().insert(
                feature.to_string(),
                if enabled { "on" } else { "off" }.to_string(),
            );
        });
    });
}

pub struct PendingExperimentalFeatureOperation {
    requirements: Vec<String>,
    revoked: Arc<std::sync::atomic::AtomicBool>,
    _subscription: Option<gpui::Subscription>,
}

impl PendingExperimentalFeatureOperation {
    pub fn new(requirements: &[String], cx: &mut App) -> anyhow::Result<Self> {
        task::validate_experimental_feature_requirements(requirements)?;
        for feature in requirements {
            anyhow::ensure!(
                cx.has_flag_by_name(feature),
                "This operation requires local opt-in to {feature}"
            );
        }
        let requirements = requirements.to_vec();
        let revoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let subscription = if requirements.is_empty() {
            None
        } else {
            Some(cx.observe_global::<feature_flags::FeatureFlagStore>({
                let requirements = requirements.clone();
                let revoked = revoked.clone();
                move |cx| {
                    if requirements
                        .iter()
                        .any(|feature| !cx.has_flag_by_name(feature))
                    {
                        revoked.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }))
        };
        Ok(Self {
            requirements,
            revoked,
            _subscription: subscription,
        })
    }

    pub fn check(&self, cx: &App) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.revoked.load(std::sync::atomic::Ordering::SeqCst),
            "Pending Task experimental opt-in was revoked; schedule a fresh Task"
        );
        for feature in &self.requirements {
            anyhow::ensure!(
                cx.has_flag_by_name(feature),
                "This operation requires local opt-in to {feature}"
            );
        }
        Ok(())
    }

    pub fn check_requirements(&self, requirements: &[String], cx: &App) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.requirements == requirements,
            "Pending operation experimental requirements changed during preparation"
        );
        self.check(cx)
    }
}

pub struct PendingExperimentalTaskLaunch {
    task_id: task::TaskId,
    operation: PendingExperimentalFeatureOperation,
}

impl PendingExperimentalTaskLaunch {
    pub fn new(launch: &task::SpawnInTerminal, cx: &mut App) -> anyhow::Result<Option<Self>> {
        validate_experimental_task_requirements(launch, cx)?;
        if launch.required_experimental_features.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            task_id: launch.id.clone(),
            operation: PendingExperimentalFeatureOperation::new(
                &launch.required_experimental_features,
                cx,
            )?,
        }))
    }

    pub fn check(&self, launch: &task::SpawnInTerminal, cx: &App) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.task_id == launch.id,
            "Pending Task identity changed during preparation"
        );
        self.operation
            .check_requirements(&launch.required_experimental_features, cx)
    }
}

pub fn validate_experimental_task_requirements(
    launch: &task::SpawnInTerminal,
    cx: &gpui::App,
) -> anyhow::Result<()> {
    task::validate_experimental_feature_requirements(&launch.required_experimental_features)?;
    for feature in &launch.required_experimental_features {
        anyhow::ensure!(
            cx.has_flag_by_name(feature),
            "This Task requires local opt-in to {feature}"
        );
    }
    Ok(())
}

#[cfg(feature = "rust-coverage")]
fn validate_coverage_task_declaration(
    run_id: &str,
    launch: &task::SpawnInTerminal,
    group: &task::TaskArtifactGroup,
) -> anyhow::Result<task::TaskArtifact> {
    anyhow::ensure!(
        run_id.len() == 36
            && run_id.bytes().enumerate().all(|(index, byte)| {
                if matches!(index, 8 | 13 | 18 | 23) {
                    byte == b'-'
                } else {
                    byte.is_ascii_hexdigit()
                }
            })
            && run_id.bytes().any(|byte| byte != b'0' && byte != b'-'),
        "Coverage Task run identity is invalid"
    );
    anyhow::ensure!(
        group.files.len() == 1,
        "Coverage capture requires exactly one report role"
    );
    group.with_resolved_files(group.files.clone())?;
    let artifact = group
        .files
        .get("report")
        .context("Coverage capture requires the report role")?
        .clone();
    let invocation_suffix = format!("-{}", run_id.replace('-', "").to_ascii_lowercase());
    anyhow::ensure!(
        artifact.kind == task::TaskArtifactKind::Data
            && Path::new(&artifact.path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.ends_with(&invocation_suffix))
            && launch
                .args
                .iter()
                .filter(|argument| *argument == &artifact.path)
                .count()
                == 1,
        "Coverage report must be uniquely bound to this Task invocation"
    );
    Ok(artifact)
}

#[cfg(feature = "rust-coverage")]
fn begin_coverage_task_capture<Owner: EventEmitter<crate::Event> + 'static>(
    provider: &Entity<crate::rust_coverage_provider::RustCoverageProviderStore>,
    buffer_store: &Entity<BufferStore>,
    worktree_store: &Entity<WorktreeStore>,
    run_id: String,
    launch: &task::SpawnInTerminal,
    artifact: task::TaskArtifact,
    working_directory: Option<PathBuf>,
    cx: &mut Context<Owner>,
) -> anyhow::Result<crate::rust_coverage_provider::RustCoverageTaskCapture> {
    use crate::source_coverage::SourceCoverageStatus;
    use std::sync::atomic::{AtomicBool, Ordering};

    crate::rust_coverage_provider::validate_capture_task_environment(&launch.env)?;
    let token = provider.update(cx, |provider, cx| provider.begin_task_capture_run(cx))?;
    let token = std::rc::Rc::new(std::cell::Cell::new(token));
    let admitted_output = std::rc::Rc::new(std::cell::RefCell::new(None));
    let cancellation = Arc::new(AtomicBool::new(false));
    let (stop, stopped) = async_channel::bounded::<()>(1);
    let provider_cleanup = provider.downgrade();
    let token_cleanup = token.clone();
    let cancellation_cleanup = cancellation.clone();
    let admitted_output_cleanup = admitted_output.clone();
    cx.spawn(async move |_, cx| match stopped.recv().await {
        Err(_) if cancellation_cleanup.load(Ordering::Relaxed) => {
            if let Err(error) = provider_cleanup.update(cx, |provider, cx| {
                provider.finish_task_capture_status_with_output(
                    token_cleanup.get(),
                    SourceCoverageStatus::Cancelled,
                    "Coverage Task capture was cancelled",
                    admitted_output_cleanup.borrow().as_ref(),
                    cx,
                );
            }) {
                log::debug!("Coverage capture owner closed during cancellation: {error:#}");
            }
        }
        Ok(()) | Err(_) => {}
    })
    .detach();
    let mut subscriptions = vec![cx.subscribe(&cx.entity(), {
        let provider = provider.clone();
        let token = token.clone();
        let cancellation = cancellation.clone();
        let stop = stop.clone();
        move |_, _, event, cx| {
            let status = match event {
                crate::Event::Closed
                | crate::Event::DisconnectedFromHost
                | crate::Event::DisconnectedFromRemote { .. } => {
                    Some(SourceCoverageStatus::Disconnected)
                }
                crate::Event::WorktreeRemoved(_)
                | crate::Event::WorktreeOrderChanged
                | crate::Event::WorktreePathsChanged { .. }
                | crate::Event::RemoteIdChanged(_)
                | crate::Event::Rejoined
                | crate::Event::Reshared
                | crate::Event::HostReshared
                | crate::Event::UpstreamSessionChanged => Some(SourceCoverageStatus::Stale),
                _ => None,
            };
            if let Some(status) = status {
                cancellation.store(true, Ordering::Relaxed);
                provider.update(cx, |provider, cx| {
                    provider.finish_task_capture_status(
                        token.get(),
                        status,
                        crate::rust_coverage_provider::COVERAGE_TASK_EXPIRED_DIAGNOSTIC,
                        cx,
                    )
                });
                stop.close();
            }
        }
    })];
    subscriptions.push(cx.observe_global::<feature_flags::FeatureFlagStore>({
        let provider = provider.clone();
        let token = token.clone();
        let cancellation = cancellation.clone();
        let stop = stop.clone();
        move |_, cx| {
            if !cx.has_flag::<RustCoverageExperimentalFeatureFlag>() {
                cancellation.store(true, Ordering::Relaxed);
                stop.close();
                provider.update(cx, |provider, cx| {
                    provider.finish_task_capture_status(
                        token.get(),
                        SourceCoverageStatus::Stale,
                        crate::rust_coverage_provider::COVERAGE_TASK_EXPIRED_DIAGNOSTIC,
                        cx,
                    )
                });
            }
        }
    }));
    subscriptions.push(provider.update(cx, |provider, cx| {
        provider.subscribe_capture_buffer_changes(buffer_store, cx)
    }));
    subscriptions.push(cx.on_release({
        let cancellation = cancellation.clone();
        let stop = stop.clone();
        move |_, _| {
            cancellation.store(true, Ordering::Relaxed);
            stop.close();
        }
    }));
    if let Some(trust) = crate::trusted_worktrees::TrustedWorktrees::try_get_global(cx) {
        let worktree_store = worktree_store.downgrade();
        subscriptions.push(cx.subscribe(&trust, {
            let provider = provider.clone();
            let token = token.clone();
            let cancellation = cancellation.clone();
            let stop = stop.clone();
            move |_, _, event, cx| {
                let (changed_store, status) = match event {
                    crate::trusted_worktrees::TrustedWorktreesEvent::Restricted(store, _) => {
                        (store, SourceCoverageStatus::Restricted)
                    }
                    crate::trusted_worktrees::TrustedWorktreesEvent::Trusted(store, _) => {
                        (store, SourceCoverageStatus::Stale)
                    }
                };
                if *changed_store == worktree_store {
                    cancellation.store(true, Ordering::Relaxed);
                    provider.update(cx, |provider, cx| {
                        provider.finish_task_capture_status(
                            token.get(),
                            status,
                            crate::rust_coverage_provider::COVERAGE_TASK_EXPIRED_DIAGNOSTIC,
                            cx,
                        )
                    });
                    stop.close();
                }
            }
        }));
    }
    Ok(crate::rust_coverage_provider::RustCoverageTaskCapture {
        provider: provider.clone(),
        token,
        run_id,
        task_id: launch.id.clone(),
        artifact,
        path: None,
        admitted_output,
        command: launch.command.clone(),
        arguments: launch.args.clone(),
        environment: launch.env.clone(),
        working_directory,
        admitted: false,
        source_baseline: None,
        cancellation,
        stop,
        _subscriptions: subscriptions,
        armed: true,
    })
}

#[cfg(feature = "rust-coverage")]
fn admit_coverage_task_sources(
    capture: &mut crate::rust_coverage_provider::RustCoverageTaskCapture,
    path: crate::ProjectPath,
    working_directory: PathBuf,
    buffer_store: &Entity<BufferStore>,
    cx: &App,
) -> anyhow::Result<()> {
    capture.path = Some(path);
    capture.working_directory = Some(working_directory);
    capture.token.set(
        capture
            .provider
            .read(cx)
            .seal_source_revision(capture.token.get(), cx)?,
    );
    capture.admitted = true;
    capture.source_baseline = Some(capture.provider.read(cx).begin_capture_source_baseline(
        capture,
        buffer_store,
        cx,
    )?);
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "rust-coverage"))]
impl TaskStore {
    pub fn run_host_coverage_prerequisite(
        &self,
        prerequisite: &crate::cargo_workspace_store::HostCoveragePrerequisite,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<Task<anyhow::Result<()>>> {
        use crate::trusted_worktrees::{TrustedWorktrees, TrustedWorktreesEvent};
        use gpui::AppContext as _;
        use settings::Settings as _;
        use terminal::{
            TaskState, TaskStatus, TerminalBuilder, terminal_settings::TerminalSettings,
        };

        let Self::Functional(StoreState {
            mode: StoreMode::Local { .. },
            task_inventory,
            worktree_store,
            ..
        }) = self
        else {
            anyhow::bail!("Coverage before-launch Task requires the producing Task store");
        };
        anyhow::ensure!(
            !TrustedWorktrees::has_restricted_worktrees(worktree_store, cx),
            "Coverage before-launch Task requires a trusted project"
        );
        let worktree_id = prerequisite.package.worktree_id;
        let package_scope = prerequisite
            .package
            .path
            .parent()
            .context("Coverage package has no directory")?;
        let tasks = task_inventory
            .read(cx)
            .saved_task_templates(worktree_id)
            .filter(|(source, _)| match source {
                TaskSourceKind::Worktree {
                    directory_in_worktree,
                    ..
                } => directory_in_worktree
                    .parent()
                    .is_some_and(|scope| package_scope.starts_with(scope)),
                TaskSourceKind::AbsPath { .. } => true,
                _ => false,
            });
        let (_, resolved) =
            resolve_task_reference(tasks, &prerequisite.reference, &prerequisite.context)?;
        let resolved = resolved.resolved;
        validate_experimental_task_requirements(&resolved, cx)?;
        let worktrees = worktree_store.read(cx);
        let directory = resolved
            .cwd
            .clone()
            .context("Coverage before-launch Task has no working directory")?;
        let path = worktrees
            .project_path_for_absolute_path(&directory, cx)
            .context("Coverage before-launch Task directory is outside the project")?;
        let worktree = worktrees
            .worktree_for_id(path.worktree_id, cx)
            .filter(|worktree| worktree.read(cx).is_visible() && worktree.read(cx).is_local())
            .context("Coverage before-launch Task requires a visible local worktree")?;
        let root = worktree.read(cx).abs_path().to_path_buf();
        anyhow::ensure!(
            directory.starts_with(&root),
            "Coverage before-launch Task is outside its worktree"
        );
        let settings = TerminalSettings::get(
            Some(SettingsLocation {
                worktree_id: path.worktree_id,
                path: &path.path,
            }),
            cx,
        )
        .clone();
        let (invalidated, invalidation) = async_channel::bounded::<()>(1);
        let trust = TrustedWorktrees::try_get_global(cx).map(|trust| {
            let worktree_store = worktree_store.downgrade();
            let invalidated = invalidated.clone();
            cx.subscribe(&trust, move |_, _, event, _| {
                if matches!(event, TrustedWorktreesEvent::Restricted(store, _) if *store == worktree_store) {
                    invalidated.close();
                }
            })
        });
        let feature = cx.observe_global::<feature_flags::FeatureFlagStore>(move |_, cx| {
            if !cx.has_flag::<RustCoverageExperimentalFeatureFlag>()
                || !cx.has_flag::<feature_flags::RustNamedCargoPresetsExperimentalFeatureFlag>()
            {
                invalidated.close();
            }
        });
        let worktree_store = worktree_store.clone();
        let (observer_released, retained_observer) = async_channel::bounded::<()>(1);
        let (report_outcome, outcome) = futures::channel::oneshot::channel();
        cx.spawn(async move |_, cx| {
            let (_trust, _feature) = (trust, feature);
            let result = async {
                anyhow::ensure!(
                    smol::fs::canonicalize(&directory).await? == directory,
                    "Coverage before-launch Task directory changed before launch"
                );
                let builder = cx.update(|cx| {
                    anyhow::ensure!(
                        !invalidation.is_closed()
                            && !observer_released.is_closed()
                            && worktree_store
                                .read(cx)
                                .worktree_for_id(path.worktree_id, cx)
                                .is_some()
                            && !TrustedWorktrees::has_restricted_worktrees(&worktree_store, cx),
                        "Coverage before-launch Task authority changed"
                    );
                    validate_experimental_task_requirements(&resolved, cx)?;
                    let (program, args) = task::ShellBuilder::new(&resolved.shell, false)
                        .non_interactive()
                        .build_with_command_intent(
                            resolved.command.clone(),
                            &resolved.args,
                            resolved.literal_command,
                            resolved.literal_prefix_args,
                        );
                    let mut env = settings.env;
                    env.extend(resolved.env.clone());
                    let (completed, completion) = async_channel::bounded(1);
                    Ok::<_, anyhow::Error>(TerminalBuilder::new(
                        Some(directory),
                        Some(TaskState {
                            spawned_task: resolved,
                            status: TaskStatus::Running,
                            completion_rx: completion,
                            structured_task: None,
                        }),
                        task::Shell::WithArguments {
                            program,
                            args,
                            title_override: None,
                        },
                        env,
                        settings.cursor_shape,
                        settings.alternate_scroll,
                        settings.max_scroll_history_lines,
                        settings.path_hyperlink_regexes,
                        settings.path_hyperlink_timeout_ms,
                        false,
                        0,
                        Some(completed),
                        cx,
                        Vec::new(),
                        util::paths::PathStyle::Unix,
                    ))
                })?;
                let builder = builder.await?;
                let terminal = cx.new(|cx| builder.subscribe(cx));
                let completion =
                    terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                let cancelled = smol::future::race(observer_released.closed(), async {
                    if invalidation.recv().await.is_ok() {
                        log::error!(
                            "Coverage Task authority channel received an unexpected message"
                        );
                    }
                });
                match futures::future::select(completion, Box::pin(cancelled)).await {
                    futures::future::Either::Left((status, _)) => {
                        anyhow::ensure!(
                            !invalidation.is_closed() && !observer_released.is_closed(),
                            "Coverage before-launch Task authority changed"
                        );
                        let status = status
                            .context("Coverage before-launch Task did not report an exit status")?;
                        anyhow::ensure!(
                            status.success(),
                            "Coverage before-launch Task exited with {status}"
                        );
                        Ok(())
                    }
                    futures::future::Either::Right(((), completion)) => {
                        terminal.update(cx, |terminal, _| terminal.kill_active_task());
                        let deadline = cx
                            .background_executor()
                            .timer(std::time::Duration::from_secs(10));
                        anyhow::ensure!(
                            matches!(
                                futures::future::select(completion, deadline).await,
                                futures::future::Either::Left((Some(_), _))
                            ),
                            "Terminal did not observe Coverage before-launch Task cancellation"
                        );
                        anyhow::bail!("Coverage before-launch Task was cancelled")
                    }
                }
            }
            .await;
            if report_outcome.send(result).is_err() {
                log::debug!("Coverage caller released the before-launch Task");
            }
        })
        .detach();
        Ok(cx.spawn(async move |_, _| {
            let _retained_observer = retained_observer;
            outcome
                .await
                .context("Coverage before-launch Task lost its host owner")?
        }))
    }

    /// The launch comes from the producing host's Task resolver. A peer request cannot supply it.
    pub fn run_host_coverage_task(
        &self,
        provider: Entity<crate::rust_coverage_provider::RustCoverageProviderStore>,
        run_id: String,
        launch: task::SpawnInTerminal,
        group: &task::TaskArtifactGroup,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<Task<anyhow::Result<crate::source_coverage::SourceCoverageSnapshot>>> {
        use crate::source_coverage::{SourceCoverageSourceChanged, SourceCoverageStatus};
        use gpui::AppContext as _;
        use terminal::{CoverageTaskReportReservation, TaskBuildInvocation, TerminalBuilder};

        let TaskStore::Functional(StoreState {
            mode: StoreMode::Local { .. },
            buffer_store,
            worktree_store,
            ..
        }) = self
        else {
            anyhow::bail!("Coverage Task requires a local producing Task store");
        };
        anyhow::ensure!(
            group.format == "llvm-cov-json-v1"
                && provider.read(cx).owns_host_task_worktrees(worktree_store)
                && !crate::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    worktree_store,
                    cx
                ),
            "Coverage Task requires the trusted producing project"
        );
        validate_experimental_task_requirements(&launch, cx)?;
        let artifact = validate_coverage_task_declaration(&run_id, &launch, group)?;
        let working_directory = launch
            .cwd
            .as_ref()
            .filter(|path| path.is_absolute())
            .context("Coverage Task requires an absolute host working directory")?;
        let worktrees = worktree_store.read(cx);
        let report_path = worktrees
            .project_path_for_absolute_path(&working_directory.join(&artifact.path), cx)
            .context("Coverage Task report binding is outside the host project")?;
        let root = worktrees
            .worktree_for_id(report_path.worktree_id, cx)
            .filter(|tree| tree.read(cx).is_visible() && tree.read(cx).is_local())
            .context("Coverage Task requires a visible local worktree")?;
        anyhow::ensure!(
            working_directory.starts_with(root.read(cx).abs_path()),
            "Coverage Task working directory is outside its worktree"
        );
        let filesystem = worktrees
            .fs()
            .context("Coverage filesystem is unavailable")?;
        let buffer_store = buffer_store
            .upgrade()
            .context("Coverage Task buffers are unavailable")?;
        let report = CoverageTaskReportReservation::new(usize::try_from(artifact.max_bytes)?)?;
        let mut arguments = launch.args.clone();
        let report_argument = launch
            .args
            .windows(2)
            .position(|pair| pair == ["--output-path", artifact.path.as_str()])
            .and_then(|index| arguments.get_mut(index + 1))
            .context("Coverage Task report must be bound to the collector output argument")?;
        *report_argument = report
            .path()
            .to_str()
            .context("Coverage Task reserved path is not Unicode")?
            .to_owned();
        let invocation = TaskBuildInvocation::new(
            launch
                .command
                .as_ref()
                .context("Coverage Task has no host executable")?
                .into(),
            arguments,
            working_directory.clone(),
            launch.env.clone().into_iter().collect(),
        )?;
        let mut capture = begin_coverage_task_capture(
            &provider,
            &buffer_store,
            worktree_store,
            run_id,
            &launch,
            artifact,
            Some(working_directory.clone()),
            cx,
        )?;
        capture._subscriptions.push(cx.subscribe(worktree_store, {
            use crate::worktree_store::WorktreeStoreEvent;
            use std::sync::atomic::Ordering;

            let provider = provider.clone();
            let token = capture.token.clone();
            let cancellation = capture.cancellation.clone();
            let stop = capture.stop.clone();
            move |_, _, event, cx| {
                if matches!(
                    event,
                    WorktreeStoreEvent::WorktreeAdded(_)
                        | WorktreeStoreEvent::WorktreeRemoved(..)
                        | WorktreeStoreEvent::WorktreeReleased(..)
                        | WorktreeStoreEvent::WorktreeOrderChanged
                        | WorktreeStoreEvent::WorktreePathPrivacyChanged(_)
                ) {
                    cancellation.store(true, Ordering::Relaxed);
                    provider.update(cx, |store, cx| {
                        store.finish_task_capture_status(
                            token.get(),
                            SourceCoverageStatus::Stale,
                            "Coverage Task worktrees changed before publication",
                            cx,
                        );
                    });
                    stop.close();
                }
            }
        }));
        admit_coverage_task_sources(
            &mut capture,
            report_path,
            working_directory.clone(),
            &buffer_store,
            cx,
        )?;
        let working_directory = working_directory.clone();
        let prepared = provider.update(cx, |store, cx| {
            store.prepare_capture_sources(capture, true, cx)
        });
        let (observer_released, retained_observer) = async_channel::bounded::<()>(1);
        let worker = cx.spawn(async move |_, cx| {
            let mut capture = prepared.await?;
            let cancelled =
                || smol::future::race(capture.stop.closed(), observer_released.closed());
            let run = async {
                anyhow::ensure!(
                    filesystem.canonicalize(&working_directory).await? == working_directory,
                    "Coverage Task working directory changed before launch"
                );
                let (builder, observation) = cx.update(|cx| {
                    anyhow::ensure!(
                        !observer_released.is_closed(),
                        "Coverage Task capture was cancelled"
                    );
                    provider
                        .read(cx)
                        .revalidate_task_capture_before_launch(&capture, cx)?;
                    TerminalBuilder::new_with_coverage_task(launch, None, invocation, report, 0, cx)
                })?;
                let builder = builder.await?;
                let terminal = cx.new(|cx| builder.subscribe(cx));
                let result =
                    futures::future::select(Box::pin(observation.finish()), Box::pin(cancelled()))
                        .await;
                let (status, report) = match result {
                    futures::future::Either::Left((result, _)) => result?,
                    futures::future::Either::Right(((), observation)) => {
                        // Closing the connection lets the helper reap its child before Terminal
                        // releases the helper process.
                        drop(observation);
                        let exit = terminal
                            .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                        let deadline = cx
                            .background_executor()
                            .timer(std::time::Duration::from_secs(10));
                        anyhow::ensure!(
                            matches!(
                                futures::future::select(exit, deadline).await,
                                futures::future::Either::Left((Some(_), _))
                            ),
                            "Terminal did not observe Coverage Task cancellation"
                        );
                        anyhow::bail!("Coverage Task capture was cancelled")
                    }
                };
                anyhow::ensure!(status.success(), "Coverage Task exited with {status}");
                let report = report.context("Coverage Task did not return a sealed report")?;
                let publication = provider.update(cx, |store, cx| {
                    store.publish_host_task_report(&capture, report, observer_released.clone(), cx)
                })?;
                match futures::future::select(publication, Box::pin(cancelled())).await {
                    futures::future::Either::Left((result, _)) => result,
                    futures::future::Either::Right(_) => {
                        anyhow::bail!("Coverage Task capture was cancelled")
                    }
                }
            }
            .await;
            if let Err(error) = &run {
                provider.update(cx, |store, cx| {
                    if observer_released.is_closed() {
                        store.finish_cancelled_run(capture.token.get(), cx);
                    } else if !store.finish_expired_task_capture(&capture, cx) {
                        let stale = error.is::<SourceCoverageSourceChanged>()
                            || !store.is_task_capture_source_revision_current(&capture, cx);
                        store.finish_task_capture_status(
                            capture.token.get(),
                            if stale {
                                SourceCoverageStatus::Stale
                            } else {
                                SourceCoverageStatus::Error
                            },
                            if stale {
                                "Coverage Task source changed before publication"
                            } else {
                                "Coverage Task failed before publishing an authenticated report"
                            },
                            cx,
                        );
                    }
                });
            }
            capture.armed = false;
            run
        });
        let (finish, outcome) = futures::channel::oneshot::channel();
        cx.spawn(async move |_, _| {
            if finish.send(worker.await).is_err() {
                log::debug!("Coverage Task observer released the result");
            }
        })
        .detach();
        Ok(cx.spawn(async move |_, _| {
            let _retained_observer = retained_observer;
            outcome.await.context("Coverage Task lost its host owner")?
        }))
    }
}

impl crate::Project {
    pub fn begin_task_artifact_capture(
        &self,
        run_id: String,
        launch: &task::SpawnInTerminal,
        group: &task::TaskArtifactGroup,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<Option<ProjectTaskArtifactCapture>> {
        validate_experimental_task_requirements(launch, cx)?;
        if group.format != "llvm-cov-json-v1" {
            return Ok(None);
        }
        #[cfg(not(feature = "rust-coverage"))]
        {
            let _ = (run_id, launch, cx);
            anyhow::bail!("This project host does not support coverage Task capture");
        }
        #[cfg(feature = "rust-coverage")]
        {
            anyhow::ensure!(
                cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
                "Rust coverage requires local experimental opt-in"
            );
            anyhow::ensure!(
                self.is_local(),
                "Coverage Task capture requires authoritative local Terminal completion; remote host capture is unavailable"
            );
            let artifact = validate_coverage_task_declaration(&run_id, launch, group)?;
            anyhow::ensure!(
                !crate::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &self.worktree_store(),
                    cx
                ),
                "Coverage capture requires trusted worktrees"
            );
            let working_directory = launch
                .cwd
                .as_deref()
                .map(|directory| {
                    self.terminal_task_working_directory(Some(directory), cx)
                        .map(|directory| directory.to_path_buf())
                        .context("Coverage Task working directory is unavailable")
                })
                .transpose()?;
            let capture = begin_coverage_task_capture(
                self.rust_coverage_provider_store(),
                self.buffer_store(),
                &self.worktree_store(),
                run_id,
                launch,
                artifact,
                working_directory,
                cx,
            )?;
            Ok(Some(ProjectTaskArtifactCapture {
                inner: ProjectTaskArtifactCaptureKind::RustCoverage(capture),
            }))
        }
    }

    pub fn admit_task_artifact_capture(
        &self,
        capture: &mut ProjectTaskArtifactCapture,
        launch: &task::SpawnInTerminal,
        cx: &App,
    ) -> anyhow::Result<()> {
        #[cfg(not(feature = "rust-coverage"))]
        {
            let _ = (&capture.inner, launch, cx);
            anyhow::bail!("Task artifact capture is unsupported");
        }
        #[cfg(feature = "rust-coverage")]
        match &mut capture.inner {
            #[cfg(feature = "rust-coverage")]
            ProjectTaskArtifactCaptureKind::RustCoverage(capture) => {
                anyhow::ensure!(
                    self.is_local()
                        && capture.is_current(cx)
                        && capture.provider.entity_id()
                            == self.rust_coverage_provider_store().entity_id(),
                    "Coverage Task capture owner became stale"
                );
                anyhow::ensure!(
                    !capture.admitted
                        && capture.task_id == launch.id
                        && capture.command == launch.command
                        && capture.arguments == launch.args
                        && capture.environment == launch.env
                        && capture
                            .working_directory
                            .as_ref()
                            .is_none_or(|directory| launch.cwd.as_ref() == Some(directory)),
                    "Coverage Task invocation changed before admission"
                );
                let working_directory = launch
                    .cwd
                    .clone()
                    .filter(|directory| directory.is_absolute())
                    .context(
                        "Coverage capture requires the frozen absolute Task working directory",
                    )?;
                let path = self
                    .find_project_path(working_directory.join(&capture.artifact.path), cx)
                    .context("Coverage Task report is outside the visible project")?;
                admit_coverage_task_sources(
                    capture,
                    path.clone(),
                    working_directory,
                    self.buffer_store(),
                    cx,
                )?;
                *capture.admitted_output.borrow_mut() = Some(path);
                Ok(())
            }
        }
    }

    pub fn prepare_task_artifact_capture_sources(
        &self,
        capture: ProjectTaskArtifactCapture,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<Task<anyhow::Result<ProjectTaskArtifactCapture>>> {
        #[cfg(not(feature = "rust-coverage"))]
        {
            let _ = (&capture.inner, cx);
            anyhow::bail!("Task artifact capture is unsupported");
        }
        #[cfg(feature = "rust-coverage")]
        match capture.inner {
            ProjectTaskArtifactCaptureKind::RustCoverage(capture) => {
                anyhow::ensure!(
                    self.is_local()
                        && capture.is_current(cx)
                        && capture.provider.entity_id()
                            == self.rust_coverage_provider_store().entity_id(),
                    "Coverage Task source preparation owner became stale"
                );
                let provider = capture.provider.clone();
                let prepared = provider.update(cx, |store, cx| {
                    store.prepare_capture_sources(capture, true, cx)
                });
                Ok(cx.spawn(async move |_, _| {
                    Ok(ProjectTaskArtifactCapture {
                        inner: ProjectTaskArtifactCaptureKind::RustCoverage(prepared.await?),
                    })
                }))
            }
        }
    }

    pub fn revalidate_task_artifact_capture(
        &self,
        capture: &ProjectTaskArtifactCapture,
        logical_launch: &task::SpawnInTerminal,
        cx: &App,
    ) -> anyhow::Result<()> {
        #[cfg(not(feature = "rust-coverage"))]
        {
            let _ = (&capture.inner, logical_launch, cx);
            anyhow::bail!("Task artifact capture is unsupported");
        }
        #[cfg(feature = "rust-coverage")]
        match &capture.inner {
            ProjectTaskArtifactCaptureKind::RustCoverage(capture) => {
                anyhow::ensure!(
                    self.is_local()
                        && capture.admitted
                        && capture.is_current(cx)
                        && capture.provider.entity_id()
                            == self.rust_coverage_provider_store().entity_id()
                        && !crate::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                            &self.worktree_store(),
                            cx
                        ),
                    "Coverage Task capture owner became stale before launch"
                );
                anyhow::ensure!(
                    capture.task_id == logical_launch.id
                        && capture.command == logical_launch.command
                        && capture.arguments == logical_launch.args
                        && capture.environment == logical_launch.env
                        && capture.working_directory == logical_launch.cwd,
                    "Coverage Task invocation changed before launch"
                );
                let working_directory = capture
                    .working_directory
                    .as_ref()
                    .filter(|directory| directory.is_absolute())
                    .context("Coverage capture needs its frozen working directory")?;
                anyhow::ensure!(
                    self.find_project_path(working_directory.join(&capture.artifact.path), cx)
                        .as_ref()
                        == capture.path.as_ref(),
                    "Coverage Task output root changed before launch"
                );
                capture
                    .provider
                    .read(cx)
                    .revalidate_task_capture_before_launch(capture, cx)
            }
        }
    }

    pub fn observe_task_artifact_capture(
        &self,
        capture: ProjectTaskArtifactCapture,
        spawned: Task<Option<anyhow::Result<std::process::ExitStatus>>>,
        cx: &mut Context<Self>,
    ) -> Task<Option<anyhow::Result<std::process::ExitStatus>>> {
        #[cfg(not(feature = "rust-coverage"))]
        {
            let _ = (&capture.inner, cx);
            spawned
        }
        #[cfg(feature = "rust-coverage")]
        cx.spawn(async move |_, cx| {
            let outcome = spawned.await;
            match capture.inner {
                #[cfg(feature = "rust-coverage")]
                ProjectTaskArtifactCaptureKind::RustCoverage(mut capture) => {
                    let provider = capture.provider.clone();
                    let token = capture.token.get();
                    if provider.update(cx, |provider, cx| {
                        provider.finish_expired_task_capture(&capture, cx)
                    }) {
                        capture.armed = false;
                    } else if outcome
                        .as_ref()
                        .is_some_and(|result| result.as_ref().is_ok_and(|status| status.success()))
                    {
                        let interpretation = provider.update(cx, |provider, cx| {
                            provider.complete_local_task_capture(capture, cx)
                        });
                        if let Err(error) = interpretation.await {
                            log::warn!("Coverage Task capture could not publish: {error:#}");
                        }
                    } else {
                        provider.update(cx, |provider, cx| {
                            let source_current =
                                provider.is_task_capture_source_revision_current(&capture, cx);
                            // Without a successful report, mapped paths are unknown; an expired
                            // admitted source epoch must not publish a current failure notice.
                            let status = if !source_current {
                                crate::source_coverage::SourceCoverageStatus::Stale
                            } else if outcome.is_none() {
                                crate::source_coverage::SourceCoverageStatus::Cancelled
                            } else {
                                crate::source_coverage::SourceCoverageStatus::Error
                            };
                            provider.finish_task_capture_status_with_output(
                                token,
                                status,
                                if source_current {
                                    "Coverage Task did not complete successfully"
                                } else {
                                    "Coverage Task source changed before unsuccessful completion"
                                },
                                capture.admitted_output.borrow().as_ref(),
                                cx,
                            )
                        });
                        capture.armed = false;
                    }
                }
            }
            outcome
        })
    }
}

#[cfg(test)]
mod experimental_tests {
    #[gpui::test]
    fn experimental_fixture_bootstraps_settings_notifications_after_client_flags(
        cx: &mut gpui::TestAppContext,
    ) {
        use feature_flags::FeatureFlagAppExt as _;
        use gpui::UpdateGlobal as _;

        let feature = "rust-native-profiling-experimental";
        cx.update(|cx| {
            cx.update_flags(false, vec![feature.to_string()]);
            super::set_experimental_feature_for_test(feature, true, cx);
        });
        cx.executor().run_until_parked();
        let requirements = vec![feature.to_string()];
        let original = cx.update(|cx| {
            super::PendingExperimentalFeatureOperation::new(&requirements, cx)
                .expect("original locally opted-in operation")
        });
        cx.executor().run_until_parked();
        for enabled in [false, true] {
            cx.update(|cx| {
                settings::SettingsStore::update_global(cx, |store, cx| {
                    store.update_user_settings(cx, |content| {
                        content.feature_flags.get_or_insert_default().insert(
                            feature.to_string(),
                            if enabled { "on" } else { "off" }.to_string(),
                        );
                    });
                });
            });
            cx.executor().run_until_parked();
        }
        cx.update(|cx| {
            assert!(cx.has_flag_by_name(feature));
            assert!(original.check(cx).is_err());
            let fresh = super::PendingExperimentalFeatureOperation::new(&requirements, cx)
                .expect("fresh operation after local recovery");
            assert!(fresh.check(cx).is_ok());
            let ordinary = super::PendingExperimentalFeatureOperation::new(&[], cx)
                .expect("ordinary operation has no experimental requirements");
            assert!(ordinary.check(cx).is_ok());
        });
    }

    #[gpui::test]
    fn experimental_task_admission_requires_local_settings_and_rechecks_revocation(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let mut launch = task::SpawnInTerminal::default();
            assert!(super::validate_experimental_task_requirements(&launch, cx).is_ok());
            launch.required_experimental_features =
                vec!["rust-named-cargo-presets-experimental".into()];
            assert!(super::validate_experimental_task_requirements(&launch, cx).is_err());
            super::set_experimental_feature_for_test(
                "rust-named-cargo-presets-experimental",
                true,
                cx,
            );
            assert!(super::validate_experimental_task_requirements(&launch, cx).is_ok());
            super::set_experimental_feature_for_test(
                "rust-named-cargo-presets-experimental",
                false,
                cx,
            );
            assert!(super::validate_experimental_task_requirements(&launch, cx).is_err());
            launch.required_experimental_features.clear();
            assert!(
                super::validate_experimental_task_requirements(&launch, cx).is_ok(),
                "ordinary authored Tasks remain available"
            );
        });
    }
}
