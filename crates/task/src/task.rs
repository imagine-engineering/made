//! Baseline interface of Tasks in Zed: all tasks in Zed are intended to use those for implementing their own logic.

mod adapter_schema;
mod debug_format;
mod serde_helpers;
pub mod static_source;
mod task_template;
mod vscode_debug_format;
mod vscode_format;

use anyhow::Context as _;
use collections::{BTreeMap, HashMap, HashSet, hash_map};
use gpui::{App, SharedString, Window};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::str::FromStr;
use std::sync::Arc;

pub const EXPERIMENTAL_TASK_POLICY_VERSION: u32 = 1;

pub use adapter_schema::{AdapterSchema, AdapterSchemas};
pub use debug_format::{
    AttachRequest, BuildTaskDefinition, DebugRequest, DebugScenario, DebugTaskFile, LaunchRequest,
    Request, TcpArgumentsTemplate, ZedDebugConfig,
};
pub use task_template::{
    DebugArgsRequest, HideStrategy, RevealStrategy, SaveStrategy, TaskHook, TaskTemplate,
    TaskTemplates, substitute_variables_in_map, substitute_variables_in_str,
};
pub use util::shell::{Shell, ShellKind};
pub use util::shell_builder::ShellBuilder;
pub use vscode_debug_format::VsCodeDebugTaskFile;
pub use vscode_format::VsCodeTaskFile;
pub use zed_actions::RevealTarget;

pub const MIN_TASK_STDOUT_REPORT_BYTES: usize = 1024 * 1024;
pub const MAX_TASK_STDOUT_REPORT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskStdoutReportFormat {
    CargoBuildJson,
    CargoRuntimePreparationJson,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskStdoutReport {
    pub format: TaskStdoutReportFormat,
    pub max_bytes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_profile: Option<TaskNativeProfileContinuation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskNativeProfileContinuation {
    pub collector_path: String,
    pub profile_path: String,
    pub workload_arguments: Vec<String>,
}

impl TaskNativeProfileContinuation {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.collector_path.len() <= 4096
                && std::path::Path::new(&self.collector_path).is_absolute()
                && !self.collector_path.chars().any(char::is_control)
                && !self.collector_path.contains(['\\', ':'])
                && !self.collector_path.contains("$ZED_")
                && !self.collector_path.contains("${ZED_")
                && !std::path::Path::new(&self.collector_path)
                    .components()
                    .any(|component| {
                        matches!(
                            component,
                            std::path::Component::ParentDir | std::path::Component::CurDir
                        )
                    }),
            "Native profile collector must be a bounded resolved absolute path"
        );
        anyhow::ensure!(
            self.profile_path.len() <= MAX_TASK_ARTIFACT_PATH_BYTES
                && !self.profile_path.contains("$ZED_")
                && !self.profile_path.contains("${ZED_")
                && !self.profile_path.ends_with(".syms.json")
                && !self.profile_path.starts_with('-'),
            "Native profile output must be a resolved primary profile base"
        );
        let stem = self
            .profile_path
            .strip_suffix(".json")
            .filter(|stem| !stem.is_empty())
            .context("Native profile output must use the .json extension")?;
        anyhow::ensure!(
            self.workload_arguments.len() <= 128,
            "Native profile workload has too many literal arguments"
        );
        let mut total_bytes = 0usize;
        for argument in &self.workload_arguments {
            anyhow::ensure!(
                argument.len() <= 4096 && !argument.contains('\0'),
                "Native profile workload argument exceeds its supported bounds"
            );
            total_bytes = total_bytes
                .checked_add(argument.len())
                .context("Native profile workload argument limit overflow")?;
            anyhow::ensure!(
                total_bytes <= 32 * 1024,
                "Native profile workload arguments exceed their total byte limit"
            );
        }
        let group = TaskArtifactGroup {
            format: "samply-0.13.1".into(),
            files: BTreeMap::from_iter([
                (
                    "profile".into(),
                    TaskArtifact {
                        path: self.profile_path.clone(),
                        kind: TaskArtifactKind::Data,
                        max_bytes: MAX_TASK_ARTIFACT_BYTES,
                    },
                ),
                (
                    "symbols".into(),
                    TaskArtifact {
                        path: format!("{stem}.syms.json"),
                        kind: TaskArtifactKind::Data,
                        max_bytes: 16 * 1024 * 1024,
                    },
                ),
            ]),
        };
        group.with_resolved_files(group.files.clone())?;
        Ok(())
    }
}

impl TaskStdoutReport {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (MIN_TASK_STDOUT_REPORT_BYTES..=MAX_TASK_STDOUT_REPORT_BYTES).contains(&self.max_bytes),
            "Task stdout report limit must be between 1 and 4 MiB"
        );
        anyhow::ensure!(
            self.format != TaskStdoutReportFormat::CargoRuntimePreparationJson
                || self.native_profile.is_some(),
            "Cargo runtime preparation requires a validated native profile continuation"
        );
        if let Some(continuation) = &self.native_profile {
            continuation.validate()?;
        }
        Ok(())
    }
}

pub const MAX_TASK_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_TASK_ARTIFACT_PATH_BYTES: usize = 1024;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskArtifactKind {
    Svg,
    Html,
    /// A machine-readable artifact consumed by a feature after task completion.
    Data,
    #[default]
    File,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskArtifact {
    pub path: String,
    #[serde(default)]
    pub kind: TaskArtifactKind,
    #[serde(default = "default_task_artifact_max_bytes")]
    pub max_bytes: u64,
}

pub const MAX_TASK_ARTIFACT_GROUP_FILES: usize = 8;
pub const MAX_TASK_ARTIFACT_GROUP_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskArtifactGroup {
    pub format: String,
    pub files: BTreeMap<String, TaskArtifact>,
}

impl TaskArtifactGroup {
    pub fn with_resolved_files(
        &self,
        files: BTreeMap<String, TaskArtifact>,
    ) -> anyhow::Result<Self> {
        let valid_identifier = |value: &str, limit: usize| {
            !value.is_empty()
                && value.len() <= limit
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        };
        anyhow::ensure!(
            valid_identifier(&self.format, 128),
            "invalid task artifact group format"
        );
        anyhow::ensure!(
            !files.is_empty() && files.len() <= MAX_TASK_ARTIFACT_GROUP_FILES,
            "task artifact group must declare between 1 and {MAX_TASK_ARTIFACT_GROUP_FILES} files"
        );
        anyhow::ensure!(
            self.files.keys().eq(files.keys()),
            "artifact resolution must preserve declared roles"
        );
        let mut total_bytes = 0u64;
        let mut paths = HashSet::default();
        for (role, artifact) in &files {
            let declared = self
                .files
                .get(role)
                .context("artifact role is not declared")?;
            anyhow::ensure!(
                artifact.kind == declared.kind && artifact.max_bytes == declared.max_bytes,
                "artifact resolution must preserve declared kind and byte limit"
            );
            anyhow::ensure!(
                valid_identifier(role, 64),
                "invalid task artifact group role"
            );
            anyhow::ensure!(
                artifact.kind == TaskArtifactKind::Data,
                "artifact groups accept only data files"
            );
            anyhow::ensure!(
                !artifact.path.contains(['\\', ':'])
                    && !artifact.path.chars().any(char::is_control)
                    && !artifact
                        .path
                        .split('/')
                        .any(|component| component.is_empty() || component == "."),
                "artifact group paths must use portable relative path components"
            );
            artifact.with_resolved_path(artifact.path.clone())?;
            anyhow::ensure!(
                paths.insert(&artifact.path),
                "artifact group files must have distinct paths"
            );
            total_bytes = total_bytes
                .checked_add(artifact.max_bytes)
                .context("task artifact group byte limit overflow")?;
        }
        anyhow::ensure!(
            total_bytes <= MAX_TASK_ARTIFACT_GROUP_BYTES,
            "task artifact group exceeds its combined byte limit"
        );
        Ok(Self {
            format: self.format.clone(),
            files,
        })
    }
}

fn default_task_artifact_max_bytes() -> u64 {
    MAX_TASK_ARTIFACT_BYTES
}

impl TaskArtifact {
    pub fn with_resolved_path(&self, path: String) -> anyhow::Result<Self> {
        use std::path::Component;

        if path.is_empty() || path.len() > MAX_TASK_ARTIFACT_PATH_BYTES {
            anyhow::bail!("task artifact path is empty or exceeds the supported length");
        }
        if path.contains("://")
            || path.starts_with('/')
            || path.starts_with('\\')
            || path.as_bytes().get(1) == Some(&b':')
            || PathBuf::from(&path).components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            anyhow::bail!("task artifact must be a project-relative path");
        }
        if self.max_bytes == 0 || self.max_bytes > MAX_TASK_ARTIFACT_BYTES {
            anyhow::bail!(
                "task artifact byte limit must be between 1 and {MAX_TASK_ARTIFACT_BYTES}"
            );
        }
        let extension = PathBuf::from(&path)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);
        match self.kind {
            TaskArtifactKind::Svg if extension.as_deref() != Some("svg") => {
                anyhow::bail!("SVG task artifacts must use the .svg extension")
            }
            TaskArtifactKind::Html if !matches!(extension.as_deref(), Some("html" | "htm")) => {
                anyhow::bail!("HTML task artifacts must use the .html or .htm extension")
            }
            TaskArtifactKind::Data if extension.as_deref() != Some("json") => {
                anyhow::bail!("data task artifacts must use the .json extension")
            }
            TaskArtifactKind::Svg
            | TaskArtifactKind::Html
            | TaskArtifactKind::Data
            | TaskArtifactKind::File => {}
        }
        Ok(Self {
            path,
            kind: self.kind,
            max_bytes: self.max_bytes,
        })
    }
}

/// Task identifier, unique within the application.
/// Based on it, task reruns and terminal tabs are managed.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
pub struct TaskId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructuredTerminalId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredTaskState {
    Queued,
    Running {
        terminal_id: Option<StructuredTerminalId>,
    },
    Completed {
        terminal_id: Option<StructuredTerminalId>,
        exit_code: Option<i32>,
        success: bool,
    },
    SpawnError {
        message: String,
    },
    Cancelled {
        terminal_id: Option<StructuredTerminalId>,
        termination_confirmed: bool,
    },
}

impl StructuredTaskState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::SpawnError { .. } | Self::Cancelled { .. }
        )
    }

    pub fn terminal_id(&self) -> Option<StructuredTerminalId> {
        match self {
            Self::Running { terminal_id }
            | Self::Completed { terminal_id, .. }
            | Self::Cancelled { terminal_id, .. } => *terminal_id,
            Self::Queued | Self::SpawnError { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredTaskLifecycleEvent {
    pub task_id: TaskId,
    pub state: StructuredTaskState,
}

type StructuredTaskSubscriber = Arc<dyn Fn(&StructuredTaskLifecycleEvent, &mut App) + Send + Sync>;
type StructuredTaskCancel = Arc<dyn Fn(&mut App) -> bool + Send + Sync>;
type StructuredTaskReveal = Arc<dyn Fn(&mut Window, &mut App) -> bool + Send + Sync>;

struct StructuredTaskHandleInner {
    task_id: TaskId,
    state: StructuredTaskState,
    subscribers: Vec<StructuredTaskSubscriber>,
    cancel: Option<StructuredTaskCancel>,
    reveal: Option<StructuredTaskReveal>,
}

#[derive(Clone)]
pub struct StructuredTaskHandle {
    inner: Arc<parking_lot::Mutex<StructuredTaskHandleInner>>,
}

impl std::fmt::Debug for StructuredTaskHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StructuredTaskHandle")
            .field("task_id", &self.task_id())
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}

impl StructuredTaskHandle {
    pub fn new(task_id: TaskId) -> Self {
        Self {
            inner: Arc::new(parking_lot::Mutex::new(StructuredTaskHandleInner {
                task_id,
                state: StructuredTaskState::Queued,
                subscribers: Vec::new(),
                cancel: None,
                reveal: None,
            })),
        }
    }

    pub fn task_id(&self) -> TaskId {
        self.inner.lock().task_id.clone()
    }

    pub fn state(&self) -> StructuredTaskState {
        self.inner.lock().state.clone()
    }

    pub fn subscribe(
        &self,
        cx: &mut App,
        subscriber: impl Fn(&StructuredTaskLifecycleEvent, &mut App) + Send + Sync + 'static,
    ) {
        let subscriber: StructuredTaskSubscriber = Arc::new(subscriber);
        let event = self.event();
        subscriber(&event, cx);
        self.inner.lock().subscribers.push(subscriber);
    }

    pub fn mark_running(&self, terminal_id: Option<StructuredTerminalId>, cx: &mut App) -> bool {
        self.transition(StructuredTaskState::Running { terminal_id }, cx)
    }

    pub fn bind_terminal(
        &self,
        terminal_id: StructuredTerminalId,
        cancel: impl Fn(&mut App) -> bool + Send + Sync + 'static,
        cx: &mut App,
    ) -> bool {
        let cancel: StructuredTaskCancel = Arc::new(cancel);
        let was_cancelled = {
            let mut inner = self.inner.lock();
            inner.cancel = Some(cancel.clone());
            matches!(inner.state, StructuredTaskState::Cancelled { .. })
        };
        if was_cancelled {
            cancel(cx);
            self.mark_cancelled(Some(terminal_id), false, cx);
            false
        } else {
            self.mark_running(Some(terminal_id), cx)
        }
    }

    pub fn set_reveal(
        &self,
        reveal: impl Fn(&mut Window, &mut App) -> bool + Send + Sync + 'static,
    ) {
        self.inner.lock().reveal = Some(Arc::new(reveal));
    }

    pub fn reveal_terminal(&self, window: &mut Window, cx: &mut App) -> bool {
        let reveal = self.inner.lock().reveal.clone();
        reveal.is_some_and(|reveal| reveal(window, cx))
    }

    pub fn cancel(&self, cx: &mut App) -> bool {
        let (state, cancel) = {
            let inner = self.inner.lock();
            (inner.state.clone(), inner.cancel.clone())
        };
        if state.is_terminal() {
            return false;
        }
        let terminal_id = state.terminal_id();
        let termination_confirmed = if let Some(cancel) = cancel {
            cancel(cx);
            false
        } else {
            matches!(state, StructuredTaskState::Queued)
        };
        self.mark_cancelled(terminal_id, termination_confirmed, cx)
    }

    pub fn mark_completed(&self, exit_status: ExitStatus, cx: &mut App) -> bool {
        self.transition(
            StructuredTaskState::Completed {
                terminal_id: self.state().terminal_id(),
                exit_code: exit_status.code(),
                success: exit_status.success(),
            },
            cx,
        )
    }

    pub fn mark_spawn_error(&self, message: impl Into<String>, cx: &mut App) -> bool {
        const MAX_ERROR_BYTES: usize = 4 * 1024;
        let mut message = message.into();
        if message.len() > MAX_ERROR_BYTES {
            let mut end = MAX_ERROR_BYTES;
            while end > 0 && !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
        }
        self.transition(StructuredTaskState::SpawnError { message }, cx)
    }

    pub fn mark_cancelled(
        &self,
        terminal_id: Option<StructuredTerminalId>,
        termination_confirmed: bool,
        cx: &mut App,
    ) -> bool {
        self.transition(
            StructuredTaskState::Cancelled {
                terminal_id,
                termination_confirmed,
            },
            cx,
        )
    }

    fn event(&self) -> StructuredTaskLifecycleEvent {
        let inner = self.inner.lock();
        StructuredTaskLifecycleEvent {
            task_id: inner.task_id.clone(),
            state: inner.state.clone(),
        }
    }

    fn transition(&self, state: StructuredTaskState, cx: &mut App) -> bool {
        let subscribers = {
            let mut inner = self.inner.lock();
            if inner.state == state {
                return false;
            }
            let cancellation_update = matches!(
                (&inner.state, &state),
                (
                    StructuredTaskState::Cancelled {
                        terminal_id: old_terminal_id,
                        termination_confirmed: old_confirmed,
                    },
                    StructuredTaskState::Cancelled {
                        terminal_id: new_terminal_id,
                        termination_confirmed: new_confirmed,
                    },
                ) if (!old_confirmed && *new_confirmed)
                    || (old_terminal_id.is_none() && new_terminal_id.is_some())
            );
            if inner.state.is_terminal() && !cancellation_update {
                return false;
            }
            let valid = matches!(
                (&inner.state, &state),
                (
                    StructuredTaskState::Queued,
                    StructuredTaskState::Running { .. }
                        | StructuredTaskState::SpawnError { .. }
                        | StructuredTaskState::Cancelled { .. }
                ) | (
                    StructuredTaskState::Running { .. },
                    StructuredTaskState::Running { .. }
                        | StructuredTaskState::Completed { .. }
                        | StructuredTaskState::SpawnError { .. }
                        | StructuredTaskState::Cancelled { .. }
                )
            ) || cancellation_update;
            if !valid {
                return false;
            }
            inner.state = state;
            inner.subscribers.clone()
        };
        let event = self.event();
        for subscriber in subscribers {
            subscriber(&event, cx);
        }
        true
    }
}

/// Contains all information needed by Zed to spawn a new terminal tab for the given task.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct SpawnInTerminal {
    /// Id of the task to use when determining task tab affinity.
    pub id: TaskId,
    /// Full unshortened form of `label` field.
    pub full_label: String,
    /// Human readable name of the terminal tab.
    pub label: String,
    /// Executable command to spawn.
    pub command: Option<String>,
    /// Arguments to the command, potentially unsubstituted,
    /// to let the shell that spawns the command to do the substitution, if needed.
    pub args: Vec<String>,
    /// A human-readable label, containing command and all of its arguments, joined and substituted.
    pub command_label: String,
    /// Keep the user-facing label when an internal wrapper changes the executable.
    pub preserve_command_label: bool,
    /// Treat the executable command as one literal shell token even without arguments.
    pub literal_command: bool,
    /// Quote the internal command and this many leading arguments as literal shell tokens.
    /// Remaining Task arguments retain their normal shell substitution behavior.
    pub literal_prefix_args: usize,
    /// Preserve provider argv through ConPTY instead of reparsing it in a Windows shell.
    pub windows_direct_program: bool,
    /// Current working directory to spawn the command into.
    pub cwd: Option<PathBuf>,
    /// Env overrides for the command, will be appended to the terminal's environment from the settings.
    pub env: HashMap<String, String>,
    /// Whether to use a new terminal tab or reuse the existing one to spawn the process.
    pub use_new_terminal: bool,
    /// Whether to allow multiple instances of the same task to be run, or rather wait for the existing ones to finish.
    pub allow_concurrent_runs: bool,
    /// What to do with the terminal pane and tab, after the command was started.
    pub reveal: RevealStrategy,
    /// Where to show tasks' terminal output.
    pub reveal_target: RevealTarget,
    /// What to do with the terminal pane and tab, after the command had finished.
    pub hide: HideStrategy,
    /// Which shell to use when spawning the task.
    pub shell: Shell,
    /// Whether to show the task summary line in the task output (success/failure).
    pub show_summary: bool,
    /// Whether to show the command line in the task output.
    pub show_command: bool,
    /// Whether to show the rerun button in the terminal tab.
    pub show_rerun: bool,
    /// Which edited buffers to save before running the task.
    pub save: SaveStrategy,
    pub required_experimental_features: Vec<String>,
}

pub fn validate_experimental_feature_requirement(feature: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(
            feature,
            "rust-coverage-experimental"
                | "rust-native-profiling-experimental"
                | "rust-named-cargo-presets-experimental"
                | "rust-exact-doctest-actions-experimental"
        ),
        "Unknown experimental Task requirement"
    );
    Ok(())
}

pub fn validate_experimental_feature_requirements(features: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(
        features.len() <= 4,
        "Too many experimental Task requirements"
    );
    for (index, feature) in features.iter().enumerate() {
        validate_experimental_feature_requirement(feature)?;
        anyhow::ensure!(
            !features[..index].contains(feature),
            "Duplicate experimental Task requirement"
        );
    }
    Ok(())
}

impl SpawnInTerminal {
    pub fn to_proto(&self) -> proto::SpawnInTerminal {
        proto::SpawnInTerminal {
            label: self.label.clone(),
            command: self.command.clone(),
            args: self.args.clone(),
            required_experimental_features: self.required_experimental_features.clone(),
            env: self
                .env
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            cwd: self
                .cwd
                .clone()
                .map(|cwd| cwd.to_string_lossy().into_owned()),
        }
    }

    pub fn from_proto(proto: proto::SpawnInTerminal) -> Self {
        Self {
            label: proto.label.clone(),
            command: proto.command.clone(),
            args: proto.args.clone(),
            required_experimental_features: proto.required_experimental_features,
            env: proto.env.into_iter().collect(),
            cwd: proto.cwd.map(PathBuf::from),
            ..Default::default()
        }
    }
}

/// The exact argument containing a single artifact's resolved output path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskArtifactArgumentBinding {
    pub argument_index: usize,
}

/// Explicit relations between an output argument and every file in an artifact group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskArtifactGroupArgumentBinding {
    pub argument_index: usize,
    pub role: String,
    pub related: Vec<TaskArtifactRelatedArgumentBinding>,
}

/// A declared extension replacement of the group's output path, optionally passed as an argument.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskArtifactRelatedArgumentBinding {
    pub role: String,
    pub extension: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argument_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedTaskArtifactGroupArgumentBinding {
    binding: TaskArtifactGroupArgumentBinding,
    unbound_group: TaskArtifactGroup,
}

impl ResolvedTaskArtifactGroupArgumentBinding {
    fn new(
        binding: &TaskArtifactGroupArgumentBinding,
        group: Option<&TaskArtifactGroup>,
        arguments: &[String],
        literal_prefix_args: usize,
    ) -> anyhow::Result<Self> {
        validate_artifact_group_binding_metadata(binding)?;
        let group = group.context("artifact group argument binding requires one group")?;
        let resolved = Self {
            binding: binding.clone(),
            unbound_group: group.clone(),
        };
        resolved.validate(group, arguments, literal_prefix_args, None)?;
        Ok(resolved)
    }

    fn validate(
        &self,
        group: &TaskArtifactGroup,
        arguments: &[String],
        literal_prefix_args: usize,
        artifact_directory: Option<&Path>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            group.format == self.unbound_group.format,
            "artifact group argument binding must preserve its format"
        );
        self.unbound_group
            .with_resolved_files(group.files.clone())?;
        validate_artifact_group_binding_metadata(&self.binding)?;
        let output = group
            .files
            .get(&self.binding.role)
            .context("artifact group argument binding output role is not declared")?;
        ResolvedTaskArtifactArgumentBinding::validate_argument(
            self.binding.argument_index,
            output,
            arguments,
            literal_prefix_args,
            artifact_directory,
        )?;
        let mut roles = HashSet::default();
        let mut indices = HashSet::default();
        roles.insert(&self.binding.role);
        indices.insert(self.binding.argument_index);
        for related in &self.binding.related {
            anyhow::ensure!(
                roles.insert(&related.role),
                "artifact group argument binding repeats a role"
            );
            let artifact = group
                .files
                .get(&related.role)
                .context("artifact group argument binding related role is not declared")?;
            let path = related_artifact_path(&output.path, &related.extension)?;
            anyhow::ensure!(
                artifact.path == path,
                "artifact group argument binding must match its declared path relation"
            );
            if artifact_directory.is_some() && related.argument_index.is_none() {
                artifact_argument_path(artifact, artifact_directory)?;
            }
            if let Some(index) = related.argument_index {
                anyhow::ensure!(
                    indices.insert(index),
                    "artifact group argument binding repeats an argument index"
                );
                ResolvedTaskArtifactArgumentBinding::validate_argument(
                    index,
                    artifact,
                    arguments,
                    literal_prefix_args,
                    artifact_directory,
                )?;
            }
        }
        anyhow::ensure!(
            roles.len() == group.files.len(),
            "artifact group argument binding must cover every declared role"
        );
        Ok(())
    }

    fn bind(&self, task: &ResolvedTask, invocation_id: &str) -> anyhow::Result<ResolvedTask> {
        anyhow::ensure!(
            task.resolved_artifact.is_none()
                && task.artifact_argument_binding.is_none()
                && task.resolved.command.is_some(),
            "artifact group argument binding requires an executable group task"
        );
        let current = task
            .resolved_artifact_group
            .as_ref()
            .context("artifact group argument binding requires one group")?;
        self.validate(
            current,
            &task.resolved.args,
            task.resolved.literal_prefix_args,
            task.resolved_artifact_directory.as_deref(),
        )?;
        let invocation_id = normalized_artifact_invocation_id(invocation_id)?;
        let output = self
            .unbound_group
            .files
            .get(&self.binding.role)
            .context("artifact group argument binding output role is not declared")?;
        let output =
            output.with_resolved_path(bound_artifact_path(&output.path, &invocation_id)?)?;
        let mut files = BTreeMap::new();
        files.insert(self.binding.role.clone(), output.clone());
        for related in &self.binding.related {
            let artifact = self
                .unbound_group
                .files
                .get(&related.role)
                .context("artifact group argument binding related role is not declared")?;
            files.insert(
                related.role.clone(),
                artifact
                    .with_resolved_path(related_artifact_path(&output.path, &related.extension)?)?,
            );
        }
        let group = self.unbound_group.with_resolved_files(files)?;
        let mut bound = task.clone();
        *bound
            .resolved
            .args
            .get_mut(self.binding.argument_index)
            .context("artifact group argument binding has no output argument")? =
            artifact_argument_path(&output, bound.resolved_artifact_directory.as_deref())?;
        for related in &self.binding.related {
            if let Some(index) = related.argument_index {
                *bound
                    .resolved
                    .args
                    .get_mut(index)
                    .context("artifact group argument binding has no related argument")? =
                    artifact_argument_path(
                        group.files.get(&related.role).context(
                            "artifact group argument binding related role is not declared",
                        )?,
                        bound.resolved_artifact_directory.as_deref(),
                    )?;
            }
        }
        self.validate(
            &group,
            &bound.resolved.args,
            bound.resolved.literal_prefix_args,
            bound.resolved_artifact_directory.as_deref(),
        )?;
        bound.resolved_artifact_group = Some(group);
        bound.refresh_artifact_command_label()?;
        Ok(bound)
    }
}

fn validate_artifact_group_binding_metadata(
    binding: &TaskArtifactGroupArgumentBinding,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        binding.related.len() < MAX_TASK_ARTIFACT_GROUP_FILES,
        "artifact group argument binding exceeds its role limit"
    );
    let valid_role = |role: &str| {
        !role.is_empty()
            && role.len() <= 64
            && role
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    };
    anyhow::ensure!(
        valid_role(&binding.role),
        "artifact group argument binding has an invalid output role"
    );
    for related in &binding.related {
        anyhow::ensure!(
            valid_role(&related.role),
            "artifact group argument binding has an invalid related role"
        );
        validate_artifact_related_extension(&related.extension)?;
    }
    Ok(())
}

fn validate_artifact_related_extension(extension: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !extension.is_empty()
            && extension.len() <= 64
            && extension.split('.').all(|part| !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))),
        "artifact group argument binding has an invalid related extension"
    );
    Ok(())
}

fn related_artifact_path(output_path: &str, extension: &str) -> anyhow::Result<String> {
    validate_artifact_related_extension(extension)?;
    let path = PathBuf::from(output_path).with_extension(extension);
    let path = path
        .to_str()
        .context("artifact group argument binding requires a text path")?;
    let filename = path
        .rsplit('/')
        .next()
        .context("artifact group argument binding has no filename")?;
    anyhow::ensure!(
        filename.len() <= 255,
        "bound artifact filename exceeds its component byte limit"
    );
    Ok(path.to_owned())
}

fn bound_artifact_path(path: &str, invocation_id: &str) -> anyhow::Result<String> {
    let (parent, filename) = path
        .rsplit_once('/')
        .map_or(("", path), |(parent, filename)| (parent, filename));
    let filename = match filename.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() => {
            format!("{stem}-{invocation_id}.{extension}")
        }
        _ => format!("{filename}-{invocation_id}"),
    };
    anyhow::ensure!(
        filename.len() <= 255,
        "bound artifact filename exceeds its component byte limit"
    );
    Ok(if parent.is_empty() {
        filename
    } else {
        format!("{parent}/{filename}")
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedTaskArtifactArgumentBinding {
    argument_index: usize,
    unbound_artifact: TaskArtifact,
}

impl ResolvedTaskArtifactArgumentBinding {
    fn new(
        binding: &TaskArtifactArgumentBinding,
        artifact: Option<&TaskArtifact>,
        arguments: &[String],
        literal_prefix_args: usize,
    ) -> anyhow::Result<Self> {
        let artifact = artifact.context("artifact argument binding requires one artifact")?;
        Self::validate_argument(
            binding.argument_index,
            artifact,
            arguments,
            literal_prefix_args,
            None,
        )?;
        Ok(Self {
            argument_index: binding.argument_index,
            unbound_artifact: artifact.clone(),
        })
    }

    fn validate_argument(
        argument_index: usize,
        artifact: &TaskArtifact,
        arguments: &[String],
        literal_prefix_args: usize,
        artifact_directory: Option<&Path>,
    ) -> anyhow::Result<()> {
        let argument = artifact_argument_path(artifact, artifact_directory)?;
        anyhow::ensure!(
            arguments.get(argument_index) == Some(&argument),
            "artifact argument binding must select the exact artifact path"
        );
        anyhow::ensure!(
            !artifact
                .path
                .chars()
                .any(|character| character.is_control() || character == '\\' || character == ':')
                && artifact
                    .path
                    .split('/')
                    .all(|component| !component.is_empty()
                        && component != "."
                        && component != ".."),
            "artifact argument binding requires a portable relative path"
        );
        if argument_index >= literal_prefix_args {
            anyhow::ensure!(
                artifact_directory.is_none(),
                "an anchored artifact argument must be a literal shell token"
            );
            anyhow::ensure!(
                artifact
                    .path
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_./-".contains(&byte)),
                "artifact argument binding requires literal arguments for shell-sensitive paths"
            );
        }
        Ok(())
    }
}

fn artifact_argument_path(
    artifact: &TaskArtifact,
    artifact_directory: Option<&Path>,
) -> anyhow::Result<String> {
    let Some(directory) = artifact_directory else {
        return Ok(artifact.path.clone());
    };
    validate_artifact_directory(directory)?;
    artifact.with_resolved_path(artifact.path.clone())?;
    let argument = directory.join(&artifact.path);
    let argument = argument
        .to_str()
        .context("artifact argument path is not UTF-8")?;
    anyhow::ensure!(
        argument.len() <= 4096,
        "anchored artifact argument exceeds its path limit"
    );
    Ok(argument.to_owned())
}

fn validate_artifact_directory(directory: &Path) -> anyhow::Result<()> {
    let text = directory
        .to_str()
        .context("artifact directory is not UTF-8")?;
    anyhow::ensure!(
        directory.is_absolute()
            && text.len() <= 4096
            && !text.chars().any(char::is_control)
            && !text
                .split(std::path::is_separator)
                .any(|component| matches!(component, "." | ".."))
            && !directory.components().any(|component| matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )),
        "artifact directory must be a bounded absolute path without parent traversal"
    );
    Ok(())
}

/// A final form of the [`TaskTemplate`], that got resolved with a particular [`TaskContext`] and now is ready to spawn the actual task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedTask {
    /// A way to distinguish tasks produced by the same template, but different contexts.
    /// NOTE: Resolved tasks may have the same labels, commands and do the same things,
    /// but still may have different ids if the context was different during the resolution.
    /// Since the template has `env` field, for a generic task that may be a bash command,
    /// so it's impossible to determine the id equality without more context in a generic case.
    pub id: TaskId,
    /// A template the task got resolved from.
    original_task: TaskTemplate,
    resolved_artifact: Option<TaskArtifact>,
    resolved_artifact_group: Option<TaskArtifactGroup>,
    resolved_artifact_directory: Option<PathBuf>,
    artifact_argument_binding: Option<ResolvedTaskArtifactArgumentBinding>,
    artifact_group_argument_binding: Option<ResolvedTaskArtifactGroupArgumentBinding>,
    /// Full, unshortened label of the task after all resolutions are made.
    pub resolved_label: String,
    /// Variables that were substituted during the task template resolution.
    substituted_variables: HashSet<VariableName>,
    /// Further actions that need to take place after the resolved task is spawned,
    /// with all task variables resolved.
    pub resolved: SpawnInTerminal,
}

impl ResolvedTask {
    /// A task template before the resolution.
    pub fn original_task(&self) -> &TaskTemplate {
        &self.original_task
    }

    pub fn stdout_report(&self) -> Option<&TaskStdoutReport> {
        self.original_task.stdout_report.as_ref()
    }

    /// Variables that were substituted during the task template resolution.
    pub fn substituted_variables(&self) -> &HashSet<VariableName> {
        &self.substituted_variables
    }

    pub fn resolved_artifact_group(&self) -> Option<&TaskArtifactGroup> {
        self.resolved_artifact_group.as_ref()
    }

    pub fn resolved_artifact(&self) -> Option<&TaskArtifact> {
        self.resolved_artifact.as_ref()
    }

    /// This is resolved path data only; the owning feature must validate the host,
    /// filesystem and directory freshness before allowing the Task to launch.
    pub fn artifact_directory(&self) -> Option<&Path> {
        self.resolved_artifact_directory.as_deref()
    }

    /// Keeps declared artifact paths relative to their reviewed anchor when the
    /// command executes from another directory. Templates retain no host authority.
    pub fn with_artifact_directory(&self, directory: &Path) -> anyhow::Result<Self> {
        validate_artifact_directory(directory)?;
        anyhow::ensure!(
            self.resolved_artifact_directory.is_none(),
            "artifact directory must be freshly reviewed before binding"
        );
        anyhow::ensure!(
            self.resolved.command.is_some(),
            "artifact directory requires an executable task"
        );
        let mut replacements = Vec::new();
        if let Some(binding) = &self.artifact_group_argument_binding {
            anyhow::ensure!(
                self.resolved_artifact.is_none() && self.artifact_argument_binding.is_none(),
                "artifact directory requires one explicitly bound artifact declaration"
            );
            let group = self
                .resolved_artifact_group
                .as_ref()
                .context("artifact directory requires a declared group")?;
            binding.validate(
                group,
                &self.resolved.args,
                self.resolved.literal_prefix_args,
                None,
            )?;
            let output = group
                .files
                .get(&binding.binding.role)
                .context("artifact directory output role is unavailable")?;
            replacements.push((
                binding.binding.argument_index,
                artifact_argument_path(output, Some(directory))?,
            ));
            for related in &binding.binding.related {
                let artifact = group
                    .files
                    .get(&related.role)
                    .context("artifact directory related role is unavailable")?;
                let argument = artifact_argument_path(artifact, Some(directory))?;
                if let Some(index) = related.argument_index {
                    replacements.push((index, argument));
                }
            }
        } else if let Some(binding) = &self.artifact_argument_binding {
            anyhow::ensure!(
                self.resolved_artifact_group.is_none(),
                "artifact directory requires one explicitly bound artifact declaration"
            );
            let artifact = self
                .resolved_artifact
                .as_ref()
                .context("artifact directory requires a declared artifact")?;
            ResolvedTaskArtifactArgumentBinding::validate_argument(
                binding.argument_index,
                artifact,
                &self.resolved.args,
                self.resolved.literal_prefix_args,
                None,
            )?;
            replacements.push((
                binding.argument_index,
                artifact_argument_path(artifact, Some(directory))?,
            ));
        } else {
            anyhow::bail!("artifact directory requires an explicit artifact argument binding");
        }
        for (index, _) in &replacements {
            anyhow::ensure!(
                *index < self.resolved.literal_prefix_args,
                "an anchored artifact argument must be a literal shell token"
            );
        }
        let mut anchored = self.clone();
        for (index, argument) in replacements {
            *anchored
                .resolved
                .args
                .get_mut(index)
                .context("artifact directory output argument is unavailable")? = argument;
        }
        anchored.resolved_artifact_directory = Some(directory.to_owned());
        if let Some(binding) = &anchored.artifact_group_argument_binding {
            binding.validate(
                anchored
                    .resolved_artifact_group
                    .as_ref()
                    .context("artifact directory requires a declared group")?,
                &anchored.resolved.args,
                anchored.resolved.literal_prefix_args,
                Some(directory),
            )?;
        } else if let Some(binding) = &anchored.artifact_argument_binding {
            ResolvedTaskArtifactArgumentBinding::validate_argument(
                binding.argument_index,
                anchored
                    .resolved_artifact
                    .as_ref()
                    .context("artifact directory requires a declared artifact")?,
                &anchored.resolved.args,
                anchored.resolved.literal_prefix_args,
                Some(directory),
            )?;
        }
        anchored.refresh_artifact_command_label()?;
        Ok(anchored)
    }

    /// Returns a per-invocation clone without changing template or task identity.
    pub fn bind_artifact_invocation(&self, invocation_id: &str) -> anyhow::Result<Self> {
        if let Some(binding) = &self.artifact_group_argument_binding {
            return binding.bind(self, invocation_id);
        }
        let Some(binding) = &self.artifact_argument_binding else {
            return Ok(self.clone());
        };
        let current_artifact = self
            .resolved_artifact
            .as_ref()
            .context("artifact argument binding requires one artifact")?;
        ResolvedTaskArtifactArgumentBinding::validate_argument(
            binding.argument_index,
            current_artifact,
            &self.resolved.args,
            self.resolved.literal_prefix_args,
            self.resolved_artifact_directory.as_deref(),
        )?;
        anyhow::ensure!(
            self.resolved_artifact_group.is_none() && self.resolved.command.is_some(),
            "artifact argument binding requires an executable single-artifact task"
        );
        let invocation_id = normalized_artifact_invocation_id(invocation_id)?;
        let path = bound_artifact_path(&binding.unbound_artifact.path, &invocation_id)?;
        let artifact = binding.unbound_artifact.with_resolved_path(path)?;
        let mut bound = self.clone();
        *bound
            .resolved
            .args
            .get_mut(binding.argument_index)
            .context("artifact argument binding has no output argument")? =
            artifact_argument_path(&artifact, bound.resolved_artifact_directory.as_deref())?;
        bound.resolved_artifact = Some(artifact);
        bound.refresh_artifact_command_label()?;
        Ok(bound)
    }

    fn refresh_artifact_command_label(&mut self) -> anyhow::Result<()> {
        if !self.resolved.preserve_command_label {
            let command = self
                .resolved
                .command
                .as_ref()
                .context("artifact argument binding requires an executable command")?;
            self.resolved.command_label =
                self.resolved
                    .args
                    .iter()
                    .fold(command.clone(), |mut label, argument| {
                        label.push(' ');
                        label.push_str(argument);
                        label
                    });
        }
        Ok(())
    }

    /// A human-readable label to display in the UI.
    pub fn display_label(&self) -> &str {
        self.resolved.label.as_str()
    }
}

fn normalized_artifact_invocation_id(invocation_id: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        invocation_id.len() == 32 || invocation_id.len() == 36,
        "artifact invocation requires a non-nil UUID"
    );
    let hyphenated = invocation_id.len() == 36;
    let mut normalized = String::with_capacity(32);
    for (index, byte) in invocation_id.bytes().enumerate() {
        if hyphenated && matches!(index, 8 | 13 | 18 | 23) {
            anyhow::ensure!(byte == b'-', "artifact invocation requires a non-nil UUID");
        } else {
            anyhow::ensure!(
                byte.is_ascii_hexdigit(),
                "artifact invocation requires a non-nil UUID"
            );
            normalized.push(char::from(byte.to_ascii_lowercase()));
        }
    }
    anyhow::ensure!(
        normalized.bytes().any(|byte| byte != b'0'),
        "artifact invocation requires a non-nil UUID"
    );
    Ok(normalized)
}

/// Variables, available for use in [`TaskContext`] when a Zed's [`TaskTemplate`] gets resolved into a [`ResolvedTask`].
/// Name of the variable must be a valid shell variable identifier, which generally means that it is
/// a word  consisting only  of alphanumeric characters and underscores,
/// and beginning with an alphabetic character or an  underscore.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum VariableName {
    /// An absolute path of the currently opened file.
    File,
    /// A path of the currently opened file (relative to worktree root).
    RelativeFile,
    /// A path of the currently opened file's directory (relative to worktree root).
    RelativeDir,
    /// The currently opened filename.
    Filename,
    /// The path to a parent directory of a currently opened file.
    Dirname,
    /// Stem (filename without extension) of the currently opened file.
    Stem,
    /// An absolute path of the currently opened worktree, that contains the file.
    WorktreeRoot,
    /// A symbol text, that contains latest cursor/selection position.
    Symbol,
    /// A row with the latest cursor/selection position.
    Row,
    /// A column with the latest cursor/selection position.
    Column,
    /// Text from the latest selection.
    SelectedText,
    /// The language of the currently opened buffer (e.g., "Rust", "Python").
    Language,
    /// The symbol selected by the symbol tagging system, specifically the @run capture in a runnables.scm
    RunnableSymbol,
    /// Open a Picker to select a process ID to use in place
    /// Can only be used to debug configurations
    PickProcessId,
    /// An absolute path of the main (original) git worktree for the current repository.
    /// For normal checkouts, this equals the worktree root. For linked worktrees,
    /// this is the original repo's working directory.
    MainGitWorktree,
    /// Full SHA for the Git commit associated with the task context.
    GitSha,
    /// Short SHA for the Git commit associated with the task context.
    GitShaShort,
    /// Name of the Git repository associated with the task context.
    GitRepositoryName,
    /// Absolute path of the Git repository associated with the task context.
    GitRepositoryPath,
    /// Name of the Git ref (branch, remote ref, or tag) associated with the task context.
    GitRef,
    /// Custom variable, provided by the plugin or other external source.
    /// Will be printed with `CUSTOM_` prefix to avoid potential conflicts with other variables.
    Custom(Cow<'static, str>),
}

impl VariableName {
    /// Generates a `$VARIABLE`-like string value to be used in templates.
    pub fn template_value(&self) -> String {
        format!("${self}")
    }
    /// Generates a `"$VARIABLE"`-like string, to be used instead of `Self::template_value` when expanded value could contain spaces or special characters.
    pub fn template_value_with_whitespace(&self) -> String {
        format!("\"${self}\"")
    }
}

impl FromStr for VariableName {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let without_prefix = s.strip_prefix(ZED_VARIABLE_NAME_PREFIX).ok_or(())?;
        let value = match without_prefix {
            "FILE" => Self::File,
            "FILENAME" => Self::Filename,
            "RELATIVE_FILE" => Self::RelativeFile,
            "RELATIVE_DIR" => Self::RelativeDir,
            "DIRNAME" => Self::Dirname,
            "STEM" => Self::Stem,
            "WORKTREE_ROOT" => Self::WorktreeRoot,
            "SYMBOL" => Self::Symbol,
            "RUNNABLE_SYMBOL" => Self::RunnableSymbol,
            "SELECTED_TEXT" => Self::SelectedText,
            "LANGUAGE" => Self::Language,
            "ROW" => Self::Row,
            "COLUMN" => Self::Column,
            "MAIN_GIT_WORKTREE" => Self::MainGitWorktree,
            "GIT_SHA" => Self::GitSha,
            "GIT_SHA_SHORT" => Self::GitShaShort,
            "GIT_REPOSITORY_NAME" => Self::GitRepositoryName,
            "GIT_REPOSITORY_PATH" => Self::GitRepositoryPath,
            "GIT_REF" => Self::GitRef,
            _ => {
                if let Some(custom_name) =
                    without_prefix.strip_prefix(ZED_CUSTOM_VARIABLE_NAME_PREFIX)
                {
                    Self::Custom(Cow::Owned(custom_name.to_owned()))
                } else {
                    return Err(());
                }
            }
        };
        Ok(value)
    }
}

/// A prefix that all [`VariableName`] variants are prefixed with when used in environment variables and similar template contexts.
pub const ZED_VARIABLE_NAME_PREFIX: &str = "ZED_";
const ZED_CUSTOM_VARIABLE_NAME_PREFIX: &str = "CUSTOM_";

impl std::fmt::Display for VariableName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::File => write!(f, "{ZED_VARIABLE_NAME_PREFIX}FILE"),
            Self::Filename => write!(f, "{ZED_VARIABLE_NAME_PREFIX}FILENAME"),
            Self::RelativeFile => write!(f, "{ZED_VARIABLE_NAME_PREFIX}RELATIVE_FILE"),
            Self::RelativeDir => write!(f, "{ZED_VARIABLE_NAME_PREFIX}RELATIVE_DIR"),
            Self::Dirname => write!(f, "{ZED_VARIABLE_NAME_PREFIX}DIRNAME"),
            Self::Stem => write!(f, "{ZED_VARIABLE_NAME_PREFIX}STEM"),
            Self::WorktreeRoot => write!(f, "{ZED_VARIABLE_NAME_PREFIX}WORKTREE_ROOT"),
            Self::Symbol => write!(f, "{ZED_VARIABLE_NAME_PREFIX}SYMBOL"),
            Self::Row => write!(f, "{ZED_VARIABLE_NAME_PREFIX}ROW"),
            Self::Column => write!(f, "{ZED_VARIABLE_NAME_PREFIX}COLUMN"),
            Self::SelectedText => write!(f, "{ZED_VARIABLE_NAME_PREFIX}SELECTED_TEXT"),
            Self::Language => write!(f, "{ZED_VARIABLE_NAME_PREFIX}LANGUAGE"),
            Self::RunnableSymbol => write!(f, "{ZED_VARIABLE_NAME_PREFIX}RUNNABLE_SYMBOL"),
            Self::PickProcessId => write!(f, "{ZED_VARIABLE_NAME_PREFIX}PICK_PID"),
            Self::MainGitWorktree => write!(f, "{ZED_VARIABLE_NAME_PREFIX}MAIN_GIT_WORKTREE"),
            Self::GitSha => write!(f, "{ZED_VARIABLE_NAME_PREFIX}GIT_SHA"),
            Self::GitShaShort => write!(f, "{ZED_VARIABLE_NAME_PREFIX}GIT_SHA_SHORT"),
            Self::GitRepositoryName => write!(f, "{ZED_VARIABLE_NAME_PREFIX}GIT_REPOSITORY_NAME"),
            Self::GitRepositoryPath => write!(f, "{ZED_VARIABLE_NAME_PREFIX}GIT_REPOSITORY_PATH"),
            Self::GitRef => write!(f, "{ZED_VARIABLE_NAME_PREFIX}GIT_REF"),
            Self::Custom(s) => write!(
                f,
                "{ZED_VARIABLE_NAME_PREFIX}{ZED_CUSTOM_VARIABLE_NAME_PREFIX}{s}"
            ),
        }
    }
}

/// Container for predefined environment variables that describe state of Zed at the time the task was spawned.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TaskVariables(HashMap<VariableName, String>);

impl TaskVariables {
    /// Inserts another variable into the container, overwriting the existing one if it already exists — in this case, the old value is returned.
    pub fn insert(&mut self, variable: VariableName, value: String) -> Option<String> {
        self.0.insert(variable, value)
    }

    /// Extends the container with another one, overwriting the existing variables on collision.
    pub fn extend(&mut self, other: Self) {
        self.0.extend(other.0);
    }
    /// Get the value associated with given variable name, if there is one.
    pub fn get(&self, key: &VariableName) -> Option<&str> {
        self.0.get(key).map(|s| s.as_str())
    }
    /// Clear out variables obtained from tree-sitter queries, which are prefixed with '_' character
    pub fn sweep(&mut self) {
        self.0.retain(|name, _| {
            if let VariableName::Custom(name) = name {
                !name.starts_with('_')
            } else {
                true
            }
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (&VariableName, &String)> {
        self.0.iter()
    }
}

impl FromIterator<(VariableName, String)> for TaskVariables {
    fn from_iter<T: IntoIterator<Item = (VariableName, String)>>(iter: T) -> Self {
        Self(HashMap::from_iter(iter))
    }
}

impl IntoIterator for TaskVariables {
    type Item = (VariableName, String);

    type IntoIter = hash_map::IntoIter<VariableName, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Keeps track of the file associated with a task and context of tasks execution (i.e. current file or current function).
/// Keeps all Zed-related state inside, used to produce a resolved task out of its template.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TaskContext {
    /// A path to a directory in which the task should be executed.
    pub cwd: Option<PathBuf>,
    /// Additional environment variables associated with a given task.
    pub task_variables: TaskVariables,
    /// Environment variables obtained when loading the project into Zed.
    /// This is the environment one would get when `cd`ing in a terminal
    /// into the project's root directory.
    pub project_env: HashMap<String, String>,
}

/// A shared reference to a [`TaskContext`], used to avoid cloning the context multiple times.
#[derive(Clone, Debug, Default)]
pub struct SharedTaskContext(Arc<TaskContext>);

impl std::ops::Deref for SharedTaskContext {
    type Target = TaskContext;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<TaskContext> for SharedTaskContext {
    fn from(context: TaskContext) -> Self {
        Self(Arc::new(context))
    }
}

/// This is a new type representing a 'tag' on a 'runnable symbol', typically a test of main() function, found via treesitter.
#[derive(Clone, Debug)]
pub struct RunnableTag(pub SharedString);

pub fn shell_from_proto(proto: proto::Shell) -> anyhow::Result<Shell> {
    let shell_type = proto.shell_type.context("invalid shell type")?;
    let shell = match shell_type {
        proto::shell::ShellType::System(_) => Shell::System,
        proto::shell::ShellType::Program(program) => Shell::Program(program),
        proto::shell::ShellType::WithArguments(program) => Shell::WithArguments {
            program: program.program,
            args: program.args,
            title_override: None,
        },
    };
    Ok(shell)
}

pub fn shell_to_proto(shell: Shell) -> proto::Shell {
    let shell_type = match shell {
        Shell::System => proto::shell::ShellType::System(proto::System {}),
        Shell::Program(program) => proto::shell::ShellType::Program(program),
        Shell::WithArguments {
            program,
            args,
            title_override: _,
        } => proto::shell::ShellType::WithArguments(proto::shell::WithArguments { program, args }),
    };
    proto::Shell {
        shell_type: Some(shell_type),
    }
}

type VsCodeEnvVariable = String;
type VsCodeCommand = String;
type ZedEnvVariable = String;

struct EnvVariableReplacer {
    variables: HashMap<VsCodeEnvVariable, ZedEnvVariable>,
    commands: HashMap<VsCodeCommand, ZedEnvVariable>,
}

impl EnvVariableReplacer {
    fn new(variables: HashMap<VsCodeEnvVariable, ZedEnvVariable>) -> Self {
        Self {
            variables,
            commands: HashMap::default(),
        }
    }

    fn with_commands(
        mut self,
        commands: impl IntoIterator<Item = (VsCodeCommand, ZedEnvVariable)>,
    ) -> Self {
        self.commands = commands.into_iter().collect();
        self
    }

    fn replace_value(&self, input: serde_json::Value) -> serde_json::Value {
        match input {
            serde_json::Value::String(s) => serde_json::Value::String(self.replace(&s)),
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.into_iter().map(|v| self.replace_value(v)).collect())
            }
            serde_json::Value::Object(obj) => serde_json::Value::Object(
                obj.into_iter()
                    .map(|(k, v)| (self.replace(&k), self.replace_value(v)))
                    .collect(),
            ),
            _ => input,
        }
    }
    // Replaces occurrences of VsCode-specific environment variables with Zed equivalents.
    fn replace(&self, input: &str) -> String {
        shellexpand::env_with_context_no_errors(&input, |var: &str| {
            // Colons denote a default value in case the variable is not set. We want to preserve that default, as otherwise shellexpand will substitute it for us.
            let colon_position = var.find(':').unwrap_or(var.len());
            let (left, right) = var.split_at(colon_position);
            if left == "env" && !right.is_empty() {
                let variable_name = &right[1..];
                return Some(format!("${{{variable_name}}}"));
            } else if left == "command" && !right.is_empty() {
                let command_name = &right[1..];
                if let Some(replacement_command) = self.commands.get(command_name) {
                    return Some(format!("${{{replacement_command}}}"));
                }
            }

            let (variable_name, default) = (left, right);
            let append_previous_default = |ret: &mut String| {
                if !default.is_empty() {
                    ret.push_str(default);
                }
            };
            if let Some(substitution) = self.variables.get(variable_name) {
                // Got a VSCode->Zed hit, perform a substitution
                let mut name = format!("${{{substitution}");
                append_previous_default(&mut name);
                name.push('}');
                return Some(name);
            }
            // This is an unknown variable.
            // We should not error out, as they may come from user environment (e.g. $PATH). That means that the variable substitution might not be perfect.
            // If there's a default, we need to return the string verbatim as otherwise shellexpand will apply that default for us.
            if !default.is_empty() {
                return Some(format!("${{{var}}}"));
            }
            // Else we can just return None and that variable will be left as is.
            None
        })
        .into_owned()
    }
}
