use anyhow::{Context as _, bail};
use collections::{HashMap, HashSet};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use util::schemars::{AllowTrailingCommas, DefaultDenyUnknownFields};
use util::serde::default_true;
use util::{ResultExt, truncate_and_remove_front};

use crate::{
    AttachRequest, ResolvedTask, ResolvedTaskArtifactArgumentBinding,
    ResolvedTaskArtifactGroupArgumentBinding, RevealTarget, Shell, SpawnInTerminal, TaskArtifact,
    TaskArtifactArgumentBinding, TaskArtifactGroup, TaskArtifactGroupArgumentBinding, TaskContext,
    TaskId, TaskStdoutReport, VariableName, ZED_VARIABLE_NAME_PREFIX,
    serde_helpers::non_empty_string_vec,
};

fn experimental_policy_is_unadmitted(version: &u32) -> bool {
    *version == 0
}

/// A template definition of a Zed task to run.
/// May use the [`VariableName`] to get the corresponding substitutions into its fields.
///
/// Template itself is not ready to spawn a task, it needs to be resolved with a [`TaskContext`] first, that
/// contains all relevant Zed state in task variables.
/// A single template may produce different tasks (or none) for different contexts.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct TaskTemplate {
    #[serde(default, skip_serializing_if = "experimental_policy_is_unadmitted")]
    pub experimental_policy_version: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_experimental_features: Vec<String>,
    /// Human readable name of the task to display in the UI.
    pub label: String,
    /// Executable command to spawn.
    pub command: String,
    /// Arguments to the command.
    #[serde(default)]
    pub args: Vec<String>,
    /// Treat argument strings as literal data and quote the resolved executable, including persisted Tasks and Rerun.
    #[serde(default, skip_serializing_if = "literal_arguments_are_disabled")]
    pub literal_arguments: bool,
    /// Env overrides for the command, will be appended to the terminal's environment from the settings.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Current working directory to spawn the command into, defaults to current project root.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Whether to use a new terminal tab or reuse the existing one to spawn the process.
    #[serde(default)]
    pub use_new_terminal: bool,
    /// Whether to allow multiple instances of the same task to be run, or rather wait for the existing ones to finish.
    #[serde(default)]
    pub allow_concurrent_runs: bool,
    /// What to do with the terminal pane and tab, after the command was started:
    /// * `always` — always show the task's pane, and focus the corresponding tab in it (default)
    // * `no_focus` — always show the task's pane, add the task's tab in it, but don't focus it
    // * `never` — do not alter focus, but still add/reuse the task's tab in its pane
    #[serde(default)]
    pub reveal: RevealStrategy,
    /// Where to place the task's terminal item after starting the task.
    /// * `dock` — in the terminal dock, "regular" terminal items' place (default).
    /// * `center` — in the central pane group, "main" editor area.
    #[serde(default)]
    pub reveal_target: RevealTarget,
    /// What to do with the terminal pane and tab, after the command had finished:
    /// * `never` — do nothing when the command finishes (default)
    /// * `always` — always hide the terminal tab, hide the pane also if it was the last tab in it
    /// * `on_success` — hide the terminal tab on task success only, otherwise behaves similar to `always`.
    #[serde(default)]
    pub hide: HideStrategy,
    /// Represents the tags which this template attaches to.
    /// Adding this removes this task from other UI and gives you ability to run it by tag.
    #[serde(default, deserialize_with = "non_empty_string_vec")]
    #[schemars(length(min = 1))]
    pub tags: Vec<String>,
    /// Which shell to use when spawning the task.
    #[serde(default)]
    pub shell: Shell,
    /// Whether to show the task line in the task output.
    #[serde(default = "default_true")]
    pub show_summary: bool,
    /// Whether to show the command line in the task output.
    #[serde(default = "default_true")]
    pub show_command: bool,
    /// Which edited buffers to save before running the task.
    #[serde(default)]
    pub save: SaveStrategy,
    /// Hooks that this task runs when emitted.
    #[serde(default)]
    pub hooks: HashSet<TaskHook>,
    /// Request bounded stdout data; this does not declare successful execution or artifact ownership.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdout_report: Option<TaskStdoutReport>,
    /// A bounded project-visible file produced by a successful task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<TaskArtifact>,
    /// Rebind this exact artifact output argument to a unique path on every invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_argument_binding: Option<TaskArtifactArgumentBinding>,
    /// Explicit named data files for a feature's completion workflow; not automatically opened.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_group: Option<TaskArtifactGroup>,
    /// Rebind an explicit artifact-group output argument and declared related paths per invocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_group_argument_binding: Option<TaskArtifactGroupArgumentBinding>,
}

#[derive(Deserialize, Eq, PartialEq, Clone, Debug)]
/// Use to represent debug request type
pub enum DebugArgsRequest {
    /// launch (program, cwd) are stored in TaskTemplate as (command, cwd)
    Launch,
    /// Attach
    Attach(AttachRequest),
}

/// What to do with the terminal pane and tab, after the command was started.
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskHook {
    #[serde(alias = "create_git_worktree")]
    CreateWorktree,
}

/// What to do with the terminal pane and tab, after the command was started.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RevealStrategy {
    /// Always show the task's pane, and focus the corresponding tab in it.
    #[default]
    Always,
    /// Always show the task's pane, add the task's tab in it, but don't focus it.
    NoFocus,
    /// Do not alter focus, but still add/reuse the task's tab in its pane.
    Never,
}

/// What to do with the terminal pane and tab, after the command has finished.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HideStrategy {
    /// Do nothing when the command finishes.
    #[default]
    Never,
    /// Always hide the terminal tab, hide the pane also if it was the last tab in it.
    Always,
    /// Hide the terminal tab on task success only, otherwise behaves similar to `Always`.
    OnSuccess,
}

/// Which edited buffers to save before running a task.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SaveStrategy {
    /// Save all edited buffers.
    All,
    /// Save the current buffer.
    Current,
    #[default]
    /// Don't save any buffers.
    None,
}

/// A group of Tasks defined in a JSON file.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TaskTemplates(pub Vec<TaskTemplate>);

impl TaskTemplates {
    pub const FILE_NAME: &str = "tasks.json";
    /// Generates JSON schema of Tasks JSON template format.
    pub fn generate_json_schema() -> serde_json::Value {
        let schema = schemars::generate::SchemaSettings::draft2019_09()
            .with_transform(DefaultDenyUnknownFields)
            .with_transform(AllowTrailingCommas)
            .into_generator()
            .root_schema_for::<Self>();

        serde_json::to_value(schema).unwrap()
    }
}

impl TaskTemplate {
    /// Replaces all `VariableName` task variables in the task template string fields.
    ///
    /// Every [`ResolvedTask`] gets a [`TaskId`], based on the `id_base` (to avoid collision with various task sources),
    /// and hashes of its template and [`TaskContext`], see [`ResolvedTask`] fields' documentation for more details.
    pub fn resolve_task(&self, id_base: &str, cx: &TaskContext) -> Option<ResolvedTask> {
        self.resolve_task_internal(id_base, cx, self.literal_arguments)
    }

    /// Provider arguments are data, so editor variables and shell expressions
    /// within them must not change the selected operation.
    pub fn resolve_task_with_literal_arguments(
        &self,
        id_base: &str,
        cx: &TaskContext,
    ) -> Option<ResolvedTask> {
        self.resolve_task_internal(id_base, cx, true)
    }

    fn resolve_task_internal(
        &self,
        id_base: &str,
        cx: &TaskContext,
        literal_arguments: bool,
    ) -> Option<ResolvedTask> {
        if self.experimental_policy_version > crate::EXPERIMENTAL_TASK_POLICY_VERSION {
            return None;
        }
        crate::validate_experimental_feature_requirements(&self.required_experimental_features)
            .log_err()?;
        if let Some(report) = self.stdout_report.as_ref() {
            report.validate().log_err()?;
        }
        if literal_arguments && self.args.iter().any(|argument| argument.contains('\0')) {
            return None;
        }
        if self.label.trim().is_empty() || self.command.trim().is_empty() {
            return None;
        }

        let mut variable_names = HashMap::default();
        let mut substituted_variables = HashSet::default();
        let task_variables = cx
            .task_variables
            .0
            .iter()
            .map(|(key, value)| {
                let key_string = key.to_string();
                if !variable_names.contains_key(&key_string) {
                    variable_names.insert(key_string.clone(), key.clone());
                }
                (key_string, value.as_str())
            })
            .collect::<HashMap<_, _>>();
        let truncated_variables = truncate_variables(&task_variables);
        let cwd = match self.cwd.as_deref() {
            Some(cwd) => {
                let substituted_cwd = substitute_all_template_variables_in_str(
                    cwd,
                    &task_variables,
                    &variable_names,
                    &mut substituted_variables,
                )?;
                Some(PathBuf::from(substituted_cwd))
            }
            None => None,
        }
        .or(cx.cwd.clone());
        let full_label = substitute_all_template_variables_in_str(
            &self.label,
            &task_variables,
            &variable_names,
            &mut substituted_variables,
        )?;

        // Arbitrarily picked threshold below which we don't truncate any variables.
        const TRUNCATION_THRESHOLD: usize = 64;

        let human_readable_label = if full_label.len() > TRUNCATION_THRESHOLD {
            substitute_all_template_variables_in_str(
                &self.label,
                &truncated_variables,
                &variable_names,
                &mut substituted_variables,
            )?
        } else {
            #[allow(
                clippy::redundant_clone,
                reason = "We want to clone the full_label to avoid borrowing it in the fold closure"
            )]
            full_label.clone()
        }
        .lines()
        .fold(String::new(), |mut string, line| {
            if string.is_empty() {
                string.push_str(line);
            } else {
                string.push_str("\\n");
                string.push_str(line);
            }
            string
        });

        let command = substitute_all_template_variables_in_str(
            &self.command,
            &task_variables,
            &variable_names,
            &mut substituted_variables,
        )?;
        if literal_arguments && command.contains('\0') {
            return None;
        }
        let args_with_substitutions = if literal_arguments {
            self.args.clone()
        } else {
            substitute_all_template_variables_in_vec(
                &self.args,
                &task_variables,
                &variable_names,
                &mut substituted_variables,
            )?
        };
        let resolved_artifact = if let Some(artifact) = &self.artifact {
            let path = substitute_all_template_variables_in_str(
                &artifact.path,
                &task_variables,
                &variable_names,
                &mut substituted_variables,
            )?;
            Some(artifact.with_resolved_path(path).log_err()?)
        } else {
            None
        };

        let resolved_artifact_group = if let Some(group) = &self.artifact_group {
            if self.artifact.is_some() || group.files.len() > crate::MAX_TASK_ARTIFACT_GROUP_FILES {
                log::error!("declare either one artifact or one bounded artifact group");
                return None;
            }
            let mut files = collections::BTreeMap::new();
            for (role, artifact) in &group.files {
                let path = substitute_all_template_variables_in_str(
                    &artifact.path,
                    &task_variables,
                    &variable_names,
                    &mut substituted_variables,
                )?;
                files.insert(role.clone(), artifact.with_resolved_path(path).log_err()?);
            }
            Some(group.with_resolved_files(files).log_err()?)
        } else {
            None
        };

        let artifact_argument_binding = self
            .artifact_argument_binding
            .as_ref()
            .map(|binding| {
                ResolvedTaskArtifactArgumentBinding::new(
                    binding,
                    resolved_artifact.as_ref(),
                    &args_with_substitutions,
                    if literal_arguments {
                        self.args.len()
                    } else {
                        0
                    },
                )
            })
            .transpose()
            .log_err()?;

        if self.artifact_argument_binding.is_some()
            && self.artifact_group_argument_binding.is_some()
        {
            log::error!("declare either a single artifact binding or an artifact group binding");
            return None;
        }
        let artifact_group_argument_binding = self
            .artifact_group_argument_binding
            .as_ref()
            .map(|binding| {
                ResolvedTaskArtifactGroupArgumentBinding::new(
                    binding,
                    resolved_artifact_group.as_ref(),
                    &args_with_substitutions,
                    if literal_arguments {
                        self.args.len()
                    } else {
                        0
                    },
                )
            })
            .transpose()
            .log_err()?;

        let task_hash = if literal_arguments {
            to_hex_hash(&(self, "literal_arguments"))
        } else {
            to_hex_hash(self)
        }
        .context("hashing task template")
        .log_err()?;
        let variables_hash = to_hex_hash(&task_variables)
            .context("hashing task variables")
            .log_err()?;
        let id = TaskId(format!("{id_base}_{task_hash}_{variables_hash}"));

        let env = {
            // Start with the project environment as the base.
            let mut env = cx.project_env.clone();

            // Extend that environment with what's defined in the TaskTemplate
            env.extend(self.env.clone());

            // Then we replace all task variables that could be set in environment variables
            let mut env = substitute_all_template_variables_in_map(
                &env,
                &task_variables,
                &variable_names,
                &mut substituted_variables,
            )?;

            // Last step: set the task variables as environment variables too
            env.extend(task_variables.into_iter().map(|(k, v)| (k, v.to_owned())));
            env
        };

        Some(ResolvedTask {
            id: id.clone(),
            substituted_variables,
            original_task: self.clone(),
            resolved_artifact,
            resolved_artifact_group,
            resolved_artifact_directory: None,
            artifact_argument_binding,
            artifact_group_argument_binding,
            resolved_label: full_label.clone(),
            resolved: SpawnInTerminal {
                required_experimental_features: self.required_experimental_features.clone(),
                id,
                cwd,
                full_label,
                label: human_readable_label,
                command_label: args_with_substitutions.iter().fold(
                    command.clone(),
                    |mut command_label, arg| {
                        command_label.push(' ');
                        command_label.push_str(arg);
                        command_label
                    },
                ),
                preserve_command_label: false,
                literal_command: literal_arguments,
                windows_direct_program: false,
                literal_prefix_args: if literal_arguments {
                    self.args.len()
                } else {
                    0
                },
                command: Some(command),
                args: args_with_substitutions,
                env,
                use_new_terminal: self.use_new_terminal,
                allow_concurrent_runs: self.allow_concurrent_runs,
                reveal: self.reveal,
                reveal_target: self.reveal_target,
                hide: self.hide,
                shell: self.shell.clone(),
                show_summary: self.show_summary,
                show_command: self.show_command,
                show_rerun: true,
                save: self.save,
            },
        })
    }

    /// Validates that all `$ZED_*` variables used in this template are known
    /// variable names, returning a vector with all of the unique unknown
    /// variables.
    ///
    /// Note that `$ZED_CUSTOM_*` variables are never considered to be invalid
    /// since those are provided dynamically by extensions.
    pub fn unknown_variables(&self) -> Vec<String> {
        let mut variables = HashSet::default();

        Self::collect_unknown_variables(&self.label, &mut variables);
        Self::collect_unknown_variables(&self.command, &mut variables);

        if !self.literal_arguments {
            self.args
                .iter()
                .for_each(|arg| Self::collect_unknown_variables(arg, &mut variables));
        }

        self.env
            .values()
            .for_each(|value| Self::collect_unknown_variables(value, &mut variables));

        if let Some(cwd) = &self.cwd {
            Self::collect_unknown_variables(cwd, &mut variables);
        }
        if let Some(artifact) = &self.artifact {
            Self::collect_unknown_variables(&artifact.path, &mut variables);
        }
        if let Some(group) = &self.artifact_group {
            for artifact in group.files.values() {
                Self::collect_unknown_variables(&artifact.path, &mut variables);
            }
        }

        variables.into_iter().collect()
    }

    fn collect_unknown_variables(template: &str, unknown: &mut HashSet<String>) {
        shellexpand::env_with_context_no_errors(template, |variable| {
            // It's possible that the variable has a default defined, which is
            // separated by a `:`, for example, `${ZED_FILE:default_value} so we
            // ensure that we're only looking at the variable name itself.
            let colon_position = variable.find(':').unwrap_or(variable.len());
            let variable_name = &variable[..colon_position];

            if variable_name.starts_with(ZED_VARIABLE_NAME_PREFIX)
                && let without_prefix = &variable_name[ZED_VARIABLE_NAME_PREFIX.len()..]
                && !without_prefix.starts_with("CUSTOM_")
                && variable_name.parse::<VariableName>().is_err()
            {
                unknown.insert(variable_name.to_string());
            }

            None::<&str>
        });
    }
}

fn literal_arguments_are_disabled(enabled: &bool) -> bool {
    !*enabled
}

const MAX_DISPLAY_VARIABLE_LENGTH: usize = 15;

fn truncate_variables(task_variables: &HashMap<String, &str>) -> HashMap<String, String> {
    task_variables
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                truncate_and_remove_front(value, MAX_DISPLAY_VARIABLE_LENGTH),
            )
        })
        .collect()
}

fn to_hex_hash(object: impl Serialize) -> anyhow::Result<String> {
    let json = serde_json_lenient::to_string(&object).context("serializing the object")?;
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    Ok(hex::encode(hasher.finalize()))
}

pub fn substitute_variables_in_str(template_str: &str, context: &TaskContext) -> Option<String> {
    let mut variable_names = HashMap::default();
    let mut substituted_variables = HashSet::default();
    let task_variables = context
        .task_variables
        .0
        .iter()
        .map(|(key, value)| {
            let key_string = key.to_string();
            if !variable_names.contains_key(&key_string) {
                variable_names.insert(key_string.clone(), key.clone());
            }
            (key_string, value.as_str())
        })
        .collect::<HashMap<_, _>>();
    substitute_all_template_variables_in_str(
        template_str,
        &task_variables,
        &variable_names,
        &mut substituted_variables,
    )
}
fn substitute_all_template_variables_in_str<A: AsRef<str>>(
    template_str: &str,
    task_variables: &HashMap<String, A>,
    variable_names: &HashMap<String, VariableName>,
    substituted_variables: &mut HashSet<VariableName>,
) -> Option<String> {
    let substituted_string = shellexpand::env_with_context(template_str, |var| {
        // Colons denote a default value in case the variable is not set. We
        // want to preserve that default, as otherwise shellexpand will
        // substitute it for us.
        let colon_position = var.find(':').unwrap_or(var.len());
        let (variable_name, default) = var.split_at(colon_position);
        if let Some(name) = task_variables.get(variable_name) {
            if let Some(substituted_variable) = variable_names.get(variable_name) {
                substituted_variables.insert(substituted_variable.clone());
            }
            // Got a task variable hit - use the variable value, ignore default
            return Ok(Some(name.as_ref().to_owned()));
        } else if variable_name.starts_with(ZED_VARIABLE_NAME_PREFIX) {
            // Unknown ZED variable - use default if available
            if !default.is_empty() {
                // Strip the colon and return the default value
                return Ok(Some(default[1..].to_owned()));
            } else {
                bail!("Unknown variable name: {variable_name}");
            }
        }
        // This is an unknown variable.
        // We should not error out, as they may come from user environment (e.g.
        // $PATH). That means that the variable substitution might not be
        // perfect. If there's a default, we need to return the string verbatim
        // as otherwise shellexpand will apply that default for us.
        if !default.is_empty() {
            return Ok(Some(format!("${{{var}}}")));
        }

        // Else we can just return None and that variable will be left as is.
        Ok(None)
    })
    .ok()?;

    Some(substituted_string.into_owned())
}

fn substitute_all_template_variables_in_vec(
    template_strs: &[String],
    task_variables: &HashMap<String, &str>,
    variable_names: &HashMap<String, VariableName>,
    substituted_variables: &mut HashSet<VariableName>,
) -> Option<Vec<String>> {
    let mut expanded = Vec::with_capacity(template_strs.len());
    for variable in template_strs {
        let new_value = substitute_all_template_variables_in_str(
            variable,
            task_variables,
            variable_names,
            substituted_variables,
        )?;

        if !new_value.is_empty() || variable.is_empty() {
            expanded.push(new_value);
        }
    }

    Some(expanded)
}

pub fn substitute_variables_in_map(
    keys_and_values: &HashMap<String, String>,
    context: &TaskContext,
) -> Option<HashMap<String, String>> {
    let mut variable_names = HashMap::default();
    let mut substituted_variables = HashSet::default();
    let task_variables = context
        .task_variables
        .0
        .iter()
        .map(|(key, value)| {
            let key_string = key.to_string();
            if !variable_names.contains_key(&key_string) {
                variable_names.insert(key_string.clone(), key.clone());
            }
            (key_string, value.as_str())
        })
        .collect::<HashMap<_, _>>();
    substitute_all_template_variables_in_map(
        keys_and_values,
        &task_variables,
        &variable_names,
        &mut substituted_variables,
    )
}
fn substitute_all_template_variables_in_map(
    keys_and_values: &HashMap<String, String>,
    task_variables: &HashMap<String, &str>,
    variable_names: &HashMap<String, VariableName>,
    substituted_variables: &mut HashSet<VariableName>,
) -> Option<HashMap<String, String>> {
    let mut new_map: HashMap<String, String> = Default::default();
    for (key, value) in keys_and_values {
        let new_value = substitute_all_template_variables_in_str(
            value,
            task_variables,
            variable_names,
            substituted_variables,
        )?;
        let new_key = substitute_all_template_variables_in_str(
            key,
            task_variables,
            variable_names,
            substituted_variables,
        )?;
        new_map.insert(new_key, new_value);
    }

    Some(new_map)
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Cow,
        path::{Path, PathBuf},
    };

    use crate::{TaskVariables, VariableName};

    use super::*;

    const TEST_ID_BASE: &str = "test_base";

    #[test]
    fn experimental_requirements_preserve_ordinary_identity_and_rerun() {
        let ordinary = TaskTemplate {
            label: "ordinary".into(),
            command: "echo".into(),
            ..Default::default()
        };
        let ordinary_json = serde_json::to_value(&ordinary).unwrap();
        assert!(
            ordinary_json
                .get("required_experimental_features")
                .is_none()
        );
        let context = TaskContext::default();
        let initial = ordinary.resolve_task(TEST_ID_BASE, &context).unwrap();
        let replay: TaskTemplate = serde_json::from_value(ordinary_json).unwrap();
        assert_eq!(
            initial.id,
            replay.resolve_task(TEST_ID_BASE, &context).unwrap().id
        );
        let selected = TaskTemplate {
            required_experimental_features: vec![
                "rust-native-profiling-experimental".into(),
                "rust-named-cargo-presets-experimental".into(),
            ],
            ..ordinary
        };
        let encoded = serde_json::to_string(&selected).unwrap();
        let replay: TaskTemplate = serde_json::from_str(&encoded).unwrap();
        let initial = selected.resolve_task(TEST_ID_BASE, &context).unwrap();
        let cached = initial.clone();
        let template = replay.resolve_task(TEST_ID_BASE, &context).unwrap();
        assert_eq!(initial.id, cached.id);
        assert_eq!(initial.id, template.id);
        assert_eq!(
            initial.resolved.required_experimental_features,
            selected.required_experimental_features
        );
        assert_eq!(
            template.resolved.required_experimental_features,
            selected.required_experimental_features
        );
        let decoded = SpawnInTerminal::from_proto(initial.resolved.to_proto());
        assert_eq!(
            decoded.required_experimental_features,
            selected.required_experimental_features
        );
        for invalid in [
            vec!["unknown".to_string()],
            vec!["rust-coverage-experimental".to_string(); 2],
            vec!["rust-coverage-experimental".to_string(); 5],
        ] {
            let rejected = TaskTemplate {
                required_experimental_features: invalid,
                ..selected.clone()
            };
            assert!(rejected.resolve_task(TEST_ID_BASE, &context).is_none());
        }
    }

    #[test]
    fn task_stdout_report_persists_and_survives_cached_and_template_rerun() {
        let mut template = group_binding_template("target/native capture/capture.json");
        template.literal_arguments = true;
        template.stdout_report = Some(crate::TaskStdoutReport {
            format: crate::TaskStdoutReportFormat::CargoBuildJson,
            max_bytes: 2 * 1024 * 1024,
            native_profile: None,
        });
        let encoded = serde_json::to_string(&template).expect("persist template");
        let restored: TaskTemplate = serde_json::from_str(&encoded).expect("load template");
        assert_eq!(restored, template);
        let resolved = restored
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("resolve");
        let original_id = resolved.id.clone();
        let cached = resolved
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("cached rerun binding");
        let reevaluated = resolved
            .original_task()
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("template rerun resolution")
            .bind_artifact_invocation("2233445566778899aabbccddeeff0011")
            .expect("template rerun binding");
        for rerun in [&cached, &reevaluated] {
            assert_eq!(rerun.stdout_report(), template.stdout_report.as_ref());
            assert_eq!(rerun.id, original_id);
            assert_eq!(rerun.original_task(), &template);
            assert_eq!(rerun.resolved.command, resolved.resolved.command);
            assert_eq!(rerun.resolved.env, resolved.resolved.env);
            assert_eq!(
                rerun.resolved.literal_prefix_args,
                resolved.resolved.literal_prefix_args
            );
        }
        assert_ne!(
            cached.resolved_artifact_group(),
            reevaluated.resolved_artifact_group()
        );
        assert_eq!(
            resolved
                .resolved_artifact_group()
                .expect("original group")
                .files
                .get("profile")
                .expect("primary artifact")
                .path,
            "target/native capture/capture.json"
        );
    }

    #[test]
    fn task_stdout_report_rejects_unknown_formats_credentials_and_invalid_limits() {
        for value in [
            serde_json::json!({"format":"arbitrary","max_bytes":1048576}),
            serde_json::json!({"format":"cargo_build_json","max_bytes":1048576,"success":true}),
            serde_json::json!({"format":"cargo_build_json","max_bytes":1048576,"token":"caller"}),
            serde_json::json!({"format":"cargo_build_json","max_bytes":1048576,"bytes":"caller"}),
        ] {
            assert!(serde_json::from_value::<crate::TaskStdoutReport>(value).is_err());
        }
        for limit in [
            0,
            crate::MIN_TASK_STDOUT_REPORT_BYTES - 1,
            crate::MAX_TASK_STDOUT_REPORT_BYTES + 1,
            usize::MAX,
        ] {
            let template = TaskTemplate {
                label: "Build".into(),
                command: "cargo".into(),
                stdout_report: Some(crate::TaskStdoutReport {
                    format: crate::TaskStdoutReportFormat::CargoBuildJson,
                    max_bytes: limit,
                    native_profile: None,
                }),
                ..Default::default()
            };
            assert!(
                template
                    .resolve_task(TEST_ID_BASE, &TaskContext::default())
                    .is_none()
            );
        }
        for limit in [
            crate::MIN_TASK_STDOUT_REPORT_BYTES,
            crate::MAX_TASK_STDOUT_REPORT_BYTES,
        ] {
            let template = TaskTemplate {
                label: "Build".into(),
                command: "cargo".into(),
                stdout_report: Some(crate::TaskStdoutReport {
                    format: crate::TaskStdoutReportFormat::CargoBuildJson,
                    max_bytes: limit,
                    native_profile: None,
                }),
                ..Default::default()
            };
            assert_eq!(
                template
                    .resolve_task(TEST_ID_BASE, &TaskContext::default())
                    .expect("bounded descriptor")
                    .stdout_report()
                    .expect("descriptor")
                    .max_bytes,
                limit
            );
        }
    }

    #[test]
    fn native_runtime_preparation_requires_validated_continuation_and_existing_limits() {
        let mut report = crate::TaskStdoutReport {
            format: crate::TaskStdoutReportFormat::CargoRuntimePreparationJson,
            max_bytes: crate::MAX_TASK_STDOUT_REPORT_BYTES,
            native_profile: None,
        };
        assert!(report.validate().is_err());
        report.native_profile = Some(crate::TaskNativeProfileContinuation {
            collector_path: "/opt/local tools/samply".into(),
            profile_path: "target/native captures/capture.json".into(),
            workload_arguments: vec![
                "argument with spaces".into(),
                "$literal;*".into(),
                "".into(),
            ],
        });
        report.validate().expect("bounded native runtime intent");
        for limit in [
            crate::MIN_TASK_STDOUT_REPORT_BYTES - 1,
            crate::MAX_TASK_STDOUT_REPORT_BYTES + 1,
        ] {
            report.max_bytes = limit;
            assert!(report.validate().is_err());
        }
        report.max_bytes = crate::MAX_TASK_STDOUT_REPORT_BYTES;
        report
            .native_profile
            .as_mut()
            .expect("continuation")
            .collector_path = "relative/samply".into();
        assert!(report.validate().is_err());
        let legacy = crate::TaskStdoutReport {
            format: crate::TaskStdoutReportFormat::CargoBuildJson,
            max_bytes: crate::MIN_TASK_STDOUT_REPORT_BYTES,
            native_profile: None,
        };
        legacy.validate().expect("ordinary legacy Cargo report");
    }

    #[test]
    fn native_runtime_preparation_history_and_rerun_preserve_intent_without_authority() {
        let template = TaskTemplate {
            label: "Prepare native profile".into(),
            command: "/opt/local tools/cargo".into(),
            args: vec![
                "run".into(),
                "--offline".into(),
                "--message-format=json".into(),
                "--bin".into(),
                "selected".into(),
                "--".into(),
                "argument with spaces".into(),
                "$literal;*".into(),
                "".into(),
            ],
            literal_arguments: true,
            cwd: Some("/workspace".into()),
            stdout_report: Some(crate::TaskStdoutReport {
                format: crate::TaskStdoutReportFormat::CargoRuntimePreparationJson,
                max_bytes: crate::MAX_TASK_STDOUT_REPORT_BYTES,
                native_profile: Some(crate::TaskNativeProfileContinuation {
                    collector_path: "/opt/local tools/samply".into(),
                    profile_path: "target/native captures/capture.json".into(),
                    workload_arguments: vec![
                        "argument with spaces".into(),
                        "$literal;*".into(),
                        "".into(),
                    ],
                }),
            }),
            ..Default::default()
        };
        let encoded = serde_json::to_value(&template).expect("persist runtime intent");
        assert_eq!(
            encoded["stdout_report"]["format"],
            "cargo_runtime_preparation_json"
        );
        for key in [
            "environment",
            "runtime",
            "token",
            "endpoint",
            "pid",
            "receipt",
            "success",
        ] {
            assert!(encoded["stdout_report"].get(key).is_none());
            assert!(
                encoded["stdout_report"]["native_profile"]
                    .get(key)
                    .is_none()
            );
            let mut fabricated = encoded.clone();
            fabricated["stdout_report"][key] = serde_json::json!("fabricated authority");
            assert!(serde_json::from_value::<TaskTemplate>(fabricated).is_err());
        }
        let restored: TaskTemplate = serde_json::from_value(encoded).expect("restore intent");
        let resolved = restored
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("resolve intent");
        let cached = resolved
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("cached recipe");
        let reevaluated = resolved
            .original_task()
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("template recipe");
        for rerun in [&cached, &reevaluated] {
            assert_eq!(rerun.stdout_report(), restored.stdout_report.as_ref());
            assert_eq!(rerun.original_task(), &restored);
            assert_eq!(rerun.id, resolved.id);
            assert_eq!(rerun.resolved, resolved.resolved);
            assert!(rerun.resolved_artifact_group().is_none());
        }
    }

    #[test]
    fn native_profile_continuation_persists_literal_intent_without_cached_authority() {
        let report = crate::TaskStdoutReport {
            format: crate::TaskStdoutReportFormat::CargoBuildJson,
            max_bytes: crate::MAX_TASK_STDOUT_REPORT_BYTES,
            native_profile: Some(crate::TaskNativeProfileContinuation {
                collector_path: "/opt/local tools/samply".into(),
                profile_path: "target/native captures/capture.json".into(),
                workload_arguments: vec!["$HOME".into(), "two words".into(), "--port=0".into()],
            }),
        };
        let template = TaskTemplate {
            label: "Build then profile".into(),
            command: "/opt/local tools/cargo".into(),
            args: vec!["build".into(), "--bin".into(), "selected".into()],
            literal_arguments: true,
            cwd: Some("/workspace".into()),
            stdout_report: Some(report.clone()),
            ..Default::default()
        };
        let encoded = serde_json::to_value(&template).expect("persist intent");
        assert_eq!(
            encoded["stdout_report"]["native_profile"]["profile_path"],
            "target/native captures/capture.json"
        );
        for key in [
            "success",
            "token",
            "version",
            "executable",
            "receipt",
            "run_id",
        ] {
            assert!(
                encoded["stdout_report"]["native_profile"]
                    .get(key)
                    .is_none()
            );
        }
        let restored: TaskTemplate = serde_json::from_value(encoded).expect("restore intent");
        let resolved = restored
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("resolve original recipe");
        let cached = resolved
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("fresh cached run");
        let reevaluated = resolved
            .original_task()
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("reevaluate original recipe")
            .bind_artifact_invocation("2233445566778899aabbccddeeff0011")
            .expect("fresh template run");
        for rerun in [&cached, &reevaluated] {
            assert_eq!(rerun.stdout_report(), Some(&report));
            assert_eq!(rerun.id, resolved.id);
            assert_eq!(rerun.resolved, resolved.resolved);
            assert_eq!(rerun.original_task(), &restored);
            assert!(rerun.resolved_artifact_group().is_none());
        }
        let mut report_only = template;
        report_only
            .stdout_report
            .as_mut()
            .expect("report")
            .native_profile = None;
        let report_only = report_only
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("ordinary report-only intent");
        assert_ne!(report_only.id, resolved.id);
        assert_eq!(report_only.resolved.command, resolved.resolved.command);
        assert_eq!(report_only.resolved.args, resolved.resolved.args);
        assert_eq!(report_only.resolved.env, resolved.resolved.env);
        assert!(serde_json::to_value(report_only.original_task()).expect("serialize report-only")
            ["stdout_report"].get("native_profile").is_none());
    }

    #[test]
    fn native_profile_continuation_rejects_credentials_unresolved_paths_and_unbounded_arguments() {
        let valid = crate::TaskNativeProfileContinuation {
            collector_path: "/opt/tools/samply".into(),
            profile_path: "target/native/capture.json".into(),
            workload_arguments: vec!["literal $HOME".into()],
        };
        valid.validate().expect("bounded literal continuation");
        for key in [
            "success",
            "token",
            "version",
            "executable",
            "receipt",
            "run_id",
        ] {
            let mut value = serde_json::to_value(&valid).expect("encode data");
            value[key] = serde_json::json!("caller");
            assert!(serde_json::from_value::<crate::TaskNativeProfileContinuation>(value).is_err());
        }
        let mut invalid = Vec::new();
        for path in [
            "samply",
            "/opt/../tools/samply",
            "/opt/$ZED_FILE/samply",
            "/opt/${ZED_FILE}/samply",
            "/opt/samply\0",
            "/opt/samply\n",
        ] {
            let mut value = valid.clone();
            value.collector_path = path.into();
            invalid.push(value);
        }
        for path in [
            "/tmp/capture.json",
            "../capture.json",
            "target//capture.json",
            "target/./capture.json",
            "target/capture.syms.json",
            "target/capture.json.gz",
            "target/$ZED_FILE.json",
            "target/${ZED_FILE}.json",
            "-capture.json",
        ] {
            let mut value = valid.clone();
            value.profile_path = path.into();
            invalid.push(value);
        }
        let mut value = valid.clone();
        value.collector_path = format!("/{}", "x".repeat(4096));
        invalid.push(value);
        let mut value = valid.clone();
        value.profile_path = format!("{}.json", "x".repeat(1020));
        invalid.push(value);
        for arguments in [
            vec!["x".into(); 129],
            vec!["x".repeat(4097)],
            vec!["x".repeat(4096); 9],
            vec!["invalid\0argument".into()],
        ] {
            let mut value = valid.clone();
            value.workload_arguments = arguments;
            invalid.push(value);
        }
        for continuation in invalid {
            let template = TaskTemplate {
                label: "Invalid native intent".into(),
                command: "/opt/tools/cargo".into(),
                stdout_report: Some(crate::TaskStdoutReport {
                    format: crate::TaskStdoutReportFormat::CargoBuildJson,
                    max_bytes: crate::MIN_TASK_STDOUT_REPORT_BYTES,
                    native_profile: Some(continuation),
                }),
                ..Default::default()
            };
            assert!(
                template
                    .resolve_task(TEST_ID_BASE, &TaskContext::default())
                    .is_none()
            );
        }
    }

    #[test]
    fn task_stdout_report_absence_preserves_ordinary_serialization_and_launch() {
        let template: TaskTemplate = serde_json::from_value(serde_json::json!({
            "label":"Ordinary", "command":"echo", "args":["$HOME"],
            "env":{"EXAMPLE":"value"}, "allow_concurrent_runs":false, "use_new_terminal":false
        }))
        .expect("ordinary template");
        let serialized = serde_json::to_value(&template).expect("serialize");
        assert!(serialized.get("stdout_report").is_none());
        let resolved = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("ordinary task");
        assert!(resolved.stdout_report().is_none());
        let legacy_hash = to_hex_hash(&template).expect("legacy hash");
        assert!(
            resolved
                .id
                .0
                .starts_with(&format!("{TEST_ID_BASE}_{legacy_hash}_"))
        );
        let mut requested = template;
        requested.stdout_report = Some(crate::TaskStdoutReport {
            format: crate::TaskStdoutReportFormat::CargoBuildJson,
            max_bytes: 1048576,
            native_profile: None,
        });
        let requested = requested
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("requested data");
        assert_ne!(requested.id, resolved.id);
        assert_eq!(requested.resolved.command, resolved.resolved.command);
        assert_eq!(requested.resolved.args, resolved.resolved.args);
        assert_eq!(requested.resolved.env, resolved.resolved.env);
        assert_eq!(requested.resolved.shell, resolved.resolved.shell);
        assert_eq!(
            requested.resolved.literal_prefix_args,
            resolved.resolved.literal_prefix_args
        );
        assert_eq!(
            requested.resolved.allow_concurrent_runs,
            resolved.resolved.allow_concurrent_runs
        );
        assert_eq!(
            requested.resolved.use_new_terminal,
            resolved.resolved.use_new_terminal
        );
    }

    fn artifact_binding_template(path: &str) -> TaskTemplate {
        TaskTemplate {
            label: "Capture".into(),
            command: "collector".into(),
            args: vec!["--output-path".into(), path.into(), "suffix=$HOME".into()],
            artifact: Some(TaskArtifact {
                path: path.into(),
                kind: crate::TaskArtifactKind::Data,
                max_bytes: crate::MAX_TASK_ARTIFACT_BYTES,
            }),
            artifact_argument_binding: Some(TaskArtifactArgumentBinding { argument_index: 1 }),
            ..Default::default()
        }
    }

    fn group_binding_template(path: &str) -> TaskTemplate {
        let symbols = std::path::Path::new(path).with_extension("syms.json");
        let symbols = symbols.to_str().expect("text fixture path");
        TaskTemplate {
            label: "Capture pair".into(),
            command: "collector".into(),
            args: vec!["--output".into(), path.into(), "suffix=$HOME".into()],
            artifact_group: Some(TaskArtifactGroup {
                format: "paired-data-v1".into(),
                files: collections::BTreeMap::from_iter([
                    (
                        "profile".into(),
                        TaskArtifact {
                            path: path.into(),
                            kind: crate::TaskArtifactKind::Data,
                            max_bytes: crate::MAX_TASK_ARTIFACT_BYTES,
                        },
                    ),
                    (
                        "symbols".into(),
                        TaskArtifact {
                            path: symbols.into(),
                            kind: crate::TaskArtifactKind::Data,
                            max_bytes: 16 * 1024 * 1024,
                        },
                    ),
                ]),
            }),
            artifact_group_argument_binding: Some(TaskArtifactGroupArgumentBinding {
                argument_index: 1,
                role: "profile".into(),
                related: vec![crate::TaskArtifactRelatedArgumentBinding {
                    role: "symbols".into(),
                    extension: "syms.json".into(),
                    argument_index: None,
                }],
            }),
            ..Default::default()
        }
    }

    #[cfg(unix)]
    #[test]
    fn task_artifact_directory_keeps_nested_execution_and_pair_destination_distinct() {
        let directory = std::path::Path::new("/owned workspace");
        let execution = directory.join("nested/member");
        let template = TaskTemplate {
            cwd: Some(execution.to_str().expect("execution cwd").into()),
            literal_arguments: true,
            ..group_binding_template("target/native captures/capture.json")
        };
        let original = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("original pair");
        let anchored = original
            .with_artifact_directory(directory)
            .expect("reviewed anchor data");
        let first = anchored
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("first pair");
        let second = first
            .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
            .expect("cached pair");
        for (bound, id) in [
            (&first, "112233445566778899aabbccddeeff00"),
            (&second, "99887766554433221100ffeeddccbbaa"),
        ] {
            let group = bound.resolved_artifact_group().expect("declared pair");
            let relative = format!("target/native captures/capture-{id}.json");
            assert_eq!(group.files["profile"].path, relative);
            assert_eq!(
                group.files["symbols"].path,
                format!("target/native captures/capture-{id}.syms.json")
            );
            assert_eq!(
                bound.resolved.args.get(1),
                Some(
                    &directory
                        .join(&relative)
                        .to_str()
                        .expect("bound output")
                        .to_owned()
                )
            );
            assert_eq!(bound.resolved.cwd.as_deref(), Some(execution.as_path()));
            assert_eq!(bound.artifact_directory(), Some(directory));
            assert_eq!(bound.resolved.args.get(2), Some(&template.args[2]));
            assert_eq!(bound.id, original.id);
            assert_eq!(bound.original_task(), &template);
        }
        let ordinary = original
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("default cwd-relative pair");
        assert_eq!(ordinary.artifact_directory(), None);
        assert_eq!(
            ordinary.resolved.args.get(1),
            Some(
                &ordinary
                    .resolved_artifact_group()
                    .expect("ordinary pair")
                    .files["profile"]
                    .path
            )
        );
        assert_eq!(original.resolved.args, template.args);
    }

    #[cfg(unix)]
    #[test]
    fn task_artifact_directory_requires_fresh_template_review_and_literal_exact_paths() {
        let template = TaskTemplate {
            literal_arguments: true,
            ..group_binding_template("target/capture.json")
        };
        let original = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("pair task");
        let directory = std::path::Path::new("/owned workspace");
        let first = original
            .with_artifact_directory(directory)
            .expect("first anchor")
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("first binding");
        let persisted: TaskTemplate = serde_json::from_value(
            serde_json::to_value(first.original_task()).expect("persistent template"),
        )
        .expect("restored template");
        let fresh = persisted
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("fresh template resolution");
        assert_eq!(
            fresh.artifact_directory(),
            None,
            "serialized data carries no reviewed host anchor"
        );
        let second = fresh
            .with_artifact_directory(directory)
            .expect("fresh host review data")
            .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
            .expect("fresh pair");
        assert_eq!(
            second,
            first
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .expect("cached fresh pair")
        );
        assert!(
            first
                .with_artifact_directory(std::path::Path::new("/other host"))
                .is_err()
        );
        for path in [
            "relative",
            "/owned/../elsewhere",
            "/owned/./elsewhere",
            "/owned\ncontrol",
        ] {
            assert!(
                original
                    .with_artifact_directory(std::path::Path::new(path))
                    .is_err()
            );
        }
        assert!(
            original
                .with_artifact_directory(std::path::Path::new(&format!("/{}", "x".repeat(4096))))
                .is_err()
        );
        let mut changed = first.clone();
        changed.resolved.args[1] =
            "/other host/target/capture-112233445566778899aabbccddeeff00.json".into();
        assert!(
            changed
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .is_err()
        );
        let mut changed = first.clone();
        changed
            .resolved_artifact_group
            .as_mut()
            .expect("current pair")
            .files
            .get_mut("profile")
            .expect("profile role")
            .path = "../escaped.json".into();
        assert!(
            changed
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .is_err()
        );
        let nonliteral = group_binding_template("target/capture.json")
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("legacy shell task");
        assert!(nonliteral.with_artifact_directory(directory).is_err());
        assert!(first.bind_artifact_invocation("not-a-uuid").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn task_artifact_directory_uses_shared_single_and_related_argument_binding() {
        let directory = std::path::Path::new("/owned workspace");
        let template = TaskTemplate {
            literal_arguments: true,
            ..artifact_binding_template("target/report.json")
        };
        let single = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("single artifact")
            .with_artifact_directory(directory)
            .expect("single anchor")
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("single binding");
        assert_eq!(
            single.resolved.args.get(1).map(String::as_str),
            Some("/owned workspace/target/report-112233445566778899aabbccddeeff00.json")
        );
        assert_eq!(
            single
                .resolved_artifact()
                .expect("relative declaration")
                .path,
            "target/report-112233445566778899aabbccddeeff00.json"
        );
        let mut template = group_binding_template("target/capture.json");
        template.literal_arguments = true;
        template.args.insert(2, "target/capture.syms.json".into());
        template
            .artifact_group_argument_binding
            .as_mut()
            .expect("group binding")
            .related[0]
            .argument_index = Some(2);
        let group = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("explicit symbols argument")
            .with_artifact_directory(directory)
            .expect("group anchor")
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("both bound arguments");
        assert_eq!(
            group.resolved.args.get(1).map(String::as_str),
            Some("/owned workspace/target/capture-112233445566778899aabbccddeeff00.json")
        );
        assert_eq!(
            group.resolved.args.get(2).map(String::as_str),
            Some("/owned workspace/target/capture-112233445566778899aabbccddeeff00.syms.json")
        );
        assert_eq!(group.resolved.args.get(3), Some(&template.args[3]));
        let mut changed = group;
        changed.resolved.args[2] = "/owned workspace/target/wrong.syms.json".into();
        assert!(
            changed
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .is_err()
        );
        let freshly_unbound = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("unbound pair");
        let mut no_command_unbound = freshly_unbound.clone();
        no_command_unbound.resolved.command = None;
        no_command_unbound.resolved.preserve_command_label = true;
        assert!(
            no_command_unbound
                .with_artifact_directory(directory)
                .is_err(),
            "preserved labels cannot disguise a missing executable"
        );
        assert_eq!(freshly_unbound.artifact_directory(), None);
        let ordinary = TaskTemplate {
            label: "Ordinary".into(),
            command: "program".into(),
            ..Default::default()
        }
        .resolve_task(TEST_ID_BASE, &TaskContext::default())
        .expect("ordinary task");
        assert!(ordinary.with_artifact_directory(directory).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn task_artifact_directory_bounds_implicit_companion_before_rebinding() {
        let invocation = "112233445566778899aabbccddeeff00";
        let template = TaskTemplate {
            literal_arguments: true,
            ..group_binding_template("target/capture.json")
        };
        let original = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("bounded original pair");
        let bound_companion = format!("target/capture-{invocation}.syms.json");
        let directory_bytes = 4096 - 1 - bound_companion.len();
        let exact = std::path::PathBuf::from(format!("/{}", "a".repeat(directory_bytes - 1)));
        let anchored = original
            .with_artifact_directory(&exact)
            .expect("bounded anchor data");
        let bound = anchored
            .bind_artifact_invocation(invocation)
            .expect("inclusive pair path bound");
        let group = bound.resolved_artifact_group().expect("bound pair");
        assert_eq!(
            exact
                .join(&group.files["symbols"].path)
                .to_str()
                .expect("companion path")
                .len(),
            4096
        );
        assert_eq!(
            bound.resolved.args.get(1).expect("profile operand").len(),
            4091
        );
        let overflow = std::path::PathBuf::from(format!("/{}", "a".repeat(directory_bytes)));
        let oversized = original
            .with_artifact_directory(&overflow)
            .expect("original paths fit before UUID binding");
        let before = oversized.clone();
        assert!(
            oversized.bind_artifact_invocation(invocation).is_err(),
            "implicit companion must not exceed the bound even when the output argv fits"
        );
        assert_eq!(
            oversized, before,
            "refused binding cannot mutate the cached task"
        );
        assert_eq!(original.artifact_directory(), None);
        let joined_original_companion_bytes = template
            .artifact_group
            .as_ref()
            .expect("original pair")
            .files["symbols"]
            .path
            .len()
            + 1;
        let oversized_initial_anchor = std::path::PathBuf::from(format!(
            "/{}",
            "a".repeat(4096 - joined_original_companion_bytes)
        ));
        assert!(
            original
                .with_artifact_directory(&oversized_initial_anchor)
                .is_err(),
            "every implicit companion is bounded before the task clone"
        );
    }

    #[cfg(unix)]
    #[test]
    fn task_artifact_directory_reaches_real_shell_as_literal_without_changing_execution() {
        let execution = std::env::temp_dir().join(format!(
            "task-artifact-directory-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("fixture clock")
                .as_nanos(),
        ));
        std::fs::create_dir(&execution).expect("owned execution fixture");
        let execution = std::fs::canonicalize(execution).expect("actual execution directory");
        let _cleanup = util::defer({
            let execution = execution.clone();
            move || std::fs::remove_dir_all(execution).expect("remove owned execution fixture")
        });
        let sentinel = execution.join("injected");
        let artifact_directory = execution.join(format!(
            "workspace '$ARTIFACT_DIRECTORY_VALUE' $(touch {}) 日本語",
            sentinel.display(),
        ));
        let output_path = "target/native captures/capture.json";
        let mut template = group_binding_template(output_path);
        template.command = "/bin/sh".into();
        template.args = vec![
            "-c".into(),
            "printf '%s\\0' \"$PWD\" \"$@\"".into(),
            "actual-argument-receiver".into(),
            output_path.into(),
            "suffix=$ARTIFACT_DIRECTORY_VALUE".into(),
        ];
        template
            .artifact_group_argument_binding
            .as_mut()
            .expect("output binding")
            .argument_index = 3;
        template.cwd = Some(execution.to_str().expect("execution path").into());
        template.literal_arguments = true;
        let original = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("literal pair declaration");
        let anchored = original
            .with_artifact_directory(&artifact_directory)
            .expect("artifact anchor data");
        for invocation in [
            "112233445566778899aabbccddeeff00",
            "99887766554433221100ffeeddccbbaa",
        ] {
            let bound = anchored
                .bind_artifact_invocation(invocation)
                .expect("fresh invocation");
            let group = bound
                .resolved_artifact_group()
                .expect("relative pair declaration");
            let expected_output = artifact_directory
                .join(&group.files["profile"].path)
                .to_str()
                .expect("expected literal output")
                .to_owned();
            let expected_arguments = [expected_output, template.args[4].clone()];
            assert_eq!(bound.resolved.args.get(3), expected_arguments.first());
            assert_eq!(bound.resolved.cwd.as_deref(), Some(execution.as_path()));
            let (program, arguments) =
                util::shell_builder::ShellBuilder::new(&Shell::Program("/bin/sh".into()), false)
                    .non_interactive()
                    .build_with_literal_prefix(
                        bound.resolved.command.clone(),
                        &bound.resolved.args,
                        bound.resolved.literal_prefix_args,
                    );
            let output = futures::executor::block_on(
                util::command::new_command(program)
                    .args(arguments)
                    .current_dir(bound.resolved.cwd.as_ref().expect("execution cwd"))
                    .env("ARTIFACT_DIRECTORY_VALUE", "must-not-expand")
                    .output(),
            )
            .expect("actual shell/printf receiver");
            assert!(
                output.status.success(),
                "actual receiver failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let expected =
                std::iter::once(execution.to_str().expect("actual child cwd").to_owned())
                    .chain(expected_arguments)
                    .collect::<Vec<_>>()
                    .iter()
                    .flat_map(|argument| {
                        argument
                            .as_bytes()
                            .iter()
                            .copied()
                            .chain(std::iter::once(0))
                    })
                    .collect::<Vec<_>>();
            assert_eq!(
                output.stdout, expected,
                "the exact anchored operand must reach the actual child"
            );
            assert!(
                !sentinel.exists(),
                "anchor data must not execute command substitution"
            );
            assert_eq!(bound.original_task(), &template);
            assert_eq!(bound.id, original.id);
        }
    }

    #[test]
    fn task_persisted_literal_arguments_keep_declared_mode_and_group_binding() {
        let legacy: TaskTemplate =
            serde_json::from_value(serde_json::json!({"label": "Legacy", "command": "collector"}))
                .expect("legacy template");
        assert!(!legacy.literal_arguments);
        assert!(
            serde_json::to_value(&legacy)
                .expect("legacy serialization")
                .get("literal_arguments")
                .is_none()
        );
        let normal_hash = to_hex_hash(&legacy).expect("legacy hash");
        assert!(
            legacy
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .expect("legacy task")
                .id
                .0
                .starts_with(&format!("{TEST_ID_BASE}_{normal_hash}_"))
        );
        let mut template = group_binding_template("target/native captures/capture.json");
        template.literal_arguments = true;
        template.args.extend([
            "$ZED_UNAVAILABLE_NATIVE_ARG".into(),
            "' ; $HOME $(noop) 日本語".into(),
            "".into(),
        ]);
        let encoded = serde_json::to_string(&TaskTemplates(vec![template.clone()]))
            .expect("persist templates");
        let TaskTemplates(mut restored) = serde_json::from_str(&encoded).expect("load templates");
        let restored = restored.pop().expect("one persisted template");
        assert_eq!(restored, template);
        let original = restored
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("ordinary persisted-template resolution uses explicit literal mode");
        assert_eq!(original.resolved.args, template.args);
        assert_eq!(original.resolved.literal_prefix_args, template.args.len());
        assert_eq!(
            original,
            restored
                .resolve_task_with_literal_arguments(TEST_ID_BASE, &TaskContext::default())
                .expect("forced mode remains equivalent")
        );
        let first = original
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("first pair");
        let cached = first
            .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
            .expect("cached pair");
        let reevaluated = first
            .original_task()
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("ordinary rerun resolver")
            .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
            .expect("reevaluated pair");
        assert_eq!(cached, reevaluated);
        assert_eq!(cached.id, original.id);
        assert_eq!(cached.original_task(), &template);
        assert_eq!(cached.resolved.args.get(3), Some(&template.args[3]));
        assert_eq!(cached.resolved.args.get(4), Some(&template.args[4]));
        let mut bad = serde_json::to_value(&template).expect("template value");
        bad["literal_arguments"] = serde_json::json!("true");
        assert!(serde_json::from_value::<TaskTemplate>(bad).is_err());
    }

    #[test]
    fn task_persisted_literal_arguments_leave_non_argument_variables_live() {
        let mut variables = TaskVariables::default();
        variables.insert(VariableName::WorktreeRoot, "/owned project".into());
        variables.insert(VariableName::File, "changed.txt".into());
        let context = TaskContext {
            task_variables: variables,
            ..Default::default()
        };
        let template = TaskTemplate {
            label: "Capture $ZED_FILE".into(),
            command: "$ZED_WORKTREE_ROOT/collector".into(),
            cwd: Some("$ZED_WORKTREE_ROOT".into()),
            args: vec!["$ZED_FILE".into(), "$ZED_UNAVAILABLE_NATIVE_ARG".into()],
            literal_arguments: true,
            ..Default::default()
        };
        assert!(template.unknown_variables().is_empty());
        let task = template
            .resolve_task(TEST_ID_BASE, &context)
            .expect("explicit literal resolution");
        assert_eq!(task.resolved.args, template.args);
        assert_eq!(task.resolved.literal_prefix_args, 2);
        assert_eq!(
            task.resolved.command.as_deref(),
            Some("/owned project/collector")
        );
        assert_eq!(task.resolved_label, "Capture changed.txt");
        assert_eq!(
            task.resolved.cwd.as_deref(),
            Some(std::path::Path::new("/owned project"))
        );
        let unknown = TaskTemplate {
            command: "$ZED_UNKNOWN_COMMAND".into(),
            ..template.clone()
        };
        assert!(
            unknown
                .unknown_variables()
                .contains(&"ZED_UNKNOWN_COMMAND".into())
        );
        assert!(unknown.resolve_task(TEST_ID_BASE, &context).is_none());
        let unsafe_argument = TaskTemplate {
            args: vec!["invalid\0argument".into()],
            ..template
        };
        assert!(
            unsafe_argument
                .resolve_task(TEST_ID_BASE, &context)
                .is_none()
        );
    }

    #[test]
    fn task_artifact_group_argument_binding_json_is_optional_and_checked() {
        let legacy = TaskTemplate {
            label: "Legacy".into(),
            command: "collector".into(),
            ..Default::default()
        };
        let value = serde_json::to_value(&legacy).expect("legacy value");
        assert!(value.get("artifact_group_argument_binding").is_none());
        let legacy_hash = to_hex_hash(&legacy).expect("legacy hash");
        assert!(
            legacy
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .expect("legacy task")
                .id
                .0
                .starts_with(&format!("{TEST_ID_BASE}_{legacy_hash}_"))
        );
        let template = group_binding_template("target/capture.json");
        let value = serde_json::to_value(&template).expect("group value");
        assert!(
            value["artifact_group_argument_binding"]["related"][0]
                .get("argument_index")
                .is_none()
        );
        assert_eq!(
            serde_json::from_value::<TaskTemplate>(value.clone()).expect("roundtrip"),
            template
        );
        for binding in [
            serde_json::json!({"argument_index": 1, "role": "profile"}),
            serde_json::json!({"argument_index": -1, "role": "profile", "related": []}),
            serde_json::json!({"argument_index": 1, "role": "profile", "related": [], "unknown": true}),
            serde_json::json!({"argument_index": 1, "role": "profile", "related": [{"role": "symbols", "extension": "syms.json", "unknown": true}]}),
        ] {
            let mut invalid = value.clone();
            invalid["artifact_group_argument_binding"] = binding;
            assert!(serde_json::from_value::<TaskTemplate>(invalid).is_err());
        }
        let schema = TaskTemplates::generate_json_schema();
        assert!(
            schema
                .to_string()
                .contains("artifact_group_argument_binding")
        );
    }

    #[test]
    fn task_artifact_group_argument_binding_cached_and_recalculated_rerun_keep_pairs_and_identity()
    {
        let template = group_binding_template("target/capture.json");
        let original = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("group task");
        let first = original
            .bind_artifact_invocation("11223344-5566-7788-99AA-BBCCDDEEFF00")
            .expect("first pair");
        let second = first
            .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
            .expect("cached pair");
        for (task, id) in [
            (&first, "112233445566778899aabbccddeeff00"),
            (&second, "99887766554433221100ffeeddccbbaa"),
        ] {
            let group = task.resolved_artifact_group().expect("bound pair");
            assert_eq!(
                group.files["profile"].path,
                format!("target/capture-{id}.json")
            );
            assert_eq!(
                group.files["symbols"].path,
                format!("target/capture-{id}.syms.json")
            );
            assert_eq!(
                task.resolved.args.get(1),
                Some(&group.files["profile"].path)
            );
            assert_eq!(group.files["symbols"].max_bytes, 16 * 1024 * 1024);
            assert_eq!(group.format, "paired-data-v1");
            assert_eq!(task.id, original.id);
            assert_eq!(task.resolved.id, original.resolved.id);
            assert_eq!(task.original_task(), &template);
        }
        assert_eq!(
            second,
            template
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .expect("recalculated task")
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .expect("new pair")
        );
        assert_eq!(original.resolved.args, template.args);
        assert_eq!(
            original.resolved_artifact_group(),
            template.artifact_group.as_ref()
        );
        assert!(
            second
                .resolved
                .command_label
                .contains("capture-99887766554433221100ffeeddccbbaa.json")
        );
        assert!(
            !second
                .resolved
                .command_label
                .contains("112233445566778899aabbccddeeff00")
        );
    }

    #[test]
    fn task_artifact_group_argument_binding_handles_exact_related_argv_and_literal_prefix() {
        let mut template = group_binding_template("target/with spaces/café.json");
        let companion = template.artifact_group.as_ref().expect("group").files["symbols"]
            .path
            .clone();
        template.args.insert(2, companion);
        template
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related[0]
            .argument_index = Some(2);
        assert!(
            template
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .is_none()
        );
        let mut original = template
            .resolve_task_with_literal_arguments(TEST_ID_BASE, &TaskContext::default())
            .expect("literal pair");
        original.resolved.args.push(" && echo $HOME".into());
        original.resolved.preserve_command_label = true;
        original.resolved.command_label = "Capture pair".into();
        let first = original
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("safe pair");
        let group = first.resolved_artifact_group().expect("pair");
        assert_eq!(
            first.resolved.args.get(1),
            Some(&group.files["profile"].path)
        );
        assert_eq!(
            first.resolved.args.get(2),
            Some(&group.files["symbols"].path)
        );
        assert_eq!(first.resolved.literal_prefix_args, 4);
        assert_eq!(
            first.resolved.args.last().map(String::as_str),
            Some(" && echo $HOME")
        );
        assert_eq!(first.resolved.command_label, "Capture pair");
        assert_eq!(first.original_task(), &template);
        let mut demoted = first.clone();
        demoted.resolved.literal_prefix_args = 0;
        assert!(
            demoted
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .is_err()
        );
    }

    #[test]
    fn task_artifact_group_argument_binding_rejects_roles_slots_and_unsafe_paths() {
        let valid = group_binding_template("target/capture.json");
        let mut cases = Vec::new();
        let mut invalid = valid.clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .role = "missing".into();
        cases.push(invalid);
        let mut invalid = valid.clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related
            .clear();
        cases.push(invalid);
        let mut invalid = valid.clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related[0]
            .role = "profile".into();
        cases.push(invalid);
        let mut invalid = valid.clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related[0]
            .role = "missing".into();
        cases.push(invalid);
        let mut invalid = valid.clone();
        let related = invalid
            .artifact_group_argument_binding
            .as_ref()
            .expect("binding")
            .related[0]
            .clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related
            .push(related);
        cases.push(invalid);
        let mut invalid = valid.clone();
        invalid.artifact_argument_binding = Some(TaskArtifactArgumentBinding { argument_index: 1 });
        cases.push(invalid);
        let mut invalid = valid.clone();
        invalid.artifact_group = None;
        cases.push(invalid);
        let mut invalid = valid.clone();
        invalid
            .artifact_group
            .as_mut()
            .expect("group")
            .files
            .get_mut("symbols")
            .expect("symbols")
            .path = "target/unrelated.json".into();
        cases.push(invalid);
        for index in [0, 2, usize::MAX] {
            let mut invalid = valid.clone();
            invalid
                .artifact_group_argument_binding
                .as_mut()
                .expect("binding")
                .argument_index = index;
            cases.push(invalid);
        }
        for index in [0, 1, 2, usize::MAX] {
            let mut invalid = valid.clone();
            invalid
                .artifact_group_argument_binding
                .as_mut()
                .expect("binding")
                .related[0]
                .argument_index = Some(index);
            cases.push(invalid);
        }
        for path in [
            "/capture.json",
            "../capture.json",
            "target//capture.json",
            "target/./capture.json",
            "target/a\\b.json",
            "target/a:b.json",
            "target/a\nb.json",
            "target/$HOME.json",
        ] {
            cases.push(group_binding_template(path));
        }
        let mut invalid = valid.clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .role = "a".repeat(65);
        cases.push(invalid);
        let mut invalid = valid.clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related[0]
            .role = "a".repeat(65);
        cases.push(invalid);
        let mut invalid = valid;
        let related = invalid
            .artifact_group_argument_binding
            .as_ref()
            .expect("binding")
            .related[0]
            .clone();
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related = vec![related; crate::MAX_TASK_ARTIFACT_GROUP_FILES];
        cases.push(invalid);
        for invalid in cases {
            assert!(
                invalid
                    .resolve_task(TEST_ID_BASE, &TaskContext::default())
                    .is_none(),
                "invalid group binding rejected"
            );
        }
    }

    #[test]
    fn task_artifact_group_argument_binding_rejects_suffix_collisions_and_bounds() {
        for extension in [
            "",
            ".json",
            "syms..json",
            "syms.json.",
            "../syms.json",
            "a/b.json",
            "a\\b.json",
            "C:syms.json",
            "a\n.json",
            "a$.json",
            "json",
        ] {
            let mut invalid = group_binding_template("target/capture.json");
            invalid
                .artifact_group_argument_binding
                .as_mut()
                .expect("binding")
                .related[0]
                .extension = extension.into();
            assert!(
                invalid
                    .resolve_task(TEST_ID_BASE, &TaskContext::default())
                    .is_none()
            );
        }
        let mut invalid = group_binding_template("target/capture.json");
        invalid
            .artifact_group_argument_binding
            .as_mut()
            .expect("binding")
            .related[0]
            .extension = "a".repeat(65);
        assert!(
            invalid
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .is_none()
        );
        let mut invalid = group_binding_template("target/capture.json");
        invalid
            .artifact_group
            .as_mut()
            .expect("group")
            .files
            .get_mut("symbols")
            .expect("symbols")
            .max_bytes = crate::MAX_TASK_ARTIFACT_BYTES + 1;
        assert!(
            invalid
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .is_none()
        );
        let mut invalid = group_binding_template("target/capture.json");
        invalid
            .artifact_group
            .as_mut()
            .expect("group")
            .files
            .insert(
                "extra".into(),
                TaskArtifact {
                    path: "target/extra.json".into(),
                    kind: crate::TaskArtifactKind::Data,
                    max_bytes: crate::MAX_TASK_ARTIFACT_BYTES,
                },
            );
        assert!(
            invalid
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .is_none(),
            "combined group payload budget remains enforced"
        );
        assert!(
            group_binding_template(&format!(
                "target/{}/capture.json",
                "a".repeat(crate::MAX_TASK_ARTIFACT_PATH_BYTES)
            ))
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .is_none()
        );
        let parent = std::iter::repeat_n("a".repeat(200), 5)
            .collect::<Vec<_>>()
            .join("/");
        let task = group_binding_template(&format!("{parent}/capture.json"))
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("unbound total path within budget");
        assert!(
            task.bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err(),
            "invocation suffix must respect total path byte limit"
        );
        // The anchor fits, but its longer explicitly declared companion does not.
        let task = group_binding_template(&format!("{}.json", "a".repeat(216)))
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("unbound component");
        assert!(
            task.bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
    }

    #[test]
    fn task_artifact_group_argument_binding_rejects_tampering_and_nil_uuid_atomically() {
        let original = group_binding_template("target/capture.json")
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("group task");
        for invalid in [
            "",
            "not-a-uuid",
            "00000000000000000000000000000000",
            "00000000-0000-0000-0000-000000000000",
        ] {
            assert!(original.bind_artifact_invocation(invalid).is_err());
        }
        let before = original.clone();
        let mut mutated = original.clone();
        mutated.resolved.args[1] = "target/substitute.json".into();
        let unchanged = mutated.clone();
        assert!(
            mutated
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
        assert_eq!(mutated, unchanged);
        let mut mutated = original.clone();
        mutated
            .resolved_artifact_group
            .as_mut()
            .expect("group")
            .format = "different-format".into();
        assert!(
            mutated
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
        let mut mutated = original.clone();
        mutated
            .resolved_artifact_group
            .as_mut()
            .expect("group")
            .files
            .get_mut("symbols")
            .expect("symbols")
            .max_bytes = 1;
        assert!(
            mutated
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
        let mut mutated = original.clone();
        mutated
            .resolved_artifact_group
            .as_mut()
            .expect("group")
            .files
            .get_mut("symbols")
            .expect("symbols")
            .path = "target/other.syms.json".into();
        assert!(
            mutated
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
        let mut mutated = original.clone();
        mutated.resolved.command = None;
        assert!(
            mutated
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
        assert_eq!(original, before);
    }

    #[test]
    fn task_artifact_group_argument_binding_unbound_arbitrary_groups_are_unchanged() {
        let mut template = group_binding_template("target/capture.json");
        template.artifact_group_argument_binding = None;
        template
            .artifact_group
            .as_mut()
            .expect("group")
            .files
            .get_mut("symbols")
            .expect("symbols")
            .path = "other/arbitrary.json".into();
        let task = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("ordinary group");
        assert_eq!(
            task.bind_artifact_invocation("not-a-uuid")
                .expect("legacy no-op"),
            task
        );
    }

    #[test]
    fn task_artifact_argument_binding_json_is_optional_and_checked() {
        let legacy_json = serde_json::json!({"label": "Capture", "command": "collector"});
        let legacy: TaskTemplate = serde_json::from_value(legacy_json).expect("legacy template");
        assert_eq!(legacy.artifact_argument_binding, None);
        let legacy_serialized = serde_json::to_value(&legacy).expect("serialize legacy");
        assert!(legacy_serialized.get("artifact_argument_binding").is_none());
        let legacy_hash = to_hex_hash(&legacy).expect("legacy template hash");
        let resolved = legacy
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("legacy resolution");
        assert!(
            resolved
                .id
                .0
                .starts_with(&format!("{TEST_ID_BASE}_{legacy_hash}_"))
        );
        let roundtrip =
            serde_json::from_value::<TaskTemplate>(legacy_serialized).expect("legacy roundtrip");
        assert_eq!(
            resolved,
            roundtrip
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .expect("legacy repeat")
        );
        let template = artifact_binding_template("target/capture.json");
        let serialized = serde_json::to_value(&template).expect("serialize binding");
        assert_eq!(serialized["artifact_argument_binding"]["argument_index"], 1);
        assert_eq!(
            serde_json::from_value::<TaskTemplate>(serialized).expect("roundtrip"),
            template
        );
        for binding in [
            serde_json::json!({"argument_index": -1}),
            serde_json::json!({"argument_index": 1, "extra": true}),
            serde_json::json!({}),
        ] {
            let mut value = serde_json::to_value(&template).expect("template value");
            value["artifact_argument_binding"] = binding;
            assert!(serde_json::from_value::<TaskTemplate>(value).is_err());
        }
    }

    #[test]
    fn task_artifact_argument_binding_resolution_rejects_invalid_slots_and_paths() {
        let context = TaskContext::default();
        let valid = artifact_binding_template("target/capture.json");
        assert!(valid.resolve_task(TEST_ID_BASE, &context).is_some());
        for index in [0, 2, usize::MAX] {
            let invalid = TaskTemplate {
                artifact_argument_binding: Some(TaskArtifactArgumentBinding {
                    argument_index: index,
                }),
                ..valid.clone()
            };
            assert!(invalid.resolve_task(TEST_ID_BASE, &context).is_none());
        }
        let missing = TaskTemplate {
            artifact: None,
            ..valid.clone()
        };
        assert!(missing.resolve_task(TEST_ID_BASE, &context).is_none());
        let group = TaskTemplate {
            artifact_group: Some(TaskArtifactGroup {
                format: "test-data".into(),
                files: Default::default(),
            }),
            ..valid
        };
        assert!(group.resolve_task(TEST_ID_BASE, &context).is_none());
        for path in [
            "/capture.json",
            "../capture.json",
            "target/../capture.json",
            "target/./capture.json",
            "target//capture.json",
            "target\\capture.json",
            "target:private/capture.json",
            "target/line\n/capture.json",
            "target/$HOME/capture.json",
            "target/with spaces/capture.json",
        ] {
            let invalid = artifact_binding_template(path);
            assert!(
                invalid.resolve_task(TEST_ID_BASE, &context).is_none(),
                "invalid bound path must reject"
            );
        }
        let mut variables = TaskVariables::default();
        variables.insert(VariableName::RelativeDir, "target".into());
        let context = TaskContext {
            task_variables: variables,
            ..Default::default()
        };
        let substituted = artifact_binding_template("$ZED_RELATIVE_DIR/capture.json")
            .resolve_task(TEST_ID_BASE, &context)
            .expect("exact resolved argument");
        assert_eq!(
            substituted.resolved_artifact().expect("artifact").path,
            "target/capture.json"
        );
        assert!(
            artifact_binding_template("$ZED_RELATIVE_DIR/capture.json")
                .resolve_task_with_literal_arguments(TEST_ID_BASE, &context)
                .is_none()
        );
    }

    #[test]
    fn task_artifact_argument_binding_clone_and_cached_rerun_keep_identity() {
        let mut template = artifact_binding_template("target/capture.json");
        template.use_new_terminal = true;
        template.allow_concurrent_runs = true;
        template.show_command = false;
        template.env.insert("CAPTURE_MODE".into(), "private".into());
        let original = template
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("resolved template");
        let first_id = "11223344-5566-7788-99AA-BBCCDDEEFF00";
        let second_id = "99887766554433221100ffeeddccbbaa";
        let first = original
            .bind_artifact_invocation(first_id)
            .expect("first invocation");
        assert_eq!(
            first.resolved_artifact().expect("artifact").path,
            "target/capture-112233445566778899aabbccddeeff00.json"
        );
        assert_eq!(
            first.resolved.args.get(1),
            Some(&first.resolved_artifact().expect("artifact").path)
        );
        assert_eq!(first.id, original.id);
        assert_eq!(first.resolved.id, original.resolved.id);
        assert_eq!(first.original_task(), &template);
        assert_eq!(original.resolved.args, template.args);
        let cached = first.clone();
        let second = cached
            .bind_artifact_invocation(second_id)
            .expect("cached rerun");
        assert_eq!(
            second.resolved_artifact().expect("artifact").path,
            "target/capture-99887766554433221100ffeeddccbbaa.json"
        );
        assert_eq!(second.id, first.id);
        assert_eq!(second.original_task(), &template);
        let mut expected = original.resolved.clone();
        expected.args[1] = second.resolved_artifact().expect("artifact").path.clone();
        expected.command_label = format!("collector {}", expected.args.join(" "));
        assert_eq!(second.resolved, expected);
        assert_eq!(
            first,
            original
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .expect("equivalent UUID")
        );
        assert_eq!(
            second,
            template
                .clone()
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .expect("recalculate rerun")
                .bind_artifact_invocation(second_id)
                .expect("new invocation")
        );
        let legacy = TaskTemplate {
            artifact_argument_binding: None,
            ..template
        }
        .resolve_task(TEST_ID_BASE, &TaskContext::default())
        .expect("legacy task");
        assert_eq!(
            legacy
                .bind_artifact_invocation("not-an-id")
                .expect("no binding is a no-op"),
            legacy
        );
    }

    #[test]
    fn task_artifact_argument_binding_preserves_supported_artifact_formats() {
        for (kind, path, expected) in [
            (
                crate::TaskArtifactKind::Data,
                "capture.json",
                "capture-112233445566778899aabbccddeeff00.json",
            ),
            (
                crate::TaskArtifactKind::Svg,
                "target/capture.svg",
                "target/capture-112233445566778899aabbccddeeff00.svg",
            ),
            (
                crate::TaskArtifactKind::Html,
                "target/capture.htm",
                "target/capture-112233445566778899aabbccddeeff00.htm",
            ),
            (
                crate::TaskArtifactKind::File,
                "target/capture",
                "target/capture-112233445566778899aabbccddeeff00",
            ),
        ] {
            let mut template = artifact_binding_template(path);
            let artifact = template.artifact.as_mut().expect("declared artifact");
            artifact.kind = kind;
            artifact.max_bytes = 4096;
            let original = template
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .expect("supported artifact");
            let bound = original
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .expect("bound artifact");
            let artifact = bound.resolved_artifact().expect("bound declaration");
            assert_eq!(artifact.path, expected);
            assert_eq!(artifact.kind, kind);
            assert_eq!(artifact.max_bytes, 4096);
            assert_eq!(
                bound.resolved.args.get(1).map(String::as_str),
                Some(expected)
            );
        }
    }

    #[test]
    fn task_artifact_argument_binding_rejects_invalid_uuid_tampering_and_overflow() {
        let original = artifact_binding_template("target/capture.json")
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("resolved");
        for invalid in [
            "",
            "private/capture",
            "00000000000000000000000000000000",
            "112233445566778899aabbccddeeff0g",
            "11223344_5566-7788-99aa-bbccddeeff00",
            "11223344-5566-7788-99aa-bbccddeeff000",
        ] {
            let error = original
                .bind_artifact_invocation(invalid)
                .expect_err("invalid UUID");
            assert_eq!(
                error.to_string(),
                "artifact invocation requires a non-nil UUID"
            );
            assert_eq!(
                original.resolved_artifact().expect("artifact").path,
                "target/capture.json"
            );
        }
        let mut tampered = original.clone();
        tampered.resolved.args[1] = "target/other.json".into();
        assert!(
            tampered
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
        let mut missing_command = original;
        missing_command.resolved.command = None;
        assert!(
            missing_command
                .bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
        let long_path = format!(
            "{}{}.json",
            format!("{}/", "a".repeat(240)).repeat(4),
            "b".repeat(55)
        );
        let long = artifact_binding_template(&long_path)
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("bounded original path");
        assert!(
            long.bind_artifact_invocation("112233445566778899aabbccddeeff00")
                .is_err()
        );
    }

    #[test]
    fn task_artifact_argument_binding_bounds_filename_component_before_mutation() {
        let invocation = "112233445566778899aabbccddeeff00";
        let exact = artifact_binding_template(&format!("target/{}.json", "a".repeat(217)))
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .expect("original filename");
        let bound = exact
            .bind_artifact_invocation(invocation)
            .expect("exact component limit");
        assert_eq!(
            bound
                .resolved_artifact()
                .expect("artifact")
                .path
                .rsplit('/')
                .next()
                .expect("filename")
                .len(),
            255
        );
        for path in [
            format!("target/{}.json", "a".repeat(218)),
            format!("target/{}.json", "é".repeat(109)),
        ] {
            let original = artifact_binding_template(&path)
                .resolve_task_with_literal_arguments(TEST_ID_BASE, &TaskContext::default())
                .expect("bounded template path");
            let before = original.clone();
            let error = original
                .bind_artifact_invocation(invocation)
                .expect_err("generated component is oversized");
            assert_eq!(
                error.to_string(),
                "bound artifact filename exceeds its component byte limit"
            );
            assert_eq!(original, before);
        }
    }

    #[test]
    fn task_artifact_argument_binding_preserves_literal_prefix_and_shell_suffix() {
        let path = "target/with spaces/$HOME/capture.json";
        let template = artifact_binding_template(path);
        let mut literal = template
            .resolve_task_with_literal_arguments(TEST_ID_BASE, &TaskContext::default())
            .expect("literal output path");
        literal.resolved.literal_prefix_args = 2;
        let bound = literal
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("literal bound path");
        assert_eq!(bound.resolved.literal_prefix_args, 2);
        assert_eq!(
            bound.resolved.args.get(2).map(String::as_str),
            Some("suffix=$HOME")
        );
        let (_, arguments) =
            util::shell_builder::ShellBuilder::new(&Shell::Program("/bin/sh".into()), false)
                .non_interactive()
                .build_with_literal_prefix(
                    bound.resolved.command.clone(),
                    &bound.resolved.args,
                    bound.resolved.literal_prefix_args,
                );
        let command = arguments.last().expect("shell command");
        assert!(
            command.contains(
                "'target/with spaces/$HOME/capture-112233445566778899aabbccddeeff00.json'"
            )
        );
        assert!(command.ends_with(" suffix=$HOME"));
        let mut unsafe_prefix = bound;
        unsafe_prefix.resolved.literal_prefix_args = 0;
        assert!(
            unsafe_prefix
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .is_err()
        );
        let legacy = TaskTemplate {
            artifact_argument_binding: None,
            ..template
        };
        assert!(
            legacy
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .is_some()
        );
        let mut labeled = literal;
        labeled.resolved.preserve_command_label = true;
        labeled.resolved.command_label = "Human capture label".into();
        let labeled = labeled
            .bind_artifact_invocation("112233445566778899aabbccddeeff00")
            .expect("preserved label");
        assert!(labeled.resolved.preserve_command_label);
        assert_eq!(labeled.resolved.command_label, "Human capture label");
    }

    #[test]
    fn literal_zero_argument_command_survives_persistence_and_rerun() {
        let template = TaskTemplate {
            label: "Installed profiler".into(),
            command: "/installed profiler".into(),
            literal_arguments: true,
            ..Default::default()
        };
        let context = TaskContext::default();
        let resolved = template
            .resolve_task(TEST_ID_BASE, &context)
            .expect("literal command");
        assert!(resolved.resolved.literal_command);
        assert_eq!(resolved.resolved.literal_prefix_args, 0);
        assert!(resolved.resolved.args.is_empty());
        let decoded: TaskTemplate =
            serde_json::from_str(&serde_json::to_string(&template).expect("persist Task intent"))
                .expect("restore Task intent");
        let rerun = decoded
            .resolve_task(TEST_ID_BASE, &context)
            .expect("reevaluated Rerun");
        assert_eq!(rerun, resolved);
        let ordinary = TaskTemplate {
            literal_arguments: false,
            ..template.clone()
        }
        .resolve_task(TEST_ID_BASE, &context)
        .expect("ordinary authored command");
        assert!(!ordinary.resolved.literal_command);
        assert_ne!(ordinary.id, resolved.id);
        assert!(
            TaskTemplate {
                command: "invalid\0program".into(),
                ..template
            }
            .resolve_task(TEST_ID_BASE, &context)
            .is_none()
        );
    }

    #[test]
    fn literal_arguments_preserve_data_and_distinguish_task_identity() {
        let mut variables = TaskVariables::default();
        variables.insert(VariableName::File, "editor-selected.java".into());
        variables.insert(VariableName::WorktreeRoot, "/owned project".into());
        let context = TaskContext {
            task_variables: variables,
            ..Default::default()
        };
        let template = TaskTemplate {
            label: "Provider selection".into(),
            command: "$ZED_WORKTREE_ROOT/wrapper".into(),
            cwd: Some("$ZED_WORKTREE_ROOT".into()),
            args: vec![
                "$ZED_FILE".into(),
                "".into(),
                "literal 日本語 ' \" $HOME".into(),
            ],
            ..Default::default()
        };
        let normal = template
            .resolve_task(TEST_ID_BASE, &context)
            .expect("existing substitution");
        assert_eq!(normal.resolved.args[0], "editor-selected.java");
        assert_eq!(normal.resolved.literal_prefix_args, 0);
        let legacy_hash = to_hex_hash(&template).expect("legacy identity hash");
        assert!(
            normal
                .id
                .0
                .starts_with(&format!("{TEST_ID_BASE}_{legacy_hash}_")),
            "normal identity format must remain unchanged"
        );
        let literal = template
            .resolve_task_with_literal_arguments(TEST_ID_BASE, &context)
            .expect("literal data");
        assert_eq!(literal.resolved.args, template.args);
        assert_eq!(literal.resolved.literal_prefix_args, template.args.len());
        assert_eq!(
            literal.resolved.command.as_deref(),
            Some("/owned project/wrapper")
        );
        assert_eq!(
            literal.resolved.cwd.as_deref(),
            Some(std::path::Path::new("/owned project"))
        );
        assert_eq!(literal.original_task(), &template);
        assert_ne!(literal.id, normal.id);
        assert_eq!(
            literal,
            template
                .resolve_task_with_literal_arguments(TEST_ID_BASE, &context)
                .expect("stable repeat")
        );
        assert!(
            !literal
                .substituted_variables()
                .contains(&VariableName::File)
        );
        let unknown = TaskTemplate {
            args: vec!["$ZED_UNAVAILABLE_PROVIDER_VALUE".into()],
            ..template.clone()
        };
        assert!(unknown.resolve_task(TEST_ID_BASE, &context).is_none());
        assert_eq!(
            unknown
                .resolve_task_with_literal_arguments(TEST_ID_BASE, &context)
                .expect("unknown variable remains data")
                .resolved
                .args,
            unknown.args
        );
        let invalid = TaskTemplate {
            args: vec!["invalid\0argument".into()],
            ..template
        };
        assert!(
            invalid
                .resolve_task_with_literal_arguments(TEST_ID_BASE, &context)
                .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn literal_arguments_reach_posix_process_without_shell_expansion() {
        let directory = std::env::temp_dir().join(format!(
            "groovy-task-arguments-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir(&directory).expect("owned argument fixture");
        let _cleanup = util::defer({
            let directory = directory.clone();
            move || std::fs::remove_dir_all(directory).expect("remove only owned argument fixture")
        });
        let sentinel = directory.join("injected");
        let data = [
            "".to_string(),
            "spaces ' quotes \" 日本語 α 📦".into(),
            "$ZED_FILE".into(),
            "${ZED_UNAVAILABLE_PROVIDER_VALUE}".into(),
            format!("$(touch {})", sentinel.display()),
            "`printf unexpected` ; & | >".into(),
        ];
        let template = TaskTemplate {
            label: "Literal process probe".into(),
            command: "/usr/bin/printf".into(),
            args: std::iter::once("%s\\0".into())
                .chain(data.iter().cloned())
                .collect(),
            ..Default::default()
        };
        let literal = template
            .resolve_task_with_literal_arguments(TEST_ID_BASE, &TaskContext::default())
            .expect("literal plan");
        let (program, arguments) =
            util::shell_builder::ShellBuilder::new(&crate::Shell::Program("/bin/sh".into()), false)
                .build_with_literal_prefix(
                    literal.resolved.command.clone(),
                    &literal.resolved.args,
                    literal.resolved.literal_prefix_args,
                );
        let output = futures::executor::block_on(
            util::command::new_command(program)
                .args(arguments)
                .current_dir(&directory)
                .output(),
        )
        .expect("owned printf process");
        assert!(
            output.status.success(),
            "probe process failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = data
            .iter()
            .flat_map(|argument| {
                argument
                    .as_bytes()
                    .iter()
                    .copied()
                    .chain(std::iter::once(0))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            output.stdout, expected,
            "literal data must arrive byte-for-byte"
        );
        assert!(
            !sentinel.exists(),
            "argument data must not execute command substitution"
        );
    }

    #[test]
    fn test_resolving_templates_with_blank_command_and_label() {
        let task_with_all_properties = TaskTemplate {
            label: "test_label".to_string(),
            command: "test_command".to_string(),
            args: vec!["test_arg".to_string()],
            env: HashMap::from_iter([("test_env_key".to_string(), "test_env_var".to_string())]),
            ..TaskTemplate::default()
        };

        for task_with_blank_property in &[
            TaskTemplate {
                label: "".to_string(),
                ..task_with_all_properties.clone()
            },
            TaskTemplate {
                command: "".to_string(),
                ..task_with_all_properties.clone()
            },
            TaskTemplate {
                label: "".to_string(),
                command: "".to_string(),
                ..task_with_all_properties
            },
        ] {
            assert_eq!(
                task_with_blank_property.resolve_task(TEST_ID_BASE, &TaskContext::default()),
                None,
                "should not resolve task with blank label and/or command: {task_with_blank_property:?}"
            );
        }
    }

    #[test]
    fn test_template_cwd_resolution() {
        let task_without_cwd = TaskTemplate {
            cwd: None,
            label: "test task".to_string(),
            command: "echo 4".to_string(),
            ..TaskTemplate::default()
        };

        let resolved_task = |task_template: &TaskTemplate, task_cx| {
            let resolved_task = task_template
                .resolve_task(TEST_ID_BASE, task_cx)
                .unwrap_or_else(|| panic!("failed to resolve task {task_without_cwd:?}"));
            assert_substituted_variables(&resolved_task, Vec::new());
            resolved_task.resolved
        };

        let cx = TaskContext {
            cwd: None,
            task_variables: TaskVariables::default(),
            project_env: HashMap::default(),
        };
        assert_eq!(
            resolved_task(&task_without_cwd, &cx).cwd,
            None,
            "When neither task nor task context have cwd, it should be None"
        );

        let context_cwd = Path::new("a").join("b").join("c");
        let cx = TaskContext {
            cwd: Some(context_cwd.clone()),
            task_variables: TaskVariables::default(),
            project_env: HashMap::default(),
        };
        assert_eq!(
            resolved_task(&task_without_cwd, &cx).cwd,
            Some(context_cwd.clone()),
            "TaskContext's cwd should be taken on resolve if task's cwd is None"
        );

        let task_cwd = Path::new("d").join("e").join("f");
        let mut task_with_cwd = task_without_cwd.clone();
        task_with_cwd.cwd = Some(task_cwd.display().to_string());
        let task_with_cwd = task_with_cwd;

        let cx = TaskContext {
            cwd: None,
            task_variables: TaskVariables::default(),
            project_env: HashMap::default(),
        };
        assert_eq!(
            resolved_task(&task_with_cwd, &cx).cwd,
            Some(task_cwd.clone()),
            "TaskTemplate's cwd should be taken on resolve if TaskContext's cwd is None"
        );

        let cx = TaskContext {
            cwd: Some(context_cwd),
            task_variables: TaskVariables::default(),
            project_env: HashMap::default(),
        };
        assert_eq!(
            resolved_task(&task_with_cwd, &cx).cwd,
            Some(task_cwd),
            "TaskTemplate's cwd should be taken on resolve if TaskContext's cwd is not None"
        );
    }

    #[test]
    fn test_template_variables_resolution() {
        let custom_variable_1 = VariableName::Custom(Cow::Borrowed("custom_variable_1"));
        let custom_variable_2 = VariableName::Custom(Cow::Borrowed("custom_variable_2"));
        let long_value = "01".repeat(MAX_DISPLAY_VARIABLE_LENGTH * 2);
        let all_variables = [
            (VariableName::Row, "1234".to_string()),
            (VariableName::Column, "5678".to_string()),
            (VariableName::File, "test_file".to_string()),
            (VariableName::SelectedText, "test_selected_text".to_string()),
            (VariableName::Symbol, long_value.clone()),
            (VariableName::WorktreeRoot, "/test_root/".to_string()),
            (
                custom_variable_1.clone(),
                "test_custom_variable_1".to_string(),
            ),
            (
                custom_variable_2.clone(),
                "test_custom_variable_2".to_string(),
            ),
        ];

        let task_with_all_variables = TaskTemplate {
            label: format!(
                "test label for {} and {}",
                VariableName::Row.template_value(),
                VariableName::Symbol.template_value(),
            ),
            command: format!(
                "echo {} {}",
                VariableName::File.template_value(),
                VariableName::Symbol.template_value(),
            ),
            args: vec![
                format!("arg1 {}", VariableName::SelectedText.template_value()),
                format!("arg2 {}", VariableName::Column.template_value()),
                format!("arg3 {}", VariableName::Symbol.template_value()),
            ],
            env: HashMap::from_iter([
                ("test_env_key".to_string(), "test_env_var".to_string()),
                (
                    "env_key_1".to_string(),
                    VariableName::WorktreeRoot.template_value(),
                ),
                (
                    "env_key_2".to_string(),
                    format!(
                        "env_var_2 {} {}",
                        custom_variable_1.template_value(),
                        custom_variable_2.template_value()
                    ),
                ),
                (
                    "env_key_3".to_string(),
                    format!("env_var_3 {}", VariableName::Symbol.template_value()),
                ),
            ]),
            ..TaskTemplate::default()
        };

        let mut first_resolved_id = None;
        for i in 0..15 {
            let resolved_task = task_with_all_variables.resolve_task(
                TEST_ID_BASE,
                &TaskContext {
                    cwd: None,
                    task_variables: TaskVariables::from_iter(all_variables.clone()),
                    project_env: HashMap::default(),
                },
            ).unwrap_or_else(|| panic!("Should successfully resolve task {task_with_all_variables:?} with variables {all_variables:?}"));

            match &first_resolved_id {
                None => first_resolved_id = Some(resolved_task.id.clone()),
                Some(first_id) => assert_eq!(
                    &resolved_task.id, first_id,
                    "Step {i}, for the same task template and context, there should be the same resolved task id"
                ),
            }

            assert_eq!(
                resolved_task.original_task, task_with_all_variables,
                "Resolved task should store its template without changes"
            );
            assert_eq!(
                resolved_task.resolved_label,
                format!("test label for 1234 and {long_value}"),
                "Resolved task label should be substituted with variables and those should not be shortened"
            );
            assert_substituted_variables(
                &resolved_task,
                all_variables.iter().map(|(name, _)| name.clone()).collect(),
            );

            let spawn_in_terminal = &resolved_task.resolved;
            assert_eq!(
                spawn_in_terminal.label,
                format!(
                    "test label for 1234 and …{}",
                    &long_value[long_value.len() - MAX_DISPLAY_VARIABLE_LENGTH..]
                ),
                "Human-readable label should have long substitutions trimmed"
            );
            assert_eq!(
                spawn_in_terminal.command.clone().unwrap(),
                format!("echo test_file {long_value}"),
                "Command should be substituted with variables and those should not be shortened"
            );
            assert_eq!(
                spawn_in_terminal.args,
                &[
                    "arg1 test_selected_text",
                    "arg2 5678",
                    "arg3 010101010101010101010101010101010101010101010101010101010101",
                ],
                "Args should be substituted with variables"
            );
            assert_eq!(
                spawn_in_terminal.command_label,
                format!(
                    "{} arg1 test_selected_text arg2 5678 arg3 {long_value}",
                    spawn_in_terminal.command.clone().unwrap()
                ),
                "Command label args should be substituted with variables and those should not be shortened"
            );

            assert_eq!(
                spawn_in_terminal
                    .env
                    .get("test_env_key")
                    .map(|s| s.as_str()),
                Some("test_env_var")
            );
            assert_eq!(
                spawn_in_terminal.env.get("env_key_1").map(|s| s.as_str()),
                Some("/test_root/")
            );
            assert_eq!(
                spawn_in_terminal.env.get("env_key_2").map(|s| s.as_str()),
                Some("env_var_2 test_custom_variable_1 test_custom_variable_2")
            );
            assert_eq!(
                spawn_in_terminal.env.get("env_key_3"),
                Some(&format!("env_var_3 {long_value}")),
                "Env vars should be substituted with variables and those should not be shortened"
            );
        }

        for i in 0..all_variables.len() {
            let mut not_all_variables = all_variables.to_vec();
            let removed_variable = not_all_variables.remove(i);
            let resolved_task_attempt = task_with_all_variables.resolve_task(
                TEST_ID_BASE,
                &TaskContext {
                    cwd: None,
                    task_variables: TaskVariables::from_iter(not_all_variables),
                    project_env: HashMap::default(),
                },
            );
            assert!(
                matches!(resolved_task_attempt, None),
                "If any of the Zed task variables is not substituted, the task should not be resolved, but got some resolution without the variable {removed_variable:?} (index {i})"
            );
        }
    }

    #[test]
    fn test_can_resolve_free_variables() {
        let task = TaskTemplate {
            label: "My task".into(),
            command: "echo".into(),
            args: vec!["$PATH".into()],
            ..TaskTemplate::default()
        };
        let resolved_task = task
            .resolve_task(TEST_ID_BASE, &TaskContext::default())
            .unwrap();
        assert_substituted_variables(&resolved_task, Vec::new());
        let resolved = resolved_task.resolved;
        assert_eq!(resolved.label, task.label);
        assert_eq!(resolved.command, Some(task.command));
        assert_eq!(resolved.args, task.args);
    }

    #[test]
    fn test_errors_on_missing_zed_variable() {
        let task = TaskTemplate {
            label: "My task".into(),
            command: "echo".into(),
            args: vec!["$ZED_VARIABLE".into()],
            ..TaskTemplate::default()
        };
        assert!(
            task.resolve_task(TEST_ID_BASE, &TaskContext::default())
                .is_none()
        );
    }

    #[test]
    fn test_symbol_dependent_tasks() {
        let task_with_all_properties = TaskTemplate {
            label: "test_label".to_string(),
            command: "test_command".to_string(),
            args: vec!["test_arg".to_string()],
            env: HashMap::from_iter([("test_env_key".to_string(), "test_env_var".to_string())]),
            ..TaskTemplate::default()
        };
        let cx = TaskContext {
            cwd: None,
            task_variables: TaskVariables::from_iter(Some((
                VariableName::Symbol,
                "test_symbol".to_string(),
            ))),
            project_env: HashMap::default(),
        };

        for (i, symbol_dependent_task) in [
            TaskTemplate {
                label: format!("test_label_{}", VariableName::Symbol.template_value()),
                ..task_with_all_properties.clone()
            },
            TaskTemplate {
                command: format!("test_command_{}", VariableName::Symbol.template_value()),
                ..task_with_all_properties.clone()
            },
            TaskTemplate {
                args: vec![format!(
                    "test_arg_{}",
                    VariableName::Symbol.template_value()
                )],
                ..task_with_all_properties.clone()
            },
            TaskTemplate {
                env: HashMap::from_iter([(
                    "test_env_key".to_string(),
                    format!("test_env_var_{}", VariableName::Symbol.template_value()),
                )]),
                ..task_with_all_properties
            },
        ]
        .into_iter()
        .enumerate()
        {
            let resolved = symbol_dependent_task
                .resolve_task(TEST_ID_BASE, &cx)
                .unwrap_or_else(|| panic!("Failed to resolve task {symbol_dependent_task:?}"));
            assert_eq!(
                resolved.substituted_variables,
                HashSet::from_iter(Some(VariableName::Symbol)),
                "(index {i}) Expected the task to depend on symbol task variable: {resolved:?}"
            )
        }
    }

    #[track_caller]
    fn assert_substituted_variables(resolved_task: &ResolvedTask, mut expected: Vec<VariableName>) {
        let mut resolved_variables = resolved_task
            .substituted_variables
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        resolved_variables.sort_by_key(|var| var.to_string());
        expected.sort_by_key(|var| var.to_string());
        assert_eq!(resolved_variables, expected)
    }

    #[test]
    fn substitute_funky_labels() {
        let faulty_go_test = TaskTemplate {
            label: format!(
                "go test {}/{}",
                VariableName::Symbol.template_value(),
                VariableName::Symbol.template_value(),
            ),
            command: "go".into(),
            args: vec![format!(
                "^{}$/^{}$",
                VariableName::Symbol.template_value(),
                VariableName::Symbol.template_value()
            )],
            ..TaskTemplate::default()
        };
        let mut context = TaskContext::default();
        context
            .task_variables
            .insert(VariableName::Symbol, "my-symbol".to_string());
        assert!(faulty_go_test.resolve_task("base", &context).is_some());
    }

    #[test]
    fn test_project_env() {
        let all_variables = [
            (VariableName::Row, "1234".to_string()),
            (VariableName::Column, "5678".to_string()),
            (VariableName::File, "test_file".to_string()),
            (VariableName::Symbol, "my symbol".to_string()),
        ];

        let template = TaskTemplate {
            label: "my task".to_string(),
            command: format!(
                "echo {} {}",
                VariableName::File.template_value(),
                VariableName::Symbol.template_value(),
            ),
            args: vec![],
            env: HashMap::from_iter([
                (
                    "TASK_ENV_VAR1".to_string(),
                    "TASK_ENV_VAR1_VALUE".to_string(),
                ),
                (
                    "TASK_ENV_VAR2".to_string(),
                    format!(
                        "env_var_2 {} {}",
                        VariableName::Row.template_value(),
                        VariableName::Column.template_value()
                    ),
                ),
                (
                    "PROJECT_ENV_WILL_BE_OVERWRITTEN".to_string(),
                    "overwritten".to_string(),
                ),
            ]),
            ..TaskTemplate::default()
        };

        let project_env = HashMap::from_iter([
            (
                "PROJECT_ENV_VAR1".to_string(),
                "PROJECT_ENV_VAR1_VALUE".to_string(),
            ),
            (
                "PROJECT_ENV_WILL_BE_OVERWRITTEN".to_string(),
                "PROJECT_ENV_WILL_BE_OVERWRITTEN_VALUE".to_string(),
            ),
        ]);

        let context = TaskContext {
            cwd: None,
            task_variables: TaskVariables::from_iter(all_variables),
            project_env,
        };

        let resolved = template
            .resolve_task(TEST_ID_BASE, &context)
            .unwrap()
            .resolved;

        assert_eq!(resolved.env["TASK_ENV_VAR1"], "TASK_ENV_VAR1_VALUE");
        assert_eq!(resolved.env["TASK_ENV_VAR2"], "env_var_2 1234 5678");
        assert_eq!(resolved.env["PROJECT_ENV_VAR1"], "PROJECT_ENV_VAR1_VALUE");
        assert_eq!(
            resolved.env["PROJECT_ENV_WILL_BE_OVERWRITTEN"],
            "overwritten"
        );
    }

    #[test]
    fn test_variable_default_values() {
        let task_with_defaults = TaskTemplate {
            label: "test with defaults".to_string(),
            command: format!(
                "echo ${{{}}}",
                VariableName::File.to_string() + ":fallback.txt"
            ),
            args: vec![
                "${ZED_MISSING_VAR:default_value}".to_string(),
                format!("${{{}}}", VariableName::Row.to_string() + ":42"),
            ],
            ..TaskTemplate::default()
        };

        // Test 1: When ZED_FILE exists, should use actual value and ignore default
        let context_with_file = TaskContext {
            cwd: None,
            task_variables: TaskVariables::from_iter(vec![
                (VariableName::File, "actual_file.rs".to_string()),
                (VariableName::Row, "123".to_string()),
            ]),
            project_env: HashMap::default(),
        };

        let resolved = task_with_defaults
            .resolve_task(TEST_ID_BASE, &context_with_file)
            .expect("Should resolve task with existing variables");

        assert_eq!(
            resolved.resolved.command.unwrap(),
            "echo actual_file.rs",
            "Should use actual ZED_FILE value, not default"
        );
        assert_eq!(
            resolved.resolved.args,
            vec!["default_value", "123"],
            "Should use default for missing var, actual value for existing var"
        );

        // Test 2: When ZED_FILE doesn't exist, should use default value
        let context_without_file = TaskContext {
            cwd: None,
            task_variables: TaskVariables::from_iter(vec![(VariableName::Row, "456".to_string())]),
            project_env: HashMap::default(),
        };

        let resolved = task_with_defaults
            .resolve_task(TEST_ID_BASE, &context_without_file)
            .expect("Should resolve task using default values");

        assert_eq!(
            resolved.resolved.command.unwrap(),
            "echo fallback.txt",
            "Should use default value when ZED_FILE is missing"
        );
        assert_eq!(
            resolved.resolved.args,
            vec!["default_value", "456"],
            "Should use defaults for missing vars"
        );

        // Test 3: Missing ZED variable without default should fail
        let task_no_default = TaskTemplate {
            label: "test no default".to_string(),
            command: "${ZED_MISSING_NO_DEFAULT}".to_string(),
            ..TaskTemplate::default()
        };

        assert!(
            task_no_default
                .resolve_task(TEST_ID_BASE, &TaskContext::default())
                .is_none(),
            "Should fail when ZED variable has no default and doesn't exist"
        );
    }

    #[test]
    fn test_unknown_variables() {
        // Variable names starting with `ZED_` that are not valid should be
        // reported.
        let label = "test unknown variables".to_string();
        let command = "$ZED_UNKNOWN".to_string();
        let task = TaskTemplate {
            label,
            command,
            ..TaskTemplate::default()
        };

        assert_eq!(task.unknown_variables(), vec!["ZED_UNKNOWN".to_string()]);

        // Variable names starting with `ZED_CUSTOM_` should never be reported,
        // as those are dynamically provided by extensions.
        let label = "test custom variables".to_string();
        let command = "$ZED_CUSTOM_UNKNOWN".to_string();
        let task = TaskTemplate {
            label,
            command,
            ..TaskTemplate::default()
        };

        assert!(task.unknown_variables().is_empty());

        // Unknown variable names with defaults should still be reported,
        // otherwise the default would always be silently used.
        let label = "test custom variables".to_string();
        let command = "${ZED_UNKNOWN:default_value}".to_string();
        let task = TaskTemplate {
            label,
            command,
            ..TaskTemplate::default()
        };

        assert_eq!(task.unknown_variables(), vec!["ZED_UNKNOWN".to_string()]);

        // Valid variable names are not reported.
        let label = "test custom variables".to_string();
        let command = "$ZED_FILE".to_string();
        let task = TaskTemplate {
            label,
            command,
            ..TaskTemplate::default()
        };
        assert!(task.unknown_variables().is_empty());
    }

    #[test]
    fn test_git_variables_resolution() {
        let task = TaskTemplate {
            label: "Show $ZED_GIT_SHA_SHORT in $ZED_GIT_REPOSITORY_NAME".to_string(),
            command: "git".to_string(),
            args: vec!["show".to_string(), "$ZED_GIT_SHA".to_string()],
            cwd: Some("$ZED_GIT_REPOSITORY_PATH".to_string()),
            env: HashMap::from_iter([("COMMIT".to_string(), "$ZED_GIT_SHA".to_string())]),
            ..TaskTemplate::default()
        };
        let sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string();
        let sha_short = "0123456".to_string();
        let repo_name = "zed".to_string();
        let repo_path = format!("/Users/example/{repo_name}");

        let context = TaskContext {
            task_variables: TaskVariables::from_iter([
                (VariableName::GitSha, sha.clone()),
                (VariableName::GitShaShort, sha_short.clone()),
                (VariableName::GitRepositoryName, repo_name.clone()),
                (VariableName::GitRepositoryPath, repo_path.clone()),
            ]),
            ..TaskContext::default()
        };

        let task = task.resolve_task(TEST_ID_BASE, &context).unwrap();
        assert_eq!(
            task.resolved_label,
            format!("Show {sha_short} in {repo_name}")
        );
        assert_eq!(task.resolved.command, Some("git".to_string()));
        assert_eq!(task.resolved.args, vec!["show".to_string(), sha.clone()]);
        assert_eq!(task.resolved.cwd, Some(PathBuf::from(repo_path)));
        assert_eq!(task.resolved.env.get("COMMIT"), Some(&sha));

        assert_substituted_variables(
            &task,
            vec![
                VariableName::GitSha,
                VariableName::GitShaShort,
                VariableName::GitRepositoryName,
                VariableName::GitRepositoryPath,
            ],
        );
    }

    #[test]
    fn test_args_produced_by_empty_variables_are_omitted() {
        let features_flag = VariableName::Custom(Cow::Borrowed("features_flag"));
        let features = VariableName::Custom(Cow::Borrowed("features"));
        let bin_name = VariableName::Custom(Cow::Borrowed("bin_name"));

        let task = TaskTemplate {
            label: "cargo run".to_string(),
            command: "cargo".to_string(),
            args: vec![
                "run".to_string(),
                "--bin".to_string(),
                bin_name.template_value(),
                features_flag.template_value(),
                features.template_value(),
                String::new(),
                format!("--config={}", features.template_value()),
            ],
            ..TaskTemplate::default()
        };

        let context = TaskContext {
            task_variables: TaskVariables::from_iter([
                (features_flag, String::new()),
                (features, String::new()),
                (bin_name, "test_bin".to_string()),
            ]),
            ..TaskContext::default()
        };

        let resolved = task
            .resolve_task(TEST_ID_BASE, &context)
            .unwrap_or_else(|| panic!("failed to resolve task {task:?}"))
            .resolved;
        assert_eq!(
            resolved.args,
            vec!["run", "--bin", "test_bin", "", "--config="],
            "args that consist entirely of variables resolved to empty strings should be omitted, \
            while literal empty args and partially substituted args should be preserved"
        );
    }

    #[test]
    fn task_artifact_groups_resolve_named_files_and_reject_invalid_contracts() {
        use crate::{TaskArtifactGroup, TaskArtifactKind};
        let group = TaskArtifactGroup {
            format: "samply-0.13.1".into(),
            files: collections::BTreeMap::from_iter([
                (
                    "profile".into(),
                    TaskArtifact {
                        path: "target/$ZED_STEM.json".into(),
                        kind: TaskArtifactKind::Data,
                        max_bytes: 32 * 1024 * 1024,
                    },
                ),
                (
                    "symbols".into(),
                    TaskArtifact {
                        path: "target/$ZED_STEM.syms.json".into(),
                        kind: TaskArtifactKind::Data,
                        max_bytes: 16 * 1024 * 1024,
                    },
                ),
            ]),
        };
        let template = TaskTemplate {
            label: "profile".into(),
            command: "profiler".into(),
            artifact_group: Some(group.clone()),
            ..Default::default()
        };
        let context = TaskContext {
            task_variables: TaskVariables::from_iter([(VariableName::Stem, "workload".into())]),
            ..Default::default()
        };
        let resolved = template
            .resolve_task(TEST_ID_BASE, &context)
            .expect("group resolves");
        let files = &resolved
            .resolved_artifact_group()
            .expect("resolved group")
            .files;
        assert_eq!(
            files.get("profile").expect("profile").path,
            "target/workload.json"
        );
        assert_eq!(
            files.get("symbols").expect("symbols").path,
            "target/workload.syms.json"
        );
        assert!(resolved.resolved_artifact().is_none());
        assert!(
            resolved
                .substituted_variables()
                .contains(&VariableName::Stem)
        );
        let serialized = serde_json::to_string(&template).expect("serialize group");
        assert_eq!(
            serde_json::from_str::<TaskTemplate>(&serialized).expect("deserialize group"),
            template
        );
        let mut unknown = template.clone();
        unknown
            .artifact_group
            .as_mut()
            .expect("group")
            .files
            .get_mut("profile")
            .expect("profile")
            .path = "target/$ZED_UNKNOWN.json".into();
        assert_eq!(unknown.unknown_variables(), vec!["ZED_UNKNOWN".to_string()]);
        for mutation in 0..12 {
            let mut invalid = template.clone();
            let group = invalid.artifact_group.as_mut().expect("group");
            match mutation {
                0 => group.files.clear(),
                1 => group.format = "bad format".into(),
                2 => {
                    group.files.insert(
                        "bad role".into(),
                        group.files.get("profile").expect("profile").clone(),
                    );
                }
                3 => {
                    group.files.get_mut("symbols").expect("symbols").path =
                        "target/$ZED_STEM.json".into()
                }
                4 => {
                    group.files.get_mut("profile").expect("profile").path = "../outside.json".into()
                }
                5 => {
                    group.files.get_mut("profile").expect("profile").path =
                        "target\\..\\outside.json".into()
                }
                6 => group.files.get_mut("profile").expect("profile").kind = TaskArtifactKind::File,
                7 => {
                    for index in 0..9 {
                        group.files.insert(
                            format!("file{index}"),
                            TaskArtifact {
                                path: format!("target/{index}.json"),
                                kind: TaskArtifactKind::Data,
                                max_bytes: 1024,
                            },
                        );
                    }
                }
                8 => {
                    invalid.artifact = Some(TaskArtifact {
                        path: "target/file.json".into(),
                        kind: TaskArtifactKind::Data,
                        max_bytes: 1024,
                    })
                }
                9 => {
                    group.files.get_mut("profile").expect("profile").path =
                        "target/name:stream.json".into()
                }
                10 => {
                    group.files.get_mut("profile").expect("profile").path =
                        "target/./profile.json".into()
                }
                11 => {
                    group.files.get_mut("profile").expect("profile").path =
                        "target/control\n.json".into()
                }
                _ => unreachable!(),
            }
            assert!(
                invalid.resolve_task(TEST_ID_BASE, &context).is_none(),
                "mutation {mutation}"
            );
        }
        let mut oversized = group.clone();
        oversized.files.insert(
            "extra".into(),
            TaskArtifact {
                path: "target/extra.json".into(),
                kind: TaskArtifactKind::Data,
                max_bytes: 32 * 1024 * 1024,
            },
        );
        assert!(
            oversized
                .with_resolved_files(oversized.files.clone())
                .is_err()
        );
        let mut changed = group.files.clone();
        changed.get_mut("symbols").expect("symbols").max_bytes += 1;
        assert!(group.with_resolved_files(changed).is_err());
        let mut missing = group.files.clone();
        missing.remove("symbols");
        assert!(group.with_resolved_files(missing).is_err());
    }

    #[test]
    fn task_artifact_paths_are_substituted_and_bounded() {
        let task = TaskTemplate {
            label: "profile".to_string(),
            command: "profiler".to_string(),
            artifact: Some(TaskArtifact {
                path: "target/$ZED_STEM.svg".to_string(),
                kind: crate::TaskArtifactKind::Svg,
                max_bytes: 1024,
            }),
            ..TaskTemplate::default()
        };
        let context = TaskContext {
            task_variables: TaskVariables::from_iter([(VariableName::Stem, "server".to_string())]),
            ..TaskContext::default()
        };
        let resolved = task
            .resolve_task(TEST_ID_BASE, &context)
            .expect("safe artifact should resolve");
        assert_eq!(
            resolved
                .resolved_artifact()
                .map(|artifact| artifact.path.as_str()),
            Some("target/server.svg")
        );
        assert!(
            resolved
                .substituted_variables()
                .contains(&VariableName::Stem)
        );

        for path in [
            "../profile.svg",
            "/tmp/profile.svg",
            "https://example.com/profile.svg",
        ] {
            let invalid = TaskTemplate {
                artifact: Some(TaskArtifact {
                    path: path.to_string(),
                    kind: crate::TaskArtifactKind::Svg,
                    max_bytes: 1024,
                }),
                ..task.clone()
            };
            assert!(invalid.resolve_task(TEST_ID_BASE, &context).is_none());
        }

        let data = TaskTemplate {
            artifact: Some(TaskArtifact {
                path: "target/coverage.json".to_string(),
                kind: crate::TaskArtifactKind::Data,
                max_bytes: 4096,
            }),
            ..task.clone()
        };
        assert!(data.resolve_task(TEST_ID_BASE, &context).is_some());
        let invalid_data = TaskTemplate {
            artifact: Some(TaskArtifact {
                path: "target/coverage.txt".to_string(),
                kind: crate::TaskArtifactKind::Data,
                max_bytes: 4096,
            }),
            ..task
        };
        assert!(invalid_data.resolve_task(TEST_ID_BASE, &context).is_none());
    }
}
