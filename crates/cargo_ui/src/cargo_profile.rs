use anyhow::{Result, bail};
use collections::HashMap;
use serde::{Deserialize, Serialize};
use settings::{CargoNativeProfilerSettingsContent, CargoProfilerSettingsContent};
use task::{TaskArtifact, TaskArtifactKind};

use crate::cargo_preset::CompiledCargoPreset;

const MAX_PROFILE_COMMAND_ITEMS: usize = 256;
const MAX_PROFILE_COMMAND_TEXT_BYTES: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExternalProfileCommand {
    pub command: String,
    pub arguments: Vec<String>,
    pub environment: HashMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CargoProfiler {
    pub command: ExternalProfileCommand,
    pub artifact: TaskArtifact,
}

impl CargoProfiler {
    pub fn from_settings(content: &CargoProfilerSettingsContent) -> Result<Self> {
        let command = ExternalProfileCommand {
            command: content.command.clone().unwrap_or_default(),
            arguments: content.args.clone().unwrap_or_default(),
            environment: HashMap::default(),
        };
        validate_external_command(&command)?;
        let kind = match content.artifact_kind.as_deref() {
            Some("svg") => TaskArtifactKind::Svg,
            Some("html") => TaskArtifactKind::Html,
            _ => bail!("profiler artifact_kind must be svg or html"),
        };
        let artifact = TaskArtifact {
            path: content.artifact_path.clone().unwrap_or_default(),
            kind,
            max_bytes: content.artifact_max_bytes.unwrap_or(0),
        };
        artifact.with_resolved_path(artifact.path.clone())?;
        Ok(Self { command, artifact })
    }

    pub fn settings_content(&self) -> CargoProfilerSettingsContent {
        CargoProfilerSettingsContent {
            command: Some(self.command.command.clone()),
            args: Some(self.command.arguments.clone()),
            artifact_path: Some(self.artifact.path.clone()),
            artifact_kind: Some(match self.artifact.kind {
                TaskArtifactKind::Svg => "svg".to_string(),
                TaskArtifactKind::Html => "html".to_string(),
                TaskArtifactKind::Data => "data".to_string(),
                TaskArtifactKind::File => "file".to_string(),
            }),
            artifact_max_bytes: Some(self.artifact.max_bytes),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoNativeProfiler {
    pub collector_path: String,
    pub profile_path: String,
}

impl CargoNativeProfiler {
    pub fn from_settings(content: &CargoNativeProfilerSettingsContent) -> Result<Self> {
        let collector_path = content.collector_path.as_deref().unwrap_or_default();
        if collector_path.is_empty()
            || collector_path.len() > MAX_PROFILE_COMMAND_TEXT_BYTES
            || !std::path::Path::new(collector_path).is_absolute()
            || collector_path.chars().any(|character| {
                character.is_whitespace()
                    || character.is_control()
                    || matches!(
                        character,
                        ';' | '|' | '&' | '<' | '>' | '`' | '$' | '\\' | '"' | '\''
                    )
            })
            || std::path::Path::new(collector_path)
                .components()
                .any(|component| {
                    matches!(
                        component,
                        std::path::Component::ParentDir | std::path::Component::CurDir
                    )
                })
        {
            bail!(
                "Native profiling requires bounded absolute installed Samply executable path supported by the Task host binding"
            );
        }
        let profile_path = content.profile_path.as_deref().unwrap_or_default();
        if profile_path.is_empty()
            || profile_path.len() > task::MAX_TASK_ARTIFACT_PATH_BYTES
            || profile_path == ".json"
            || profile_path.contains("$ZED_")
            || profile_path.contains("${ZED_")
            || !profile_path.ends_with(".json")
            || profile_path.ends_with(".syms.json")
        {
            bail!(
                "Native profiling requires a resolved JSON profile output distinct from its symbols companion"
            );
        }
        TaskArtifact {
            path: profile_path.to_owned(),
            kind: TaskArtifactKind::Data,
            max_bytes: 32 * 1024 * 1024,
        }
        .with_resolved_path(profile_path.to_owned())?;
        Ok(Self {
            collector_path: collector_path.to_owned(),
            profile_path: profile_path.to_owned(),
        })
    }

    pub fn settings_content(&self) -> CargoNativeProfilerSettingsContent {
        CargoNativeProfilerSettingsContent {
            collector_path: Some(self.collector_path.clone()),
            profile_path: Some(self.profile_path.clone()),
        }
    }
}

pub(crate) fn validate_native_profile_selection(
    selection: &crate::CargoActionSelection,
    preset: &crate::CargoPreset,
) -> Result<()> {
    let kind_matches = matches!(
        (&selection.node_kind, &selection.target),
        (
            crate::CargoActionNodeKind::Target(crate::CargoActionTargetKind::Binary),
            Some(crate::CargoTargetSelector::Binary(_))
        ) | (
            crate::CargoActionNodeKind::Target(crate::CargoActionTargetKind::Example),
            Some(crate::CargoTargetSelector::Example(_))
        )
    );
    if !kind_matches {
        bail!(
            "Selected native profiling currently supports binary and example targets; test and bench native capture remain incomplete"
        );
    }
    if preset.toolchain.is_some()
        || preset.target_triple.is_some()
        || !preset.features.is_empty()
        || preset.default_features.is_some()
        || !preset.args.is_empty()
        || !matches!(
            preset.working_directory,
            crate::CargoWorkingDirectoryPolicy::Context
                | crate::CargoWorkingDirectoryPolicy::Workspace
        )
    {
        bail!(
            "Selected native profiling currently requires the host default toolchain and workspace cwd; custom toolchain, target, features and Build arguments remain incomplete"
        );
    }
    if preset.profile.as_ref().is_some_and(|profile| {
        profile.is_empty()
            || profile.len() > project::cargo_workspace::MAX_CARGO_CONFIGURATION_FIELD_BYTES
            || profile.starts_with('-')
            || !profile
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    }) {
        bail!("Selected native profiling requires a bounded Cargo profile name");
    }
    let profiler = preset.native_profiler.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Configure an explicit native profiler in the active Cargo preset")
    })?;
    CargoNativeProfiler::from_settings(&profiler.settings_content())?;
    if preset.profiler.is_some() {
        bail!("Choose either the external profiler or native profiler for this Cargo preset");
    }
    Ok(())
}

pub(crate) fn native_profile_selected_target(
    selection: &crate::CargoActionSelection,
    snapshot: &project::cargo_workspace::CargoWorkspaceSnapshot,
) -> Result<(
    project::cargo_workspace::CargoWorkspaceKey,
    String,
    project::cargo_workspace::CargoTargetModel,
)> {
    let (target_name, target_kind) = match (&selection.node_kind, &selection.target) {
        (
            crate::CargoActionNodeKind::Target(crate::CargoActionTargetKind::Binary),
            Some(crate::CargoTargetSelector::Binary(name)),
        ) => (name, project::cargo_workspace::CargoTargetKind::Binary),
        (
            crate::CargoActionNodeKind::Target(crate::CargoActionTargetKind::Example),
            Some(crate::CargoTargetSelector::Example(name)),
        ) => (name, project::cargo_workspace::CargoTargetKind::Example),
        _ => bail!("Selected native profiling requires an exact binary or example target"),
    };
    let mut workspaces = snapshot.workspaces.iter().filter(|workspace| {
        workspace.root_manifest.as_ref() == Some(&selection.workspace_manifest)
            && workspace.key.root.worktree_id == selection.worktree_id
    });
    let workspace = workspaces.next().ok_or_else(|| {
        anyhow::anyhow!("The selected native profiling workspace is no longer present")
    })?;
    if workspaces.next().is_some() {
        bail!("The selected native profiling workspace is ambiguous");
    }
    let mut members = workspace.members.iter().filter(|member| {
        Some(&member.manifest_path) == selection.package_manifest.as_ref()
            && Some(&member.name) == selection.package_name.as_ref()
    });
    let member = members.next().ok_or_else(|| {
        anyhow::anyhow!("The selected native profiling package is no longer present")
    })?;
    if members.next().is_some() {
        bail!("The selected native profiling package is ambiguous");
    }
    let mut targets = member
        .targets
        .iter()
        .filter(|target| &target.name == target_name && target.kind == target_kind);
    let target = targets.next().ok_or_else(|| {
        anyhow::anyhow!("The selected native profiling target is no longer present")
    })?;
    if targets.next().is_some() || target.source_path.is_none() {
        bail!("The selected native profiling target has no unique mapped source");
    }
    Ok((workspace.key.clone(), member.id.clone(), target.clone()))
}

pub(crate) fn native_profile_build_template(
    preset: &crate::CargoPreset,
    recipe: project::cargo_workspace_store::NativeCargoBuildRecipe,
) -> Result<task::TaskTemplate> {
    let profiler = preset.native_profiler.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Configure an explicit native profiler in the active Cargo preset")
    })?;
    let continuation = task::TaskNativeProfileContinuation {
        collector_path: profiler.collector_path.clone(),
        profile_path: profiler.profile_path.clone(),
        workload_arguments: preset.trailing_args.clone(),
    };
    let stdout_report = task::TaskStdoutReport {
        format: task::TaskStdoutReportFormat::CargoRuntimePreparationJson,
        max_bytes: project::cargo_workspace::MAX_CARGO_BUILD_MESSAGE_BYTES,
        native_profile: Some(continuation),
    };
    stdout_report.validate()?;
    if !recipe.program.is_absolute() || !recipe.working_directory.is_absolute() {
        bail!("Native profiling needs an absolute host-owned Build recipe");
    }
    Ok(task::TaskTemplate {
        experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
        label: format!("Native Profile (macOS): {}", preset.label),
        command: recipe
            .program
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Native Build program path is not UTF-8"))?
            .to_owned(),
        args: recipe.arguments,
        env: recipe.task_environment,
        cwd: Some(
            recipe
                .working_directory
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Native Build working directory is not UTF-8"))?
                .to_owned(),
        ),
        literal_arguments: true,
        stdout_report: Some(stdout_report),
        required_experimental_features: vec!["rust-native-profiling-experimental".into()],
        reveal: preset.presentation.reveal,
        reveal_target: preset.presentation.reveal_target,
        hide: preset.presentation.hide,
        save: preset.presentation.save,
        use_new_terminal: preset.presentation.use_new_terminal,
        allow_concurrent_runs: preset.presentation.allow_concurrent_runs,
        tags: vec!["cargo-profile".into(), "cargo-native-profile".into()],
        ..task::TaskTemplate::default()
    })
}

pub fn compile_profile_plan(
    mut compiled_cargo_context: CompiledCargoPreset,
    external_command: ExternalProfileCommand,
    artifact: TaskArtifact,
) -> Result<CompiledCargoPreset> {
    validate_external_command(&external_command)?;
    if !artifact.path.contains("$ZED_") {
        artifact.with_resolved_path(artifact.path.clone())?;
    }

    compiled_cargo_context.task_template.label =
        format!("Profile {}", compiled_cargo_context.task_template.label);
    compiled_cargo_context.task_template.command = external_command.command;
    compiled_cargo_context.task_template.args = external_command.arguments;
    compiled_cargo_context.task_template.literal_arguments = true;
    compiled_cargo_context
        .task_template
        .env
        .extend(external_command.environment);
    compiled_cargo_context
        .task_template
        .tags
        .retain(|tag| !tag.starts_with("cargo-"));
    compiled_cargo_context
        .task_template
        .tags
        .extend(["cargo-profile".to_string(), "external-tool".to_string()]);
    compiled_cargo_context.task_template.artifact = Some(artifact);
    Ok(compiled_cargo_context)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplyCollector {
    pub command: String,
    pub version: String,
}

pub fn compile_native_profile_plan(
    mut compiled_cargo_context: CompiledCargoPreset,
    host_os: &str,
    collector: &SamplyCollector,
    workload: ExternalProfileCommand,
    profile_path: &str,
) -> Result<CompiledCargoPreset> {
    compiled_cargo_context.task_template = project::cargo_workspace::compile_native_profile_task(
        compiled_cargo_context.task_template,
        host_os,
        &project::cargo_workspace::NativeProfileCollector {
            command: collector.command.clone(),
            version: collector.version.clone(),
        },
        project::cargo_workspace::NativeProfileCommand {
            command: workload.command,
            arguments: workload.arguments,
            environment: workload.environment,
        },
        profile_path,
    )?;
    Ok(compiled_cargo_context)
}

fn validate_external_command(command: &ExternalProfileCommand) -> Result<()> {
    if command.command.trim().is_empty()
        || command.command.len() > MAX_PROFILE_COMMAND_TEXT_BYTES
        || command.arguments.len() > MAX_PROFILE_COMMAND_ITEMS
        || command.environment.len() > MAX_PROFILE_COMMAND_ITEMS
    {
        bail!("external profile command is empty or exceeds the supported bounds");
    }
    if command
        .arguments
        .iter()
        .chain(command.environment.keys())
        .chain(command.environment.values())
        .any(|value| value.len() > MAX_PROFILE_COMMAND_TEXT_BYTES)
    {
        bail!("external profile command field exceeds the supported length");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_preset::{
        CargoCompileContext, CargoPreset, CargoPresetScope, CargoSubcommand,
        CargoWorkingDirectoryPolicy, compile_preset,
    };
    use task::{TaskArtifactKind, TaskContext, TaskVariables, VariableName};

    #[test]
    fn cargo_profile_plan_uses_an_explicit_external_task_and_declared_artifact() {
        let compiled = compile_preset(
            &CargoPreset {
                label: "release binary".to_string(),
                subcommand: CargoSubcommand::Run,
                scope: CargoPresetScope::Package,
                package: Some("app".to_string()),
                working_directory: CargoWorkingDirectoryPolicy::Package,
                ..CargoPreset::ephemeral_default(CargoSubcommand::Run)
            },
            &CargoCompileContext {
                workspace_cwd: Some("/workspace".to_string()),
                package_cwd: Some("/workspace/app".to_string()),
                ..CargoCompileContext::default()
            },
            None,
        )
        .expect("Cargo context should compile");
        let plan = compile_profile_plan(
            compiled,
            ExternalProfileCommand {
                command: "my-profiler".to_string(),
                arguments: vec![
                    "record".to_string(),
                    "--output".to_string(),
                    "target/profile.svg".to_string(),
                    "cargo".to_string(),
                    "run".to_string(),
                    "--package".to_string(),
                    "app".to_string(),
                ],
                environment: HashMap::from_iter([(
                    "PROFILE_MODE".to_string(),
                    "sampling".to_string(),
                )]),
            },
            TaskArtifact {
                path: "target/profile.svg".to_string(),
                kind: TaskArtifactKind::Svg,
                max_bytes: 1024 * 1024,
            },
        )
        .expect("explicit external profile plan should compile");

        assert_eq!(plan.task_template.command, "my-profiler");
        assert_eq!(plan.task_template.cwd.as_deref(), Some("/workspace/app"));
        assert_eq!(
            plan.task_template
                .artifact
                .as_ref()
                .map(|artifact| artifact.path.as_str()),
            Some("target/profile.svg")
        );
        assert!(
            plan.task_template
                .tags
                .contains(&"external-tool".to_string())
        );
        let resolved = plan
            .task_template
            .resolve_task("cargo-profile", &TaskContext::default())
            .expect("profile task should resolve without a new runner");
        assert_eq!(
            resolved
                .resolved_artifact()
                .map(|artifact| artifact.path.as_str()),
            Some("target/profile.svg")
        );
    }

    #[test]
    fn cargo_profile_plan_rejects_unsafe_or_oversized_declarations() {
        let compiled = compile_preset(
            &CargoPreset::ephemeral_default(CargoSubcommand::Build),
            &CargoCompileContext::default(),
            None,
        )
        .expect("default Cargo context should compile");
        let command = ExternalProfileCommand {
            command: "profiler".to_string(),
            arguments: Vec::new(),
            environment: HashMap::default(),
        };
        assert!(
            compile_profile_plan(
                compiled.clone(),
                command.clone(),
                TaskArtifact {
                    path: "../private/profile.svg".to_string(),
                    kind: TaskArtifactKind::Svg,
                    max_bytes: 1024,
                },
            )
            .is_err()
        );
        assert!(
            compile_profile_plan(
                compiled,
                command,
                TaskArtifact {
                    path: "target/profile.svg".to_string(),
                    kind: TaskArtifactKind::Svg,
                    max_bytes: task::MAX_TASK_ARTIFACT_BYTES + 1,
                },
            )
            .is_err()
        );
    }

    fn native_context() -> CompiledCargoPreset {
        compile_preset(
            &CargoPreset {
                label: "selected app".to_string(),
                pre_launch_task: Some("prepare application".to_string()),
                environment: HashMap::from_iter([(
                    "PRESET_ENV".to_string(),
                    "preserved".to_string(),
                )]),
                working_directory: CargoWorkingDirectoryPolicy::Workspace,
                ..CargoPreset::ephemeral_default(CargoSubcommand::Run)
            },
            &CargoCompileContext {
                workspace_cwd: Some("/host/workspace".to_string()),
                ..CargoCompileContext::default()
            },
            None,
        )
        .expect("native Cargo context")
    }

    fn native_collector() -> SamplyCollector {
        SamplyCollector {
            command: "/host/bin/samply".to_string(),
            version: "0.13.1".to_string(),
        }
    }

    fn native_task_context() -> TaskContext {
        TaskContext {
            task_variables: TaskVariables::from_iter([
                (VariableName::File, "selected.rs".to_string()),
                (VariableName::WorktreeRoot, "/host/workspace".to_string()),
            ]),
            ..TaskContext::default()
        }
    }

    fn native_workload() -> ExternalProfileCommand {
        ExternalProfileCommand {
            command: "/host/workspace/target/debug/selected app".to_string(),
            arguments: vec![
                "--output".to_string(),
                "workload data".to_string(),
                "--symbol-server".to_string(),
                "$ZED_FILE".to_string(),
                "${ZED_FILE}".to_string(),
                "${ZED_UNKNOWN:fallback}".to_string(),
            ],
            environment: HashMap::from_iter([("WORKLOAD_ENV".to_string(), "explicit".to_string())]),
        }
    }

    #[test]
    fn cargo_native_profile_plan_declares_exact_pair_and_preserves_context() {
        let context = native_context();
        let expected_context = context.task_context.clone();
        let expected_save = context.task_template.save;
        let plan = compile_native_profile_plan(
            context,
            "macos",
            &native_collector(),
            native_workload(),
            "target/native capture.json",
        )
        .expect("explicit native plan");
        assert_eq!(plan.task_context, expected_context);
        assert_eq!(plan.pre_launch_task.as_deref(), Some("prepare application"));
        assert_eq!(plan.task_template.cwd.as_deref(), Some("/host/workspace"));
        assert_eq!(plan.task_template.save, expected_save);
        assert_eq!(
            plan.task_template.env.get("PRESET_ENV").map(String::as_str),
            Some("preserved")
        );
        assert_eq!(
            plan.task_template
                .env
                .get("WORKLOAD_ENV")
                .map(String::as_str),
            Some("explicit")
        );
        assert!(plan.task_template.artifact.is_none());
        assert!(plan.task_template.literal_arguments);
        assert!(
            !plan
                .task_template
                .tags
                .contains(&"external-tool".to_string())
        );
        let resolved = plan
            .task_template
            .resolve_task("native-profile", &native_task_context())
            .expect("real generic Task resolution");
        assert_eq!(
            resolved.resolved.literal_prefix_args,
            resolved.resolved.args.len()
        );
        assert_eq!(resolved.resolved.args, plan.task_template.args);
        let group = resolved
            .resolved_artifact_group()
            .expect("resolved native pair");
        assert_eq!(group.format, "samply-0.13.1");
        assert_eq!(group.files.len(), 2);
        let profile = group.files.get("profile").expect("profile role");
        let symbols = group.files.get("symbols").expect("symbols role");
        assert_eq!(profile.path, "target/native capture.json");
        assert_eq!(symbols.path, "target/native capture.syms.json");
        assert_eq!(profile.kind, TaskArtifactKind::Data);
        assert_eq!(symbols.kind, TaskArtifactKind::Data);
        assert_eq!(profile.max_bytes, 32 * 1024 * 1024);
        assert_eq!(symbols.max_bytes, 16 * 1024 * 1024);
        assert_eq!(plan.task_template.command, "/host/bin/samply");
        assert_eq!(
            plan.task_template.args,
            [
                "record",
                "--save-only",
                "--no-open",
                "--main-thread-only",
                "--unstable-presymbolicate",
                "--include-args=0",
                "--duration",
                "30",
                "-r",
                "100",
                "-o",
                "target/native capture.json",
                "--",
                "/host/workspace/target/debug/selected app",
                "--output",
                "workload data",
                "--symbol-server",
                "$ZED_FILE",
                "${ZED_FILE}",
                "${ZED_UNKNOWN:fallback}"
            ]
        );
    }

    #[test]
    fn cargo_native_profile_plan_rejects_unsupported_host_or_collector() {
        for host in ["linux", "windows", "unknown", ""] {
            assert!(
                compile_native_profile_plan(
                    native_context(),
                    host,
                    &native_collector(),
                    native_workload(),
                    "target/capture.json"
                )
                .is_err()
            );
        }
        for version in ["0.13.0", "0.13.2", "", "0.13.1-custom"] {
            let mut collector = native_collector();
            collector.version = version.to_string();
            assert!(
                compile_native_profile_plan(
                    native_context(),
                    "macos",
                    &collector,
                    native_workload(),
                    "target/capture.json"
                )
                .is_err()
            );
        }
        for command in [
            "samply",
            "~/bin/samply",
            "/host/bin/samply; echo private",
            "/host/application support/samply",
        ] {
            let mut collector = native_collector();
            collector.command = command.to_string();
            assert!(
                compile_native_profile_plan(
                    native_context(),
                    "macos",
                    &collector,
                    native_workload(),
                    "target/capture.json"
                )
                .is_err()
            );
        }
    }

    #[test]
    fn cargo_native_profile_plan_rejects_invalid_pair_and_workload() {
        for (path, substituted_path) in [
            ("target/$ZED_FILE.json", "target/selected.rs.json"),
            ("target/${ZED_FILE}.json", "target/selected.rs.json"),
            (
                "target/${ZED_UNKNOWN:fallback}.json",
                "target/fallback.json",
            ),
        ] {
            let generic_template = task::TaskTemplate {
                label: "literal metadata precondition".to_string(),
                command: "/host/bin/samply".to_string(),
                args: vec![path.to_string()],
                literal_arguments: true,
                artifact: Some(TaskArtifact {
                    path: path.to_string(),
                    kind: TaskArtifactKind::Data,
                    max_bytes: 1024,
                }),
                ..task::TaskTemplate::default()
            };
            let generic_resolved = generic_template
                .resolve_task("metadata-precondition", &native_task_context())
                .expect("generic metadata substitutes real Task context");
            assert_eq!(
                generic_resolved.resolved.args.first().map(String::as_str),
                Some(path)
            );
            assert_eq!(
                generic_resolved
                    .resolved_artifact()
                    .expect("metadata artifact")
                    .path,
                substituted_path
            );
            assert!(
                compile_native_profile_plan(
                    native_context(),
                    "macos",
                    &native_collector(),
                    native_workload(),
                    path,
                )
                .is_err()
            );
        }
        for path in [
            "target/capture.json.gz",
            "target/capture.syms.json",
            "../capture.json",
            "/host/capture.json",
            "https://example.invalid/capture.json",
            "target\\capture.json",
            "./capture.json",
            "target//capture.json",
            "-capture.json",
            "target/$ZED_FILE.json",
            "target/capture",
        ] {
            assert!(
                compile_native_profile_plan(
                    native_context(),
                    "macos",
                    &native_collector(),
                    native_workload(),
                    path
                )
                .is_err()
            );
        }
        for command in [
            "cargo",
            "target/selected-app",
            "$ZED_WORKTREE_ROOT/selected-app",
            "/host/$ZED_FILE",
            "/host/${ZED_FILE}",
            "/host/${ZED_UNKNOWN:fallback}",
            "/host/workload=private",
            "/host/invalid\0workload",
        ] {
            let mut workload = native_workload();
            workload.command = command.to_string();
            assert!(
                compile_native_profile_plan(
                    native_context(),
                    "macos",
                    &native_collector(),
                    workload,
                    "target/capture.json"
                )
                .is_err()
            );
        }
        let longest_profile = format!(
            "{}.json",
            "p".repeat(task::MAX_TASK_ARTIFACT_PATH_BYTES - 5)
        );
        assert!(
            compile_native_profile_plan(
                native_context(),
                "macos",
                &native_collector(),
                native_workload(),
                &longest_profile
            )
            .is_err()
        );
        let mut workload = native_workload();
        workload.arguments = vec!["arg".to_string(); MAX_PROFILE_COMMAND_ITEMS];
        assert!(
            compile_native_profile_plan(
                native_context(),
                "macos",
                &native_collector(),
                workload,
                "target/capture.json"
            )
            .is_err()
        );
        let mut workload = native_workload();
        workload.arguments.push("invalid\0argument".to_string());
        assert!(
            compile_native_profile_plan(
                native_context(),
                "macos",
                &native_collector(),
                workload,
                "target/capture.json"
            )
            .is_err()
        );
    }

    #[test]
    fn cargo_native_profile_plan_binds_pairs_for_cached_and_persisted_rerun() {
        let plan = compile_native_profile_plan(
            native_context(),
            "macos",
            &native_collector(),
            native_workload(),
            "target/native capture.json",
        )
        .expect("native plan");
        let original = plan
            .task_template
            .resolve_task("native-profile", &native_task_context())
            .expect("unbound native task");
        let first = original
            .bind_artifact_invocation("11223344-5566-7788-99AA-BBCCDDEEFF00")
            .expect("first invocation pair");
        let second = first
            .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
            .expect("cached rerun pair");
        for (bound, identifier) in [
            (&first, "112233445566778899aabbccddeeff00"),
            (&second, "99887766554433221100ffeeddccbbaa"),
        ] {
            let group = bound.resolved_artifact_group().expect("bound group");
            let profile = group.files.get("profile").expect("profile role");
            let symbols = group.files.get("symbols").expect("symbols role");
            assert_eq!(
                profile.path,
                format!("target/native capture-{identifier}.json")
            );
            assert_eq!(
                symbols.path,
                format!("target/native capture-{identifier}.syms.json")
            );
            assert_eq!(bound.resolved.args.get(11), Some(&profile.path));
            assert_eq!(group.format, "samply-0.13.1");
            assert_eq!(profile.max_bytes, 32 * 1024 * 1024);
            assert_eq!(symbols.max_bytes, 16 * 1024 * 1024);
            assert_eq!(
                bound.resolved.args.get(17..),
                original.resolved.args.get(17..)
            );
            assert_eq!(bound.id, original.id);
            assert_eq!(bound.original_task(), &plan.task_template);
        }
        let persisted =
            serde_json::to_string(second.original_task()).expect("persist original template");
        let restored: task::TaskTemplate =
            serde_json::from_str(&persisted).expect("restore original template");
        assert!(restored.literal_arguments);
        assert_eq!(
            second,
            restored
                .resolve_task("native-profile", &native_task_context())
                .expect("restored context")
                .bind_artifact_invocation("99887766554433221100ffeeddccbbaa")
                .expect("persisted rerun pair")
        );
        assert_eq!(
            original.resolved.args.get(11).map(String::as_str),
            Some("target/native capture.json")
        );
    }

    #[test]
    fn native_profile_settings_round_trip_is_declarative_without_authority() {
        let content = CargoNativeProfilerSettingsContent {
            collector_path: Some("/opt/tools/samply".into()),
            profile_path: Some("target/native-capture.json".into()),
        };
        let declaration =
            CargoNativeProfiler::from_settings(&content).expect("bounded explicit native settings");
        assert_eq!(declaration.settings_content(), content);
        let serialized = serde_json::to_value(content).expect("native settings JSON");
        assert!(serialized.get("version").is_none());
        assert!(serialized.get("success").is_none());
        assert!(
            serde_json::from_value::<CargoNativeProfilerSettingsContent>(
                serde_json::json!({"cargo_path": "/opt/tools/cargo", "success": true})
            )
            .is_err()
        );
    }

    #[test]
    fn native_profile_settings_reject_unsafe_and_unresolved_inputs() {
        let valid = CargoNativeProfilerSettingsContent {
            collector_path: Some("/opt/tools/samply".into()),
            profile_path: Some("target/native-capture.json".into()),
        };
        for command in [
            "",
            "cargo",
            "/tmp/../cargo",
            "/tmp/$(collector)",
            "/tmp/tool\n",
            "/tmp/tool with spaces",
        ] {
            let mut invalid = valid.clone();
            invalid.collector_path = Some(command.into());
            assert!(CargoNativeProfiler::from_settings(&invalid).is_err());
        }
        for output in [
            "",
            "../profile.json",
            "target/capture.syms.json",
            "target/$ZED_FILE.json",
            "target/${ZED_FILE}.json",
            "target/capture.svg",
            ".json",
        ] {
            let mut invalid = valid.clone();
            invalid.profile_path = Some(output.into());
            assert!(CargoNativeProfiler::from_settings(&invalid).is_err());
        }
        let mut invalid = valid;
        invalid.collector_path = Some(format!("/{}", "x".repeat(MAX_PROFILE_COMMAND_TEXT_BYTES)));
        assert!(CargoNativeProfiler::from_settings(&invalid).is_err());
    }

    fn native_selected_fixture() -> (
        crate::CargoActionSelection,
        project::cargo_workspace::CargoWorkspaceSnapshot,
    ) {
        use project::cargo_workspace::*;
        let worktree_id = project::WorktreeId::from_proto(7);
        let path = |value: &str| project::ProjectPath {
            worktree_id,
            path: util::rel_path::RelPath::new(
                std::path::Path::new(value),
                util::paths::PathStyle::Unix,
            )
            .expect("fixture project path")
            .into(),
        };
        let target = CargoTargetModel {
            name: "app".into(),
            kind: CargoTargetKind::Binary,
            crate_types: vec!["bin".into()],
            source_path: Some(path("member/src/main.rs")),
            source_display_path: None,
            required_features: vec!["selected-feature".into()],
            edition: "2021".into(),
        };
        let selection = crate::CargoActionSelection {
            node_kind: crate::CargoActionNodeKind::Target(crate::CargoActionTargetKind::Binary),
            worktree_id,
            workspace_name: "fixture".into(),
            workspace_manifest: path("Cargo.toml"),
            package_name: Some("member".into()),
            package_manifest: Some(path("member/Cargo.toml")),
            target: Some(crate::CargoTargetSelector::Binary("app".into())),
            has_bench_targets: false,
        };
        let snapshot = CargoWorkspaceSnapshot {
            revision: 1,
            input_fingerprint: 2,
            failures: Vec::new(),
            completeness: CargoSnapshotCompleteness::Complete,
            workspaces: vec![CargoWorkspaceModel {
                key: CargoWorkspaceKey { root: path("") },
                root_manifest: Some(path("Cargo.toml")),
                display_name: "fixture".into(),
                is_virtual: true,
                configuration: CargoWorkspaceConfiguration::unresolved(),
                dependency_graph: None,
                members: vec![CargoPackageModel {
                    id: "metadata-model-id".into(),
                    name: "member".into(),
                    version: "0.1.0".into(),
                    manifest_path: path("member/Cargo.toml"),
                    is_default_member: true,
                    targets: vec![target],
                    features: Vec::new(),
                    dependencies: Vec::new(),
                }],
            }],
        };
        (selection, snapshot)
    }

    #[test]
    fn native_profile_selection_retains_exact_current_target_data() {
        let (selection, mut snapshot) = native_selected_fixture();
        let (_, member, selected_target) = native_profile_selected_target(&selection, &snapshot)
            .expect("current exact selection data");
        assert_eq!(member, "metadata-model-id");
        assert_eq!(selected_target.required_features, vec!["selected-feature"]);
        let workspace = snapshot.workspaces.first_mut().expect("workspace");
        let package = workspace.members.first_mut().expect("package");
        let target = package.targets.first_mut().expect("target");
        target.kind = project::cargo_workspace::CargoTargetKind::Example;
        let mut example = selection;
        example.node_kind =
            crate::CargoActionNodeKind::Target(crate::CargoActionTargetKind::Example);
        example.target = Some(crate::CargoTargetSelector::Example("app".into()));
        assert!(native_profile_selected_target(&example, &snapshot).is_ok());
    }

    #[test]
    fn native_profile_selection_rejects_stale_ambiguous_and_unsupported_contexts() {
        let (selection, snapshot) = native_selected_fixture();
        let mut preset = crate::CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.native_profiler = Some(
            CargoNativeProfiler::from_settings(&CargoNativeProfilerSettingsContent {
                collector_path: Some("/opt/tools/samply".into()),
                profile_path: Some("target/capture.json".into()),
            })
            .expect("declarative profiler"),
        );
        validate_native_profile_selection(&selection, &preset).expect("supported default context");
        for profile in ["dev", "release", "native-review"] {
            let mut named = preset.clone();
            named.profile = Some(profile.into());
            validate_native_profile_selection(&selection, &named)
                .expect("bounded profile is declarative; current host metadata decides admission");
        }

        let mut wrong = selection.clone();
        wrong.target = Some(crate::CargoTargetSelector::Test("app".into()));
        assert!(validate_native_profile_selection(&wrong, &preset).is_err());
        assert!(native_profile_selected_target(&wrong, &snapshot).is_err());
        let mut stale = selection.clone();
        stale.package_name = Some("removed".into());
        assert!(native_profile_selected_target(&stale, &snapshot).is_err());
        let mut ambiguous = snapshot.clone();
        let duplicate = ambiguous.workspaces.first().expect("workspace").clone();
        ambiguous.workspaces.push(duplicate);
        assert!(native_profile_selected_target(&selection, &ambiguous).is_err());
        let mut ambiguous = snapshot;
        let package = ambiguous
            .workspaces
            .first_mut()
            .expect("workspace")
            .members
            .first_mut()
            .expect("package");
        package
            .targets
            .push(package.targets.first().expect("target").clone());
        assert!(native_profile_selected_target(&selection, &ambiguous).is_err());
        for variant in 0..8 {
            let mut unsupported = preset.clone();
            match variant {
                0 => unsupported.profile = Some("--config".into()),
                1 => unsupported.toolchain = Some("nightly".into()),
                2 => unsupported.target_triple = Some("aarch64-apple-darwin".into()),
                3 => unsupported.features.push("extra".into()),
                4 => unsupported.default_features = Some(false),
                5 => unsupported.args.push("--config".into()),
                6 => unsupported.working_directory = crate::CargoWorkingDirectoryPolicy::Package,
                _ => {
                    unsupported.working_directory =
                        crate::CargoWorkingDirectoryPolicy::Custom("/tmp/elsewhere".into())
                }
            }
            assert!(validate_native_profile_selection(&selection, &unsupported).is_err());
        }
    }

    #[test]
    fn native_profile_build_template_retains_declarative_intent_for_every_rerun() {
        let mut preset = crate::CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.native_profiler = Some(
            CargoNativeProfiler::from_settings(&CargoNativeProfilerSettingsContent {
                collector_path: Some("/opt/tools/samply".into()),
                profile_path: Some("target/capture.json".into()),
            })
            .expect("native declaration"),
        );
        preset.trailing_args = vec![
            "argument with spaces".into(),
            "$(literal-workload)".into(),
            "a; b".into(),
        ];
        preset.presentation.save = task::SaveStrategy::All;
        let build_arguments = vec![
            "run".into(),
            "--offline".into(),
            "--message-format=json".into(),
            "--package".into(),
            "member".into(),
            "--bin".into(),
            "app".into(),
            "--target-dir".into(),
            "/host/workspace/target".into(),
            "--".into(),
        ]
        .into_iter()
        .chain(preset.trailing_args.iter().cloned())
        .collect::<Vec<String>>();
        let recipe = project::cargo_workspace_store::NativeCargoBuildRecipe {
            program: "/opt/tools/cargo".into(),
            arguments: build_arguments.clone(),
            working_directory: "/host/workspace".into(),
            task_environment: HashMap::from_iter([("PRESET_ENV".into(), "literal value".into())]),
        };
        let template =
            native_profile_build_template(&preset, recipe).expect("declarative Build template");
        assert_eq!(template.command, "/opt/tools/cargo");
        assert_eq!(template.args, build_arguments);
        assert_eq!(template.save, task::SaveStrategy::All);
        assert!(template.literal_arguments);
        assert!(template.artifact_group.is_none());
        let report = template
            .stdout_report
            .as_ref()
            .expect("existing bounded report request");
        assert_eq!(
            report.format,
            task::TaskStdoutReportFormat::CargoRuntimePreparationJson
        );
        let continuation = report
            .native_profile
            .as_ref()
            .expect("capture continuation data");
        assert_eq!(continuation.workload_arguments, preset.trailing_args);
        assert_eq!(continuation.collector_path, "/opt/tools/samply");
        assert_eq!(continuation.profile_path, "target/capture.json");
        let context = task::TaskContext::default();
        let initial = template
            .resolve_task("Cargo", &context)
            .expect("initial Build intent");
        let cached = initial.clone();
        let persisted: task::TaskTemplate =
            serde_json::from_value(serde_json::to_value(&template).expect("persisted intent JSON"))
                .expect("declarative reload");
        let reevaluated = persisted
            .resolve_task("Cargo", &context)
            .expect("persisted template Rerun intent");
        for rerun in [cached, reevaluated] {
            assert_eq!(rerun.id, initial.id);
            assert_eq!(rerun.resolved.command.as_deref(), Some("/opt/tools/cargo"));
            assert_eq!(rerun.resolved.args, build_arguments);
            assert_eq!(rerun.stdout_report(), template.stdout_report.as_ref());
            assert_eq!(rerun.original_task(), &template);
        }
    }

    #[test]
    fn native_profile_runtime_template_keeps_literal_selection_and_renews_only_data() {
        let mut preset = crate::CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.native_profiler = Some(
            CargoNativeProfiler::from_settings(&CargoNativeProfilerSettingsContent {
                collector_path: Some("/opt/tools/samply".into()),
                profile_path: Some("target/capture.json".into()),
            })
            .expect("native declaration"),
        );
        let context = TaskContext {
            cwd: Some("/different/editor/directory".into()),
            task_variables: TaskVariables::from_iter([(VariableName::File, "changed.rs".into())]),
            ..Default::default()
        };
        for selector in ["--bin", "--example"] {
            for operands in [
                Vec::<String>::new(),
                vec![
                    "".into(),
                    "$ZED_FILE".into(),
                    "${ZED_FILE}".into(),
                    "$(literal); * --target".into(),
                ],
            ] {
                preset.trailing_args = operands.clone();
                let arguments = [
                    "run",
                    "--offline",
                    "--message-format=json",
                    "--package",
                    "raw-member",
                    selector,
                    "selected-target",
                    "--target-dir",
                    "/host/workspace/target",
                    "--",
                ]
                .into_iter()
                .map(String::from)
                .chain(operands.iter().cloned())
                .collect::<Vec<_>>();
                let template = native_profile_build_template(
                    &preset,
                    project::cargo_workspace_store::NativeCargoBuildRecipe {
                        program: "/opt/tools/cargo".into(),
                        arguments: arguments.clone(),
                        working_directory: "/host/workspace".into(),
                        task_environment: HashMap::from_iter([(
                            "PRESET_ENV".into(),
                            "value".into(),
                        )]),
                    },
                )
                .expect("runtime preparation data template");
                let serialized = serde_json::to_value(&template).expect("declarative history");
                let persisted: task::TaskTemplate =
                    serde_json::from_value(serialized.clone()).expect("persisted runtime template");
                let initial = template
                    .resolve_task("Cargo", &context)
                    .expect("initial intent");
                let reevaluated = persisted
                    .resolve_task("Cargo", &context)
                    .expect("renewed template intent");
                let mut cached = initial.clone();
                for identifier in [
                    "112233445566778899aabbccddeeff00",
                    "99887766554433221100ffeeddccbbaa",
                ] {
                    cached = cached
                        .bind_artifact_invocation(identifier)
                        .expect("Build has no capture artifact authority");
                    for replay in [&cached, &reevaluated] {
                        assert_eq!(replay.id, initial.id);
                        assert_eq!(replay.original_task(), &template);
                        assert_eq!(replay.resolved.args, arguments);
                        assert_eq!(replay.resolved.literal_prefix_args, arguments.len());
                        assert_eq!(
                            replay.resolved.cwd.as_deref(),
                            Some(std::path::Path::new("/host/workspace"))
                        );
                        assert!(replay.resolved_artifact_group().is_none());
                        assert!(replay.artifact_directory().is_none());
                        let report = replay.stdout_report().expect("finite runtime descriptor");
                        assert_eq!(
                            report.format,
                            task::TaskStdoutReportFormat::CargoRuntimePreparationJson
                        );
                        assert_eq!(
                            report
                                .native_profile
                                .as_ref()
                                .expect("data continuation")
                                .workload_arguments,
                            operands
                        );
                    }
                }
                let report = serialized
                    .get("stdout_report")
                    .expect("serialized report data");
                let report = report.as_object().expect("report object");
                assert!(report.keys().all(|name| matches!(
                    name.as_str(),
                    "format" | "max_bytes" | "native_profile"
                )));
                let continuation = report
                    .get("native_profile")
                    .expect("continuation data")
                    .as_object()
                    .expect("continuation object");
                assert!(continuation.keys().all(|name| matches!(
                    name.as_str(),
                    "collector_path" | "profile_path" | "workload_arguments"
                )));
            }
        }
    }

    #[test]
    fn native_profile_runtime_template_refuses_oversized_or_invalid_continuation() {
        let mut preset = crate::CargoPreset::ephemeral_default(crate::CargoSubcommand::Run);
        preset.native_profiler = Some(
            CargoNativeProfiler::from_settings(&CargoNativeProfilerSettingsContent {
                collector_path: Some("/opt/tools/samply".into()),
                profile_path: Some("target/capture.json".into()),
            })
            .expect("native declaration"),
        );
        for operands in [
            vec![String::new(); 129],
            vec!["x".repeat(4097)],
            vec!["nul\0operand".into()],
        ] {
            preset.trailing_args = operands;
            assert!(
                native_profile_build_template(
                    &preset,
                    project::cargo_workspace_store::NativeCargoBuildRecipe {
                        program: "/opt/tools/cargo".into(),
                        arguments: vec!["run".into()],
                        working_directory: "/host/workspace".into(),
                        task_environment: HashMap::default(),
                    },
                )
                .is_err()
            );
        }
    }
}
