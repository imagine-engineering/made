use feature_flags::{
    FeatureFlagAppExt as _, RustCoverageExperimentalFeatureFlag,
    RustNamedCargoPresetsExperimentalFeatureFlag, RustNativeProfilingExperimentalFeatureFlag,
};

use anyhow::{Result, anyhow, bail};
use gpui::{Context, Window, actions};
use project::{
    ProjectPath, TaskSourceKind, WorktreeId,
    rust_coverage_provider::{MAX_RUST_COVERAGE_ARTIFACT_BYTES, RUST_COVERAGE_PROVIDER_ID},
    source_coverage::{
        SourceCoverageProviderId, SourceCoverageSourceChanged, SourceCoverageStatus,
    },
};
use task::{DebugScenario, SharedTaskContext, TaskArtifactKind, TaskContext, TaskTemplate};
use workspace::{
    Toast, Workspace,
    notifications::NotificationId,
    tasks::{ScheduledTaskResult, TaskInvocationContext},
};

use crate::{
    CARGO_COVERAGE_FAILURE_GUIDANCE, CargoCompileContext, CargoPreset, CargoPresetScope,
    CargoSubcommand, CargoTargetSelector, compile_coverage_plan, compile_debug_scenario,
    compile_preset,
};

actions!(
    cargo_actions,
    [
        BuildSelected,
        CheckSelected,
        RunSelected,
        RunWithCoverageSelected,
        ProfileSelected,
        TestSelected,
        BenchSelected,
        DebugSelected,
        DocSelected,
        ClippySelected,
        FmtSelected,
        CleanSelected,
        TreeSelected
    ]
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoAction {
    Build,
    Check,
    Run,
    RunWithCoverage,
    Profile,
    Test,
    Bench,
    Debug,
    Doc,
    Clippy,
    Fmt,
    Clean,
    Tree,
}

impl CargoAction {
    pub const ALL: [Self; 13] = [
        Self::Build,
        Self::Check,
        Self::Run,
        Self::RunWithCoverage,
        Self::Profile,
        Self::Test,
        Self::Bench,
        Self::Debug,
        Self::Doc,
        Self::Clippy,
        Self::Fmt,
        Self::Clean,
        Self::Tree,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Build => "Build",
            Self::Check => "Check",
            Self::Run => "Run",
            Self::RunWithCoverage => "Run with Coverage",
            Self::Profile => "Profile",
            Self::Test => "Test",
            Self::Bench => "Bench",
            Self::Debug => "Debug",
            Self::Doc => "Doc",
            Self::Clippy => "Clippy",
            Self::Fmt => "Fmt",
            Self::Clean => "Clean",
            Self::Tree => "Tree",
        }
    }

    fn subcommand(self, selection: &CargoActionSelection) -> Result<CargoSubcommand> {
        match self {
            Self::Build => Ok(CargoSubcommand::Build),
            Self::Check => Ok(CargoSubcommand::Check),
            Self::Run | Self::RunWithCoverage => Ok(CargoSubcommand::Run),
            Self::Test => Ok(CargoSubcommand::Test),
            Self::Bench => Ok(CargoSubcommand::Bench),
            Self::Doc => Ok(CargoSubcommand::Doc),
            Self::Clippy => Ok(CargoSubcommand::Clippy),
            Self::Fmt => Ok(CargoSubcommand::Fmt),
            Self::Clean => Ok(CargoSubcommand::Clean),
            Self::Tree => Ok(CargoSubcommand::Tree),
            Self::Debug | Self::Profile => match selection.node_kind {
                CargoActionNodeKind::Target(CargoActionTargetKind::Binary)
                | CargoActionNodeKind::Target(CargoActionTargetKind::Example) => {
                    Ok(CargoSubcommand::Run)
                }
                CargoActionNodeKind::Target(CargoActionTargetKind::Test) => {
                    Ok(CargoSubcommand::Test)
                }
                CargoActionNodeKind::Target(CargoActionTargetKind::Bench) => {
                    Ok(CargoSubcommand::Bench)
                }
                _ => bail!("the selected Cargo node cannot run this action"),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoActionTargetKind {
    Library,
    Binary,
    Example,
    Test,
    Bench,
    BuildScript,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoActionNodeKind {
    Workspace,
    Package,
    Target(CargoActionTargetKind),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoActionSelection {
    pub node_kind: CargoActionNodeKind,
    pub worktree_id: WorktreeId,
    pub workspace_name: String,
    pub workspace_manifest: ProjectPath,
    pub package_name: Option<String>,
    pub package_manifest: Option<ProjectPath>,
    pub target: Option<CargoTargetSelector>,
    pub has_bench_targets: bool,
}

impl CargoActionSelection {
    pub fn scope(&self) -> CargoPresetScope {
        match self.node_kind {
            CargoActionNodeKind::Workspace => CargoPresetScope::Workspace,
            CargoActionNodeKind::Package | CargoActionNodeKind::Target(_) => {
                CargoPresetScope::Package
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CargoActionRuntime {
    pub trusted: bool,
    pub connected: bool,
    pub writable: bool,
    pub cargo_available: bool,
    pub host_capable: bool,
}

impl Default for CargoActionRuntime {
    fn default() -> Self {
        Self {
            trusted: true,
            connected: true,
            writable: true,
            cargo_available: true,
            host_capable: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoActionAvailability {
    pub action: CargoAction,
    pub enabled: bool,
    pub reason: Option<String>,
    pub accessibility_label: String,
}

pub fn cargo_action_availability(
    selection: &CargoActionSelection,
    runtime: CargoActionRuntime,
) -> Vec<CargoActionAvailability> {
    CargoAction::ALL
        .into_iter()
        .map(|action| {
            let reason = action_inapplicable_reason(action, selection)
                .map(str::to_string)
                .or_else(|| runtime_denial_reason(runtime).map(str::to_string));
            let accessibility_label = reason
                .as_ref()
                .map(|reason| format!("{} unavailable: {reason}", action.label()))
                .unwrap_or_else(|| format!("{} selected Cargo item", action.label()));
            CargoActionAvailability {
                action,
                enabled: reason.is_none(),
                reason,
                accessibility_label,
            }
        })
        .collect()
}

pub fn cargo_action_availability_with_preset(
    selection: &CargoActionSelection,
    runtime: CargoActionRuntime,
    preset: Option<&CargoPreset>,
) -> Vec<CargoActionAvailability> {
    let mut entries = cargo_action_availability(selection, runtime);
    if let Some(preset) = preset.filter(|preset| preset.native_profiler.is_some()) {
        for entry in &mut entries {
            if entry.action == CargoAction::Profile {
                entry.reason = entry.reason.take().or_else(|| {
                    crate::cargo_profile::validate_native_profile_selection(selection, preset)
                        .err()
                        .map(|error| error.to_string())
                });
                entry.enabled = entry.reason.is_none();
                entry.accessibility_label = entry
                    .reason
                    .as_ref()
                    .map(|reason| {
                        format!("{} unavailable: {reason}", preset.profile_action_label())
                    })
                    .unwrap_or_else(|| {
                        format!("{} selected Cargo item", preset.profile_action_label())
                    });
            }
        }
    }
    entries
}

fn runtime_denial_reason(runtime: CargoActionRuntime) -> Option<&'static str> {
    if !runtime.trusted {
        Some("trust this worktree before running Cargo commands")
    } else if !runtime.connected {
        Some("the authoritative project host is disconnected")
    } else if !runtime.writable {
        Some("multiplayer guests with read-only access cannot run Cargo commands")
    } else if !runtime.cargo_available {
        Some("Cargo is not available on the authoritative project host")
    } else if !runtime.host_capable {
        Some("the authoritative project host does not support Tasks and DAP")
    } else {
        None
    }
}

fn action_inapplicable_reason(
    action: CargoAction,
    selection: &CargoActionSelection,
) -> Option<&'static str> {
    use CargoActionNodeKind::{Package, Target, Workspace};
    use CargoActionTargetKind::{Bench, Binary, BuildScript, Example, Library, Other, Test};

    let applicable = match (selection.node_kind, action) {
        (
            Workspace | Package,
            CargoAction::Build
            | CargoAction::Check
            | CargoAction::Test
            | CargoAction::Doc
            | CargoAction::Clippy
            | CargoAction::Fmt
            | CargoAction::Clean
            | CargoAction::Tree,
        ) => true,
        (Workspace | Package, CargoAction::Bench) => selection.has_bench_targets,
        (
            Workspace | Package,
            CargoAction::Run
            | CargoAction::RunWithCoverage
            | CargoAction::Debug
            | CargoAction::Profile,
        ) => false,
        (
            Target(Library),
            CargoAction::Build
            | CargoAction::Check
            | CargoAction::Test
            | CargoAction::Doc
            | CargoAction::Clippy,
        ) => true,
        (
            Target(Binary | Example),
            CargoAction::Build | CargoAction::Check | CargoAction::Doc | CargoAction::Clippy,
        ) => true,
        (
            Target(Binary | Example),
            CargoAction::Run
            | CargoAction::RunWithCoverage
            | CargoAction::Test
            | CargoAction::Debug,
        ) => true,
        (
            Target(Test),
            CargoAction::Build | CargoAction::Check | CargoAction::Test | CargoAction::Clippy,
        ) => true,
        (Target(Test), CargoAction::Debug) => true,
        (Target(Binary | Example | Test | Bench), CargoAction::Profile) => true,
        (
            Target(Bench),
            CargoAction::Build | CargoAction::Check | CargoAction::Bench | CargoAction::Clippy,
        ) => true,
        (Target(Bench), CargoAction::Debug) => true,
        (Target(BuildScript | Other), CargoAction::Build | CargoAction::Check) => true,
        _ => false,
    };
    (!applicable).then_some(match action {
        CargoAction::Run | CargoAction::RunWithCoverage => {
            "Run requires a binary or example target"
        }
        CargoAction::Bench => "Bench requires a bench target",
        CargoAction::Debug => "Debug requires a binary, example, test, or bench target",
        CargoAction::Profile => "Profile requires a binary, example, test, or bench target",
        CargoAction::Test => "Test is not applicable to this target kind",
        CargoAction::Doc => "Doc requires a workspace, package, library, binary, or example",
        CargoAction::Clippy => "Clippy is not applicable to this target kind",
        CargoAction::Fmt | CargoAction::Clean | CargoAction::Tree => {
            "the action requires a workspace or package"
        }
        CargoAction::Build | CargoAction::Check => "the action is not applicable to this node",
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CargoActionPlan {
    NativeProfile {
        selection: CargoActionSelection,
        preset: CargoPreset,
        context: TaskContext,
        worktree_id: WorktreeId,
        pre_launch_task: Option<String>,
        requires_named_preset: bool,
    },
    Task {
        source: TaskSourceKind,
        template: TaskTemplate,
        context: TaskContext,
        worktree_id: WorktreeId,
        pre_launch_task: Option<String>,
        requires_named_preset: bool,
        omit_history: bool,
    },
    Debug {
        scenario: DebugScenario,
        context: TaskContext,
        worktree_id: WorktreeId,
        pre_launch_task: Option<String>,
        requires_named_preset: bool,
    },
}

impl CargoActionPlan {
    fn validate_experimental_access(&self, cx: &gpui::App) -> Result<()> {
        let requires_named_preset = match self {
            Self::Task {
                requires_named_preset,
                ..
            }
            | Self::Debug {
                requires_named_preset,
                ..
            }
            | Self::NativeProfile {
                requires_named_preset,
                ..
            } => *requires_named_preset,
        };
        anyhow::ensure!(
            !requires_named_preset || cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
            "Named Cargo preset opt-in was revoked before planning"
        );
        match self {
            Self::NativeProfile { .. } => anyhow::ensure!(
                cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
                "Native profiling requires the rust-native-profiling-experimental opt-in"
            ),
            Self::Task { template, .. } => ensure_experimental_template_access(template, cx)?,
            Self::Debug { .. } => {}
        }
        Ok(())
    }

    fn experimental_requirements(&self) -> Result<Vec<String>> {
        let (requires_named_preset, template, native) = match self {
            Self::Task {
                requires_named_preset,
                template,
                ..
            } => (*requires_named_preset, Some(template), false),
            Self::Debug {
                requires_named_preset,
                scenario,
                ..
            } => (
                *requires_named_preset,
                match scenario.build.as_ref() {
                    Some(task::BuildTaskDefinition::Template { task_template, .. }) => {
                        Some(task_template)
                    }
                    _ => None,
                },
                false,
            ),
            Self::NativeProfile {
                requires_named_preset,
                ..
            } => (*requires_named_preset, None, true),
        };
        let mut requirements = Vec::new();
        if requires_named_preset {
            requirements.push("rust-named-cargo-presets-experimental".to_string());
        }
        if native {
            requirements.push("rust-native-profiling-experimental".to_string());
        }
        if let Some(template) = template {
            task::validate_experimental_feature_requirements(
                &template.required_experimental_features,
            )?;
            for feature in &template.required_experimental_features {
                if !requirements.contains(feature) {
                    requirements.push(feature.clone());
                }
            }
            let coverage = template
                .artifact_group
                .as_ref()
                .is_some_and(|group| group.format == crate::CARGO_COVERAGE_ARTIFACT_FORMAT)
                || template.artifact.as_ref().is_some_and(|artifact| {
                    artifact.kind == TaskArtifactKind::Data
                        && artifact.path == crate::CARGO_COVERAGE_ARTIFACT_PATH
                });
            let native = template
                .stdout_report
                .as_ref()
                .is_some_and(|report| report.native_profile.is_some());
            for (required, feature) in [
                (coverage, "rust-coverage-experimental"),
                (native, "rust-native-profiling-experimental"),
            ] {
                if required
                    && !requirements
                        .iter()
                        .any(|requirement| requirement == feature)
                {
                    requirements.push(feature.to_string());
                }
            }
        }
        task::validate_experimental_feature_requirements(&requirements)?;
        Ok(requirements)
    }

    fn pre_launch(&self) -> Option<(&str, &TaskContext, WorktreeId)> {
        match self {
            Self::Task {
                context,
                worktree_id,
                pre_launch_task,
                ..
            }
            | Self::Debug {
                context,
                worktree_id,
                pre_launch_task,
                ..
            }
            | Self::NativeProfile {
                context,
                worktree_id,
                pre_launch_task,
                ..
            } => pre_launch_task
                .as_deref()
                .map(|reference| (reference, context, *worktree_id)),
        }
    }

    fn clear_pre_launch(&mut self) {
        match self {
            Self::Task {
                pre_launch_task, ..
            }
            | Self::Debug {
                pre_launch_task, ..
            }
            | Self::NativeProfile {
                pre_launch_task, ..
            } => *pre_launch_task = None,
        }
    }
}

pub fn plan_cargo_action(
    action: CargoAction,
    selection: &CargoActionSelection,
    active_preset: Option<&CargoPreset>,
    base_context: &TaskContext,
) -> Result<CargoActionPlan> {
    plan_cargo_action_with_confirmation(action, selection, active_preset, base_context, false)
}

pub fn plan_cargo_action_with_confirmation(
    action: CargoAction,
    selection: &CargoActionSelection,
    active_preset: Option<&CargoPreset>,
    base_context: &TaskContext,
    clean_confirmed: bool,
) -> Result<CargoActionPlan> {
    if let Some(reason) = action_inapplicable_reason(action, selection) {
        bail!("{} is unavailable: {reason}", action.label());
    }
    if action == CargoAction::Clean && !clean_confirmed {
        bail!("Clean requires explicit confirmation before scheduling");
    }
    let subcommand = action.subcommand(selection)?;
    let mut preset = active_preset
        .cloned()
        .unwrap_or_else(|| CargoPreset::ephemeral_default(subcommand));
    preset.scope = selection.scope();
    preset.package = selection.package_name.clone();
    preset.target = selection.target.clone();

    if action == CargoAction::Profile && preset.native_profiler.is_some() {
        crate::cargo_profile::validate_native_profile_selection(selection, &preset)?;
        return Ok(CargoActionPlan::NativeProfile {
            selection: selection.clone(),
            pre_launch_task: preset.pre_launch_task.clone(),
            requires_named_preset: active_preset.is_some(),
            preset,
            context: selection_task_context(base_context, &selection.workspace_manifest)?,
            worktree_id: selection.worktree_id,
        });
    }

    let workspace_cwd = manifest_directory_template(&selection.workspace_manifest);
    let package_cwd = selection
        .package_manifest
        .as_ref()
        .map(manifest_directory_template);
    let mut compiled = compile_preset(
        &preset,
        &CargoCompileContext {
            workspace_name: Some(selection.workspace_name.clone()),
            workspace_cwd: Some(workspace_cwd),
            package_name: selection.package_name.clone(),
            package_cwd,
        },
        Some(subcommand),
    )?;
    if active_preset.is_some() {
        compiled
            .task_template
            .required_experimental_features
            .push("rust-named-cargo-presets-experimental".into());
    }
    normalize_extended_action_args(action, &mut compiled.task_template.args);
    if action == CargoAction::RunWithCoverage {
        compiled = compile_coverage_plan(compiled)?;
    }
    if action == CargoAction::Profile {
        let profiler = preset.profiler.as_ref().ok_or_else(|| {
            anyhow!(
                "Configure a profiler command and bounded artifact in the active Cargo preset first"
            )
        })?;
        compiled = crate::compile_profile_plan(
            compiled,
            profiler.command.clone(),
            profiler.artifact.clone(),
        )?;
    }
    let context = selection_task_context(base_context, &selection.workspace_manifest)?;
    if action == CargoAction::Debug {
        return Ok(CargoActionPlan::Debug {
            scenario: compile_debug_scenario(&compiled, None)?,
            context,
            worktree_id: selection.worktree_id,
            pre_launch_task: compiled.pre_launch_task,
            requires_named_preset: active_preset.is_some(),
        });
    }
    Ok(CargoActionPlan::Task {
        source: TaskSourceKind::Language {
            name: "Cargo".into(),
        },
        template: compiled.task_template,
        context,
        worktree_id: selection.worktree_id,
        pre_launch_task: compiled.pre_launch_task,
        requires_named_preset: active_preset.is_some(),
        omit_history: false,
    })
}

fn normalize_extended_action_args(action: CargoAction, args: &mut Vec<String>) {
    let subcommand_index = usize::from(
        args.first()
            .is_some_and(|argument| argument.starts_with('+')),
    );
    let scope_index = subcommand_index + 1;
    match action {
        CargoAction::Fmt
            if args
                .get(scope_index)
                .is_some_and(|argument| argument == "--workspace") =>
        {
            args[scope_index] = "--all".to_string();
        }
        CargoAction::Clean
            if args
                .get(scope_index)
                .is_some_and(|argument| argument == "--workspace") =>
        {
            args.remove(scope_index);
        }
        CargoAction::Tree => {
            let delimiter = args
                .iter()
                .position(|argument| argument == "--")
                .unwrap_or(args.len());
            args.splice(
                delimiter..delimiter,
                ["--locked".to_string(), "--offline".to_string()],
            );
        }
        _ => {}
    }
}

fn manifest_directory_template(manifest: &ProjectPath) -> String {
    let root = task::VariableName::WorktreeRoot.template_value();
    manifest
        .path
        .parent()
        .filter(|parent| !parent.is_empty())
        .map(|parent| format!("{root}/{}", parent.as_unix_str()))
        .unwrap_or(root)
}

fn selection_task_context(
    base_context: &TaskContext,
    workspace_manifest: &ProjectPath,
) -> Result<TaskContext> {
    let mut context = base_context.clone();
    let worktree_root = base_context
        .cwd
        .as_ref()
        .ok_or_else(|| anyhow!("the selected worktree has no task working directory"))?;
    let workspace_directory = workspace_manifest
        .path
        .parent()
        .map(|parent| worktree_root.join(parent.as_std_path()))
        .unwrap_or_else(|| worktree_root.clone());
    context.cwd = Some(workspace_directory);
    Ok(context)
}

pub trait CargoActionDispatcher {
    fn validate_plan(&mut self, _plan: &CargoActionPlan) -> Result<()> {
        Ok(())
    }
    fn report_plan_error(&mut self, error: anyhow::Error) {
        log::error!("Cargo planning failed: {error:#}");
    }

    fn schedule_native_profile(
        &mut self,
        selection: CargoActionSelection,
        preset: CargoPreset,
        context: TaskContext,
    );

    fn schedule_task(
        &mut self,
        source: TaskSourceKind,
        template: TaskTemplate,
        context: TaskContext,
        omit_history: bool,
    );

    fn start_debug_session(
        &mut self,
        scenario: DebugScenario,
        context: TaskContext,
        worktree_id: WorktreeId,
    );

    fn run_pre_launch_task(
        &mut self,
        reference: String,
        context: TaskContext,
        worktree_id: WorktreeId,
        continuation: CargoActionPlan,
    );
}

pub fn dispatch_cargo_action(
    mut plan: CargoActionPlan,
    dispatcher: &mut impl CargoActionDispatcher,
) {
    if let Err(error) = dispatcher.validate_plan(&plan) {
        dispatcher.report_plan_error(error);
        return;
    }
    let pre_launch = plan.pre_launch().map(|(reference, context, worktree_id)| {
        (reference.to_string(), context.clone(), worktree_id)
    });
    if let Some((reference, context, worktree_id)) = pre_launch {
        plan.clear_pre_launch();
        dispatcher.run_pre_launch_task(reference, context, worktree_id, plan);
        return;
    }
    match plan {
        CargoActionPlan::NativeProfile {
            selection,
            preset,
            context,
            ..
        } => dispatcher.schedule_native_profile(selection, preset, context),
        CargoActionPlan::Task {
            source,
            template,
            context,
            worktree_id: _,
            pre_launch_task: _,
            omit_history,
            ..
        } => dispatcher.schedule_task(source, template, context, omit_history),
        CargoActionPlan::Debug {
            scenario,
            context,
            worktree_id,
            ..
        } => dispatcher.start_debug_session(scenario, context, worktree_id),
    }
}

fn ensure_experimental_template_access(template: &TaskTemplate, cx: &gpui::App) -> Result<()> {
    for requirement in &template.required_experimental_features {
        anyhow::ensure!(
            cx.has_flag_by_name(requirement),
            "Cargo Task requires the {requirement} opt-in"
        );
    }
    let coverage = template
        .artifact_group
        .as_ref()
        .is_some_and(|group| group.format == crate::CARGO_COVERAGE_ARTIFACT_FORMAT)
        || template.artifact.as_ref().is_some_and(|artifact| {
            artifact.kind == TaskArtifactKind::Data
                && artifact.path == crate::CARGO_COVERAGE_ARTIFACT_PATH
        });
    anyhow::ensure!(
        !coverage || cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
        "Rust coverage requires the rust-coverage-experimental opt-in"
    );
    let native = template
        .stdout_report
        .as_ref()
        .is_some_and(|report| report.native_profile.is_some());
    anyhow::ensure!(
        !native || cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
        "Native profiling requires the rust-native-profiling-experimental opt-in"
    );
    Ok(())
}

pub struct WorkspaceCargoActionDispatcher<'a, 'cx> {
    workspace: &'a mut Workspace,
    window: &'a mut Window,
    cx: &'a mut Context<'cx, Workspace>,
}

impl<'a, 'cx> WorkspaceCargoActionDispatcher<'a, 'cx> {
    pub fn new(
        workspace: &'a mut Workspace,
        window: &'a mut Window,
        cx: &'a mut Context<'cx, Workspace>,
    ) -> Self {
        Self {
            workspace,
            window,
            cx,
        }
    }
}

impl CargoActionDispatcher for WorkspaceCargoActionDispatcher<'_, '_> {
    fn validate_plan(&mut self, plan: &CargoActionPlan) -> Result<()> {
        plan.validate_experimental_access(self.cx)
    }
    fn report_plan_error(&mut self, error: anyhow::Error) {
        self.workspace.show_toast(
            Toast::new(
                NotificationId::unique::<CargoNativeProfileNotification>(),
                format!("Cargo action could not start: {error}"),
            ),
            self.cx,
        );
    }

    fn schedule_native_profile(
        &mut self,
        selection: CargoActionSelection,
        preset: CargoPreset,
        context: TaskContext,
    ) {
        let project = self.workspace.project().clone();
        let workspace = self.cx.entity().downgrade();
        self.cx
            .spawn_in(self.window, async move |_, cx| {
                let result = async {
                    project.read_with(cx, |_, cx| {
                        crate::ensure_experimental_cargo_action(
                            CargoAction::Profile,
                            Some(&preset),
                            cx,
                        )
                    })?;
                    crate::cargo_profile::validate_native_profile_selection(&selection, &preset)?;
                    let store = project.read_with(cx, |project, cx| {
                    anyhow::ensure!(
                        project.is_local() && !project.is_shared() && !project.is_disconnected(cx),
                        "Selected native profiling requires an unshared connected local macOS host"
                    );
                    Ok(project.cargo_workspace_store().clone())
                })?;
                    let recipe = store.update(cx, |store, cx| {
                        let (snapshot, current) = store.cached_snapshot(cx).ok_or_else(|| {
                            anyhow!("Refresh Cargo metadata before native profiling")
                        })?;
                        anyhow::ensure!(
                            current,
                            "Refresh stale Cargo metadata before native profiling"
                        );
                        let (workspace, member, target) =
                            crate::cargo_profile::native_profile_selected_target(
                                &selection, snapshot,
                            )?;
                        store.prepare_native_cargo_runtime_recipe(
                            workspace,
                            member,
                            target,
                            preset.profile.clone(),
                            preset.environment.clone(),
                            preset.trailing_args.clone(),
                            cx,
                        )
                    })?;
                    let recipe = recipe.await?;
                    let mut template =
                        crate::cargo_profile::native_profile_build_template(&preset, recipe)?;
                    template.required_experimental_features = vec![
                        "rust-native-profiling-experimental".into(),
                        "rust-named-cargo-presets-experimental".into(),
                    ];
                    workspace.update_in(cx, |workspace, window, cx| {
                        anyhow::ensure!(
                            workspace.project().entity_id() == project.entity_id(),
                            "The native profiling Project changed while preparing the action"
                        );
                        crate::ensure_experimental_cargo_action(
                            CargoAction::Profile,
                            Some(&preset),
                            cx,
                        )?;
                        workspace.schedule_task(
                            TaskSourceKind::Language {
                                name: "Cargo".into(),
                            },
                            &template,
                            &context,
                            false,
                            window,
                            cx,
                        );
                        anyhow::Ok(())
                    })??;
                    anyhow::Ok(())
                }
                .await;
                if let Err(error) = result {
                    #[cfg(test)]
                    eprintln!("native selected action could not start: {error:#}");
                    if let Err(closed) = workspace.update(cx, |workspace, cx| {
                        workspace.show_toast(
                            Toast::new(
                                NotificationId::unique::<CargoNativeProfileNotification>(),
                                format!("Native profiling could not start: {error}"),
                            ),
                            cx,
                        );
                    }) {
                        log::debug!(
                            "Native profiling workspace closed before reporting failure: {closed:#}"
                        );
                    }
                }
            })
            .detach();
    }

    fn schedule_task(
        &mut self,
        source: TaskSourceKind,
        template: TaskTemplate,
        context: TaskContext,
        omit_history: bool,
    ) {
        if let Err(error) = ensure_experimental_template_access(&template, self.cx) {
            self.report_plan_error(error);
            return;
        }
        if template
            .artifact_group
            .as_ref()
            .is_some_and(|group| group.format == crate::CARGO_COVERAGE_ARTIFACT_FORMAT)
            || template
                .artifact
                .as_ref()
                .is_some_and(|artifact| artifact.kind == TaskArtifactKind::Data)
        {
            self.schedule_coverage_task(source, template, context, omit_history);
            return;
        }
        self.workspace.schedule_task(
            source,
            &template,
            &context,
            omit_history,
            self.window,
            self.cx,
        );
    }

    fn start_debug_session(
        &mut self,
        scenario: DebugScenario,
        context: TaskContext,
        worktree_id: WorktreeId,
    ) {
        self.workspace.start_debug_session(
            scenario,
            SharedTaskContext::from(context),
            None,
            Some(worktree_id),
            self.window,
            self.cx,
        );
    }

    fn run_pre_launch_task(
        &mut self,
        reference: String,
        context: TaskContext,
        worktree_id: WorktreeId,
        continuation: CargoActionPlan,
    ) {
        let pending = match continuation
            .experimental_requirements()
            .and_then(|requirements| {
                project::task_store::PendingExperimentalFeatureOperation::new(
                    &requirements,
                    self.cx,
                )
            }) {
            Ok(pending) => pending,
            Err(error) => {
                self.report_plan_error(error);
                return;
            }
        };
        let workspace = self.cx.entity().downgrade();
        self.workspace.schedule_task_reference_with_completion(
            reference,
            worktree_id,
            context,
            false,
            move |result, cx| {
                if result != ScheduledTaskResult::Success {
                    return;
                }
                if let Err(error) = workspace.update_in(cx, |workspace, window, cx| {
                    let mut dispatcher = WorkspaceCargoActionDispatcher::new(workspace, window, cx);
                    if let Err(error) =
                        continuation
                            .experimental_requirements()
                            .and_then(|requirements| {
                                pending.check_requirements(&requirements, dispatcher.cx)
                            })
                    {
                        dispatcher.report_plan_error(error);
                        return;
                    }
                    drop(pending);
                    dispatch_cargo_action(continuation, &mut dispatcher);
                }) {
                    log::debug!("Cargo action was dropped after its pre-launch task: {error:#}");
                }
            },
            self.window,
            self.cx,
        );
    }
}

struct CargoNativeProfileNotification;

struct CargoCoverageNotification;

fn cancel_coverage_run(
    provider: &gpui::Entity<project::rust_coverage_provider::RustCoverageProviderStore>,
    coverage: &gpui::Entity<project::source_coverage::SourceCoverageStore>,
    token: project::rust_coverage_provider::RustCoverageRunToken,
    status: SourceCoverageStatus,
    cx: &mut gpui::App,
) {
    if !provider.read(cx).owns_run(token) {
        return;
    }
    if let Err(error) = coverage.update(cx, |store, cx| {
        store.mark_provider_status(
            SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.to_string()),
            status,
            Some("Coverage Task expired because the project session or trust changed".into()),
            cx,
        )
    }) {
        log::error!("Coverage expired state could not be published: {error:#}");
    }
    provider.update(cx, |provider, _| provider.cancel_run(token));
}

fn finish_expired_coverage_run(
    workspace: &gpui::WeakEntity<Workspace>,
    provider: &gpui::Entity<project::rust_coverage_provider::RustCoverageProviderStore>,
    coverage: &gpui::Entity<project::source_coverage::SourceCoverageStore>,
    token: project::rust_coverage_provider::RustCoverageRunToken,
    cx: &mut gpui::AsyncApp,
) {
    let owned = cx.update(|cx| {
        if !provider.read(cx).owns_run(token) {
            return false;
        }
        let status = coverage
            .read(cx)
            .selected_snapshot()
            .map(|snapshot| snapshot.status)
            .filter(|status| {
                matches!(
                    status,
                    SourceCoverageStatus::Restricted | SourceCoverageStatus::Disconnected
                )
            })
            .unwrap_or(SourceCoverageStatus::Stale);
        cancel_coverage_run(provider, coverage, token, status, cx);
        true
    });
    if owned {
        if let Err(error) = workspace.update(cx, |workspace, cx| {
            workspace.show_toast(
                Toast::new(
                    NotificationId::unique::<CargoCoverageNotification>(),
                    "Coverage Task expired; refresh the project and run coverage again",
                ),
                cx,
            );
        }) {
            log::debug!("Coverage expiry notification owner expired: {error:#}");
        }
    }
}

impl WorkspaceCargoActionDispatcher<'_, '_> {
    fn schedule_coverage_task(
        &mut self,
        source: TaskSourceKind,
        mut template: TaskTemplate,
        context: TaskContext,
        omit_history: bool,
    ) {
        if !self.cx.has_flag::<RustCoverageExperimentalFeatureFlag>() {
            self.report_plan_error(anyhow!(
                "Rust coverage requires the rust-coverage-experimental opt-in"
            ));
            return;
        }
        let grouped_coverage = template
            .artifact_group
            .as_ref()
            .is_some_and(|group| group.format == crate::CARGO_COVERAGE_ARTIFACT_FORMAT);
        let local_capture = self.workspace.project().read(self.cx).is_local() && grouped_coverage;
        if grouped_coverage && !local_capture {
            let Some(mut group) = template.artifact_group.take() else {
                return;
            };
            let Some(binding) = template.artifact_group_argument_binding.take() else {
                self.workspace.show_toast(
                    Toast::new(
                        NotificationId::unique::<CargoCoverageNotification>(),
                        "Coverage Task has no output binding",
                    ),
                    self.cx,
                );
                return;
            };
            if group.files.len() != 1 || binding.role != "report" || !binding.related.is_empty() {
                self.workspace.show_toast(
                    Toast::new(
                        NotificationId::unique::<CargoCoverageNotification>(),
                        "Coverage Task report declaration is invalid",
                    ),
                    self.cx,
                );
                return;
            }
            template.artifact = group.files.remove("report");
            template.artifact_argument_binding = Some(task::TaskArtifactArgumentBinding {
                argument_index: binding.argument_index,
            });
        }
        let Some(resolved) = template.resolve_task(&source.to_id_base(), &context) else {
            self.workspace.show_toast(
                Toast::new(
                    NotificationId::unique::<CargoCoverageNotification>(),
                    "Run with Coverage could not resolve its task context",
                ),
                self.cx,
            );
            return;
        };
        if local_capture {
            self.workspace.schedule_resolved_task(
                source,
                resolved,
                omit_history,
                self.window,
                self.cx,
            );
            return;
        }
        if resolved.resolved_artifact().is_none() {
            self.workspace.show_toast(
                Toast::new(
                    NotificationId::unique::<CargoCoverageNotification>(),
                    "Run with Coverage has no declared report artifact",
                ),
                self.cx,
            );
            return;
        }
        let project = self.workspace.project().clone();
        let source_coverage_store = project.read(self.cx).source_coverage_store().clone();
        let provider = project.read(self.cx).rust_coverage_provider_store().clone();
        let run_token = match provider.update(self.cx, |provider, cx| provider.begin_run(cx)) {
            Ok(token) => token,
            Err(error) => {
                self.workspace.show_toast(
                    Toast::new(
                        NotificationId::unique::<CargoCoverageNotification>(),
                        format!("Run with Coverage could not be admitted: {error}"),
                    ),
                    self.cx,
                );
                return;
            }
        };
        let mut subscriptions = vec![self.cx.subscribe(&project, {
            let provider = provider.clone();
            let coverage = source_coverage_store.clone();
            move |_, _, event, cx| {
                if matches!(
                    event,
                    project::Event::Closed
                        | project::Event::DisconnectedFromHost
                        | project::Event::DisconnectedFromRemote { .. }
                        | project::Event::WorktreeRemoved(_)
                        | project::Event::WorktreePathsChanged { .. }
                        | project::Event::RemoteIdChanged(_)
                        | project::Event::Reshared
                        | project::Event::HostReshared
                        | project::Event::UpstreamSessionChanged
                        | project::Event::Rejoined
                ) {
                    let status = if matches!(
                        event,
                        project::Event::Closed
                            | project::Event::DisconnectedFromHost
                            | project::Event::DisconnectedFromRemote { .. }
                    ) {
                        SourceCoverageStatus::Disconnected
                    } else {
                        SourceCoverageStatus::Stale
                    };
                    cancel_coverage_run(&provider, &coverage, run_token, status, cx);
                }
            }
        })];
        if let Some(trust) = project::trusted_worktrees::TrustedWorktrees::try_get_global(self.cx) {
            let worktree_store = project.read(self.cx).worktree_store().downgrade();
            subscriptions.push(
                self.cx.subscribe(&trust, {
                    let provider = provider.clone();
                    let coverage = source_coverage_store.clone();
                    move |_, _, event, cx| {
                        let changed_worktree_store = match event {
                            project::trusted_worktrees::TrustedWorktreesEvent::Trusted(
                                store,
                                _,
                            )
                            | project::trusted_worktrees::TrustedWorktreesEvent::Restricted(
                                store,
                                _,
                            ) => store,
                        };
                        if *changed_worktree_store == worktree_store {
                            let status = if matches!(
                                event,
                                project::trusted_worktrees::TrustedWorktreesEvent::Restricted(..)
                            ) {
                                SourceCoverageStatus::Restricted
                            } else {
                                SourceCoverageStatus::Stale
                            };
                            cancel_coverage_run(&provider, &coverage, run_token, status, cx);
                        }
                    }
                }),
            );
        }
        if let Err(error) = source_coverage_store.update(self.cx, |store, cx| {
            store.mark_provider_status(
                SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.to_string()),
                SourceCoverageStatus::Loading,
                None,
                cx,
            )
        }) {
            log::error!("Coverage loading state could not be published: {error:#}");
        }
        let workspace = self.cx.entity().downgrade();
        let run_token = std::rc::Rc::new(std::cell::Cell::new(run_token));
        let admitted_invocation =
            std::rc::Rc::new(std::cell::RefCell::new(None::<TaskInvocationContext>));
        self.workspace.schedule_resolved_task_with_invocation_admission_preflight_and_completion(
            source,
            resolved,
            omit_history,
            {
                let provider = provider.clone();
                let run_token = run_token.clone();
                move |_, cx| provider.read(cx).is_run_current(run_token.get(), cx)
            },
            {
                let provider = provider.clone();
                let coverage = source_coverage_store.clone();
                let run_token = run_token.clone();
                let admitted_invocation = admitted_invocation.clone();
                move |invocation, workspace, cx| {
                    let sealed = provider.read_with(cx, |provider, cx| provider.seal_source_revision(run_token.get(), cx));
                    match sealed {
                        Ok(sealed) => {
                            run_token.set(sealed);
                            *admitted_invocation.borrow_mut() = Some(invocation.clone());
                            Ok(())
                        }
                        Err(error) => {
                            let owned = provider.read(cx).owns_run(run_token.get());
                            let status = coverage.read(cx).selected_snapshot()
                                .map(|snapshot| snapshot.status)
                                .filter(|status| matches!(status, SourceCoverageStatus::Disconnected | SourceCoverageStatus::Restricted))
                                .unwrap_or(SourceCoverageStatus::Stale);
                            cancel_coverage_run(&provider, &coverage, run_token.get(), status, cx);
                            if owned {
                                workspace.show_toast(Toast::new(
                                    NotificationId::unique::<CargoCoverageNotification>(),
                                    "Coverage Task expired before launch; refresh the project and run coverage again",
                                ), cx);
                            }
                            Err(error)
                        }
                    }
                }
            },
            move |result, cx| {
                let run_token = run_token.get();
                if !provider.read_with(cx, |provider, cx| provider.is_run_current(run_token, cx)) {
                    finish_expired_coverage_run(&workspace, &provider, &source_coverage_store, run_token, cx);
                    return;
                }
                match result {
                ScheduledTaskResult::Success => {
                    let interpretation: Result<_> = (|| {
                        let invocation = admitted_invocation.borrow_mut().take()
                            .ok_or_else(|| anyhow!("Coverage Task has no admitted invocation"))?;
                        let artifact = invocation.artifact()
                            .ok_or_else(|| anyhow!("Coverage Task has no admitted report artifact"))?;
                        let task_working_directory = invocation.working_directory();
                        let artifact_path = project.read_with(cx, |project, cx| {
                            if project.is_disconnected(cx)
                                || project.remote_client().is_some_and(|client| {
                                    client.read(cx).connection().is_none()
                                })
                                || project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                                    &project.worktree_store(), cx,
                                )
                            {
                                bail!("Coverage requires a connected, trusted project host");
                            }
                            let working_directory = task_working_directory
                                .ok_or_else(|| anyhow!("Coverage Task has no project working directory"))?;
                            project
                                .find_project_path(working_directory.join(&artifact.path), cx)
                                .ok_or_else(|| anyhow!("Coverage report is outside the visible project"))
                        })?;
                        let max_bytes = usize::try_from(artifact.max_bytes)
                            .unwrap_or(usize::MAX)
                            .min(MAX_RUST_COVERAGE_ARTIFACT_BYTES);
                        Ok(provider.update(cx, |provider, cx| {
                            provider.interpret_artifact_for_run(run_token, artifact_path, max_bytes, cx)
                        }))
                    })();
                    match interpretation {
                        Ok(interpretation) => {
                            cx.spawn(async move |cx| {
                                let _subscriptions = subscriptions;
                                let result = interpretation.await;
                                if !provider.read_with(cx, |provider, cx| provider.is_run_current(run_token, cx)) {
                                    finish_expired_coverage_run(&workspace, &provider, &source_coverage_store, run_token, cx);
                                    return;
                                }
                                let message = match result {
                                    Ok(snapshot) => format!(
                                        "Coverage loaded for {} file(s)",
                                        snapshot.files.len()
                                    ),
                                    Err(error) => {
                                        let status = if error.is::<SourceCoverageSourceChanged>() {
                                            SourceCoverageStatus::Stale
                                        } else {
                                            SourceCoverageStatus::Error
                                        };
                                        if let Err(status_error) = source_coverage_store.update(
                                            cx,
                                            |store, cx| {
                                                store.mark_provider_status(
                                                    SourceCoverageProviderId(
                                                        RUST_COVERAGE_PROVIDER_ID.to_string(),
                                                    ),
                                                    status,
                                                    Some(error.to_string()),
                                                    cx,
                                                )
                                            },
                                        ) {
                                            log::error!(
                                                "Coverage error state could not be published: {status_error:#}"
                                            );
                                        }
                                        if status == SourceCoverageStatus::Stale {
                                            format!("Coverage is stale: {error}. Run coverage again")
                                        } else {
                                            format!("Coverage report could not be interpreted: {error}")
                                        }
                                    }
                                };
                                workspace
                                    .update(cx, |workspace, cx| {
                                        workspace.show_toast(
                                            Toast::new(
                                                NotificationId::unique::<
                                                    CargoCoverageNotification,
                                                >(),
                                                message,
                                            ),
                                            cx,
                                        )
                                    })
                                    .ok();
                            })
                            .detach();
                        }
                        Err(error) => {
                            log::error!("Coverage artifact could not be scheduled: {error:#}");
                            if let Err(status_error) = source_coverage_store.update(
                                cx,
                                |store, cx| {
                                    store.mark_provider_status(
                                        SourceCoverageProviderId(
                                            RUST_COVERAGE_PROVIDER_ID.to_string(),
                                        ),
                                        SourceCoverageStatus::Error,
                                        Some("Coverage artifact validation failed".to_string()),
                                        cx,
                                    )
                                },
                            ) {
                                log::error!(
                                    "Coverage error state could not be published: {status_error:#}"
                                );
                            }
                            workspace
                                .update(cx, |workspace, cx| {
                                    workspace.show_toast(
                                        Toast::new(
                                            NotificationId::unique::<CargoCoverageNotification>(),
                                            "Coverage report could not be validated on the project host",
                                        ),
                                        cx,
                                    )
                                })
                                .ok();
                        }
                    }
                }
                ScheduledTaskResult::Failure | ScheduledTaskResult::SpawnFailed => {
                    if let Err(error) = source_coverage_store.update(cx, |store, cx| {
                        store.mark_provider_status(
                            SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.to_string()),
                            SourceCoverageStatus::Error,
                            Some(CARGO_COVERAGE_FAILURE_GUIDANCE.to_string()),
                            cx,
                        )
                    }) {
                        log::error!("Coverage failure state could not be published: {error:#}");
                    }
                    workspace
                        .update(cx, |workspace, cx| {
                            workspace.show_toast(
                                Toast::new(
                                    NotificationId::unique::<CargoCoverageNotification>(),
                                    CARGO_COVERAGE_FAILURE_GUIDANCE,
                                ),
                                cx,
                            )
                        })
                        .ok();
                }
                ScheduledTaskResult::Cancelled => {
                    provider.update(cx, |provider, cx| provider.finish_cancelled_run(run_token, cx));
                }
                }
            },
            self.window,
            self.cx,
        );
    }
}

#[cfg(test)]
mod tests {
    fn spawn_prepared_recorded_terminal<T: workspace::TerminalProvider + 'static>(
        terminal: T,
        launch: task::SpawnInTerminal,
        handle: Option<task::StructuredTaskHandle>,
        prepare: workspace::TaskLaunchPreparation,
        window: &mut Window,
        cx: &mut gpui::App,
    ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
        window.to_async(cx).spawn(async move |cx| {
            let original = launch.clone();
            let prepared = match prepare(launch, cx).await {
                Ok(prepared) => prepared,
                Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                Err(error) => return Some(Err(error)),
            };
            if let Err(error) = prepared.validate_task_identity(&original) {
                return Some(Err(error));
            }
            let result = match cx.update(|window, cx| match handle {
                Some(handle) => terminal.spawn_structured(prepared.launch, handle, window, cx),
                None => terminal.spawn(prepared.launch, window, cx),
            }) {
                Ok(result) => result,
                Err(error) => return Some(Err(error)),
            };
            match prepared.completion {
                Some(complete) => complete(result, cx).await,
                None => result.await,
            }
        })
    }

    fn compile_legacy_coverage_plan(
        compiled: crate::cargo_preset::CompiledCargoPreset,
    ) -> Result<crate::cargo_preset::CompiledCargoPreset> {
        let mut compiled = compile_coverage_plan(compiled)?;
        // These cases retain the legacy single-report dispatcher contract; grouped
        // capture is exercised separately through the common Task owner and real PTYs.
        let mut group = compiled
            .task_template
            .artifact_group
            .take()
            .expect("coverage group");
        let binding = compiled
            .task_template
            .artifact_group_argument_binding
            .take()
            .expect("coverage binding");
        compiled.task_template.artifact = group.files.remove("report");
        compiled.task_template.artifact_argument_binding =
            Some(task::TaskArtifactArgumentBinding {
                argument_index: binding.argument_index,
            });
        Ok(compiled)
    }

    use std::path::PathBuf;

    use collections::HashMap;
    use project::WorktreeId;
    use task::{BuildTaskDefinition, RevealStrategy, RevealTarget, SaveStrategy};
    use util::rel_path::RelPath;

    use super::*;

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_native_profile_helper_uses_production_cli_before_test_harness(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::AppContext as _;
        use std::{process::Stdio, time::Duration};
        use util::macos_task_completion::{TASK_COMPLETION_TOKEN_ENV, TaskCompletionListener};

        cx.executor().allow_parking();
        for exit_code in [0, 7] {
            let helper = std::env::current_exe().expect("source-matched test helper");
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("private listener");
            let mut command = std::process::Command::new(&helper);
            let arguments = [
                "space value",
                "$literal",
                "quote'\"",
                "; no shell",
                "Unicode 雪",
            ];
            command
                .args([
                    std::ffi::OsStr::new("--internal-task-observer"),
                    std::ffi::OsStr::new("--socket"),
                    listener.path().as_os_str(),
                    std::ffi::OsStr::new("--"),
                    std::ffi::OsStr::new("/bin/sh"),
                ])
                .args([
                    "-c",
                    r#"[ -z "${ZED_PROFILE_TASK_TOKEN+x}" ] || exit 49; code=$1; shift; printf '%s\0' "$@"; exit "$code""#,
                    "production-cli-helper",
                    &exit_code.to_string(),
                ])
                .args(arguments)
                .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32));
            let child =
                util::process::Child::spawn(command, Stdio::null(), Stdio::piped(), Stdio::piped())
                    .expect("actual helper process");
            let observation = cx.background_spawn(async move { listener.observe().await });
            let completion = futures::future::select(
                Box::pin(async { futures::join!(observation, child.output()) }),
                Box::pin(cx.background_executor.timer(Duration::from_secs(10))),
            )
            .await;
            let (observed, output) = match completion {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => panic!("actual helper CLI did not complete"),
            };
            assert_eq!(
                observed.expect("kernel child status").code(),
                Some(exit_code)
            );
            let output = output.expect("actual helper output");
            assert_eq!(output.status.code(), Some(exit_code));
            assert!(output.stderr.is_empty());
            let expected: Vec<u8> = arguments
                .into_iter()
                .flat_map(|argument| argument.bytes().chain(std::iter::once(0)))
                .collect();
            assert_eq!(
                output.stdout, expected,
                "literal child argv without libtest output"
            );
        }
    }

    #[gpui::test]
    async fn cargo_coverage_action_overlapping_runs_bind_distinct_artifacts(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_invocation_capture(CoverageInvocationReport::Valid, false, cx).await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_invocation_does_not_recover_missing_output_from_shared_path(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_invocation_capture(CoverageInvocationReport::Missing, false, cx).await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_invocation_rejects_private_report_symlink(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_invocation_capture(CoverageInvocationReport::PrivateSymlink, false, cx)
            .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_invocation_rejects_report_above_declared_limit(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_invocation_capture(CoverageInvocationReport::Oversized, false, cx).await;
    }

    #[gpui::test]
    async fn cargo_coverage_registered_rerun_captures_fresh_local_generations(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_invocation_capture(CoverageInvocationReport::Valid, true, cx).await;
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum CoverageInvocationReport {
        Valid,
        Missing,
        PrivateSymlink,
        Oversized,
    }

    async fn check_coverage_invocation_capture(
        newest_report: CoverageInvocationReport,
        exercise_rerun: bool,
        cx: &mut gpui::TestAppContext,
    ) {
        use futures::channel::oneshot;
        use gpui::{App, AppContext as _};
        use project::Fs as _;
        use settings::SettingsStore;
        use std::{cell::RefCell, process::ExitStatus, rc::Rc, sync::Arc};

        fn report(covered_lines: usize) -> String {
            let mut report: serde_json::Value = serde_json::from_str(include_str!(
                "../../project/test_data/source_coverage/llvm3-run.json"
            ))
            .expect("actual LLVM coverage fixture");
            *report
                .get_mut("data")
                .and_then(|data| data.get_mut(0))
                .and_then(|data| data.get_mut("files"))
                .and_then(|files| files.get_mut(0))
                .and_then(|file| file.get_mut("summary"))
                .and_then(|summary| summary.get_mut("lines"))
                .and_then(|lines| lines.get_mut("covered"))
                .expect("actual fixture line summary") = covered_lines.into();
            serde_json::to_string(&report)
                .expect("coverage fixture serialization")
                .replace("/workspace/", "/fixture/")
        }

        struct HeldCollector {
            fs: Arc<project::FakeFs>,
            launches: Rc<RefCell<Vec<task::SpawnInTerminal>>>,
            releases: Rc<RefCell<Vec<Option<oneshot::Sender<()>>>>>,
            newest_report: CoverageInvocationReport,
        }
        impl workspace::TerminalProvider for HeldCollector {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<ExitStatus>>> {
                let terminal = Self {
                    fs: self.fs.clone(),
                    launches: self.launches.clone(),
                    releases: self.releases.clone(),
                    newest_report: self.newest_report,
                };
                window.to_async(cx).spawn(async move |cx| {
                    let original = launch.clone();
                    let prepared = match prepare(launch, cx).await {
                        Ok(prepared) => prepared,
                        Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    if let Err(error) = prepared.validate_task_identity(&original) {
                        return Some(Err(error));
                    }
                    let result = match cx.update(|window, cx| match handle {
                        Some(handle) => {
                            terminal.spawn_structured(prepared.launch, handle, window, cx)
                        }
                        None => terminal.spawn(prepared.launch, window, cx),
                    }) {
                        Ok(result) => result,
                        Err(error) => return Some(Err(error)),
                    };
                    match prepared.completion {
                        Some(complete) => complete(result, cx).await,
                        None => result.await,
                    }
                })
            }

            fn spawn(
                &self,
                launch: task::SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<ExitStatus>>> {
                let output_argument = launch
                    .args
                    .iter()
                    .position(|argument| argument == "--output-path")
                    .and_then(|index| launch.args.get(index + 1))
                    .expect("actual collector output argument");
                let output = launch
                    .cwd
                    .as_ref()
                    .expect("actual collector cwd")
                    .join(output_argument);
                let index = self.launches.borrow().len();
                self.launches.borrow_mut().push(launch);
                let (release, released) = oneshot::channel();
                self.releases.borrow_mut().push(Some(release));
                let fs = self.fs.clone();
                let newest_report = self.newest_report;
                cx.background_spawn(async move {
                    released.await.expect("release actual scheduled collector");
                    if index == 0 {
                        fs.insert_file(&output, report(2).into_bytes()).await;
                    } else {
                        match newest_report {
                            CoverageInvocationReport::Valid => {
                                fs.insert_file(&output, report(3).into_bytes()).await;
                            }
                            CoverageInvocationReport::Missing => {}
                            CoverageInvocationReport::PrivateSymlink => {
                                fs.create_symlink(&output, "/fixture/.env".into())
                                    .await
                                    .expect("actual private artifact alias");
                            }
                            CoverageInvocationReport::Oversized => {
                                let mut contents = report(3).into_bytes();
                                contents.push(b' ');
                                fs.insert_file(&output, contents).await;
                            }
                        }
                    }
                    Some(Ok(ExitStatus::default()))
                })
            }
        }

        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
            tasks_ui::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                ".git": {},
                ".gitignore": "target\n",
                ".env": report(3),
                "Cargo.toml": "[package]\nname = \"fixture\"\nversion = \"0.1.0\"",
                "src": {"bin": {"fixture.rs": "fn main() {}"}},
                "target": {"zed-coverage.json": report(1)}
            }),
        )
        .await;
        let project = project::Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let launches = Rc::new(RefCell::new(Vec::new()));
        let releases = Rc::new(RefCell::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(HeldCollector {
                fs: fs.clone(),
                launches: launches.clone(),
                releases: releases.clone(),
                newest_report,
            });
        });
        let mut compiled = compile_coverage_plan(
            compile_preset(
                &CargoPreset::ephemeral_default(CargoSubcommand::Run),
                &CargoCompileContext {
                    workspace_cwd: Some("/fixture".into()),
                    ..CargoCompileContext::default()
                },
                None,
            )
            .expect("actual Cargo Run preset"),
        )
        .expect("actual coverage collector preset");
        compiled.task_template.save = SaveStrategy::None;
        compiled.task_template.allow_concurrent_runs = true;
        compiled
            .task_template
            .artifact_group
            .as_mut()
            .expect("coverage declaration")
            .files
            .get_mut("report")
            .expect("coverage report role")
            .max_bytes = report(3).len() as u64;
        if exercise_rerun {
            let persisted =
                serde_json::to_string(&task::TaskTemplates(vec![compiled.task_template]))
                    .expect("persist actual coverage template");
            let task::TaskTemplates(mut loaded) =
                serde_json::from_str(&persisted).expect("load actual persisted coverage template");
            compiled.task_template = loaded.pop().expect("one persisted coverage template");
        }
        let original_template = compiled.task_template.clone();
        let contexts = workspace
            .update_in(cx, |workspace, window, cx| {
                tasks_ui::task_contexts(workspace, window, cx)
            })
            .await;
        let initial_context = contexts
            .active_context()
            .expect("same production worktree context used by reevaluated Rerun")
            .clone();
        assert_eq!(
            initial_context.cwd.as_deref(),
            Some(std::path::Path::new("/fixture"))
        );
        fs.pause_events();
        for index in 0..2 {
            workspace.update_in(cx, |workspace, window, cx| {
                WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                    TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    compiled.task_template.clone(),
                    initial_context.clone(),
                    false,
                );
            });
            cx.run_until_parked();
            assert_eq!(launches.borrow().len(), index + 1);
            assert_eq!(releases.borrow().len(), index + 1);
        }
        let (old_launch, current_launch) = {
            let launches = launches.borrow();
            (
                launches.first().expect("held older Task").clone(),
                launches.get(1).expect("held latest Task").clone(),
            )
        };
        assert_eq!(old_launch.id, current_launch.id);
        assert_eq!(old_launch.command, current_launch.command);
        let output_argument = |launch: &task::SpawnInTerminal| {
            launch
                .args
                .iter()
                .position(|argument| argument == "--output-path")
                .and_then(|index| launch.args.get(index + 1))
                .expect("collector output")
                .clone()
        };
        let old_output = output_argument(&old_launch);
        let current_output = output_argument(&current_launch);
        assert_ne!(old_output, current_output);
        for output in [&old_output, &current_output] {
            assert!(output.starts_with("target/zed-coverage-"));
            assert!(output.ends_with(".json"));
            assert_ne!(output, crate::CARGO_COVERAGE_ARTIFACT_PATH);
        }
        let history = workspace.read_with(cx, |workspace, cx| {
            workspace
                .project()
                .read(cx)
                .task_store()
                .read(cx)
                .task_inventory()
                .cloned()
                .expect("production Task history")
        });
        let (_, cached) = history.read_with(cx, |history, _| {
            history
                .last_scheduled_task(None)
                .expect("latest bound Task history")
        });
        assert_eq!(cached.id, current_launch.id);
        assert_eq!(
            cached
                .resolved_artifact_group()
                .expect("bound coverage group")
                .files
                .get("report")
                .expect("bound artifact")
                .path,
            current_output
        );
        assert_eq!(cached.original_task(), &original_template);
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        store.read_with(cx, |store, _| {
            assert_eq!(
                store.selected_snapshot().expect("held newest run").status,
                SourceCoverageStatus::Loading
            );
        });
        releases
            .borrow_mut()
            .get_mut(1)
            .expect("newest release")
            .take()
            .expect("single newest release")
            .send(())
            .expect("newest Task remains held");
        cx.run_until_parked();
        let completed = store.read_with(cx, |store, _| {
            store
                .selected_snapshot()
                .expect("newest completion")
                .clone()
        });
        if newest_report == CoverageInvocationReport::Valid {
            assert_eq!(completed.status, SourceCoverageStatus::Current);
            assert_eq!(completed.files.len(), 1);
            assert_eq!(
                completed
                    .files
                    .first()
                    .expect("covered source")
                    .covered_lines,
                3
            );
            project.read_with(cx, |project, cx| {
                let output_path = project
                    .find_project_path(
                        current_launch
                            .cwd
                            .as_ref()
                            .expect("latest cwd")
                            .join(&current_output),
                        cx,
                    )
                    .expect("bound output project path");
                assert!(
                    project
                        .entry_for_path(&output_path, cx)
                        .expect("explicitly refreshed ignored output")
                        .is_ignored
                );
            });
        } else {
            assert_eq!(
                completed.status,
                SourceCoverageStatus::Error,
                "{newest_report:?}: {completed:?}"
            );
            assert!(completed.files.is_empty());
        }
        assert_eq!(
            fs.load("/fixture/target/zed-coverage.json".as_ref())
                .await
                .expect("original shared path retained"),
            report(1)
        );
        releases
            .borrow_mut()
            .first_mut()
            .expect("older release")
            .take()
            .expect("single older release")
            .send(())
            .expect("older terminal was not globally serialized");
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert_eq!(store.selected_snapshot(), Some(&completed))
        });
        let old_path = old_launch
            .cwd
            .as_ref()
            .expect("older cwd")
            .join(&old_output);
        assert_eq!(
            fs.load(&old_path)
                .await
                .expect("obsolete collector wrote its own output"),
            report(2)
        );
        if newest_report == CoverageInvocationReport::Valid {
            let current_path = current_launch
                .cwd
                .as_ref()
                .expect("latest cwd")
                .join(&current_output);
            assert_eq!(
                fs.load(&current_path)
                    .await
                    .expect("latest output survives old capture"),
                report(3)
            );
        }
        if exercise_rerun {
            use gpui::Focusable as _;
            let mut output_paths = collections::HashSet::from_iter([old_output, current_output]);
            let mut previous = completed;
            for (offset, reevaluate_context) in [false, true, false, true].into_iter().enumerate() {
                workspace.update_in(cx, |workspace, window, cx| {
                    window.focus(&workspace.focus_handle(cx), cx);
                });
                cx.dispatch_action(tasks_ui::Rerun {
                    reevaluate_context,
                    ..Default::default()
                });
                cx.run_until_parked();
                let index = offset + 2;
                assert_eq!(launches.borrow().len(), index + 1);
                let launch = launches
                    .borrow()
                    .get(index)
                    .expect("actual registered Rerun")
                    .clone();
                assert_eq!(launch.id, current_launch.id);
                assert_eq!(launch.cwd, current_launch.cwd);
                let output = output_argument(&launch);
                assert!(
                    output_paths.insert(output.clone()),
                    "each Rerun gets one fresh output"
                );
                assert_eq!(
                    output.matches('-').count(),
                    2,
                    "Rerun does not append another UUID"
                );
                let (_, cached) = history.read_with(cx, |history, _| {
                    history
                        .last_scheduled_task(None)
                        .expect("latest actual Rerun history")
                });
                assert_eq!(cached.id, launch.id);
                assert_eq!(cached.original_task(), &original_template);
                assert_eq!(
                    cached
                        .resolved_artifact_group()
                        .expect("latest capture intent")
                        .files
                        .get("report")
                        .expect("report role")
                        .path,
                    output
                );
                let loading = store.read_with(cx, |store, _| {
                    store
                        .selected_snapshot()
                        .expect("Rerun immediately enters Loading")
                        .clone()
                });
                assert_eq!(loading.status, SourceCoverageStatus::Loading);
                assert!(loading.generation > previous.generation);
                releases
                    .borrow_mut()
                    .get_mut(index)
                    .expect("held Rerun release")
                    .take()
                    .expect("one Rerun release")
                    .send(())
                    .expect("common scheduler retained actual future");
                cx.run_until_parked();
                let published = store.read_with(cx, |store, _| {
                    store
                        .selected_snapshot()
                        .expect("Rerun publishes through common capture owner")
                        .clone()
                });
                assert_eq!(published.status, SourceCoverageStatus::Current);
                assert_eq!(published.generation, loading.generation);
                assert_eq!(published.files.len(), 1);
                assert_eq!(
                    published
                        .files
                        .first()
                        .expect("Rerun source summary")
                        .covered_lines,
                    3
                );
                previous = published;
            }
        }
        fs.unpause_events_and_flush();
        project.update(cx, |project, cx| project.close(cx));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn cargo_coverage_action_rejects_obsolete_task_success_failure_and_cancellation(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::{cell::RefCell, os::unix::process::ExitStatusExt, process::ExitStatus, rc::Rc};

        use futures::channel::oneshot;
        use gpui::{App, AppContext as _};
        use settings::SettingsStore;

        type Completion = oneshot::Sender<Option<Result<ExitStatus>>>;

        #[derive(Clone)]
        struct Terminal {
            completions: Rc<RefCell<Vec<Option<Completion>>>>,
            fs: std::sync::Arc<project::FakeFs>,
        }

        impl workspace::TerminalProvider for Terminal {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                spawn_prepared_recorded_terminal(self.clone(), launch, handle, prepare, window, cx)
            }

            fn spawn(
                &self,
                task: task::SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<ExitStatus>>> {
                let (completion, result) = oneshot::channel();
                self.completions.borrow_mut().push(Some(completion));
                let fs = self.fs.clone();
                let artifact = task.cwd.as_ref().expect("coverage launch directory").join(
                    task.args
                        .get(
                            task.args
                                .iter()
                                .position(|argument| argument == "--output-path")
                                .expect("coverage output flag")
                                + 1,
                        )
                        .expect("coverage output argument"),
                );
                cx.background_spawn(async move {
                    let result = result.await.expect("controlled Task completion");
                    if result
                        .as_ref()
                        .is_some_and(|result| result.as_ref().is_ok_and(|status| status.success()))
                    {
                        let report =
                            include_str!("../../project/test_data/source_coverage/llvm3-run.json")
                                .replace("/workspace/", "/fixture/");
                        fs.insert_file(&artifact, report.into_bytes()).await;
                    }
                    result
                })
            }
        }

        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                "Cargo.toml": "[package]\nname = \"fixture\"\nversion = \"0.1.0\"",
                "src": {"bin": {"fixture.rs": "fn main() {}"}},
                "target": {}
            }),
        )
        .await;
        let project = project::Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let completions = Rc::new(RefCell::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(Terminal {
                completions: completions.clone(),
                fs,
            });
        });
        let mut compiled = compile_coverage_plan(
            compile_preset(
                &CargoPreset::ephemeral_default(CargoSubcommand::Run),
                &CargoCompileContext {
                    workspace_cwd: Some("/fixture".to_string()),
                    ..CargoCompileContext::default()
                },
                None,
            )
            .expect("run plan"),
        )
        .expect("coverage plan");
        compiled.task_template.save = SaveStrategy::None;
        compiled.task_template.allow_concurrent_runs = true;
        for index in 0..5 {
            let mut template = compiled.task_template.clone();
            template.label = format!("Coverage run {index}");
            workspace.update_in(cx, |workspace, window, cx| {
                WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                    TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    template,
                    TaskContext {
                        cwd: Some(PathBuf::from("/fixture")),
                        ..TaskContext::default()
                    },
                    false,
                );
            });
            cx.run_until_parked();
        }
        assert_eq!(completions.borrow().len(), 5);
        completions
            .borrow_mut()
            .get_mut(4)
            .expect("newest run")
            .take()
            .expect("newest run completion")
            .send(Some(Ok(ExitStatus::default())))
            .expect("newest Task is pending");
        cx.run_until_parked();
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let current = store.read_with(cx, |store, _| {
            store.selected_snapshot().expect("newest coverage").clone()
        });
        assert_eq!(current.status, SourceCoverageStatus::Current);
        assert_eq!(current.files.len(), 1);
        assert_eq!(
            current.files.first().expect("covered source").covered_lines,
            3
        );
        for (index, result) in [
            Some(Ok(ExitStatus::default())),
            Some(Ok(ExitStatus::from_raw(1 << 8))),
            None,
            Some(Err(anyhow!("controlled spawn failure"))),
        ]
        .into_iter()
        .enumerate()
        {
            completions
                .borrow_mut()
                .get_mut(index)
                .expect("obsolete run")
                .take()
                .expect("obsolete run completion")
                .send(result)
                .expect("obsolete Task is pending");
            cx.run_until_parked();
            store.read_with(cx, |store, _| {
                assert_eq!(
                    store.selected_snapshot(),
                    Some(&current),
                    "obsolete Task {index}"
                );
            });
        }
        for (outcome, source_changed, expected_status) in [
            (None, false, SourceCoverageStatus::Cancelled),
            (
                Some(Ok(ExitStatus::default())),
                false,
                SourceCoverageStatus::Current,
            ),
            (None, true, SourceCoverageStatus::Stale),
            (
                Some(Ok(ExitStatus::default())),
                false,
                SourceCoverageStatus::Current,
            ),
        ] {
            let completion_index = completions.borrow().len();
            workspace.update_in(cx, |workspace, window, cx| {
                WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                    TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    compiled.task_template.clone(),
                    TaskContext {
                        cwd: Some("/fixture".into()),
                        ..TaskContext::default()
                    },
                    false,
                );
            });
            cx.run_until_parked();
            let loading = store.read_with(cx, |store, _| {
                store
                    .selected_snapshot()
                    .expect("fresh scheduled capture")
                    .clone()
            });
            assert_eq!(loading.status, SourceCoverageStatus::Loading);
            assert!(loading.generation > current.generation);
            if source_changed {
                let source = project.read_with(cx, |project, cx| {
                    project
                        .find_project_path("/fixture/src/bin/fixture.rs", cx)
                        .expect("admitted source journal path")
                });
                store.update(cx, |store, cx| {
                    store.invalidate_paths(std::slice::from_ref(&source), cx)
                });
                store.read_with(cx, |store, _| {
                    assert_eq!(
                        store
                            .selected_snapshot()
                            .expect("pending changed source")
                            .status,
                        SourceCoverageStatus::Loading,
                        "source journal mutation alone must not masquerade as a terminal outcome"
                    );
                });
            }
            completions
                .borrow_mut()
                .get_mut(completion_index)
                .expect("fresh scheduled Task")
                .take()
                .expect("pending completion")
                .send(outcome)
                .expect("fresh Task is pending");
            cx.run_until_parked();
            store.read_with(cx, |store, _| {
                let completed = store
                    .selected_snapshot()
                    .expect("current scheduled outcome");
                assert_eq!(completed.status, expected_status);
                assert_eq!(completed.generation, loading.generation);
                assert_eq!(
                    completed.files.is_empty(),
                    expected_status != SourceCoverageStatus::Current
                );
            });
        }
        cx.update(|_, cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let worktrees = project.read_with(cx, |project, _| project.worktree_store());
        cx.update(|_, cx| {
            project::trusted_worktrees::track_worktree_trust(
                worktrees.clone(),
                None,
                None,
                None,
                cx,
            )
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("trust global")
        });
        let trust_current_worktree = |cx: &mut gpui::VisualTestContext| {
            let worktree = project.read_with(cx, |project, cx| {
                project
                    .visible_worktrees(cx)
                    .next()
                    .expect("visible fixture")
                    .read(cx)
                    .id()
            });
            trust.update(cx, |trust, cx| {
                trust.trust(
                    &worktrees,
                    collections::HashSet::from_iter([
                        project::trusted_worktrees::PathTrust::Worktree(worktree),
                    ]),
                    cx,
                )
            });
            worktree
        };
        trust_current_worktree(cx);
        cx.run_until_parked();
        for restricted in [true, false] {
            let completed_before = completions.borrow().len();
            workspace.update_in(cx, |workspace, window, cx| {
                WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                    TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    compiled.task_template.clone(),
                    TaskContext {
                        cwd: Some("/fixture".into()),
                        ..TaskContext::default()
                    },
                    false,
                );
            });
            cx.run_until_parked();
            store.read_with(cx, |store, _| {
                assert_eq!(
                    store.selected_snapshot().expect("pending coverage").status,
                    SourceCoverageStatus::Loading
                )
            });
            let worktree = project.read_with(cx, |project, cx| {
                project
                    .visible_worktrees(cx)
                    .next()
                    .expect("current fixture")
                    .read(cx)
                    .id()
            });
            let expected = if restricted {
                trust.update(cx, |trust, cx| {
                    trust.restrict(
                        worktrees.downgrade(),
                        collections::HashSet::from_iter([
                            project::trusted_worktrees::PathTrust::Worktree(worktree),
                        ]),
                        cx,
                    )
                });
                SourceCoverageStatus::Restricted
            } else {
                project.update(cx, |project, cx| project.remove_worktree(worktree, cx));
                SourceCoverageStatus::Stale
            };
            cx.run_until_parked();
            let expired = store.read_with(cx, |store, _| {
                let snapshot = store
                    .selected_snapshot()
                    .expect("expired pending coverage")
                    .clone();
                assert_eq!(
                    snapshot.status, expected,
                    "pending Loading must end on lifecycle revocation"
                );
                snapshot
            });
            completions
                .borrow_mut()
                .get_mut(completed_before)
                .expect("expired Task")
                .take()
                .expect("held expired completion")
                .send(Some(Ok(ExitStatus::default())))
                .expect("expired Task is pending");
            cx.run_until_parked();
            store.read_with(cx, |store, _| {
                assert_eq!(
                    store.selected_snapshot(),
                    Some(&expired),
                    "obsolete success cannot replace lifecycle state"
                )
            });
            if !restricted {
                let (worktree, _) = project
                    .update(cx, |project, cx| {
                        project.find_or_create_worktree("/fixture", true, cx)
                    })
                    .await
                    .expect("restore worktree");
                worktree
                    .read_with(cx, |worktree, _| {
                        worktree.as_local().expect("local worktree").scan_complete()
                    })
                    .await;
            }
            trust_current_worktree(cx);
            cx.run_until_parked();
            let recovery_index = completions.borrow().len();
            workspace.update_in(cx, |workspace, window, cx| {
                WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                    TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    compiled.task_template.clone(),
                    TaskContext {
                        cwd: Some("/fixture".into()),
                        ..TaskContext::default()
                    },
                    false,
                );
            });
            cx.run_until_parked();
            completions
                .borrow_mut()
                .get_mut(recovery_index)
                .expect("recovery Task")
                .take()
                .expect("held recovery completion")
                .send(Some(Ok(ExitStatus::default())))
                .expect("recovery Task is pending");
            cx.run_until_parked();
            store.read_with(cx, |store, _| {
                assert_eq!(
                    store
                        .selected_snapshot()
                        .expect("fresh lifecycle recovery")
                        .status,
                    SourceCoverageStatus::Current
                )
            });
        }
        workspace.update_in(cx, |workspace, window, cx| {
            WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                TaskSourceKind::Language {
                    name: "Cargo".into(),
                },
                compiled.task_template,
                TaskContext {
                    cwd: Some(PathBuf::from("/fixture")),
                    ..TaskContext::default()
                },
                false,
            );
        });
        cx.run_until_parked();
        assert_eq!(completions.borrow().len(), 14);
        project.update(cx, |project, cx| project.close(cx));
        cx.run_until_parked();
        let closed = store.read_with(cx, |store, _| {
            let snapshot = store.selected_snapshot().expect("closed coverage").clone();
            assert_eq!(snapshot.status, SourceCoverageStatus::Disconnected);
            snapshot
        });
        completions
            .borrow_mut()
            .get_mut(13)
            .expect("closed run")
            .take()
            .expect("closed run completion")
            .send(Some(Ok(ExitStatus::default())))
            .expect("closed Task is pending");
        cx.run_until_parked();
        store.read_with(cx, |store, _| {
            assert_eq!(
                store.selected_snapshot(),
                Some(&closed),
                "Task success after project close must not republish coverage"
            );
        });
        cx.read(|cx| {
            assert!(
                cx.windows().is_empty(),
                "Project.close removes the Workspace window"
            );
        });
    }

    #[gpui::test]
    async fn cargo_coverage_action_current_save_seals_after_dirty_source_save(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::UnchangedSource,
            true,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_all_save_seals_after_dirty_source_save(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::All,
            CoverageCaptureScenario::UnchangedSource,
            true,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_source_change_during_task_is_stale(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::ChangedSource,
            true,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_worktree_reorder_success(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::WorktreeReordered(ScheduledTaskResult::Success),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_worktree_reorder_failure(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::WorktreeReordered(ScheduledTaskResult::Failure),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_worktree_reorder_spawnfailed(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::WorktreeReordered(ScheduledTaskResult::SpawnFailed),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_worktree_reorder_cancelled(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::WorktreeReordered(ScheduledTaskResult::Cancelled),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_source_journal_overflow_success(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::SourceJournalOverflow(ScheduledTaskResult::Success),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_source_journal_overflow_failure(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::SourceJournalOverflow(ScheduledTaskResult::Failure),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_source_journal_overflow_spawnfailed(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::SourceJournalOverflow(ScheduledTaskResult::SpawnFailed),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_during_task_on_source_journal_overflow_cancelled(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::SourceJournalOverflow(ScheduledTaskResult::Cancelled),
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_after_local_artifact_parse(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::ParsedSessionExpired,
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_rejects_source_change_after_local_artifact_parse(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::ParsedSourceChanged,
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_rejects_trust_revocation_after_local_artifact_parse(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::ParsedRestricted,
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_rejects_worktree_removal_after_local_artifact_parse(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::ParsedWorktreeRemoved,
            false,
            cx,
        )
        .await;
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::UnchangedSource,
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_rejects_project_close_after_local_artifact_parse(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::ParsedProjectClosed,
            false,
            cx,
        )
        .await;
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::UnchangedSource,
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_malformed_task_artifact_is_error(cx: &mut gpui::TestAppContext) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::MalformedArtifact,
            true,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_single_current_save_seals_after_dirty_source_save(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::Current,
            CoverageCaptureScenario::UnchangedSource,
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_single_all_save_seals_after_dirty_source_save(
        cx: &mut gpui::TestAppContext,
    ) {
        check_coverage_action_saves_before_sealing(
            SaveStrategy::All,
            CoverageCaptureScenario::UnchangedSource,
            false,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn cargo_coverage_action_owned_session_expires_while_waiting_for_save_admission(
        cx: &mut gpui::TestAppContext,
    ) {
        use futures::channel::oneshot;
        use gpui::{App, AppContext as _, UpdateGlobal as _};
        use settings::SettingsStore;
        use std::{cell::RefCell, rc::Rc, sync::Arc};

        #[derive(Clone)]
        struct Terminal {
            fs: Arc<project::FakeFs>,
            launches: Rc<RefCell<Vec<String>>>,
        }
        impl workspace::TerminalProvider for Terminal {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                spawn_prepared_recorded_terminal(self.clone(), launch, handle, prepare, window, cx)
            }

            fn spawn(
                &self,
                task: task::SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                self.launches.borrow_mut().push(task.command_label);
                if task.command.as_deref() != Some("cargo") {
                    return gpui::Task::ready(Some(Ok(std::process::ExitStatus::default())));
                }
                let fs = self.fs.clone();
                let artifact = task.cwd.as_ref().expect("coverage launch directory").join(
                    task.args
                        .get(
                            task.args
                                .iter()
                                .position(|argument| argument == "--output-path")
                                .expect("coverage output flag")
                                + 1,
                        )
                        .expect("coverage output argument"),
                );
                cx.background_spawn(async move {
                    let report =
                        include_str!("../../project/test_data/source_coverage/llvm3-run.json")
                            .replace("/workspace/", "/fixture/");
                    fs.insert_file(&artifact, report.into_bytes()).await;
                    Some(Ok(std::process::ExitStatus::default()))
                })
            }
        }
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.workspace.use_system_path_prompts = Some(false);
                });
            });
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                "src": {"bin": {"fixture.rs": "fn main() {\n    work();\n}\n"}},
                "target": {}
            }),
        )
        .await;
        let project = project::Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let buffer = project
            .update(cx, |project, cx| project.create_buffer(None, true, cx))
            .await
            .expect("untitled buffer");
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let untitled = workspace.update_in(cx, |workspace, window, cx| {
            let editor =
                cx.new(|cx| editor::Editor::for_buffer(buffer, Some(project.clone()), window, cx));
            editor.update(cx, |editor, cx| {
                editor.edit(
                    [(
                        editor::MultiBufferOffset(0)..editor::MultiBufferOffset(0),
                        "fn predecessor() {}\n",
                    )],
                    cx,
                )
            });
            workspace.add_item_to_active_pane(Box::new(editor.clone()), None, true, window, cx);
            editor
        });
        let (release_save, held_save) = oneshot::channel();
        let held_save = RefCell::new(Some(held_save));
        let prompts = Rc::new(RefCell::new(0));
        let launches = Rc::new(RefCell::new(Vec::new()));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(Terminal {
                fs: fs.clone(),
                launches: launches.clone(),
            });
            let prompts = prompts.clone();
            workspace.set_prompt_for_new_path(Box::new(move |_, _, _, _, _| {
                *prompts.borrow_mut() += 1;
                held_save
                    .borrow_mut()
                    .take()
                    .expect("only predecessor requests a save path")
            }));
        });
        let predecessor = TaskTemplate {
            label: "predecessor held configured save".into(),
            command: "echo".into(),
            args: vec!["predecessor".into()],
            save: SaveStrategy::Current,
            ..TaskTemplate::default()
        }
        .resolve_task("coverage-prerequisite-predecessor", &TaskContext::default())
        .expect("predecessor Task");
        let predecessor_result = Rc::new(RefCell::new(None));
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_admission_and_completion(
                TaskSourceKind::UserInput,
                predecessor,
                false,
                {
                    let untitled = untitled.clone();
                    move |_, cx| {
                        assert!(
                            !untitled.read(cx).buffer().read(cx).is_dirty(cx),
                            "predecessor admission follows its real configured save"
                        );
                        Ok(())
                    }
                },
                {
                    let predecessor_result = predecessor_result.clone();
                    move |result, _| *predecessor_result.borrow_mut() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(
            *prompts.borrow(),
            1,
            "actual configured save is held at its path prompt"
        );
        assert!(launches.borrow().is_empty());
        assert!(untitled.read_with(cx, |editor, cx| editor.buffer().read(cx).is_dirty(cx)));
        let source = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/fixture/src/bin/fixture.rs", cx)
                .expect("mapped source")
        });
        workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.open_path(source, None, true, window, cx)
            })
            .await
            .expect("activate mapped source");
        let active = workspace.read_with(cx, |workspace, cx| {
            workspace
                .active_item_as::<editor::Editor>(cx)
                .expect("mapped source Editor")
        });
        active.update(cx, |editor, cx| {
            editor.edit(
                [(
                    editor::MultiBufferOffset(0)..editor::MultiBufferOffset(0),
                    "// newest dirty source\n",
                )],
                cx,
            )
        });
        cx.run_until_parked();
        let original_source = fs
            .read_file_sync("/fixture/src/bin/fixture.rs")
            .expect("mapped source before newest configured save");
        let writes_before = fs.write_count_for_path("/fixture/src/bin/fixture.rs");
        let mut compiled = compile_legacy_coverage_plan(
            compile_preset(
                &CargoPreset::ephemeral_default(CargoSubcommand::Run),
                &CargoCompileContext {
                    workspace_cwd: Some("/fixture".into()),
                    ..CargoCompileContext::default()
                },
                None,
            )
            .expect("run plan"),
        )
        .expect("coverage plan");
        compiled.task_template.save = SaveStrategy::Current;
        workspace.update_in(cx, |workspace, window, cx| {
            WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                TaskSourceKind::Language {
                    name: "Cargo".into(),
                },
                compiled.task_template.clone(),
                TaskContext {
                    cwd: Some("/fixture".into()),
                    ..TaskContext::default()
                },
                false,
            );
        });
        cx.run_until_parked();
        let coverage = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        coverage.read_with(cx, |coverage, _| {
            assert_eq!(
                coverage
                    .selected_snapshot()
                    .expect("newest held capture")
                    .status,
                SourceCoverageStatus::Loading
            );
        });
        assert!(
            launches.borrow().is_empty(),
            "predecessor still holds prerequisite gate"
        );
        assert_eq!(
            fs.write_count_for_path("/fixture/src/bin/fixture.rs"),
            writes_before
        );
        coverage.update(cx, |coverage, cx| {
            coverage.invalidate(SourceCoverageStatus::Stale, cx)
        });
        coverage.read_with(cx, |coverage, _| {
            assert_eq!(
                coverage
                    .selected_snapshot()
                    .expect("expired first capture")
                    .status,
                SourceCoverageStatus::Loading,
                "session expiry alone cannot invalidate files absent from first capture"
            );
        });
        release_save
            .send(Some(vec![PathBuf::from("/fixture/predecessor.rs")]))
            .expect("predecessor awaits its real configured save prompt");
        cx.run_until_parked();
        assert_eq!(
            *predecessor_result.borrow(),
            Some(ScheduledTaskResult::Success)
        );
        assert_eq!(
            launches.borrow().len(),
            1,
            "expired newest coverage must not launch"
        );
        assert_eq!(*prompts.borrow(), 1);
        assert_eq!(
            fs.write_count_for_path("/fixture/src/bin/fixture.rs"),
            writes_before,
            "expired newest admission must not perform its configured source save"
        );
        assert_eq!(
            fs.read_file_sync("/fixture/src/bin/fixture.rs")
                .expect("source remains on disk"),
            original_source
        );
        assert!(active.read_with(cx, |editor, cx| editor.buffer().read(cx).is_dirty(cx)));
        coverage.read_with(cx, |coverage, _| {
            let snapshot = coverage
                .selected_snapshot()
                .expect("owned expired cancellation");
            assert_eq!(snapshot.status, SourceCoverageStatus::Stale);
            assert!(snapshot.files.is_empty());
            assert_eq!(
                snapshot.diagnostic.as_deref(),
                Some("Coverage Task expired because the project session or trust changed")
            );
        });
        workspace.read_with(cx, |workspace, _| {
            assert!(
                workspace
                    .message_notification_for_test(&NotificationId::unique::<
                        CargoCoverageNotification,
                    >())
                    .is_some(),
                "owned expired preflight must expose its bounded guidance"
            );
        });
        assert_eq!(
            active.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        workspace.update_in(cx, |workspace, window, cx| {
            WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                TaskSourceKind::Language {
                    name: "Cargo".into(),
                },
                compiled.task_template,
                TaskContext {
                    cwd: Some("/fixture".into()),
                    ..TaskContext::default()
                },
                false,
            );
        });
        cx.run_until_parked();
        assert_eq!(
            launches.borrow().len(),
            2,
            "fresh admitted coverage recovers normally"
        );
        assert!(!active.read_with(cx, |editor, cx| editor.buffer().read(cx).is_dirty(cx)));
        coverage.read_with(cx, |coverage, _| {
            assert_eq!(
                coverage
                    .selected_snapshot()
                    .expect("fresh coverage after expiry")
                    .status,
                SourceCoverageStatus::Current
            );
        });
        assert!(
            active
                .read_with(cx, |editor, _| editor.source_coverage_gutter_counts())
                .0
                > 0
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum CoverageCaptureScenario {
        UnchangedSource,
        ChangedSource,
        MalformedArtifact,
        WorktreeReordered(ScheduledTaskResult),
        SourceJournalOverflow(ScheduledTaskResult),
        ParsedSessionExpired,
        ParsedSourceChanged,
        ParsedRestricted,
        ParsedWorktreeRemoved,
        ParsedProjectClosed,
    }

    async fn check_coverage_action_saves_before_sealing(
        save: SaveStrategy,
        scenario: CoverageCaptureScenario,
        schedule_superseded_save: bool,
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::{App, AppContext as _, Entity};
        use project::Fs as _;
        use settings::SettingsStore;
        use std::{cell::RefCell, rc::Rc, sync::Arc};

        #[derive(Clone)]
        struct Terminal {
            fs: Arc<project::FakeFs>,
            active: Entity<editor::Editor>,
            inactive: Entity<editor::Editor>,
            dirty_at_launch: Rc<RefCell<Vec<(bool, bool)>>>,
            scenario: CoverageCaptureScenario,
            completion: Rc<RefCell<Option<futures::channel::oneshot::Receiver<()>>>>,
        }
        impl workspace::TerminalProvider for Terminal {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                spawn_prepared_recorded_terminal(self.clone(), launch, handle, prepare, window, cx)
            }

            fn spawn(
                &self,
                task: task::SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                self.dirty_at_launch.borrow_mut().push((
                    self.active.read(cx).buffer().read(cx).is_dirty(cx),
                    self.inactive.read(cx).buffer().read(cx).is_dirty(cx),
                ));
                let fs = self.fs.clone();
                let scenario = self.scenario;
                let artifact = task.cwd.as_ref().expect("coverage launch directory").join(
                    task.args
                        .get(
                            task.args
                                .iter()
                                .position(|argument| argument == "--output-path")
                                .expect("coverage output flag")
                                + 1,
                        )
                        .expect("coverage output argument"),
                );
                let completion = self
                    .completion
                    .borrow_mut()
                    .take()
                    .expect("only newest Task may reach the terminal");
                cx.background_spawn(async move {
                    completion.await.expect("release the admitted latest Task");
                    match scenario {
                        CoverageCaptureScenario::WorktreeReordered(
                            ScheduledTaskResult::Failure,
                        )
                        | CoverageCaptureScenario::SourceJournalOverflow(
                            ScheduledTaskResult::Failure,
                        ) => {
                            #[cfg(unix)]
                            {
                                use std::os::unix::process::ExitStatusExt;
                                return Some(Ok(std::process::ExitStatus::from_raw(1 << 8)));
                            }
                            #[cfg(windows)]
                            {
                                use std::os::windows::process::ExitStatusExt;
                                return Some(Ok(std::process::ExitStatus::from_raw(1)));
                            }
                        }
                        CoverageCaptureScenario::WorktreeReordered(
                            ScheduledTaskResult::SpawnFailed,
                        )
                        | CoverageCaptureScenario::SourceJournalOverflow(
                            ScheduledTaskResult::SpawnFailed,
                        ) => {
                            return Some(Err(anyhow!("controlled terminal launch failure")));
                        }
                        CoverageCaptureScenario::WorktreeReordered(
                            ScheduledTaskResult::Cancelled,
                        )
                        | CoverageCaptureScenario::SourceJournalOverflow(
                            ScheduledTaskResult::Cancelled,
                        ) => {
                            return None;
                        }
                        _ => {}
                    }
                    let source = fs
                        .load("/fixture/src/bin/fixture.rs".as_ref())
                        .await
                        .expect("saved coverage source");
                    assert!(
                        source.starts_with("// active dirty\n"),
                        "configured save must reach disk before the Task launches"
                    );
                    let report = if scenario == CoverageCaptureScenario::MalformedArtifact {
                        "{}".to_string()
                    } else {
                        include_str!("../../project/test_data/source_coverage/llvm3-run.json")
                            .replace("/workspace/", "/fixture/")
                    };
                    fs.insert_file(&artifact, report.into_bytes()).await;
                    Some(Ok(std::process::ExitStatus::default()))
                })
            }
        }
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
            if scenario == CoverageCaptureScenario::ParsedRestricted {
                project::trusted_worktrees::init(collections::HashMap::default(), cx);
            }
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/fixture", serde_json::json!({
            "src": {"bin": {"fixture.rs": "fn main() {}\n"}, "inactive.rs": "fn inactive() {}\n"},
            "target": {}
        })).await;
        let roots = if matches!(scenario, CoverageCaptureScenario::WorktreeReordered(_)) {
            fs.insert_tree(
                "/otherfixture",
                serde_json::json!({"independent.rs": "fn independent() {}"}),
            )
            .await;
            vec![PathBuf::from("/fixture"), PathBuf::from("/otherfixture")]
        } else {
            vec![PathBuf::from("/fixture")]
        };
        let project = if scenario == CoverageCaptureScenario::ParsedRestricted {
            let project = project::Project::test_with_worktree_trust(
                fs.clone(),
                roots.iter().map(PathBuf::as_path),
                cx,
            )
            .await;
            let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
            let paths = project.read_with(cx, |project, cx| {
                project
                    .visible_worktrees(cx)
                    .map(|worktree| {
                        project::trusted_worktrees::PathTrust::Worktree(worktree.read(cx).id())
                    })
                    .collect()
            });
            cx.update(|cx| {
                project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                    .expect("registered trust service")
                    .update(cx, |trust, cx| trust.trust(&worktree_store, paths, cx));
            });
            project
        } else {
            project::Project::test(fs.clone(), roots.iter().map(PathBuf::as_path), cx).await
        };
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let mut editors = Vec::new();
        for source in ["/fixture/src/inactive.rs", "/fixture/src/bin/fixture.rs"] {
            let path = project.read_with(cx, |project, cx| {
                project.find_project_path(source, cx).expect("source path")
            });
            workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.open_path(path, None, true, window, cx)
                })
                .await
                .expect("open source Editor");
            editors.push(workspace.read_with(cx, |workspace, cx| {
                workspace
                    .active_item_as::<editor::Editor>(cx)
                    .expect("active source Editor")
            }));
        }
        let inactive = editors.first().expect("inactive source").clone();
        let active = editors.get(1).expect("active source").clone();
        inactive.update(cx, |editor, cx| {
            editor.edit(
                [(
                    editor::MultiBufferOffset(0)..editor::MultiBufferOffset(0),
                    "// inactive dirty\n",
                )],
                cx,
            )
        });
        active.update(cx, |editor, cx| {
            editor.edit(
                [(
                    editor::MultiBufferOffset(0)..editor::MultiBufferOffset(0),
                    "// active dirty\n",
                )],
                cx,
            )
        });
        cx.run_until_parked();
        assert!(active.read_with(cx, |editor, cx| editor.buffer().read(cx).is_dirty(cx)));
        assert!(inactive.read_with(cx, |editor, cx| editor.buffer().read(cx).is_dirty(cx)));
        let dirty_at_launch = Rc::new(RefCell::new(Vec::new()));
        let (release, completion) = futures::channel::oneshot::channel();
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(Terminal {
                fs: fs.clone(),
                active: active.clone(),
                inactive: inactive.clone(),
                dirty_at_launch: dirty_at_launch.clone(),
                scenario,
                completion: Rc::new(RefCell::new(Some(completion))),
            })
        });
        let mut compiled = compile_legacy_coverage_plan(
            compile_preset(
                &CargoPreset::ephemeral_default(CargoSubcommand::Run),
                &CargoCompileContext {
                    workspace_cwd: Some("/fixture".into()),
                    ..CargoCompileContext::default()
                },
                None,
            )
            .expect("run plan"),
        )
        .expect("coverage plan");
        compiled.task_template.save = save;
        let labels = if schedule_superseded_save {
            vec!["superseded source save", "latest source save"]
        } else {
            vec!["single source save"]
        };
        for label in labels {
            let mut template = compiled.task_template.clone();
            template.label = label.into();
            workspace.update_in(cx, |workspace, window, cx| {
                WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                    TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    template,
                    TaskContext {
                        cwd: Some("/fixture".into()),
                        ..TaskContext::default()
                    },
                    false,
                );
            });
        }
        cx.run_until_parked();
        assert_eq!(
            *dirty_at_launch.borrow(),
            [(false, save == SaveStrategy::Current)]
        );
        workspace.read_with(cx, |workspace, _| {
            assert!(
                !workspace
                    .notification_ids()
                    .contains(&NotificationId::unique::<CargoCoverageNotification>()),
                "obsolete save admission must not replace the newest Task's notification"
            );
        });
        let coverage = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        coverage.read_with(cx, |coverage, _| {
            let snapshot = coverage
                .selected_snapshot()
                .expect("admitted Task is loading");
            assert_eq!(snapshot.status, SourceCoverageStatus::Loading);
            assert!(
                snapshot.files.is_empty(),
                "first capture has no previous mapped files"
            );
        });
        if scenario == CoverageCaptureScenario::ChangedSource {
            active.update(cx, |editor, cx| {
                editor.edit(
                    [(
                        editor::MultiBufferOffset(0)..editor::MultiBufferOffset(0),
                        "// changed during coverage Task\n",
                    )],
                    cx,
                )
            });
            cx.run_until_parked();
            coverage.read_with(cx, |coverage, _| {
                assert_eq!(
                    coverage
                        .selected_snapshot()
                        .expect("held first capture")
                        .status,
                    SourceCoverageStatus::Loading,
                    "a mapped source edit must be classified when the artifact is interpreted"
                );
            });
        }
        if matches!(scenario, CoverageCaptureScenario::WorktreeReordered(_)) {
            let worktrees = project.read_with(cx, |project, cx| {
                project
                    .visible_worktrees(cx)
                    .map(|worktree| worktree.read(cx).id())
                    .collect::<Vec<_>>()
            });
            project.update(cx, |project, cx| {
                project
                    .move_worktree(
                        *worktrees.first().expect("first visible fixture"),
                        *worktrees.get(1).expect("second visible fixture"),
                        cx,
                    )
                    .expect("actual worktree reorder");
            });
        } else if matches!(scenario, CoverageCaptureScenario::SourceJournalOverflow(_)) {
            let worktree_id = project.read_with(cx, |project, cx| {
                project
                    .find_project_path("/fixture/src/bin/fixture.rs", cx)
                    .expect("mapped source")
                    .worktree_id
            });
            let paths = (0..=project::source_coverage::MAX_SOURCE_COVERAGE_FILES)
                .map(|index| ProjectPath {
                    worktree_id,
                    path: RelPath::from_unix_str(&format!("changed-source-{index}.rs"))
                        .expect("changed path")
                        .into(),
                })
                .collect::<Vec<_>>();
            coverage.update(cx, |coverage, cx| coverage.invalidate_paths(&paths, cx));
        }
        if matches!(
            scenario,
            CoverageCaptureScenario::WorktreeReordered(_)
                | CoverageCaptureScenario::SourceJournalOverflow(_)
        ) {
            cx.run_until_parked();
            coverage.read_with(cx, |coverage, _| {
                let snapshot = coverage
                    .selected_snapshot()
                    .expect("held owned expired capture");
                assert_eq!(snapshot.status, SourceCoverageStatus::Loading);
                assert!(snapshot.files.is_empty());
            });
        }
        let parse_gate = if matches!(
            scenario,
            CoverageCaptureScenario::ParsedSessionExpired
                | CoverageCaptureScenario::ParsedSourceChanged
                | CoverageCaptureScenario::ParsedRestricted
                | CoverageCaptureScenario::ParsedWorktreeRemoved
                | CoverageCaptureScenario::ParsedProjectClosed
        ) {
            let provider = project.read_with(cx, |project, _| {
                project.rust_coverage_provider_store().clone()
            });
            Some(provider.update(cx, |provider, _| provider.gate_next_local_parse_for_test()))
        } else {
            None
        };
        release
            .send(())
            .expect("latest Task is held at actual terminal completion");
        let mut revoked_during_parse = None;
        if let Some((started, release_parse)) = parse_gate {
            started
                .await
                .expect("terminal Success reached actual local artifact parse");
            coverage.read_with(cx, |coverage, _| {
                assert_eq!(
                    coverage
                        .selected_snapshot()
                        .expect("parsed capture not published")
                        .status,
                    SourceCoverageStatus::Loading
                );
            });
            match scenario {
                CoverageCaptureScenario::ParsedSessionExpired => {
                    coverage.update(cx, |coverage, cx| {
                        coverage.invalidate(SourceCoverageStatus::Stale, cx)
                    });
                }
                CoverageCaptureScenario::ParsedSourceChanged => {
                    active.update(cx, |editor, cx| {
                        editor.edit(
                            [(
                                editor::MultiBufferOffset("// active dirty\n".len())
                                    ..editor::MultiBufferOffset("// active dirty\n".len()),
                                "// source changed after actual coverage parse\n",
                            )],
                            cx,
                        );
                    });
                }
                CoverageCaptureScenario::ParsedRestricted => {
                    let worktree_store =
                        project.read_with(cx, |project, _| project.worktree_store());
                    let worktree_id = project.read_with(cx, |project, cx| {
                        project
                            .visible_worktrees(cx)
                            .next()
                            .expect("visible fixture")
                            .read(cx)
                            .id()
                    });
                    cx.update(|_, cx| {
                        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                            .expect("registered trust service")
                            .update(cx, |trust, cx| {
                                trust.restrict(
                                    worktree_store.downgrade(),
                                    collections::HashSet::from_iter([
                                        project::trusted_worktrees::PathTrust::Worktree(
                                            worktree_id,
                                        ),
                                    ]),
                                    cx,
                                );
                            });
                    });
                }
                CoverageCaptureScenario::ParsedWorktreeRemoved => {
                    let worktree_id = project.read_with(cx, |project, cx| {
                        project
                            .visible_worktrees(cx)
                            .next()
                            .expect("mapped fixture worktree")
                            .read(cx)
                            .id()
                    });
                    project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
                }
                CoverageCaptureScenario::ParsedProjectClosed => {
                    project.update(cx, |project, cx| project.close(cx));
                }
                _ => panic!("only parse-gated scenarios reach this transition"),
            }
            cx.run_until_parked();
            if matches!(
                scenario,
                CoverageCaptureScenario::ParsedRestricted
                    | CoverageCaptureScenario::ParsedProjectClosed
                    | CoverageCaptureScenario::ParsedWorktreeRemoved
            ) {
                revoked_during_parse = Some(coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage.selected_snapshot().expect("revoked held parse").clone();
                    let expected = match scenario {
                        CoverageCaptureScenario::ParsedRestricted => SourceCoverageStatus::Restricted,
                        CoverageCaptureScenario::ParsedProjectClosed => SourceCoverageStatus::Disconnected,
                        CoverageCaptureScenario::ParsedWorktreeRemoved => SourceCoverageStatus::Stale,
                        _ => panic!("only actual lifecycle revocations record their held snapshot"),
                    };
                    assert_eq!(snapshot.status, expected, "lifecycle subscription must clear first-capture Loading while parsing is held");
                    assert!(snapshot.files.is_empty());
                    snapshot
                }));
            }
            release_parse
                .send(())
                .expect("actual local parse awaits release");
        }
        cx.run_until_parked();
        coverage.read_with(cx, |coverage, _| {
            let snapshot = coverage
                .selected_snapshot()
                .expect("coverage after configured source save");
            let expected_status = match scenario {
                CoverageCaptureScenario::UnchangedSource => SourceCoverageStatus::Current,
                CoverageCaptureScenario::ChangedSource
                | CoverageCaptureScenario::WorktreeReordered(_)
                | CoverageCaptureScenario::SourceJournalOverflow(_)
                | CoverageCaptureScenario::ParsedSessionExpired
                | CoverageCaptureScenario::ParsedSourceChanged
                | CoverageCaptureScenario::ParsedWorktreeRemoved => SourceCoverageStatus::Stale,
                CoverageCaptureScenario::ParsedRestricted => SourceCoverageStatus::Restricted,
                CoverageCaptureScenario::ParsedProjectClosed => SourceCoverageStatus::Disconnected,
                CoverageCaptureScenario::MalformedArtifact => SourceCoverageStatus::Error,
            };
            assert_eq!(snapshot.status, expected_status, "{snapshot:?}");
            if let Some(revoked) = revoked_during_parse.as_ref() {
                assert_eq!(snapshot, revoked, "delayed parsed result must preserve exact lifecycle outcome");
            }
            if scenario == CoverageCaptureScenario::UnchangedSource {
                assert_eq!(snapshot.files.len(), 1);
                assert_eq!(snapshot.files.first().expect("covered source").covered_lines, 3);
            } else {
                assert!(snapshot.files.is_empty(), "unverified report must not publish source ranges");
                let projection = language_tools::source_coverage_summary::SourceCoverageSummaryProjection::project(
                    snapshot,
                    language_tools::source_coverage_summary::SourceCoverageFilter::All,
                );
                assert!(projection.roots.iter().all(|root| root.children.is_empty()),
                    "rejected report must not expose source navigation targets");
            }
        });
        if matches!(
            scenario,
            CoverageCaptureScenario::WorktreeReordered(_)
                | CoverageCaptureScenario::SourceJournalOverflow(_)
                | CoverageCaptureScenario::ParsedSessionExpired
        ) {
            coverage.read_with(cx, |coverage, _| {
                assert_eq!(
                    coverage
                        .selected_snapshot()
                        .expect("expired capture retired")
                        .diagnostic
                        .as_deref(),
                    Some("Coverage Task expired because the project session or trust changed")
                );
            });
            workspace.read_with(cx, |workspace, _| {
                assert!(
                    workspace
                        .message_notification_for_test(&NotificationId::unique::<
                            CargoCoverageNotification,
                        >())
                        .is_some(),
                    "owned expired completion must expose bounded recovery guidance"
                );
            });
        }
        let gutters = active.read_with(cx, |editor, _| editor.source_coverage_gutter_counts());
        if scenario == CoverageCaptureScenario::UnchangedSource {
            assert!(
                gutters.0 > 0,
                "verified report must reach the mounted Editor"
            );
        } else {
            assert_eq!(
                gutters,
                (0, 0),
                "unverified report must not show current gutters"
            );
        }
        if matches!(
            scenario,
            CoverageCaptureScenario::WorktreeReordered(_)
                | CoverageCaptureScenario::SourceJournalOverflow(_)
                | CoverageCaptureScenario::ParsedSessionExpired
                | CoverageCaptureScenario::ParsedSourceChanged
                | CoverageCaptureScenario::ParsedRestricted
        ) {
            if scenario == CoverageCaptureScenario::ParsedRestricted {
                let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
                let paths = project.read_with(cx, |project, cx| {
                    project
                        .visible_worktrees(cx)
                        .map(|worktree| {
                            project::trusted_worktrees::PathTrust::Worktree(worktree.read(cx).id())
                        })
                        .collect()
                });
                cx.update(|_, cx| {
                    project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                        .expect("registered trust service")
                        .update(cx, |trust, cx| trust.trust(&worktree_store, paths, cx));
                });
                cx.run_until_parked();
            }
            let (release_fresh, completion) = futures::channel::oneshot::channel();
            workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(Terminal {
                    fs: fs.clone(),
                    active: active.clone(),
                    inactive: inactive.clone(),
                    dirty_at_launch: dirty_at_launch.clone(),
                    scenario: CoverageCaptureScenario::UnchangedSource,
                    completion: Rc::new(RefCell::new(Some(completion))),
                });
            });
            workspace.update_in(cx, |workspace, window, cx| {
                WorkspaceCargoActionDispatcher::new(workspace, window, cx).schedule_coverage_task(
                    TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    compiled.task_template.clone(),
                    TaskContext {
                        cwd: Some("/fixture".into()),
                        ..TaskContext::default()
                    },
                    false,
                );
            });
            cx.run_until_parked();
            assert_eq!(
                dirty_at_launch.borrow().len(),
                2,
                "fresh Task launches after expired outcome"
            );
            release_fresh.send(()).expect("fresh recovery Task is held");
            cx.run_until_parked();
            coverage.read_with(cx, |coverage, _| {
                let snapshot = coverage
                    .selected_snapshot()
                    .expect("fresh recovered coverage");
                assert_eq!(snapshot.status, SourceCoverageStatus::Current);
                assert_eq!(snapshot.files.len(), 1);
            });
            assert!(
                active
                    .read_with(cx, |editor, _| editor.source_coverage_gutter_counts())
                    .0
                    > 0,
                "fresh Task recovers current mounted source gutters"
            );
        }
        if scenario == CoverageCaptureScenario::ParsedProjectClosed {
            cx.read(|cx| {
                assert!(
                    cx.windows().is_empty(),
                    "actual Project.close retires its Workspace"
                )
            });
        } else {
            cx.deactivate_window();
            container.update_in(cx, |_, window, _| window.remove_window());
        }
    }

    fn path(worktree_id: WorktreeId, path: &'static str) -> ProjectPath {
        ProjectPath {
            worktree_id,
            path: RelPath::from_unix_str(path)
                .expect("fixture path should be relative")
                .into(),
        }
    }

    fn selection(kind: CargoActionNodeKind) -> CargoActionSelection {
        let worktree_id = WorktreeId::from_proto(7);
        CargoActionSelection {
            node_kind: kind,
            worktree_id,
            workspace_name: "workspace".to_string(),
            workspace_manifest: path(worktree_id, "nested/Cargo.toml"),
            package_name: Some("member with spaces".to_string()),
            package_manifest: Some(path(worktree_id, "nested/member/Cargo.toml")),
            target: None,
            has_bench_targets: false,
        }
    }

    #[test]
    fn cargo_actions_eligibility_table_and_runtime_denials_are_accessible() {
        use CargoActionNodeKind::{Package, Target, Workspace};
        use CargoActionTargetKind::{Bench, Binary, BuildScript, Library, Test};

        let cases = [
            (
                Workspace,
                vec![
                    CargoAction::Build,
                    CargoAction::Check,
                    CargoAction::Test,
                    CargoAction::Doc,
                    CargoAction::Clippy,
                    CargoAction::Fmt,
                    CargoAction::Clean,
                    CargoAction::Tree,
                ],
            ),
            (
                Package,
                vec![
                    CargoAction::Build,
                    CargoAction::Check,
                    CargoAction::Test,
                    CargoAction::Doc,
                    CargoAction::Clippy,
                    CargoAction::Fmt,
                    CargoAction::Clean,
                    CargoAction::Tree,
                ],
            ),
            (
                Target(Library),
                vec![
                    CargoAction::Build,
                    CargoAction::Check,
                    CargoAction::Test,
                    CargoAction::Doc,
                    CargoAction::Clippy,
                ],
            ),
            (
                Target(Binary),
                vec![
                    CargoAction::Build,
                    CargoAction::Check,
                    CargoAction::Run,
                    CargoAction::RunWithCoverage,
                    CargoAction::Profile,
                    CargoAction::Test,
                    CargoAction::Debug,
                    CargoAction::Doc,
                    CargoAction::Clippy,
                ],
            ),
            (
                Target(Test),
                vec![
                    CargoAction::Build,
                    CargoAction::Check,
                    CargoAction::Profile,
                    CargoAction::Test,
                    CargoAction::Debug,
                    CargoAction::Clippy,
                ],
            ),
            (
                Target(Bench),
                vec![
                    CargoAction::Build,
                    CargoAction::Check,
                    CargoAction::Profile,
                    CargoAction::Bench,
                    CargoAction::Debug,
                    CargoAction::Clippy,
                ],
            ),
            (
                Target(BuildScript),
                vec![CargoAction::Build, CargoAction::Check],
            ),
        ];
        for (kind, expected) in cases {
            let enabled = cargo_action_availability(&selection(kind), Default::default())
                .into_iter()
                .filter(|availability| availability.enabled)
                .map(|availability| availability.action)
                .collect::<Vec<_>>();
            assert_eq!(enabled, expected, "unexpected eligibility for {kind:?}");
        }

        let denial_cases = [
            (
                CargoActionRuntime {
                    trusted: false,
                    ..Default::default()
                },
                "trust this worktree",
            ),
            (
                CargoActionRuntime {
                    connected: false,
                    ..Default::default()
                },
                "disconnected",
            ),
            (
                CargoActionRuntime {
                    writable: false,
                    ..Default::default()
                },
                "read-only",
            ),
            (
                CargoActionRuntime {
                    cargo_available: false,
                    ..Default::default()
                },
                "Cargo is not available",
            ),
            (
                CargoActionRuntime {
                    host_capable: false,
                    ..Default::default()
                },
                "does not support Tasks and DAP",
            ),
        ];
        for (runtime, expected_reason) in denial_cases {
            let availability = cargo_action_availability(&selection(Target(Binary)), runtime);
            assert!(availability.iter().all(|entry| !entry.enabled));
            assert!(
                availability
                    .iter()
                    .find(|entry| entry.action == CargoAction::Build)
                    .is_some_and(|entry| entry.accessibility_label.contains(expected_reason))
            );
        }
    }

    #[gpui::test]
    async fn cargo_experimental_opt_ins_reject_disabled_and_revoke_pending_plans(
        cx: &mut gpui::TestAppContext,
    ) {
        let selected = selection(CargoActionNodeKind::Package);
        let context = TaskContext {
            cwd: Some("/fixture".into()),
            ..Default::default()
        };
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Build);
        preset.pre_launch_task = Some("authored preparation".into());
        let ordinary = plan_cargo_action(CargoAction::Build, &selected, None, &context)
            .expect("established default planning");
        let named = plan_cargo_action(CargoAction::Build, &selected, Some(&preset), &context)
            .expect("bounded named DATA plan");
        let mut coverage_selection =
            selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        coverage_selection.target = Some(CargoTargetSelector::Binary("fixture-main".into()));
        let coverage = plan_cargo_action(
            CargoAction::RunWithCoverage,
            &coverage_selection,
            None,
            &context,
        )
        .expect("bounded coverage DATA plan");
        let restored = match &named {
            CargoActionPlan::Task { template, .. } => serde_json::from_str::<TaskTemplate>(
                &serde_json::to_string(template).expect("literal Task serialization"),
            )
            .expect("cached/template requirements survive"),
            _ => panic!("named Build must remain a Task"),
        };
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            cx.update_flags(true, vec![]);
            assert!(
                ordinary.validate_experimental_access(cx).is_ok(),
                "staff does not remove established Cargo actions"
            );
            assert!(
                named.validate_experimental_access(cx).is_err(),
                "staff is not experimental opt-in"
            );
            assert!(coverage.validate_experimental_access(cx).is_err());
            assert!(ensure_experimental_template_access(&restored, cx).is_err());
            assert!(
                named.pre_launch().is_some(),
                "gate preserves original prelaunch continuation"
            );
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            assert!(named.validate_experimental_access(cx).is_ok());
            assert!(coverage.validate_experimental_access(cx).is_ok());
            assert!(ensure_experimental_template_access(&restored, cx).is_ok());
            let mut continuation = named.clone();
            continuation.clear_pre_launch();
            crate::set_experimental_cargo_opt_ins_for_test(false, cx);
            assert!(
                continuation.validate_experimental_access(cx).is_err(),
                "late successful authored preparation cannot restart revoked named planning"
            );
            assert!(
                ensure_experimental_template_access(&restored, cx).is_err(),
                "serialized requirements do not disappear on cached replay"
            );
            assert!(ordinary.validate_experimental_access(cx).is_ok());
        });
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn cargo_pre_launch_off_then_on_requires_fresh_continuation_without_cancelling_prerequisite(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::App;
        use std::{
            cell::{Cell, RefCell},
            rc::Rc,
            sync::Arc,
            time::{Duration, Instant},
        };
        type Release = Box<dyn Fn(&mut App)>;
        struct ActualPreLaunchTerminal {
            project: gpui::Entity<project::Project>,
            controls: Rc<RefCell<Vec<Release>>>,
            completed: Rc<Cell<usize>>,
        }
        impl ActualPreLaunchTerminal {
            fn spawn_actual(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: Option<workspace::TaskLaunchPreparation>,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                let project = self.project.clone();
                let controls = self.controls.clone();
                let completed = self.completed.clone();
                window.spawn(cx, async move |cx| {
                    let prepared = match prepare {
                        Some(prepare) => match prepare(launch, cx).await {
                            Ok(prepared) => prepared,
                            Err(error) => return Some(Err(error)),
                        },
                        None => {
                            assert!(
                                launch.required_experimental_features.is_empty(),
                                "marked continuation must consume original preparation"
                            );
                            workspace::PreparedTaskLaunch {
                                launch,
                                completion: None,
                                native_terminal: None,
                                pending_launch: None,
                            }
                        }
                    };
                    let terminal = match project
                        .update(cx, |project, cx| {
                            project.create_terminal_task_with_pending_launch(
                                prepared.launch,
                                handle,
                                prepared.native_terminal,
                                prepared.pending_launch,
                                cx,
                            )
                        })
                        .await
                    {
                        Ok(terminal) => terminal,
                        Err(error) => return Some(Err(error)),
                    };
                    controls.borrow_mut().push(Box::new({
                        let terminal = terminal.clone();
                        move |cx| {
                            terminal.update(cx, |terminal, _| terminal.input(b"release\n".to_vec()))
                        }
                    }));
                    let status = terminal
                        .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                        .await;
                    completed.set(completed.get() + 1);
                    let result = gpui::Task::ready(status.map(Ok));
                    match prepared.completion {
                        Some(complete) => complete(result, cx).await,
                        None => result.await,
                    }
                })
            }
        }
        impl workspace::TerminalProvider for ActualPreLaunchTerminal {
            fn spawn(
                &self,
                launch: task::SpawnInTerminal,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                self.spawn_actual(launch, None, None, window, cx)
            }
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> gpui::Task<Option<Result<std::process::ExitStatus>>> {
                self.spawn_actual(launch, handle, Some(prepare), window, cx)
            }
        }
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            crate::init(cx);
        });
        let directory = tempfile::tempdir().expect("owned pre-launch fixture");
        let started = directory.path().join("started");
        let finished = directory.path().join("finished");
        let continued = directory.path().join("continued");
        let project = project::Project::test(
            Arc::new(fs::RealFs::new(None, cx.executor())),
            [directory.path()],
            cx,
        )
        .await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let controls = Rc::new(RefCell::new(Vec::new()));
        let completed = Rc::new(Cell::new(0));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(ActualPreLaunchTerminal {
                project: project.clone(),
                controls: controls.clone(),
                completed: completed.clone(),
            })
        });
        let worktree_id = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .next()
                .expect("actual fixture Worktree")
                .read(cx)
                .id()
        });
        let prerequisite = TaskTemplate {
            label: "original ordinary prerequisite".into(),
            command: "/bin/sh".into(),
            args: vec!["-c".into(), "printf started > \"$PRE_LAUNCH_STARTED\"; read release; printf finished > \"$PRE_LAUNCH_FINISHED\"".into()],
            literal_arguments: true,
            env: [("PRE_LAUNCH_STARTED".into(), started.to_str().expect("fixture UTF-8").into()), ("PRE_LAUNCH_FINISHED".into(), finished.to_str().expect("fixture UTF-8").into())].into_iter().collect(),
            save: task::SaveStrategy::None,
            ..Default::default()
        };
        let task_json =
            serde_json::to_string(&vec![prerequisite]).expect("actual existing Task-file format");
        let task_path = directory.path().join("tasks.json");
        project
            .read_with(cx, |project, cx| {
                project
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .cloned()
                    .expect("existing Task inventory")
            })
            .update(cx, |inventory, _| {
                inventory.update_file_based_tasks(
                    project::task_store::TaskSettingsLocation::Global(&task_path),
                    Some(&task_json),
                )
            })
            .expect("register actual ordinary Task reference");
        for (phase, named) in [(0usize, true), (1, true), (2, false)] {
            if phase != 0 {
                std::fs::remove_file(&started).expect("retire owned previous start marker");
                std::fs::remove_file(&finished).expect("retire owned previous finish marker");
            }
            if phase == 2 {
                std::fs::remove_file(&continued).expect("retire owned continuation marker");
            }
            let plan = CargoActionPlan::Task {
                source: TaskSourceKind::UserInput,
                template: TaskTemplate {
                    label: "literal experimental continuation".into(),
                    command: "/bin/sh".into(),
                    args: vec![
                        "-c".into(),
                        "printf continued > \"$PRE_LAUNCH_CONTINUED\"".into(),
                    ],
                    literal_arguments: true,
                    env: [(
                        "PRE_LAUNCH_CONTINUED".into(),
                        continued.to_str().expect("fixture UTF-8").into(),
                    )]
                    .into_iter()
                    .collect(),
                    required_experimental_features: if named {
                        vec!["rust-named-cargo-presets-experimental".into()]
                    } else {
                        Vec::new()
                    },
                    experimental_policy_version: 1,
                    save: task::SaveStrategy::None,
                    ..Default::default()
                },
                context: TaskContext {
                    cwd: Some(directory.path().to_path_buf()),
                    ..Default::default()
                },
                worktree_id,
                pre_launch_task: Some("original ordinary prerequisite".into()),
                requires_named_preset: named,
                omit_history: false,
            };
            workspace.update_in(cx, |workspace, window, cx| {
                dispatch_cargo_action(
                    plan,
                    &mut WorkspaceCargoActionDispatcher::new(workspace, window, cx),
                )
            });
            let deadline = Instant::now() + Duration::from_secs(30);
            while !started.is_file() {
                assert!(
                    Instant::now() < deadline,
                    "actual prerequisite failed to reach stdin hold"
                );
                cx.background_executor
                    .timer(Duration::from_millis(10))
                    .await;
            }
            assert!(!finished.exists());
            if phase != 1 {
                cx.update(|_, cx| crate::set_experimental_cargo_opt_ins_for_test(false, cx));
                cx.update(|_, cx| crate::set_experimental_cargo_opt_ins_for_test(true, cx));
            }
            assert!(!continued.exists());
            let index = if phase == 0 {
                0
            } else if phase == 1 {
                1
            } else {
                3
            };
            cx.update(|_, cx| {
                controls
                    .borrow()
                    .get(index)
                    .expect("actual prerequisite Terminal")(cx)
            });
            let expected_completed = if phase == 0 {
                1
            } else if phase == 1 {
                3
            } else {
                5
            };
            while completed.get() != expected_completed {
                assert!(
                    Instant::now() < deadline,
                    "actual child/continuation did not retire within original deadline"
                );
                cx.background_executor
                    .timer(Duration::from_millis(10))
                    .await;
            }
            cx.run_until_parked();
            assert_eq!(
                std::fs::read_to_string(&finished)
                    .expect("ordinary prerequisite completion marker"),
                "finished"
            );
            if phase == 0 {
                assert!(
                    !continued.exists(),
                    "Off then On cannot resume original experimental continuation"
                );
                assert_eq!(
                    controls.borrow().len(),
                    1,
                    "ordinary prerequisite finishes without experimental child launch"
                );
            } else {
                assert_eq!(
                    std::fs::read_to_string(&continued)
                        .expect("actual fresh/ordinary continuation marker"),
                    "continued"
                );
            }
        }
    }

    #[derive(Default)]
    struct FakeDispatcher {
        plans: Vec<CargoActionPlan>,
        pre_launch_plans: Vec<(String, TaskContext, WorktreeId, CargoActionPlan)>,
    }

    impl CargoActionDispatcher for FakeDispatcher {
        fn schedule_native_profile(
            &mut self,
            selection: CargoActionSelection,
            preset: CargoPreset,
            context: TaskContext,
        ) {
            let worktree_id = selection.worktree_id;
            self.plans.push(CargoActionPlan::NativeProfile {
                selection,
                preset,
                context,
                worktree_id,
                pre_launch_task: None,
                requires_named_preset: false,
            });
        }

        fn schedule_task(
            &mut self,
            source: TaskSourceKind,
            template: TaskTemplate,
            context: TaskContext,
            omit_history: bool,
        ) {
            self.plans.push(CargoActionPlan::Task {
                source,
                template,
                context,
                worktree_id: WorktreeId::from_proto(0),
                pre_launch_task: None,
                requires_named_preset: false,
                omit_history,
            });
        }

        fn start_debug_session(
            &mut self,
            scenario: DebugScenario,
            context: TaskContext,
            worktree_id: WorktreeId,
        ) {
            self.plans.push(CargoActionPlan::Debug {
                scenario,
                context,
                worktree_id,
                pre_launch_task: None,
                requires_named_preset: false,
            });
        }

        fn run_pre_launch_task(
            &mut self,
            reference: String,
            context: TaskContext,
            worktree_id: WorktreeId,
            continuation: CargoActionPlan,
        ) {
            self.pre_launch_plans
                .push((reference, context, worktree_id, continuation));
        }
    }

    #[test]
    fn cargo_profile_action_requires_configuration_and_dispatches_explicit_arguments() {
        let binary = selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        let context = TaskContext {
            cwd: Some(PathBuf::from("/remote/worktree")),
            ..Default::default()
        };
        assert!(plan_cargo_action(CargoAction::Profile, &binary, None, &context).is_err());
        let declaration = settings::CargoProfilerSettingsContent {
            command: Some("installed-profiler".into()),
            args: Some(vec!["record".into(), "argument with spaces".into()]),
            artifact_path: Some("target/profile.svg".into()),
            artifact_kind: Some("svg".into()),
            artifact_max_bytes: Some(1024),
        };
        let mut preset = crate::parse_preset_content(
            "profile",
            &settings::CargoPresetSettingsContent {
                subcommand: Some("run".into()),
                profiler: Some(declaration.clone()),
                ..Default::default()
            },
        )
        .expect("configured preset");
        preset
            .environment
            .insert("PROFILE_MODE".into(), "local".into());
        let plan = plan_cargo_action(CargoAction::Profile, &binary, Some(&preset), &context)
            .expect("explicit profile action");
        let mut dispatcher = FakeDispatcher::default();
        dispatch_cargo_action(plan, &mut dispatcher);
        let Some(CargoActionPlan::Task {
            template, context, ..
        }) = dispatcher.plans.first()
        else {
            panic!("Profile must dispatch through Tasks")
        };
        assert_eq!(template.command, "installed-profiler");
        assert_eq!(template.args, vec!["record", "argument with spaces"]);
        assert_eq!(
            template.env.get("PROFILE_MODE").map(String::as_str),
            Some("local")
        );
        assert_eq!(
            template.artifact.as_ref().map(|a| a.path.as_str()),
            Some("target/profile.svg")
        );
        assert_eq!(
            context.cwd.as_deref(),
            Some(std::path::Path::new("/remote/worktree/nested"))
        );
        assert_eq!(
            preset
                .profiler
                .as_ref()
                .expect("profiler")
                .settings_content(),
            declaration
        );
        let mut invalid = declaration;
        invalid.artifact_path = Some("../private/profile.svg".into());
        assert!(crate::CargoProfiler::from_settings(&invalid).is_err());
    }

    #[test]
    fn cargo_actions_route_exact_templates_and_debug_scenarios_to_fake_dispatcher() {
        let mut binary = selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        binary.target = Some(CargoTargetSelector::Binary("bin;not-shell".to_string()));
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.environment =
            HashMap::from_iter([("TOKEN".to_string(), "value with spaces".to_string())]);
        preset.args = vec!["--config".to_string(), "x='$(not-shell)'".to_string()];
        preset.presentation.save = SaveStrategy::All;
        preset.presentation.reveal = RevealStrategy::NoFocus;
        preset.presentation.reveal_target = RevealTarget::Center;
        let base_context = TaskContext {
            cwd: Some(PathBuf::from("/remote/worktree")),
            ..TaskContext::default()
        };

        let task_plan = plan_cargo_action(CargoAction::Run, &binary, Some(&preset), &base_context)
            .expect("binary run should compile");
        let mut dispatcher = FakeDispatcher::default();
        dispatch_cargo_action(task_plan, &mut dispatcher);
        let CargoActionPlan::Task {
            source,
            template,
            context,
            omit_history,
            ..
        } = &dispatcher.plans[0]
        else {
            panic!("run should route to the task scheduler")
        };
        assert!(matches!(source, TaskSourceKind::Language { name } if name == "Cargo"));
        assert_eq!(
            template.args,
            vec![
                "run",
                "--package",
                "member with spaces",
                "--bin",
                "bin;not-shell",
                "--config",
                "x='$(not-shell)'",
            ]
        );
        assert_eq!(template.env["TOKEN"], "value with spaces");
        assert_eq!(template.save, SaveStrategy::All);
        assert_eq!(template.reveal, RevealStrategy::NoFocus);
        assert_eq!(template.reveal_target, RevealTarget::Center);
        assert_eq!(
            context.cwd.as_deref(),
            Some(PathBuf::from("/remote/worktree/nested").as_path())
        );
        assert!(!omit_history);

        let debug_plan =
            plan_cargo_action(CargoAction::Debug, &binary, Some(&preset), &base_context)
                .expect("binary debug should compile");
        dispatch_cargo_action(debug_plan, &mut dispatcher);
        let CargoActionPlan::Debug {
            scenario,
            context,
            worktree_id,
            ..
        } = &dispatcher.plans[1]
        else {
            panic!("debug should route to DAP")
        };
        assert_eq!(*worktree_id, binary.worktree_id);
        assert_eq!(context.cwd, Some(PathBuf::from("/remote/worktree/nested")));
        let Some(BuildTaskDefinition::Template {
            task_template,
            locator_name,
        }) = scenario.build.as_ref()
        else {
            panic!("debug should contain a Cargo build task")
        };
        assert_eq!(
            task_template.args.first().map(String::as_str),
            Some("build")
        );
        assert_eq!(locator_name.as_deref(), Some("rust-cargo-locator"));
    }

    #[test]
    fn rust_workspace_comprehensive_fixture_compiles_the_selected_preset_without_tools() {
        let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../project/test_data/cargo_workspace/comprehensive-v1.json"
        ))
        .expect("comprehensive Cargo fixture should parse");
        let package = fixture["roots"][0]["metadata"]["packages"][0]["name"]
            .as_str()
            .expect("fixture package should have a name");
        let binary = fixture["roots"][0]["metadata"]["packages"][0]["targets"]
            .as_array()
            .and_then(|targets| {
                targets.iter().find(|target| {
                    target["kind"]
                        .as_array()
                        .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin"))
                })
            })
            .and_then(|target| target["name"].as_str())
            .expect("fixture package should have a binary");
        let mut selection = selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        selection.package_name = Some(package.to_string());
        selection.target = Some(CargoTargetSelector::Binary(binary.to_string()));
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.toolchain = Some("stable".to_string());
        preset.features = vec!["extra".to_string()];
        let context = TaskContext {
            cwd: Some(PathBuf::from("/fixture")),
            ..TaskContext::default()
        };
        let CargoActionPlan::Task { template, .. } = plan_cargo_action(
            CargoAction::RunWithCoverage,
            &selection,
            Some(&preset),
            &context,
        )
        .expect("fixture selection should compile") else {
            panic!("coverage selection should compile to an ordinary task")
        };
        assert_eq!(
            template.args,
            [
                "+stable",
                "llvm-cov",
                "--json",
                "--output-path",
                crate::CARGO_COVERAGE_ARTIFACT_PATH,
                "run",
                "--package",
                package,
                "--bin",
                binary,
                "--features",
                "extra",
            ]
        );
        assert_eq!(
            template
                .artifact_group
                .as_ref()
                .and_then(|group| group.files.get("report"))
                .map(|artifact| artifact.kind),
            Some(TaskArtifactKind::Data)
        );
    }

    #[test]
    fn cargo_pre_launch_task_gates_the_main_action_without_embedding_a_command() {
        let binary = selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.pre_launch_task = Some("Generate bindings".to_string());
        let base_context = TaskContext {
            cwd: Some(PathBuf::from("/remote/worktree")),
            ..TaskContext::default()
        };
        let plan = plan_cargo_action(CargoAction::Run, &binary, Some(&preset), &base_context)
            .expect("pre-launch preset should compile");
        let mut dispatcher = FakeDispatcher::default();
        dispatch_cargo_action(plan, &mut dispatcher);

        assert!(dispatcher.plans.is_empty());
        assert_eq!(dispatcher.pre_launch_plans.len(), 1);
        let (reference, context, worktree_id, continuation) = dispatcher
            .pre_launch_plans
            .pop()
            .expect("gate should be recorded");
        assert_eq!(reference, "Generate bindings");
        assert_eq!(context.cwd, Some(PathBuf::from("/remote/worktree/nested")));
        assert_eq!(worktree_id, binary.worktree_id);
        assert!(continuation.pre_launch().is_none());

        dispatch_cargo_action(continuation, &mut dispatcher);
        assert_eq!(dispatcher.plans.len(), 1);
    }

    fn task_args(plan: CargoActionPlan) -> Vec<String> {
        let CargoActionPlan::Task { template, .. } = plan else {
            panic!("extended Cargo actions should route through Tasks")
        };
        assert_eq!(template.command, "cargo");
        assert!(!template.allow_concurrent_runs);
        template.args
    }

    #[test]
    fn cargo_extended_action_matrix_uses_exact_safe_argv() {
        let workspace = selection(CargoActionNodeKind::Workspace);
        let package = selection(CargoActionNodeKind::Package);
        let binary = selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        let base_context = TaskContext {
            cwd: Some(PathBuf::from("/remote/worktree")),
            ..TaskContext::default()
        };

        let cases = [
            (CargoAction::Doc, &workspace, vec!["doc", "--workspace"]),
            (
                CargoAction::Clippy,
                &package,
                vec!["clippy", "--package", "member with spaces"],
            ),
            (CargoAction::Fmt, &workspace, vec!["fmt", "--all"]),
            (
                CargoAction::Fmt,
                &package,
                vec!["fmt", "--package", "member with spaces"],
            ),
            (CargoAction::Clean, &workspace, vec!["clean"]),
            (
                CargoAction::Clean,
                &package,
                vec!["clean", "--package", "member with spaces"],
            ),
            (
                CargoAction::Tree,
                &workspace,
                vec!["tree", "--workspace", "--locked", "--offline"],
            ),
            (
                CargoAction::Tree,
                &package,
                vec![
                    "tree",
                    "--package",
                    "member with spaces",
                    "--locked",
                    "--offline",
                ],
            ),
        ];
        for (action, selection, expected) in cases {
            let plan = plan_cargo_action_with_confirmation(
                action,
                selection,
                None,
                &base_context,
                action == CargoAction::Clean,
            )
            .expect("approved Cargo action should plan");
            assert_eq!(task_args(plan), expected, "unexpected argv for {action:?}");
        }

        for action in [CargoAction::Fmt, CargoAction::Clean, CargoAction::Tree] {
            let availability = cargo_action_availability(&binary, Default::default())
                .into_iter()
                .find(|availability| availability.action == action)
                .expect("every action should have an availability row");
            assert!(!availability.enabled);
            assert!(
                availability
                    .accessibility_label
                    .contains("requires a workspace or package")
            );
        }

        let source = include_str!("cargo_actions.rs");
        assert!(!source.contains(concat!("CargoAction::", "Update")));
        assert!(!source.contains(concat!("Self::", "Update")));
    }

    #[test]
    fn cargo_clean_confirmation_is_required_before_a_plan_can_be_scheduled() {
        let workspace = selection(CargoActionNodeKind::Workspace);
        let base_context = TaskContext {
            cwd: Some(PathBuf::from("/remote/worktree")),
            ..TaskContext::default()
        };
        let error = plan_cargo_action(CargoAction::Clean, &workspace, None, &base_context)
            .expect_err("unconfirmed Clean must not produce a task plan");
        assert!(error.to_string().contains("explicit confirmation"));

        let plan = plan_cargo_action_with_confirmation(
            CargoAction::Clean,
            &workspace,
            None,
            &base_context,
            true,
        )
        .expect("confirmed Clean should produce a task plan");
        assert_eq!(task_args(plan), vec!["clean"]);
    }

    #[test]
    fn cargo_actions_have_no_direct_process_execution_path() {
        let (source, _) = include_str!("cargo_actions.rs")
            .split_once("\n#[cfg(test)]\nmod tests {")
            .expect("the production action owner precedes its test module");
        assert!(!source.contains(concat!("std::process::", "Command")));
        assert!(!source.contains(concat!("Command", "::new")));
        assert!(source.contains("schedule_task"));
        assert!(source.contains("start_debug_session"));
    }

    #[test]
    fn cargo_native_profile_selected_action_preserves_request_and_prelaunch_owner() {
        let mut selected = selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        selected.target = Some(CargoTargetSelector::Binary("app".into()));
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.native_profiler = Some(
            crate::CargoNativeProfiler::from_settings(
                &settings::CargoNativeProfilerSettingsContent {
                    collector_path: Some("/opt/tools/samply".into()),
                    profile_path: Some("target/capture.json".into()),
                },
            )
            .expect("declarative native profiler"),
        );
        preset.pre_launch_task = Some("prepare-workload".into());
        preset.trailing_args = vec!["argument with spaces".into(), "$(literal)".into()];
        let context = TaskContext {
            cwd: Some("/host/worktree".into()),
            ..Default::default()
        };
        let plan = plan_cargo_action(CargoAction::Profile, &selected, Some(&preset), &context)
            .expect("selected native action data");
        let mut dispatcher = FakeDispatcher::default();
        dispatch_cargo_action(plan, &mut dispatcher);
        assert!(dispatcher.plans.is_empty(), "prelaunch stays Tasks-owned");
        let (reference, _, worktree_id, continuation) = dispatcher
            .pre_launch_plans
            .pop()
            .expect("existing task-reference continuation");
        assert_eq!(reference, "prepare-workload");
        assert_eq!(worktree_id, selected.worktree_id);
        dispatch_cargo_action(continuation, &mut dispatcher);
        let Some(CargoActionPlan::NativeProfile {
            selection,
            preset: emitted,
            context,
            ..
        }) = dispatcher.plans.first()
        else {
            panic!("native action must dispatch its Build intent request");
        };
        assert_eq!(selection, &selected);
        assert_eq!(emitted.native_profiler, preset.native_profiler);
        assert_eq!(emitted.trailing_args, preset.trailing_args);
        assert_eq!(
            context.cwd.as_deref(),
            Some(std::path::Path::new("/host/worktree/nested"))
        );
        assert!(emitted.profiler.is_none());
    }

    #[test]
    fn cargo_native_profile_action_name_and_denials_preserve_external_mode() {
        let mut selected = selection(CargoActionNodeKind::Target(CargoActionTargetKind::Binary));
        selected.target = Some(CargoTargetSelector::Binary("app".into()));
        let baseline = cargo_action_availability(&selected, CargoActionRuntime::default());
        assert_eq!(
            cargo_action_availability_with_preset(&selected, CargoActionRuntime::default(), None),
            baseline
        );
        let mut native = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        native.native_profiler = Some(
            crate::CargoNativeProfiler::from_settings(
                &settings::CargoNativeProfilerSettingsContent {
                    collector_path: Some("/opt/tools/samply".into()),
                    profile_path: Some("target/capture.json".into()),
                },
            )
            .expect("explicit native mode"),
        );
        let availability = cargo_action_availability_with_preset(
            &selected,
            CargoActionRuntime::default(),
            Some(&native),
        );
        let action = availability
            .iter()
            .find(|entry| entry.action == CargoAction::Profile)
            .expect("Profile action");
        assert!(action.enabled);
        assert_eq!(
            action.accessibility_label,
            "Native Profile (macOS) selected Cargo item"
        );
        for unsupported in [CargoActionTargetKind::Test, CargoActionTargetKind::Bench] {
            selected.node_kind = CargoActionNodeKind::Target(unsupported);
            selected.target = Some(CargoTargetSelector::Test("app".into()));
            let availability = cargo_action_availability_with_preset(
                &selected,
                CargoActionRuntime::default(),
                Some(&native),
            );
            let action = availability
                .iter()
                .find(|entry| entry.action == CargoAction::Profile)
                .expect("Profile denial");
            assert!(!action.enabled);
            assert!(
                action
                    .accessibility_label
                    .contains("Native Profile (macOS) unavailable")
            );
            assert!(
                action
                    .reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("remain incomplete"))
            );
        }
    }
}
