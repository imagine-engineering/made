use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use collections::HashMap;
use serde::{Deserialize, Serialize};
use settings::{CargoPresetSettingsContent, CargoSettingsContent, MergeFromTrait as _, Settings};
use task::{
    BuildTaskDefinition, DebugScenario, HideStrategy, RevealStrategy, RevealTarget, SaveStrategy,
    TaskTemplate,
};

pub const CARGO_PRESET_SCHEMA_VERSION: u32 = 2;
pub const CARGO_PRESET_WORKSPACE_STATE_VERSION: u32 = 1;
const CARGO_LOCATOR_NAME: &str = "rust-cargo-locator";
const MAX_PRESETS: usize = 256;
const MAX_PRESET_IDENTIFIER_BYTES: usize = 128;
const MAX_PRESET_TEXT_BYTES: usize = 4096;
const MAX_PRESET_ITEMS: usize = 256;
const MAX_PRESET_DIAGNOSTICS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CargoSubcommand {
    Build,
    Check,
    Run,
    Test,
    Bench,
    Doc,
    Clippy,
    Fmt,
    Clean,
    Tree,
}

impl CargoSubcommand {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Check => "check",
            Self::Run => "run",
            Self::Test => "test",
            Self::Bench => "bench",
            Self::Doc => "doc",
            Self::Clippy => "clippy",
            Self::Fmt => "fmt",
            Self::Clean => "clean",
            Self::Tree => "tree",
        }
    }
}

impl TryFrom<&str> for CargoSubcommand {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "build" => Ok(Self::Build),
            "check" => Ok(Self::Check),
            "run" => Ok(Self::Run),
            "test" => Ok(Self::Test),
            "bench" => Ok(Self::Bench),
            "doc" => Ok(Self::Doc),
            "clippy" => Ok(Self::Clippy),
            "fmt" => Ok(Self::Fmt),
            "clean" => Ok(Self::Clean),
            "tree" => Ok(Self::Tree),
            _ => bail!("unsupported Cargo subcommand `{value}`"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CargoPresetScope {
    Workspace,
    Package,
}

impl TryFrom<&str> for CargoPresetScope {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "workspace" => Ok(Self::Workspace),
            "package" => Ok(Self::Package),
            _ => bail!("unsupported Cargo scope `{value}`"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum CargoTargetSelector {
    Library,
    Binary(String),
    Example(String),
    Test(String),
    Bench(String),
    AllTargets,
}

impl CargoTargetSelector {
    fn append_args(&self, args: &mut Vec<String>) {
        match self {
            Self::Library => args.push("--lib".to_string()),
            Self::Binary(name) => {
                args.push("--bin".to_string());
                args.push(name.clone());
            }
            Self::Example(name) => {
                args.push("--example".to_string());
                args.push(name.clone());
            }
            Self::Test(name) => {
                args.push("--test".to_string());
                args.push(name.clone());
            }
            Self::Bench(name) => {
                args.push("--bench".to_string());
                args.push(name.clone());
            }
            Self::AllTargets => args.push("--all-targets".to_string()),
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Library => "lib".to_string(),
            Self::Binary(name) => format!("bin {name}"),
            Self::Example(name) => format!("example {name}"),
            Self::Test(name) => format!("test {name}"),
            Self::Bench(name) => format!("bench {name}"),
            Self::AllTargets => "all targets".to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CargoWorkingDirectoryPolicy {
    Context,
    Workspace,
    Package,
    Custom(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CargoTaskPresentation {
    pub reveal: RevealStrategy,
    pub reveal_target: RevealTarget,
    pub hide: HideStrategy,
    pub save: SaveStrategy,
    pub use_new_terminal: bool,
    pub allow_concurrent_runs: bool,
}

impl Default for CargoTaskPresentation {
    fn default() -> Self {
        Self {
            reveal: RevealStrategy::Always,
            reveal_target: RevealTarget::Dock,
            hide: HideStrategy::Never,
            save: SaveStrategy::None,
            use_new_terminal: false,
            allow_concurrent_runs: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoPreset {
    pub id: String,
    pub label: String,
    pub subcommand: CargoSubcommand,
    pub scope: CargoPresetScope,
    pub package: Option<String>,
    pub target: Option<CargoTargetSelector>,
    pub profile: Option<String>,
    pub profiler: Option<crate::cargo_profile::CargoProfiler>,
    pub native_profiler: Option<crate::cargo_profile::CargoNativeProfiler>,
    pub features: Vec<String>,
    pub default_features: Option<bool>,
    pub target_triple: Option<String>,
    pub toolchain: Option<String>,
    pub pre_launch_task: Option<String>,
    pub args: Vec<String>,
    pub trailing_args: Vec<String>,
    pub environment: HashMap<String, String>,
    pub working_directory: CargoWorkingDirectoryPolicy,
    pub presentation: CargoTaskPresentation,
}

impl CargoPreset {
    pub fn ephemeral_default(subcommand: CargoSubcommand) -> Self {
        Self {
            id: "cargo-default".to_string(),
            label: format!("Cargo {}", subcommand.as_str()),
            subcommand,
            scope: CargoPresetScope::Workspace,
            package: None,
            target: None,
            profile: None,
            profiler: None,
            native_profiler: None,
            features: Vec::new(),
            default_features: None,
            target_triple: None,
            toolchain: None,
            pre_launch_task: None,
            args: Vec::new(),
            trailing_args: Vec::new(),
            environment: HashMap::default(),
            working_directory: CargoWorkingDirectoryPolicy::Context,
            presentation: CargoTaskPresentation::default(),
        }
    }

    pub fn profile_action_label(&self) -> &'static str {
        if self.native_profiler.is_some() {
            "Native Profile (macOS)"
        } else {
            "Profile"
        }
    }

    pub fn environment_keys(&self) -> Vec<String> {
        let mut keys = self.environment.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        keys
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoPresetDiagnostic {
    pub preset_id: Option<String>,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct CargoPresetSettings {
    pub presets: BTreeMap<String, CargoPreset>,
    pub diagnostics: Vec<CargoPresetDiagnostic>,
}

impl Settings for CargoPresetSettings {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        content
            .project
            .cargo
            .as_ref()
            .map(parse_settings_content)
            .unwrap_or_default()
    }
}

pub fn merge_preset_content(
    user: &CargoSettingsContent,
    project: &CargoSettingsContent,
) -> CargoSettingsContent {
    let mut merged = user.clone();
    merged.merge_from(project);
    merged
}

pub fn parse_settings_content(content: &CargoSettingsContent) -> CargoPresetSettings {
    let schema_version = content
        .schema_version
        .unwrap_or(CARGO_PRESET_SCHEMA_VERSION);
    if !matches!(schema_version, 1 | CARGO_PRESET_SCHEMA_VERSION) {
        return CargoPresetSettings {
            presets: BTreeMap::new(),
            diagnostics: vec![CargoPresetDiagnostic {
                preset_id: None,
                message: bounded_message(format!(
                    "Unsupported Cargo preset schema version {schema_version}; expected {CARGO_PRESET_SCHEMA_VERSION}"
                )),
            }],
        };
    }

    let mut parsed = CargoPresetSettings::default();
    let mut entries = content.presets.iter().collect::<Vec<_>>();
    entries.sort_by_key(|(preset_id, _)| *preset_id);
    if entries.len() > MAX_PRESETS {
        parsed.diagnostics.push(CargoPresetDiagnostic {
            preset_id: None,
            message: format!(
                "Cargo presets were truncated from {} to {MAX_PRESETS} entries",
                entries.len()
            ),
        });
        entries.truncate(MAX_PRESETS);
    }
    for (id, content) in entries {
        match parse_preset(id, content) {
            Ok(preset) => {
                parsed.presets.insert(id.clone(), preset);
            }
            Err(error) if parsed.diagnostics.len() < MAX_PRESET_DIAGNOSTICS => {
                parsed.diagnostics.push(CargoPresetDiagnostic {
                    preset_id: Some(bounded_text(id)),
                    message: bounded_message(error.to_string()),
                });
            }
            Err(_) => {}
        }
    }
    parsed
}

fn parse_preset(id: &str, content: &CargoPresetSettingsContent) -> Result<CargoPreset> {
    validate_identifier("preset identifier", id)?;
    let subcommand: CargoSubcommand = content
        .subcommand
        .as_deref()
        .ok_or_else(|| anyhow!("missing required `subcommand`"))?
        .try_into()?;
    let scope = content.scope.as_deref().unwrap_or("workspace").try_into()?;
    let target = parse_target(
        content.target_kind.as_deref(),
        content.target_name.as_deref(),
    )?;
    let working_directory = match content.working_directory.as_deref().unwrap_or("context") {
        "context" => CargoWorkingDirectoryPolicy::Context,
        "workspace" => CargoWorkingDirectoryPolicy::Workspace,
        "package" => CargoWorkingDirectoryPolicy::Package,
        "custom" => CargoWorkingDirectoryPolicy::Custom(
            content
                .custom_working_directory
                .as_deref()
                .filter(|path| !path.trim().is_empty())
                .ok_or_else(|| anyhow!("custom working directory requires a non-empty path"))?
                .to_string(),
        ),
        value => bail!("unsupported working-directory policy `{value}`"),
    };
    let presentation = CargoTaskPresentation {
        reveal: parse_reveal(content.reveal.as_deref())?,
        reveal_target: parse_reveal_target(content.reveal_target.as_deref())?,
        hide: parse_hide(content.hide.as_deref())?,
        save: parse_save(content.save.as_deref())?,
        use_new_terminal: content.use_new_terminal.unwrap_or(false),
        allow_concurrent_runs: content.allow_concurrent_runs.unwrap_or(false),
    };
    let features = bounded_items("features", content.features.clone().unwrap_or_default())?;
    let args = bounded_items("args", content.args.clone().unwrap_or_default())?;
    let trailing_args = bounded_items(
        "trailing_args",
        content.trailing_args.clone().unwrap_or_default(),
    )?;
    let environment = content.environment.clone().unwrap_or_default();
    if environment.len() > MAX_PRESET_ITEMS {
        bail!("environment contains more than {MAX_PRESET_ITEMS} entries");
    }
    for (key, value) in &environment {
        validate_identifier("environment key", key)?;
        validate_text("environment value", value)?;
    }
    for (field, value) in [
        ("label", content.label.as_deref()),
        ("package", content.package.as_deref()),
        ("profile", content.profile.as_deref()),
        ("target_triple", content.target_triple.as_deref()),
        ("toolchain", content.toolchain.as_deref()),
        ("pre_launch_task", content.pre_launch_task.as_deref()),
    ] {
        if let Some(value) = value {
            validate_text(field, value)?;
        }
    }

    if content.profiler.is_some() && content.native_profiler.is_some() {
        bail!("Choose either the external profiler or native profiler for this Cargo preset");
    }

    Ok(CargoPreset {
        id: id.to_string(),
        label: content
            .label
            .clone()
            .unwrap_or_else(|| format!("Cargo {}", subcommand.as_str())),
        subcommand,
        scope,
        package: content.package.clone(),
        target,
        profile: content.profile.clone(),
        profiler: content
            .profiler
            .as_ref()
            .map(crate::cargo_profile::CargoProfiler::from_settings)
            .transpose()?,
        native_profiler: content
            .native_profiler
            .as_ref()
            .map(crate::cargo_profile::CargoNativeProfiler::from_settings)
            .transpose()?,
        features,
        default_features: content.default_features,
        target_triple: content.target_triple.clone(),
        toolchain: content.toolchain.clone(),
        pre_launch_task: content.pre_launch_task.clone(),
        args,
        trailing_args,
        environment,
        working_directory,
        presentation,
    })
}

pub fn parse_preset_content(id: &str, content: &CargoPresetSettingsContent) -> Result<CargoPreset> {
    parse_preset(id, content)
}

fn parse_target(kind: Option<&str>, name: Option<&str>) -> Result<Option<CargoTargetSelector>> {
    let Some(kind) = kind else {
        if name.is_some() {
            bail!("target_name requires target_kind");
        }
        return Ok(None);
    };
    let named = |label: &str| -> Result<String> {
        let name = name
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| anyhow!("{label} target requires target_name"))?;
        validate_text("target_name", name)?;
        Ok(name.to_string())
    };
    match kind {
        "lib" => Ok(Some(CargoTargetSelector::Library)),
        "bin" => Ok(Some(CargoTargetSelector::Binary(named("binary")?))),
        "example" => Ok(Some(CargoTargetSelector::Example(named("example")?))),
        "test" => Ok(Some(CargoTargetSelector::Test(named("test")?))),
        "bench" => Ok(Some(CargoTargetSelector::Bench(named("bench")?))),
        "all_targets" => Ok(Some(CargoTargetSelector::AllTargets)),
        _ => bail!("unsupported Cargo target kind `{kind}`"),
    }
}

fn parse_reveal(value: Option<&str>) -> Result<RevealStrategy> {
    match value.unwrap_or("always") {
        "always" => Ok(RevealStrategy::Always),
        "no_focus" => Ok(RevealStrategy::NoFocus),
        "never" => Ok(RevealStrategy::Never),
        value => bail!("unsupported task reveal policy `{value}`"),
    }
}

fn parse_reveal_target(value: Option<&str>) -> Result<RevealTarget> {
    match value.unwrap_or("dock") {
        "dock" => Ok(RevealTarget::Dock),
        "center" => Ok(RevealTarget::Center),
        value => bail!("unsupported task reveal target `{value}`"),
    }
}

fn parse_hide(value: Option<&str>) -> Result<HideStrategy> {
    match value.unwrap_or("never") {
        "never" => Ok(HideStrategy::Never),
        "always" => Ok(HideStrategy::Always),
        "on_success" => Ok(HideStrategy::OnSuccess),
        value => bail!("unsupported task hide policy `{value}`"),
    }
}

fn parse_save(value: Option<&str>) -> Result<SaveStrategy> {
    match value.unwrap_or("none") {
        "none" => Ok(SaveStrategy::None),
        "current" => Ok(SaveStrategy::Current),
        "all" => Ok(SaveStrategy::All),
        value => bail!("unsupported task save policy `{value}`"),
    }
}

fn bounded_items(field: &str, values: Vec<String>) -> Result<Vec<String>> {
    if values.len() > MAX_PRESET_ITEMS {
        bail!("{field} contains more than {MAX_PRESET_ITEMS} entries");
    }
    for value in &values {
        validate_text(field, value)?;
    }
    Ok(values)
}

fn validate_identifier(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must not be empty");
    }
    if value.len() > MAX_PRESET_IDENTIFIER_BYTES {
        bail!("{field} exceeds {MAX_PRESET_IDENTIFIER_BYTES} bytes");
    }
    Ok(())
}

fn validate_text(field: &str, value: &str) -> Result<()> {
    if value.len() > MAX_PRESET_TEXT_BYTES {
        bail!("{field} exceeds {MAX_PRESET_TEXT_BYTES} bytes");
    }
    Ok(())
}

fn bounded_text(value: &str) -> String {
    let mut value = value.to_string();
    while value.len() > MAX_PRESET_IDENTIFIER_BYTES {
        value.pop();
    }
    value
}

fn bounded_message(value: String) -> String {
    let mut value = value;
    while value.len() > MAX_PRESET_TEXT_BYTES {
        value.pop();
    }
    value
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CargoCompileContext {
    pub workspace_name: Option<String>,
    pub workspace_cwd: Option<String>,
    pub package_name: Option<String>,
    pub package_cwd: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoTaskContextInputs {
    pub scope: CargoPresetScope,
    pub workspace_cwd: Option<String>,
    pub package_cwd: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledCargoPreset {
    pub task_template: TaskTemplate,
    pub task_context: CargoTaskContextInputs,
    pub pre_launch_task: Option<String>,
}

pub fn compile_preset(
    preset: &CargoPreset,
    context: &CargoCompileContext,
    subcommand_override: Option<CargoSubcommand>,
) -> Result<CompiledCargoPreset> {
    let subcommand = subcommand_override.unwrap_or(preset.subcommand);
    let mut args = Vec::new();
    if let Some(toolchain) = &preset.toolchain {
        args.push(format!("+{toolchain}"));
    }
    args.push(subcommand.as_str().to_string());
    match preset.scope {
        CargoPresetScope::Workspace => args.push("--workspace".to_string()),
        CargoPresetScope::Package => {
            let package = preset
                .package
                .as_ref()
                .or(context.package_name.as_ref())
                .ok_or_else(|| anyhow!("package scope requires a selected package"))?;
            args.push("--package".to_string());
            args.push(package.clone());
        }
    }
    if let Some(target) = &preset.target {
        target.append_args(&mut args);
    }
    if let Some(profile) = &preset.profile {
        args.push("--profile".to_string());
        args.push(profile.clone());
    }
    if preset.default_features == Some(false) {
        args.push("--no-default-features".to_string());
    }
    let mut features = preset.features.clone();
    features.sort();
    features.dedup();
    if !features.is_empty() {
        args.push("--features".to_string());
        args.push(features.join(","));
    }
    if let Some(target_triple) = &preset.target_triple {
        args.push("--target".to_string());
        args.push(target_triple.clone());
    }
    args.extend(preset.args.iter().cloned());
    if !preset.trailing_args.is_empty() {
        args.push("--".to_string());
        args.extend(preset.trailing_args.iter().cloned());
    }
    let cwd = match &preset.working_directory {
        CargoWorkingDirectoryPolicy::Context => None,
        CargoWorkingDirectoryPolicy::Workspace => context.workspace_cwd.clone(),
        CargoWorkingDirectoryPolicy::Package => context
            .package_cwd
            .clone()
            .or_else(|| context.workspace_cwd.clone()),
        CargoWorkingDirectoryPolicy::Custom(path) => Some(path.clone()),
    };
    let label = if preset.label.trim().is_empty() {
        format!("Cargo {}", subcommand.as_str())
    } else if subcommand == preset.subcommand {
        preset.label.clone()
    } else {
        format!("Cargo {} ({})", subcommand.as_str(), preset.label)
    };
    Ok(CompiledCargoPreset {
        task_template: TaskTemplate {
            experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
            label,
            command: "cargo".to_string(),
            args,
            env: preset.environment.clone(),
            cwd,
            use_new_terminal: preset.presentation.use_new_terminal,
            allow_concurrent_runs: preset.presentation.allow_concurrent_runs,
            reveal: preset.presentation.reveal,
            reveal_target: preset.presentation.reveal_target,
            hide: preset.presentation.hide,
            tags: vec![
                "cargo-preset".to_string(),
                format!("cargo-{}", subcommand.as_str()),
            ],
            save: preset.presentation.save,
            ..TaskTemplate::default()
        },
        task_context: CargoTaskContextInputs {
            scope: preset.scope,
            workspace_cwd: context.workspace_cwd.clone(),
            package_cwd: context.package_cwd.clone(),
        },
        pre_launch_task: preset.pre_launch_task.clone(),
    })
}

pub fn compile_debug_scenario(
    compiled: &CompiledCargoPreset,
    adapter: Option<&str>,
) -> Result<DebugScenario> {
    let mut build_template = compiled.task_template.clone();
    let subcommand_index = usize::from(
        build_template
            .args
            .first()
            .is_some_and(|argument| argument.starts_with('+')),
    );
    let action = build_template
        .args
        .get_mut(subcommand_index)
        .ok_or_else(|| anyhow!("Cargo task has no subcommand"))?;
    match action.as_str() {
        "run" => *action = "build".to_string(),
        "test" | "bench" => {
            let delimiter = build_template
                .args
                .iter()
                .position(|argument| argument == "--")
                .unwrap_or(build_template.args.len());
            if !build_template.args[..delimiter]
                .iter()
                .any(|argument| argument == "--no-run")
            {
                build_template
                    .args
                    .insert(delimiter, "--no-run".to_string());
            }
        }
        "build" => {}
        unsupported => bail!("Cargo `{unsupported}` cannot produce a debug scenario"),
    }
    let adapter = adapter.unwrap_or("CodeLLDB");
    Ok(DebugScenario {
        adapter: adapter.to_string().into(),
        label: format!("Debug {}", compiled.task_template.label).into(),
        build: Some(BuildTaskDefinition::Template {
            task_template: build_template,
            locator_name: Some(CARGO_LOCATOR_NAME.into()),
        }),
        config: if adapter == "CodeLLDB" {
            serde_json::json!({ "sourceLanguages": ["rust"] })
        } else {
            serde_json::Value::Null
        },
        tcp_connection: None,
    })
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CargoSafeSelectionState {
    pub scope: Option<CargoPresetScope>,
    pub package: Option<String>,
    pub target: Option<CargoTargetSelector>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CargoPresetWorkspaceState {
    #[serde(default = "workspace_state_version")]
    pub schema_version: u32,
    #[serde(default, alias = "active_preset")]
    pub active_preset_id: Option<String>,
    #[serde(default)]
    pub selection: CargoSafeSelectionState,
}

impl Default for CargoPresetWorkspaceState {
    fn default() -> Self {
        Self {
            schema_version: CARGO_PRESET_WORKSPACE_STATE_VERSION,
            active_preset_id: None,
            selection: CargoSafeSelectionState::default(),
        }
    }
}

fn workspace_state_version() -> u32 {
    CARGO_PRESET_WORKSPACE_STATE_VERSION
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RecoveredCargoPresetWorkspaceState {
    pub state: CargoPresetWorkspaceState,
    pub notice: Option<String>,
    pub rewrite: bool,
}

pub fn recover_workspace_state(
    serialized: Option<&str>,
    available_presets: &BTreeMap<String, CargoPreset>,
) -> RecoveredCargoPresetWorkspaceState {
    let Some(serialized) = serialized else {
        return RecoveredCargoPresetWorkspaceState::default();
    };
    let mut state = match serde_json::from_str::<CargoPresetWorkspaceState>(serialized) {
        Ok(state) if state.schema_version == CARGO_PRESET_WORKSPACE_STATE_VERSION => state,
        Ok(state) => {
            return RecoveredCargoPresetWorkspaceState {
                state: CargoPresetWorkspaceState::default(),
                notice: Some(format!(
                    "Cargo preset state version {} is unsupported; using Cargo defaults",
                    state.schema_version
                )),
                rewrite: true,
            };
        }
        Err(error) => {
            return RecoveredCargoPresetWorkspaceState {
                state: CargoPresetWorkspaceState::default(),
                notice: Some(bounded_message(format!(
                    "Cargo preset state could not be restored; using Cargo defaults: {error}"
                ))),
                rewrite: true,
            };
        }
    };
    let mut notice = None;
    let mut rewrite = false;
    if let Some(active_id) = state.active_preset_id.as_ref()
        && !available_presets.contains_key(active_id)
    {
        notice = Some(format!(
            "Cargo preset `{}` is unavailable; using Cargo defaults",
            bounded_text(active_id)
        ));
        state.active_preset_id = None;
        rewrite = true;
    }
    if state
        .selection
        .package
        .as_ref()
        .is_some_and(|package| package.len() > MAX_PRESET_TEXT_BYTES)
    {
        state.selection.package = None;
        notice = Some("Cargo package selection was invalid; using the selected scope".to_string());
        rewrite = true;
    }
    RecoveredCargoPresetWorkspaceState {
        state,
        notice,
        rewrite,
    }
}

pub fn persistence_summary(
    state: &CargoPresetWorkspaceState,
    presets: &BTreeMap<String, CargoPreset>,
) -> String {
    let active = state
        .active_preset_id
        .as_deref()
        .unwrap_or("Cargo defaults");
    let environment_keys = state
        .active_preset_id
        .as_ref()
        .and_then(|id| presets.get(id))
        .map(CargoPreset::environment_keys)
        .unwrap_or_default();
    format!(
        "active={active}; environment_keys={}",
        environment_keys.join(",")
    )
}

#[cfg(test)]
mod tests {
    use settings::{CargoPresetSettingsContent, CargoSettingsContent};

    use super::*;

    fn content(subcommand: &str) -> CargoPresetSettingsContent {
        CargoPresetSettingsContent {
            label: Some("Adversarial preset".to_string()),
            subcommand: Some(subcommand.to_string()),
            scope: Some("package".to_string()),
            package: Some("pkg with spaces".to_string()),
            target_kind: Some("bin".to_string()),
            target_name: Some("bin;echo-not-shell".to_string()),
            profile: Some("ship".to_string()),
            profiler: None,
            native_profiler: None,
            features: Some(vec!["z".to_string(), "a feature".to_string()]),
            default_features: Some(false),
            target_triple: Some("wasm32-unknown-unknown".to_string()),
            toolchain: None,
            pre_launch_task: None,
            args: Some(vec!["--config".to_string(), "x='$(nope)'".to_string()]),
            trailing_args: Some(vec!["argument with spaces".to_string()]),
            environment: Some(HashMap::from_iter([
                ("TOKEN".to_string(), "secret value".to_string()),
                ("RUSTFLAGS".to_string(), "-C target-cpu=native".to_string()),
            ])),
            working_directory: Some("package".to_string()),
            custom_working_directory: None,
            reveal: Some("no_focus".to_string()),
            reveal_target: Some("dock".to_string()),
            hide: Some("on_success".to_string()),
            save: Some("all".to_string()),
            use_new_terminal: Some(true),
            allow_concurrent_runs: Some(true),
        }
    }

    #[test]
    fn cargo_preset_project_precedence_and_invalid_entry_isolation() {
        let user = CargoSettingsContent {
            schema_version: Some(1),
            presets: HashMap::from_iter([
                ("shared".to_string(), content("build")),
                ("user-only".to_string(), content("check")),
            ]),
        };
        let mut project_shared = CargoPresetSettingsContent::default();
        project_shared.subcommand = Some("test".to_string());
        project_shared.profile = Some("project-profile".to_string());
        let mut invalid = CargoPresetSettingsContent::default();
        invalid.subcommand = Some("fetch-and-install".to_string());
        let project = CargoSettingsContent {
            schema_version: Some(1),
            presets: HashMap::from_iter([
                ("shared".to_string(), project_shared),
                ("invalid".to_string(), invalid),
            ]),
        };
        let parsed = parse_settings_content(&merge_preset_content(&user, &project));
        assert_eq!(parsed.presets.len(), 2);
        assert_eq!(parsed.presets["shared"].subcommand, CargoSubcommand::Test);
        assert_eq!(
            parsed.presets["shared"].profile.as_deref(),
            Some("project-profile")
        );
        assert_eq!(parsed.diagnostics.len(), 1);
        assert_eq!(parsed.diagnostics[0].preset_id.as_deref(), Some("invalid"));
    }

    #[test]
    fn cargo_preset_compiler_preserves_argv_env_and_dap_shape() {
        let preset =
            parse_preset("adversarial", &content("run")).expect("fixture preset should validate");
        let compiled = compile_preset(
            &preset,
            &CargoCompileContext {
                workspace_name: Some("workspace".to_string()),
                workspace_cwd: Some("/workspace".to_string()),
                package_name: Some("selected".to_string()),
                package_cwd: Some("/workspace/pkg".to_string()),
            },
            None,
        )
        .expect("fixture should compile");
        assert_eq!(compiled.task_template.command, "cargo");
        assert_eq!(
            compiled.task_template.args,
            vec![
                "run",
                "--package",
                "pkg with spaces",
                "--bin",
                "bin;echo-not-shell",
                "--profile",
                "ship",
                "--no-default-features",
                "--features",
                "a feature,z",
                "--target",
                "wasm32-unknown-unknown",
                "--config",
                "x='$(nope)'",
                "--",
                "argument with spaces",
            ]
        );
        assert_eq!(
            compiled.task_template.env["TOKEN"],
            "secret value".to_string()
        );
        assert_eq!(
            compiled.task_template.cwd.as_deref(),
            Some("/workspace/pkg")
        );
        let scenario = compile_debug_scenario(&compiled, None)
            .expect("run preset should produce a debug scenario");
        let Some(BuildTaskDefinition::Template {
            task_template,
            locator_name,
        }) = scenario.build
        else {
            panic!("debug scenario should contain a task template")
        };
        assert_eq!(
            task_template.args.first().map(String::as_str),
            Some("build")
        );
        assert_eq!(locator_name.as_deref(), Some(CARGO_LOCATOR_NAME));
    }

    #[test]
    fn cargo_preset_v2_migrates_v1_and_compiles_toolchain_and_pre_launch_reference() {
        let version_one = CargoSettingsContent {
            schema_version: Some(1),
            presets: HashMap::from_iter([("legacy".to_string(), content("check"))]),
        };
        let migrated = parse_settings_content(&version_one);
        assert!(migrated.diagnostics.is_empty());
        assert!(migrated.presets["legacy"].toolchain.is_none());
        assert!(migrated.presets["legacy"].pre_launch_task.is_none());

        let mut version_two_content = content("run");
        version_two_content.toolchain = Some("nightly-2026-08-01".to_string());
        version_two_content.pre_launch_task = Some("Generate bindings".to_string());
        let version_two = CargoSettingsContent {
            schema_version: Some(2),
            presets: HashMap::from_iter([("v2".to_string(), version_two_content)]),
        };
        let parsed = parse_settings_content(&version_two);
        let compiled = compile_preset(
            &parsed.presets["v2"],
            &CargoCompileContext {
                workspace_name: Some("workspace".to_string()),
                workspace_cwd: Some("/workspace".to_string()),
                package_name: Some("package".to_string()),
                package_cwd: Some("/workspace/package".to_string()),
            },
            None,
        )
        .expect("version-two preset should compile");
        assert_eq!(
            compiled.task_template.args.first().map(String::as_str),
            Some("+nightly-2026-08-01")
        );
        assert_eq!(
            compiled.task_template.args.get(1).map(String::as_str),
            Some("run")
        );
        assert_eq!(
            compiled.pre_launch_task.as_deref(),
            Some("Generate bindings")
        );

        let scenario = compile_debug_scenario(&compiled, None)
            .expect("toolchain preset should compile to a debug scenario");
        let Some(BuildTaskDefinition::Template { task_template, .. }) = scenario.build else {
            panic!("debug scenario should contain a task template")
        };
        assert_eq!(task_template.args[0], "+nightly-2026-08-01");
        assert_eq!(task_template.args[1], "build");
    }

    #[test]
    fn cargo_preset_workspace_state_falls_back_without_serializing_values() {
        let preset =
            parse_preset("adversarial", &content("test")).expect("fixture preset should validate");
        let presets = BTreeMap::from_iter([(preset.id.clone(), preset)]);
        let state = CargoPresetWorkspaceState {
            schema_version: 1,
            active_preset_id: Some("adversarial".to_string()),
            selection: CargoSafeSelectionState {
                scope: Some(CargoPresetScope::Package),
                package: Some("member".to_string()),
                target: Some(CargoTargetSelector::Test("case".to_string())),
            },
        };
        let serialized = serde_json::to_string(&state).expect("state should serialize");
        assert!(!serialized.contains("secret value"));
        assert!(!persistence_summary(&state, &presets).contains("secret value"));
        assert!(persistence_summary(&state, &presets).contains("RUSTFLAGS,TOKEN"));

        let recovered = recover_workspace_state(Some(&serialized), &BTreeMap::new());
        assert!(recovered.state.active_preset_id.is_none());
        assert!(recovered.rewrite);
        assert!(
            recovered
                .notice
                .as_deref()
                .is_some_and(|notice| notice.contains("unavailable"))
        );
    }

    #[test]
    fn cargo_preset_native_profiler_is_explicit_and_exclusive() {
        let native = settings::CargoNativeProfilerSettingsContent {
            collector_path: Some("/opt/tools/samply".into()),
            profile_path: Some("target/native-capture.json".into()),
        };
        let mut declaration = CargoPresetSettingsContent {
            subcommand: Some("run".into()),
            native_profiler: Some(native.clone()),
            ..Default::default()
        };
        let parsed = parse_preset("native", &declaration).expect("native preset");
        assert!(parsed.profiler.is_none());
        assert_eq!(
            parsed
                .native_profiler
                .expect("native declaration")
                .settings_content(),
            native
        );
        declaration.profiler = Some(settings::CargoProfilerSettingsContent {
            command: Some("external-profiler".into()),
            args: Some(Vec::new()),
            artifact_path: Some("target/profile.svg".into()),
            artifact_kind: Some("svg".into()),
            artifact_max_bytes: Some(1024),
        });
        assert!(parse_preset("conflicting", &declaration).is_err());
        assert!(
            CargoPreset::ephemeral_default(CargoSubcommand::Run)
                .native_profiler
                .is_none()
        );
    }
}
