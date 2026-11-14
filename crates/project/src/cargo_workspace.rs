use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use cargo_metadata::{
    Dependency, DependencyKind, Message, Metadata, Node, Package, PackageId, Target, TargetKind,
};
use sha2::{Digest as _, Sha256};

use crate::ProjectPath;

pub const CARGO_METADATA_FORMAT_VERSION: u64 = 1;
pub const MAX_CARGO_CONFIGURATION_DIAGNOSTICS: usize = 32;
pub const MAX_CARGO_CONFIGURATION_ITEMS: usize = 128;
pub const MAX_CARGO_CONFIGURATION_FIELD_BYTES: usize = 512;
pub const MAX_RUST_TOOLCHAIN_FILE_BYTES: usize = 64 * 1024;
pub const MAX_RUSTC_VERBOSE_VERSION_BYTES: usize = 64 * 1024;
pub const MAX_CARGO_DEPENDENCY_INSTANCES: usize = 32;
pub const MAX_CARGO_DEPENDENCY_FEATURES: usize = 256;
pub const MAX_CARGO_METADATA_BYTES: usize = 10 * 1024 * 1024;
pub const MAX_CARGO_GRAPH_NODES: usize = 4096;
pub const MAX_CARGO_GRAPH_EDGES: usize = 32768;
pub const MAX_CARGO_SNAPSHOT_BYTES: usize = 8 * 1024 * 1024;
const MAX_CARGO_RETAINED_ITEMS: usize = 65536;
const CARGO_GRAPH_FORMAT_VERSION: u32 = 1;
pub(crate) const MAX_CARGO_PASSIVE_CONFIGURATION_BYTES: usize = 64 * 1024;

pub const MAX_CARGO_BUILD_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_CARGO_BUILD_MESSAGE_LINES: usize = 50_000;
pub const MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES: usize = 1024;
pub const MAX_CARGO_BUILD_MESSAGE_FIELDS: usize = 65_536;
pub const MAX_CARGO_BUILD_ARTIFACTS: usize = 2000;

pub struct CargoMessageDecoder<'a> {
    lines: std::iter::Take<std::iter::Enumerate<std::str::Lines<'a>>>,
    byte_limit_reached: bool,
    line_limit_reached: bool,
    field_limits: Option<(usize, usize)>,
}

impl<'a> CargoMessageDecoder<'a> {
    pub fn new(input: &'a str, byte_limit: usize, line_limit: usize) -> Self {
        let mut end = input.len().min(byte_limit);
        while !input.is_char_boundary(end) {
            end -= 1;
        }
        let bounded = &input[..end];
        Self {
            lines: bounded.lines().enumerate().take(line_limit),
            byte_limit_reached: end < input.len(),
            line_limit_reached: bounded.lines().count() > line_limit,
            field_limits: None,
        }
    }

    pub fn require_bounded_fields(mut self, field_bytes: usize, fields: usize) -> Self {
        self.field_limits = Some((field_bytes, fields));
        self
    }

    pub fn byte_limit_reached(&self) -> bool {
        self.byte_limit_reached
    }

    pub fn line_limit_reached(&self) -> bool {
        self.line_limit_reached
    }
}

impl Iterator for CargoMessageDecoder<'_> {
    type Item = std::result::Result<Message, String>;

    fn next(&mut self) -> Option<Self::Item> {
        for (line_index, line) in self.lines.by_ref() {
            if line.trim().is_empty() {
                continue;
            }
            let value = match serde_json::from_str::<serde_json::Value>(line) {
                Ok(value) => value,
                Err(error) => {
                    return Some(Err(format!(
                        "Cargo JSON line {} is malformed: {error}",
                        line_index + 1
                    )));
                }
            };
            let reason = value.get("reason").and_then(serde_json::Value::as_str);
            if !matches!(
                reason,
                Some(
                    "compiler-artifact"
                        | "compiler-message"
                        | "build-script-executed"
                        | "build-finished"
                )
            ) {
                return Some(Err(format!(
                    "Cargo JSON line {} has an unknown record kind",
                    line_index + 1
                )));
            }
            if let Some((field_bytes, fields)) = self.field_limits {
                let mut remaining = fields;
                if !cargo_message_fields_are_bounded(&value, field_bytes, &mut remaining) {
                    return Some(Err(format!(
                        "Cargo JSON line {} exceeds field bounds",
                        line_index + 1
                    )));
                }
            }
            return Some(serde_json::from_value::<Message>(value).map_err(|error| {
                format!("Cargo JSON line {} is incomplete: {error}", line_index + 1)
            }));
        }
        None
    }
}

fn cargo_message_fields_are_bounded(
    value: &serde_json::Value,
    field_bytes: usize,
    remaining: &mut usize,
) -> bool {
    let Some(next) = remaining.checked_sub(1) else {
        return false;
    };
    *remaining = next;
    match value {
        serde_json::Value::String(value) => value.len() <= field_bytes,
        serde_json::Value::Array(values) => values
            .iter()
            .all(|value| cargo_message_fields_are_bounded(value, field_bytes, remaining)),
        serde_json::Value::Object(values) => values.iter().all(|(key, value)| {
            key.len() <= field_bytes
                && cargo_message_fields_are_bounded(value, field_bytes, remaining)
        }),
        _ => true,
    }
}

pub const MAX_CARGO_NATIVE_TARGET_POLICY_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoNativeExecutionPolicy {
    Program,
    Test { harness: bool },
    Bench { harness: bool },
}

impl CargoNativeExecutionPolicy {
    fn expected_test_mode(self, kind: &CargoTargetKind) -> Result<bool> {
        match (self, kind) {
            (Self::Program, CargoTargetKind::Binary | CargoTargetKind::Example) => Ok(false),
            (Self::Test { .. }, CargoTargetKind::Test)
            | (Self::Bench { .. }, CargoTargetKind::Bench) => Ok(true),
            _ => bail!("Native Cargo execution policy does not match its selected target kind"),
        }
    }
}

/// Returns bounded manifest data, not authority to execute a selected workload.
/// Cargo's artifact test bit records the compile mode even for harness=false.
#[cfg(test)]
pub(crate) fn parse_native_cargo_execution_policy(
    manifest: &str,
    manifest_path: &Path,
    target_kind: &CargoTargetKind,
    target_name: &str,
    source_path: &Path,
    edition: &str,
) -> Result<CargoNativeExecutionPolicy> {
    parse_native_cargo_execution_policy_with_workspace_manifest(
        manifest,
        manifest_path,
        target_kind,
        target_name,
        source_path,
        edition,
        None,
    )
}

pub(crate) fn parse_native_cargo_execution_policy_with_workspace_manifest(
    manifest: &str,
    manifest_path: &Path,
    target_kind: &CargoTargetKind,
    target_name: &str,
    source_path: &Path,
    edition: &str,
    workspace_manifest: Option<&str>,
) -> Result<CargoNativeExecutionPolicy> {
    let (section, directory, auto_field) = match target_kind {
        CargoTargetKind::Binary | CargoTargetKind::Example => {
            return Ok(CargoNativeExecutionPolicy::Program);
        }
        CargoTargetKind::Test => ("test", "tests", "autotests"),
        CargoTargetKind::Bench => ("bench", "benches", "autobenches"),
        _ => bail!("Native Cargo execution policy requires an executable target"),
    };
    ensure!(
        manifest.len() <= MAX_CARGO_NATIVE_TARGET_POLICY_BYTES,
        "Native Cargo target manifest exceeds its byte limit"
    );
    ensure!(
        !target_name.is_empty()
            && target_name.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
            && !target_name
                .chars()
                .any(|value| value.is_control() || matches!(value, '/' | '\\' | ':')),
        "Native Cargo target name is invalid"
    );
    validate_native_cargo_path(manifest_path)?;
    validate_native_cargo_path(source_path)?;
    let root = manifest_path
        .parent()
        .context("Native Cargo package directory is unavailable")?;
    let value = toml::from_str::<toml::Value>(manifest)
        .map_err(|_| anyhow::anyhow!("Native Cargo target manifest is invalid"))?;
    let package = value
        .get("package")
        .and_then(toml::Value::as_table)
        .context("Native Cargo target package declaration is unavailable")?;
    let package_edition = match package.get("edition") {
        Some(toml::Value::String(edition)) => edition.clone(),
        Some(toml::Value::Table(declaration)) => {
            ensure!(
                declaration.len() == 1
                    && declaration.get("workspace").and_then(toml::Value::as_bool) == Some(true),
                "Native Cargo inherited package edition declaration is invalid"
            );
            let workspace_manifest = workspace_manifest.context(
                "Native Cargo inherited package edition requires a reviewed workspace declaration",
            )?;
            ensure!(
                workspace_manifest.len() <= MAX_CARGO_NATIVE_TARGET_POLICY_BYTES,
                "Native Cargo workspace manifest exceeds its byte limit"
            );
            let workspace = toml::from_str::<toml::Value>(workspace_manifest).map_err(|_| {
                anyhow::anyhow!("Native Cargo workspace edition declaration is invalid")
            })?;
            workspace
                .get("workspace")
                .and_then(|value| value.get("package"))
                .and_then(|value| value.get("edition"))
                .and_then(toml::Value::as_str)
                .context("Native Cargo workspace package edition is unavailable")?
                .to_owned()
        }
        Some(_) => bail!("Native Cargo package edition declaration is invalid"),
        None => "2015".to_owned(),
    };
    let declarations = match value.get(section) {
        Some(value) => value
            .as_array()
            .context("Native Cargo target declarations are invalid")?
            .as_slice(),
        None => &[],
    };
    ensure!(
        declarations.len() <= MAX_CARGO_CONFIGURATION_ITEMS,
        "Native Cargo target declaration count exceeds its limit"
    );
    let conventional_sources = [
        root.join(directory).join(format!("{target_name}.rs")),
        root.join(directory).join(target_name).join("main.rs"),
    ];
    let mut selected_harness = None;
    let mut selected_edition = None;
    for declaration in declarations {
        let declaration = declaration
            .as_table()
            .context("Native Cargo target declaration is invalid")?;
        let name = declaration
            .get("name")
            .and_then(toml::Value::as_str)
            .context("Native Cargo target declaration has no supported explicit name")?;
        ensure!(
            !name.is_empty()
                && name.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                && !name.chars().any(char::is_control),
            "Native Cargo target declaration name is invalid"
        );
        let harness = match declaration.get("harness") {
            Some(value) => value
                .as_bool()
                .context("Native Cargo target harness declaration is invalid")?,
            None => true,
        };
        if name != target_name {
            continue;
        }
        ensure!(
            selected_harness.is_none(),
            "Native Cargo selected target declaration is ambiguous"
        );
        let declared_source = match declaration.get("path") {
            Some(value) => {
                let path = value
                    .as_str()
                    .context("Native Cargo target source declaration is invalid")?;
                ensure!(
                    !path.is_empty()
                        && path.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                        && !path.chars().any(char::is_control),
                    "Native Cargo target source declaration exceeds its bound"
                );
                let path = normalize_native_manifest_target_path(root, Path::new(path))?;
                ensure!(
                    &path == source_path,
                    "Native Cargo target source declaration does not match metadata"
                );
                true
            }
            None => conventional_sources.iter().any(|path| path == source_path),
        };
        ensure!(
            declared_source,
            "Native Cargo target has no supported inferred source declaration"
        );
        selected_edition = match declaration.get("edition") {
            Some(value) => Some(
                value
                    .as_str()
                    .context("Native Cargo target edition declaration is invalid")?,
            ),
            None => None,
        };
        selected_harness = Some(harness);
    }
    let effective_edition = selected_edition.unwrap_or(package_edition.as_str());
    ensure!(
        matches!(effective_edition, "2015" | "2018" | "2021" | "2024")
            && effective_edition == edition,
        "Native Cargo target edition does not match its reviewed manifest"
    );
    let harness = if let Some(harness) = selected_harness {
        harness
    } else {
        ensure!(
            matches!(edition, "2015" | "2018" | "2021" | "2024"),
            "Native Cargo target edition is unsupported"
        );
        let auto = match package.get(auto_field) {
            Some(value) => value
                .as_bool()
                .context("Native Cargo target autodiscovery declaration is invalid")?,
            None => {
                edition != "2015"
                    || !["lib", "bin", "example", "test", "bench"]
                        .iter()
                        .any(|name| value.get(*name).is_some())
            }
        };
        ensure!(
            auto,
            "Native Cargo selected target is not enabled by its manifest"
        );
        ensure!(
            conventional_sources.iter().any(|path| path == source_path),
            "Native Cargo selected target has no supported autodiscovered source"
        );
        true
    };
    Ok(match target_kind {
        CargoTargetKind::Test => CargoNativeExecutionPolicy::Test { harness },
        CargoTargetKind::Bench => CargoNativeExecutionPolicy::Bench { harness },
        _ => bail!("Native Cargo target policy changed during resolution"),
    })
}

fn normalize_native_manifest_target_path(root: &Path, path: &Path) -> Result<PathBuf> {
    let candidate = root.join(path);
    let mut normalized = PathBuf::new();
    for component in candidate.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                ensure!(
                    normalized.pop(),
                    "Native Cargo target source escapes its filesystem root"
                );
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    validate_native_cargo_path(&normalized)?;
    Ok(normalized)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoExecutableSelection {
    pub package_id: String,
    pub target_kind: CargoTargetKind,
    pub target_name: String,
    pub source_path: PathBuf,
    pub execution_policy: CargoNativeExecutionPolicy,
}

/// Returns parsed executable data only. The caller must independently bind producer
/// completion, source freshness and host/filesystem authority before capture.
pub fn resolve_native_cargo_executable(
    input: &[u8],
    selection: &CargoExecutableSelection,
    producer_succeeded: bool,
) -> Result<PathBuf> {
    ensure!(
        matches!(
            selection.target_kind,
            CargoTargetKind::Binary | CargoTargetKind::Example
        ),
        "Native capture executable selection currently requires a binary or example"
    );
    Ok(resolve_native_cargo_build_output(input, selection, producer_succeeded, None)?.executable)
}

pub const MAX_CARGO_NATIVE_RUNTIME_PATHS: usize = 4096;
const MAX_CARGO_NATIVE_RUNTIME_PATH_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoNativeRuntimeLibraryRecord {
    package_id: String,
    linked_paths: Vec<PathBuf>,
    output_directory: Option<PathBuf>,
}

impl CargoNativeRuntimeLibraryRecord {
    pub fn package_id(&self) -> &str {
        &self.package_id
    }
    pub fn linked_paths(&self) -> &[PathBuf] {
        &self.linked_paths
    }
    pub fn output_directory(&self) -> Option<&Path> {
        self.output_directory.as_deref()
    }
}

pub(crate) struct CargoNativeBuildOutput {
    pub(crate) executable: PathBuf,
    #[cfg(any(target_os = "macos", test))]
    pub(crate) runtime_libraries: Option<Vec<CargoNativeRuntimeLibraryRecord>>,
}

pub(crate) fn native_cargo_build_profile(arguments: &[String]) -> Result<Option<&str>> {
    let suffix = arguments
        .get(8..)
        .context("Native Build recipe is incomplete")?;
    let profile = match suffix {
        [] => None,
        [flag, _] if flag == "--features" => None,
        [flag, profile] if flag == "--profile" => Some(profile.as_str()),
        [features, _, flag, profile] if features == "--features" && flag == "--profile" => {
            Some(profile.as_str())
        }
        _ => bail!("Native Build option sequence is not canonical"),
    };
    if let Some(profile) = profile {
        ensure!(
            !profile.is_empty()
                && !profile.starts_with('-')
                && profile.len() <= MAX_CARGO_CONFIGURATION_FIELD_BYTES
                && profile
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "Native Build profile name is invalid"
        );
    }
    Ok(profile)
}

pub(crate) fn native_cargo_runtime_logical_arguments(
    selected_build: &[String],
    workload_arguments: &[String],
) -> Result<Vec<String>> {
    ensure!(
        matches!(selected_build.len(), 8 | 10 | 12)
            && selected_build
                .first()
                .is_some_and(|argument| argument == "build")
            && selected_build
                .get(1)
                .is_some_and(|argument| argument == "--message-format=json")
            && selected_build
                .get(2)
                .is_some_and(|argument| argument == "--package")
            && selected_build
                .get(4)
                .is_some_and(|argument| matches!(argument.as_str(), "--bin" | "--example"))
            && selected_build
                .get(6)
                .is_some_and(|argument| argument == "--target-dir")
            && native_cargo_build_profile(selected_build).is_ok(),
        "Native runtime preparation requires an exact selected Cargo Build recipe"
    );
    ensure!(
        selected_build.iter().all(|argument| !argument.is_empty()
            && argument.len() <= 4096
            && !argument.chars().any(char::is_control)),
        "Native runtime selection operand exceeds its bound"
    );
    validate_native_cargo_path(Path::new(
        selected_build
            .get(7)
            .context("Native runtime output disappeared")?,
    ))?;
    ensure!(
        workload_arguments.len() <= 128,
        "Native runtime workload argument count exceeds its bound"
    );
    let mut bytes = 0_usize;
    for argument in workload_arguments {
        bytes = bytes
            .checked_add(argument.len())
            .context("Native runtime workload argument size overflow")?;
        ensure!(
            argument.len() <= 4096 && !argument.contains('\0') && bytes <= 32 * 1024,
            "Native runtime workload arguments exceed their bound"
        );
    }
    let mut arguments = vec!["run".into(), "--offline".into()];
    arguments.extend(selected_build.iter().skip(1).cloned());
    arguments.push("--".into());
    arguments.extend(workload_arguments.iter().cloned());
    Ok(arguments)
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn native_cargo_runtime_selected_build_arguments(
    runtime_arguments: &[String],
    workload_arguments: &[String],
) -> Result<Vec<String>> {
    let operands = workload_arguments
        .len()
        .checked_add(1)
        .context("Native runtime logical recipe count overflow")?;
    let delimiter = runtime_arguments
        .len()
        .checked_sub(operands)
        .context("Native runtime logical recipe is incomplete")?;
    ensure!(
        matches!(delimiter, 9 | 11 | 13)
            && runtime_arguments
                .first()
                .is_some_and(|argument| argument == "run")
            && runtime_arguments
                .get(1)
                .is_some_and(|argument| argument == "--offline")
            && runtime_arguments
                .get(delimiter)
                .is_some_and(|argument| argument == "--")
            && runtime_arguments.get(delimiter + 1..) == Some(workload_arguments),
        "Native runtime logical Task differs from its literal continuation"
    );
    let mut selected_build = vec!["build".into()];
    selected_build.extend(
        runtime_arguments
            .iter()
            .skip(2)
            .take(delimiter - 2)
            .cloned(),
    );
    ensure!(
        native_cargo_runtime_logical_arguments(&selected_build, workload_arguments)?
            == runtime_arguments,
        "Native runtime logical Task is not an exact selected recipe"
    );
    Ok(selected_build)
}

pub(crate) fn resolve_native_cargo_build_output(
    input: &[u8],
    selection: &CargoExecutableSelection,
    producer_succeeded: bool,
    runtime_packages: Option<&BTreeSet<String>>,
) -> Result<CargoNativeBuildOutput> {
    ensure!(producer_succeeded, "Cargo producer did not succeed");
    ensure!(
        input.len() <= MAX_CARGO_BUILD_MESSAGE_BYTES,
        "Cargo build JSON byte limit reached"
    );
    let input = std::str::from_utf8(input).context("Cargo build JSON is not UTF-8")?;
    let expected_kind = match selection.target_kind {
        CargoTargetKind::Binary => TargetKind::Bin,
        CargoTargetKind::Example => TargetKind::Example,
        CargoTargetKind::Test => TargetKind::Test,
        CargoTargetKind::Bench => TargetKind::Bench,
        _ => bail!("Native capture executable selection requires an exact executable target"),
    };
    let expected_test_mode = selection
        .execution_policy
        .expected_test_mode(&selection.target_kind)?;
    ensure!(
        [&selection.package_id, &selection.target_name]
            .into_iter()
            .all(|value| !value.is_empty()
                && value.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                && !value.chars().any(char::is_control)),
        "Cargo executable selection fields are invalid"
    );
    validate_native_cargo_path(&selection.source_path)?;
    let decoder = CargoMessageDecoder::new(
        input,
        MAX_CARGO_BUILD_MESSAGE_BYTES,
        MAX_CARGO_BUILD_MESSAGE_LINES,
    )
    .require_bounded_fields(
        MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES,
        MAX_CARGO_BUILD_MESSAGE_FIELDS,
    );
    ensure!(
        !decoder.byte_limit_reached() && !decoder.line_limit_reached(),
        "Cargo build JSON limit reached"
    );
    if let Some(packages) = runtime_packages {
        ensure!(
            packages.len() <= MAX_CARGO_GRAPH_NODES
                && packages.contains(&selection.package_id)
                && packages.iter().all(|package| !package.is_empty()
                    && package.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                    && !package.chars().any(char::is_control)),
            "Native Cargo runtime package closure is invalid"
        );
    }
    let mut runtime_libraries = runtime_packages.map(|_| Vec::new());
    let mut scripts = 0_usize;
    let mut runtime_paths = 0_usize;
    let mut runtime_bytes = 0_usize;
    let mut executable = None;
    let mut artifacts = 0;
    let mut finished = false;
    for message in decoder {
        let message = message.map_err(|error| anyhow::anyhow!(error))?;
        ensure!(
            !finished,
            "Cargo build JSON contains records after build-finished"
        );
        match message {
            Message::CompilerArtifact(artifact) => {
                artifacts += 1;
                ensure!(
                    artifacts <= MAX_CARGO_BUILD_ARTIFACTS,
                    "Cargo build artifact limit reached"
                );
                if artifact.package_id.repr != selection.package_id
                    || artifact.target.name != selection.target_name
                    || artifact.target.kind.as_slice() != std::slice::from_ref(&expected_kind)
                    || artifact.target.src_path.as_std_path() != selection.source_path
                {
                    continue;
                }
                ensure!(
                    artifact.profile.test == expected_test_mode,
                    "Selected Cargo artifact compile mode does not match its execution policy"
                );
                let path = artifact
                    .executable
                    .context("Selected Cargo artifact has no executable")?
                    .into_std_path_buf();
                validate_native_cargo_path(&path)?;
                ensure!(
                    executable.is_none(),
                    "Cargo build JSON has ambiguous selected executables"
                );
                executable = Some(path);
            }
            Message::BuildFinished(result) => {
                ensure!(result.success, "Cargo reported an unsuccessful build");
                finished = true;
            }
            Message::BuildScriptExecuted(script) => {
                if let Some(packages) = runtime_packages {
                    scripts += 1;
                    ensure!(
                        scripts <= MAX_CARGO_BUILD_ARTIFACTS,
                        "Cargo runtime build-script record limit reached"
                    );
                    if !packages.contains(&script.package_id.repr) {
                        continue;
                    }
                    ensure!(
                        script.linked_paths.len() <= MAX_CARGO_CONFIGURATION_ITEMS,
                        "Cargo runtime linked-path record limit reached"
                    );
                    let output_directory = script.out_dir.into_std_path_buf();
                    let mut linked_paths = Vec::new();
                    for path in script.linked_paths {
                        let path = path.into_std_path_buf();
                        let text = path.to_str().context("Cargo runtime path is not UTF-8")?;
                        ensure!(
                            !text.is_empty()
                                && text.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                                && !text.chars().any(char::is_control),
                            "Cargo runtime path exceeds its bounds"
                        );
                        runtime_paths = runtime_paths
                            .checked_add(1)
                            .context("Cargo runtime path count overflow")?;
                        runtime_bytes = runtime_bytes
                            .checked_add(text.len())
                            .context("Cargo runtime path byte count overflow")?;
                        ensure!(
                            runtime_paths <= MAX_CARGO_NATIVE_RUNTIME_PATHS
                                && runtime_bytes <= MAX_CARGO_NATIVE_RUNTIME_PATH_BYTES,
                            "Cargo runtime retained-path limit reached"
                        );
                        linked_paths.push(path);
                    }
                    let output_text = output_directory
                        .to_str()
                        .context("Cargo runtime output is not UTF-8")?;
                    ensure!(
                        output_text.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                            && !output_text.chars().any(char::is_control),
                        "Cargo runtime output exceeds its bounds"
                    );
                    runtime_bytes = runtime_bytes
                        .checked_add(output_text.len())
                        .context("Cargo runtime path byte count overflow")?;
                    ensure!(
                        runtime_bytes <= MAX_CARGO_NATIVE_RUNTIME_PATH_BYTES,
                        "Cargo runtime retained-path limit reached"
                    );
                    runtime_libraries
                        .as_mut()
                        .context("Cargo runtime retention owner disappeared")?
                        .push(CargoNativeRuntimeLibraryRecord {
                            package_id: script.package_id.repr,
                            linked_paths,
                            output_directory: (!output_text.is_empty()).then_some(output_directory),
                        });
                }
            }
            Message::CompilerMessage(_) => {}
            _ => bail!("Cargo emitted an unsupported build message"),
        }
    }
    ensure!(
        finished,
        "Cargo build JSON has no successful build-finished record"
    );
    Ok(CargoNativeBuildOutput {
        executable: executable.context("Cargo build JSON has no exact selected executable")?,
        #[cfg(any(target_os = "macos", test))]
        runtime_libraries,
    })
}

pub(crate) fn validate_native_cargo_path(path: &Path) -> Result<()> {
    let path_text = path
        .to_str()
        .context("Cargo executable path is not UTF-8")?;
    ensure!(
        path.is_absolute()
            && path_text.len() <= MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
            && !path_text.chars().any(char::is_control)
            && !path.components().any(|component| matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )),
        "Cargo executable path is not a bounded absolute path"
    );
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CargoConfigurationString {
    pub value: String,
    pub base_directory: PathBuf,
}

impl CargoConfigurationString {
    pub fn executable(&self) -> PathBuf {
        let path = Path::new(&self.value);
        if path.is_absolute()
            || (!self.value.contains('/') && !(cfg!(windows) && self.value.contains('\\')))
        {
            path.to_path_buf()
        } else {
            self.base_directory.join(path)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CargoConfigurationEnvironment {
    String(CargoConfigurationString),
    Table {
        value: Option<CargoConfigurationString>,
        force: Option<bool>,
        relative: Option<bool>,
    },
}

impl CargoConfigurationEnvironment {
    pub fn resolve(&self) -> Result<(std::ffi::OsString, bool)> {
        let (value, force, relative) = match self {
            Self::String(value) => (value, false, false),
            Self::Table {
                value,
                force,
                relative,
            } => (
                value
                    .as_ref()
                    .context("Cargo environment value is missing")?,
                force.unwrap_or(false),
                relative.unwrap_or(false),
            ),
        };
        let resolved = if relative {
            value.base_directory.join(&value.value).into_os_string()
        } else {
            value.value.clone().into()
        };
        Ok((resolved, force))
    }

    fn merge(&mut self, incoming: Self) -> Result<()> {
        match (self, incoming) {
            (Self::String(current), Self::String(incoming)) => *current = incoming,
            (
                Self::Table {
                    value,
                    force,
                    relative,
                },
                Self::Table {
                    value: incoming_value,
                    force: incoming_force,
                    relative: incoming_relative,
                },
            ) => {
                if incoming_value.is_some() {
                    *value = incoming_value;
                }
                if incoming_force.is_some() {
                    *force = incoming_force;
                }
                if incoming_relative.is_some() {
                    *relative = incoming_relative;
                }
            }
            _ => bail!("Cargo environment configuration has conflicting value types"),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CargoConfigurationInclude {
    pub path: PathBuf,
    pub optional: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CargoHostExecutionConfiguration {
    pub native_runtime_target_or_runner_configured: bool,
    pub compiler: Option<CargoConfigurationString>,
    pub documentation_generator: Option<CargoConfigurationString>,
    pub wrapper: Option<CargoConfigurationString>,
    pub workspace_wrapper: Option<CargoConfigurationString>,
    pub environment: BTreeMap<String, CargoConfigurationEnvironment>,
    pub includes: Vec<CargoConfigurationInclude>,
}

impl CargoHostExecutionConfiguration {
    pub fn merge_values(&mut self, incoming: Self) -> Result<()> {
        self.native_runtime_target_or_runner_configured |=
            incoming.native_runtime_target_or_runner_configured;
        if incoming.compiler.is_some() {
            self.compiler = incoming.compiler;
        }
        if incoming.documentation_generator.is_some() {
            self.documentation_generator = incoming.documentation_generator;
        }
        if incoming.wrapper.is_some() {
            self.wrapper = incoming.wrapper;
        }
        if incoming.workspace_wrapper.is_some() {
            self.workspace_wrapper = incoming.workspace_wrapper;
        }
        for (name, value) in incoming.environment {
            if let Some(current) = self.environment.get_mut(&name) {
                current.merge(value)?;
            } else {
                self.environment.insert(name, value);
            }
        }
        if self.environment.len() > MAX_CARGO_CONFIGURATION_ITEMS {
            bail!("Cargo environment configuration exceeds the supported entry limit");
        }
        // Includes are loaded before the containing file, not merged as ordinary arrays.
        Ok(())
    }
}

pub(crate) fn parse_cargo_host_execution_configuration(
    contents: &str,
    source_path: &Path,
) -> Result<CargoHostExecutionConfiguration> {
    if contents.len() > MAX_CARGO_PASSIVE_CONFIGURATION_BYTES {
        bail!("Cargo configuration exceeds the supported byte limit");
    }
    let document = toml::from_str::<toml::Value>(contents)
        .map_err(|_| anyhow::anyhow!("invalid Cargo configuration TOML contents"))?;
    let base_directory = source_path
        .parent()
        .and_then(Path::parent)
        .context("Cargo configuration source directory is unavailable")?;
    let string = |value: &toml::Value| -> Result<CargoConfigurationString> {
        Ok(CargoConfigurationString {
            value: value
                .as_str()
                .context("Cargo execution configuration value must be a string")?
                .to_string(),
            base_directory: base_directory.to_path_buf(),
        })
    };
    let mut configuration = CargoHostExecutionConfiguration::default();
    configuration.native_runtime_target_or_runner_configured = document.get("unstable").is_some()
        || document
            .get("build")
            .is_some_and(|build| build.get("target").is_some())
        || document.get("target").is_some_and(|targets| {
            targets.as_table().is_none_or(|targets| {
                targets.values().any(|target| {
                    target
                        .as_table()
                        .is_none_or(|target| target.contains_key("runner"))
                })
            })
        });
    if let Some(build) = document.get("build") {
        let build = build
            .as_table()
            .context("Cargo build configuration must be a table")?;
        configuration.compiler = build.get("rustc").map(&string).transpose()?;
        configuration.documentation_generator = build.get("rustdoc").map(&string).transpose()?;
        configuration.wrapper = build.get("rustc-wrapper").map(&string).transpose()?;
        configuration.workspace_wrapper = build
            .get("rustc-workspace-wrapper")
            .map(&string)
            .transpose()?;
    }
    if let Some(environment) = document.get("env") {
        let environment = environment
            .as_table()
            .context("Cargo environment configuration must be a table")?;
        if environment.len() > MAX_CARGO_CONFIGURATION_ITEMS {
            bail!("Cargo environment configuration exceeds the supported entry limit");
        }
        for (name, value) in environment {
            if name.is_empty()
                || name.contains(['\0', '='])
                || name.len() > MAX_CARGO_CONFIGURATION_FIELD_BYTES
            {
                bail!("Cargo environment configuration has an invalid variable name");
            }
            let value = if let Some(table) = value.as_table() {
                let flag = |name: &str| -> Result<Option<bool>> {
                    table
                        .get(name)
                        .map(|value| {
                            value
                                .as_bool()
                                .context("Cargo environment configuration flag must be boolean")
                        })
                        .transpose()
                };
                CargoConfigurationEnvironment::Table {
                    value: table.get("value").map(&string).transpose()?,
                    force: flag("force")?,
                    relative: flag("relative")?,
                }
            } else {
                CargoConfigurationEnvironment::String(string(value)?)
            };
            configuration.environment.insert(name.clone(), value);
        }
    }
    if let Some(includes) = document.get("include") {
        let includes = includes
            .as_array()
            .context("Cargo configuration include must be an array")?;
        if includes.len() > MAX_CARGO_CONFIGURATION_ITEMS {
            bail!("Cargo configuration include exceeds the supported entry limit");
        }
        for include in includes {
            let (path, optional) = if let Some(table) = include.as_table() {
                let path = table
                    .get("path")
                    .and_then(toml::Value::as_str)
                    .context("Cargo configuration include path is missing or invalid")?;
                let optional = table
                    .get("optional")
                    .map(|value| {
                        value
                            .as_bool()
                            .context("Cargo configuration include optional flag must be boolean")
                    })
                    .transpose()?
                    .unwrap_or(false);
                (path, optional)
            } else {
                (
                    include
                        .as_str()
                        .context("Cargo configuration include path must be a string")?,
                    false,
                )
            };
            if !path.ends_with(".toml") {
                bail!("Cargo configuration include path must end in .toml");
            }
            let path = Path::new(path);
            configuration.includes.push(CargoConfigurationInclude {
                path: source_path
                    .parent()
                    .context("Cargo configuration include source is unavailable")?
                    .join(path),
                optional,
            });
        }
    }
    Ok(configuration)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoWorkspaceSnapshot {
    pub revision: u64,
    pub input_fingerprint: u64,
    pub workspaces: Vec<CargoWorkspaceModel>,
    pub failures: Vec<CargoCandidateFailure>,
    pub omitted: CargoDiscoveryOmissions,
    pub completeness: CargoSnapshotCompleteness,
}

pub const MAX_CARGO_SNAPSHOT_ENTRIES: usize = 128;
pub const MAX_CARGO_DISCOVERY_CANDIDATES: usize = 512;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CargoDiscoveryOmissions {
    pub candidates: u32,
    pub workspaces: u32,
    pub failures: u32,
}

impl CargoDiscoveryOmissions {
    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoSnapshotCompleteness {
    Complete,
    Partial,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CargoWorkspaceKey {
    pub root: ProjectPath,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoWorkspaceModel {
    pub key: CargoWorkspaceKey,
    pub root_manifest: Option<ProjectPath>,
    pub display_name: String,
    pub is_virtual: bool,
    pub members: Vec<CargoPackageModel>,
    pub configuration: CargoWorkspaceConfiguration,
    pub dependency_graph: Option<CargoDependencyGraph>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoDependencyGraph {
    pub format_version: u32,
    pub metadata_identity: String,
    pub lock_identity: Option<String>,
    pub nodes: Vec<CargoDependencyGraphNode>,
    pub edges: Vec<CargoDependencyGraphEdge>,
    pub roots: Vec<u32>,
    pub completeness: CargoSnapshotCompleteness,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoDependencyGraphNode {
    pub identity: String,
    pub name: String,
    pub version: String,
    pub source_kind: CargoDependencySourceKind,
    pub source_identity: String,
    pub crates_io: bool,
    pub workspace_manifest: Option<ProjectPath>,
    pub enabled_features: Vec<String>,
    pub lock_status: CargoDependencyLockStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoDependencyGraphEdge {
    pub parent: u32,
    pub child: u32,
    pub alias: String,
    pub kind: CargoDependencyKind,
    pub target: Option<String>,
    pub version_requirement: Option<String>,
    pub workspace_inherited: bool,
}

pub fn cargo_evidence_identity(value: &[u8]) -> String {
    Sha256::digest(value)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_crates_io_source(source: &str) -> bool {
    matches!(
        source,
        "registry+https://github.com/rust-lang/crates.io-index"
            | "registry+https://index.crates.io/"
            | "sparse+https://index.crates.io/"
    )
}

fn cargo_source_identity(source: &str) -> String {
    cargo_evidence_identity(
        if is_crates_io_source(source) {
            "registry+https://github.com/rust-lang/crates.io-index"
        } else {
            source
        }
        .as_bytes(),
    )
}

fn is_cargo_evidence_identity(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl CargoDependencyGraph {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format_version == CARGO_GRAPH_FORMAT_VERSION,
            "Unsupported Cargo dependency graph version"
        );
        ensure!(
            is_cargo_evidence_identity(&self.metadata_identity),
            "Invalid Cargo metadata identity"
        );
        ensure!(
            self.lock_identity
                .as_deref()
                .is_none_or(is_cargo_evidence_identity),
            "Invalid Cargo lock identity"
        );
        ensure!(
            self.nodes.len() <= MAX_CARGO_GRAPH_NODES
                && self.edges.len() <= MAX_CARGO_GRAPH_EDGES
                && self.roots.len() <= MAX_CARGO_GRAPH_NODES,
            "Cargo graph count limit exceeded"
        );
        ensure!(
            self.completeness != CargoSnapshotCompleteness::Complete
                || (!self.nodes.is_empty() && !self.roots.is_empty()),
            "Empty Cargo graph coverage is incomplete"
        );
        let mut identities = BTreeSet::new();
        for node in &self.nodes {
            ensure!(
                is_cargo_evidence_identity(&node.identity)
                    && is_cargo_evidence_identity(&node.source_identity)
                    && identities.insert(&node.identity),
                "Invalid or duplicate Cargo graph identity"
            );
            ensure!(
                !node.crates_io
                    || (node.source_kind == CargoDependencySourceKind::Registry
                        && node.source_identity
                            == cargo_source_identity(
                                "registry+https://github.com/rust-lang/crates.io-index"
                            )),
                "Invalid public Cargo source provenance"
            );
            ensure!(
                !node.name.is_empty()
                    && node.name.len() <= 512
                    && !node.name.chars().any(char::is_control)
                    && semver::Version::parse(&node.version).is_ok(),
                "Invalid Cargo graph package"
            );
            ensure!(
                node.enabled_features.len() <= MAX_CARGO_DEPENDENCY_FEATURES,
                "Cargo graph feature limit exceeded"
            );
        }
        let mut edges = BTreeSet::new();
        for edge in &self.edges {
            ensure!(
                (edge.parent as usize) < self.nodes.len()
                    && (edge.child as usize) < self.nodes.len(),
                "Cargo graph edge references absent node"
            );
            ensure!(
                edges.insert((
                    edge.parent,
                    edge.child,
                    &edge.alias,
                    edge.kind,
                    &edge.target
                )),
                "Duplicate Cargo graph edge"
            );
        }
        let roots: BTreeSet<_> = self.roots.iter().collect();
        ensure!(
            roots.len() == self.roots.len()
                && self
                    .roots
                    .iter()
                    .all(|root| (*root as usize) < self.nodes.len()),
            "Invalid Cargo graph root"
        );
        Ok(())
    }
}

struct CargoEvidenceBudget {
    bytes: usize,
    items: usize,
}

impl CargoEvidenceBudget {
    fn field(&mut self, value: &str) -> Result<()> {
        ensure!(value.len() <= 16 * 1024, "Invalid Cargo retained field");
        self.items = self
            .items
            .checked_add(1)
            .context("Cargo item count overflow")?;
        self.bytes = self
            .bytes
            .checked_add(value.len() + 64)
            .context("Cargo evidence byte count overflow")?;
        ensure!(
            self.items <= MAX_CARGO_RETAINED_ITEMS && self.bytes <= MAX_CARGO_SNAPSHOT_BYTES,
            "Cargo retained evidence limit exceeded"
        );
        Ok(())
    }

    fn path(&mut self, value: &ProjectPath) -> Result<()> {
        self.field(value.path.as_unix_str())
    }

    fn optional(&mut self, value: Option<&str>) -> Result<()> {
        if let Some(value) = value {
            self.field(value)?;
        }
        Ok(())
    }
}

pub fn validate_cargo_snapshot_bounds(snapshot: &CargoWorkspaceSnapshot) -> Result<()> {
    validate_cargo_projection_bounds(
        &snapshot.workspaces.iter().collect::<Vec<_>>(),
        &snapshot.failures.iter().collect::<Vec<_>>(),
    )
}

pub fn validate_cargo_workspace_bounds(workspace: &CargoWorkspaceModel) -> Result<()> {
    validate_cargo_projection_bounds(&[workspace], &[])
}

pub fn validate_cargo_replacement_bounds(
    previous: &CargoWorkspaceSnapshot,
    workspace: &CargoWorkspaceModel,
    manifest: &ProjectPath,
) -> Result<()> {
    validate_cargo_snapshot_bounds(previous)?;
    let mut workspaces = previous
        .workspaces
        .iter()
        .filter(|existing| existing.key != workspace.key)
        .collect::<Vec<_>>();
    workspaces.push(workspace);
    let failures = previous
        .failures
        .iter()
        .filter(|failure| failure.manifest_path != *manifest)
        .collect::<Vec<_>>();
    validate_cargo_projection_bounds(&workspaces, &failures)
}

fn validate_cargo_projection_bounds(
    workspaces: &[&CargoWorkspaceModel],
    failures: &[&CargoCandidateFailure],
) -> Result<()> {
    ensure!(
        workspaces.len() <= MAX_CARGO_SNAPSHOT_ENTRIES,
        "Cargo workspace limit ({MAX_CARGO_SNAPSHOT_ENTRIES}) exceeded by {} entries",
        workspaces.len().saturating_sub(MAX_CARGO_SNAPSHOT_ENTRIES)
    );
    ensure!(
        failures.len() <= MAX_CARGO_SNAPSHOT_ENTRIES,
        "Cargo failed manifest limit ({MAX_CARGO_SNAPSHOT_ENTRIES}) exceeded by {} entries",
        failures.len().saturating_sub(MAX_CARGO_SNAPSHOT_ENTRIES)
    );
    let mut budget = CargoEvidenceBudget { bytes: 0, items: 0 };
    for failure in failures {
        budget.path(&failure.manifest_path)?;
        budget.field(&failure.message)?;
    }
    for workspace in workspaces {
        budget.path(&workspace.key.root)?;
        if let Some(path) = &workspace.root_manifest {
            budget.path(path)?;
        }
        budget.field(&workspace.display_name)?;
        for package in &workspace.members {
            budget.field(&package.id)?;
            budget.field(&package.name)?;
            budget.field(&package.version)?;
            budget.path(&package.manifest_path)?;
            for target in &package.targets {
                budget.field(&target.name)?;
                budget.field(&target.edition)?;
                if let CargoTargetKind::Other(value) = &target.kind {
                    budget.field(value)?;
                }
                if let Some(path) = &target.source_path {
                    budget.path(path)?;
                }
                budget.optional(target.source_display_path.as_deref())?;
                for field in target.crate_types.iter().chain(&target.required_features) {
                    budget.field(field)?;
                }
            }
            for feature in &package.features {
                budget.field(&feature.name)?;
                for field in &feature.expands {
                    budget.field(field)?;
                }
            }
            for dependency in &package.dependencies {
                budget.field(&dependency.declaration_name)?;
                budget.field(&dependency.version_requirement)?;
                for field in [
                    &dependency.rename,
                    &dependency.target,
                    &dependency.resolved_name,
                    &dependency.resolved_version,
                ] {
                    budget.optional(field.as_deref())?;
                }
                for path in [
                    &dependency.resolved_workspace_member,
                    &dependency.local_manifest,
                    &dependency.declaration_manifest,
                ]
                .into_iter()
                .flatten()
                {
                    budget.path(path)?;
                }
                for feature in &dependency.requested_features {
                    budget.field(feature)?;
                }
                ensure!(
                    dependency.resolved_instances.len() <= MAX_CARGO_DEPENDENCY_INSTANCES,
                    "Cargo resolved instance limit exceeded"
                );
                for resolved in &dependency.resolved_instances {
                    budget.field(&resolved.name)?;
                    budget.field(&resolved.version)?;
                    ensure!(
                        resolved.enabled_features.len() <= MAX_CARGO_DEPENDENCY_FEATURES,
                        "Cargo feature limit exceeded"
                    );
                    for feature in &resolved.enabled_features {
                        budget.field(feature)?;
                    }
                    for path in [&resolved.workspace_member, &resolved.local_manifest]
                        .into_iter()
                        .flatten()
                    {
                        budget.path(path)?;
                    }
                }
            }
        }
        let configuration = &workspace.configuration;
        for profile in &configuration.profiles {
            budget.field(&profile.name)?;
        }
        if let Some(toolchain) = &configuration.declared_toolchain {
            budget.path(&toolchain.source_path)?;
            budget.optional(toolchain.channel.as_deref())?;
            for field in toolchain.components.iter().chain(&toolchain.targets) {
                budget.field(field)?;
            }
        }
        budget.optional(configuration.host_compiler.release.as_deref())?;
        budget.optional(configuration.host_compiler.host_target.as_deref())?;
        for diagnostic in &configuration.diagnostics {
            budget.field(&diagnostic.message)?;
            if let Some(path) = &diagnostic.source_path {
                budget.path(path)?;
            }
        }
        if let Some(graph) = &workspace.dependency_graph {
            graph.validate()?;
            budget.field(&graph.metadata_identity)?;
            budget.optional(graph.lock_identity.as_deref())?;
            for node in &graph.nodes {
                for field in [
                    &node.identity,
                    &node.name,
                    &node.version,
                    &node.source_identity,
                ] {
                    budget.field(field)?;
                }
                if let Some(path) = &node.workspace_manifest {
                    budget.path(path)?;
                }
                for feature in &node.enabled_features {
                    budget.field(feature)?;
                }
            }
            for edge in &graph.edges {
                budget.field(&edge.alias)?;
                budget.optional(edge.target.as_deref())?;
                budget.optional(edge.version_requirement.as_deref())?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoWorkspaceConfiguration {
    pub profiles: Vec<CargoProfileModel>,
    pub declared_toolchain: Option<CargoToolchainModel>,
    pub host_compiler: CargoHostCompilerModel,
    pub cargo_target: CargoTargetConfiguration,
    pub diagnostics: Vec<CargoConfigurationDiagnostic>,
    pub completeness: CargoConfigurationCompleteness,
}

impl CargoWorkspaceConfiguration {
    pub fn unresolved() -> Self {
        Self {
            profiles: implicit_cargo_profiles(),
            declared_toolchain: None,
            host_compiler: CargoHostCompilerModel::unknown(),
            cargo_target: CargoTargetConfiguration::UnresolvedCargoDefault,
            diagnostics: Vec::new(),
            completeness: CargoConfigurationCompleteness::Complete,
        }
    }

    pub fn retain_stale_safe_facts(&mut self, previous: &Self) {
        if self.host_compiler.status != CargoHostCompilerStatus::Available
            && previous.host_compiler.status == CargoHostCompilerStatus::Available
        {
            self.host_compiler.release = previous.host_compiler.release.clone();
            self.host_compiler.host_target = previous.host_compiler.host_target.clone();
            self.host_compiler.stale = true;
        }
        if self.declared_toolchain.is_none()
            && self.completeness == CargoConfigurationCompleteness::Partial
        {
            self.declared_toolchain = previous.declared_toolchain.clone();
        }
    }

    pub fn add_diagnostic(
        &mut self,
        source_path: Option<ProjectPath>,
        category: CargoConfigurationDiagnosticCategory,
        message: impl AsRef<str>,
    ) {
        self.completeness = CargoConfigurationCompleteness::Partial;
        if self.diagnostics.len() >= MAX_CARGO_CONFIGURATION_DIAGNOSTICS {
            return;
        }
        self.diagnostics.push(CargoConfigurationDiagnostic {
            source_path,
            category,
            message: bounded_configuration_field(message.as_ref()),
        });
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoConfigurationCompleteness {
    Complete,
    Partial,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoProfileModel {
    pub name: String,
    pub origin: CargoProfileOrigin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoProfileOrigin {
    Implicit,
    Declared,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoToolchainModel {
    pub source_path: ProjectPath,
    pub format: CargoToolchainFormat,
    pub channel: Option<String>,
    pub components: Vec<String>,
    pub targets: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoToolchainFormat {
    Toml,
    Legacy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RustToolchainConfiguration {
    pub format: CargoToolchainFormat,
    pub channel: Option<String>,
    pub path: Option<PathBuf>,
    pub components: Option<Vec<String>>,
    pub targets: Option<Vec<String>>,
    pub profile: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoHostCompilerModel {
    pub status: CargoHostCompilerStatus,
    pub release: Option<String>,
    pub host_target: Option<String>,
    pub stale: bool,
}

impl CargoHostCompilerModel {
    pub fn unknown() -> Self {
        Self {
            status: CargoHostCompilerStatus::Unknown,
            release: None,
            host_target: None,
            stale: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoHostCompilerStatus {
    Unknown,
    Available,
    Restricted,
    Missing,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoTargetConfiguration {
    UnresolvedCargoDefault,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoConfigurationDiagnostic {
    pub source_path: Option<ProjectPath>,
    pub category: CargoConfigurationDiagnosticCategory,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoConfigurationDiagnosticCategory {
    Manifest,
    Toolchain,
    CompilerProbe,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoPackageModel {
    pub id: String,
    pub name: String,
    pub version: String,
    pub manifest_path: ProjectPath,
    pub is_default_member: bool,
    pub targets: Vec<CargoTargetModel>,
    pub features: Vec<CargoFeatureModel>,
    pub dependencies: Vec<CargoDependencyModel>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoTargetModel {
    pub name: String,
    pub kind: CargoTargetKind,
    pub crate_types: Vec<String>,
    pub source_path: Option<ProjectPath>,
    pub source_display_path: Option<String>,
    pub required_features: Vec<String>,
    pub edition: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum CargoTargetKind {
    Library,
    Binary,
    Example,
    Test,
    Bench,
    BuildScript,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoFeatureModel {
    pub name: String,
    pub defined: bool,
    pub enabled: CargoFeatureEnabled,
    pub expands: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoFeatureEnabled {
    Enabled,
    Disabled,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoDependencyModel {
    pub declaration_name: String,
    pub rename: Option<String>,
    pub kind: CargoDependencyKind,
    pub version_requirement: String,
    pub optional: bool,
    pub uses_default_features: bool,
    pub requested_features: Vec<String>,
    pub target: Option<String>,
    pub source_kind: CargoDependencySourceKind,
    pub resolved_name: Option<String>,
    pub resolved_version: Option<String>,
    pub resolved_workspace_member: Option<ProjectPath>,
    pub local_manifest: Option<ProjectPath>,
    pub declaration_manifest: Option<ProjectPath>,
    pub declaration_origin: CargoDependencyDeclarationOrigin,
    pub resolved_instances: Vec<CargoResolvedDependencyModel>,
    pub resolution_truncated: bool,
    pub feature_causality: CargoDependencyFeatureCausality,
    pub cycle_detected: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoDependencyDeclarationOrigin {
    Direct,
    WorkspaceInherited,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoDependencyFeatureCausality {
    Validated,
    Ambiguous,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoResolvedDependencyModel {
    pub name: String,
    pub version: String,
    pub source_kind: CargoDependencySourceKind,
    pub enabled_features: Vec<String>,
    pub lock_status: CargoDependencyLockStatus,
    pub workspace_member: Option<ProjectPath>,
    pub local_manifest: Option<ProjectPath>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoDependencyLockStatus {
    Locked,
    NotLocked,
    MissingLockfile,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum CargoDependencyKind {
    Normal,
    Development,
    Build,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoDependencySourceKind {
    Path,
    Registry,
    Git,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoCandidateFailure {
    pub manifest_path: ProjectPath,
    pub category: CargoWorkspaceErrorCategory,
    pub message: String,
    pub has_stale_model: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoWorkspaceErrorCategory {
    Restricted,
    CargoNotFound,
    CargoFailed,
    InvalidMetadata,
    UnsupportedMetadata,
    Disconnected,
    Cancelled,
    Internal,
}

pub fn implicit_cargo_profiles() -> Vec<CargoProfileModel> {
    ["dev", "release"]
        .into_iter()
        .map(|name| CargoProfileModel {
            name: name.to_string(),
            origin: CargoProfileOrigin::Implicit,
        })
        .collect()
}

pub fn parse_cargo_profiles(manifest: &str) -> Result<Vec<CargoProfileModel>> {
    let value = toml::from_str::<toml::Value>(manifest).context("invalid Cargo manifest TOML")?;
    let Some(profile_value) = value.get("profile") else {
        return Ok(implicit_cargo_profiles());
    };
    let profiles = profile_value
        .as_table()
        .context("Cargo manifest profile must be a table")?;
    let mut names = BTreeSet::new();
    for (name, descriptor) in profiles {
        if name.is_empty() || name.len() > MAX_CARGO_CONFIGURATION_FIELD_BYTES {
            bail!("Cargo profile name is empty or exceeds the supported length");
        }
        if !descriptor.is_table() {
            bail!("Cargo profile {name} must be a table");
        }
        names.insert(name.clone());
    }
    if names.len() > MAX_CARGO_CONFIGURATION_ITEMS {
        bail!("Cargo manifest declares more than {MAX_CARGO_CONFIGURATION_ITEMS} profiles");
    }

    let mut result = implicit_cargo_profiles();
    for name in names {
        if let Some(implicit) = result.iter_mut().find(|profile| profile.name == name) {
            implicit.origin = CargoProfileOrigin::Declared;
        } else {
            result.push(CargoProfileModel {
                name,
                origin: CargoProfileOrigin::Declared,
            });
        }
    }
    Ok(result)
}

pub fn parse_rust_toolchain(
    source_path: ProjectPath,
    contents: &str,
) -> Result<CargoToolchainModel> {
    let configuration = parse_rust_toolchain_configuration(
        contents,
        source_path.path.file_name() == Some("rust-toolchain.toml"),
    )?;
    Ok(CargoToolchainModel {
        source_path,
        format: configuration.format,
        channel: configuration.channel,
        components: configuration.components.unwrap_or_default(),
        targets: configuration.targets.unwrap_or_default(),
    })
}

pub(crate) fn parse_rust_toolchain_configuration(
    contents: &str,
    toml_only: bool,
) -> Result<RustToolchainConfiguration> {
    if contents.len() > MAX_RUST_TOOLCHAIN_FILE_BYTES {
        bail!("rust-toolchain file exceeds the supported byte limit");
    }
    if contents.lines().count() == 1 && !toml_only {
        let selection = contents.trim();
        if selection.is_empty() {
            bail!("rust-toolchain file is empty");
        }
        if selection.len() > MAX_CARGO_CONFIGURATION_FIELD_BYTES {
            bail!("legacy rust-toolchain channel exceeds the supported length");
        }
        let (channel, path) = if Path::new(selection).is_absolute() {
            (None, Some(PathBuf::from(selection)))
        } else {
            (Some(selection.to_string()), None)
        };
        return Ok(RustToolchainConfiguration {
            format: CargoToolchainFormat::Legacy,
            channel,
            path,
            components: None,
            targets: None,
            profile: None,
        });
    }

    // TOML parser errors can include arbitrary source text and private host paths.
    let value = toml::from_str::<toml::Value>(contents)
        .map_err(|_| anyhow::anyhow!("invalid rust-toolchain TOML contents"))?;
    let toolchain = value
        .get("toolchain")
        .and_then(toml::Value::as_table)
        .context("rust-toolchain.toml is missing a [toolchain] table")?;
    let channel = optional_toolchain_string(toolchain, "channel")?;
    let path = optional_toolchain_string(toolchain, "path")?.map(PathBuf::from);
    let components = toolchain
        .contains_key("components")
        .then(|| optional_toolchain_string_array(toolchain, "components"))
        .transpose()?;
    let targets = toolchain
        .contains_key("targets")
        .then(|| optional_toolchain_string_array(toolchain, "targets"))
        .transpose()?;
    let profile = optional_toolchain_string(toolchain, "profile")?;
    // Rustup 1.27.1 and 1.28.0 do not count profile alone as a selection.
    if channel.is_none() && path.is_none() && components.is_none() && targets.is_none() {
        bail!("rust-toolchain.toml does not declare a supported toolchain field");
    }
    if let Some(path) = &path {
        if channel.is_some() {
            bail!("rust-toolchain cannot specify both channel and path");
        }
        if !path.is_absolute() {
            bail!("rust-toolchain path must be absolute");
        }
        if components.is_some() || targets.is_some() || profile.is_some() {
            bail!("rust-toolchain path cannot specify components, targets or profile");
        }
    }
    if profile
        .as_deref()
        .is_some_and(|value| !matches!(value, "minimal" | "default" | "complete"))
    {
        bail!("rust-toolchain profile is invalid");
    }
    Ok(RustToolchainConfiguration {
        format: CargoToolchainFormat::Toml,
        channel,
        path,
        components,
        targets,
        profile,
    })
}

pub fn parse_rustc_verbose_version(output: &[u8]) -> Result<CargoHostCompilerModel> {
    if output.len() > MAX_RUSTC_VERBOSE_VERSION_BYTES {
        bail!("rustc -vV output exceeds the supported limit");
    }
    let output = std::str::from_utf8(output).context("rustc -vV output is not valid UTF-8")?;
    let release = rustc_field(output, "release:").context("rustc -vV output is missing release")?;
    let host_target = rustc_field(output, "host:").context("rustc -vV output is missing host")?;
    Ok(CargoHostCompilerModel {
        status: CargoHostCompilerStatus::Available,
        release: Some(release),
        host_target: Some(host_target),
        stale: false,
    })
}

fn optional_toolchain_string(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
) -> Result<Option<String>> {
    let Some(value) = table.get(key) else {
        return Ok(None);
    };
    let value = value
        .as_str()
        .with_context(|| format!("rust-toolchain {key} must be a string"))?;
    if value.len() > MAX_CARGO_CONFIGURATION_FIELD_BYTES {
        bail!("rust-toolchain {key} exceeds the supported length");
    }
    Ok(Some(value.to_string()))
}

fn optional_toolchain_string_array(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
) -> Result<Vec<String>> {
    let Some(value) = table.get(key) else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .with_context(|| format!("rust-toolchain {key} must be an array"))?;
    if values.len() > MAX_CARGO_CONFIGURATION_ITEMS {
        bail!("rust-toolchain {key} exceeds the supported item limit");
    }
    values
        .iter()
        .map(|value| {
            let value = value
                .as_str()
                .with_context(|| format!("rust-toolchain {key} entries must be strings"))?;
            if value.len() > MAX_CARGO_CONFIGURATION_FIELD_BYTES {
                bail!("rust-toolchain {key} entry exceeds the supported length");
            }
            Ok(value.to_string())
        })
        .collect()
}

fn rustc_field(output: &str, prefix: &str) -> Option<String> {
    output
        .lines()
        .find_map(|line| line.strip_prefix(prefix).map(str::trim))
        .filter(|value| !value.is_empty())
        .map(bounded_configuration_field)
}

fn bounded_configuration_field(value: &str) -> String {
    let mut boundary = value.len().min(MAX_CARGO_CONFIGURATION_FIELD_BYTES);
    while !value.is_char_boundary(boundary) {
        boundary = boundary.saturating_sub(1);
    }
    value[..boundary].to_string()
}

pub fn parse_metadata(output: &[u8]) -> Result<Metadata> {
    ensure!(
        output.len() <= MAX_CARGO_METADATA_BYTES,
        "Cargo metadata exceeds the supported byte limit"
    );
    let value: serde_json::Value =
        serde_json::from_slice(output).context("invalid Cargo metadata JSON")?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .context("Cargo metadata is missing a numeric format version")?;
    if version != CARGO_METADATA_FORMAT_VERSION {
        bail!("unsupported Cargo metadata format version {version}");
    }
    serde_json::from_value(value).context("invalid Cargo metadata structure")
}

pub fn workspace_from_metadata(
    metadata: &Metadata,
    mut resolve_path: impl FnMut(&Path) -> Option<ProjectPath>,
) -> Result<CargoWorkspaceModel> {
    let root = resolve_path(metadata.workspace_root.as_std_path())
        .context("Cargo workspace root is outside visible worktrees")?;
    let root_manifest = resolve_path(metadata.workspace_root.join("Cargo.toml").as_std_path());
    let workspace_ids: BTreeSet<&PackageId> = metadata.workspace_members.iter().collect();
    let default_ids: BTreeSet<&PackageId> = if metadata.workspace_default_members.is_available() {
        metadata.workspace_default_members.iter().collect()
    } else {
        BTreeSet::new()
    };
    let packages_by_id: HashMap<&PackageId, &Package> = metadata
        .packages
        .iter()
        .map(|package| (&package.id, package))
        .collect();
    let nodes_by_id: HashMap<&PackageId, &Node> = metadata
        .resolve
        .iter()
        .flat_map(|resolve| &resolve.nodes)
        .map(|node| (&node.id, node))
        .collect();

    let mut members = Vec::new();
    for package in metadata
        .packages
        .iter()
        .filter(|package| workspace_ids.contains(&package.id))
    {
        let Some(manifest_path) = resolve_path(package.manifest_path.as_std_path()) else {
            continue;
        };
        members.push(convert_package(
            package,
            manifest_path,
            default_ids.contains(&package.id),
            nodes_by_id.get(&package.id).copied(),
            &nodes_by_id,
            &packages_by_id,
            &workspace_ids,
            &mut resolve_path,
        ));
    }

    if !metadata.workspace_members.is_empty() && members.is_empty() {
        bail!("Cargo workspace has no members in visible worktrees");
    }

    members.sort_by(|left, right| {
        right
            .is_default_member
            .cmp(&left.is_default_member)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.version.cmp(&right.version))
            .then_with(|| left.manifest_path.cmp(&right.manifest_path))
    });

    let display_name = root
        .path
        .file_name()
        .map(|name| name.to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "Cargo workspace".to_string());

    Ok(CargoWorkspaceModel {
        key: CargoWorkspaceKey { root },
        root_manifest,
        display_name,
        is_virtual: metadata.root_package().is_none(),
        members,
        configuration: CargoWorkspaceConfiguration::unresolved(),
        dependency_graph: Some(dependency_graph_from_metadata(
            metadata,
            &packages_by_id,
            &workspace_ids,
            &mut resolve_path,
        )?),
    })
}

fn dependency_graph_from_metadata(
    metadata: &Metadata,
    packages: &HashMap<&PackageId, &Package>,
    workspace_ids: &BTreeSet<&PackageId>,
    resolve_path: &mut impl FnMut(&Path) -> Option<ProjectPath>,
) -> Result<CargoDependencyGraph> {
    let mut graph = CargoDependencyGraph {
        format_version: CARGO_GRAPH_FORMAT_VERSION,
        metadata_identity: cargo_evidence_identity(&serde_json::to_vec(metadata)?),
        lock_identity: None,
        nodes: Vec::new(),
        edges: Vec::new(),
        roots: Vec::new(),
        completeness: CargoSnapshotCompleteness::Complete,
    };
    let Some(resolve) = &metadata.resolve else {
        graph.completeness = CargoSnapshotCompleteness::Partial;
        return Ok(graph);
    };
    let mut indexes = BTreeMap::new();
    for resolved in &resolve.nodes {
        if graph.nodes.len() >= MAX_CARGO_GRAPH_NODES {
            graph.completeness = CargoSnapshotCompleteness::Partial;
            break;
        }
        let Some(package) = packages.get(&resolved.id) else {
            graph.completeness = CargoSnapshotCompleteness::Partial;
            continue;
        };
        let workspace_member = workspace_ids.contains(&package.id);
        let workspace_manifest = workspace_member
            .then(|| resolve_path(package.manifest_path.as_std_path()))
            .flatten();
        if workspace_member && workspace_manifest.is_none() {
            graph.completeness = CargoSnapshotCompleteness::Partial;
            continue;
        }
        let source = package.source.as_ref().map(ToString::to_string);
        let source_kind = match source.as_deref() {
            None => CargoDependencySourceKind::Path,
            Some(source) if source.starts_with("registry+") || source.starts_with("sparse+") => {
                CargoDependencySourceKind::Registry
            }
            Some(source) if source.starts_with("git+") => CargoDependencySourceKind::Git,
            Some(_) => CargoDependencySourceKind::Other,
        };
        let crates_io = source.as_deref().is_some_and(is_crates_io_source);
        let index = u32::try_from(graph.nodes.len())?;
        ensure!(
            indexes.insert(&resolved.id, index).is_none(),
            "Duplicate Cargo resolved node"
        );
        if workspace_member {
            graph.roots.push(index);
        }
        let mut enabled_features = resolved.features.clone();
        enabled_features.sort();
        enabled_features.dedup();
        if enabled_features.len() > MAX_CARGO_DEPENDENCY_FEATURES {
            enabled_features.truncate(MAX_CARGO_DEPENDENCY_FEATURES);
            graph.completeness = CargoSnapshotCompleteness::Partial;
        }
        graph.nodes.push(CargoDependencyGraphNode {
            identity: cargo_evidence_identity(package.id.repr.as_bytes()),
            name: package.name.clone(),
            version: package.version.to_string(),
            source_kind,
            source_identity: cargo_source_identity(source.as_deref().unwrap_or(&package.id.repr)),
            crates_io,
            workspace_manifest,
            enabled_features,
            lock_status: CargoDependencyLockStatus::Unknown,
        });
    }
    let mut unique_edges = BTreeSet::new();
    'resolved_edges: for resolved in &resolve.nodes {
        let Some(parent) = indexes.get(&resolved.id).copied() else {
            continue;
        };
        let package = packages.get(&resolved.id).copied();
        for dependency in &resolved.deps {
            let Some(child) = indexes.get(&dependency.pkg).copied() else {
                graph.completeness = CargoSnapshotCompleteness::Partial;
                continue;
            };
            let kinds = dependency
                .dep_kinds
                .iter()
                .map(|kind| (kind.kind, kind.target.as_ref().map(ToString::to_string)))
                .collect::<Vec<_>>();
            let kinds = if kinds.is_empty() {
                graph.completeness = CargoSnapshotCompleteness::Partial;
                vec![(DependencyKind::Unknown, None)]
            } else {
                kinds
            };
            for (kind, target) in kinds {
                let kind = match kind {
                    DependencyKind::Normal => CargoDependencyKind::Normal,
                    DependencyKind::Development => CargoDependencyKind::Development,
                    DependencyKind::Build => CargoDependencyKind::Build,
                    DependencyKind::Unknown => CargoDependencyKind::Unknown,
                };
                if graph.edges.len() >= MAX_CARGO_GRAPH_EDGES {
                    graph.completeness = CargoSnapshotCompleteness::Partial;
                    break 'resolved_edges;
                }
                if !unique_edges.insert((
                    parent,
                    child,
                    dependency.name.clone(),
                    kind,
                    target.clone(),
                )) {
                    continue;
                }
                let declarations = package
                    .into_iter()
                    .flat_map(|package| &package.dependencies)
                    .filter(|declaration| {
                        declaration
                            .rename
                            .as_ref()
                            .unwrap_or(&declaration.name)
                            .replace('-', "_")
                            == dependency.name
                            && declaration.target.as_ref().map(ToString::to_string) == target
                            && matches!(
                                (declaration.kind, kind),
                                (DependencyKind::Normal, CargoDependencyKind::Normal)
                                    | (
                                        DependencyKind::Development,
                                        CargoDependencyKind::Development
                                    )
                                    | (DependencyKind::Build, CargoDependencyKind::Build)
                            )
                    })
                    .collect::<Vec<_>>();
                graph.edges.push(CargoDependencyGraphEdge {
                    parent,
                    child,
                    alias: dependency.name.clone(),
                    kind,
                    target,
                    version_requirement: (declarations.len() == 1)
                        .then(|| {
                            declarations
                                .first()
                                .map(|declaration| declaration.req.to_string())
                        })
                        .flatten(),
                    workspace_inherited: false,
                });
            }
        }
        if resolved.deps.is_empty() && !resolved.dependencies.is_empty() {
            graph.completeness = CargoSnapshotCompleteness::Partial;
        }
    }
    if graph.nodes.is_empty() || graph.roots.is_empty() {
        graph.completeness = CargoSnapshotCompleteness::Partial;
    }
    graph.validate()?;
    Ok(graph)
}

fn convert_package(
    package: &Package,
    manifest_path: ProjectPath,
    is_default_member: bool,
    node: Option<&Node>,
    nodes_by_id: &HashMap<&PackageId, &Node>,
    packages_by_id: &HashMap<&PackageId, &Package>,
    workspace_ids: &BTreeSet<&PackageId>,
    resolve_path: &mut impl FnMut(&Path) -> Option<ProjectPath>,
) -> CargoPackageModel {
    let mut targets = package
        .targets
        .iter()
        .map(|target| convert_target(target, resolve_path))
        .collect::<Vec<_>>();
    targets.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.source_display_path.cmp(&right.source_display_path))
    });

    let mut feature_names: BTreeSet<String> = package.features.keys().cloned().collect();
    if let Some(node) = node {
        feature_names.extend(node.features.iter().cloned());
    }
    let features = feature_names
        .into_iter()
        .map(|name| {
            let expands = package.features.get(&name).cloned().unwrap_or_default();
            CargoFeatureModel {
                defined: package.features.contains_key(&name),
                enabled: match node {
                    Some(node) if node.features.contains(&name) => CargoFeatureEnabled::Enabled,
                    Some(_) => CargoFeatureEnabled::Disabled,
                    None => CargoFeatureEnabled::Unknown,
                },
                name,
                expands,
            }
        })
        .collect();

    let mut dependencies = package
        .dependencies
        .iter()
        .map(|dependency| {
            convert_dependency(
                dependency,
                node,
                nodes_by_id,
                packages_by_id,
                workspace_ids,
                resolve_path,
            )
        })
        .collect::<Vec<_>>();
    dependencies.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| dependency_label(left).cmp(dependency_label(right)))
            .then_with(|| left.target.cmp(&right.target))
            .then_with(|| left.version_requirement.cmp(&right.version_requirement))
    });

    CargoPackageModel {
        id: stable_package_id(&manifest_path, &package.name, &package.version.to_string()),
        name: package.name.clone(),
        version: package.version.to_string(),
        manifest_path,
        is_default_member,
        targets,
        features,
        dependencies,
    }
}

pub(crate) fn stable_package_id(manifest_path: &ProjectPath, name: &str, version: &str) -> String {
    format!(
        "{}:{}:{name}@{version}",
        manifest_path.worktree_id.to_proto(),
        manifest_path.path.as_unix_str()
    )
}

fn convert_target(
    target: &Target,
    resolve_path: &mut impl FnMut(&Path) -> Option<ProjectPath>,
) -> CargoTargetModel {
    let source_path = resolve_path(target.src_path.as_std_path());
    let source_display_path = source_path
        .as_ref()
        .map(|path| path.path.as_unix_str().to_string());
    CargoTargetModel {
        name: target.name.clone(),
        kind: normalize_target_kind(&target.kind),
        crate_types: target.crate_types.iter().map(ToString::to_string).collect(),
        source_path,
        source_display_path,
        required_features: target.required_features.clone(),
        edition: target.edition.to_string(),
    }
}

pub(crate) fn normalize_target_kind(kinds: &[TargetKind]) -> CargoTargetKind {
    let Some(kind) = kinds.first() else {
        return CargoTargetKind::Other("unknown".to_string());
    };
    match kind {
        TargetKind::Lib
        | TargetKind::RLib
        | TargetKind::DyLib
        | TargetKind::CDyLib
        | TargetKind::StaticLib
        | TargetKind::ProcMacro => CargoTargetKind::Library,
        TargetKind::Bin => CargoTargetKind::Binary,
        TargetKind::Example => CargoTargetKind::Example,
        TargetKind::Test => CargoTargetKind::Test,
        TargetKind::Bench => CargoTargetKind::Bench,
        TargetKind::CustomBuild => CargoTargetKind::BuildScript,
        TargetKind::Unknown(value) => CargoTargetKind::Other(value.clone()),
        _ => CargoTargetKind::Other(kind.to_string()),
    }
}

fn convert_dependency(
    dependency: &Dependency,
    node: Option<&Node>,
    nodes_by_id: &HashMap<&PackageId, &Node>,
    packages_by_id: &HashMap<&PackageId, &Package>,
    workspace_ids: &BTreeSet<&PackageId>,
    resolve_path: &mut impl FnMut(&Path) -> Option<ProjectPath>,
) -> CargoDependencyModel {
    let alias = dependency.rename.as_deref().unwrap_or(&dependency.name);
    let matching_packages = node
        .into_iter()
        .flat_map(|node| &node.deps)
        .filter(|node_dependency| node_dependency.name == alias)
        .filter_map(|node_dependency| {
            packages_by_id
                .get(&node_dependency.pkg)
                .copied()
                .map(|package| (node_dependency, package))
        })
        .collect::<Vec<_>>();
    let resolved_package = matching_packages.first().map(|(_, package)| *package);
    let local_manifest = dependency
        .path
        .as_ref()
        .and_then(|path| resolve_path(path.join("Cargo.toml").as_std_path()))
        .or_else(|| {
            resolved_package.and_then(|package| resolve_path(package.manifest_path.as_std_path()))
        });
    let resolved_workspace_member = resolved_package
        .filter(|package| workspace_ids.contains(&package.id))
        .and_then(|package| resolve_path(package.manifest_path.as_std_path()));
    let resolution_truncated = matching_packages.len() > MAX_CARGO_DEPENDENCY_INSTANCES;
    let resolved_instances = matching_packages
        .into_iter()
        .take(MAX_CARGO_DEPENDENCY_INSTANCES)
        .map(|(node_dependency, package)| {
            let workspace_member = workspace_ids
                .contains(&package.id)
                .then(|| resolve_path(package.manifest_path.as_std_path()))
                .flatten();
            let local_manifest = dependency
                .path
                .as_ref()
                .and_then(|path| resolve_path(path.join("Cargo.toml").as_std_path()))
                .or_else(|| resolve_path(package.manifest_path.as_std_path()));
            let mut enabled_features = nodes_by_id
                .get(&node_dependency.pkg)
                .map(|node| node.features.clone())
                .unwrap_or_default();
            enabled_features.sort();
            enabled_features.dedup();
            enabled_features.truncate(MAX_CARGO_DEPENDENCY_FEATURES);
            CargoResolvedDependencyModel {
                name: package.name.clone(),
                version: package.version.to_string(),
                source_kind: dependency_source_kind(dependency),
                enabled_features,
                lock_status: CargoDependencyLockStatus::Unknown,
                workspace_member,
                local_manifest,
            }
        })
        .collect::<Vec<_>>();

    CargoDependencyModel {
        declaration_name: dependency.name.clone(),
        rename: dependency.rename.clone(),
        kind: match dependency.kind {
            DependencyKind::Normal => CargoDependencyKind::Normal,
            DependencyKind::Development => CargoDependencyKind::Development,
            DependencyKind::Build => CargoDependencyKind::Build,
            DependencyKind::Unknown => CargoDependencyKind::Unknown,
        },
        version_requirement: dependency.req.to_string(),
        optional: dependency.optional,
        uses_default_features: dependency.uses_default_features,
        requested_features: dependency.features.clone(),
        target: dependency.target.as_ref().map(ToString::to_string),
        source_kind: dependency_source_kind(dependency),
        resolved_name: resolved_package.map(|package| package.name.clone()),
        resolved_version: resolved_package.map(|package| package.version.to_string()),
        resolved_workspace_member,
        local_manifest,
        declaration_manifest: None,
        declaration_origin: CargoDependencyDeclarationOrigin::Unknown,
        feature_causality: if resolved_instances.len() == 1 {
            CargoDependencyFeatureCausality::Validated
        } else if resolved_instances.len() > 1 {
            CargoDependencyFeatureCausality::Ambiguous
        } else {
            CargoDependencyFeatureCausality::Unknown
        },
        resolved_instances,
        resolution_truncated,
        cycle_detected: false,
    }
}

fn dependency_source_kind(dependency: &Dependency) -> CargoDependencySourceKind {
    if dependency.path.is_some() {
        CargoDependencySourceKind::Path
    } else if dependency
        .source
        .as_deref()
        .is_some_and(|source| source.starts_with("git+"))
    {
        CargoDependencySourceKind::Git
    } else if dependency.registry.is_some()
        || dependency
            .source
            .as_deref()
            .is_some_and(|source| source.starts_with("registry+"))
    {
        CargoDependencySourceKind::Registry
    } else {
        CargoDependencySourceKind::Other
    }
}

pub fn enrich_dependency_provenance(
    workspace: &mut CargoWorkspaceModel,
    manifests: &BTreeMap<ProjectPath, String>,
    lock_contents: Option<&str>,
) {
    let lock_packages = lock_contents.and_then(parse_lock_packages);
    let lock_state = match (lock_contents, lock_packages.as_ref()) {
        (None, _) => CargoDependencyLockStatus::MissingLockfile,
        (Some(_), None) => CargoDependencyLockStatus::Unknown,
        (Some(_), Some(_)) => CargoDependencyLockStatus::NotLocked,
    };
    let adjacency = workspace
        .members
        .iter()
        .map(|package| {
            (
                package.manifest_path.clone(),
                package
                    .dependencies
                    .iter()
                    .flat_map(|dependency| &dependency.resolved_instances)
                    .filter_map(|resolved| resolved.workspace_member.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for package in &mut workspace.members {
        let manifest = manifests
            .get(&package.manifest_path)
            .and_then(|contents| toml::from_str::<toml::Value>(contents).ok());
        for dependency in &mut package.dependencies {
            dependency.declaration_manifest = manifests
                .contains_key(&package.manifest_path)
                .then(|| package.manifest_path.clone());
            dependency.declaration_origin = manifest
                .as_ref()
                .and_then(|manifest| manifest_dependency_value(manifest, dependency))
                .map(|value| {
                    if value
                        .as_table()
                        .and_then(|table| table.get("workspace"))
                        .and_then(toml::Value::as_bool)
                        == Some(true)
                    {
                        CargoDependencyDeclarationOrigin::WorkspaceInherited
                    } else {
                        CargoDependencyDeclarationOrigin::Direct
                    }
                })
                .unwrap_or(CargoDependencyDeclarationOrigin::Unknown);

            for resolved in &mut dependency.resolved_instances {
                resolved.lock_status = lock_packages
                    .as_ref()
                    .map(|packages| {
                        if packages.iter().any(|package| package.matches(resolved)) {
                            CargoDependencyLockStatus::Locked
                        } else {
                            CargoDependencyLockStatus::NotLocked
                        }
                    })
                    .unwrap_or(lock_state);
                dependency.cycle_detected |=
                    resolved.workspace_member.as_ref().is_some_and(|target| {
                        dependency_path_reaches(target, &package.manifest_path, &adjacency)
                    });
            }
        }
    }
    if let Some(graph) = &mut workspace.dependency_graph {
        graph.lock_identity =
            lock_contents.map(|contents| cargo_evidence_identity(contents.as_bytes()));
        for node in &mut graph.nodes {
            node.lock_status = lock_packages
                .as_ref()
                .map(|packages| {
                    if packages.iter().any(|package| {
                        package.name == node.name
                            && package.version == node.version
                            && if node.source_kind == CargoDependencySourceKind::Path {
                                package.source_identity.is_none()
                            } else {
                                package.source_identity.as_deref()
                                    == Some(node.source_identity.as_str())
                            }
                    }) {
                        CargoDependencyLockStatus::Locked
                    } else {
                        CargoDependencyLockStatus::NotLocked
                    }
                })
                .unwrap_or(lock_state);
        }
        for edge in &mut graph.edges {
            let declaration_origin = graph
                .nodes
                .get(edge.parent as usize)
                .and_then(|node| node.workspace_manifest.as_ref())
                .and_then(|manifest| {
                    workspace
                        .members
                        .iter()
                        .find(|member| &member.manifest_path == manifest)
                })
                .and_then(|member| {
                    member.dependencies.iter().find(|dependency| {
                        dependency
                            .rename
                            .as_ref()
                            .unwrap_or(&dependency.declaration_name)
                            .replace('-', "_")
                            == edge.alias
                            && dependency.kind == edge.kind
                            && dependency.target == edge.target
                    })
                });
            edge.workspace_inherited = declaration_origin.is_some_and(|declaration| {
                declaration.declaration_origin
                    == CargoDependencyDeclarationOrigin::WorkspaceInherited
            });
        }
    }
}

fn manifest_dependency_value<'a>(
    manifest: &'a toml::Value,
    dependency: &CargoDependencyModel,
) -> Option<&'a toml::Value> {
    let section = match dependency.kind {
        CargoDependencyKind::Normal => "dependencies",
        CargoDependencyKind::Development => "dev-dependencies",
        CargoDependencyKind::Build => "build-dependencies",
        CargoDependencyKind::Unknown => return None,
    };
    let key = dependency
        .rename
        .as_deref()
        .unwrap_or(&dependency.declaration_name);
    let root_match = manifest
        .get(section)
        .and_then(toml::Value::as_table)
        .and_then(|table| table.get(key));
    if dependency.target.is_none() && root_match.is_some() {
        return root_match;
    }
    manifest
        .get("target")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(|targets| targets.values())
        .filter_map(toml::Value::as_table)
        .filter_map(|target| target.get(section))
        .filter_map(toml::Value::as_table)
        .find_map(|dependencies| dependencies.get(key))
        .or(root_match)
}

#[derive(Clone, Debug)]
struct LockPackage {
    name: String,
    version: String,
    source_kind: CargoDependencySourceKind,
    source_identity: Option<String>,
}

impl LockPackage {
    fn matches(&self, resolved: &CargoResolvedDependencyModel) -> bool {
        self.name == resolved.name
            && self.version == resolved.version
            && (self.source_kind == resolved.source_kind
                || matches!(resolved.source_kind, CargoDependencySourceKind::Path))
    }
}

fn parse_lock_packages(contents: &str) -> Option<Vec<LockPackage>> {
    let value = toml::from_str::<toml::Value>(contents).ok()?;
    let packages = value.get("package")?.as_array()?;
    let mut result = Vec::with_capacity(packages.len());
    for package in packages {
        let package = package.as_table()?;
        let name = package.get("name")?.as_str()?;
        let version = package.get("version")?.as_str()?;
        let source_kind = package
            .get("source")
            .and_then(toml::Value::as_str)
            .map(|source| {
                if source.starts_with("registry+") {
                    CargoDependencySourceKind::Registry
                } else if source.starts_with("git+") {
                    CargoDependencySourceKind::Git
                } else {
                    CargoDependencySourceKind::Other
                }
            })
            .unwrap_or(CargoDependencySourceKind::Path);
        result.push(LockPackage {
            name: name.to_string(),
            version: version.to_string(),
            source_kind,
            source_identity: package
                .get("source")
                .and_then(toml::Value::as_str)
                .map(cargo_source_identity),
        });
    }
    Some(result)
}

fn dependency_path_reaches(
    start: &ProjectPath,
    destination: &ProjectPath,
    adjacency: &BTreeMap<ProjectPath, Vec<ProjectPath>>,
) -> bool {
    let mut pending = vec![start.clone()];
    let mut visited = BTreeSet::new();
    while let Some(path) = pending.pop() {
        if &path == destination {
            return true;
        }
        if !visited.insert(path.clone()) || visited.len() > 10_000 {
            continue;
        }
        if let Some(next) = adjacency.get(&path) {
            pending.extend(next.iter().cloned());
        }
    }
    false
}

fn dependency_label(dependency: &CargoDependencyModel) -> &str {
    dependency
        .rename
        .as_deref()
        .unwrap_or(&dependency.declaration_name)
}

pub fn deduplicate_workspaces(
    workspaces: impl IntoIterator<Item = CargoWorkspaceModel>,
) -> Vec<CargoWorkspaceModel> {
    let mut by_key = BTreeMap::new();
    for workspace in workspaces {
        by_key.entry(workspace.key.clone()).or_insert(workspace);
    }
    by_key.into_values().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_runtime_recipe_inverse_renews_selected_shape_and_literal_operands() {
        let base = [
            "build",
            "--message-format=json",
            "--package",
            "selected-package",
            "--bin",
            "selected-target",
            "--target-dir",
            "/selected/output",
        ]
        .map(String::from)
        .to_vec();
        let operands = vec!["".into(), "$ZED_FILE;* --target".into(), "--".into()];
        for selected in [base.clone(), {
            let mut example = base;
            example[4] = "--example".into();
            example.extend(["--features".into(), "selected-feature".into()]);
            example
        }] {
            let logical = native_cargo_runtime_logical_arguments(&selected, &operands)
                .expect("literal logical recipe");
            assert_eq!(
                native_cargo_runtime_selected_build_arguments(&logical, &operands)
                    .expect("same selected recipe"),
                selected
            );
            let mut changed = operands.clone();
            changed[1] = "different literal workload".into();
            assert!(native_cargo_runtime_selected_build_arguments(&logical, &changed).is_err());
            let mut flags = logical.clone();
            flags[1] = "--target".into();
            assert!(native_cargo_runtime_selected_build_arguments(&flags, &operands).is_err());
            let mut hidden = logical.clone();
            hidden.insert(2, "--config=authored.runner".into());
            assert!(native_cargo_runtime_selected_build_arguments(&hidden, &operands).is_err());
        }
    }

    #[test]
    fn native_runtime_named_profile_recipe_roundtrip_rejects_noncanonical_options() {
        let base = [
            "build",
            "--message-format=json",
            "--package",
            "selected-package",
            "--bin",
            "selected-target",
            "--target-dir",
            "/selected/output",
        ]
        .map(String::from)
        .to_vec();
        let operands = vec![String::new(), "literal --profile ; $HOME".into()];
        for kind in ["--bin", "--example"] {
            for features in [false, true] {
                for profile in ["dev", "release", "native-review"] {
                    let mut selected = base.clone();
                    selected[4] = kind.into();
                    if features {
                        selected.extend(["--features".into(), "required-one,required-two".into()]);
                    }
                    selected.extend(["--profile".into(), profile.into()]);
                    assert_eq!(
                        native_cargo_build_profile(&selected).expect("canonical profile"),
                        Some(profile)
                    );
                    let runtime = native_cargo_runtime_logical_arguments(&selected, &operands)
                        .expect("literal runtime profile");
                    assert_eq!(
                        native_cargo_runtime_selected_build_arguments(&runtime, &operands)
                            .expect("exact selected inverse"),
                        selected
                    );
                    for tail in [
                        vec!["--profile", "other"],
                        vec!["--target", "foreign"],
                        vec!["--no-default-features"],
                        vec!["--config", "authored.runner=foreign"],
                    ] {
                        let mut changed = selected.clone();
                        changed.extend(tail.into_iter().map(str::to_owned));
                        assert!(
                            native_cargo_runtime_logical_arguments(&changed, &operands).is_err()
                        );
                    }
                }
            }
        }
        for name in ["", "--config", "has space", "bad\nprofile"] {
            let mut invalid = base.clone();
            invalid.extend(["--profile".into(), name.into()]);
            assert!(native_cargo_runtime_logical_arguments(&invalid, &operands).is_err());
        }
    }

    #[test]
    fn native_runtime_configuration_records_target_runner_and_unstable_routes() {
        let path = Path::new("/workspace/.cargo/config.toml");
        for contents in [
            "[build]\ntarget = \"aarch64-apple-darwin\"\n",
            "[target.aarch64-apple-darwin]\nrunner = [\"configured-runner\"]\n",
            "[target.\"cfg(unix)\"]\nrunner = \"configured-runner\"\n",
            "[unstable]\nhost-config = true\n",
        ] {
            assert!(
                parse_cargo_host_execution_configuration(contents, path)
                    .expect("existing shared typed parser")
                    .native_runtime_target_or_runner_configured
            );
        }
        let supported = parse_cargo_host_execution_configuration("[build]\nrustc = \"compiler\"\nrustc-wrapper = \"wrapper\"\n[env]\nCONFIGURED_VALUE = \"literal\"\n", path)
            .expect("ordinary compiler configuration data remains supported");
        assert!(!supported.native_runtime_target_or_runner_configured);
        let mut merged = supported;
        merged
            .merge_values(
                parse_cargo_host_execution_configuration(
                    "[build]\ntarget = \"aarch64-apple-darwin\"\n",
                    path,
                )
                .expect("target configured"),
            )
            .expect("shared merge");
        assert!(merged.native_runtime_target_or_runner_configured);
    }

    #[test]
    fn native_cargo_runtime_logical_recipe_preserves_exact_selection_and_literal_arguments() {
        let base = [
            "build",
            "--message-format=json",
            "--package",
            "package",
            "--bin",
            "program",
            "--target-dir",
            "/workspace/target output",
        ]
        .map(str::to_owned)
        .to_vec();
        let operands = ["", "$ZED_FILE ; touch absent", "unicode-λ", "--help"]
            .map(str::to_owned)
            .to_vec();
        let arguments = native_cargo_runtime_logical_arguments(&base, &operands)
            .expect("bounded logical recipe");
        assert_eq!(
            &arguments[..9],
            &[
                "run",
                "--offline",
                "--message-format=json",
                "--package",
                "package",
                "--bin",
                "program",
                "--target-dir",
                "/workspace/target output"
            ]
        );
        assert_eq!(arguments.get(9).map(String::as_str), Some("--"));
        assert_eq!(&arguments[10..], operands);
        assert!(!arguments.iter().any(|argument| argument == "--config"
            || argument == "--target"
            || argument.contains("runtime-observer")));
        let mut example = base;
        example[4] = "--example".into();
        example.extend(["--features".into(), "required-one,required-two".into()]);
        let arguments = native_cargo_runtime_logical_arguments(&example, &[])
            .expect("exact example with required features");
        assert_eq!(arguments.get(5).map(String::as_str), Some("--example"));
        assert_eq!(
            &arguments[9..],
            &["--features", "required-one,required-two", "--"]
        );
    }

    #[test]
    fn native_cargo_runtime_logical_recipe_rejects_unreviewed_shapes() {
        let base = [
            "build",
            "--message-format=json",
            "--package",
            "package",
            "--bin",
            "program",
            "--target-dir",
            "/workspace/target",
        ]
        .map(str::to_owned)
        .to_vec();
        for (index, value) in [
            (0, "test"),
            (1, "--message-format=human"),
            (2, "--workspace"),
            (4, "--test"),
            (6, "--target"),
            (7, "relative-target"),
        ] {
            let mut changed = base.clone();
            changed[index] = value.into();
            assert!(
                native_cargo_runtime_logical_arguments(&changed, &[]).is_err(),
                "changed operand {index}"
            );
        }
        let mut extra = base;
        extra.extend(["--config".into(), "authored.runner=unissued".into()]);
        assert!(native_cargo_runtime_logical_arguments(&extra, &[]).is_err());
    }

    #[test]
    fn native_cargo_runtime_logical_recipe_enforces_inclusive_workload_bounds() {
        let base = [
            "build",
            "--message-format=json",
            "--package",
            "package",
            "--bin",
            "program",
            "--target-dir",
            "/workspace/target",
        ]
        .map(str::to_owned)
        .to_vec();
        assert!(native_cargo_runtime_logical_arguments(&base, &vec![String::new(); 128]).is_ok());
        assert!(native_cargo_runtime_logical_arguments(&base, &vec![String::new(); 129]).is_err());
        assert!(native_cargo_runtime_logical_arguments(&base, &vec!["x".repeat(4096); 8]).is_ok());
        assert!(native_cargo_runtime_logical_arguments(&base, &vec!["x".repeat(4096); 9]).is_err());
        assert!(native_cargo_runtime_logical_arguments(&base, &["x".repeat(4097)]).is_err());
        assert!(native_cargo_runtime_logical_arguments(&base, &["nul\0operand".into()]).is_err());
    }

    use std::sync::Arc;

    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    fn project_path(path: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(1),
            path: Arc::from(RelPath::from_unix_str(path).expect("fixture path should be valid")),
        }
    }

    #[test]
    fn cargo_workspace_execution_configuration_preserves_field_origins() {
        let directory = tempfile::tempdir().expect("configuration fixture");
        let included = directory.path().join(".cargo/fragments/compiler.toml");
        let containing = directory.path().join(".cargo/config.toml");
        let mut configuration = parse_cargo_host_execution_configuration(
            "[build]\nrustc = '../tooling/included-rustc'\nrustdoc = 'tools/rustdoc'\n[env.RUSTUP_AUTO_INSTALL]\nvalue = 'child-value'\nrelative = true\nforce = true\n",
            &included,
        )
        .expect("included configuration");
        let own = parse_cargo_host_execution_configuration(
            "[build]\nrustc = 'tooling/selected-rustc'\n[env.RUSTUP_AUTO_INSTALL]\nforce = false\n",
            &containing,
        )
        .expect("containing configuration");
        configuration.merge_values(own).expect("table field merge");
        assert_eq!(
            configuration
                .compiler
                .expect("selected compiler")
                .executable(),
            directory.path().join("tooling/selected-rustc")
        );
        assert_eq!(
            configuration
                .documentation_generator
                .expect("included documentation generator")
                .executable(),
            directory.path().join(".cargo/tools/rustdoc")
        );
        let (value, force) = configuration
            .environment
            .get("RUSTUP_AUTO_INSTALL")
            .expect("merged environment entry")
            .resolve()
            .expect("merged value");
        assert_eq!(
            value,
            directory.path().join(".cargo/child-value").into_os_string()
        );
        assert!(!force, "an explicit false overrides the included true");
    }

    #[test]
    fn cargo_workspace_execution_configuration_keeps_lookup_and_include_modes() {
        let directory = tempfile::tempdir().expect("configuration fixture");
        let source = directory.path().join(".cargo/config.toml");
        let configuration = parse_cargo_host_execution_configuration(
            "include = ['first.toml', {path = 'optional.toml', optional = true}, '.toml']\n[build]\nrustc = 'rustc'\nrustc-wrapper = ''\n[env]\nCUSTOM = 'preserved'\n",
            &source,
        )
        .expect("lookup and include configuration");
        assert_eq!(
            configuration
                .compiler
                .expect("compiler lookup")
                .executable(),
            Path::new("rustc")
        );
        assert_eq!(
            configuration.wrapper.expect("explicit empty wrapper").value,
            ""
        );
        assert_eq!(
            configuration.includes,
            [
                CargoConfigurationInclude {
                    path: directory.path().join(".cargo/first.toml"),
                    optional: false,
                },
                CargoConfigurationInclude {
                    path: directory.path().join(".cargo/optional.toml"),
                    optional: true,
                },
                CargoConfigurationInclude {
                    path: directory.path().join(".cargo/.toml"),
                    optional: false,
                },
            ]
        );
        assert_eq!(
            configuration
                .environment
                .get("CUSTOM")
                .expect("unrelated env")
                .resolve()
                .expect("unrelated env value"),
            (std::ffi::OsString::from("preserved"), false)
        );
    }

    #[test]
    fn cargo_workspace_execution_configuration_rejects_conflicting_and_missing_env() {
        let directory = tempfile::tempdir().expect("configuration fixture");
        let source = directory.path().join(".cargo/config.toml");
        let mut lower =
            parse_cargo_host_execution_configuration("[env]\nRUSTUP_AUTO_INSTALL = '1'\n", &source)
                .expect("string environment");
        let higher = parse_cargo_host_execution_configuration(
            "[env.RUSTUP_AUTO_INSTALL]\nvalue = '0'\nforce = true\n",
            &source,
        )
        .expect("table environment");
        lower
            .merge_values(higher)
            .expect_err("Cargo rejects string/table conflicts");
        let missing = parse_cargo_host_execution_configuration(
            "[env.RUSTUP_AUTO_INSTALL]\nforce = true\n",
            &source,
        )
        .expect("partial table may obtain value from an include");
        missing
            .environment
            .get("RUSTUP_AUTO_INSTALL")
            .expect("partial value")
            .resolve()
            .expect_err("unmerged missing value is invalid");
    }

    #[test]
    fn cargo_workspace_execution_configuration_bounds_and_sanitizes_input() {
        let source = Path::new("/private/config-fixture/.cargo/config.toml");
        for contents in [
            "[build]\nrustc = 42\n".to_string(),
            "[env]\nCUSTOM = { value = 'ok', force = 'private' }\n".to_string(),
            "include = ['not-toml']\n".to_string(),
            "include = [{path='ok.toml', optional=1}]\n".to_string(),
            "[env]\nCUSTOM = 'private missing quote".to_string(),
            "#".repeat(MAX_CARGO_PASSIVE_CONFIGURATION_BYTES + 1),
        ] {
            let error = parse_cargo_host_execution_configuration(&contents, source)
                .expect_err("invalid configuration rejected");
            assert!(!format!("{error:#}").contains("private"));
            assert!(!format!("{error:#}").contains("CUSTOM"));
        }
        let entries = (0..=MAX_CARGO_CONFIGURATION_ITEMS)
            .map(|index| format!("VARIABLE_{index} = 'value'\n"))
            .collect::<String>();
        parse_cargo_host_execution_configuration(&format!("[env]\n{entries}"), source)
            .expect_err("environment inventory bounded");
    }

    #[test]
    fn cargo_workspace_profiles_include_implicit_and_declared_profiles() {
        let profiles = parse_cargo_profiles(include_str!(
            "../test_data/cargo_workspace/profiles-custom.toml"
        ))
        .expect("custom profile fixture should parse");
        assert_eq!(
            profiles,
            vec![
                CargoProfileModel {
                    name: "dev".to_string(),
                    origin: CargoProfileOrigin::Declared,
                },
                CargoProfileModel {
                    name: "release".to_string(),
                    origin: CargoProfileOrigin::Implicit,
                },
                CargoProfileModel {
                    name: "ci".to_string(),
                    origin: CargoProfileOrigin::Declared,
                },
                CargoProfileModel {
                    name: "ship".to_string(),
                    origin: CargoProfileOrigin::Declared,
                },
            ]
        );
        assert!(
            parse_cargo_profiles(include_str!(
                "../test_data/cargo_workspace/profiles-malformed.toml"
            ))
            .is_err()
        );
    }

    #[test]
    fn cargo_workspace_toolchain_formats_are_bounded_declarations() {
        let toml = parse_rust_toolchain(
            project_path("rust-toolchain.toml"),
            include_str!("../test_data/cargo_workspace/rust-toolchain.toml"),
        )
        .expect("TOML toolchain fixture should parse");
        assert_eq!(toml.format, CargoToolchainFormat::Toml);
        assert_eq!(toml.channel.as_deref(), Some("stable"));
        assert_eq!(toml.components, ["rustfmt", "clippy"]);
        assert_eq!(toml.targets, ["wasm32-unknown-unknown"]);

        let legacy = parse_rust_toolchain(
            project_path("rust-toolchain"),
            include_str!("../test_data/cargo_workspace/rust-toolchain-legacy"),
        )
        .expect("legacy toolchain fixture should parse");
        assert_eq!(legacy.format, CargoToolchainFormat::Legacy);
        assert_eq!(legacy.channel.as_deref(), Some("1.90.0"));
    }

    #[test]
    fn cargo_workspace_toolchain_parser_honors_file_mode_and_field_presence() {
        let with_comment = parse_rust_toolchain(
            project_path("rust-toolchain.toml"),
            "# installed compiler\n[toolchain]\nchannel = \"stable\"\n",
        )
        .expect("commented TOML declaration should parse");
        assert_eq!(with_comment.format, CargoToolchainFormat::Toml);
        assert_eq!(with_comment.channel.as_deref(), Some("stable"));
        assert!(parse_rust_toolchain(project_path("rust-toolchain.toml"), "stable\n").is_err());
        assert!(parse_rust_toolchain(project_path("rust-toolchain"), "stable\nnightly\n").is_err());

        for contents in [
            "[toolchain]\ncomponents = []\n",
            "[toolchain]\ntargets = []\n",
            "[toolchain]\ncomponents = []\nprofile = \"minimal\"\n",
        ] {
            let parsed = parse_rust_toolchain_configuration(contents, true)
                .expect("empty array presence is a valid default-toolchain request");
            assert_eq!(parsed.channel, None);
            assert!(parsed.components.is_some() || parsed.targets.is_some());
        }
        for contents in [
            "",
            " \n",
            "[toolchain]\n",
            "[toolchain]\nprofile = \"minimal\"\n",
            "[toolchain]\nchannel = \"stable\"\nprofile = \"unknown\"\n",
            "[toolchain]\ncomponents = [1]\n",
            "[toolchain]\nchannel = 1\n",
        ] {
            assert!(parse_rust_toolchain_configuration(contents, true).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn cargo_workspace_toolchain_parser_keeps_host_paths_out_of_normalized_models() {
        let host_path = "/private/toolchains/compiler";
        for (contents, toml_only) in [
            (format!("{host_path}\n"), false),
            (format!("[toolchain]\npath = {host_path:?}\n"), true),
        ] {
            let parsed = parse_rust_toolchain_configuration(&contents, toml_only)
                .expect("absolute toolchain should parse");
            assert_eq!(parsed.path.as_deref(), Some(Path::new(host_path)));
            assert_eq!(parsed.channel, None);
            let normalized = parse_rust_toolchain(
                project_path(if toml_only {
                    "rust-toolchain.toml"
                } else {
                    "rust-toolchain"
                }),
                &contents,
            )
            .expect("path declaration should normalize without a host path");
            assert_eq!(normalized.channel, None);
            assert!(!format!("{normalized:?}").contains(host_path));
        }
        for options in [
            "channel = \"stable\"",
            "components = []",
            "targets = []",
            "profile = \"minimal\"",
        ] {
            let error = parse_rust_toolchain_configuration(
                &format!("[toolchain]\npath = {host_path:?}\n{options}\n"),
                true,
            )
            .expect_err("path toolchains reject any competing option presence");
            assert!(!format!("{error:#}").contains(host_path));
        }
        assert!(
            parse_rust_toolchain_configuration("[toolchain]\npath = \"relative/compiler\"\n", true)
                .is_err()
        );
    }

    #[test]
    fn cargo_workspace_toolchain_parser_bounds_input_and_sanitizes_errors() {
        assert!(
            parse_rust_toolchain_configuration(
                &" ".repeat(MAX_RUST_TOOLCHAIN_FILE_BYTES + 1),
                true
            )
            .is_err()
        );
        let long_field = "s".repeat(MAX_CARGO_CONFIGURATION_FIELD_BYTES + 1);
        for field in ["channel", "path", "profile"] {
            assert!(
                parse_rust_toolchain_configuration(
                    &format!("[toolchain]\n{field} = {long_field:?}\n"),
                    true,
                )
                .is_err()
            );
        }
        for field in ["components", "targets"] {
            assert!(
                parse_rust_toolchain_configuration(
                    &format!("[toolchain]\n{field} = [{long_field:?}]\n"),
                    true,
                )
                .is_err()
            );
            let entries = vec!["\"rustfmt\""; MAX_CARGO_CONFIGURATION_ITEMS + 1].join(",");
            assert!(
                parse_rust_toolchain_configuration(
                    &format!("[toolchain]\n{field} = [{entries}]\n"),
                    true,
                )
                .is_err()
            );
        }
        let private_value = "/private/secret/toolchain";
        let error = parse_rust_toolchain_configuration(
            &format!("[toolchain]\npath = \"{private_value}"),
            true,
        )
        .expect_err("malformed TOML should fail without including its contents");
        assert!(!format!("{error:#}").contains(private_value));
    }

    #[test]
    fn cargo_workspace_rustc_probe_parses_only_release_and_host() {
        let compiler = parse_rustc_verbose_version(include_bytes!(
            "../test_data/cargo_workspace/rustc-vv.txt"
        ))
        .expect("rustc fixture should parse");
        assert_eq!(compiler.status, CargoHostCompilerStatus::Available);
        assert_eq!(compiler.release.as_deref(), Some("1.90.0"));
        assert_eq!(
            compiler.host_target.as_deref(),
            Some("x86_64-unknown-linux-gnu")
        );
        assert!(parse_rustc_verbose_version(b"rustc 1.90.0\n").is_err());
    }

    #[test]
    fn cargo_workspace_configuration_can_retain_safe_stale_facts() {
        let mut previous = CargoWorkspaceConfiguration::unresolved();
        previous.host_compiler = CargoHostCompilerModel {
            status: CargoHostCompilerStatus::Available,
            release: Some("1.90.0".to_string()),
            host_target: Some("x86_64-unknown-linux-gnu".to_string()),
            stale: false,
        };
        previous.declared_toolchain = Some(
            parse_rust_toolchain(
                project_path("rust-toolchain.toml"),
                include_str!("../test_data/cargo_workspace/rust-toolchain.toml"),
            )
            .expect("toolchain fixture should parse"),
        );
        let mut current = CargoWorkspaceConfiguration::unresolved();
        current.host_compiler.status = CargoHostCompilerStatus::Failed;
        current.add_diagnostic(
            None,
            CargoConfigurationDiagnosticCategory::CompilerProbe,
            "probe failed",
        );
        current.retain_stale_safe_facts(&previous);
        assert!(current.host_compiler.stale);
        assert_eq!(
            current.host_compiler.release,
            previous.host_compiler.release
        );
        assert_eq!(current.declared_toolchain, previous.declared_toolchain);
    }

    #[test]
    fn cargo_dependency_provenance_combines_manifest_lock_resolution_and_cycles() {
        let package_a_manifest = project_path("a/Cargo.toml");
        let package_b_manifest = project_path("b/Cargo.toml");
        let dependency = |name: &str, workspace_member: ProjectPath| CargoDependencyModel {
            declaration_name: name.to_string(),
            rename: Some(format!("{name}_alias")),
            kind: CargoDependencyKind::Normal,
            version_requirement: "^1".to_string(),
            optional: true,
            uses_default_features: false,
            requested_features: vec!["derive".to_string()],
            target: None,
            source_kind: CargoDependencySourceKind::Path,
            resolved_name: Some(name.to_string()),
            resolved_version: Some("1.0.0".to_string()),
            resolved_workspace_member: Some(workspace_member.clone()),
            local_manifest: Some(workspace_member.clone()),
            declaration_manifest: None,
            declaration_origin: CargoDependencyDeclarationOrigin::Unknown,
            resolved_instances: vec![CargoResolvedDependencyModel {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                source_kind: CargoDependencySourceKind::Path,
                enabled_features: vec!["derive".to_string()],
                lock_status: CargoDependencyLockStatus::Unknown,
                workspace_member: Some(workspace_member.clone()),
                local_manifest: Some(workspace_member),
            }],
            resolution_truncated: false,
            feature_causality: CargoDependencyFeatureCausality::Validated,
            cycle_detected: false,
        };
        let package = |name: &str, manifest_path: ProjectPath, dependency: CargoDependencyModel| {
            CargoPackageModel {
                id: name.to_string(),
                name: name.to_string(),
                version: "1.0.0".to_string(),
                manifest_path,
                is_default_member: false,
                targets: Vec::new(),
                features: Vec::new(),
                dependencies: vec![dependency],
            }
        };
        let mut workspace = CargoWorkspaceModel {
            key: CargoWorkspaceKey {
                root: project_path(""),
            },
            root_manifest: Some(project_path("Cargo.toml")),
            display_name: "fixture".to_string(),
            is_virtual: true,
            members: vec![
                package(
                    "a",
                    package_a_manifest.clone(),
                    dependency("b", package_b_manifest.clone()),
                ),
                package(
                    "b",
                    package_b_manifest.clone(),
                    dependency("a", package_a_manifest.clone()),
                ),
            ],
            dependency_graph: None,
            configuration: CargoWorkspaceConfiguration::unresolved(),
        };
        let manifests = BTreeMap::from([
            (
                package_a_manifest.clone(),
                "[dependencies]\nb_alias = { workspace = true, features = [\"derive\"] }\n"
                    .to_string(),
            ),
            (
                package_b_manifest,
                "[dependencies]\na_alias = { path = \"../a\" }\n".to_string(),
            ),
        ]);
        let lock = "version = 4\n[[package]]\nname = \"a\"\nversion = \"1.0.0\"\n[[package]]\nname = \"b\"\nversion = \"1.0.0\"\n";
        enrich_dependency_provenance(&mut workspace, &manifests, Some(lock));

        let dependency_a = &workspace.members[0].dependencies[0];
        assert_eq!(
            dependency_a.declaration_origin,
            CargoDependencyDeclarationOrigin::WorkspaceInherited
        );
        assert_eq!(
            dependency_a.resolved_instances[0].lock_status,
            CargoDependencyLockStatus::Locked
        );
        assert!(dependency_a.cycle_detected);
        assert_eq!(
            dependency_a.declaration_manifest.as_ref(),
            Some(&package_a_manifest)
        );

        enrich_dependency_provenance(&mut workspace, &manifests, Some("malformed = ["));
        assert_eq!(
            workspace.members[0].dependencies[0].resolved_instances[0].lock_status,
            CargoDependencyLockStatus::Unknown
        );
        assert_eq!(
            workspace.members[1].dependencies[0].declaration_origin,
            CargoDependencyDeclarationOrigin::Direct
        );
    }
    #[test]
    fn cargo_native_target_policy_inherited_edition_joins_bounded_workspace_data() {
        let member = "[package]\nname='member'\nversion='0.1.0'\nedition.workspace=true\n";
        let manifest_path = Path::new("/workspace/member/Cargo.toml");
        let workspace = "[workspace]\nmembers=['member']\n[workspace.package]\nedition='2024'\n";
        for (kind, directory) in [
            (CargoTargetKind::Test, "tests"),
            (CargoTargetKind::Bench, "benches"),
        ] {
            let source = PathBuf::from(format!("/workspace/member/{directory}/selected.rs"));
            let inherited = parse_native_cargo_execution_policy_with_workspace_manifest(
                member,
                manifest_path,
                &kind,
                "selected",
                &source,
                "2024",
                Some(workspace),
            )
            .expect("reviewed inherited edition data");
            assert_eq!(
                inherited,
                if kind == CargoTargetKind::Test {
                    CargoNativeExecutionPolicy::Test { harness: true }
                } else {
                    CargoNativeExecutionPolicy::Bench { harness: true }
                }
            );
            let section = if kind == CargoTargetKind::Test {
                "test"
            } else {
                "bench"
            };
            let explicit =
                format!("{member}[[{section}]]\nname='selected'\nedition='2018'\nharness=false\n");
            let policy = parse_native_cargo_execution_policy_with_workspace_manifest(
                &explicit,
                manifest_path,
                &kind,
                "selected",
                &source,
                "2018",
                Some(workspace),
            )
            .expect("target override over inherited package edition");
            assert_eq!(
                policy,
                if kind == CargoTargetKind::Test {
                    CargoNativeExecutionPolicy::Test { harness: false }
                } else {
                    CargoNativeExecutionPolicy::Bench { harness: false }
                }
            );
            assert!(
                policy
                    .expected_test_mode(&kind)
                    .expect("exact compile mode")
            );
            assert!(
                parse_native_cargo_execution_policy_with_workspace_manifest(
                    member,
                    manifest_path,
                    &kind,
                    "selected",
                    &source,
                    "2021",
                    Some(workspace)
                )
                .is_err(),
                "workspace edition cannot replace mismatched typed metadata"
            );
        }
    }

    #[test]
    fn cargo_native_target_policy_inherited_edition_requires_complete_current_declarations() {
        let member = "[package]\nname='member'\nversion='0.1.0'\nedition.workspace=true\n";
        let path = Path::new("/workspace/member/Cargo.toml");
        let source = Path::new("/workspace/member/tests/selected.rs");
        let oversized = " ".repeat(MAX_CARGO_NATIVE_TARGET_POLICY_BYTES + 1);
        for workspace in [
            None,
            Some("invalid=["),
            Some("[workspace]"),
            Some("[workspace.package]\nedition=2024"),
            Some("[workspace.package]\nedition={workspace=true}"),
            Some("[workspace.package]\nedition='private-edition-sentinel'"),
            Some(oversized.as_str()),
        ] {
            let error = parse_native_cargo_execution_policy_with_workspace_manifest(
                member,
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2024",
                workspace,
            )
            .expect_err("missing or invalid inheritance remains unavailable");
            assert!(!error.to_string().contains("private-edition-sentinel"));
        }
        for declaration in [
            "edition.workspace=false",
            "edition={workspace=true,other=true}",
            "edition=2024",
        ] {
            let member = format!("[package]\nname='member'\nversion='0.1.0'\n{declaration}\n");
            assert!(
                parse_native_cargo_execution_policy_with_workspace_manifest(
                    &member,
                    path,
                    &CargoTargetKind::Test,
                    "selected",
                    source,
                    "2024",
                    Some("[workspace.package]\nedition='2024'")
                )
                .is_err()
            );
        }
        let direct = "[package]\nname='member'\nversion='0.1.0'\nedition='2024'\n";
        assert!(
            parse_native_cargo_execution_policy_with_workspace_manifest(
                direct,
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2024",
                Some("invalid=[")
            )
            .is_ok(),
            "irrelevant root data does not discard direct package edition"
        );
        assert!(
            parse_native_cargo_execution_policy(
                member,
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2024"
            )
            .is_err(),
            "no guessed root inheritance on legacy data-only entrypoint"
        );
    }

    #[test]
    fn cargo_native_target_policy_retains_explicit_and_autodiscovered_harness_modes() {
        let manifest_path = Path::new("/workspace/member/Cargo.toml");
        for (kind, directory, section) in [
            (CargoTargetKind::Test, "tests", "test"),
            (CargoTargetKind::Bench, "benches", "bench"),
        ] {
            for harness in [true, false] {
                let manifest = format!(
                    "[package]\nname='member'\nversion='0.1.0'\nedition='2024'\n[[{section}]]\nname='selected'\npath='custom/selected.rs'\nharness={harness}\n"
                );
                let policy = parse_native_cargo_execution_policy(
                    &manifest,
                    manifest_path,
                    &kind,
                    "selected",
                    Path::new("/workspace/member/custom/selected.rs"),
                    "2024",
                )
                .expect("exact declared policy");
                assert_eq!(
                    policy,
                    if kind == CargoTargetKind::Test {
                        CargoNativeExecutionPolicy::Test { harness }
                    } else {
                        CargoNativeExecutionPolicy::Bench { harness }
                    }
                );
                assert!(
                    policy.expected_test_mode(&kind).expect("test compile mode"),
                    "Cargo test compile mode remains true for custom harness"
                );
                assert!(policy.expected_test_mode(&CargoTargetKind::Binary).is_err());
            }
            for source in [
                format!("/workspace/member/{directory}/selected.rs"),
                format!("/workspace/member/{directory}/selected/main.rs"),
            ] {
                let policy = parse_native_cargo_execution_policy(
                    "[package]\nname='member'\nversion='0.1.0'\nedition='2024'\n",
                    manifest_path,
                    &kind,
                    "selected",
                    Path::new(&source),
                    "2024",
                )
                .expect("autodiscovered default policy");
                assert_eq!(
                    policy,
                    if kind == CargoTargetKind::Test {
                        CargoNativeExecutionPolicy::Test { harness: true }
                    } else {
                        CargoNativeExecutionPolicy::Bench { harness: true }
                    }
                );
            }
        }
        for kind in [CargoTargetKind::Binary, CargoTargetKind::Example] {
            assert_eq!(
                parse_native_cargo_execution_policy(
                    "",
                    manifest_path,
                    &kind,
                    "selected",
                    Path::new("/workspace/member/src/main.rs"),
                    "2024"
                )
                .expect("unchanged Program mode"),
                CargoNativeExecutionPolicy::Program
            );
        }
    }

    #[test]
    fn cargo_native_target_policy_rejects_stale_ambiguous_malformed_and_oversized_data() {
        let path = Path::new("/workspace/member/Cargo.toml");
        let source = Path::new("/workspace/member/tests/selected.rs");
        let header = "[package]\nname='member'\nversion='0.1.0'\nedition='2024'\n";
        let mut exact = header.to_owned();
        exact.push_str(&" ".repeat(MAX_CARGO_NATIVE_TARGET_POLICY_BYTES - exact.len()));
        assert!(
            parse_native_cargo_execution_policy(
                &exact,
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2024"
            )
            .is_ok(),
            "inclusive exact manifest byte bound"
        );
        for suffix in [
            "autotests=false\n",
            "autotests='false'\n",
            "[[test]]\nname='selected'\nharness='false'\n",
            "[[test]]\nname='selected'\npath='tests/other.rs'\n",
            "[[test]]\nname='selected'\n[[test]]\nname='selected'\n",
            "[[test]]\npath='tests/selected.rs'\nharness=false\n",
            "[test]\nname='selected'\n",
        ] {
            assert!(
                parse_native_cargo_execution_policy(
                    &format!("{header}{suffix}"),
                    path,
                    &CargoTargetKind::Test,
                    "selected",
                    source,
                    "2024"
                )
                .is_err(),
                "{suffix}"
            );
        }
        for manifest in [
            "invalid=[".into(),
            " ".repeat(MAX_CARGO_NATIVE_TARGET_POLICY_BYTES + 1),
            format!(
                "{header}{}",
                "[[test]]\nname='other'\n".repeat(MAX_CARGO_CONFIGURATION_ITEMS + 1)
            ),
        ] {
            assert!(
                parse_native_cargo_execution_policy(
                    &manifest,
                    path,
                    &CargoTargetKind::Test,
                    "selected",
                    source,
                    "2024"
                )
                .is_err()
            );
        }
        assert!(
            parse_native_cargo_execution_policy(
                header,
                path,
                &CargoTargetKind::Test,
                "selected",
                Path::new("/workspace/member/custom/selected.rs"),
                "2024"
            )
            .is_err()
        );
        assert!(
            parse_native_cargo_execution_policy(
                header,
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "unknown-edition"
            )
            .is_err()
        );
        assert!(
            parse_native_cargo_execution_policy(
                header,
                Path::new("relative/Cargo.toml"),
                &CargoTargetKind::Test,
                "selected",
                source,
                "2024"
            )
            .is_err()
        );
    }

    #[test]
    fn cargo_native_target_policy_checks_source_normalization_and_2015_autodiscovery() {
        let path = Path::new("/workspace/member/Cargo.toml");
        let source = Path::new("/workspace/shared/selected.rs");
        let manifest = "[package]\nname='member'\nversion='0.1.0'\n[[test]]\nname='selected'\npath='../shared/selected.rs'\nharness=false\n";
        assert_eq!(
            parse_native_cargo_execution_policy(
                manifest,
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2015"
            )
            .expect("exact shared source data; host sealing still required"),
            CargoNativeExecutionPolicy::Test { harness: false }
        );
        let header = "[package]\nname='member'\nversion='0.1.0'\n";
        let source = Path::new("/workspace/member/tests/selected.rs");
        assert!(
            parse_native_cargo_execution_policy(
                &format!("{header}[[bin]]\nname='cli'\n"),
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2015"
            )
            .is_err()
        );
        assert!(
            parse_native_cargo_execution_policy(
                &format!("{header}autotests=true\n[[bin]]\nname='cli'\n"),
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2015"
            )
            .is_ok()
        );
        assert!(
            parse_native_cargo_execution_policy(
                &format!("{header}edition='2018'\n[[bin]]\nname='cli'\n"),
                path,
                &CargoTargetKind::Test,
                "selected",
                source,
                "2018"
            )
            .is_ok()
        );
        let error = parse_native_cargo_execution_policy(
            &format!(
                "{header}edition='2024'\n[[test]]\nname='selected'\npath='private-sentinel'\n"
            ),
            path,
            &CargoTargetKind::Test,
            "selected",
            source,
            "2024",
        )
        .expect_err("source mismatch");
        assert!(!error.to_string().contains("private-sentinel"));
    }

    fn native_runtime_script(package: &str, paths: &[String]) -> serde_json::Value {
        serde_json::json!({"reason":"build-script-executed", "package_id":package,
            "linked_libs":[], "linked_paths":paths, "cfgs":[], "env":[],
            "out_dir":"/fixture/target/debug/build/selected/out"})
    }

    #[test]
    fn cargo_native_runtime_records_retain_exact_package_paths_and_legacy_behavior() {
        let selection = native_build_selection();
        let dependency = "path+file:///dependency#dependency@1.0.0";
        let packages = BTreeSet::from_iter([selection.package_id.clone(), dependency.into()]);
        let records = [
            native_runtime_script(
                dependency,
                &["native=/fixture/target/debug/build/dependency/out".into()],
            ),
            native_runtime_script(
                "path+file:///unrelated#unrelated@1.0.0",
                &["/unrelated/path".into()],
            ),
            native_runtime_script(
                &selection.package_id,
                &["/fixture/target/debug/build/selected/out".into()],
            ),
            native_build_artifact(),
            native_build_finished(true),
        ];
        let bytes = native_build_stream(&records);
        let output = resolve_native_cargo_build_output(&bytes, &selection, true, Some(&packages))
            .expect("bounded actual report data");
        assert_eq!(
            output.executable,
            PathBuf::from("/fixture/target/debug/deps/cli")
        );
        let libraries = output.runtime_libraries.expect("present metadata closure");
        assert_eq!(libraries.len(), 2);
        assert_eq!(libraries[0].package_id(), dependency);
        assert_eq!(
            libraries[0].linked_paths(),
            &[PathBuf::from(
                "native=/fixture/target/debug/build/dependency/out"
            )]
        );
        assert_eq!(libraries[1].package_id(), selection.package_id);
        assert_eq!(
            libraries[1].output_directory(),
            Some(Path::new("/fixture/target/debug/build/selected/out"))
        );
        assert_eq!(
            resolve_native_cargo_executable(&bytes, &selection, true)
                .expect("unchanged public executable"),
            output.executable
        );
        assert!(
            resolve_native_cargo_build_output(&bytes, &selection, true, None)
                .expect("explicit unavailable runtime data")
                .runtime_libraries
                .is_none()
        );
    }

    #[test]
    fn cargo_native_runtime_records_preserve_failure_and_bounded_retention() {
        let selection = native_build_selection();
        let packages = BTreeSet::from_iter([selection.package_id.clone()]);
        let exact = (0..MAX_CARGO_CONFIGURATION_ITEMS)
            .map(|index| format!("/fixture/target/library-{index}"))
            .collect::<Vec<_>>();
        let report = |paths: &[String]| {
            native_build_stream(&[
                native_runtime_script(&selection.package_id, paths),
                native_build_artifact(),
                native_build_finished(true),
            ])
        };
        assert_eq!(
            resolve_native_cargo_build_output(&report(&exact), &selection, true, Some(&packages))
                .expect("inclusive record bound")
                .runtime_libraries
                .expect("runtime records")[0]
                .linked_paths()
                .len(),
            MAX_CARGO_CONFIGURATION_ITEMS
        );
        for paths in [
            vec!["bad\npath".into()],
            vec!["x".repeat(MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES + 1)],
            vec!["".into()],
            vec!["/fixture/lib".into(); MAX_CARGO_CONFIGURATION_ITEMS + 1],
        ] {
            assert!(
                resolve_native_cargo_build_output(
                    &report(&paths),
                    &selection,
                    true,
                    Some(&packages)
                )
                .is_err()
            );
        }
        let mut records = vec![
            native_runtime_script(&selection.package_id, &exact);
            MAX_CARGO_NATIVE_RUNTIME_PATHS / exact.len()
        ];
        records.extend([native_build_artifact(), native_build_finished(true)]);
        assert!(
            resolve_native_cargo_build_output(
                &native_build_stream(&records),
                &selection,
                true,
                Some(&packages)
            )
            .is_ok()
        );
        records.insert(
            0,
            native_runtime_script(&selection.package_id, &["/fixture/one-too-many".into()]),
        );
        assert!(
            resolve_native_cargo_build_output(
                &native_build_stream(&records),
                &selection,
                true,
                Some(&packages)
            )
            .is_err()
        );
        let wide_paths =
            vec![format!("/fixture/{}", "x".repeat(600)); MAX_CARGO_CONFIGURATION_ITEMS];
        let mut byte_overflow = vec![native_runtime_script(&selection.package_id, &wide_paths); 16];
        byte_overflow.extend([native_build_artifact(), native_build_finished(true)]);
        assert!(
            resolve_native_cargo_build_output(
                &native_build_stream(&byte_overflow),
                &selection,
                true,
                Some(&packages)
            )
            .is_err(),
            "aggregate retained byte cap independent of path/line/report caps"
        );
        let mut too_many_scripts =
            vec![native_runtime_script(&selection.package_id, &[]); MAX_CARGO_BUILD_ARTIFACTS];
        too_many_scripts.extend([native_build_artifact(), native_build_finished(true)]);
        assert!(
            resolve_native_cargo_build_output(
                &native_build_stream(&too_many_scripts),
                &selection,
                true,
                Some(&packages)
            )
            .is_ok()
        );
        too_many_scripts.insert(0, native_runtime_script(&selection.package_id, &[]));
        assert!(
            resolve_native_cargo_build_output(
                &native_build_stream(&too_many_scripts),
                &selection,
                true,
                Some(&packages)
            )
            .is_err()
        );
        for success in [false, true] {
            let failed = native_build_stream(&[
                native_runtime_script(&selection.package_id, &exact),
                native_build_artifact(),
                native_build_finished(false),
            ]);
            assert!(
                resolve_native_cargo_build_output(&failed, &selection, success, Some(&packages))
                    .is_err()
            );
        }
        let after_finished = native_build_stream(&[
            native_build_artifact(),
            native_build_finished(true),
            native_runtime_script(&selection.package_id, &exact),
        ]);
        assert!(
            resolve_native_cargo_build_output(&after_finished, &selection, true, Some(&packages))
                .is_err()
        );
        let wrong_closure = BTreeSet::from_iter(["other".into()]);
        assert!(
            resolve_native_cargo_build_output(
                &report(&exact),
                &selection,
                true,
                Some(&wrong_closure)
            )
            .is_err()
        );
    }

    #[test]
    fn cargo_native_executable_retains_typed_test_bench_compile_mode_without_launch_authority() {
        for (kind, raw_kind, directory) in [
            (CargoTargetKind::Test, "test", "tests"),
            (CargoTargetKind::Bench, "bench", "benches"),
        ] {
            for harness in [true, false] {
                let mut selection = native_build_selection();
                selection.target_kind = kind.clone();
                selection.source_path = PathBuf::from(format!("/fixture/{directory}/cli.rs"));
                selection.execution_policy = if kind == CargoTargetKind::Test {
                    CargoNativeExecutionPolicy::Test { harness }
                } else {
                    CargoNativeExecutionPolicy::Bench { harness }
                };
                let mut artifact = native_build_artifact();
                artifact["target"]["kind"] = serde_json::json!([raw_kind]);
                artifact["target"]["src_path"] =
                    selection.source_path.to_str().expect("source").into();
                artifact["profile"]["test"] = true.into();
                let successful =
                    native_build_stream(&[artifact.clone(), native_build_finished(true)]);
                assert!(
                    resolve_native_cargo_build_output(&successful, &selection, true, None).is_ok()
                );
                assert!(resolve_native_cargo_executable(&successful, &selection, true).is_err());
                assert!(
                    resolve_native_cargo_build_output(&successful, &selection, false, None)
                        .is_err()
                );
                let mut wrong_mode = artifact.clone();
                wrong_mode["profile"]["test"] = false.into();
                assert!(
                    resolve_native_cargo_build_output(
                        &native_build_stream(&[wrong_mode, native_build_finished(true)]),
                        &selection,
                        true,
                        None
                    )
                    .is_err()
                );
                assert!(
                    resolve_native_cargo_build_output(
                        &native_build_stream(&[
                            artifact.clone(),
                            artifact,
                            native_build_finished(true)
                        ]),
                        &selection,
                        true,
                        None
                    )
                    .is_err()
                );
                selection.execution_policy = CargoNativeExecutionPolicy::Program;
                assert!(
                    resolve_native_cargo_build_output(&successful, &selection, true, None).is_err()
                );
            }
        }
    }

    fn native_build_selection() -> CargoExecutableSelection {
        CargoExecutableSelection {
            package_id: "path+file:///fixture#rust-fixture@0.1.0".into(),
            target_kind: CargoTargetKind::Binary,
            target_name: "cli".into(),
            source_path: "/fixture/src/bin/cli.rs".into(),
            execution_policy: CargoNativeExecutionPolicy::Program,
        }
    }

    fn native_build_artifact() -> serde_json::Value {
        let mut artifact: serde_json::Value = serde_json::from_str(
            include_str!("../test_data/rust_test_provider/cargo_messages.jsonl")
                .lines()
                .nth(1)
                .expect("binary fixture"),
        )
        .expect("typed fixture");
        artifact["profile"]["test"] = false.into();
        artifact
    }

    fn native_build_stream(records: &[serde_json::Value]) -> Vec<u8> {
        records
            .iter()
            .map(|record| serde_json::to_string(record).expect("fixture JSON"))
            .collect::<Vec<_>>()
            .join("\n")
            .into_bytes()
    }

    fn native_build_finished(success: bool) -> serde_json::Value {
        serde_json::json!({"reason":"build-finished","success":success})
    }

    #[test]
    fn cargo_native_executable_selects_exact_binary_and_example() {
        let selection = native_build_selection();
        let artifact = native_build_artifact();
        let mut other = artifact.clone();
        other["package_id"] = "path+file:///dependency#dependency@1.0.0".into();
        other["target"]["name"] = "cli".into();
        other["executable"] = "/fixture/target/debug/other".into();
        assert_eq!(
            resolve_native_cargo_executable(
                &native_build_stream(&[other, artifact.clone(), native_build_finished(true)]),
                &selection,
                true
            )
            .expect("exact binary"),
            PathBuf::from("/fixture/target/debug/deps/cli")
        );
        let mut example = artifact;
        example["target"]["kind"] = serde_json::json!(["example"]);
        example["target"]["name"] = "demo".into();
        example["target"]["src_path"] = "/fixture/examples/demo.rs".into();
        example["executable"] = "/fixture/target/debug/examples/demo".into();
        let example_selection = CargoExecutableSelection {
            target_kind: CargoTargetKind::Example,
            target_name: "demo".into(),
            source_path: "/fixture/examples/demo.rs".into(),
            ..selection
        };
        assert_eq!(
            resolve_native_cargo_executable(
                &native_build_stream(&[example, native_build_finished(true)]),
                &example_selection,
                true
            )
            .expect("exact example"),
            PathBuf::from("/fixture/target/debug/examples/demo")
        );
    }

    #[test]
    fn cargo_native_executable_rejects_cross_selection_and_duplicates() {
        for mutate in [
            "package", "kind", "name", "source", "test", "missing", "relative", "parent",
        ] {
            let mut artifact = native_build_artifact();
            match mutate {
                "package" => artifact["package_id"] = "path+file:///other#other@1.0.0".into(),
                "kind" => artifact["target"]["kind"] = serde_json::json!(["bin", "example"]),
                "name" => artifact["target"]["name"] = "other".into(),
                "source" => artifact["target"]["src_path"] = "/fixture/other.rs".into(),
                "test" => artifact["profile"]["test"] = true.into(),
                "missing" => artifact["executable"] = serde_json::Value::Null,
                "relative" => artifact["executable"] = "target/debug/cli".into(),
                "parent" => artifact["executable"] = "/fixture/../private/cli".into(),
                _ => panic!("invalid mutation fixture"),
            }
            assert!(
                resolve_native_cargo_executable(
                    &native_build_stream(&[artifact, native_build_finished(true)]),
                    &native_build_selection(),
                    true
                )
                .is_err(),
                "{mutate}"
            );
        }
        let artifact = native_build_artifact();
        for second in [artifact.clone(), {
            let mut different = artifact.clone();
            different["executable"] = "/fixture/target/release/cli".into();
            different
        }] {
            assert!(
                resolve_native_cargo_executable(
                    &native_build_stream(&[artifact.clone(), second, native_build_finished(true)]),
                    &native_build_selection(),
                    true
                )
                .is_err()
            );
        }
    }

    #[test]
    fn cargo_native_executable_requires_complete_successful_producer_data() {
        let artifact = native_build_artifact();
        let selection = native_build_selection();
        for records in [
            vec![artifact.clone()],
            vec![artifact.clone(), native_build_finished(false)],
            vec![
                artifact.clone(),
                native_build_finished(true),
                native_build_finished(true),
            ],
            vec![native_build_finished(true), artifact.clone()],
        ] {
            assert!(
                resolve_native_cargo_executable(&native_build_stream(&records), &selection, true)
                    .is_err()
            );
        }
        let complete = native_build_stream(&[artifact, native_build_finished(true)]);
        assert!(resolve_native_cargo_executable(&complete, &selection, false).is_err());
        for suffix in [
            b"\n{bad".as_slice(),
            b"\n{\"reason\":\"other\"}".as_slice(),
            b"\n\xff".as_slice(),
        ] {
            let mut malformed = complete.clone();
            malformed.extend_from_slice(suffix);
            assert!(resolve_native_cargo_executable(&malformed, &selection, true).is_err());
        }
    }

    #[test]
    fn cargo_native_executable_enforces_byte_line_field_and_artifact_bounds() {
        let selection = native_build_selection();
        let complete = native_build_stream(&[native_build_artifact(), native_build_finished(true)]);
        let mut exact_bytes = complete.clone();
        exact_bytes.resize(MAX_CARGO_BUILD_MESSAGE_BYTES, b' ');
        assert!(resolve_native_cargo_executable(&exact_bytes, &selection, true).is_ok());
        exact_bytes.push(b' ');
        assert!(resolve_native_cargo_executable(&exact_bytes, &selection, true).is_err());
        let mut exact_lines = complete;
        exact_lines.extend_from_slice("\n".repeat(MAX_CARGO_BUILD_MESSAGE_LINES - 1).as_bytes());
        assert!(resolve_native_cargo_executable(&exact_lines, &selection, true).is_ok());
        exact_lines.push(b'\n');
        assert!(resolve_native_cargo_executable(&exact_lines, &selection, true).is_err());
        let mut exact_field = native_build_artifact();
        exact_field["features"] =
            serde_json::json!(["x".repeat(MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES)]);
        assert!(
            resolve_native_cargo_executable(
                &native_build_stream(&[exact_field, native_build_finished(true)]),
                &selection,
                true
            )
            .is_ok()
        );
        assert!(
            resolve_native_cargo_executable(
                &vec![b' '; MAX_CARGO_BUILD_MESSAGE_BYTES + 1],
                &selection,
                true
            )
            .is_err()
        );
        assert!(
            resolve_native_cargo_executable(
                "\n".repeat(MAX_CARGO_BUILD_MESSAGE_LINES + 1).as_bytes(),
                &selection,
                true
            )
            .is_err()
        );
        let mut oversized = native_build_artifact();
        oversized["features"] =
            serde_json::json!(["x".repeat(MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES + 1)]);
        assert!(
            resolve_native_cargo_executable(
                &native_build_stream(&[oversized, native_build_finished(true)]),
                &selection,
                true
            )
            .is_err()
        );
        let mut other = native_build_artifact();
        other["target"]["name"] = "other".into();
        let mut exact_artifacts = vec![other.clone(); MAX_CARGO_BUILD_ARTIFACTS - 1];
        exact_artifacts.push(native_build_artifact());
        exact_artifacts.push(native_build_finished(true));
        assert!(
            resolve_native_cargo_executable(
                &native_build_stream(&exact_artifacts),
                &selection,
                true
            )
            .is_ok()
        );
        let mut many = vec![other; MAX_CARGO_BUILD_ARTIFACTS + 1];
        many.push(native_build_artifact());
        many.push(native_build_finished(true));
        assert!(
            resolve_native_cargo_executable(&native_build_stream(&many), &selection, true).is_err()
        );
        let oversized_fields = serde_json::json!({"reason":"build-finished","success":true,"extra":vec![0;MAX_CARGO_BUILD_MESSAGE_FIELDS]});
        assert!(
            resolve_native_cargo_executable(
                &native_build_stream(&[oversized_fields]),
                &selection,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn cargo_message_decoder_preserves_tolerant_line_diagnostics_and_utf8_bounds() {
        let input = "\n{bad\n{\"reason\":\"other\"}\n{\"reason\":\"build-finished\"}\n{\"reason\":\"build-finished\",\"success\":true}\n";
        let records = CargoMessageDecoder::new(input, input.len(), 10).collect::<Vec<_>>();
        assert_eq!(records.len(), 4);
        assert!(
            records[0]
                .as_ref()
                .expect_err("malformed")
                .starts_with("Cargo JSON line 2 is malformed:")
        );
        assert_eq!(
            records[1].as_ref().expect_err("unknown"),
            "Cargo JSON line 3 has an unknown record kind"
        );
        assert!(
            records[2]
                .as_ref()
                .expect_err("incomplete")
                .starts_with("Cargo JSON line 4 is incomplete:")
        );
        assert!(matches!(records[3], Ok(Message::BuildFinished(_))));
        let decoder = CargoMessageDecoder::new("é", 1, 10);
        assert!(decoder.byte_limit_reached());
        assert_eq!(decoder.count(), 0);
        let decoder = CargoMessageDecoder::new(input, input.len(), 2);
        assert!(decoder.line_limit_reached());
        assert_eq!(decoder.count(), 1);
    }
}

const MAX_NATIVE_PROFILE_COMMAND_ITEMS: usize = 256;
const MAX_NATIVE_PROFILE_COMMAND_TEXT_BYTES: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeProfileCommand {
    pub command: String,
    pub arguments: Vec<String>,
    pub environment: collections::HashMap<String, String>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeProfileCollector {
    pub command: String,
    pub version: String,
}

pub fn compile_native_profile_task(
    mut task_template: task::TaskTemplate,
    host_os: &str,
    collector: &NativeProfileCollector,
    workload: NativeProfileCommand,
    profile_path: &str,
) -> Result<task::TaskTemplate> {
    task::validate_experimental_feature_requirements(
        &task_template.required_experimental_features,
    )?;
    match host_os {
        "macos" => {}
        "linux" => bail!("Native profiling on Linux is unsupported in the approved macOS phase"),
        "windows" => {
            bail!("Native profiling on Windows is unsupported in the approved macOS phase")
        }
        _ => bail!("Native profiling requires a known authoritative macOS host"),
    }
    if collector.version != "0.13.1" {
        bail!("Native profiling requires user-installed Samply 0.13.1 on the project host");
    }
    if !collector.command.starts_with('/')
        || collector.command.chars().any(|character| {
            character.is_whitespace()
                || character.is_control()
                || matches!(
                    character,
                    ';' | '|' | '&' | '<' | '>' | '`' | '$' | '\\' | '"' | '\''
                )
        })
    {
        bail!(
            "Native profiling requires a resolved collector executable supported by the Task host binding"
        );
    }
    validate_native_profile_command(&workload)?;
    if !workload.command.starts_with('/')
        || workload.command.contains('=')
        || workload.command.contains("$ZED_")
        || workload.command.contains("${ZED_")
        || workload.command.chars().any(char::is_control)
        || workload
            .arguments
            .iter()
            .chain(workload.environment.keys())
            .chain(workload.environment.values())
            .any(|value| value.contains('\0'))
    {
        bail!(
            "Native profiling requires a resolved workload executable and valid structured arguments"
        );
    }
    if task_template.stdout_report.is_some()
        || task_template.artifact.is_some()
        || task_template.artifact_argument_binding.is_some()
        || task_template.artifact_group_argument_binding.is_some()
        || task_template.artifact_group.is_some()
    {
        bail!("Native profiling requires an unadorned Cargo context");
    }
    if profile_path.contains("$ZED_") || profile_path.contains("${ZED_") {
        bail!("Native profiling requires a resolved output path before literal Task planning");
    }
    let profile_stem = profile_path
        .strip_suffix(".json")
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("Native profiling requires an explicit uncompressed JSON output")
        })?;
    if profile_path.ends_with(".syms.json") || profile_path.starts_with('-') {
        bail!("Native profiling requires a profile output distinct from its symbols companion");
    }
    let group = task::TaskArtifactGroup {
        format: "samply-0.13.1".to_string(),
        files: BTreeMap::from_iter([
            (
                "profile".to_string(),
                task::TaskArtifact {
                    path: profile_path.to_string(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 32 * 1024 * 1024,
                },
            ),
            (
                "symbols".to_string(),
                task::TaskArtifact {
                    path: format!("{profile_stem}.syms.json"),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 16 * 1024 * 1024,
                },
            ),
        ]),
    };
    let group = group.with_resolved_files(group.files.clone())?;
    let mut arguments = vec![
        "record".to_string(),
        "--save-only".to_string(),
        "--no-open".to_string(),
        "--main-thread-only".to_string(),
        "--unstable-presymbolicate".to_string(),
        "--include-args=0".to_string(),
        "--duration".to_string(),
        "30".to_string(),
        "-r".to_string(),
        "100".to_string(),
        "-o".to_string(),
        profile_path.to_string(),
        "--".to_string(),
        workload.command,
    ];
    arguments.extend(workload.arguments);
    task_template.env.extend(workload.environment);
    validate_native_profile_command(&NativeProfileCommand {
        command: collector.command.clone(),
        arguments: arguments.clone(),
        environment: task_template.env.clone(),
    })?;
    task_template.label = format!("Native Profile {}", task_template.label);
    task_template.experimental_policy_version = task::EXPERIMENTAL_TASK_POLICY_VERSION;
    if !task_template
        .required_experimental_features
        .iter()
        .any(|flag| flag == "rust-native-profiling-experimental")
    {
        task_template
            .required_experimental_features
            .push("rust-native-profiling-experimental".into());
    }
    task_template.command = collector.command.clone();
    task_template.literal_arguments = true;
    task_template.args = arguments;
    task_template.tags.retain(|tag| !tag.starts_with("cargo-"));
    task_template
        .tags
        .extend(["cargo-profile".to_string(), "native-profile".to_string()]);
    task_template.artifact_group = Some(group);
    task_template.artifact_group_argument_binding = Some(task::TaskArtifactGroupArgumentBinding {
        argument_index: 11,
        role: "profile".to_string(),
        related: vec![task::TaskArtifactRelatedArgumentBinding {
            role: "symbols".to_string(),
            extension: "syms.json".to_string(),
            argument_index: None,
        }],
    });
    Ok(task_template)
}

fn validate_native_profile_command(command: &NativeProfileCommand) -> Result<()> {
    if command.command.trim().is_empty()
        || command.command.len() > MAX_NATIVE_PROFILE_COMMAND_TEXT_BYTES
        || command.arguments.len() > MAX_NATIVE_PROFILE_COMMAND_ITEMS
        || command.environment.len() > MAX_NATIVE_PROFILE_COMMAND_ITEMS
    {
        bail!("external profile command is empty or exceeds the supported bounds");
    }
    if command
        .arguments
        .iter()
        .chain(command.environment.keys())
        .chain(command.environment.values())
        .any(|value| value.len() > MAX_NATIVE_PROFILE_COMMAND_TEXT_BYTES)
    {
        bail!("external profile command field exceeds the supported length");
    }
    Ok(())
}

#[cfg(test)]
#[test]
fn native_profile_task_retains_required_opt_ins_and_current_policy() {
    let template = task::TaskTemplate {
        required_experimental_features: vec!["rust-named-cargo-presets-experimental".into()],
        ..Default::default()
    };
    let compiled = compile_native_profile_task(
        template,
        "macos",
        &NativeProfileCollector {
            command: "/installed/samply".into(),
            version: "0.13.1".into(),
        },
        NativeProfileCommand {
            command: "/workspace/target/workload".into(),
            arguments: vec![],
            environment: collections::HashMap::default(),
        },
        "target/capture.json",
    )
    .expect("current native capture plan");
    assert_eq!(
        compiled.experimental_policy_version,
        task::EXPERIMENTAL_TASK_POLICY_VERSION
    );
    assert_eq!(
        compiled.required_experimental_features,
        vec![
            "rust-named-cargo-presets-experimental",
            "rust-native-profiling-experimental",
        ]
    );
}

#[cfg(test)]
#[test]
fn native_profile_task_rejects_inherited_build_report() {
    let task = task::TaskTemplate {
        stdout_report: Some(task::TaskStdoutReport {
            format: task::TaskStdoutReportFormat::CargoBuildJson,
            max_bytes: 1024 * 1024,
            native_profile: None,
        }),
        ..Default::default()
    };
    assert!(
        compile_native_profile_task(
            task,
            "macos",
            &NativeProfileCollector {
                command: "/installed/samply".into(),
                version: "0.13.1".into(),
            },
            NativeProfileCommand {
                command: "/workspace/target/workload".into(),
                arguments: vec![],
                environment: collections::HashMap::default()
            },
            "target/capture.json"
        )
        .is_err()
    );
}
